# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed latent dataset adapter for official Franka EE10 trajectories."""

from __future__ import annotations

import copy
import hashlib
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from n0_twam.data.ee10_alignment import build_franka_ee10_latent_targets
from n0_twam.integrations.worldarena.franka_views import (
    FrankaDatasetView,
    load_franka_view,
)
from n0_twam.tactile_profiles import (
    VISION_ONLY,
    validate_tactile_profile_config,
)

from .lerobot_latent_dataset import LatentLeRobotDataset, recursive_find_file

DATASET_ADAPTER = "worldarena_franka_ee10"


def _validate_config(config: object) -> None:
    expected = {
        "dataset_adapter": DATASET_ADAPTER,
        "action_schema": "ee20_absee",
        "action_dim": 20,
        "action_per_frame": 6,
        "tactile_profile": VISION_ONLY,
        "tactile_mode": "disabled",
        "use_local_tactile": False,
        "use_contact_gate": False,
    }
    for field, value in expected.items():
        if getattr(config, field, None) != value:
            raise ValueError(
                f"Franka dataset config mismatch for {field}: "
                f"{getattr(config, field, None)!r} vs {value!r}"
            )
    if list(getattr(config, "used_action_channel_ids", [])) != list(range(10)):
        raise ValueError("Franka active model action channels must be exactly 0..9")
    if list(getattr(config, "tactile_keys", [])):
        raise ValueError("Franka no-tactile dataset requires tactile_keys=[]")
    normalizer = getattr(config, "norm_stat", None)
    if not isinstance(normalizer, dict):
        raise ValueError("Franka dataset requires a normalizer object")
    for field in ("q01", "q99"):
        values = np.asarray(normalizer.get(field), dtype=np.float32)
        if values.shape != (20,) or not np.isfinite(values).all():
            raise ValueError(f"Franka normalizer {field} must have 20 finite values")
    q01 = np.asarray(normalizer["q01"], dtype=np.float32)
    q99 = np.asarray(normalizer["q99"], dtype=np.float32)
    if not np.array_equal(q01[10:], np.full(10, -1.0, dtype=np.float32)):
        raise ValueError("Franka inactive q01 channels must be -1")
    if not np.array_equal(q99[10:], np.full(10, 1.0, dtype=np.float32)):
        raise ValueError("Franka inactive q99 channels must be 1")


def _load_view(config: object) -> FrankaDatasetView:
    path = getattr(config, "dataset_view_path", None)
    if path is None:
        raise ValueError("Franka training requires dataset_view_path")
    view = load_franka_view(Path(str(path)))
    expected_id = getattr(config, "train_view_id", None)
    if view.view_id != expected_id:
        raise ValueError(
            f"Franka view ID mismatch: {view.view_id!r} vs {expected_id!r}"
        )
    return view


def _construct(repo_id: str, config: object) -> "FrankaLatentLeRobotDataset":
    return FrankaLatentLeRobotDataset(repo_id=repo_id, config=config)


class MultiLatentLeRobotFrankaDataset(torch.utils.data.Dataset):
    """One physical all600 repo filtered by an immutable development/final view."""

    def __init__(self, config, num_init_worker: int = 1) -> None:
        _validate_config(config)
        view = _load_view(config)
        dataset_root = Path(str(config.dataset_path)).expanduser().resolve(strict=True)
        repo_paths = sorted(
            str(Path(path).parents[1])
            for path in recursive_find_file(dataset_root, "info.json")
        )
        if tuple(Path(path).resolve(strict=True) for path in repo_paths) != (
            dataset_root,
        ):
            raise ValueError(
                "Franka dataset adapter requires exactly the declared all600 repo"
            )
        selected_config = copy.copy(config)
        selected_config.selected_lerobot_episode_ids = tuple(
            entry.lerobot_episode_id for entry in view.entries
        )
        worker_count = max(
            1,
            min(
                int(getattr(config, "num_init_worker", num_init_worker)),
                len(repo_paths),
            ),
        )
        constructor = partial(_construct, config=selected_config)
        if worker_count == 1:
            self._datasets = [constructor(repo_paths[0])]
        else:
            with Pool(worker_count) as pool:
                self._datasets = pool.map(constructor, repo_paths)
        validate_tactile_profile_config(
            config,
            repo_names=(dataset.repo_name for dataset in self._datasets),
        )
        self.dataset_view = view
        self._offsets: list[int] = []
        total = 0
        for dataset in self._datasets:
            self._offsets.append(total)
            total += len(dataset)
        self._length = total
        if self._length == 0:
            raise ValueError("Franka dataset contains no valid latent segments")
        self._sample_signatures = tuple(
            signature
            for dataset in self._datasets
            for signature in dataset.sample_signatures
        )
        sample_tasks: list[str] = []
        sample_ids: list[str] = []
        sample_episode_ids: list[int] = []
        for dataset in self._datasets:
            entries = dataset._view_entries_by_episode
            for meta in dataset.new_metas:
                episode_id = int(meta["episode_index"])
                entry = entries[episode_id]
                sample_episode_ids.append(episode_id)
                sample_tasks.append(entry.task)
                payload = {
                    "view_sha256": view.view_sha256,
                    "episode_id": episode_id,
                    "source_sha256": entry.source_sha256,
                    "start_frame": int(meta["start_frame"]),
                    "end_frame": int(meta["end_frame"]),
                    "crop_window_policy_id": str(config.crop_window_policy_id),
                }
                sample_ids.append(
                    hashlib.sha256(
                        json.dumps(
                            payload,
                            ensure_ascii=True,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest()
                )
        if len(sample_ids) != self._length or len(set(sample_ids)) != self._length:
            raise ValueError("Franka sample identity inventory is incomplete")
        self.sample_tasks = tuple(sample_tasks)
        self.sample_ids = tuple(sample_ids)
        self.sample_episode_ids = tuple(sample_episode_ids)

    def __len__(self) -> int:
        return self._length

    @property
    def sample_signatures(self) -> tuple[tuple[int, int], ...]:
        return self._sample_signatures

    def set_epoch(self, epoch: int) -> None:
        for dataset in self._datasets:
            dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        for dataset, offset in zip(reversed(self._datasets), reversed(self._offsets)):
            if index >= offset:
                sample = dataset[index - offset]
                forbidden = {
                    key
                    for key in sample
                    if key.startswith("tactile_") and key != "tactile_cond_drop"
                }
                if forbidden:
                    raise RuntimeError(
                        f"Franka sample contains tactile tensors: {sorted(forbidden)}"
                    )
                return sample
        raise RuntimeError("unreachable Franka dataset index")


class FrankaLatentLeRobotDataset(LatentLeRobotDataset):
    """Build next-end-pose EE20 targets while retaining the released 20D head."""

    def __init__(self, repo_id: str, config=None) -> None:
        _validate_config(config)
        self.franka_slots_per_frame = 6
        super().__init__(repo_id=repo_id, config=config)
        view = _load_view(config)
        self._view_entries_by_episode = {
            entry.lerobot_episode_id: entry for entry in view.entries
        }
        seen = {int(meta["episode_index"]) for meta in self.new_metas}
        expected = set(self._view_entries_by_episode)
        if seen != expected:
            raise ValueError(
                "Franka active view has missing/unexpected usable episodes: "
                f"missing={sorted(expected - seen)}, unexpected={sorted(seen - expected)}"
            )

    def _sample_action_slots_per_frame(self) -> int:
        return self.franka_slots_per_frame

    def _hf_data_columns(self) -> list[str]:
        return ["action", "observation.state"]

    def _action_post_process_with_context(
        self,
        local_start_frame,
        local_end_frame,
        latent_frame_ids,
        data_dict,
    ):
        if "observation.state" not in data_dict:
            raise ValueError(
                "Franka samples require observation.state for cold anchoring"
            )
        targets = build_franka_ee10_latent_targets(
            local_start_frame=int(local_start_frame),
            local_end_frame=int(local_end_frame),
            latent_frame_ids=latent_frame_ids,
            converted_actions=data_dict["action"],
            converted_states=data_dict["observation.state"],
            action_q01=np.asarray(self.config.norm_stat["q01"], dtype=np.float32)[:10],
            action_q99=np.asarray(self.config.norm_stat["q99"], dtype=np.float32)[:10],
            expected_slots_per_frame=self.franka_slots_per_frame,
        )
        return (
            torch.from_numpy(targets.actions).float(),
            torch.from_numpy(targets.valid_mask).bool(),
        )


__all__ = (
    "DATASET_ADAPTER",
    "FrankaLatentLeRobotDataset",
    "MultiLatentLeRobotFrankaDataset",
)
