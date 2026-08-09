# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict view-bound qpos8 next-step dataset adapter for UniVTAC Track 3.1."""

import copy
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch

from n0_twam.actions import build_action_codec_from_config
from n0_twam.data.qpos8_alignment import build_qpos8_latent_targets
from n0_twam.integrations.univtac.dataset_view import (
    DatasetView,
    load_dataset_view,
)
from n0_twam.integrations.univtac.view_binding import build_view_sample_metadata

from .lerobot_latent_dataset import LatentLeRobotDataset, recursive_find_file


def _construct_qpos8_dataset(repo_id: str, config) -> "Qpos8LatentLeRobotDataset":
    return Qpos8LatentLeRobotDataset(repo_id=repo_id, config=config)


def _load_active_view(config: object) -> DatasetView | None:
    raw_path = getattr(config, "dataset_view_path", None)
    if raw_path is None:
        return None
    view = load_dataset_view(Path(str(raw_path)))
    expected_view_id = getattr(config, "train_view_id", None)
    if view.view_id != expected_view_id:
        raise ValueError(
            f"dataset view ID mismatch: {view.view_id!r} vs {expected_view_id!r}"
        )
    expected_manifest = getattr(config, "source_manifest_sha256", None)
    if (
        expected_manifest is not None
        and view.source_manifest_sha256 != expected_manifest
    ):
        raise ValueError("dataset view source manifest differs from the normalizer")
    dataset_path = Path(str(getattr(config, "dataset_path"))).resolve(strict=True)
    if dataset_path.name != view.physical_split:
        raise ValueError(
            "dataset view physical split does not match dataset_path: "
            f"{view.physical_split!r} vs {dataset_path.name!r}"
        )
    return view


def _construct_all_qpos8_datasets(config, num_init_worker: int) -> list:
    repo_paths = sorted(
        str(Path(path).parents[1])
        for path in recursive_find_file(config.dataset_path, "info.json")
    )
    if not repo_paths:
        raise ValueError(f"no LeRobot repositories found under {config.dataset_path}")
    active_view = _load_active_view(config)
    if active_view is not None:
        expected_root = Path(str(config.dataset_path)).resolve(strict=True)
        resolved_repositories = tuple(
            Path(path).resolve(strict=True) for path in repo_paths
        )
        if resolved_repositories != (expected_root,):
            raise ValueError(
                "view-bound qpos8 training requires exactly the declared physical "
                f"repository and rejects recursively discovered extras: "
                f"{resolved_repositories!r}"
            )
        view_config = copy.copy(config)
        view_config.selected_lerobot_episode_ids = tuple(
            entry.lerobot_episode_id for entry in active_view.entries
        )
        view_config.active_dataset_view = active_view
        config = view_config
    worker_count = max(1, min(int(num_init_worker), len(repo_paths)))
    constructor = partial(_construct_qpos8_dataset, config=config)
    if worker_count == 1:
        return [constructor(path) for path in repo_paths]
    with Pool(worker_count) as pool:
        return pool.map(constructor, repo_paths)


class MultiLatentLeRobotQpos8Dataset(torch.utils.data.Dataset):
    """Fail-closed collection of qpos8 LeRobot repositories."""

    def __init__(self, config, num_init_worker: int = 8) -> None:
        worker_count = int(getattr(config, "num_init_worker", num_init_worker))
        self._datasets = _construct_all_qpos8_datasets(config, worker_count)
        self.dataset_view = _load_active_view(config)
        self._offsets: list[int] = []
        total = 0
        for dataset in self._datasets:
            self._offsets.append(total)
            total += len(dataset)
        self._length = total
        if self._length == 0:
            raise ValueError("qpos8 dataset contains no valid latent segments")
        self._sample_signatures: tuple[tuple[int, int], ...] = tuple(
            signature
            for dataset in self._datasets
            for signature in [
                (len(dataset.used_video_keys), len(dataset.used_tactile_keys))
            ]
            for _ in range(len(dataset))
        )
        if self.dataset_view is not None:
            metadata = build_view_sample_metadata(
                view=self.dataset_view,
                datasets=self._datasets,
                crop_window_policy_id=str(
                    getattr(
                        config,
                        "crop_window_policy_id",
                        "checkpointed_rng_stream_crop_v1",
                    )
                ),
            )
            self.sample_tasks = metadata["sample_tasks"]
            self.sample_ids = metadata["sample_ids"]
            self.sample_source_episode_ids = metadata["sample_source_episode_ids"]
            self.sample_relative_paths = metadata["sample_relative_paths"]
            dimensions = (
                self._sample_signatures,
                self.sample_tasks,
                self.sample_ids,
                self.sample_source_episode_ids,
                self.sample_relative_paths,
            )
            if any(len(values) != self._length for values in dimensions):
                raise RuntimeError("view-bound qpos8 sample metadata is incomplete")

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        """Propagate sampler epoch to deterministic per-episode crop selection."""
        for dataset in self._datasets:
            dataset.set_epoch(epoch)

    @property
    def sample_signatures(self) -> tuple[tuple[int, int], ...]:
        return self._sample_signatures

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        for dataset, offset in zip(
            reversed(self._datasets),
            reversed(self._offsets),
        ):
            if index >= offset:
                return dataset[index - offset]
        raise RuntimeError("unreachable qpos8 dataset index")


class Qpos8LatentLeRobotDataset(LatentLeRobotDataset):
    """Latent N0 sample builder for converted ``joint[t+1]`` actions."""

    def __init__(self, repo_id: str, config=None) -> None:
        self.qpos8_codec = build_action_codec_from_config(config)
        if self.qpos8_codec.spec.name != "qpos8_next_step":
            raise ValueError("Qpos8 dataset requires action_schema=qpos8_next_step")
        self.qpos8_slots_per_frame = int(getattr(config, "action_per_frame", 4))
        super().__init__(repo_id=repo_id, config=config)

    def _action_post_process(
        self,
        local_start_frame,
        local_end_frame,
        latent_frame_ids,
        action,
    ):
        targets = build_qpos8_latent_targets(
            local_start_frame=int(local_start_frame),
            local_end_frame=int(local_end_frame),
            latent_frame_ids=latent_frame_ids,
            converted_actions=action,
            codec=self.qpos8_codec,
            expected_slots_per_frame=self.qpos8_slots_per_frame,
        )
        return (
            torch.from_numpy(targets.actions).float(),
            torch.from_numpy(targets.valid_mask).bool(),
        )
