# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed latent dataset adapter for AgileX qpos14 repositories."""

from __future__ import annotations

import copy
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from n0_twam.actions import build_action_codec_from_config
from n0_twam.data.qpos14_alignment import build_qpos14_latent_targets
from n0_twam.tactile_profiles import (
    MIXED,
    VISION_ONLY,
    VISION_TACTILE,
    validate_tactile_profile_config,
)

from .agilex_sample_contract import (
    ACTION_SCHEMA,
    DATASET_ADAPTER,
    build_action_index_grid,
    build_agilex_sample_shape_signature,
    canonicalize_agilex_sample,
    content_addressed_contact_drop,
    validate_digest,
)
from .sample_shape_signature import SampleShapeSignature

try:
    from .lerobot_latent_dataset import LatentLeRobotDataset, recursive_find_file
except ModuleNotFoundError as error:  # pragma: no cover - optional unit environment
    if error.name != "lerobot":
        raise

    class LatentLeRobotDataset:  # type: ignore[no-redef]
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise ImportError("LeRobot is required for the AgileX dataset adapter")

    def recursive_find_file(*_args: object, **_kwargs: object) -> list[str]:
        raise ImportError("LeRobot is required for the AgileX dataset adapter")


def _validate_config(config: object) -> None:
    expected = {
        "dataset_adapter": DATASET_ADAPTER,
        "action_schema": ACTION_SCHEMA,
        "action_dim": 14,
    }
    for field, value in expected.items():
        if getattr(config, field, None) != value:
            raise ValueError(f"AgileX dataset config mismatch for {field}")
    if list(getattr(config, "used_action_channel_ids", [])) != list(range(14)):
        raise ValueError("AgileX active action channels must be exactly 0..13")
    per_repo_offsets = getattr(config, "per_repo_action_offsets_per_anchor", None)
    if not isinstance(per_repo_offsets, dict) or not per_repo_offsets:
        raise ValueError(
            "AgileX config requires per-repository action offsets per anchor"
        )
    horizon = int(getattr(config, "action_per_frame", -1))
    for repo_name, raw_offsets in per_repo_offsets.items():
        offsets = tuple(raw_offsets)
        if (
            not isinstance(repo_name, str)
            or not repo_name
            or not offsets
            or len(offsets) != horizon
            or any(type(value) is not int for value in offsets)
            or offsets[0] != 0
            or any(right <= left for left, right in zip(offsets, offsets[1:]))
        ):
            raise ValueError("AgileX config has invalid per-repository action offsets")
    profile = getattr(config, "tactile_profile", None)
    if profile not in (VISION_TACTILE, MIXED, VISION_ONLY):
        raise ValueError("AgileX dataset requires a canonical tactile profile")
    for field in ("repo_route_manifest_sha256", "temporal_alignment_contract_sha256"):
        validate_digest(str(getattr(config, field, "")), label=field)
    selected = tuple(getattr(config, "selected_repo_names", ()))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("AgileX config requires unique selected repositories")
    for field in (
        "per_repo_action_offsets_per_anchor",
        "per_repo_route_identity",
        "per_repo_temporal_alignment_identity",
        "per_repo_wrench_keys",
    ):
        mapping = getattr(config, field, None)
        if not isinstance(mapping, dict) or set(mapping) != set(selected):
            raise ValueError(f"AgileX config has incomplete {field}")
    for field in (
        "per_repo_route_identity",
        "per_repo_temporal_alignment_identity",
    ):
        for value in getattr(config, field).values():
            validate_digest(str(value), label=field)
    normalizer = getattr(config, "norm_stat", None)
    if not isinstance(normalizer, dict):
        raise ValueError("AgileX dataset requires q01/q99 normalizer")
    for field in ("q01", "q99"):
        values = np.asarray(normalizer.get(field), dtype=np.float32)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError(f"AgileX normalizer {field} must be 14 finite values")
    _validate_wrench_routing(config, selected_repo_names=selected)


def _validate_wrench_routing(
    config: object,
    *,
    selected_repo_names: tuple[str, ...],
) -> None:
    """Validate the signed wrench roster shared by dataset and serving."""

    per_repo = getattr(config, "per_repo_wrench_keys")
    routed_keys: set[str] = set()
    for repo_name in selected_repo_names:
        raw_keys = per_repo[repo_name]
        if not isinstance(raw_keys, (list, tuple)):
            raise ValueError("AgileX per-repository wrench keys must be sequences")
        keys = tuple(raw_keys)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("AgileX wrench routes must contain non-empty strings")
        if len(keys) != len(set(keys)):
            raise ValueError("AgileX wrench routes must not contain duplicates")
        routed_keys.update(keys)

    raw_sensor_map = getattr(config, "wrench_sensor_id_map", None)
    if not isinstance(raw_sensor_map, dict):
        raise ValueError("AgileX wrench_sensor_id_map must be a mapping")
    sensor_map = dict(raw_sensor_map)
    if set(sensor_map) != routed_keys:
        raise ValueError(
            "AgileX wrench_sensor_id_map must exactly cover routed wrench keys"
        )

    wrench_arm_count = getattr(config, "wrench_arm_count", None)
    max_wrench_streams = getattr(config, "max_wrench_streams", None)
    if (
        type(wrench_arm_count) is not int
        or wrench_arm_count <= 0
        or type(max_wrench_streams) is not int
        or max_wrench_streams != wrench_arm_count
    ):
        raise ValueError(
            "AgileX wrench_arm_count and max_wrench_streams must be equal "
            "positive integers"
        )
    sensor_ids = tuple(sensor_map.values())
    if any(
        type(sensor_id) is not int or sensor_id < 0 or sensor_id >= wrench_arm_count
        for sensor_id in sensor_ids
    ):
        raise ValueError(
            "AgileX wrench sensor IDs must be integers within wrench_arm_count"
        )
    if len(sensor_ids) != len(set(sensor_ids)):
        raise ValueError("AgileX wrench sensor IDs must be unique")


def _construct(repo_id: str, config: object) -> "AgileXLatentLeRobotDataset":
    return AgileXLatentLeRobotDataset(repo_id=repo_id, config=config)


class AgileXLatentLeRobotDataset(LatentLeRobotDataset):  # type: ignore[misc]
    """Align row-native qpos14 actions and emit canonical contact masks."""

    def __init__(self, repo_id: str, config: Any = None) -> None:
        _validate_config(config)
        self.qpos14_codec = build_action_codec_from_config(config)
        repo_name = Path(repo_id).name
        per_repo_offsets = config.per_repo_action_offsets_per_anchor
        self.action_offsets_per_anchor = tuple(
            int(value) for value in per_repo_offsets.get(repo_name, ())
        )
        if not self.action_offsets_per_anchor:
            raise ValueError(
                f"AgileX repo has no action offset contract: {repo_name!r}"
            )
        self.wrench_arm_count = config.wrench_arm_count
        self.max_wrench_streams = self.wrench_arm_count
        self.wrench_sensor_id_map = dict(config.wrench_sensor_id_map)
        self._range_context: dict[str, Any] | None = None
        self._aligned_context: dict[str, Any] | None = None
        super().__init__(repo_id=repo_id, config=config)
        per_repo_wrench = getattr(config, "per_repo_wrench_keys", None) or {}
        self.used_wrench_keys = tuple(per_repo_wrench.get(self.repo_name, ()))
        available = set(getattr(self.hf_dataset, "column_names", ()))
        required = {"action", "action.valid", *self.used_wrench_keys}
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"AgileX converted repo is missing columns: {missing}")
        self._hf_agilex_view = self.hf_dataset.with_format(
            type="torch", columns=sorted(required), output_all_columns=False
        )
        route_ids = getattr(config, "per_repo_route_identity", None) or {}
        temporal_ids = (
            getattr(config, "per_repo_temporal_alignment_identity", None) or {}
        )
        self.repo_route_identity = validate_digest(
            str(route_ids.get(self.repo_name, "")), label="repo route identity"
        )
        self.temporal_alignment_identity = validate_digest(
            str(temporal_ids.get(self.repo_name, "")),
            label="temporal alignment identity",
        )

    def _sample_action_slots_per_frame(self) -> int:
        return len(self.action_offsets_per_anchor)

    def _get_range_hf_data(self, start_frame: int, end_frame: int) -> dict[str, Any]:
        if self._range_context is not None:
            raise RuntimeError("AgileX dataset item loading is not reentrant")
        batch = self._hf_agilex_view[start_frame:end_frame]
        self._range_context = dict(batch)
        return {"action": batch["action"]}

    def _action_post_process(
        self,
        local_start_frame: int,
        local_end_frame: int,
        latent_frame_ids: npt.ArrayLike,
        action: npt.ArrayLike,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self._range_context
        if context is None:
            raise RuntimeError("AgileX action alignment has no source context")
        row_valid = np.asarray(context["action.valid"]).reshape(-1)
        anchors = np.asarray(latent_frame_ids, dtype=np.int64).reshape(-1)[::4]
        indices, valid = build_action_index_grid(
            latent_anchor_row_ids=anchors,
            action_offsets_per_anchor=self.action_offsets_per_anchor,
            action_row_valid=row_valid,
            action_row_count=int(np.asarray(action).shape[0]),
            action_index_origin=int(local_start_frame),
        )
        targets = build_qpos14_latent_targets(
            converted_actions=action,
            codec=self.qpos14_codec,
            action_indices_per_anchor=indices,
            action_valid_mask=valid,
            latent_anchor_row_ids=anchors,
            action_index_origin=int(local_start_frame),
        )
        wrench = torch.zeros(
            (len(anchors), self.max_wrench_streams, 6), dtype=torch.float32
        )
        wrench_available = torch.zeros(
            (len(anchors), self.max_wrench_streams), dtype=torch.bool
        )
        relative_anchors = anchors - int(local_start_frame)
        if np.any(relative_anchors < 0) or np.any(relative_anchors >= len(row_valid)):
            raise ValueError("AgileX wrench anchors exceed the source range")
        for key in self.used_wrench_keys:
            slot = self.wrench_sensor_id_map[key]
            values = torch.as_tensor(context[key], dtype=torch.float32)
            selected = values[torch.as_tensor(relative_anchors, dtype=torch.long)]
            if (
                selected.shape != (len(anchors), 6)
                or not torch.isfinite(selected).all()
            ):
                raise ValueError(f"AgileX wrench column is invalid: {key}")
            wrench[:, slot] = selected
            wrench_available[:, slot] = True
        self._aligned_context = {
            "wrench": wrench,
            "wrench_available_mask": wrench_available,
        }
        return (
            torch.from_numpy(targets.actions.copy()).float(),
            torch.from_numpy(targets.model_valid_mask.copy()).bool(),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        meta = self.new_metas[index]
        try:
            sample = super().__getitem__(index)
            aligned = self._aligned_context
            if aligned is None:
                raise RuntimeError("AgileX item did not complete action alignment")
            frames = int(sample["actions"].shape[1])
            if "tactile_global_latent" in sample:
                sensors = int(sample["tactile_global_latent"].shape[0])
                tactile_available = torch.ones((frames, sensors), dtype=torch.bool)
            else:
                tactile_available = torch.zeros((frames, 0), dtype=torch.bool)
            profile = str(self.config.tactile_profile)
            has_tactile = sensors > 0 if "tactile_global_latent" in sample else False
            if profile == VISION_ONLY or not has_tactile:
                contact_drop = True
            elif profile == VISION_TACTILE:
                contact_drop = False
            else:
                contact_drop = content_addressed_contact_drop(
                    seed=int(getattr(self.config, "seed", 0)),
                    epoch=int(self.crop_epoch),
                    repo_id=self.repo_name,
                    episode_id=int(meta["episode_index"]),
                    frame_anchor=int(meta["start_frame"]),
                    probability=float(getattr(self.config, "tactile_cfg_prob", 0.1)),
                )
            return canonicalize_agilex_sample(
                sample,
                tactile_available_mask=tactile_available,
                wrench=aligned["wrench"],
                wrench_available_mask=aligned["wrench_available_mask"],
                contact_cond_drop=contact_drop,
                repo_route_identity=self.repo_route_identity,
                temporal_alignment_identity=self.temporal_alignment_identity,
            )
        finally:
            self._range_context = None
            self._aligned_context = None


class MultiLatentLeRobotAgileXDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    """Multiple explicitly routed AgileX repositories."""

    def __init__(self, config: Any, num_init_worker: int = 8) -> None:
        _validate_config(config)
        self.max_wrench_streams = int(getattr(config, "max_wrench_streams", 0))
        paths = sorted(
            str(Path(path).parents[1])
            for path in recursive_find_file(config.dataset_path, "info.json")
        )
        if not paths:
            raise ValueError("no AgileX LeRobot repositories found")
        worker_count = max(
            1, min(int(getattr(config, "num_init_worker", num_init_worker)), len(paths))
        )
        constructor = partial(_construct, config=copy.copy(config))
        if worker_count == 1:
            self._datasets = [constructor(path) for path in paths]
        else:
            with Pool(worker_count) as pool:
                self._datasets = pool.map(constructor, paths)
        validate_tactile_profile_config(
            config, repo_names=(dataset.repo_name for dataset in self._datasets)
        )
        self._offsets: list[int] = []
        total = 0
        for dataset in self._datasets:
            self._offsets.append(total)
            total += len(dataset)
        self._length = total
        if total == 0:
            raise ValueError("AgileX dataset contains no valid latent segments")
        self._sample_signatures: tuple[SampleShapeSignature, ...] | None = None

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        for dataset in self._datasets:
            dataset.set_epoch(epoch)

    @property
    def sample_signatures(self) -> tuple[SampleShapeSignature, ...]:
        """Return the exact post-canonicalization signature for every sample."""

        if self._sample_signatures is None:
            signatures = tuple(
                build_agilex_sample_shape_signature(
                    signature,
                    max_wrench_streams=self.max_wrench_streams,
                )
                for dataset in self._datasets
                for signature in dataset.sample_signatures
            )
            if len(signatures) != self._length:
                raise ValueError(
                    "AgileX child sample-signature count differs from dataset length"
                )
            self._sample_signatures = signatures
        return self._sample_signatures

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        for dataset, offset in zip(reversed(self._datasets), reversed(self._offsets)):
            if index >= offset:
                return dataset[index - offset]
        raise RuntimeError("unreachable AgileX dataset index")


__all__ = (
    "ACTION_SCHEMA",
    "DATASET_ADAPTER",
    "AgileXLatentLeRobotDataset",
    "MultiLatentLeRobotAgileXDataset",
    "build_action_index_grid",
    "build_agilex_sample_shape_signature",
    "canonicalize_agilex_sample",
    "content_addressed_contact_drop",
)
