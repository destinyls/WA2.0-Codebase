# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Hashable tensor-shape identities for heterogeneous latent samples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LatentTensorSpec:
    """Metadata needed to derive one decoded latent tensor shape."""

    channels: int
    frames: int
    height: int
    width: int
    dtype: str

    def validate(self, *, label: str) -> None:
        dimensions = (self.channels, self.frames, self.height, self.width)
        if any(isinstance(value, bool) or int(value) <= 0 for value in dimensions):
            raise ValueError(f"{label} dimensions must be positive integers")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError(f"{label} dtype must be non-empty")


TensorShapeEntry = tuple[str, tuple[int, ...], str]
SampleShapeSignature = tuple[TensorShapeEntry, ...]


def episode_selection_sha256(episode_ids: Sequence[int] | None) -> str | None:
    """Return a stable identity for one explicit episode selection."""

    if episode_ids is None:
        return None
    normalized = []
    for episode_id in episode_ids:
        if isinstance(episode_id, bool) or int(episode_id) < 0:
            raise ValueError("episode IDs must be non-negative integers")
        normalized.append(int(episode_id))
    raw = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _uniform_video_shape(
    specs: Sequence[LatentTensorSpec],
) -> tuple[int, int, int, int, str]:
    if not specs:
        raise ValueError("a latent sample requires at least one video stream")
    for index, spec in enumerate(specs):
        spec.validate(label=f"video[{index}]")
    first = specs[0]
    for spec in specs[1:]:
        if (
            spec.channels,
            spec.frames,
            spec.height,
            spec.dtype,
        ) != (
            first.channels,
            first.frames,
            first.height,
            first.dtype,
        ):
            raise ValueError(
                "video streams must share channel/frame/height/dtype for concat"
            )
    return (
        first.channels,
        first.frames,
        first.height,
        sum(spec.width for spec in specs),
        first.dtype,
    )


def _uniform_tactile_shape(
    global_specs: Sequence[LatentTensorSpec],
    local_specs: Sequence[LatentTensorSpec],
    *,
    video_frames: int,
) -> tuple[int, int, int, int, int, str] | None:
    if not global_specs and not local_specs:
        return None
    if not global_specs or len(global_specs) != len(local_specs):
        raise ValueError("global/local tactile stream rosters must match")
    for index, (global_spec, local_spec) in enumerate(
        zip(global_specs, local_specs, strict=True)
    ):
        global_spec.validate(label=f"tactile_global[{index}]")
        local_spec.validate(label=f"tactile_local[{index}]")
        if global_spec != local_spec:
            raise ValueError("global/local tactile tensor shapes must match exactly")
        if global_spec.frames != video_frames:
            raise ValueError("tactile/video latent frame counts must match")
    first = global_specs[0]
    if any(spec != first for spec in global_specs[1:]):
        raise ValueError("tactile sensors must have identical stackable shapes")
    return (
        len(global_specs),
        first.channels,
        first.frames,
        first.height,
        first.width,
        first.dtype,
    )


def build_sample_shape_signature(
    *,
    video_specs: Sequence[LatentTensorSpec],
    text_shape: Sequence[int],
    text_dtype: str,
    frame_ids: Sequence[int],
    max_latent_frames: int,
    action_dim: int,
    action_slots_per_frame: int | None = None,
    tactile_global_specs: Sequence[LatentTensorSpec] = (),
    tactile_local_specs: Sequence[LatentTensorSpec] = (),
    raw_tactile_evaluation: bool = False,
) -> SampleShapeSignature:
    """Return the exact output tensor roster used by default collation."""

    channels, full_frames, height, total_width, video_dtype = _uniform_video_shape(
        video_specs
    )
    tactile_shape = _uniform_tactile_shape(
        tactile_global_specs,
        tactile_local_specs,
        video_frames=full_frames,
    )
    if isinstance(max_latent_frames, bool) or int(max_latent_frames) < 0:
        raise ValueError("max_latent_frames must be a non-negative integer")
    if isinstance(action_dim, bool) or int(action_dim) <= 0:
        raise ValueError("action_dim must be a positive integer")
    resolved_frame_ids = tuple(int(value) for value in frame_ids)
    if len(resolved_frame_ids) < 2:
        raise ValueError("frame_ids must contain at least two entries")
    deltas = tuple(
        right - left
        for left, right in zip(
            resolved_frame_ids,
            resolved_frame_ids[1:],
            strict=False,
        )
    )
    if not deltas or any(value <= 0 for value in deltas):
        raise ValueError("frame_ids must be strictly increasing")

    effective_frames = full_frames
    effective_frame_id_count = len(resolved_frame_ids)
    if max_latent_frames > 0 and full_frames > max_latent_frames:
        effective_frames = int(max_latent_frames)
        effective_frame_id_count = (effective_frames - 1) * 4 + 1
        if effective_frame_id_count > len(resolved_frame_ids):
            raise ValueError("frame_ids cannot represent the requested latent crop")
    action_frames = (effective_frame_id_count - 1) // 4 + 1
    if action_slots_per_frame is None:
        if any(value != deltas[0] for value in deltas):
            raise ValueError(
                "the generic action aligner requires uniformly spaced frame_ids"
            )
        action_slots = deltas[0] * 4
    else:
        if isinstance(action_slots_per_frame, bool) or int(action_slots_per_frame) <= 0:
            raise ValueError("action_slots_per_frame must be a positive integer")
        action_slots = int(action_slots_per_frame)
        anchors = resolved_frame_ids[:effective_frame_id_count:4]
        anchor_deltas = tuple(
            right - left for left, right in zip(anchors, anchors[1:], strict=False)
        )
        if anchor_deltas and any(value != anchor_deltas[0] for value in anchor_deltas):
            raise ValueError("latent action anchors must be uniformly spaced")

    entries: list[TensorShapeEntry] = [
        ("actions", (int(action_dim), action_frames, action_slots, 1), "torch.float32"),
        (
            "actions_mask",
            (int(action_dim), action_frames, action_slots, 1),
            "torch.bool",
        ),
        ("latents", (channels, effective_frames, height, total_width), video_dtype),
        ("tactile_cond_drop", (), "torch.bool"),
        ("text_emb", tuple(int(value) for value in text_shape), str(text_dtype)),
    ]
    if tactile_shape is not None:
        sensors, tactile_channels, _, tactile_height, tactile_width, tactile_dtype = (
            tactile_shape
        )
        tactile_output_shape = (
            sensors,
            tactile_channels,
            effective_frames,
            tactile_height,
            tactile_width,
        )
        entries.extend(
            [
                ("tactile_global_latent", tactile_output_shape, tactile_dtype),
                ("tactile_local_latent", tactile_output_shape, tactile_dtype),
                ("tactile_sensor_ids", (sensors,), "torch.int64"),
            ]
        )
    if raw_tactile_evaluation:
        entries.extend(
            [
                ("lerobot_episode_index", (), "torch.int64"),
                ("source_row_ids", (effective_frame_id_count,), "torch.int64"),
                ("source_step_ids", (effective_frame_id_count,), "torch.int64"),
            ]
        )
    return tuple(sorted(entries))


__all__ = (
    "LatentTensorSpec",
    "SampleShapeSignature",
    "TensorShapeEntry",
    "build_sample_shape_signature",
    "episode_selection_sha256",
)
