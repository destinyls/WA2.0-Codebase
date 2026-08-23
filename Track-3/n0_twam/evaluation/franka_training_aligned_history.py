# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Reconstruct the causal RGB prefix used by the frozen Franka encoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from n0_twam.data.video_decode import decode_sampled_video_frames
from n0_twam.integrations.univtac.dataset_view import (
    select_content_addressed_crop_start,
)


@dataclass(frozen=True)
class TrainingAlignedVideoHistory:
    """One immutable 10 Hz, ``4*k+1`` causal RGB prefix."""

    frames: tuple[Mapping[str, npt.NDArray[np.uint8]], ...]
    episode_id: int
    crop_latent_index: int
    source_frame_ids: tuple[int, ...]


def _local_dataset(dataset: Any, sample_index: int) -> tuple[Any, int]:
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(sample_index)
    for child, offset in zip(reversed(dataset._datasets), reversed(dataset._offsets)):
        if sample_index >= offset:
            return child, sample_index - offset
    raise RuntimeError("unreachable Franka dataset index")


def build_training_aligned_video_history(
    dataset: Any,
    *,
    sample_index: int,
    max_latent_frames: int,
) -> TrainingAlignedVideoHistory:
    """Decode the exact training prefix ending at a cropped latent frame."""

    if max_latent_frames <= 0:
        raise ValueError("max_latent_frames must be positive")
    child, local_index = _local_dataset(dataset, sample_index)
    meta = child.new_metas[local_index]
    episode_id = int(meta["episode_index"])
    start_frame = int(meta["start_frame"])
    end_frame = int(meta["end_frame"])
    if start_frame != 0 or end_frame <= start_frame:
        raise ValueError("training-aligned Franka evaluation requires full episodes")

    latent_data = child._get_range_latent_data(start_frame, end_frame, episode_id)
    camera_keys = tuple(child.used_video_keys)
    if not camera_keys:
        raise ValueError("Franka evaluation has no RGB camera keys")
    first_key = camera_keys[0]
    num_latent_frames = int(latent_data[f"{first_key}.latent_num_frames"])
    if num_latent_frames > max_latent_frames:
        if child.crop_window_policy_id != "epoch_content_addressed_crop_v1":
            raise ValueError("evaluation requires content-addressed latent cropping")
        entry = child._view_entries_by_episode[episode_id]
        crop_latent_index = select_content_addressed_crop_start(
            entry,
            seed=child.crop_seed,
            epoch=child.crop_epoch,
            num_latent_frames=num_latent_frames,
            max_latent_frames=max_latent_frames,
        )
    else:
        crop_latent_index = 0

    prefix_length = crop_latent_index * 4 + 1
    decoded_by_key: dict[str, npt.NDArray[np.uint8]] = {}
    reference_frame_ids: tuple[int, ...] | None = None
    for key in camera_keys:
        payload_frame_ids = tuple(int(value) for value in latent_data[f"{key}.frame_ids"])
        if len(payload_frame_ids) < prefix_length:
            raise ValueError("cached latent has an incomplete RGB frame inventory")
        source_video = (
            Path(child.root)
            / "videos"
            / "chunk-000"
            / key
            / f"episode_{episode_id:06d}.mp4"
        )
        frames, frame_ids = decode_sampled_video_frames(
            source_video=source_video,
            start_timestamp=0.0,
            end_timestamp=(end_frame - start_frame) / int(latent_data[f"{key}.ori_fps"]),
            length=end_frame - start_frame,
            width=int(latent_data[f"{key}.video_width"]),
            height=int(latent_data[f"{key}.video_height"]),
            target_fps=int(latent_data[f"{key}.fps"]),
            ori_fps=int(latent_data[f"{key}.ori_fps"]),
        )
        decoded_ids = tuple(int(value) for value in frame_ids)
        if decoded_ids != payload_frame_ids:
            raise ValueError("raw RGB decode no longer matches cached latent frame IDs")
        if reference_frame_ids is None:
            reference_frame_ids = decoded_ids
        elif decoded_ids != reference_frame_ids:
            raise ValueError("Franka cameras use different training frame grids")
        decoded_by_key[key] = np.ascontiguousarray(frames[:prefix_length])

    history = tuple(
        {
            key: np.ascontiguousarray(decoded_by_key[key][frame_index])
            for key in camera_keys
        }
        for frame_index in range(prefix_length)
    )
    if len(history) % 4 != 1 or reference_frame_ids is None:
        raise RuntimeError("invalid training-aligned RGB history")
    return TrainingAlignedVideoHistory(
        frames=history,
        episode_id=episode_id,
        crop_latent_index=crop_latent_index,
        source_frame_ids=reference_frame_ids[:prefix_length],
    )


__all__ = ("TrainingAlignedVideoHistory", "build_training_aligned_video_history")
