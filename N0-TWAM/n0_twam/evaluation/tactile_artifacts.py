# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Write decoded tactile pairs and bind the exact frame indices emitted."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from n0_twam.evaluation.video_writer import write_rgb_mp4_pyav


def _safe_component(value: str, *, label: str) -> str:
    if Path(value).name != value or value in {"", ".", ".."}:
        raise ValueError(f"{label} must be one safe path component")
    return value


def _rgb_frames(
    video: torch.Tensor, frame_indices: range
) -> list[npt.NDArray[np.uint8]]:
    frames = []
    for frame_index in frame_indices:
        array = video[0, :, frame_index].permute(1, 2, 0).numpy()
        if array.dtype != np.uint8:
            raise ValueError("decoded tactile metric frames must be uint8")
        frames.append(cast(npt.NDArray[np.uint8], array))
    return frames


def save_tactile_metric_frames(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    output_dir: Path,
    *,
    task: str,
    sample_id: str,
    sensor: str,
    fps: int,
) -> dict[str, object]:
    """Write paired PNGs and verified future-only MP4s for one tactile stream."""

    if prediction.shape != ground_truth.shape or prediction.ndim != 5:
        raise ValueError("decoded tactile prediction/ground truth shapes must match")
    if prediction.shape[0] != 1 or prediction.shape[1] != 3:
        raise ValueError("decoded tactile videos must have shape [1,3,F,H,W]")
    decoded_frame_count = int(prediction.shape[2])
    if decoded_frame_count <= 1:
        raise ValueError("tactile metric output requires a scored future frame")
    task = _safe_component(task, label="task")
    sample_id = _safe_component(sample_id, label="sample_id")
    sensor = _safe_component(sensor, label="sensor")
    root = Path(output_dir)
    written_indices = list(range(decoded_frame_count))
    for root_name, video in (
        ("prediction", prediction),
        ("ground_truth", ground_truth),
    ):
        target = root / root_name / task / sample_id / sensor
        target.mkdir(parents=True, exist_ok=True)
        for frame_index, frame in zip(
            written_indices,
            _rgb_frames(video, range(decoded_frame_count)),
        ):
            Image.fromarray(frame).save(target / f"frame_{frame_index:06d}.png")

    scored_indices = list(range(1, decoded_frame_count))
    video_name = f"{task}__{sample_id}__{sensor}.mp4"
    codecs = {}
    for directory_name, video in (
        ("generate_videos", prediction),
        ("gt_videos", ground_truth),
    ):
        directory = root / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        codecs[directory_name] = write_rgb_mp4_pyav(
            directory / video_name,
            _rgb_frames(video, range(1, decoded_frame_count)),
            fps=fps,
        )
    return {
        "decoded_frame_count": decoded_frame_count,
        "written_frame_indices": written_indices,
        "metric_scored_frame_indices": scored_indices,
        "reference_layout_mp4_frame_count": len(scored_indices),
        "verified_h264_codecs": codecs,
    }
