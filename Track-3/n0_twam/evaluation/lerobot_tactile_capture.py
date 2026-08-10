# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Capture the exact converted LeRobot-H264 tactile rows used by Stage A."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt

TACTILE_FRAME_COUNT = 17
TACTILE_FRAME_SHAPE = (128, 128, 3)
CANONICAL_TACTILE_KEYS = (
    "observation.images.tactile_a",
    "observation.images.tactile_b",
)


def _resolve_lerobot_video(
    dataset_root: Path,
    *,
    feature_name: str,
    episode_index: int,
) -> Path:
    candidates = sorted(
        path.resolve(strict=True)
        for path in (dataset_root / "videos").glob(
            f"chunk-*/{feature_name}/episode_{episode_index:06d}.mp4"
        )
        if path.is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            "Stage-A requires exactly one converted LeRobot H264 video for "
            f"{feature_name}/episode_{episode_index:06d}, found {len(candidates)}"
        )
    return candidates[0]


def _decode_h264_rows(
    video_path: Path, source_row_ids: npt.NDArray[np.int64]
) -> npt.NDArray[np.uint8]:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise ImportError("PyAV is required for Stage-A LeRobot H264 capture") from exc
    decoded = []
    with av.open(str(video_path)) as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(
                f"Stage-A H264 must contain one video stream: {video_path}"
            )
        for frame in container.decode(streams[0]):
            rgb = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
            decoded.append(np.ascontiguousarray(resized))
    if not decoded or int(source_row_ids[-1]) >= len(decoded):
        raise ValueError(f"Stage-A H264 rows are incomplete: {video_path}")
    selected = np.stack([decoded[int(row_id)] for row_id in source_row_ids])
    if selected.shape != (TACTILE_FRAME_COUNT, *TACTILE_FRAME_SHAPE):
        raise RuntimeError("Stage-A H264 capture returned an invalid RGB video")
    if selected.dtype != np.uint8:
        raise RuntimeError("Stage-A H264 capture must remain uint8")
    return np.ascontiguousarray(selected)


def capture_lerobot_h264(
    *,
    dataset_root: Path,
    episode_index: int,
    source_row_ids: npt.NDArray[np.int64],
    tactile_keys: Sequence[str],
) -> npt.NDArray[np.uint8]:
    """Return left/right rows as uint8 ``[2,17,128,128,3]``."""

    if tuple(tactile_keys) != CANONICAL_TACTILE_KEYS:
        raise ValueError("Stage-A requires the canonical left/right tactile key order")
    return np.stack(
        [
            _decode_h264_rows(
                _resolve_lerobot_video(
                    dataset_root,
                    feature_name=feature_name,
                    episode_index=episode_index,
                ),
                source_row_ids,
            )
            for feature_name in tactile_keys
        ]
    )
