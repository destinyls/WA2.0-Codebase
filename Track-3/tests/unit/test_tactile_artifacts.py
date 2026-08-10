# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt
import pytest
import torch

from n0_twam.evaluation.tactile_artifacts import save_tactile_metric_frames


def test_decoded_five_frame_latent_window_binds_seventeen_pixel_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_frame_counts = []

    def _write(
        path: Path,
        frames: Iterable[npt.NDArray[np.uint8]],
        *,
        fps: int,
    ) -> str:
        assert fps == 15
        selected = tuple(frames)
        encoded_frame_counts.append((path, len(selected)))
        return "libx264"

    monkeypatch.setattr(
        "n0_twam.evaluation.tactile_artifacts.write_rgb_mp4_pyav",
        _write,
    )
    prediction = torch.zeros(1, 3, 17, 16, 16, dtype=torch.uint8)
    ground_truth = torch.ones_like(prediction)

    result = save_tactile_metric_frames(
        prediction,
        ground_truth,
        tmp_path,
        task="lift_bottle",
        sample_id="sample_000000",
        sensor="sensor_0",
        fps=15,
    )

    assert result["decoded_frame_count"] == 17
    assert result["written_frame_indices"] == list(range(17))
    assert result["metric_scored_frame_indices"] == list(range(1, 17))
    assert result["reference_layout_mp4_frame_count"] == 16
    assert [count for _, count in encoded_frame_counts] == [16, 16]
    prediction_frames = sorted(
        (tmp_path / "prediction/lift_bottle/sample_000000/sensor_0").glob("frame_*.png")
    )
    assert len(prediction_frames) == 17
