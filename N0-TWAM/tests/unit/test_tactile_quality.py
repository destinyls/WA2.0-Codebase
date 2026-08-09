# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from n0_twam.evaluation.tactile_quality import (
    INTERNAL_OFFLINE_PROTOCOL,
    TACTILE_METRIC_DOMAIN,
    evaluate_tactile_prediction_quality,
)


def _write_rgb(root: Path, relative_path: str, value: int) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def test_tactile_quality_scores_future_frames_and_real_tasks(tmp_path: Path) -> None:
    prediction_root = tmp_path / "prediction"
    ground_truth_root = tmp_path / "ground_truth"
    for root in (prediction_root, ground_truth_root):
        _write_rgb(root, "insert_HDMI/episode_000/tactile_a/frame_000.png", 10)
        _write_rgb(root, "insert_HDMI/episode_000/tactile_a/frame_001.png", 20)
        _write_rgb(root, "lift_bottle/episode_000/tactile_b/frame_000.png", 30)
        _write_rgb(root, "lift_bottle/episode_000/tactile_b/frame_001.png", 40)

    report = evaluate_tactile_prediction_quality(
        prediction_root=prediction_root,
        ground_truth_root=ground_truth_root,
        skip_first_frames=1,
    )

    assert report["protocol"] == INTERNAL_OFFLINE_PROTOCOL
    assert report["leaderboard_compatible"] is False
    assert report["aggregation"] == "macro_over_videos"
    assert report["overall"]["video_count"] == 2
    assert report["overall"]["frame_count"] == 2
    assert report["overall"]["psnr"] == "Infinity"
    assert report["overall"]["ssim"] == pytest.approx(1.0)
    assert report["pixel_domain"] == TACTILE_METRIC_DOMAIN
    assert set(report["by_task"]) == {"insert_HDMI", "lift_bottle"}
    assert report["official_reference"]["public_frame_policy"] == (
        "scores_every_frame_present_in_each_input_mp4"
    )
    assert "not_exact_official_metric" in report["official_reference"]["compatibility"]


def test_mixed_perfect_and_error_frames_have_finite_robust_psnr(
    tmp_path: Path,
) -> None:
    prediction_root = tmp_path / "prediction"
    ground_truth_root = tmp_path / "ground_truth"
    base = "insert_HDMI/episode_000/tactile_a"
    for root in (prediction_root, ground_truth_root):
        _write_rgb(root, f"{base}/frame_000.png", 0)
        _write_rgb(root, f"{base}/frame_001.png", 20)
    _write_rgb(prediction_root, f"{base}/frame_002.png", 0)
    _write_rgb(ground_truth_root, f"{base}/frame_002.png", 10)

    report = evaluate_tactile_prediction_quality(
        prediction_root=prediction_root,
        ground_truth_root=ground_truth_root,
    )

    assert math.isfinite(report["overall"]["psnr"])
    assert report["overall"]["perfect_psnr_frame_count"] == 1
    assert report["official_reference_macro"]["average_psnr"] == "Infinity"


def test_macro_over_videos_is_not_frame_weighted(tmp_path: Path) -> None:
    prediction_root = tmp_path / "prediction"
    ground_truth_root = tmp_path / "ground_truth"
    videos = {
        "insert_HDMI/episode_000/tactile_a": (10, 1),
        "lift_bottle/episode_001/tactile_a": (100, 3),
    }
    for base, (error, scored_frames) in videos.items():
        for root in (prediction_root, ground_truth_root):
            _write_rgb(root, f"{base}/frame_000.png", 0)
        for index in range(1, scored_frames + 1):
            _write_rgb(prediction_root, f"{base}/frame_{index:03d}.png", 0)
            _write_rgb(ground_truth_root, f"{base}/frame_{index:03d}.png", error)

    report = evaluate_tactile_prediction_quality(
        prediction_root=prediction_root,
        ground_truth_root=ground_truth_root,
    )
    expected = (20 * math.log10(255 / 10) + 20 * math.log10(255 / 100)) / 2

    assert report["overall"]["psnr"] == pytest.approx(expected)
    assert report["overall"]["video_count"] == 2


def test_tactile_quality_detects_frame_set_mismatch(tmp_path: Path) -> None:
    prediction_root = tmp_path / "prediction"
    ground_truth_root = tmp_path / "ground_truth"
    base = "insert_HDMI/episode_000/tactile_a"
    for root in (prediction_root, ground_truth_root):
        _write_rgb(root, f"{base}/frame_000.png", 0)
        _write_rgb(root, f"{base}/frame_001.png", 10)
    _write_rgb(ground_truth_root, f"{base}/frame_002.png", 0)

    with pytest.raises(ValueError, match="frame sets differ"):
        evaluate_tactile_prediction_quality(
            prediction_root=prediction_root,
            ground_truth_root=ground_truth_root,
        )


def test_tactile_quality_rejects_non_contiguous_frame_indices(tmp_path: Path) -> None:
    prediction_root = tmp_path / "prediction"
    ground_truth_root = tmp_path / "ground_truth"
    base = "insert_HDMI/episode_000/tactile_a"
    for root in (prediction_root, ground_truth_root):
        _write_rgb(root, f"{base}/frame_000.png", 0)
        _write_rgb(root, f"{base}/frame_002.png", 10)

    with pytest.raises(ValueError, match="contiguous"):
        evaluate_tactile_prediction_quality(
            prediction_root=prediction_root,
            ground_truth_root=ground_truth_root,
        )
