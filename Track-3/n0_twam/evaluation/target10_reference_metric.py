# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict Target-10 adapter around the pinned Stage-1 tactile metric."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np
import numpy.typing as npt

from n0_twam.evaluation.sealed_artifact_io import sha256_file
from n0_twam.evaluation.target10_reference_contract import (
    FUTURE_FRAME_INDICES,
    REFERENCE_CONTRACT,
    TARGET_EPISODE_IDS,
    TARGET_TASKS,
    validate_reference_metric,
)


@dataclass(frozen=True)
class ReferenceVideoPair:
    """One exact Target-10 prediction/ground-truth tactile video pair."""

    task: str
    sample_id: str
    sample_index: int
    episode_id: int
    prediction_path: Path
    ground_truth_path: Path


def _load_metric_module(path: Path) -> ModuleType:
    resolved = validate_reference_metric(path)
    module_name = f"n0_twam_reference_metric_{sha256_file(resolved)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the pinned reference metric")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "TOTAL_FRAMES": 9,
        "FUTURE_FRAME_INDICES": FUTURE_FRAME_INDICES,
        "SSIM_BACKEND": REFERENCE_CONTRACT.ssim_backend,
        "METRIC_DOMAIN": REFERENCE_CONTRACT.metric_domain,
        "PSNR_CAP_DB": REFERENCE_CONTRACT.psnr_cap_db,
    }
    for key, value in expected.items():
        if getattr(module, key, None) != value:
            raise ValueError(f"pinned reference metric constant drift: {key}")
    shapes = getattr(module, "VIDEO_SPATIAL_SHAPES", None)
    if not isinstance(shapes, dict) or shapes.get("tactile") != (192, 512):
        raise ValueError("pinned reference metric tactile shape has drifted")
    for function_name in (
        "evaluate_pair",
        "read_video_rgb",
        "_frame_psnr",
        "_frame_ssim",
    ):
        if not callable(getattr(module, function_name, None)):
            raise ValueError(f"pinned reference metric lacks {function_name}")
    return module


def _validate_pairs(
    pairs: Sequence[ReferenceVideoPair],
) -> tuple[ReferenceVideoPair, ...]:
    selected = tuple(pairs)
    expected = tuple(
        (task, f"sample_{sample_index:03d}", sample_index, episode_id)
        for task in TARGET_TASKS
        for sample_index, episode_id in enumerate(TARGET_EPISODE_IDS)
    )
    actual = tuple(
        (pair.task, pair.sample_id, pair.sample_index, pair.episode_id)
        for pair in selected
    )
    if actual != expected:
        raise ValueError(
            "reference video pairs do not match the ordered Target-10 roster"
        )
    for pair in selected:
        for path in (pair.prediction_path, pair.ground_truth_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"reference metric input must be a regular file: {path}"
                )
    return selected


def _sensor_metrics(
    module: ModuleType,
    prediction: npt.NDArray[np.uint8],
    ground_truth: npt.NDArray[np.uint8],
) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    for sensor_index, sensor_name in enumerate(REFERENCE_CONTRACT.sensor_order):
        start = sensor_index * 256
        stop = start + 256
        pred_sensor = prediction[:, :, start:stop]
        gt_sensor = ground_truth[:, :, start:stop]
        frame_psnr = tuple(
            float(module._frame_psnr(pred_sensor[index], gt_sensor[index]))
            for index in FUTURE_FRAME_INDICES
        )
        frame_ssim = tuple(
            float(module._frame_ssim(pred_sensor[index], gt_sensor[index]))
            for index in FUTURE_FRAME_INDICES
        )
        diagnostics[sensor_name] = {
            "frame_psnr": list(frame_psnr),
            "frame_ssim": list(frame_ssim),
            "average_psnr": float(np.mean(frame_psnr)),
            "average_ssim": float(np.mean(frame_ssim)),
        }
    return diagnostics


def _numeric(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"reference metric row {key} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"reference metric row {key} is non-finite")
    return result


def evaluate_reference_video_pairs(
    *,
    pairs: Sequence[ReferenceVideoPair],
    metric_script: Path,
) -> dict[str, object]:
    """Evaluate exactly 80 decoded future-frame pairs and macro by task."""

    selected = _validate_pairs(pairs)
    module = _load_metric_module(metric_script)
    episode_rows: list[dict[str, object]] = []
    for pair in selected:
        metric = module.evaluate_pair(
            pair.prediction_path,
            pair.ground_truth_path,
            "tactile",
        )
        frame_psnr = tuple(float(value) for value in metric.frame_psnr)
        frame_ssim = tuple(float(value) for value in metric.frame_ssim)
        if (
            metric.total_frames != 9
            or tuple(metric.evaluated_frame_indices) != FUTURE_FRAME_INDICES
            or len(frame_psnr) != 8
            or len(frame_ssim) != 8
            or not all(math.isfinite(value) for value in (*frame_psnr, *frame_ssim))
        ):
            raise ValueError("pinned reference metric returned an invalid pair result")
        prediction = module.read_video_rgb(pair.prediction_path)
        ground_truth = module.read_video_rgb(pair.ground_truth_path)
        episode_rows.append(
            {
                "task": pair.task,
                "sample_id": pair.sample_id,
                "sample_index": pair.sample_index,
                "episode_id": pair.episode_id,
                "prediction_file": pair.prediction_path.name,
                "ground_truth_file": pair.ground_truth_path.name,
                "prediction_sha256": sha256_file(pair.prediction_path),
                "ground_truth_sha256": sha256_file(pair.ground_truth_path),
                "total_frames": 9,
                "evaluated_frame_indices": list(FUTURE_FRAME_INDICES),
                "frame_psnr": list(frame_psnr),
                "frame_ssim": list(frame_ssim),
                "average_psnr": float(np.mean(frame_psnr)),
                "average_ssim": float(np.mean(frame_ssim)),
                "per_sensor_diagnostic": _sensor_metrics(
                    module, prediction, ground_truth
                ),
            }
        )
    task_rows: list[dict[str, object]] = []
    for task in TARGET_TASKS:
        group = [row for row in episode_rows if row["task"] == task]
        if len(group) != 5:
            raise ValueError("reference task aggregation is incomplete")
        task_rows.append(
            {
                "task": task,
                "episode_count": 5,
                "future_frame_pairs": 40,
                "average_psnr": float(
                    np.mean([_numeric(row, "average_psnr") for row in group])
                ),
                "average_ssim": float(
                    np.mean([_numeric(row, "average_ssim") for row in group])
                ),
            }
        )
    overall = {
        "task_count": 2,
        "episode_count": 10,
        "future_frame_pairs": 80,
        "average_psnr": float(
            np.mean([_numeric(row, "average_psnr") for row in task_rows])
        ),
        "average_ssim": float(
            np.mean([_numeric(row, "average_ssim") for row in task_rows])
        ),
    }
    return {
        "metric_script_sha256": sha256_file(validate_reference_metric(metric_script)),
        "domain": REFERENCE_CONTRACT.metric_domain,
        "ssim_backend": REFERENCE_CONTRACT.ssim_backend,
        "psnr_cap_db": REFERENCE_CONTRACT.psnr_cap_db,
        "aggregation": REFERENCE_CONTRACT.aggregation,
        "counts": {
            "episodes": len(episode_rows),
            "future_frame_pairs": len(episode_rows) * len(FUTURE_FRAME_INDICES),
        },
        "per_episode": episode_rows,
        "per_task": task_rows,
        "overall": overall,
    }


__all__ = ("ReferenceVideoPair", "evaluate_reference_video_pairs")
