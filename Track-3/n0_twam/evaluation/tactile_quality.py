# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Internal decoded-residual tactile metrics with video-macro aggregation."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt
from PIL import Image
from skimage.metrics import structural_similarity

IMAGE_SUFFIXES = frozenset((".png", ".jpg", ".jpeg"))
TACTILE_METRIC_DOMAIN = "decoded_global_tactile_residual_rgb_uint8"
INTERNAL_OFFLINE_PROTOCOL = "internal_offline"
OFFICIAL_REFERENCE_URL = "https://v2.world-arena.ai/"
OFFICIAL_REFERENCE_PATH = "visual-tactile_world_model_pipeline/metric/val_psnr_ssim.py"


@dataclass(frozen=True)
class TactileImagePair:
    relative_path: str
    task: str
    episode: str
    sensor: str
    prediction_path: Path
    ground_truth_path: Path


@dataclass(frozen=True)
class TactileFrameMetric:
    relative_path: str
    task: str
    episode: str
    sensor: str
    mse: float
    psnr: float
    ssim: float


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    )


def _discover_images(root: Path) -> dict[str, Path]:
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(resolved_root)
    images = {
        path.relative_to(resolved_root).as_posix(): path
        for path in resolved_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not images:
        raise ValueError(f"no tactile images found under {resolved_root}")
    return images


def _pair_metadata(relative_path: str) -> tuple[str, str, str]:
    parts = Path(relative_path).parts
    if len(parts) != 4:
        raise ValueError(
            "tactile metric layout must be task/episode/sensor/frame.ext, got "
            f"{relative_path!r}"
        )
    return parts[0], parts[1], parts[2]


def discover_tactile_image_pairs(
    *,
    prediction_root: Path,
    ground_truth_root: Path,
    skip_first_frames: int = 1,
) -> tuple[TactileImagePair, ...]:
    """Match exact paths and exclude conditioning frames from every video."""

    if skip_first_frames < 1:
        raise ValueError("internal offline evaluation must skip its conditioning frame")
    predictions = _discover_images(prediction_root)
    ground_truth = _discover_images(ground_truth_root)
    if predictions.keys() != ground_truth.keys():
        missing = sorted(ground_truth.keys() - predictions.keys(), key=_natural_key)
        unexpected = sorted(predictions.keys() - ground_truth.keys(), key=_natural_key)
        raise ValueError(
            "prediction/ground-truth frame sets differ: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    grouped: dict[str, list[str]] = defaultdict(list)
    for relative_path in predictions:
        _pair_metadata(relative_path)
        grouped[Path(relative_path).parent.as_posix()].append(relative_path)
    selected: list[str] = []
    for parent in sorted(grouped, key=_natural_key):
        ordered = sorted(grouped[parent], key=_natural_key)
        frame_indices = []
        for relative_path in ordered:
            match = re.fullmatch(r"frame_(\d+)", Path(relative_path).stem)
            if match is None:
                raise ValueError(f"invalid tactile frame name: {relative_path}")
            frame_indices.append(int(match.group(1)))
        if frame_indices != list(range(len(frame_indices))):
            raise ValueError(
                f"tactile frame indices must be contiguous from zero: {parent}"
            )
        selected.extend(ordered[skip_first_frames:])
    if not selected:
        raise ValueError("no scored tactile frames remain after condition-frame skip")

    pairs = []
    for relative_path in selected:
        task, episode, sensor = _pair_metadata(relative_path)
        pairs.append(
            TactileImagePair(
                relative_path=relative_path,
                task=task,
                episode=episode,
                sensor=sensor,
                prediction_path=predictions[relative_path],
                ground_truth_path=ground_truth[relative_path],
            )
        )
    return tuple(pairs)


def _load_rgb(path: Path) -> npt.NDArray[np.uint8]:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"expected RGB tactile image, got {image.mode}: {path}")
        array = np.asarray(image, dtype=np.uint8).copy()
    if min(array.shape[:2]) < 11:
        raise ValueError(f"SSIM requires tactile image dimensions >= 11: {path}")
    return array


def _psnr_from_mse(mse: float) -> float:
    if mse == 0.0:
        return math.inf
    return float(10.0 * math.log10((255.0**2) / mse))


def score_tactile_pair(pair: TactileImagePair) -> TactileFrameMetric:
    prediction = _load_rgb(pair.prediction_path)
    ground_truth = _load_rgb(pair.ground_truth_path)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"tactile frame shape mismatch for {pair.relative_path}: "
            f"{prediction.shape} vs {ground_truth.shape}"
        )
    error = prediction.astype(np.float64) - ground_truth.astype(np.float64)
    mse = float(np.mean(np.square(error)))
    ssim = structural_similarity(
        ground_truth,
        prediction,
        channel_axis=-1,
        data_range=255,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    return TactileFrameMetric(
        relative_path=pair.relative_path,
        task=pair.task,
        episode=pair.episode,
        sensor=pair.sensor,
        mse=mse,
        psnr=_psnr_from_mse(mse),
        ssim=float(ssim),
    )


def _json_number(value: float) -> float | str:
    return "Infinity" if math.isinf(value) else float(value)


def _summary_float(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        raise TypeError("video summary value is not numeric")
    return float(value)


def _video_key(metric: TactileFrameMetric) -> tuple[str, str, str]:
    return metric.task, metric.episode, metric.sensor


def _video_summary(metrics: Iterable[TactileFrameMetric]) -> dict[str, object]:
    selected = tuple(metrics)
    if not selected:
        raise ValueError("cannot aggregate an empty video")
    mean_mse = float(np.mean([metric.mse for metric in selected]))
    frame_psnr_mean = float(np.mean([metric.psnr for metric in selected]))
    return {
        "frame_count": len(selected),
        "mse": mean_mse,
        "psnr_aggregate_mse": _json_number(_psnr_from_mse(mean_mse)),
        "psnr_frame_mean_official_formula": _json_number(frame_psnr_mean),
        "ssim_frame_mean": float(np.mean([metric.ssim for metric in selected])),
        "perfect_psnr_frame_count": sum(math.isinf(metric.psnr) for metric in selected),
    }


def _group_videos(
    metrics: Iterable[TactileFrameMetric],
) -> dict[tuple[str, str, str], list[TactileFrameMetric]]:
    grouped: dict[tuple[str, str, str], list[TactileFrameMetric]] = defaultdict(list)
    for metric in metrics:
        grouped[_video_key(metric)].append(metric)
    return grouped


def _macro_aggregate(metrics: Iterable[TactileFrameMetric]) -> dict[str, object]:
    selected = tuple(metrics)
    videos = _group_videos(selected)
    if not videos:
        raise ValueError("cannot aggregate an empty metric set")
    summaries = [_video_summary(items) for items in videos.values()]
    robust_psnr = [_summary_float(item["psnr_aggregate_mse"]) for item in summaries]
    ssim_values = [_summary_float(item["ssim_frame_mean"]) for item in summaries]
    perfect_counts = [
        int(_summary_float(item["perfect_psnr_frame_count"])) for item in summaries
    ]
    return {
        "video_count": len(summaries),
        "frame_count": len(selected),
        "psnr": _json_number(float(np.mean(robust_psnr))),
        "ssim": float(np.mean(ssim_values)),
        "perfect_psnr_frame_count": sum(perfect_counts),
        "psnr_method": "per_video_aggregate_mse_then_macro_over_videos",
        "ssim_method": "per_video_frame_mean_then_macro_over_videos",
    }


def _official_reference_macro(
    metrics: Iterable[TactileFrameMetric],
) -> dict[str, object]:
    videos = _group_videos(metrics)
    summaries = [_video_summary(items) for items in videos.values()]
    psnr = [
        _summary_float(item["psnr_frame_mean_official_formula"]) for item in summaries
    ]
    ssim_values = [_summary_float(item["ssim_frame_mean"]) for item in summaries]
    return {
        "video_count": len(summaries),
        "average_psnr": _json_number(float(np.mean(psnr))),
        "average_ssim": float(np.mean(ssim_values)),
        "aggregation": "macro_over_videos_after_per_video_frame_mean",
        "numeric_compatibility": "not_claimed_for_internal_residual_domain",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_tactile_prediction_quality(
    *,
    prediction_root: Path,
    ground_truth_root: Path,
    skip_first_frames: int = 1,
) -> dict[str, object]:
    """Evaluate internal residual videos; this cannot attest closed-loop evidence."""

    pairs = discover_tactile_image_pairs(
        prediction_root=prediction_root,
        ground_truth_root=ground_truth_root,
        skip_first_frames=skip_first_frames,
    )
    metrics = tuple(score_tactile_pair(pair) for pair in pairs)
    by_task: dict[str, list[TactileFrameMetric]] = defaultdict(list)
    by_sensor: dict[str, list[TactileFrameMetric]] = defaultdict(list)
    identity = hashlib.sha256()
    for pair, metric in zip(pairs, metrics):
        by_task[metric.task].append(metric)
        by_sensor[metric.sensor].append(metric)
        identity.update(pair.relative_path.encode("utf-8"))
        identity.update(_sha256_file(pair.prediction_path).encode("ascii"))
        identity.update(_sha256_file(pair.ground_truth_path).encode("ascii"))

    per_video = []
    for (task, episode, sensor), items in sorted(
        _group_videos(metrics).items(), key=lambda item: _natural_key("/".join(item[0]))
    ):
        per_video.append(
            {
                "video_id": f"{task}/{episode}/{sensor}",
                "task": task,
                "episode": episode,
                "sensor": sensor,
                **_video_summary(items),
            }
        )
    return {
        "schema_version": 1,
        "metric": "tactile_prediction_quality",
        "protocol": INTERNAL_OFFLINE_PROTOCOL,
        "leaderboard_compatible": False,
        "pixel_domain": TACTILE_METRIC_DOMAIN,
        "psnr_data_range": 255,
        "aggregation": "macro_over_videos",
        "official_reference": {
            "url": OFFICIAL_REFERENCE_URL,
            "local_path": OFFICIAL_REFERENCE_PATH,
            "directory_contract": "generate_videos/ and gt_videos/ same-name mp4",
            "aggregation": "macro_over_videos_after_per_video_frame_mean",
            "max_val": 255,
            "public_frame_policy": "scores_every_frame_present_in_each_input_mp4",
            "internal_frame_policy": "frame_0_condition_excluded_before_scoring",
            "compatibility": (
                "directory_and_aggregation_reference_only_not_exact_official_metric"
            ),
        },
        "ssim": {
            "implementation": "skimage.metrics.structural_similarity",
            "gaussian_weights": True,
            "sigma": 1.5,
            "use_sample_covariance": False,
            "channel_axis": -1,
            "data_range": 255,
        },
        "skip_first_frames_per_video": skip_first_frames,
        "input_pairs_sha256": identity.hexdigest(),
        "overall": _macro_aggregate(metrics),
        "official_reference_macro": _official_reference_macro(metrics),
        "by_task": {
            key: _macro_aggregate(value) for key, value in sorted(by_task.items())
        },
        "by_sensor": {
            key: _macro_aggregate(value) for key, value in sorted(by_sensor.items())
        },
        "per_video": per_video,
        "frames": [
            {
                "relative_path": metric.relative_path,
                "task": metric.task,
                "episode": metric.episode,
                "sensor": metric.sensor,
                "mse": metric.mse,
                "psnr": _json_number(metric.psnr),
                "ssim": metric.ssim,
            }
            for metric in metrics
        ],
    }
