# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""PSNR, SSIM, and position metrics for Franka future predictions."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
import numpy.typing as npt
from skimage.metrics import structural_similarity

from n0_twam.evaluation.franka_atomic_io import publish_atomic_file
from n0_twam.evaluation.franka_prediction_artifact import (
    PREDICTION_ARTIFACT_TYPE,
    PREDICTION_SCHEMA_VERSION,
    capture_prediction_input,
    load_franka_predictions,
    require_prediction_unchanged,
)
from n0_twam.evaluation.sealed_artifact_io import canonical_json, validate_sha256
from n0_twam.integrations.worldarena.franka_views import build_standard_franka_views

REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT_TYPE = "n0_twam_track32_offline_metrics"
PSNR_CAP_DB = 100.0
_METRIC_NAMES = ("psnr_db", "ssim", "position_mae_cm", "position_rmse_cm")
_STRUCTURAL_SIMILARITY = cast(Callable[..., float], structural_similarity)
_SSIM_PARAMETERS = {
    "channel_axis": -1,
    "data_range": 255,
    "win_size": 7,
    "gaussian_weights": False,
    "use_sample_covariance": True,
    "K1": 0.01,
    "K2": 0.03,
}


def _rgb_metrics(
    prediction: npt.NDArray[np.uint8], target: npt.NDArray[np.uint8]
) -> tuple[float, float]:
    delta = prediction.astype(np.float64) / 255.0 - target.astype(np.float64) / 255.0
    mse = float(np.mean(np.square(delta)))
    psnr = min(PSNR_CAP_DB, -10.0 * np.log10(max(mse, 10.0**-10)))
    ssim = float(_STRUCTURAL_SIMILARITY(target, prediction, **_SSIM_PARAMETERS))
    if not np.isfinite(psnr) or not np.isfinite(ssim):
        raise ValueError("PSNR/SSIM evaluation returned a non-finite value")
    return float(psnr), ssim


def _mean(
    records: Sequence[Mapping[str, float]],
    *,
    names: Sequence[str] = _METRIC_NAMES,
) -> dict[str, float]:
    if not records:
        raise ValueError("cannot aggregate an empty metric group")
    return {
        name: float(np.mean([record[name] for record in records])) for name in names
    }


def _write_new_json(path: Path, payload: object) -> str:
    raw = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )

    def _writer(handle: BinaryIO) -> None:
        handle.write(raw)

    def _validator(candidate: bytes) -> None:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("metric report must contain one JSON object")

    published = publish_atomic_file(
        output=path,
        writer=_writer,
        validator=_validator,
        label="metric report",
    )
    return published.sha256


def _require_expected_identity(
    metadata: Mapping[str, object],
    *,
    checkpoint_identity_sha256: str,
    dataset_view_id: str,
    dataset_view_sha256: str,
    decoder_sha256: str,
) -> None:
    expected = {
        "checkpoint_identity_sha256": validate_sha256(
            checkpoint_identity_sha256, label="expected checkpoint identity SHA256"
        ),
        "dataset_view_id": dataset_view_id,
        "dataset_view_sha256": validate_sha256(
            dataset_view_sha256, label="expected dataset view SHA256"
        ),
        "decoder_sha256": validate_sha256(
            decoder_sha256, label="expected decoder SHA256"
        ),
    }
    if not isinstance(dataset_view_id, str) or not dataset_view_id:
        raise ValueError("expected dataset view ID must be non-empty")
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"prediction artifact {field} does not match expectation")


def evaluate_franka_offline_predictions(
    *,
    predictions: Path,
    output: Path,
    checkpoint_identity_sha256: str,
    dataset_view_id: str,
    dataset_view_sha256: str,
    decoder_sha256: str,
) -> dict[str, object]:
    """Score a frozen future-prediction artifact and seal a JSON report."""

    snapshot = capture_prediction_input(predictions)
    destination_input = Path(output).expanduser()
    if destination_input.exists() or destination_input.is_symlink():
        raise FileExistsError(f"metric report already exists: {destination_input}")
    destination = destination_input.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"metric report already exists: {destination}")
    artifact = load_franka_predictions(snapshot.raw)
    _require_expected_identity(
        artifact.metadata,
        checkpoint_identity_sha256=checkpoint_identity_sha256,
        dataset_view_id=dataset_view_id,
        dataset_view_sha256=dataset_view_sha256,
        decoder_sha256=decoder_sha256,
    )

    per_sample: list[dict[str, object]] = []
    all_metrics: list[dict[str, float]] = []
    task_metrics: defaultdict[str, list[Mapping[str, float]]] = defaultdict(list)
    view_metrics: defaultdict[str, list[Mapping[str, float]]] = defaultdict(list)
    valid_rgb_frames = 0
    valid_actions = 0
    for sample_index, (sample_id, task_id) in enumerate(
        zip(artifact.sample_ids, artifact.task_ids)
    ):
        frame_scores: list[tuple[float, float]] = []
        for view_index, view_name in enumerate(artifact.view_names):
            view_scores = [
                _rgb_metrics(
                    artifact.predicted_rgb[sample_index, view_index, frame_index],
                    artifact.target_rgb[sample_index, view_index, frame_index],
                )
                for frame_index in np.flatnonzero(
                    artifact.video_valid[sample_index, view_index]
                )
            ]
            frame_scores.extend(view_scores)
            view_metrics[view_name].append(
                {
                    "psnr_db": float(np.mean([score[0] for score in view_scores])),
                    "ssim": float(np.mean([score[1] for score in view_scores])),
                    "position_mae_cm": 0.0,
                    "position_rmse_cm": 0.0,
                }
            )
        action_rows = artifact.action_valid[sample_index]
        position_delta = artifact.predicted_end_pose[
            sample_index, action_rows, :3
        ].astype(np.float64) - artifact.target_end_pose[
            sample_index, action_rows, :3
        ].astype(
            np.float64
        )
        distances_cm = np.linalg.norm(position_delta, axis=1) * 100.0
        metrics = {
            "psnr_db": float(np.mean([score[0] for score in frame_scores])),
            "ssim": float(np.mean([score[1] for score in frame_scores])),
            "position_mae_cm": float(np.mean(distances_cm)),
            "position_rmse_cm": float(np.sqrt(np.mean(np.square(distances_cm)))),
        }
        per_sample.append(
            {
                "sample_id": sample_id,
                "task_id": task_id,
                "valid_rgb_frame_count": len(frame_scores),
                "valid_action_count": int(action_rows.sum()),
                **metrics,
            }
        )
        all_metrics.append(metrics)
        task_metrics[task_id].append(metrics)
        valid_rgb_frames += len(frame_scores)
        valid_actions += int(action_rows.sum())

    standard_view = build_standard_franka_views().get(
        str(artifact.metadata["dataset_view_id"])
    )
    canonical_view_roster_match = standard_view is not None and (
        artifact.metadata["dataset_view_sha256"] == standard_view.view_sha256
        and tuple(artifact.lerobot_episode_ids)
        == tuple(entry.lerobot_episode_id for entry in standard_view.entries)
        and tuple(artifact.task_ids)
        == tuple(entry.task for entry in standard_view.entries)
    )
    core: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "status": "complete",
        "execution_tier": "offline_reference_metrics",
        "organizer_evaluation_completed": False,
        "real_robot_evaluation_completed": False,
        "prediction_artifact": str(snapshot.path),
        "prediction_artifact_sha256": snapshot.sha256,
        "prediction_contract": dict(artifact.metadata),
        "metric_contract": {
            "rgb_range": "uint8_0_255",
            "conditioning_frame_included": False,
            "psnr": "per_frame_rgb_mse_then_episode_macro_mean",
            "psnr_cap_db": PSNR_CAP_DB,
            "ssim": "skimage_structural_similarity_explicit_parameters",
            "ssim_backend": f"scikit-image-{version('scikit-image')}",
            "ssim_parameters": dict(_SSIM_PARAMETERS),
            "position_mae_cm": "episode_mean_l2_xyz_error_cm_then_macro_mean",
            "position_rmse_cm": "episode_root_mean_squared_l2_xyz_error_cm_then_macro_mean",
            "aggregation": "episode_macro_mean",
        },
        "sample_count": len(per_sample),
        "valid_rgb_frame_count": valid_rgb_frames,
        "valid_action_count": valid_actions,
        "canonical_view_roster_match": canonical_view_roster_match,
        "internal_holdout_complete": False,
        "generalization_claim_valid": False,
        "generalization_claim_reason": (
            "offline prediction NPZ does not independently prove canonical target bytes; "
            "use these metrics as non-organizer proxy evidence"
        ),
        "overall": _mean(all_metrics),
        "per_task": {
            task: {"sample_count": len(task_metrics[task]), **_mean(task_metrics[task])}
            for task in sorted(task_metrics)
        },
        "per_view": {
            view: {
                "sample_count": len(view_metrics[view]),
                **_mean(view_metrics[view], names=("psnr_db", "ssim")),
            }
            for view in artifact.view_names
        },
        "per_sample": per_sample,
    }
    semantic_core = {
        key: value for key, value in core.items() if key != "prediction_artifact"
    }
    report = {
        **core,
        "report_identity_sha256": hashlib.sha256(
            canonical_json(semantic_core)
        ).hexdigest(),
    }
    require_prediction_unchanged(snapshot)
    report_file_sha256 = _write_new_json(destination, report)
    return {
        **report,
        "report": str(destination.resolve(strict=True)),
        "report_file_sha256": report_file_sha256,
    }


__all__ = (
    "PREDICTION_ARTIFACT_TYPE",
    "PREDICTION_SCHEMA_VERSION",
    "PSNR_CAP_DB",
    "REPORT_SCHEMA_VERSION",
    "evaluate_franka_offline_predictions",
)
