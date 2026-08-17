# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Offline PSNR/SSIM and native qpos14 metrics for AgileX predictions."""

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

from n0_twam.evaluation.agilex_prediction_artifact import (
    capture_prediction_input,
    load_agilex_predictions,
    require_prediction_unchanged,
)
from n0_twam.evaluation.franka_atomic_io import publish_atomic_file
from n0_twam.evaluation.sealed_artifact_io import canonical_json, validate_sha256

REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT_TYPE = "n0_twam_track32_agilex_offline_metrics"
PSNR_CAP_DB = 100.0
_METRICS = ("psnr_db", "ssim", "qpos14_mae", "qpos14_rmse")
_SSIM = cast(Callable[..., float], structural_similarity)
_SSIM_PARAMETERS = {
    "channel_axis": -1,
    "data_range": 255,
    "win_size": 7,
    "gaussian_weights": False,
    "use_sample_covariance": True,
    "K1": 0.01,
    "K2": 0.03,
}
_QPOS_GROUPS = {
    "left_arm": tuple(range(0, 6)),
    "left_gripper": (6,),
    "right_arm": tuple(range(7, 13)),
    "right_gripper": (13,),
}


def _rgb(
    prediction: npt.NDArray[np.uint8], target: npt.NDArray[np.uint8]
) -> tuple[float, float]:
    delta = prediction.astype(np.float64) / 255.0 - target.astype(np.float64) / 255.0
    mse = float(np.mean(np.square(delta)))
    psnr = min(PSNR_CAP_DB, -10.0 * np.log10(max(mse, 10.0**-10)))
    ssim = float(_SSIM(target, prediction, **_SSIM_PARAMETERS))
    if not np.isfinite((psnr, ssim)).all():
        raise ValueError("AgileX RGB metrics returned non-finite values")
    return float(psnr), ssim


def _mean(
    records: Sequence[Mapping[str, float]],
    *,
    names: Sequence[str] = _METRICS,
) -> dict[str, float]:
    if not records:
        raise ValueError("cannot aggregate an empty AgileX metric group")
    return {
        name: float(np.mean([record[name] for record in records])) for name in names
    }


def _qpos_metrics(delta: npt.NDArray[np.float64]) -> tuple[float, float]:
    return float(np.mean(np.abs(delta))), float(np.sqrt(np.mean(np.square(delta))))


def _write_json(path: Path, payload: object) -> str:
    raw = (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )

    def writer(handle: BinaryIO) -> None:
        handle.write(raw)

    def validator(candidate: bytes) -> None:
        if not isinstance(json.loads(candidate), dict):
            raise ValueError("AgileX metric report must be a JSON object")

    return publish_atomic_file(
        output=path,
        writer=writer,
        validator=validator,
        label="AgileX metric report",
    ).sha256


def _require_identity(
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
        raise ValueError("expected AgileX dataset view ID must be non-empty")
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"prediction artifact {field} does not match expectation")


def evaluate_agilex_offline_predictions(
    *,
    predictions: Path,
    output: Path,
    checkpoint_identity_sha256: str,
    dataset_view_id: str,
    dataset_view_sha256: str,
    decoder_sha256: str,
) -> dict[str, object]:
    """Score one frozen AgileX prediction artifact and seal its report."""

    snapshot = capture_prediction_input(predictions)
    lexical_output = Path(output).expanduser()
    if lexical_output.exists() or lexical_output.is_symlink():
        raise FileExistsError(f"AgileX metric report already exists: {lexical_output}")
    destination = lexical_output.resolve(strict=False)
    artifact = load_agilex_predictions(snapshot.raw)
    _require_identity(
        artifact.metadata,
        checkpoint_identity_sha256=checkpoint_identity_sha256,
        dataset_view_id=dataset_view_id,
        dataset_view_sha256=dataset_view_sha256,
        decoder_sha256=decoder_sha256,
    )

    per_sample: list[dict[str, object]] = []
    all_metrics: list[Mapping[str, float]] = []
    by_task: defaultdict[str, list[Mapping[str, float]]] = defaultdict(list)
    by_view: defaultdict[str, list[Mapping[str, float]]] = defaultdict(list)
    group_deltas: dict[str, list[npt.NDArray[np.float64]]] = {
        name: [] for name in _QPOS_GROUPS
    }
    valid_rgb_frames = 0
    valid_actions = 0
    for sample_index, (sample_id, task_id, repo_id) in enumerate(
        zip(artifact.sample_ids, artifact.task_ids, artifact.repo_ids, strict=True)
    ):
        rgb_scores: list[tuple[float, float]] = []
        for view_index, view_name in enumerate(artifact.view_names):
            view_scores = [
                _rgb(
                    artifact.predicted_rgb[sample_index, view_index, frame_index],
                    artifact.target_rgb[sample_index, view_index, frame_index],
                )
                for frame_index in np.flatnonzero(
                    artifact.video_valid[sample_index, view_index]
                )
            ]
            rgb_scores.extend(view_scores)
            by_view[view_name].append(
                {
                    "psnr_db": float(np.mean([item[0] for item in view_scores])),
                    "ssim": float(np.mean([item[1] for item in view_scores])),
                }
            )
        valid = artifact.action_valid[sample_index]
        delta = artifact.predicted_qpos14[sample_index, valid].astype(np.float64)
        delta -= artifact.target_qpos14[sample_index, valid].astype(np.float64)
        qpos_mae, qpos_rmse = _qpos_metrics(delta)
        for name, indices in _QPOS_GROUPS.items():
            group_deltas[name].append(delta[:, indices])
        metrics = {
            "psnr_db": float(np.mean([item[0] for item in rgb_scores])),
            "ssim": float(np.mean([item[1] for item in rgb_scores])),
            "qpos14_mae": qpos_mae,
            "qpos14_rmse": qpos_rmse,
        }
        record = {
            "sample_id": sample_id,
            "task_id": task_id,
            "repo_id": repo_id,
            "contact_condition_present": bool(
                artifact.contact_condition_present[sample_index]
            ),
            "valid_rgb_frame_count": len(rgb_scores),
            "valid_action_count": int(valid.sum()),
            **metrics,
        }
        per_sample.append(record)
        all_metrics.append(metrics)
        by_task[task_id].append(metrics)
        valid_rgb_frames += len(rgb_scores)
        valid_actions += int(valid.sum())

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
            "psnr": "per_frame_rgb_mse_then_sample_macro_mean",
            "psnr_cap_db": PSNR_CAP_DB,
            "ssim": "skimage_structural_similarity_explicit_parameters",
            "ssim_backend": f"scikit-image-{version('scikit-image')}",
            "ssim_parameters": dict(_SSIM_PARAMETERS),
            "qpos14_mae": "sample_elementwise_absolute_error_then_macro_mean",
            "qpos14_rmse": "sample_elementwise_root_mean_square_then_macro_mean",
            "qpos_units": "native_dataset_joint_and_gripper_units",
            "aggregation": "sample_macro_mean",
        },
        "sample_count": len(per_sample),
        "valid_rgb_frame_count": valid_rgb_frames,
        "valid_action_count": valid_actions,
        "internal_holdout_complete": False,
        "generalization_claim_valid": False,
        "generalization_claim_reason": (
            "offline predictions are proxy evidence and do not replace organizer robot evaluation"
        ),
        "overall": _mean(all_metrics),
        "per_task": {
            task: {"sample_count": len(records), **_mean(records)}
            for task, records in sorted(by_task.items())
        },
        "per_view": {
            view: {
                "sample_count": len(by_view[view]),
                **_mean(by_view[view], names=("psnr_db", "ssim")),
            }
            for view in artifact.view_names
        },
        "qpos_groups": {
            name: dict(
                zip(
                    ("mae", "rmse"),
                    _qpos_metrics(np.concatenate(group_deltas[name], axis=0)),
                    strict=True,
                )
            )
            for name in _QPOS_GROUPS
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
    report_sha256 = _write_json(destination, report)
    return {
        **report,
        "report": str(destination.resolve(strict=True)),
        "report_file_sha256": report_sha256,
    }


__all__ = (
    "PSNR_CAP_DB",
    "REPORT_ARTIFACT_TYPE",
    "REPORT_SCHEMA_VERSION",
    "evaluate_agilex_offline_predictions",
)
