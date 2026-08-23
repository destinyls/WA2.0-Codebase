# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Training-corpus Franka trajectory metrics and explicit acceptance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from n0_twam.evaluation.franka_atomic_io import publish_atomic_file
from n0_twam.evaluation.franka_prediction_artifact import (
    capture_prediction_input,
    load_franka_predictions,
    require_prediction_unchanged,
)
from n0_twam.evaluation.sealed_artifact_io import canonical_json

REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT_TYPE = "n0_twam_track32_franka_training_fit"


@dataclass(frozen=True)
class WipeTrainingFitGates:
    """First-stage engineering gates; these are not organizer thresholds."""

    offset1_median_cm: float = 0.5
    offset1_p95_cm: float = 1.5
    overall_p95_cm: float = 2.0
    absolute_z_bias_cm: float = 0.2
    z_p95_absolute_cm: float = 1.0
    rotation_p95_deg: float = 2.0


def _stats(values: npt.NDArray[np.float64]) -> dict[str, float]:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("metric values must be a finite non-empty vector")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def _validate_inputs(
    predicted: npt.ArrayLike,
    target: npt.ArrayLike,
    valid: npt.ArrayLike,
    action_offsets: Sequence[int],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    tuple[int, ...],
]:
    prediction = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    mask = np.asarray(valid)
    offsets = tuple(int(value) for value in action_offsets)
    if prediction.shape != reference.shape or prediction.ndim != 3:
        raise ValueError("Franka trajectories must share shape [sample,action,8]")
    if prediction.shape[-1] != 8 or not np.isfinite(prediction).all():
        raise ValueError("Franka trajectory values must be finite end-pose8")
    if not np.isfinite(reference).all():
        raise ValueError("Franka trajectory targets must be finite")
    if mask.dtype != np.bool_ or mask.shape != prediction.shape[:2]:
        raise ValueError("Franka trajectory validity must be bool [sample,action]")
    if len(offsets) != prediction.shape[1] or not offsets:
        raise ValueError("action offsets must match the action horizon")
    if offsets[0] <= 0 or any(b <= a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("action offsets must be positive and strictly increasing")
    if not bool(mask.any()):
        raise ValueError("Franka trajectory evaluation has no valid actions")
    return prediction, reference, mask, offsets


def _rotation_errors_deg(
    predicted: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    predicted_q = predicted[..., 3:7]
    target_q = target[..., 3:7]
    predicted_norm = np.linalg.norm(predicted_q, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(target_q, axis=-1, keepdims=True)
    if np.any(predicted_norm <= 0.0) or np.any(target_norm <= 0.0):
        raise ValueError("Franka trajectory contains a zero quaternion")
    predicted_q = predicted_q / predicted_norm
    target_q = target_q / target_norm
    cosine = np.abs(np.sum(predicted_q * target_q, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(cosine, 0.0, 1.0)))


def _gate(value: float, threshold: float) -> dict[str, object]:
    return {
        "value": value,
        "operator": "<=",
        "threshold": threshold,
        "passed": value <= threshold,
    }


def summarize_franka_training_fit(
    *,
    predicted: npt.ArrayLike,
    target: npt.ArrayLike,
    valid: npt.ArrayLike,
    action_offsets: Sequence[int],
    gates: WipeTrainingFitGates = WipeTrainingFitGates(),
) -> dict[str, object]:
    """Return physical trajectory errors and the first-stage Wipe gate result."""

    prediction, reference, mask, offsets = _validate_inputs(
        predicted, target, valid, action_offsets
    )
    xyz_delta_cm = (prediction[..., :3] - reference[..., :3]) * 100.0
    distance_cm = np.linalg.norm(xyz_delta_cm, axis=-1)
    rotation_deg = _rotation_errors_deg(prediction, reference)
    gripper_abs = np.abs(prediction[..., 7] - reference[..., 7])
    valid_xyz = xyz_delta_cm[mask]
    valid_distance = distance_cm[mask]
    valid_rotation = rotation_deg[mask]
    valid_gripper = gripper_abs[mask]

    per_axis: dict[str, dict[str, float]] = {}
    for axis_index, axis in enumerate(("x", "y", "z")):
        values = valid_xyz[:, axis_index]
        per_axis[axis] = {
            "signed_bias_cm": float(np.mean(values)),
            "mae_cm": float(np.mean(np.abs(values))),
            "p95_absolute_cm": float(np.percentile(np.abs(values), 95)),
            "max_absolute_cm": float(np.max(np.abs(values))),
            "rmse_cm": float(np.sqrt(np.mean(np.square(values)))),
        }

    per_horizon: list[dict[str, object]] = []
    for horizon_index, offset in enumerate(offsets):
        horizon_mask = mask[:, horizon_index]
        if not bool(horizon_mask.any()):
            continue
        per_horizon.append(
            {
                "action_offset": offset,
                "valid_count": int(horizon_mask.sum()),
                "position_cm": _stats(distance_cm[horizon_mask, horizon_index]),
                "rotation_deg": _stats(rotation_deg[horizon_mask, horizon_index]),
                "gripper_absolute_wire": _stats(
                    gripper_abs[horizon_mask, horizon_index]
                ),
            }
        )
    offset1 = next(
        entry for entry in per_horizon if int(entry["action_offset"]) == offsets[0]
    )
    offset1_position = offset1["position_cm"]
    if not isinstance(offset1_position, Mapping):
        raise RuntimeError("offset-one position metrics are malformed")

    overall_position = _stats(valid_distance)
    overall_rotation = _stats(valid_rotation)
    gate_results = {
        "offset1_position_median_cm": _gate(
            float(offset1_position["median"]), gates.offset1_median_cm
        ),
        "offset1_position_p95_cm": _gate(
            float(offset1_position["p95"]), gates.offset1_p95_cm
        ),
        "overall_position_p95_cm": _gate(
            overall_position["p95"], gates.overall_p95_cm
        ),
        "absolute_z_bias_cm": _gate(
            abs(per_axis["z"]["signed_bias_cm"]), gates.absolute_z_bias_cm
        ),
        "z_p95_absolute_cm": _gate(
            per_axis["z"]["p95_absolute_cm"], gates.z_p95_absolute_cm
        ),
        "rotation_p95_deg": _gate(
            overall_rotation["p95"], gates.rotation_p95_deg
        ),
    }
    return {
        "valid_action_count": int(mask.sum()),
        "position_frame": "robot_base_absolute",
        "quaternion_order": "XYZW",
        "position_cm": overall_position,
        "rotation_deg": overall_rotation,
        "gripper_absolute_wire": _stats(valid_gripper),
        "per_axis": per_axis,
        "per_horizon": per_horizon,
        "threshold_coverage": {
            "le_0_5_cm": float(np.mean(valid_distance <= 0.5)),
            "le_1_0_cm": float(np.mean(valid_distance <= 1.0)),
            "le_2_0_cm": float(np.mean(valid_distance <= 2.0)),
        },
        "gates": gate_results,
        "all_gates_passed": all(bool(item["passed"]) for item in gate_results.values()),
    }


def write_franka_training_fit_report(*, predictions: Path, output: Path) -> dict[str, object]:
    """Score one sealed prediction artifact and publish a no-clobber report."""

    snapshot = capture_prediction_input(predictions)
    artifact = load_franka_predictions(snapshot.raw)
    trajectory = summarize_franka_training_fit(
        predicted=artifact.predicted_end_pose,
        target=artifact.target_end_pose,
        valid=artifact.action_valid,
        action_offsets=tuple(int(value) for value in artifact.metadata["action_offsets"]),
    )
    core = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "status": "complete",
        "evidence_tier": "training_corpus_regression_diagnostic",
        "organizer_evaluation_completed": False,
        "real_robot_evaluation_completed": False,
        "prediction_artifact": str(snapshot.path),
        "prediction_artifact_sha256": snapshot.sha256,
        "prediction_contract": dict(artifact.metadata),
        "trajectory": trajectory,
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
    raw = (
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )

    def _writer(handle: BinaryIO) -> None:
        handle.write(raw)

    def _validator(candidate: bytes) -> None:
        if json.loads(candidate) != report:
            raise ValueError("training-fit report changed during publication")

    require_prediction_unchanged(snapshot)
    published = publish_atomic_file(
        output=output,
        writer=_writer,
        validator=_validator,
        label="Franka training-fit report",
    )
    return {
        **report,
        "report": str(published.path.resolve(strict=True)),
        "report_file_sha256": published.sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_franka_training_fit_report(
        predictions=args.predictions,
        output=args.output,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if bool(result["trajectory"]["all_gates_passed"]) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "WipeTrainingFitGates",
    "summarize_franka_training_fit",
    "write_franka_training_fit_report",
)
