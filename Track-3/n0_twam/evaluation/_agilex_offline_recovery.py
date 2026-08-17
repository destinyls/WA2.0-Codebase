# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict recovery of already-generated AgileX offline artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import BinaryIO

from n0_twam.evaluation.franka_atomic_io import publish_atomic_file
from n0_twam.evaluation.franka_prediction_io import (
    StablePredictionInput,
    capture_prediction_input,
    require_prediction_unchanged,
)
from n0_twam.evaluation.sealed_artifact_io import canonical_json, validate_sha256

from .agilex_evaluation_view import (
    load_agilex_evaluation_view,
    load_agilex_evaluation_view_bytes,
)
from .agilex_offline_metrics import REPORT_ARTIFACT_TYPE, REPORT_SCHEMA_VERSION
from .agilex_prediction_artifact import (
    AgileXPredictions,
    load_agilex_predictions,
    verify_agilex_prediction_file,
)

_METRIC_NAMES = frozenset(("psnr_db", "ssim", "qpos14_mae", "qpos14_rmse"))
_QPOS_GROUPS = frozenset(("left_arm", "left_gripper", "right_arm", "right_gripper"))
_METRICS_FIELDS = frozenset(
    (
        "schema_version",
        "artifact_type",
        "status",
        "execution_tier",
        "organizer_evaluation_completed",
        "real_robot_evaluation_completed",
        "prediction_artifact",
        "prediction_artifact_sha256",
        "prediction_contract",
        "metric_contract",
        "sample_count",
        "valid_rgb_frame_count",
        "valid_action_count",
        "internal_holdout_complete",
        "generalization_claim_valid",
        "generalization_claim_reason",
        "overall",
        "per_task",
        "per_view",
        "qpos_groups",
        "per_sample",
        "report_identity_sha256",
    )
)


def _json_object(snapshot: StablePredictionInput, *, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"{label} contains non-finite constant {value}")

    try:
        payload = json.loads(
            snapshot.raw.decode("utf-8"), parse_constant=reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _finite_metric_record(
    value: object, *, names: frozenset[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != names:
        raise ValueError(f"{label} fields differ from the metric contract")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value.values()
    ):
        raise ValueError(f"{label} must contain finite numbers")
    return value


def _validate_metrics_report(
    snapshot: StablePredictionInput,
    *,
    prediction: StablePredictionInput,
    artifact: AgileXPredictions,
) -> dict[str, object]:
    report = _json_object(snapshot, label="AgileX metrics report")
    if set(report) != _METRICS_FIELDS:
        raise ValueError("AgileX metrics report fields differ from schema")
    if (
        report["schema_version"] != REPORT_SCHEMA_VERSION
        or report["artifact_type"] != REPORT_ARTIFACT_TYPE
        or report["status"] != "complete"
        or report["execution_tier"] != "offline_reference_metrics"
        or report["organizer_evaluation_completed"] is not False
        or report["real_robot_evaluation_completed"] is not False
        or report["internal_holdout_complete"] is not False
        or report["generalization_claim_valid"] is not False
    ):
        raise ValueError("AgileX metrics report contract is incompatible")
    if report["prediction_artifact_sha256"] != prediction.sha256:
        raise ValueError("AgileX metrics prediction artifact SHA256 mismatch")
    if report["prediction_contract"] != dict(artifact.metadata):
        raise ValueError("AgileX metrics prediction contract mismatch")
    identity = validate_sha256(
        report["report_identity_sha256"], label="AgileX metric report identity"
    )
    semantic = {
        key: value
        for key, value in report.items()
        if key not in {"prediction_artifact", "report_identity_sha256"}
    }
    if hashlib.sha256(canonical_json(semantic)).hexdigest() != identity:
        raise ValueError("AgileX metric report identity mismatch")
    sample_count = len(artifact.sample_ids)
    if report["sample_count"] != sample_count:
        raise ValueError("AgileX metric sample count mismatch")
    if report["valid_rgb_frame_count"] != int(artifact.video_valid.sum()) or report[
        "valid_action_count"
    ] != int(artifact.action_valid.sum()):
        raise ValueError("AgileX metric valid-item counts mismatch")
    _finite_metric_record(report["overall"], names=_METRIC_NAMES, label="overall")
    per_sample = report["per_sample"]
    if not isinstance(per_sample, list) or len(per_sample) != sample_count:
        raise ValueError("AgileX per-sample metric roster mismatch")
    for index, record in enumerate(per_sample):
        expected = {
            "sample_id": artifact.sample_ids[index],
            "task_id": artifact.task_ids[index],
            "repo_id": artifact.repo_ids[index],
            "contact_condition_present": bool(
                artifact.contact_condition_present[index]
            ),
            "valid_rgb_frame_count": int(artifact.video_valid[index].sum()),
            "valid_action_count": int(artifact.action_valid[index].sum()),
        }
        if not isinstance(record, Mapping) or any(
            record.get(name) != value for name, value in expected.items()
        ):
            raise ValueError("AgileX per-sample metric identity mismatch")
        _finite_metric_record(
            {name: record.get(name) for name in _METRIC_NAMES},
            names=_METRIC_NAMES,
            label="per-sample metrics",
        )
    if not isinstance(report["per_task"], Mapping) or set(report["per_task"]) != set(
        artifact.task_ids
    ):
        raise ValueError("AgileX per-task metric roster mismatch")
    if not isinstance(report["per_view"], Mapping) or set(report["per_view"]) != set(
        artifact.view_names
    ):
        raise ValueError("AgileX per-view metric roster mismatch")
    if (
        not isinstance(report["qpos_groups"], Mapping)
        or set(report["qpos_groups"]) != _QPOS_GROUPS
    ):
        raise ValueError("AgileX qpos metric group roster mismatch")
    return report


def _copy_snapshot(
    snapshot: StablePredictionInput,
    *,
    output: Path,
    validator: Callable[[bytes], object],
    label: str,
) -> str:
    def writer(handle: BinaryIO) -> None:
        handle.write(snapshot.raw)

    published = publish_atomic_file(
        output=output,
        writer=writer,
        validator=validator,
        label=label,
    )
    if published.sha256 != snapshot.sha256:
        raise RuntimeError(f"{label} copied SHA256 mismatch")
    return published.sha256


def recover_agilex_offline_closeout(
    *,
    train_request: Path,
    policy_config: Path,
    evaluation_view: Path,
    predictions: Path,
    metrics: Path,
    replay_observation: Path,
    output_root: Path,
    new_root: Callable[[Path], Path],
    build_closeout: Callable[..., dict[str, object]],
    publish_closeout: Callable[[Path, dict[str, object]], str],
    prepare_environment: Callable[..., dict[str, object]],
    replay_steps: int = 7,
    control_hz: float = 10.0,
    seed: int = 20260813,
    device: str = "0",
) -> dict[str, object]:
    """Replay and close out already-generated, hash-bound AgileX artifacts."""

    inputs = {
        "policy_config": capture_prediction_input(policy_config),
        "evaluation_view": capture_prediction_input(evaluation_view),
        "predictions": capture_prediction_input(predictions),
        "metrics": capture_prediction_input(metrics),
        "replay_observation": capture_prediction_input(replay_observation),
    }
    if len({snapshot.path for snapshot in inputs.values()}) != len(inputs):
        raise ValueError("AgileX closeout recovery inputs must be distinct files")
    load_agilex_evaluation_view_bytes(inputs["evaluation_view"].raw)
    view = load_agilex_evaluation_view(inputs["evaluation_view"].path)
    artifact = load_agilex_predictions(inputs["predictions"].raw)
    metadata = artifact.metadata
    if (
        metadata["dataset_view_id"] != view.view_id
        or metadata["dataset_view_sha256"] != view.view_sha256
    ):
        raise ValueError("AgileX prediction and evaluation view identities differ")
    expected_roster = tuple(
        (
            f"{entry.repo_id}:episode_{entry.episode_id:06d}",
            entry.task_id,
            entry.repo_id,
        )
        for entry in view.entries
    )
    actual_roster = tuple(
        zip(artifact.sample_ids, artifact.task_ids, artifact.repo_ids, strict=True)
    )
    if actual_roster != expected_roster:
        raise ValueError("AgileX prediction and evaluation view rosters differ")
    if metadata["prediction_mode"] != "policy_action":
        raise ValueError("AgileX closeout recovery requires policy-action predictions")
    if metadata["seed"] != seed:
        raise ValueError("AgileX recovery seed differs from prediction identity")
    metrics_report = _validate_metrics_report(
        inputs["metrics"], prediction=inputs["predictions"], artifact=artifact
    )

    from n0_twam.integrations.worldarena.agilex_policy_io import (
        load_agilex_policy_config,
    )
    from n0_twam.integrations.worldarena.agilex_policy_replay import (
        _observation,
        run_agilex_policy_replay,
    )

    policy = load_agilex_policy_config(inputs["policy_config"].path)
    observation = _observation(inputs["replay_observation"].raw)
    if (
        policy.checkpoint_identity_sha256 != metadata["checkpoint_identity_sha256"]
        or policy.policy.tactile_profile != metadata["tactile_profile"]
    ):
        raise ValueError("AgileX policy and prediction identities differ")
    if observation["task_id"] not in {entry.task_id for entry in view.entries}:
        raise ValueError(
            "AgileX replay observation task is absent from evaluation view"
        )
    for snapshot in inputs.values():
        require_prediction_unchanged(snapshot)
    prepare_environment(train_request=train_request, device=device, seed=seed)
    for snapshot in inputs.values():
        require_prediction_unchanged(snapshot)

    root = new_root(output_root)
    paths = {
        "view": root / "evaluation_view.json",
        "predictions": root / "predictions.npz",
        "metrics": root / "metrics.json",
        "observation": root / "replay_observation.npz",
        "replay": root / "policy_replay.json",
        "closeout": root / "closeout.json",
    }
    copied_hashes = {
        "view": _copy_snapshot(
            inputs["evaluation_view"],
            output=paths["view"],
            validator=load_agilex_evaluation_view_bytes,
            label="AgileX recovery evaluation view",
        ),
        "predictions": _copy_snapshot(
            inputs["predictions"],
            output=paths["predictions"],
            validator=load_agilex_predictions,
            label="AgileX recovery predictions",
        ),
        "metrics": _copy_snapshot(
            inputs["metrics"],
            output=paths["metrics"],
            validator=lambda raw: json.loads(raw),
            label="AgileX recovery metrics",
        ),
        "observation": _copy_snapshot(
            inputs["replay_observation"],
            output=paths["observation"],
            validator=_observation,
            label="AgileX recovery replay observation",
        ),
    }
    verify_agilex_prediction_file(
        paths["predictions"], expected_sha256=copied_hashes["predictions"]
    )
    copied_prediction = capture_prediction_input(paths["predictions"])
    copied_metrics = capture_prediction_input(paths["metrics"])
    _validate_metrics_report(
        copied_metrics, prediction=copied_prediction, artifact=artifact
    )
    if copied_metrics.sha256 != copied_hashes["metrics"]:
        raise RuntimeError("AgileX recovery metrics copied SHA256 mismatch")
    replay = run_agilex_policy_replay(
        config_path=inputs["policy_config"].path,
        observation_path=paths["observation"],
        steps=replay_steps,
        output=paths["replay"],
        control_hz=control_hz,
        require_realtime=False,
    )
    for snapshot in inputs.values():
        require_prediction_unchanged(snapshot)
    core = build_closeout(
        metadata=metadata,
        view_path=paths["view"],
        prediction_path=paths["predictions"],
        metrics_path=paths["metrics"],
        observation_path=paths["observation"],
        replay_path=paths["replay"],
        prediction_sha256=copied_hashes["predictions"],
        metrics_sha256=copied_hashes["metrics"],
        replay=replay,
        sample_count=len(artifact.sample_ids),
        metrics=metrics_report["overall"],
    )
    closeout_sha256 = publish_closeout(paths["closeout"], core)
    return {
        **core,
        "closeout": str(paths["closeout"]),
        "closeout_file_sha256": closeout_sha256,
    }


__all__ = ("recover_agilex_offline_closeout",)
