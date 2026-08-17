# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from n0_twam.cli import run_cli
from n0_twam.evaluation.agilex_evaluation_view import (
    AgileXEvaluationEntry,
    publish_agilex_evaluation_view,
)
from n0_twam.evaluation.agilex_offline_metrics import (
    evaluate_agilex_offline_predictions,
)
from n0_twam.evaluation.agilex_offline_pipeline import (
    recover_agilex_offline_closeout,
    run_agilex_offline_evaluation,
)
from n0_twam.evaluation.agilex_prediction_artifact import (
    PREDICTION_ARTIFACT_TYPE,
)
from n0_twam.evaluation.franka_prediction_io import capture_prediction_input

_SHA = "a" * 64


def _recovery_inputs(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source"
    source.mkdir()
    view_path = source / "evaluation_view.json"
    view = publish_agilex_evaluation_view(
        output=view_path,
        view_id="agilex-recovery-view",
        entries=(
            AgileXEvaluationEntry(
                repo_id="agilex_rgb",
                episode_id=0,
                task_id="clean_table",
            ),
            AgileXEvaluationEntry(
                repo_id="agilex_touch",
                episode_id=0,
                task_id="insert",
            ),
        ),
    )
    predictions = source / "predictions.npz"
    target_rgb = np.zeros((2, 3, 1, 8, 8, 3), dtype=np.uint8)
    target_qpos = np.zeros((2, 1, 14), dtype=np.float32)
    np.savez(
        predictions,
        schema_version=np.asarray(1, dtype=np.int64),
        artifact_type=np.asarray(PREDICTION_ARTIFACT_TYPE),
        checkpoint_identity_sha256=np.asarray(_SHA),
        dataset_view_id=np.asarray("agilex-recovery-view"),
        dataset_view_sha256=np.asarray(view["view_sha256"]),
        decoder_sha256=np.asarray("b" * 64),
        seed=np.asarray(20260813, dtype=np.int64),
        run_role=np.asarray("final_refit"),
        prediction_mode=np.asarray("policy_action"),
        tactile_profile=np.asarray("mixed"),
        view_names=np.asarray(("top", "wrist_l", "wrist_r")),
        frame_offsets=np.asarray((1,), dtype=np.int64),
        action_offsets=np.asarray((1,), dtype=np.int64),
        sample_ids=np.asarray(
            ("agilex_rgb:episode_000000", "agilex_touch:episode_000000")
        ),
        task_ids=np.asarray(("clean_table", "insert")),
        repo_ids=np.asarray(("agilex_rgb", "agilex_touch")),
        contact_condition_present=np.asarray((False, True), dtype=np.bool_),
        predicted_rgb=target_rgb.copy(),
        target_rgb=target_rgb,
        video_valid=np.ones(target_rgb.shape[:3], dtype=np.bool_),
        predicted_qpos14=target_qpos.copy(),
        target_qpos14=target_qpos,
        action_valid=np.ones(target_qpos.shape[:2], dtype=np.bool_),
    )
    metrics = source / "metrics.json"
    evaluate_agilex_offline_predictions(
        predictions=predictions,
        output=metrics,
        checkpoint_identity_sha256=_SHA,
        dataset_view_id="agilex-recovery-view",
        dataset_view_sha256=str(view["view_sha256"]),
        decoder_sha256="b" * 64,
    )
    observation = source / "replay_observation.npz"
    np.savez(
        observation,
        schema_version=np.asarray(1, dtype=np.int64),
        task_id=np.asarray("insert"),
        images=np.zeros((3, 8, 8, 3), dtype=np.uint8),
        joint_qpos=np.zeros(14, dtype=np.float32),
        execution_dt_s=np.asarray(0.1, dtype=np.float64),
        tactile_keys=np.asarray((), dtype=np.str_),
        tactile_images=np.empty((0, 1, 1, 3), dtype=np.uint8),
        wrench_keys=np.asarray((), dtype=np.str_),
        wrench=np.empty((0, 6), dtype=np.float32),
    )
    config = source / "policy.json"
    config.write_text("{}\n", encoding="utf-8")
    return {
        "policy_config": config,
        "evaluation_view": view_path,
        "predictions": predictions,
        "metrics": metrics,
        "replay_observation": observation,
    }


def _mock_recovery_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "n0_twam.evaluation.agilex_offline_pipeline."
        "prepare_agilex_evaluation_environment",
        lambda **_: {"profile": "mixed"},
    )
    policy = SimpleNamespace(
        checkpoint_identity_sha256=_SHA,
        policy=SimpleNamespace(tactile_profile="mixed"),
    )
    monkeypatch.setattr(
        "n0_twam.integrations.worldarena.agilex_policy_io." "load_agilex_policy_config",
        lambda _: policy,
    )

    def replay(**kwargs):
        payload = {
            "status": "complete",
            "realtime_assessment": {"sample_count": 7, "realtime_pass": False},
        }
        kwargs["output"].write_text(json.dumps(payload), encoding="utf-8")
        return {
            **payload,
            "receipt_file_sha256": capture_prediction_input(kwargs["output"]).sha256,
        }

    monkeypatch.setattr(
        "n0_twam.integrations.worldarena.agilex_policy_replay."
        "run_agilex_policy_replay",
        replay,
    )


def test_one_command_pipeline_seals_all_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "n0_twam.evaluation.agilex_offline_pipeline."
        "prepare_agilex_evaluation_environment",
        lambda **kwargs: {"profile": "mixed"},
    )

    def view(**kwargs):
        kwargs["output"].write_text("{}", encoding="utf-8")
        return {"status": "complete"}

    def generation(**kwargs):
        kwargs["output"].write_bytes(b"prediction")
        kwargs["replay_observation_output"].write_bytes(b"observation")
        return {"sha256": _SHA, "sample_count": 10}

    metadata = {
        "checkpoint_identity_sha256": _SHA,
        "dataset_view_id": "proxy-v1",
        "dataset_view_sha256": "b" * 64,
        "decoder_sha256": "c" * 64,
    }

    def metrics(**kwargs):
        kwargs["output"].write_text("{}", encoding="utf-8")
        return {
            "report_file_sha256": "d" * 64,
            "overall": {"psnr_db": 21.0, "qpos14_mae": 0.1},
        }

    def replay(**kwargs):
        kwargs["output"].write_text("{}", encoding="utf-8")
        return {
            "receipt_file_sha256": "e" * 64,
            "realtime_assessment": {"realtime_pass": False},
        }

    monkeypatch.setattr(
        "n0_twam.evaluation.agilex_offline_pipeline." "_build_proxy_view",
        view,
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.agilex_offline_pipeline."
        "generate_agilex_offline_predictions",
        generation,
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.agilex_offline_pipeline." "verify_agilex_prediction_file",
        lambda *args, **kwargs: SimpleNamespace(metadata=metadata),
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.agilex_offline_pipeline."
        "evaluate_agilex_offline_predictions",
        metrics,
    )
    monkeypatch.setattr(
        "n0_twam.integrations.worldarena.agilex_policy_replay."
        "run_agilex_policy_replay",
        replay,
    )

    result = run_agilex_offline_evaluation(
        train_request=tmp_path / "request.json",
        policy_config=tmp_path / "policy.json",
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "evaluation",
    )

    assert result["status"] == "complete"
    assert result["execution_tier"] == "training_distribution_offline_proxy"
    assert result["independent_holdout"] is False
    assert result["sample_count"] == 10
    assert result["metrics"]["psnr_db"] == 21.0
    assert (tmp_path / "evaluation/closeout.json").is_file()


def test_public_pipeline_cli_dispatches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "n0_twam.evaluation.agilex_offline_pipeline." "run_agilex_offline_evaluation",
        lambda **kwargs: {"status": "complete", "output": str(kwargs["output_root"])},
    )

    result = run_cli(
        [
            "track32",
            "agilex-offline-eval",
            "--train-request",
            str(tmp_path / "request.json"),
            "--config",
            str(tmp_path / "policy.json"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path / "evaluation"),
        ]
    )

    assert result["status"] == "complete"


def test_closeout_recovery_copies_four_artifacts_then_replays_and_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _recovery_inputs(tmp_path)
    _mock_recovery_runtime(monkeypatch)
    monkeypatch.setattr(
        "n0_twam.evaluation.agilex_offline_pipeline."
        "generate_agilex_offline_predictions",
        lambda **_: pytest.fail("recovery must not regenerate predictions"),
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.agilex_offline_pipeline."
        "evaluate_agilex_offline_predictions",
        lambda **_: pytest.fail("recovery must not recompute metrics"),
    )
    output = tmp_path / "recovered"

    result = recover_agilex_offline_closeout(
        train_request=tmp_path / "request.json",
        **inputs,
        output_root=output,
        replay_steps=7,
        control_hz=10.0,
    )

    expected = {
        "evaluation_view.json",
        "predictions.npz",
        "metrics.json",
        "replay_observation.npz",
        "policy_replay.json",
        "closeout.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    for name in expected - {"policy_replay.json", "closeout.json"}:
        assert (output / name).read_bytes() == inputs[
            {
                "evaluation_view.json": "evaluation_view",
                "predictions.npz": "predictions",
                "metrics.json": "metrics",
                "replay_observation.npz": "replay_observation",
            }[name]
        ].read_bytes()
    assert result["status"] == "complete"
    assert result["sample_count"] == 2
    assert result["checkpoint_identity_sha256"] == _SHA
    assert result["metrics"]["qpos14_mae"] == 0.0
    closeout = json.loads((output / "closeout.json").read_text(encoding="utf-8"))
    assert (
        closeout["prediction_file_sha256"]
        == capture_prediction_input(inputs["predictions"]).sha256
    )
    assert (
        closeout["metrics_file_sha256"]
        == capture_prediction_input(inputs["metrics"]).sha256
    )


def test_closeout_recovery_rejects_metrics_binding_before_creating_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _recovery_inputs(tmp_path)
    _mock_recovery_runtime(monkeypatch)
    payload = json.loads(inputs["metrics"].read_text(encoding="utf-8"))
    payload["prediction_artifact_sha256"] = "f" * 64
    inputs["metrics"].chmod(0o644)
    inputs["metrics"].write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "recovered"

    with pytest.raises(ValueError, match="prediction artifact SHA256"):
        recover_agilex_offline_closeout(
            train_request=tmp_path / "request.json",
            **inputs,
            output_root=output,
        )

    assert not output.exists()


def test_public_closeout_recovery_cli_dispatches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "n0_twam.evaluation.agilex_offline_pipeline." "recover_agilex_offline_closeout",
        lambda **kwargs: {"status": "complete", "output": str(kwargs["output_root"])},
    )

    result = run_cli(
        [
            "track32",
            "agilex-offline-closeout",
            "--train-request",
            str(tmp_path / "request.json"),
            "--config",
            str(tmp_path / "policy.json"),
            "--evaluation-view",
            str(tmp_path / "evaluation_view.json"),
            "--predictions",
            str(tmp_path / "predictions.npz"),
            "--metrics",
            str(tmp_path / "metrics.json"),
            "--replay-observation",
            str(tmp_path / "replay_observation.npz"),
            "--output-root",
            str(tmp_path / "recovered"),
        ]
    )

    assert result["status"] == "complete"
