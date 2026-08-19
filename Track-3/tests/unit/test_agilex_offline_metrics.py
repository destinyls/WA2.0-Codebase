# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from n0_twam.cli import run_cli
from n0_twam.evaluation.agilex_offline_metrics import (
    PSNR_CAP_DB,
    evaluate_agilex_offline_predictions,
)
from n0_twam.evaluation.agilex_prediction_artifact import (
    PREDICTION_ARTIFACT_TYPE,
)

CHECKPOINT_SHA = "a" * 64
DATASET_SHA = "b" * 64
DECODER_SHA = "c" * 64


def _predictions(
    tmp_path: Path,
    *,
    pixel_offset: int = 0,
    qpos_offset: float = 0.0,
) -> Path:
    target_rgb = np.zeros((2, 3, 2, 8, 8, 3), dtype=np.uint8)
    target_qpos = np.zeros((2, 3, 14), dtype=np.float32)
    path = tmp_path / "agilex-predictions.npz"
    np.savez(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        artifact_type=np.asarray(PREDICTION_ARTIFACT_TYPE),
        checkpoint_identity_sha256=np.asarray(CHECKPOINT_SHA),
        dataset_view_id=np.asarray("agilex-unit-view"),
        dataset_view_sha256=np.asarray(DATASET_SHA),
        decoder_sha256=np.asarray(DECODER_SHA),
        seed=np.asarray(20260813, dtype=np.int64),
        run_role=np.asarray("development"),
        prediction_mode=np.asarray("policy_action"),
        tactile_profile=np.asarray("mixed"),
        view_names=np.asarray(("top", "wrist_l", "wrist_r")),
        frame_offsets=np.asarray((1, 2), dtype=np.int64),
        action_offsets=np.asarray((1, 2, 3), dtype=np.int64),
        sample_ids=np.asarray(("sample-a", "sample-b")),
        task_ids=np.asarray(("insert", "wipe_table")),
        repo_ids=np.asarray(("agilex_touch", "agilex_vision")),
        contact_condition_present=np.asarray((True, False), dtype=np.bool_),
        predicted_rgb=np.full_like(target_rgb, pixel_offset),
        target_rgb=target_rgb,
        video_valid=np.ones(target_rgb.shape[:3], dtype=np.bool_),
        predicted_qpos14=np.full_like(target_qpos, qpos_offset),
        target_qpos14=target_qpos,
        action_valid=np.ones(target_qpos.shape[:2], dtype=np.bool_),
    )
    return path


def _evaluate(path: Path, output: Path) -> dict[str, object]:
    return evaluate_agilex_offline_predictions(
        predictions=path,
        output=output,
        checkpoint_identity_sha256=CHECKPOINT_SHA,
        dataset_view_id="agilex-unit-view",
        dataset_view_sha256=DATASET_SHA,
        decoder_sha256=DECODER_SHA,
    )


def test_perfect_agilex_predictions_seal_rgb_and_qpos_metrics(
    tmp_path: Path,
) -> None:
    output = tmp_path / "metrics.json"
    result = _evaluate(_predictions(tmp_path), output)

    assert result["status"] == "complete"
    assert result["execution_tier"] == "offline_reference_metrics"
    assert result["organizer_evaluation_completed"] is False
    assert result["real_robot_evaluation_completed"] is False
    assert result["overall"] == pytest.approx(
        {
            "psnr_db": PSNR_CAP_DB,
            "ssim": 1.0,
            "qpos14_mae": 0.0,
            "qpos14_rmse": 0.0,
        }
    )
    assert set(result["per_view"]) == {"top", "wrist_l", "wrist_r"}
    assert set(result["per_task"]) == {"insert", "wipe_table"}
    assert set(result["qpos_groups"]) == {
        "left_arm",
        "left_gripper",
        "right_arm",
        "right_gripper",
    }
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["report_identity_sha256"]) == 64
    assert len(result["report_file_sha256"]) == 64


def test_agilex_metrics_use_elementwise_qpos_mae_and_rmse(
    tmp_path: Path,
) -> None:
    result = _evaluate(
        _predictions(tmp_path, pixel_offset=10, qpos_offset=0.25),
        tmp_path / "metrics.json",
    )

    assert result["overall"]["psnr_db"] == pytest.approx(20.0 * np.log10(255.0 / 10.0))
    assert result["overall"]["qpos14_mae"] == pytest.approx(0.25)
    assert result["overall"]["qpos14_rmse"] == pytest.approx(0.25)
    assert result["qpos_groups"]["left_arm"]["mae"] == pytest.approx(0.25)


def test_agilex_scorer_rejects_identity_mismatch_without_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "metrics.json"
    with pytest.raises(ValueError, match="checkpoint_identity_sha256"):
        evaluate_agilex_offline_predictions(
            predictions=_predictions(tmp_path),
            output=output,
            checkpoint_identity_sha256="d" * 64,
            dataset_view_id="agilex-unit-view",
            dataset_view_sha256=DATASET_SHA,
            decoder_sha256=DECODER_SHA,
        )
    assert not output.exists()


def test_public_cli_scores_agilex_predictions(tmp_path: Path) -> None:
    output = tmp_path / "metrics.json"
    result = run_cli(
        (
            "track32",
            "agilex-score-predictions",
            "--predictions",
            str(_predictions(tmp_path)),
            "--output",
            str(output),
            "--checkpoint-identity-sha256",
            CHECKPOINT_SHA,
            "--dataset-view-id",
            "agilex-unit-view",
            "--dataset-view-sha256",
            DATASET_SHA,
            "--decoder-sha256",
            DECODER_SHA,
        )
    )

    assert result["overall"]["psnr_db"] == PSNR_CAP_DB
    assert output.is_file()
