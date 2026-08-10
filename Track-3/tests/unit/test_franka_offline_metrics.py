# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from n0_twam.cli import run_cli
from n0_twam.evaluation.franka_offline_metrics import (
    PREDICTION_ARTIFACT_TYPE,
    PSNR_CAP_DB,
    evaluate_franka_offline_predictions,
)
from n0_twam.evaluation.sealed_artifact_io import canonical_json
from n0_twam.integrations.worldarena.franka_views import build_standard_franka_views

CHECKPOINT_SHA = "a" * 64
DATASET_SHA = "b" * 64
DECODER_SHA = "c" * 64


def _prediction_artifact(
    tmp_path: Path,
    *,
    tasks: tuple[str, ...] = ("clear_up", "pour", "wipe"),
    run_role: str = "development",
    dataset_view_id: str = "unit_test_view",
    pixel_offset: int = 0,
    position_offsets_m: tuple[float, float] = (0.0, 0.0),
    frame_offsets: tuple[int, ...] = (1, 2),
    lerobot_episode_ids: tuple[int, ...] | None = None,
) -> Path:
    sample_count = len(tasks)
    target_rgb = np.zeros(
        (sample_count, 2, len(frame_offsets), 8, 8, 3), dtype=np.uint8
    )
    predicted_rgb = np.full_like(target_rgb, pixel_offset)
    target_pose = np.zeros((sample_count, 2, 8), dtype=np.float32)
    predicted_pose = target_pose.copy()
    target_pose[..., 3] = 1.0
    predicted_pose[..., 3] = 1.0
    predicted_pose[:, 0, 0] = position_offsets_m[0]
    predicted_pose[:, 1, 0] = position_offsets_m[1]
    if lerobot_episode_ids is None:
        task_counts = {task: 0 for task in ("clear_up", "pour", "wipe")}
        derived_ids: list[int] = []
        for task in tasks:
            task_index = ("clear_up", "pour", "wipe").index(task)
            derived_ids.append(task_index * 200 + task_counts[task])
            task_counts[task] += 1
        lerobot_episode_ids = tuple(derived_ids)
    standard_view = build_standard_franka_views().get(dataset_view_id)
    dataset_view_sha256 = (
        standard_view.view_sha256 if standard_view is not None else DATASET_SHA
    )
    path = tmp_path / "predictions.npz"
    np.savez(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        artifact_type=np.asarray(PREDICTION_ARTIFACT_TYPE),
        checkpoint_identity_sha256=np.asarray(CHECKPOINT_SHA),
        dataset_view_id=np.asarray(dataset_view_id),
        dataset_view_sha256=np.asarray(dataset_view_sha256),
        decoder_sha256=np.asarray(DECODER_SHA),
        seed=np.asarray(20260810, dtype=np.int64),
        run_role=np.asarray(run_role),
        prediction_mode=np.asarray("policy_action"),
        view_names=np.asarray(("cam_high", "cam_left_wrist")),
        frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
        action_offsets=np.asarray((1, 2), dtype=np.int64),
        sample_ids=np.asarray(
            tuple(f"sample-{index}" for index in range(sample_count))
        ),
        task_ids=np.asarray(tasks),
        lerobot_episode_ids=np.asarray(lerobot_episode_ids, dtype=np.int64),
        predicted_rgb=predicted_rgb,
        target_rgb=target_rgb,
        video_valid=np.ones(target_rgb.shape[:3], dtype=np.bool_),
        predicted_end_pose=predicted_pose,
        target_end_pose=target_pose,
        action_valid=np.ones(target_pose.shape[:2], dtype=np.bool_),
    )
    return path


def _evaluate(predictions: Path, output: Path) -> dict[str, object]:
    with np.load(predictions, allow_pickle=False) as archive:
        dataset_view_id = str(archive["dataset_view_id"].item())
        dataset_view_sha256 = str(archive["dataset_view_sha256"].item())
    return evaluate_franka_offline_predictions(
        predictions=predictions,
        output=output,
        checkpoint_identity_sha256=CHECKPOINT_SHA,
        dataset_view_id=dataset_view_id,
        dataset_view_sha256=dataset_view_sha256,
        decoder_sha256=DECODER_SHA,
    )


def test_perfect_future_predictions_have_perfect_metrics_and_sealed_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "metrics.json"
    result = _evaluate(_prediction_artifact(tmp_path), output)

    assert result["status"] == "complete"
    assert result["organizer_evaluation_completed"] is False
    assert result["real_robot_evaluation_completed"] is False
    assert result["overall"] == pytest.approx(
        {
            "psnr_db": PSNR_CAP_DB,
            "ssim": 1.0,
            "position_mae_cm": 0.0,
            "position_rmse_cm": 0.0,
        }
    )
    assert set(result["per_task"]) == {"clear_up", "pour", "wipe"}
    assert set(result["per_view"]) == {"cam_high", "cam_left_wrist"}
    assert result["internal_holdout_complete"] is False
    assert result["generalization_claim_valid"] is False
    payload = json.loads(output.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"report_identity_sha256", "prediction_artifact"}
    }
    assert (
        payload["report_identity_sha256"]
        == hashlib.sha256(canonical_json(core)).hexdigest()
    )
    assert len(result["report_file_sha256"]) == 64


def test_position_metrics_are_l2_mae_and_rmse_in_centimeters(tmp_path: Path) -> None:
    predictions = _prediction_artifact(
        tmp_path,
        pixel_offset=10,
        position_offsets_m=(0.01, 0.03),
    )
    result = _evaluate(predictions, tmp_path / "metrics.json")

    assert result["overall"]["psnr_db"] == pytest.approx(20.0 * np.log10(255.0 / 10.0))
    assert result["overall"]["ssim"] == pytest.approx(0.06105490481444098)
    assert result["overall"]["position_mae_cm"] == pytest.approx(2.0)
    assert result["overall"]["position_rmse_cm"] == pytest.approx(np.sqrt(5.0))


def test_canonical_internal_holdout_roster_never_self_certifies_target_bytes(
    tmp_path: Path,
) -> None:
    validation = build_standard_franka_views()["franka_dev_validation60_v1"]
    tasks = tuple(entry.task for entry in validation.entries)
    episode_ids = tuple(entry.lerobot_episode_id for entry in validation.entries)
    development = _prediction_artifact(
        tmp_path,
        tasks=tasks,
        dataset_view_id="franka_dev_validation60_v1",
        lerobot_episode_ids=episode_ids,
    )
    development_result = _evaluate(development, tmp_path / "development.json")
    assert development_result["canonical_view_roster_match"] is True
    assert development_result["internal_holdout_complete"] is False
    assert development_result["generalization_claim_valid"] is False

    final_dir = tmp_path / "final"
    final_dir.mkdir()
    final_refit = _prediction_artifact(
        final_dir,
        tasks=tasks,
        run_role="final_refit",
        dataset_view_id="franka_dev_validation60_v1",
        lerobot_episode_ids=episode_ids,
    )
    final_result = _evaluate(final_refit, tmp_path / "final.json")
    assert final_result["canonical_view_roster_match"] is True
    assert final_result["internal_holdout_complete"] is False
    assert final_result["generalization_claim_valid"] is False


def test_scorer_rejects_identity_mismatch_and_conditioning_frame(
    tmp_path: Path,
) -> None:
    predictions = _prediction_artifact(tmp_path)
    output = tmp_path / "metrics.json"
    with pytest.raises(ValueError, match="checkpoint_identity_sha256"):
        evaluate_franka_offline_predictions(
            predictions=predictions,
            output=output,
            checkpoint_identity_sha256="d" * 64,
            dataset_view_id="unit_test_view",
            dataset_view_sha256=DATASET_SHA,
            decoder_sha256=DECODER_SHA,
        )
    assert not output.exists()

    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    invalid = _prediction_artifact(invalid_dir, frame_offsets=(0, 1))
    with pytest.raises(ValueError, match="exclude offset zero"):
        _evaluate(invalid, output)
    assert not output.exists()


def test_scorer_rejects_prediction_symlink_and_existing_output(tmp_path: Path) -> None:
    predictions = _prediction_artifact(tmp_path)
    link = tmp_path / "prediction-link.npz"
    link.symlink_to(predictions)
    output = tmp_path / "metrics.json"
    with pytest.raises(ValueError, match="non-symlink"):
        _evaluate(link, output)
    assert not output.exists()

    output.write_text("reserved", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _evaluate(predictions, output)
    assert output.read_text(encoding="utf-8") == "reserved"


def test_public_cli_scores_prediction_artifact(tmp_path: Path) -> None:
    predictions = _prediction_artifact(tmp_path)
    output = tmp_path / "metrics.json"
    result = run_cli(
        (
            "track32",
            "score-predictions",
            "--predictions",
            str(predictions),
            "--output",
            str(output),
            "--checkpoint-identity-sha256",
            CHECKPOINT_SHA,
            "--dataset-view-id",
            "unit_test_view",
            "--dataset-view-sha256",
            DATASET_SHA,
            "--decoder-sha256",
            DECODER_SHA,
        )
    )

    assert result["overall"]["psnr_db"] == PSNR_CAP_DB
    assert output.is_file()


def test_public_cli_materializes_prediction_artifact(tmp_path: Path) -> None:
    source = _prediction_artifact(tmp_path)
    with np.load(source, allow_pickle=False) as archive:
        metadata = {
            name: archive[name].item()
            for name in (
                "checkpoint_identity_sha256",
                "dataset_view_id",
                "dataset_view_sha256",
                "decoder_sha256",
                "seed",
                "run_role",
                "prediction_mode",
            )
        }
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
            if name
            not in {
                "schema_version",
                "artifact_type",
                *metadata,
            }
        }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    arrays_path = tmp_path / "arrays.npz"
    np.savez(arrays_path, **arrays)
    source.unlink()

    result = run_cli(
        (
            "track32",
            "pack-predictions",
            "--metadata",
            str(metadata_path),
            "--arrays",
            str(arrays_path),
            "--output",
            str(source),
        )
    )

    assert result["status"] == "complete"
    assert len(result["sha256"]) == 64
    assert _evaluate(source, tmp_path / "packed-metrics.json")["status"] == "complete"
