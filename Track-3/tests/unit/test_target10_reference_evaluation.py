from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from n0_twam.evaluation import target10_causal_input as causal_module
from n0_twam.evaluation.sealed_artifact_io import read_json_object
from n0_twam.evaluation.target10_golden_calibration import (
    PublishedGoldenScoreMismatch,
    format_published_score,
)
from n0_twam.evaluation.target10_prediction_artifact_v3 import (
    Target10PredictionSample,
    verify_target10_prediction_artifact,
    write_target10_prediction_artifact,
)
from n0_twam.evaluation.target10_reference_contract import (
    REFERENCE_CONTRACT,
    REFERENCE_METADATA_SHA256,
    REFERENCE_METRIC_SHA256,
    TARGET_EPISODE_IDS,
    TARGET_RAW_ROWS,
    TARGET_TASKS,
    load_target10_reference_roster,
    validate_view_against_reference_roster,
)
from n0_twam.evaluation.target10_reference_materializer import (
    reconstruct_reference_prediction,
    resize_reference_rgb,
)
from n0_twam.evaluation.target10_reference_metric import (
    ReferenceVideoPair,
    evaluate_reference_video_pairs,
)
from n0_twam.evaluation.target10_reference_request import (
    CALIBRATION_POLICY_ALLOW_PROTOCOL_ALIGNED_REPORT,
    CALIBRATION_POLICY_REQUIRE_PUBLISHED_GOLDEN,
    Target10ReferenceEvaluationRequest,
)
from n0_twam.evaluation.video_writer import write_rgb_mp4_reference

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = Path(
    os.environ.get(
        "N0_TARGET10_REFERENCE_ASSET_ROOT",
        REPO_ROOT.parent / "visual-tactile_world_model_pipeline",
    )
)
REFERENCE_METADATA = REFERENCE_ROOT / "metadata_val.json"
REFERENCE_METRIC = Path(
    os.environ.get(
        "N0_TARGET10_REFERENCE_METRIC",
        REFERENCE_ROOT / "metric" / "stage1_holdout_metrics.py",
    )
)
HEX = "a" * 64


def _request_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 6,
        "checkpoint": str(tmp_path / "checkpoint"),
        "checkpoint_sha256": HEX,
        "vae": str(tmp_path / "vae"),
        "base_model": str(tmp_path / "base_model"),
        "empty_embedding": str(tmp_path / "empty_emb.pt"),
        "empty_embedding_sha256": "8" * 64,
        "lerobot_root": str(tmp_path / "lerobot"),
        "normalizer": str(tmp_path / "normalizer.json"),
        "normalizer_sha256": "9" * 64,
        "normalizer_source_view": str(tmp_path / "normalizer_source_view.json"),
        "evaluation_view_manifest": str(tmp_path / "view.json"),
        "reference_metadata": str(REFERENCE_METADATA),
        "raw_root": str(tmp_path / "raw"),
        "manifest": str(tmp_path / "manifest.json"),
        "conversion_report": str(tmp_path / "conversion.json"),
        "official_metric_script": str(REFERENCE_METRIC),
        "golden_root": str(tmp_path / "golden"),
        "golden_manifest": str(tmp_path / "golden.json"),
        "golden_manifest_sha256": HEX,
        "calibration_policy": CALIBRATION_POLICY_REQUIRE_PUBLISHED_GOLDEN,
        "output_root": str(tmp_path / "output"),
        "config_name": "track31_univtac",
        "device": "cuda:0",
        "hip_visible_devices": "0",
        "n_steps": 50,
        "seed": 2026,
    }


def test_reference_contract_binds_exact_sibling_assets() -> None:
    assert hashlib.sha256(REFERENCE_METADATA.read_bytes()).hexdigest() == (
        REFERENCE_METADATA_SHA256
    )
    assert hashlib.sha256(REFERENCE_METRIC.read_bytes()).hexdigest() == (
        REFERENCE_METRIC_SHA256
    )
    roster = load_target10_reference_roster(REFERENCE_METADATA)
    assert roster == tuple(
        (f"{task}/clean/{episode}.hdf5", task, episode)
        for task in TARGET_TASKS
        for episode in TARGET_EPISODE_IDS
    )
    assert TARGET_RAW_ROWS == (0, 5, 10, 15, 20, 25, 30, 35, 40)
    assert REFERENCE_CONTRACT.model_latent_frames == 11
    assert REFERENCE_CONTRACT.model_decoded_frame_count == 41
    assert REFERENCE_CONTRACT.selected_output_indices == TARGET_RAW_ROWS
    assert REFERENCE_CONTRACT.decoded_frame_count == 9
    assert REFERENCE_CONTRACT.expected_future_frame_pairs == 80
    assert REFERENCE_CONTRACT.sampling_steps == 50
    assert REFERENCE_CONTRACT.evaluation_seed == 2026


def test_reference_roster_uses_source_episode_not_global_lerobot_id() -> None:
    entries = tuple(
        SimpleNamespace(
            relative_path=f"{task}/clean/{episode_id}.hdf5",
            task=task,
            source_episode_id=episode_id,
            lerobot_episode_id=global_id,
        )
        for global_id, (task, episode_id) in enumerate(
            (
                (task, episode_id)
                for task in TARGET_TASKS
                for episode_id in TARGET_EPISODE_IDS
            ),
            start=5,
        )
    )
    roster = tuple(
        (entry.relative_path, entry.task, entry.source_episode_id) for entry in entries
    )

    bound = validate_view_against_reference_roster(
        SimpleNamespace(entries=entries), roster
    )

    assert [row["episode_id"] for row in bound] == [0, 1, 2, 3, 5] * 2
    assert [row["lerobot_episode_id"] for row in bound] == list(range(5, 15))


def test_continuous41_selection_uses_exact_raw_time_grid() -> None:
    from script.track3_1.generate_target10_reference_prediction import (
        select_target10_reference_grid,
    )

    continuous = np.empty((2, 41, 128, 128, 3), dtype=np.float32)
    for frame_index in range(41):
        continuous[:, frame_index].fill(frame_index / 100.0)
    selected = select_target10_reference_grid(continuous)
    assert selected.shape == (2, 9, 128, 128, 3)
    np.testing.assert_allclose(
        selected[:, :, 0, 0, 0],
        np.asarray(
            [[value / 100.0 for value in TARGET_RAW_ROWS]] * 2,
            dtype=np.float32,
        ),
    )


def test_evaluation_runtime_paths_are_from_sealed_provenance(tmp_path: Path) -> None:
    from script.track3_1.generate_target10_reference_prediction import (
        _evaluation_runtime_paths,
    )

    evaluation_view = tmp_path / "frozen_target10_v1.json"
    evaluation_view.write_text("{}\n", encoding="utf-8")
    dataset_path = tmp_path / "frozen40"
    dataset_path.mkdir()
    resolved_view, resolved_dataset = _evaluation_runtime_paths(
        {
            "artifact_files": {"evaluation_view": {"path": str(evaluation_view)}},
            "dataset_path": str(dataset_path),
        }
    )
    assert resolved_view == evaluation_view.resolve()
    assert resolved_dataset == dataset_path.resolve()


@pytest.mark.parametrize("forbidden", ["fps", "official_metric_sha256", "resize"])
def test_reference_request_rejects_protocol_overrides(
    tmp_path: Path, forbidden: str
) -> None:
    payload = _request_payload(tmp_path)
    payload[forbidden] = 10
    with pytest.raises(ValueError, match="unexpected"):
        Target10ReferenceEvaluationRequest.from_json_dict(payload)


@pytest.mark.parametrize(("key", "value"), [("n_steps", 8), ("seed", 20260801)])
def test_reference_request_rejects_sampling_protocol_override(
    tmp_path: Path, key: str, value: int
) -> None:
    payload = _request_payload(tmp_path)
    payload[key] = value
    with pytest.raises(ValueError, match="fixes n_steps=50 and seed=2026"):
        Target10ReferenceEvaluationRequest.from_json_dict(payload)


def test_reference_request_rejects_unknown_calibration_policy(tmp_path: Path) -> None:
    payload = _request_payload(tmp_path)
    payload["calibration_policy"] = "ignore_golden"
    with pytest.raises(ValueError, match="calibration_policy must be one of"):
        Target10ReferenceEvaluationRequest.from_json_dict(payload)


def test_bfloat16_causal_tensor_roundtrip() -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).to(torch.bfloat16)
    payload, dtype_name = causal_module._numpy_payload(value)
    restored = causal_module._torch_payload(payload, dtype_name)
    assert restored.dtype == torch.bfloat16
    assert torch.equal(restored, value)


def test_empty_embedding_identity_is_explicit_and_hashed(tmp_path: Path) -> None:
    empty_embedding = tmp_path / "empty_emb.pt"
    empty_embedding.write_bytes(b"empty embedding")
    expected_sha256 = hashlib.sha256(empty_embedding.read_bytes()).hexdigest()

    identity = causal_module._empty_embedding_identity(
        empty_embedding,
        expected_sha256=expected_sha256,
    )

    assert identity == {
        "path": str(empty_embedding.resolve()),
        "size_bytes": len(b"empty embedding"),
        "file_sha256": expected_sha256,
    }
    with pytest.raises(ValueError, match="differs from the request"):
        causal_module._empty_embedding_identity(
            empty_embedding,
            expected_sha256="0" * 64,
        )


def test_empty_embedding_identity_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.pt"
    target.write_bytes(b"empty embedding")
    symlink = tmp_path / "empty_emb.pt"
    symlink.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        causal_module._empty_embedding_identity(
            symlink,
            expected_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        )


def test_reference_reconstruction_and_resize_contract() -> None:
    raw = np.zeros((2, 9, 64, 80, 3), dtype=np.uint8)
    raw[:, :, :, :, 0] = 20
    residual = np.zeros((2, 9, 128, 128, 3), dtype=np.float32)
    residual[:, 1:, :, :, 1] = 0.25
    prediction, ground_truth = reconstruct_reference_prediction(raw, residual)
    assert prediction.shape == ground_truth.shape == (9, 192, 512, 3)
    assert prediction.dtype == ground_truth.dtype == np.uint8
    assert np.all(prediction[1:, :, :, 1] == 64)
    assert resize_reference_rgb(raw[0, 0]).shape == (192, 256, 3)


def test_reference_writer_and_pinned_metric_aggregate(tmp_path: Path) -> None:
    prediction = np.zeros((9, 192, 512, 3), dtype=np.uint8)
    ground_truth = prediction.copy()
    prediction_path = tmp_path / "pred.mp4"
    ground_truth_path = tmp_path / "gt.mp4"
    evidence = write_rgb_mp4_reference(prediction_path, prediction, fps=2)
    write_rgb_mp4_reference(ground_truth_path, ground_truth, fps=2)
    assert evidence["pixel_format"] == "yuv420p"
    pairs = [
        ReferenceVideoPair(
            task=task,
            sample_id=f"sample_{sample_index:03d}",
            sample_index=sample_index,
            episode_id=episode_id,
            prediction_path=prediction_path,
            ground_truth_path=ground_truth_path,
        )
        for task in TARGET_TASKS
        for sample_index, episode_id in enumerate(TARGET_EPISODE_IDS)
    ]
    report = evaluate_reference_video_pairs(pairs=pairs, metric_script=REFERENCE_METRIC)
    assert report["counts"] == {"episodes": 10, "future_frame_pairs": 80}
    assert report["overall"] == {
        "task_count": 2,
        "episode_count": 10,
        "future_frame_pairs": 80,
        "average_psnr": 100.0,
        "average_ssim": 1.0,
    }


def test_prediction_artifact_v3_roundtrip_and_tamper(tmp_path: Path) -> None:
    samples = []
    for index, (task, episode_id) in enumerate(
        (task, episode) for task in TARGET_TASKS for episode in TARGET_EPISODE_IDS
    ):
        samples.append(
            Target10PredictionSample(
                sample_id=f"sample_{index:02d}",
                dataset_index=index,
                episode_index=episode_id,
                task=task,
                source_relative_path=f"{task}/clean/{episode_id}.hdf5",
                signed_residual=np.zeros((2, 9, 128, 128, 3), dtype=np.float32),
            )
        )
    artifact_path = tmp_path / "artifact"
    write_target10_prediction_artifact(
        artifact_path,
        samples=samples,
        source_manifest_sha256=HEX,
        conversion_report_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        decoder_identity_sha256="d" * 64,
        evaluation_view_id="frozen_target10_v1",
        evaluation_view_sha256="e" * 64,
        generation_provenance={"test": True},
    )
    verified = verify_target10_prediction_artifact(artifact_path)
    assert verified.signed_residual.shape == (10, 2, 9, 128, 128, 3)
    metadata_path = artifact_path / "artifact.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["decoded_frame_count"] = 17
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_target10_prediction_artifact(artifact_path)


def test_published_score_rounding_is_explicit() -> None:
    assert format_published_score(21.255, places=2) == "21.26"
    assert format_published_score(0.7455, places=3) == "0.746"


@pytest.mark.parametrize(
    ("calibration_policy", "score_mismatch", "expected_comparable"),
    (
        (CALIBRATION_POLICY_REQUIRE_PUBLISHED_GOLDEN, False, True),
        (CALIBRATION_POLICY_ALLOW_PROTOCOL_ALIGNED_REPORT, True, False),
    ),
)
def test_pipeline_executes_golden_before_hcu_and_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_policy: str,
    score_mismatch: bool,
    expected_comparable: bool,
) -> None:
    from n0_twam.evaluation import target10_reference_pipeline as pipeline

    payload = _request_payload(tmp_path)
    payload["calibration_policy"] = calibration_policy
    request = Target10ReferenceEvaluationRequest.from_json_dict(payload)
    view = SimpleNamespace(view_id="frozen_target10_v1", view_sha256="f" * 64)
    causal_dataset_identity = {
        "source_manifest_sha256": "1" * 64,
        "normalizer_sha256": "7" * 64,
        "conversion_report_sha256": "2" * 64,
        "evaluation_view_id": view.view_id,
        "evaluation_view_sha256": view.view_sha256,
        "encoder_source_identity": {"identity_sha256": "8" * 64},
        "latent_inventory_validation": {
            "video_inventory_sha256": "9" * 64,
            "tactile_inventory_sha256": "a" * 64,
        },
        "empty_embedding": {
            "path": str(tmp_path / "empty_emb.pt"),
            "size_bytes": 4,
            "file_sha256": "8" * 64,
        },
    }
    input_identity = {
        "manifest_payload_sha256": "1" * 64,
        "conversion_report_payload_sha256": "2" * 64,
        "vae_decoder_identity_sha256": "3" * 64,
        "strict_checkpoint_identity": {
            "schema_version": 1,
            "identity_sha256": "b" * 64,
        },
        "causal_dataset_provenance": causal_dataset_identity,
    }
    events: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "normalize_reference_request",
        lambda value: (value, view, input_identity),
    )

    def calibrate(**kwargs: object) -> dict[str, object]:
        events.append("golden")
        if score_mismatch:
            raise PublishedGoldenScoreMismatch(
                {
                    "status": "failed_score_mismatch",
                    "published_display": {"psnr": "26.46", "ssim": "0.891"},
                    "published_score_comparable": False,
                }
            )
        return {"status": "pass", "published_score_comparable": True}

    monkeypatch.setattr(pipeline, "calibrate_target10_golden", calibrate)
    causal = SimpleNamespace(
        seal_sha256="4" * 64,
        metadata={
            "evaluation_view_id": view.view_id,
            "evaluation_view_sha256": view.view_sha256,
            "dataset_provenance": causal_dataset_identity,
        },
    )

    def build_causal(output: Path, **kwargs: object) -> Path:
        events.append("causal")
        output.mkdir()
        return output

    monkeypatch.setattr(pipeline, "build_target10_causal_input_bundle", build_causal)
    monkeypatch.setattr(
        pipeline, "verify_target10_causal_input_bundle", lambda path: causal
    )
    monkeypatch.setattr(
        pipeline,
        "_run_stage_a",
        lambda *args, **kwargs: events.append("hcu"),
    )
    artifact = SimpleNamespace(
        root=request.output_root / "prediction_artifact",
        seal_sha256="5" * 64,
        file_sha256={"artifact.json": "6" * 64},
        metadata={
            "checkpoint_sha256": request.checkpoint_sha256,
            "source_manifest_sha256": "1" * 64,
            "conversion_report_sha256": "2" * 64,
            "evaluation_view_id": view.view_id,
            "evaluation_view_sha256": view.view_sha256,
            "decoder_identity_sha256": "3" * 64,
            "generation_provenance": {
                "config_name": request.config_name,
                "n_steps": request.n_steps,
                "seed_contract": {"seed": request.seed},
                "conditioning_bundle": {"seal_sha256": "4" * 64},
                "dataset_provenance": causal_dataset_identity,
                "strict_checkpoint_identity": input_identity[
                    "strict_checkpoint_identity"
                ],
                "empty_embedding": causal_dataset_identity["empty_embedding"],
                "checkpoint_dataset_binding": {
                    "manifest_sha256": "1" * 64,
                    "normalizer_sha256": "7" * 64,
                    "conversion_report_sha256": "2" * 64,
                },
            },
        },
    )
    monkeypatch.setattr(
        pipeline, "verify_target10_prediction_artifact", lambda path: artifact
    )
    monkeypatch.setattr(
        pipeline,
        "verify_materialized_target10_reference",
        lambda **kwargs: events.append("recompute"),
    )

    def metric_evaluator(**kwargs: object) -> dict[str, object]:
        events.append("metric")
        output = Path(str(kwargs["output"]))
        output.mkdir()
        report = {
            "protocol": REFERENCE_CONTRACT.contract_id,
            "contract_sha256": REFERENCE_CONTRACT.sha256,
            "prediction_artifact": {"seal_sha256": artifact.seal_sha256},
            "evaluation_view": {
                "view_id": view.view_id,
                "view_sha256": view.view_sha256,
            },
            "metric": {
                "metric_script_sha256": REFERENCE_METRIC_SHA256,
                "counts": {"episodes": 10, "future_frame_pairs": 80},
                "overall": {"average_psnr": 22.0, "average_ssim": 0.75},
            },
        }
        (output / "tactile_prediction_quality.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    summary = pipeline.run_target10_reference_evaluation(
        request, metric_evaluator=metric_evaluator
    )
    assert events == ["golden", "causal", "hcu", "metric", "recompute"]
    assert summary["published_score_comparable"] is expected_comparable
    if expected_comparable:
        assert "published_reference" in summary
        assert "reported_reference_numeric_only" not in summary
    else:
        assert "published_reference" not in summary
        assert summary["reported_reference_numeric_only"]["comparison_status"] == (
            "not_published_golden_calibrated"
        )
    assert (
        read_json_object(request.output_root / "evaluation_receipt.json")["score"][
            "display_psnr"
        ]
        == "22.00"
    )


def test_strict_calibration_policy_rejects_score_mismatch_before_hcu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from n0_twam.evaluation import target10_reference_pipeline as pipeline

    request = Target10ReferenceEvaluationRequest.from_json_dict(
        _request_payload(tmp_path)
    )

    def reject(**kwargs: object) -> dict[str, object]:
        raise PublishedGoldenScoreMismatch(
            {
                "status": "failed_score_mismatch",
                "published_display": {"psnr": "26.46", "ssim": "0.891"},
                "published_score_comparable": False,
            }
        )

    monkeypatch.setattr(pipeline, "calibrate_target10_golden", reject)
    with pytest.raises(PublishedGoldenScoreMismatch):
        pipeline._calibrate_golden(request)


def test_stage_a_environment_comes_only_from_request_and_checkpoint(
    tmp_path: Path,
) -> None:
    from n0_twam.evaluation import target10_reference_pipeline as pipeline

    payload = _request_payload(tmp_path)
    request = Target10ReferenceEvaluationRequest.from_json_dict(payload)
    request.checkpoint.mkdir()
    (request.checkpoint / "train_meta.json").write_text(
        json.dumps(
            {
                "training_profile_id": "multitask_pretrain_v1",
                "run_role": "final_refit",
            }
        ),
        encoding="utf-8",
    )

    environment = pipeline._stage_a_environment(
        request,
        base_environment={"PRESERVED": "yes", "N0_BASE_MODEL": "/wrong"},
    )

    assert environment["PRESERVED"] == "yes"
    assert environment["N0_TRACK31_ARTIFACT_ROOT"] == str(
        request.conversion_report.parent
    )
    assert environment["N0_TRACK31_LEROBOT_ROOT"] == str(request.lerobot_root)
    assert environment["N0_TRACK31_MANIFEST_PATH"] == str(request.manifest)
    assert environment["N0_TRACK31_NORMALIZER_PATH"] == str(request.normalizer)
    assert environment["N0_TRACK31_NORMALIZER_SOURCE_VIEW_PATH"] == str(
        request.normalizer_source_view
    )
    assert environment["N0_BASE_MODEL"] == str(request.base_model)
    assert environment["N0_EMPTY_EMBEDDING"] == str(request.empty_embedding)
    assert environment["N0_EMPTY_EMBEDDING_SHA256"] == (request.empty_embedding_sha256)
    assert environment["N0_RELEASED_CHECKPOINT"] == str(request.base_model)
    assert environment["N0_TRACK31_TRAIN_PROFILE"] == "multitask_pretrain_v1"
    assert environment["N0_TRACK31_RUN_ROLE"] == "final_refit"
    assert environment["HIP_VISIBLE_DEVICES"] == "0"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["PYTHONHASHSEED"] == "2026"


def test_reused_reference_artifact_rejects_strict_checkpoint_drift() -> None:
    from n0_twam.evaluation.target10_reference_reuse import (
        artifact_matches_reference_request,
    )

    request = SimpleNamespace(
        checkpoint_sha256="1" * 64,
        config_name="track31_univtac",
        n_steps=50,
        seed=2026,
    )
    view = SimpleNamespace(view_id="frozen_target10_v1", view_sha256="2" * 64)
    dataset = {
        "source_manifest_sha256": "3" * 64,
        "conversion_report_sha256": "4" * 64,
        "normalizer_sha256": "5" * 64,
        "evaluation_view_id": view.view_id,
        "evaluation_view_sha256": view.view_sha256,
        "empty_embedding": {"file_sha256": "6" * 64},
    }
    strict_checkpoint = {
        "schema_version": 1,
        "identity_sha256": "7" * 64,
    }
    identity = {
        "manifest_payload_sha256": "3" * 64,
        "conversion_report_payload_sha256": "4" * 64,
        "vae_decoder_identity_sha256": "8" * 64,
        "causal_dataset_provenance": dataset,
        "strict_checkpoint_identity": strict_checkpoint,
    }
    provenance = {
        "config_name": request.config_name,
        "n_steps": request.n_steps,
        "seed_contract": {"seed": request.seed},
        "conditioning_bundle": {"seal_sha256": "9" * 64},
        "dataset_provenance": dataset,
        "strict_checkpoint_identity": strict_checkpoint,
        "empty_embedding": dataset["empty_embedding"],
        "checkpoint_dataset_binding": {
            "manifest_sha256": "3" * 64,
            "conversion_report_sha256": "4" * 64,
            "normalizer_sha256": "5" * 64,
        },
    }
    artifact = SimpleNamespace(
        metadata={
            "checkpoint_sha256": request.checkpoint_sha256,
            "source_manifest_sha256": "3" * 64,
            "conversion_report_sha256": "4" * 64,
            "evaluation_view_id": view.view_id,
            "evaluation_view_sha256": view.view_sha256,
            "decoder_identity_sha256": "8" * 64,
            "generation_provenance": provenance,
        }
    )
    assert artifact_matches_reference_request(
        artifact,
        request=request,
        view=view,
        input_identity=identity,
        causal_seal_sha256="9" * 64,
    )
    drifted = json.loads(json.dumps(provenance))
    drifted["strict_checkpoint_identity"]["identity_sha256"] = "a" * 64
    artifact.metadata["generation_provenance"] = drifted
    assert not artifact_matches_reference_request(
        artifact,
        request=request,
        view=view,
        input_identity=identity,
        causal_seal_sha256="9" * 64,
    )


def test_v18_request_rejects_silently_ignored_overrides(tmp_path: Path) -> None:
    from script.track3_1.run_target10_tactile_evaluation_v18_step1500 import main

    with pytest.raises(SystemExit):
        main(
            [
                "--request",
                str(tmp_path / "request.json"),
                "--checkpoint",
                str(tmp_path / "different_checkpoint"),
            ]
        )


def test_reused_causal_bundle_rejects_encoder_or_inventory_drift() -> None:
    from n0_twam.evaluation.target10_reference_pipeline import (
        _require_causal_dataset_binding,
    )

    expected = {
        "encoder_source_identity": {"identity_sha256": "a" * 64},
        "latent_inventory_validation": {
            "video_inventory_sha256": "b" * 64,
            "tactile_inventory_sha256": "c" * 64,
        },
        "normalizer_source_view_sha256": "d" * 64,
    }
    _require_causal_dataset_binding(expected, {"causal_dataset_provenance": expected})
    drifted = json.loads(json.dumps(expected))
    drifted["encoder_source_identity"]["identity_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="another data bundle"):
        _require_causal_dataset_binding(
            drifted, {"causal_dataset_provenance": expected}
        )


def test_reference_metric_receipt_rejects_report_tamper(tmp_path: Path) -> None:
    from n0_twam.evaluation.target10_reference_receipts import (
        verify_reference_metric_receipt,
        write_reference_metric_receipt,
    )

    metric_root = tmp_path / "metric"
    metric_root.mkdir()
    report_path = metric_root / "tactile_prediction_quality.json"
    report_path.write_text('{"score": 1}\n', encoding="utf-8")
    golden_path = tmp_path / "golden_calibration.json"
    golden_path.write_text('{"status": "pass"}\n', encoding="utf-8")
    artifact = SimpleNamespace(
        seal_sha256="a" * 64,
        file_sha256={"artifact.json": "b" * 64},
    )
    receipt_path = tmp_path / "metric_receipt.json"
    arguments = {
        "metric_root": metric_root,
        "report_path": report_path,
        "artifact": artifact,
        "request_sha256": "c" * 64,
        "input_identity": {"manifest": "d" * 64},
        "evaluation_view_id": "frozen_target10_v1",
        "evaluation_view_sha256": "e" * 64,
        "golden_calibration_path": golden_path,
    }
    write_reference_metric_receipt(receipt_path, **arguments)
    verify_reference_metric_receipt(receipt_path, **arguments)
    report_path.write_text('{"score": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="drifted"):
        verify_reference_metric_receipt(receipt_path, **arguments)
