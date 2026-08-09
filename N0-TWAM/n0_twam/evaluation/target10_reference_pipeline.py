# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""One-command end-to-end Target-10 reference9 evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from n0_twam.evaluation.sealed_artifact_io import (
    canonical_json,
    read_json_object,
    sha256_file,
)
from n0_twam.evaluation.tactile_provenance import write_json_atomic
from n0_twam.evaluation.target10_causal_input import (
    build_target10_causal_input_bundle,
    verify_target10_causal_input_bundle,
)
from n0_twam.evaluation.target10_evaluation_receipts import (
    _EvaluationLock,
    _acquire_lock,
)
from n0_twam.evaluation.target10_golden_calibration import (
    PublishedGoldenScoreMismatch,
    calibrate_target10_golden,
    format_published_score,
)
from n0_twam.evaluation.target10_prediction_artifact_v3 import (
    verify_target10_prediction_artifact,
)
from n0_twam.evaluation.target10_reference_contract import REFERENCE_CONTRACT
from n0_twam.evaluation.target10_reference_materializer import (
    materialize_and_evaluate_target10_reference,
)
from n0_twam.evaluation.target10_reference_preflight import (
    normalize_reference_request,
)
from n0_twam.evaluation.target10_reference_receipts import (
    verify_reference_metric_receipt,
    write_reference_metric_receipt,
)
from n0_twam.evaluation.target10_reference_reuse import (
    artifact_matches_reference_request,
    validate_reference_metric_report,
    verify_materialized_target10_reference,
)
from n0_twam.evaluation.target10_reference_request import (
    CALIBRATION_POLICY_ALLOW_PROTOCOL_ALIGNED_REPORT,
    REQUEST_SCHEMA_VERSION,
    Target10ReferenceEvaluationRequest,
    load_target10_reference_request,
)

PIPELINE_SCHEMA_VERSION = 6


ReferenceMetricEvaluator = Callable[..., dict[str, object]]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _request_sha256(
    request: Target10ReferenceEvaluationRequest,
    input_identity: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        canonical_json(
            {"request": request.to_json_dict(), "input_identity": dict(input_identity)}
        )
    ).hexdigest()


def _write_state(root: Path, request_sha256: str, stage: str, status: str) -> None:
    write_json_atomic(
        root / "evaluation_state.json",
        {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "stage": stage,
            "status": status,
            "updated_at_utc": _now_utc(),
        },
    )


def _ensure_request_receipt(
    root: Path,
    request: Target10ReferenceEvaluationRequest,
    request_sha256: str,
    input_identity: Mapping[str, object],
) -> None:
    path = root / "evaluation_request.json"
    payload = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "request": request.to_json_dict(),
        "input_identity": dict(input_identity),
    }
    if path.exists():
        if read_json_object(path) != payload:
            raise ValueError("output_root belongs to another reference request")
    else:
        write_json_atomic(path, payload)


def _require_causal_dataset_binding(
    causal_dataset: object,
    input_identity: Mapping[str, object],
) -> None:
    """Require a reused causal bundle to match every audited dataset input."""

    expected = input_identity.get("causal_dataset_provenance")
    if (
        not isinstance(causal_dataset, Mapping)
        or not isinstance(expected, Mapping)
        or dict(causal_dataset) != dict(expected)
    ):
        raise ValueError("causal input bundle belongs to another data bundle")


def _calibrate_golden(
    request: Target10ReferenceEvaluationRequest,
) -> tuple[dict[str, object], bool]:
    """Run the golden gate, allowing only an explicit score-mismatch fallback."""

    try:
        calibration = calibrate_target10_golden(
            golden_root=request.golden_root,
            golden_manifest_path=request.golden_manifest,
            expected_manifest_sha256=request.golden_manifest_sha256,
            metric_script=request.official_metric_script,
        )
    except PublishedGoldenScoreMismatch as exc:
        if (
            request.calibration_policy
            != CALIBRATION_POLICY_ALLOW_PROTOCOL_ALIGNED_REPORT
        ):
            raise
        calibration = dict(exc.evidence)
    passed = (
        calibration.get("status") == "pass"
        and calibration.get("published_score_comparable") is True
    )
    allowed_failure = (
        calibration.get("status") == "failed_score_mismatch"
        and calibration.get("published_score_comparable") is False
        and request.calibration_policy
        == CALIBRATION_POLICY_ALLOW_PROTOCOL_ALIGNED_REPORT
    )
    if not passed and not allowed_failure:
        raise RuntimeError("golden calibration returned an invalid status")
    return calibration, passed


def _stage_a_environment(
    request: Target10ReferenceEvaluationRequest,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Bind the inference config import to the already-audited request paths."""

    train_meta = read_json_object(request.checkpoint / "train_meta.json")
    training_profile = train_meta.get("training_profile_id")
    run_role = train_meta.get("run_role")
    if not isinstance(training_profile, str) or not training_profile:
        raise ValueError("checkpoint train_meta lacks training_profile_id")
    if not isinstance(run_role, str) or not run_role:
        raise ValueError("checkpoint train_meta lacks run_role")
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(
        {
            "N0_TRACK31_ARTIFACT_ROOT": str(request.conversion_report.parent),
            "N0_TRACK31_LEROBOT_ROOT": str(request.lerobot_root),
            "N0_TRACK31_MANIFEST_PATH": str(request.manifest),
            "N0_TRACK31_NORMALIZER_PATH": str(request.normalizer),
            "N0_TRACK31_NORMALIZER_SOURCE_VIEW_PATH": str(
                request.normalizer_source_view
            ),
            "N0_BASE_MODEL": str(request.base_model),
            "N0_EMPTY_EMBEDDING": str(request.empty_embedding),
            "N0_EMPTY_EMBEDDING_SHA256": request.empty_embedding_sha256,
            "N0_RELEASED_CHECKPOINT": str(request.base_model),
            "N0_TRACK31_TRAIN_PROFILE": training_profile,
            "N0_TRACK31_RUN_ROLE": run_role,
            "PYTHONHASHSEED": str(request.seed),
        }
    )
    if request.hip_visible_devices is not None:
        environment["HIP_VISIBLE_DEVICES"] = request.hip_visible_devices
        environment["CUDA_VISIBLE_DEVICES"] = request.hip_visible_devices
    return environment


def _run_stage_a(
    request: Target10ReferenceEvaluationRequest,
    *,
    causal_input: Path,
    artifact_path: Path,
    input_identity: Mapping[str, object],
    log_path: Path,
) -> None:
    strict_checkpoint = input_identity.get("strict_checkpoint_identity")
    strict_checkpoint_sha256 = (
        strict_checkpoint.get("identity_sha256")
        if isinstance(strict_checkpoint, Mapping)
        else None
    )
    if not isinstance(strict_checkpoint_sha256, str):
        raise ValueError("strict checkpoint identity is missing from preflight")
    command = [
        sys.executable,
        "-m",
        "n0_twam.evaluation.target10_reference_generation",
        "--config-name",
        request.config_name,
        "--ckpt",
        str(request.checkpoint),
        "--vae",
        str(request.vae),
        "--causal-input-bundle",
        str(causal_input),
        "--empty-embedding",
        str(request.empty_embedding),
        "--empty-embedding-sha256",
        request.empty_embedding_sha256,
        "--output",
        str(artifact_path),
        "--source-manifest-sha256",
        str(input_identity["manifest_payload_sha256"]),
        "--conversion-report-sha256",
        str(input_identity["conversion_report_payload_sha256"]),
        "--strict-checkpoint-identity-sha256",
        strict_checkpoint_sha256,
        "--n-steps",
        str(request.n_steps),
        "--seed",
        str(request.seed),
        "--device",
        request.device,
    ]
    environment = _stage_a_environment(request)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_now_utc()}] command={json.dumps(command)}\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"reference Stage A failed ({return_code}); see {log_path}")


def run_target10_reference_evaluation(
    request: Target10ReferenceEvaluationRequest,
    *,
    metric_evaluator: ReferenceMetricEvaluator = (
        materialize_and_evaluate_target10_reference
    ),
) -> dict[str, object]:
    """Execute the complete reference pipeline and return the final score receipt."""

    request, view, input_identity = normalize_reference_request(request)
    request_sha = _request_sha256(request, input_identity)
    root = request.output_root
    root.mkdir(parents=True, exist_ok=True)
    lock: _EvaluationLock = _acquire_lock(root, schema_version=PIPELINE_SCHEMA_VERSION)
    request_bound = False
    try:
        _ensure_request_receipt(root, request, request_sha, input_identity)
        request_bound = True
        _write_state(root, request_sha, "golden_calibration", "running")
        golden, published_score_comparable = _calibrate_golden(request)
        write_json_atomic(root / "golden_calibration.json", golden)
        golden_calibration_path = root / "golden_calibration.json"

        causal_path = root / "causal_input_bundle"
        _write_state(root, request_sha, "build_causal_input", "running")
        if causal_path.exists():
            causal = verify_target10_causal_input_bundle(causal_path)
        else:
            build_target10_causal_input_bundle(
                causal_path,
                config_name=request.config_name,
                evaluation_view_path=request.evaluation_view_manifest,
                lerobot_root=request.lerobot_root,
                manifest_path=request.manifest,
                conversion_report_path=request.conversion_report,
                normalizer_path=request.normalizer,
                normalizer_source_view_path=request.normalizer_source_view,
                base_model_path=request.base_model,
                empty_embedding_path=request.empty_embedding,
                expected_manifest_sha256=str(input_identity["manifest_payload_sha256"]),
                expected_normalizer_sha256=request.normalizer_sha256,
                expected_empty_embedding_sha256=(request.empty_embedding_sha256),
            )
            causal = verify_target10_causal_input_bundle(causal_path)
        if (
            causal.metadata.get("evaluation_view_id") != view.view_id
            or causal.metadata.get("evaluation_view_sha256") != view.view_sha256
        ):
            raise ValueError("causal input bundle belongs to another view")
        causal_dataset = causal.metadata.get("dataset_provenance")
        _require_causal_dataset_binding(causal_dataset, input_identity)

        artifact_path = root / "prediction_artifact"
        _write_state(root, request_sha, "generate_prediction", "running")
        reused_artifact = artifact_path.exists()
        if not reused_artifact:
            _run_stage_a(
                request,
                causal_input=causal_path,
                artifact_path=artifact_path,
                input_identity=input_identity,
                log_path=root / "logs" / "stage_a.log",
            )
        artifact = verify_target10_prediction_artifact(artifact_path)
        if not artifact_matches_reference_request(
            artifact,
            request=request,
            view=view,
            input_identity=input_identity,
            causal_seal_sha256=causal.seal_sha256,
        ):
            raise ValueError("reference prediction artifact disagrees with the request")

        metric_root = root / "reference_tactile_quality"
        metric_report_path = metric_root / "tactile_prediction_quality.json"
        metric_receipt_path = root / "reference_tactile_quality_receipt.json"
        _write_state(root, request_sha, "evaluate_reference_tactile", "running")
        reused_metric = metric_root.exists()
        if reused_metric:
            report = read_json_object(metric_report_path)
            verify_reference_metric_receipt(
                metric_receipt_path,
                metric_root=metric_root,
                report_path=metric_report_path,
                artifact=artifact,
                request_sha256=request_sha,
                input_identity=input_identity,
                evaluation_view_id=view.view_id,
                evaluation_view_sha256=view.view_sha256,
                golden_calibration_path=golden_calibration_path,
            )
        else:
            metric_evaluator(
                prediction_artifact=artifact_path,
                evaluation_view_path=request.evaluation_view_manifest,
                reference_metadata_path=request.reference_metadata,
                raw_root=request.raw_root,
                manifest_path=request.manifest,
                conversion_report_path=request.conversion_report,
                metric_script=request.official_metric_script,
                output=metric_root,
            )
            report = read_json_object(metric_report_path)
        validate_reference_metric_report(report, artifact=artifact, view=view)
        verify_materialized_target10_reference(
            metric_root=metric_root,
            metric_script=request.official_metric_script,
            report=report,
        )
        if not reused_metric:
            write_reference_metric_receipt(
                metric_receipt_path,
                metric_root=metric_root,
                report_path=metric_report_path,
                artifact=artifact,
                request_sha256=request_sha,
                input_identity=input_identity,
                evaluation_view_id=view.view_id,
                evaluation_view_sha256=view.view_sha256,
                golden_calibration_path=golden_calibration_path,
            )
        metric = report["metric"]
        assert isinstance(metric, Mapping)
        overall = metric["overall"]
        assert isinstance(overall, Mapping)
        n0_psnr = float(overall["average_psnr"])
        n0_ssim = float(overall["average_ssim"])
        summary: dict[str, object] = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "status": "complete",
            "request_sha256": request_sha,
            "protocol": REFERENCE_CONTRACT.contract_id,
            "contract_sha256": REFERENCE_CONTRACT.sha256,
            "calibration_policy": request.calibration_policy,
            "calibration_status": golden["status"],
            "published_score_comparable": published_score_comparable,
            "leaderboard_compatible": False,
            "organizer_contract_confirmed": False,
            "sequence_length_status": REFERENCE_CONTRACT.sequence_length_status,
            "score": {
                "average_psnr": n0_psnr,
                "average_ssim": n0_ssim,
                "display_psnr": format_published_score(n0_psnr, places=2),
                "display_ssim": format_published_score(n0_ssim, places=3),
            },
            "golden_calibration": str(root / "golden_calibration.json"),
            "golden_calibration_sha256": sha256_file(golden_calibration_path),
            "prediction_artifact": str(artifact.root),
            "metric_report": str(metric_report_path),
            "metric_receipt": str(metric_receipt_path),
            "metric_receipt_sha256": sha256_file(metric_receipt_path),
            "reused_prediction_artifact": reused_artifact,
            "reused_metric_report": reused_metric,
        }
        reference_comparison = {
            "display_psnr": REFERENCE_CONTRACT.published_psnr_display,
            "display_ssim": REFERENCE_CONTRACT.published_ssim_display,
            "delta_psnr": n0_psnr - float(REFERENCE_CONTRACT.published_psnr_display),
            "delta_ssim": n0_ssim - float(REFERENCE_CONTRACT.published_ssim_display),
        }
        if published_score_comparable:
            summary["published_reference"] = reference_comparison
        else:
            summary["reported_reference_numeric_only"] = {
                **reference_comparison,
                "comparison_status": "not_published_golden_calibrated",
            }
        write_json_atomic(root / "evaluation_receipt.json", summary)
        _write_state(root, request_sha, "complete", "complete")
        return summary
    except Exception:
        if request_bound:
            _write_state(root, request_sha, "failed", "failed")
        raise
    finally:
        lock.release()


__all__ = (
    "REQUEST_SCHEMA_VERSION",
    "Target10ReferenceEvaluationRequest",
    "load_target10_reference_request",
    "run_target10_reference_evaluation",
)
