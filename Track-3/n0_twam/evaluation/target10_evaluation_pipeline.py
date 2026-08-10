# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Resumable one-command orchestration for the frozen Target-10 protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol

from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
    validate_recorded_transformer_identity,
    validate_transformer_identity_match,
)
from n0_twam.evaluation.raw_tactile_quality import evaluate_raw_tactile_quality
from n0_twam.evaluation.target10_evaluation_receipts import (
    _EvaluationLock,
    _acquire_lock as _acquire_receipt_lock,
    _metric_receipt_matches,
    _write_metric_receipt,
)
from n0_twam.evaluation.sealed_artifact_io import (
    canonical_json,
    read_json_object,
    sha256_file,
    validate_sha256,
)
from n0_twam.evaluation.tactile_prediction_artifact import (
    VerifiedTactilePredictionArtifact,
    verify_tactile_prediction_artifact,
)
from n0_twam.evaluation.tactile_provenance import (
    audit_vae_decoder,
    write_json_atomic,
)
from n0_twam.evaluation.target10_view_contract import load_canonical_target10_view
from n0_twam.integrations.univtac.dataset_view import DatasetView
from n0_twam.integrations.univtac.artifact_validation import (
    verify_conversion_report,
    verify_manifest_payload,
)

REQUEST_SCHEMA_VERSION = 1
PIPELINE_SCHEMA_VERSION = 1
_REQUEST_REQUIRED_FIELDS = {
    "schema_version",
    "checkpoint",
    "checkpoint_sha256",
    "vae",
    "evaluation_view_manifest",
    "raw_root",
    "manifest",
    "conversion_report",
    "official_metric_script",
    "official_metric_sha256",
    "output_root",
}
_REQUEST_OPTIONAL_FIELDS = {
    "config_name",
    "device",
    "fps",
    "hip_visible_devices",
    "n_steps",
    "seed",
}


class _MetricEvaluator(Protocol):
    """Callable signature used to keep orchestration unit-testable."""

    def __call__(
        self,
        *,
        prediction_artifact: Path,
        evaluation_view_path: Path,
        raw_root: Path,
        manifest_path: Path,
        conversion_report_path: Path,
        output: Path,
        official_metric_script: Path,
        expected_official_metric_sha256: str,
        fps: int = 10,
    ) -> dict[str, object]: ...


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: object, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _path_field(payload: Mapping[str, object], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"evaluation request {key} must be a non-empty path string")
    return Path(value).expanduser()


def _string_field(
    payload: Mapping[str, object], key: str, *, default: str | None = None
) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"evaluation request {key} must be a non-empty string")
    return value


@dataclass(frozen=True)
class Target10EvaluationRequest:
    """All explicit, reproducibility-relevant inputs for one Target-10 run."""

    checkpoint: Path
    checkpoint_sha256: str
    vae: Path
    evaluation_view_manifest: Path
    raw_root: Path
    manifest: Path
    conversion_report: Path
    official_metric_script: Path
    official_metric_sha256: str
    output_root: Path
    config_name: str = "track31_univtac"
    device: str = "cuda:0"
    fps: int = 10
    hip_visible_devices: str | None = None
    n_steps: int = 8
    seed: int = 20260801

    @classmethod
    def from_json_dict(
        cls, payload: Mapping[str, object]
    ) -> "Target10EvaluationRequest":
        unexpected = set(payload) - (
            _REQUEST_REQUIRED_FIELDS | _REQUEST_OPTIONAL_FIELDS
        )
        missing = _REQUEST_REQUIRED_FIELDS - set(payload)
        if unexpected or missing:
            details = []
            if missing:
                details.append("missing=" + ",".join(sorted(missing)))
            if unexpected:
                details.append("unexpected=" + ",".join(sorted(unexpected)))
            raise ValueError("invalid evaluation request fields: " + "; ".join(details))
        if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
            raise ValueError("unsupported evaluation request schema_version")
        hip_visible_devices = payload.get("hip_visible_devices")
        if hip_visible_devices is not None and (
            not isinstance(hip_visible_devices, str) or not hip_visible_devices
        ):
            raise ValueError("hip_visible_devices must be a non-empty string or null")
        return cls(
            checkpoint=_path_field(payload, "checkpoint"),
            checkpoint_sha256=validate_sha256(
                payload.get("checkpoint_sha256"), label="checkpoint_sha256"
            ),
            vae=_path_field(payload, "vae"),
            evaluation_view_manifest=_path_field(payload, "evaluation_view_manifest"),
            raw_root=_path_field(payload, "raw_root"),
            manifest=_path_field(payload, "manifest"),
            conversion_report=_path_field(payload, "conversion_report"),
            official_metric_script=_path_field(payload, "official_metric_script"),
            official_metric_sha256=validate_sha256(
                payload.get("official_metric_sha256"),
                label="official_metric_sha256",
            ),
            output_root=_path_field(payload, "output_root"),
            config_name=_string_field(
                payload, "config_name", default="track31_univtac"
            ),
            device=_string_field(payload, "device", default="cuda:0"),
            fps=_positive_int(payload.get("fps", 10), label="fps"),
            hip_visible_devices=hip_visible_devices,
            n_steps=_positive_int(payload.get("n_steps", 8), label="n_steps"),
            seed=_nonnegative_int(payload.get("seed", 20260801), label="seed"),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "vae": str(self.vae),
            "evaluation_view_manifest": str(self.evaluation_view_manifest),
            "raw_root": str(self.raw_root),
            "manifest": str(self.manifest),
            "conversion_report": str(self.conversion_report),
            "official_metric_script": str(self.official_metric_script),
            "official_metric_sha256": self.official_metric_sha256,
            "output_root": str(self.output_root),
            "config_name": self.config_name,
            "device": self.device,
            "fps": self.fps,
            "hip_visible_devices": self.hip_visible_devices,
            "n_steps": self.n_steps,
            "seed": self.seed,
        }


def load_target10_evaluation_request(path: Path) -> Target10EvaluationRequest:
    """Read the immutable operator request used by the one-command CLI."""

    payload = read_json_object(Path(path).resolve(strict=True))
    return Target10EvaluationRequest.from_json_dict(payload)


def _resolved_input_path(path: Path, *, label: str, is_dir: bool) -> Path:
    resolved = Path(path).resolve(strict=True)
    if is_dir != resolved.is_dir():
        expected = "directory" if is_dir else "regular file"
        raise ValueError(f"{label} must be a {expected}: {resolved}")
    return resolved


def _checkpoint_root_and_train_meta(checkpoint: Path) -> tuple[Path, Path]:
    """Resolve a checkpoint root and its small sidecar without reading weights."""

    transformer_dir = checkpoint / "transformer"
    checkpoint_root = checkpoint if transformer_dir.is_dir() else checkpoint.parent
    train_meta = checkpoint_root / "train_meta.json"
    if not train_meta.is_file():
        raise FileNotFoundError(f"checkpoint train_meta does not exist: {train_meta}")
    return checkpoint_root, train_meta


def _preflight_input_identity(
    request: Target10EvaluationRequest, view: DatasetView
) -> dict[str, object]:
    """Bind model, decoder, manifest, conversion, and raw Target-10 bytes."""

    checkpoint_root, train_meta_path = _checkpoint_root_and_train_meta(
        request.checkpoint
    )
    train_meta = read_json_object(train_meta_path)
    recorded_transformer_identity = validate_recorded_transformer_identity(
        train_meta.get("transformer_identity"), expected_action_dim=8
    )
    actual_transformer_identity = audit_transformer_checkpoint(
        checkpoint_root / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
        expected_action_dim=8,
    )
    transformer_identity = validate_transformer_identity_match(
        recorded_transformer_identity,
        actual_transformer_identity,
        expected_action_dim=8,
        label="Target-10 checkpoint",
    )
    if transformer_identity["sha256"] != request.checkpoint_sha256:
        raise ValueError("checkpoint_sha256 differs from checkpoint train_meta")
    decoder = audit_vae_decoder(request.vae)
    decoder_identity_sha256 = hashlib.sha256(canonical_json(decoder)).hexdigest()
    manifest_payload = read_json_object(request.manifest)
    manifest_sha256, _ = verify_manifest_payload(manifest_payload)
    conversion_payload = read_json_object(request.conversion_report)
    _, conversion_report_sha256 = verify_conversion_report(
        conversion_payload,
        manifest_sha256=manifest_sha256,
    )
    raw_files = []
    for entry in view.entries:
        raw_path = (request.raw_root / entry.relative_path).resolve(strict=True)
        if not raw_path.is_file() or not raw_path.is_relative_to(request.raw_root):
            raise ValueError(
                "Target-10 raw HDF5 must remain a regular file under raw_root: "
                f"{entry.relative_path}"
            )
        actual_sha256 = sha256_file(raw_path)
        if actual_sha256 != entry.source_sha256:
            raise ValueError(
                "Target-10 raw HDF5 differs from the frozen view: "
                f"{entry.relative_path}"
            )
        raw_files.append(
            {
                "relative_path": entry.relative_path,
                "size_bytes": raw_path.stat().st_size,
                "sha256": actual_sha256,
            }
        )
    return {
        "evaluation_view": {
            "view_id": view.view_id,
            "view_sha256": view.view_sha256,
            "tasks": list(view.tasks),
            "per_task_counts": view.per_task_counts,
            "selection_method": view.selection_method,
        },
        "checkpoint_train_meta_sha256": sha256_file(train_meta_path),
        "checkpoint_transformer_identity": transformer_identity,
        "vae_decoder_identity_sha256": decoder_identity_sha256,
        "manifest_file_sha256": sha256_file(request.manifest),
        "manifest_payload_sha256": manifest_sha256,
        "conversion_report_file_sha256": sha256_file(request.conversion_report),
        "conversion_report_payload_sha256": conversion_report_sha256,
        "target10_raw_hdf5": raw_files,
    }


def _normalize_and_validate_request(
    request: Target10EvaluationRequest,
) -> tuple[Target10EvaluationRequest, DatasetView, dict[str, object]]:
    """Resolve all inputs before generation or raw HDF5 reads begin."""

    if request.device != "cuda:0":
        raise ValueError(
            "Target-10 generation supports only logical cuda:0; use "
            "hip_visible_devices to select a physical accelerator"
        )
    checkpoint = _resolved_input_path(
        request.checkpoint, label="checkpoint", is_dir=True
    )
    vae = _resolved_input_path(request.vae, label="VAE", is_dir=True)
    view_path = _resolved_input_path(
        request.evaluation_view_manifest,
        label="evaluation_view_manifest",
        is_dir=False,
    )
    raw_root = _resolved_input_path(request.raw_root, label="raw_root", is_dir=True)
    manifest = _resolved_input_path(request.manifest, label="manifest", is_dir=False)
    conversion = _resolved_input_path(
        request.conversion_report,
        label="conversion_report",
        is_dir=False,
    )
    metric_script = _resolved_input_path(
        request.official_metric_script,
        label="official_metric_script",
        is_dir=False,
    )
    if sha256_file(metric_script) != request.official_metric_sha256:
        raise ValueError("official_metric_script SHA256 does not match the request")
    view = load_canonical_target10_view(
        view_path=view_path,
        manifest_path=manifest,
    )
    output_root = request.output_root.expanduser().resolve()
    protected_directories = (checkpoint, vae, raw_root)
    if any(
        output_root == protected
        or output_root.is_relative_to(protected)
        or protected.is_relative_to(output_root)
        for protected in protected_directories
    ):
        raise ValueError("output_root must be disjoint from input artifact directories")
    resolved_request = Target10EvaluationRequest(
        checkpoint=checkpoint,
        checkpoint_sha256=request.checkpoint_sha256,
        vae=vae,
        evaluation_view_manifest=view_path,
        raw_root=raw_root,
        manifest=manifest,
        conversion_report=conversion,
        official_metric_script=metric_script,
        official_metric_sha256=request.official_metric_sha256,
        output_root=output_root,
        config_name=request.config_name,
        device=request.device,
        fps=request.fps,
        hip_visible_devices=request.hip_visible_devices,
        n_steps=request.n_steps,
        seed=request.seed,
    )
    return resolved_request, view, _preflight_input_identity(resolved_request, view)


def _request_sha256(
    request: Target10EvaluationRequest, input_identity: Mapping[str, object]
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "request": request.to_json_dict(),
                "input_identity": dict(input_identity),
            }
        )
    ).hexdigest()


def _write_state(
    root: Path,
    *,
    request_sha256: str,
    stage: str,
    status: str,
    detail: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "stage": stage,
        "status": status,
        "updated_at_utc": _now_utc(),
    }
    if detail is not None:
        payload["detail"] = detail
    write_json_atomic(root / "evaluation_state.json", payload)


def _ensure_request_receipt(
    root: Path,
    request: Target10EvaluationRequest,
    request_sha256: str,
    input_identity: Mapping[str, object],
) -> None:
    receipt_path = root / "evaluation_request.json"
    if receipt_path.exists():
        existing = read_json_object(receipt_path)
        if (
            existing.get("request_sha256") != request_sha256
            or existing.get("request") != request.to_json_dict()
            or existing.get("input_identity") != dict(input_identity)
        ):
            raise ValueError(
                "output_root already belongs to a different evaluation request"
            )
        return
    write_json_atomic(
        receipt_path,
        {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "request": request.to_json_dict(),
            "input_identity": dict(input_identity),
        },
    )


def _acquire_lock(root: Path) -> _EvaluationLock:
    """Acquire one persistent OS-level lock for this output root."""

    return _acquire_receipt_lock(root, schema_version=PIPELINE_SCHEMA_VERSION)


def _artifact_matches_request(
    artifact: VerifiedTactilePredictionArtifact,
    *,
    request: Target10EvaluationRequest,
    view: DatasetView,
    input_identity: Mapping[str, object],
) -> bool:
    metadata = artifact.metadata
    samples = metadata.get("samples")
    provenance = metadata.get("generation_provenance")
    seed_contract = (
        provenance.get("seed_contract") if isinstance(provenance, Mapping) else None
    )
    return (
        metadata.get("checkpoint_sha256") == request.checkpoint_sha256
        and metadata.get("evaluation_view_id") == view.view_id
        and metadata.get("evaluation_view_sha256") == view.view_sha256
        and metadata.get("decoder_identity_sha256")
        == input_identity.get("vae_decoder_identity_sha256")
        and metadata.get("conditioning_protocol_id") == "causal_future_only_v1"
        and isinstance(samples, list)
        and len(samples) == len(view.entries)
        and isinstance(provenance, Mapping)
        and provenance.get("config_name") == request.config_name
        and provenance.get("n_steps") == request.n_steps
        and isinstance(seed_contract, Mapping)
        and seed_contract.get("seed") == request.seed
    )


def _metric_matches_artifact(
    report_path: Path,
    *,
    artifact: VerifiedTactilePredictionArtifact,
    request: Target10EvaluationRequest,
    view: DatasetView,
    request_sha256: str,
    input_identity: Mapping[str, object],
    metric_receipt_path: Path,
) -> bool:
    try:
        receipt = read_json_object(metric_receipt_path)
    except ValueError:
        return False
    return _metric_report_matches_artifact(
        report_path,
        artifact=artifact,
        request=request,
        view=view,
    ) and _metric_receipt_matches(
        receipt,
        schema_version=PIPELINE_SCHEMA_VERSION,
        metric_root=report_path.parent,
        report_path=report_path,
        artifact=artifact,
        request_sha256=request_sha256,
        input_identity=input_identity,
        evaluation_view_id=view.view_id,
        evaluation_view_sha256=view.view_sha256,
        official_metric_script_sha256=request.official_metric_sha256,
    )


def _metric_report_matches_artifact(
    report_path: Path,
    *,
    artifact: VerifiedTactilePredictionArtifact,
    request: Target10EvaluationRequest,
    view: DatasetView,
) -> bool:
    """Verify the semantic bindings present in the Stage-B report itself."""

    try:
        report = read_json_object(report_path)
    except ValueError:
        return False
    artifact_report = report.get("prediction_artifact")
    evaluation_view = report.get("evaluation_view")
    official_script = report.get("official_script")
    return (
        isinstance(artifact_report, dict)
        and artifact_report.get("file_sha256") == artifact.file_sha256
        and isinstance(evaluation_view, dict)
        and evaluation_view.get("view_id") == view.view_id
        and evaluation_view.get("view_sha256") == view.view_sha256
        and isinstance(official_script, dict)
        and official_script.get("sha256") == request.official_metric_sha256
        and isinstance(report.get("official_metrics"), dict)
    )


def _run_stage_a(
    request: Target10EvaluationRequest, artifact_path: Path, log_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(
            repo_root
            / "script"
            / "track3_1"
            / "generate_tactile_prediction_artifact.py"
        ),
        "--config-name",
        request.config_name,
        "--ckpt",
        str(request.checkpoint),
        "--vae",
        str(request.vae),
        "--output",
        str(artifact_path),
        "--evaluation-view-manifest",
        str(request.evaluation_view_manifest),
        "--protocol",
        "causal_future_only_v1",
        "--n-steps",
        str(request.n_steps),
        "--seed",
        str(request.seed),
        "--device",
        request.device,
    ]
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(request.seed)
    if request.hip_visible_devices is not None:
        environment["HIP_VISIBLE_DEVICES"] = request.hip_visible_devices
        environment["CUDA_VISIBLE_DEVICES"] = request.hip_visible_devices
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"[{_now_utc()}] command={json.dumps(command)}\n")
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
            log_handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        message = "Stage A generation failed with exit code "
        raise RuntimeError(message + f"{return_code}; see {log_path}")


def _final_summary(
    *,
    request_sha256: str,
    artifact: VerifiedTactilePredictionArtifact,
    metric_report: Mapping[str, object],
    output_root: Path,
    reused_artifact: bool,
    reused_metrics: bool,
) -> dict[str, object]:
    metrics = metric_report.get("official_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("raw tactile metric report lacks official_metrics")
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "status": "complete",
        "request_sha256": request_sha256,
        "prediction_artifact": str(artifact.root),
        "prediction_artifact_file_sha256": artifact.file_sha256,
        "metric_report": str(
            output_root / "raw_tactile_quality" / "raw_tactile_quality.json"
        ),
        "average_psnr": metrics.get("average_psnr"),
        "average_ssim": metrics.get("average_ssim"),
        "reused_prediction_artifact": reused_artifact,
        "reused_metric_report": reused_metrics,
        "leaderboard_compatible": False,
    }


def run_target10_evaluation(
    request: Target10EvaluationRequest,
    *,
    metric_evaluator: _MetricEvaluator = evaluate_raw_tactile_quality,
) -> dict[str, object]:
    """Execute or safely resume Stage-A generation and Stage-B raw evaluation."""

    request, view, input_identity = _normalize_and_validate_request(request)
    request_sha256 = _request_sha256(request, input_identity)
    root = request.output_root
    root.mkdir(parents=True, exist_ok=True)
    evaluation_lock = _acquire_lock(root)
    artifact_path = root / "prediction_artifact"
    metric_root = root / "raw_tactile_quality"
    metric_report_path = metric_root / "raw_tactile_quality.json"
    metric_receipt_path = root / "raw_tactile_quality_receipt.json"
    request_bound = False
    try:
        _ensure_request_receipt(root, request, request_sha256, input_identity)
        request_bound = True
        _write_state(
            root,
            request_sha256=request_sha256,
            stage="preflight",
            status="running",
        )
        reused_artifact = artifact_path.exists()
        if reused_artifact:
            artifact = verify_tactile_prediction_artifact(artifact_path)
            if not _artifact_matches_request(
                artifact,
                request=request,
                view=view,
                input_identity=input_identity,
            ):
                raise ValueError("existing prediction artifact disagrees with request")
        else:
            _write_state(
                root,
                request_sha256=request_sha256,
                stage="generate_prediction_artifact",
                status="running",
            )
            _run_stage_a(request, artifact_path, root / "logs" / "stage_a.log")
            artifact = verify_tactile_prediction_artifact(artifact_path)
            if not _artifact_matches_request(
                artifact,
                request=request,
                view=view,
                input_identity=input_identity,
            ):
                raise RuntimeError(
                    "generated prediction artifact disagrees with request"
                )

        reused_metrics = metric_root.exists()
        if reused_metrics:
            if not metric_receipt_path.exists():
                raise ValueError(
                    "existing raw metric output has no sealed receipt; use a new "
                    "output_root and rerun Stage B"
                )
            if not _metric_report_matches_artifact(
                metric_report_path,
                artifact=artifact,
                request=request,
                view=view,
            ):
                raise ValueError("existing raw metric report disagrees with request")
            if not _metric_matches_artifact(
                metric_report_path,
                artifact=artifact,
                request=request,
                view=view,
                request_sha256=request_sha256,
                input_identity=input_identity,
                metric_receipt_path=metric_receipt_path,
            ):
                raise ValueError("existing raw metric report disagrees with request")
            metric_report = read_json_object(metric_report_path)
        else:
            _write_state(
                root,
                request_sha256=request_sha256,
                stage="evaluate_raw_tactile_quality",
                status="running",
            )
            metric_report = metric_evaluator(
                prediction_artifact=artifact_path,
                evaluation_view_path=request.evaluation_view_manifest,
                raw_root=request.raw_root,
                manifest_path=request.manifest,
                conversion_report_path=request.conversion_report,
                official_metric_script=request.official_metric_script,
                expected_official_metric_sha256=request.official_metric_sha256,
                output=metric_root,
                fps=request.fps,
            )
            if not _metric_report_matches_artifact(
                metric_report_path,
                artifact=artifact,
                request=request,
                view=view,
            ):
                raise RuntimeError("generated raw metric report disagrees with request")
            metric_report = read_json_object(metric_report_path)
            _write_metric_receipt(
                metric_receipt_path,
                schema_version=PIPELINE_SCHEMA_VERSION,
                metric_root=metric_root,
                report_path=metric_report_path,
                artifact=artifact,
                request_sha256=request_sha256,
                input_identity=input_identity,
                evaluation_view_id=view.view_id,
                evaluation_view_sha256=view.view_sha256,
                official_metric_script_sha256=request.official_metric_sha256,
            )
        summary = _final_summary(
            request_sha256=request_sha256,
            artifact=artifact,
            metric_report=metric_report,
            output_root=root,
            reused_artifact=reused_artifact,
            reused_metrics=reused_metrics,
        )
        write_json_atomic(root / "evaluation_receipt.json", summary)
        _write_state(
            root,
            request_sha256=request_sha256,
            stage="complete",
            status="complete",
        )
        return summary
    except Exception as exc:
        if request_bound:
            _write_state(
                root,
                request_sha256=request_sha256,
                stage="failed",
                status="failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
        raise
    finally:
        if sys.exc_info()[0] is None:
            evaluation_lock.release()
        else:
            try:
                evaluation_lock.release()
            except OSError:
                pass
