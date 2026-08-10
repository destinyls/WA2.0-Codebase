# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Input normalization for calibrated Target-10 evaluation."""

from __future__ import annotations

import hashlib
from typing import Mapping

from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.evaluation.target10_evaluation_pipeline import (
    Target10EvaluationRequest,
    _checkpoint_root_and_train_meta,
    _normalize_and_validate_request,
)
from n0_twam.evaluation.target10_reference_contract import (
    REFERENCE_CONTRACT,
    REFERENCE_METRIC_SHA256,
    load_target10_reference_roster,
    validate_reference_metric,
    validate_view_against_reference_roster,
)
from n0_twam.evaluation.target10_reference_request import (
    Target10ReferenceEvaluationRequest,
)
from n0_twam.evaluation.sealed_artifact_io import (
    read_json_object,
    sha256_file,
    validate_sha256,
)
from n0_twam.evaluation.tactile_provenance import audit_evaluation_dataset
from n0_twam.integrations.univtac.artifact_validation import (
    verify_normalizer_payload,
)
from n0_twam.integrations.univtac.dataset_view import DatasetView


def normalize_reference_request(
    request: Target10ReferenceEvaluationRequest,
) -> tuple[Target10ReferenceEvaluationRequest, DatasetView, dict[str, object]]:
    """Resolve all inputs and bind both independent Target-10 rosters."""

    metric_script = validate_reference_metric(request.official_metric_script)
    legacy = Target10EvaluationRequest(
        checkpoint=request.checkpoint,
        checkpoint_sha256=request.checkpoint_sha256,
        vae=request.vae,
        evaluation_view_manifest=request.evaluation_view_manifest,
        raw_root=request.raw_root,
        manifest=request.manifest,
        conversion_report=request.conversion_report,
        official_metric_script=metric_script,
        official_metric_sha256=REFERENCE_METRIC_SHA256,
        output_root=request.output_root,
        config_name=request.config_name,
        device=request.device,
        fps=REFERENCE_CONTRACT.fps,
        hip_visible_devices=request.hip_visible_devices,
        n_steps=request.n_steps,
        seed=request.seed,
    )
    normalized, view, identity = _normalize_and_validate_request(legacy)
    metadata = request.reference_metadata.resolve(strict=True)
    validate_view_against_reference_roster(
        view, load_target10_reference_roster(metadata)
    )
    raw_golden_root = request.golden_root.expanduser()
    if raw_golden_root.is_symlink():
        raise ValueError("golden_root must not be a symlink")
    golden_root = raw_golden_root.resolve(strict=True)
    golden_manifest = request.golden_manifest.resolve(strict=True)
    if not golden_root.is_dir() or not golden_manifest.is_file():
        raise ValueError("golden calibration inputs are invalid")
    base_model = request.base_model.resolve(strict=True)
    raw_empty_embedding = request.empty_embedding.expanduser()
    if raw_empty_embedding.is_symlink():
        raise ValueError("empty_embedding must not be a symlink")
    empty_embedding = raw_empty_embedding.resolve(strict=True)
    lerobot_root = request.lerobot_root.resolve(strict=True)
    normalizer = request.normalizer.resolve(strict=True)
    normalizer_source_view = request.normalizer_source_view.resolve(strict=True)
    if (
        not base_model.is_dir()
        or not empty_embedding.is_file()
        or not lerobot_root.is_dir()
        or not (lerobot_root / view.physical_split).is_dir()
        or not normalizer.is_file()
        or not normalizer_source_view.is_file()
    ):
        raise ValueError("explicit causal dataset/model inputs are invalid")
    empty_embedding_sha256 = sha256_file(empty_embedding)
    if empty_embedding_sha256 != request.empty_embedding_sha256:
        raise ValueError("empty embedding SHA256 differs from the request")
    checkpoint_root, _ = _checkpoint_root_and_train_meta(normalized.checkpoint)
    checkpoint_snapshot = capture_strict_checkpoint_snapshot(checkpoint_root)
    strict_checkpoint_identity = build_strict_checkpoint_identity(checkpoint_snapshot)
    runtime_source_identity = checkpoint_snapshot.runtime_source_identity
    if not isinstance(runtime_source_identity, Mapping):
        raise ValueError("checkpoint runtime_source_identity is missing")
    checkpoint_empty_embedding_sha256 = validate_sha256(
        runtime_source_identity.get("empty_embedding_sha256"),
        label="checkpoint empty embedding SHA256",
    )
    if checkpoint_empty_embedding_sha256 != empty_embedding_sha256:
        raise ValueError("empty embedding differs from the checkpoint training input")
    actual_normalizer_sha, _ = verify_normalizer_payload(
        read_json_object(normalizer),
        manifest_sha256=str(identity["manifest_payload_sha256"]),
    )
    if actual_normalizer_sha != request.normalizer_sha256:
        raise ValueError("normalizer SHA256 differs from the request")
    causal_dataset_provenance = {
        **audit_evaluation_dataset(
            dataset_path=(lerobot_root / view.physical_split),
            manifest_path=normalized.manifest,
            conversion_report_path=normalized.conversion_report,
            normalizer_path=normalizer,
            evaluation_view_path=normalized.evaluation_view_manifest,
            normalizer_source_view_path=normalizer_source_view,
            expected_manifest_sha256=str(identity["manifest_payload_sha256"]),
            expected_normalizer_sha256=actual_normalizer_sha,
            base_model_path=base_model,
        ),
        "empty_embedding": {
            "path": str(empty_embedding),
            "size_bytes": empty_embedding.stat().st_size,
            "file_sha256": empty_embedding_sha256,
        },
    }
    output_root = normalized.output_root
    protected_directories = {
        normalized.checkpoint,
        normalized.vae,
        normalized.raw_root,
        base_model,
        lerobot_root,
        golden_root,
        golden_manifest.parent,
        metadata.parent,
        metric_script.parent,
    }
    protected_files = {
        normalized.evaluation_view_manifest,
        normalized.manifest,
        normalized.conversion_report,
        golden_manifest,
        metadata,
        metric_script,
        normalizer,
        normalizer_source_view,
        empty_embedding,
    }
    if any(
        output_root == path
        or output_root.is_relative_to(path)
        or path.is_relative_to(output_root)
        for path in protected_directories
    ) or any(
        output_root == path or path.is_relative_to(output_root)
        for path in protected_files
    ):
        raise ValueError("output_root must be disjoint from every reference input")
    resolved = Target10ReferenceEvaluationRequest(
        checkpoint=normalized.checkpoint,
        checkpoint_sha256=normalized.checkpoint_sha256,
        vae=normalized.vae,
        base_model=base_model,
        empty_embedding=empty_embedding,
        empty_embedding_sha256=empty_embedding_sha256,
        lerobot_root=lerobot_root,
        normalizer=normalizer,
        normalizer_sha256=actual_normalizer_sha,
        normalizer_source_view=normalizer_source_view,
        evaluation_view_manifest=normalized.evaluation_view_manifest,
        reference_metadata=metadata,
        raw_root=normalized.raw_root,
        manifest=normalized.manifest,
        conversion_report=normalized.conversion_report,
        official_metric_script=metric_script,
        golden_root=golden_root,
        golden_manifest=golden_manifest,
        golden_manifest_sha256=request.golden_manifest_sha256,
        calibration_policy=request.calibration_policy,
        output_root=output_root,
        config_name=normalized.config_name,
        device=normalized.device,
        hip_visible_devices=normalized.hip_visible_devices,
        n_steps=normalized.n_steps,
        seed=normalized.seed,
    )
    return (
        resolved,
        view,
        {
            **identity,
            "reference_contract_sha256": REFERENCE_CONTRACT.sha256,
            "reference_metadata_sha256": hashlib.sha256(
                metadata.read_bytes()
            ).hexdigest(),
            "reference_metric_sha256": REFERENCE_METRIC_SHA256,
            "golden_manifest_sha256": request.golden_manifest_sha256,
            "strict_checkpoint_identity": strict_checkpoint_identity,
            "normalizer_sha256": actual_normalizer_sha,
            "normalizer_file_sha256": hashlib.sha256(
                normalizer.read_bytes()
            ).hexdigest(),
            "normalizer_source_view_file_sha256": hashlib.sha256(
                normalizer_source_view.read_bytes()
            ).hexdigest(),
            "base_model_path": str(base_model),
            "empty_embedding": {
                "path": str(empty_embedding),
                "size_bytes": empty_embedding.stat().st_size,
                "file_sha256": empty_embedding_sha256,
                "checkpoint_file_sha256": checkpoint_empty_embedding_sha256,
            },
            "lerobot_root": str(lerobot_root),
            "causal_dataset_provenance": causal_dataset_provenance,
        },
    )


__all__ = ("normalize_reference_request",)
