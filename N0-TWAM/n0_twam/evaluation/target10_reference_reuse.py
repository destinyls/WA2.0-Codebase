# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Semantic verification for reusable Target-10 reference reports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from n0_twam.evaluation.target10_prediction_artifact_v3 import (
    VerifiedTarget10PredictionArtifact,
)
from n0_twam.evaluation.target10_reference_contract import (
    REFERENCE_CONTRACT,
    REFERENCE_METRIC_SHA256,
    TARGET_EPISODE_IDS,
    TARGET_TASKS,
)
from n0_twam.evaluation.target10_reference_metric import (
    ReferenceVideoPair,
    evaluate_reference_video_pairs,
)
from n0_twam.evaluation.target10_reference_request import (
    Target10ReferenceEvaluationRequest,
)
from n0_twam.integrations.univtac.dataset_view import DatasetView


def validate_reference_metric_report(
    report: Mapping[str, object],
    *,
    artifact: VerifiedTarget10PredictionArtifact,
    view: DatasetView,
) -> None:
    """Validate the report's immutable identities and finite summary."""

    artifact_row = report.get("prediction_artifact")
    view_row = report.get("evaluation_view")
    metric = report.get("metric")
    overall = metric.get("overall") if isinstance(metric, Mapping) else None
    values = (
        overall.get("average_psnr") if isinstance(overall, Mapping) else None,
        overall.get("average_ssim") if isinstance(overall, Mapping) else None,
    )
    if (
        report.get("protocol") != REFERENCE_CONTRACT.contract_id
        or report.get("contract_sha256") != REFERENCE_CONTRACT.sha256
        or not isinstance(artifact_row, Mapping)
        or artifact_row.get("seal_sha256") != artifact.seal_sha256
        or not isinstance(view_row, Mapping)
        or view_row.get("view_id") != view.view_id
        or view_row.get("view_sha256") != view.view_sha256
        or not isinstance(metric, Mapping)
        or metric.get("metric_script_sha256") != REFERENCE_METRIC_SHA256
        or metric.get("counts") != {"episodes": 10, "future_frame_pairs": 80}
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        )
    ):
        raise ValueError("reference metric report disagrees with the request")


def verify_materialized_target10_reference(
    *,
    metric_root: Path,
    metric_script: Path,
    report: Mapping[str, object],
) -> None:
    """Recompute both score trees from MP4 bytes before report reuse."""

    root = Path(metric_root).resolve(strict=True)
    pairs: list[ReferenceVideoPair] = []
    persistence_pairs: list[ReferenceVideoPair] = []
    for task in TARGET_TASKS:
        for sample_index, episode_id in enumerate(TARGET_EPISODE_IDS):
            sample_id = f"sample_{sample_index:03d}"
            ground_truth = root / "artifacts" / task / f"{sample_id}_gt_tactile.mp4"
            pairs.append(
                ReferenceVideoPair(
                    task=task,
                    sample_id=sample_id,
                    sample_index=sample_index,
                    episode_id=episode_id,
                    prediction_path=(
                        root / "artifacts" / task / f"{sample_id}_pred_tactile.mp4"
                    ),
                    ground_truth_path=ground_truth,
                )
            )
            persistence_pairs.append(
                ReferenceVideoPair(
                    task=task,
                    sample_id=sample_id,
                    sample_index=sample_index,
                    episode_id=episode_id,
                    prediction_path=(
                        root / "persistence" / task / f"{sample_id}_pred_tactile.mp4"
                    ),
                    ground_truth_path=ground_truth,
                )
            )
    metric = evaluate_reference_video_pairs(pairs=pairs, metric_script=metric_script)
    persistence = evaluate_reference_video_pairs(
        pairs=persistence_pairs,
        metric_script=metric_script,
    )
    if (
        report.get("metric") != metric
        or report.get("persistence_baseline") != persistence
    ):
        raise ValueError("reference report differs from recomputed MP4 metrics")


def artifact_matches_reference_request(
    artifact: VerifiedTarget10PredictionArtifact,
    *,
    request: Target10ReferenceEvaluationRequest,
    view: DatasetView,
    input_identity: Mapping[str, object],
    causal_seal_sha256: str,
) -> bool:
    """Bind a reusable prediction to model, data, view, seed, and causal input."""

    provenance = artifact.metadata.get("generation_provenance")
    if not isinstance(provenance, Mapping):
        return False
    seed_contract = provenance.get("seed_contract")
    causal = provenance.get("conditioning_bundle")
    dataset = provenance.get("dataset_provenance")
    expected_dataset = input_identity.get("causal_dataset_provenance")
    strict_checkpoint = provenance.get("strict_checkpoint_identity")
    expected_strict_checkpoint = input_identity.get("strict_checkpoint_identity")
    checkpoint_binding = provenance.get("checkpoint_dataset_binding")
    return (
        artifact.metadata.get("checkpoint_sha256") == request.checkpoint_sha256
        and artifact.metadata.get("source_manifest_sha256")
        == input_identity.get("manifest_payload_sha256")
        and artifact.metadata.get("conversion_report_sha256")
        == input_identity.get("conversion_report_payload_sha256")
        and artifact.metadata.get("evaluation_view_id") == view.view_id
        and artifact.metadata.get("evaluation_view_sha256") == view.view_sha256
        and artifact.metadata.get("decoder_identity_sha256")
        == input_identity.get("vae_decoder_identity_sha256")
        and provenance.get("config_name") == request.config_name
        and provenance.get("n_steps") == request.n_steps
        and isinstance(seed_contract, Mapping)
        and seed_contract.get("seed") == request.seed
        and isinstance(causal, Mapping)
        and causal.get("seal_sha256") == causal_seal_sha256
        and isinstance(dataset, Mapping)
        and isinstance(expected_dataset, Mapping)
        and dict(dataset) == dict(expected_dataset)
        and isinstance(strict_checkpoint, Mapping)
        and isinstance(expected_strict_checkpoint, Mapping)
        and dict(strict_checkpoint) == dict(expected_strict_checkpoint)
        and provenance.get("empty_embedding") == dataset.get("empty_embedding")
        and dataset.get("evaluation_view_id") == view.view_id
        and dataset.get("evaluation_view_sha256") == view.view_sha256
        and isinstance(checkpoint_binding, Mapping)
        and checkpoint_binding.get("manifest_sha256")
        == dataset.get("source_manifest_sha256")
        and checkpoint_binding.get("normalizer_sha256")
        == dataset.get("normalizer_sha256")
        and checkpoint_binding.get("conversion_report_sha256")
        == dataset.get("conversion_report_sha256")
    )


__all__ = (
    "artifact_matches_reference_request",
    "validate_reference_metric_report",
    "verify_materialized_target10_reference",
)
