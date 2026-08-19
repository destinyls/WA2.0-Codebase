# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict completion gate for an AgileX training invocation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
    validate_sha256,
    validate_transformer_identity_match,
)
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.checkpointing.training_lineage import (
    load_validated_action_migration_report_for_contract,
    validate_action_migration_report_for_contract,
)
from n0_twam.configs.twam_track3_agilex_recipe import (
    build_agilex_resume_recipe_contract,
    validate_agilex_resume_recipe_contract,
)
from n0_twam.distributed.optimizer_checkpoint import (
    OPTIMIZER_DCP_DIRNAME,
    build_training_execution_contract,
    validate_optimizer_checkpoint,
    validate_training_execution_contract,
)
from n0_twam.embodiments import AGILEX_ACTION_SCHEMA
from n0_twam.track31.local_provenance import LocalProvenance

from .preflight import (
    agilex_profile_id,
    build_artifact_identity,
    validate_qpos14_stage_b_checkpoint,
)
from .request import AgileXTrainRequest


def _expected_execution_contract(
    request: AgileXTrainRequest,
) -> dict[str, object]:
    hcu = request.runtime.accelerator_profile == "hcu_performance"
    return cast(
        dict[str, object],
        build_training_execution_contract(
            max_latent_frames=request.train.max_latent_frames,
            gradient_accumulation_steps=request.train.gradient_accumulation_steps,
            batch_size=request.train.batch_size,
            load_worker=0,
            num_steps=request.train.num_steps,
            lr_schedule="cosine",
            warmup_steps=20,
            lr_min_ratio=0.1,
            activation_checkpointing=not hcu,
            attention_contract={
                "attention_backend": ("grouped_flash_attn" if hcu else "grouped_sdpa"),
                "grouped_sdpa_max_query_tokens": 16384,
                "mot_cross_attention_backend": "flash_attn" if hcu else "sdpa",
            },
        ),
    )


def _verify_physical_checkpoint(
    request: AgileXTrainRequest,
    *,
    checkpoint: Path,
    train_meta: Mapping[str, object],
    training_state: Mapping[str, object],
    completion: Mapping[str, object],
    snapshot_gradient_accumulation_steps: int,
) -> None:
    """Bind completion sidecars to the actual model and optimizer bytes."""

    actual_transformer = audit_transformer_checkpoint(
        checkpoint / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
        expected_action_dim=14,
    )
    for label, payload in (
        ("train metadata", train_meta),
        ("training state", training_state),
        ("completion marker", completion),
    ):
        validate_transformer_identity_match(
            payload.get("transformer_identity"),
            actual_transformer,
            expected_action_dim=14,
            label=f"completed AgileX {label}",
        )

    optimizer_sha256 = validate_sha256(
        training_state.get("optimizer_inventory_sha256"),
        label="completed AgileX optimizer inventory SHA256",
    )
    if completion.get("optimizer_inventory_sha256") != optimizer_sha256:
        raise ValueError("completed AgileX optimizer identity differs across sidecars")
    validate_optimizer_checkpoint(
        checkpoint / OPTIMIZER_DCP_DIRNAME,
        expected_inventory_sha256=optimizer_sha256,
    )

    requested_accumulation = request.train.gradient_accumulation_steps
    if snapshot_gradient_accumulation_steps != requested_accumulation:
        raise ValueError("completed AgileX gradient accumulation differs from request")
    saved_execution = training_state.get("training_execution_contract")
    if (
        train_meta.get("training_execution_contract") != saved_execution
        or completion.get("training_execution_contract") != saved_execution
    ):
        raise ValueError(
            "completed AgileX training execution contract differs across sidecars"
        )
    validate_training_execution_contract(
        saved_execution,
        current_contract=_expected_execution_contract(request),
    )


def _validate_completed_migration_lineage(
    request: AgileXTrainRequest,
    migration: Mapping[str, object],
) -> None:
    """Bind init20 directly, or Stage-B to the qpos14 parent's report."""

    paths = request.paths
    if paths.init_from is None:
        return
    if migration.get("source_checkpoint_sha256") == paths.init_transformer_sha256:
        return
    validate_qpos14_stage_b_checkpoint(
        paths.init_from,
        expected_transformer_sha256=paths.init_transformer_sha256,
        expected_profile_id=agilex_profile_id(request.profile),
        expected_run_role=request.train.run_role,
        expected_artifact_identity=build_artifact_identity(request),
    )
    parent_migration = load_validated_action_migration_report_for_contract(
        paths.init_from,
        source_action_dim=20,
        source_action_schema="ee20_pi05",
        target_action_dim=14,
        target_action_schema=AGILEX_ACTION_SCHEMA,
        initialized_target_only_prefixes=("local_tactile_", "agilex_wrench_"),
    )
    if parent_migration != migration:
        raise ValueError(
            "completed AgileX migration lineage differs from Stage-B parent"
        )


def verify_completed_checkpoint(
    request: AgileXTrainRequest,
    provenance: LocalProvenance,
) -> tuple[Path, dict[str, object]]:
    """Require an exact qpos14/profile/artifact/lineage/provenance checkpoint."""

    checkpoint = (
        request.paths.output_root
        / "checkpoints"
        / f"checkpoint_step_{request.train.stop_after_step}"
    )
    snapshot = capture_strict_checkpoint_snapshot(checkpoint)
    if snapshot.step != request.train.stop_after_step:
        raise ValueError("completed AgileX checkpoint step differs from request")
    if snapshot.world_size != len(request.runtime.devices):
        raise ValueError("completed AgileX checkpoint world_size differs from request")
    if snapshot.transformer_config.get("action_schema") != AGILEX_ACTION_SCHEMA:
        raise ValueError("completed AgileX checkpoint is not qpos14")
    if snapshot.transformer_config.get("action_dim") != 14:
        raise ValueError("completed AgileX transformer action_dim is not 14")
    _verify_physical_checkpoint(
        request,
        checkpoint=snapshot.checkpoint_root,
        train_meta=snapshot.train_meta,
        training_state=snapshot.training_state,
        completion=snapshot.completion,
        snapshot_gradient_accumulation_steps=(snapshot.gradient_accumulation_steps),
    )
    expected_artifacts = build_artifact_identity(request)
    expected_profile = agilex_profile_id(request.profile)
    payloads = (snapshot.train_meta, snapshot.training_state, snapshot.completion)
    for payload in payloads:
        if payload.get("track32_artifact_identity") != expected_artifacts:
            raise ValueError("completed AgileX artifact identity differs")
    meta = snapshot.train_meta
    if (
        meta.get("track32_profile_id") != expected_profile
        or meta.get("action_schema") != AGILEX_ACTION_SCHEMA
        or meta.get("action_dim") != 14
        or meta.get("tactile_profile") != request.profile
        or meta.get("run_role") != request.train.run_role
        or meta.get("accelerator_profile") != request.runtime.accelerator_profile
    ):
        raise ValueError("completed AgileX checkpoint metadata differs from request")
    lineage = meta.get("training_lineage")
    if not isinstance(lineage, Mapping) or any(
        payload.get("training_lineage") != lineage for payload in payloads[1:]
    ):
        raise ValueError("completed AgileX training lineage is inconsistent")
    expected_lineage = {
        "repo_route_manifest_source_file_sha256": (
            request.paths.repo_route_manifest_sha256
        ),
        "temporal_alignment_source_file_sha256": (
            request.paths.temporal_alignment_sha256
        ),
        "normalizer_sha256": request.paths.normalizer_sha256,
        "action_schema": AGILEX_ACTION_SCHEMA,
        "tactile_profile": request.profile,
        "resume_recipe_contract": build_agilex_resume_recipe_contract(request.train),
    }
    if any(lineage.get(key) != value for key, value in expected_lineage.items()):
        raise ValueError("completed AgileX training lineage differs from request")
    validate_agilex_resume_recipe_contract(
        lineage.get("resume_recipe_contract"),
        current=request.train,
    )
    if (
        snapshot.runtime_source_identity != provenance.runtime_source_identity
        or snapshot.checkpoint_invocation_identity
        != provenance.checkpoint_invocation_identity
    ):
        raise ValueError("completed AgileX runtime provenance differs from launch")
    if snapshot.action_migration_report is None:
        raise ValueError("completed AgileX checkpoint lost action migration lineage")
    migration = validate_action_migration_report_for_contract(
        snapshot.action_migration_report,
        source_action_dim=20,
        source_action_schema="ee20_pi05",
        target_action_dim=14,
        target_action_schema=AGILEX_ACTION_SCHEMA,
        initialized_target_only_prefixes=("local_tactile_", "agilex_wrench_"),
    )
    _validate_completed_migration_lineage(request, migration)
    return checkpoint.resolve(strict=True), build_strict_checkpoint_identity(snapshot)


__all__ = ("verify_completed_checkpoint",)
