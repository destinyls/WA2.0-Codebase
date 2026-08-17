# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Checkpoint lineage validation for optional Track 3.1 Stage B training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from n0_twam.configs.twam_track31_training_profiles import (
    MULTITASK_PRETRAIN_PROFILE,
    resolve_track31_run_role,
)
from n0_twam.data.track31_training_identity import (
    validate_checkpoint_latent_inventory_binding,
)
from n0_twam.distributed.optimizer_checkpoint import (
    OPTIMIZER_DCP_DIRNAME,
    OPTIMIZER_STATE_FORMAT,
    STRICT_CHECKPOINT_SCHEMA_VERSION,
    validate_optimizer_checkpoint,
)

from .identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
    validate_recorded_transformer_identity,
    validate_sha256,
    validate_transformer_identity_match,
)
from .compatibility import ACTION_PROJECTION_KEYS
from .runtime_provenance import validate_checkpoint_runtime_provenance
from .sidecar_snapshot import (
    capture_sidecar_snapshot,
    capture_stable_json_file,
    strict_integer,
)
from .stage_a_artifact_identity import validate_stage_a_artifact_identity
from .strict_resume import expected_sidecar_paths

STAGE_A_RUNTIME_LINEAGE_SCHEMA_VERSION = 1


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def load_validated_action_migration_report(
    checkpoint_dir: Path,
    *,
    target_action_schema: str = "qpos8_next_step",
) -> dict[str, object]:
    """Load the original EE20-to-qpos8 migration report without rewriting it."""
    snapshot = capture_stable_json_file(
        Path(checkpoint_dir) / "action_migration_report.json",
        label="action migration report",
    )
    return validate_action_migration_report(
        snapshot.json_object(label="action migration report"),
        target_action_schema=target_action_schema,
    )


def load_validated_action_migration_report_for_contract(
    checkpoint_dir: Path,
    *,
    source_action_dim: int,
    source_action_schema: str,
    target_action_dim: int,
    target_action_schema: str,
    initialized_target_only_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Load and validate a dimension-agnostic migration sidecar."""

    snapshot = capture_stable_json_file(
        Path(checkpoint_dir) / "action_migration_report.json",
        label="action migration report",
    )
    return validate_action_migration_report_for_contract(
        snapshot.json_object(label="action migration report"),
        source_action_dim=source_action_dim,
        source_action_schema=source_action_schema,
        target_action_dim=target_action_dim,
        target_action_schema=target_action_schema,
        initialized_target_only_prefixes=initialized_target_only_prefixes,
    )


def _strict_positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a JSON array of non-empty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicate keys")
    return result


def validate_action_migration_report_for_contract(
    report: Mapping[str, object],
    *,
    source_action_dim: int,
    source_action_schema: str,
    target_action_dim: int,
    target_action_schema: str,
    initialized_target_only_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Validate a dimension-agnostic action migration and exact key partition.

    Unlike the legacy Track 3.1 helper below, this validator accepts an explicit
    source/target contract and therefore supports new action spaces without
    weakening the historical EE20-to-qpos8 lineage checks.
    """

    expected_source_dim = _strict_positive_integer(
        source_action_dim, label="source action dimension"
    )
    expected_target_dim = _strict_positive_integer(
        target_action_dim, label="target action dimension"
    )
    if not source_action_schema or not target_action_schema:
        raise ValueError("action migration schemas must be non-empty")
    if (
        report.get("source_action_dim") != expected_source_dim
        or report.get("source_action_schema") != source_action_schema
        or report.get("target_action_dim") != expected_target_dim
        or report.get("target_action_schema") != target_action_schema
    ):
        raise ValueError("action migration contract does not match source/target")
    if report.get("compatibility") != "migrate_action":
        raise ValueError("action migration compatibility must be migrate_action")

    source_identity = validate_recorded_transformer_identity(
        report.get("source_transformer_identity"),
        expected_action_dim=expected_source_dim,
    )
    source_sha256 = validate_sha256(
        report.get("source_checkpoint_sha256"),
        label="migration source checkpoint SHA256",
    )
    if source_identity["sha256"] != source_sha256:
        raise ValueError(
            "action migration report source transformer identity and SHA256 "
            "are inconsistent"
        )

    plan = report.get("plan")
    required_plan_fields = {
        "copied_keys",
        "reset_keys",
        "missing_target_keys",
        "unexpected_source_keys",
        "shape_mismatches",
    }
    if not isinstance(plan, Mapping) or set(plan) != required_plan_fields:
        raise ValueError("action migration plan has an invalid field set")
    copied = _string_list(plan.get("copied_keys"), label="copied key partition")
    reset = _string_list(plan.get("reset_keys"), label="reset key partition")
    target_only = _string_list(
        plan.get("missing_target_keys"), label="target-only key partition"
    )
    unexpected = _string_list(
        plan.get("unexpected_source_keys"), label="unexpected source key partition"
    )
    if unexpected:
        raise ValueError("action migration contains unexpected source keys")
    mismatches = plan.get("shape_mismatches")
    if not isinstance(mismatches, list) or mismatches:
        raise ValueError("action migration contains shape mismatches")
    if set(reset) != set(ACTION_PROJECTION_KEYS):
        raise ValueError("action migration reset key partition is incomplete")
    if any(
        not any(key.startswith(prefix) for prefix in initialized_target_only_prefixes)
        for key in target_only
    ):
        raise ValueError("action migration contains an unapproved target-only key")
    if (
        set(copied) & set(reset)
        or set(copied) & set(target_only)
        or set(reset) & set(target_only)
    ):
        raise ValueError("action migration key partitions must be disjoint")
    return dict(report)


def validate_action_migration_report(
    report: Mapping[str, object],
    *,
    target_action_schema: str = "qpos8_next_step",
) -> dict[str, object]:
    """Validate migration provenance already captured from stable bytes."""

    if (
        report.get("source_action_dim") != 20
        or report.get("source_action_schema") != "ee20_pi05"
        or report.get("target_action_dim") != 8
        or report.get("target_action_schema") != target_action_schema
    ):
        raise ValueError(
            "action migration report must preserve the original 20D EE to "
            "8D qpos migration"
        )
    source_identity = validate_recorded_transformer_identity(
        report.get("source_transformer_identity"),
        expected_action_dim=20,
    )
    source_sha256 = validate_sha256(
        report.get("source_checkpoint_sha256"),
        label="migration source checkpoint SHA256",
    )
    if source_identity["sha256"] != source_sha256:
        raise ValueError(
            "action migration report source transformer identity and SHA256 "
            "are inconsistent"
        )
    return dict(report)


def validate_parent_training_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_training_profile_id: str,
) -> dict[str, object]:
    """Validate a weights-only parent and its inherited migration provenance."""
    root = Path(checkpoint_dir)
    train_meta = _load_json_object(
        root / "train_meta.json",
        label="parent train metadata",
    )
    actual_profile_id = train_meta.get("training_profile_id")
    if actual_profile_id != expected_training_profile_id:
        raise ValueError(
            "parent training profile mismatch: "
            f"{actual_profile_id!r} vs {expected_training_profile_id!r}"
        )
    if train_meta.get("action_schema") != "qpos8_next_step":
        raise ValueError("parent checkpoint action schema must be qpos8_next_step")

    actual_identity = audit_transformer_checkpoint(
        root / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
        expected_action_dim=8,
    )
    validate_transformer_identity_match(
        train_meta.get("transformer_identity"),
        actual_identity,
        expected_action_dim=8,
        label="parent checkpoint",
    )
    migration_report = load_validated_action_migration_report(root)
    return {
        "training_profile_id": expected_training_profile_id,
        "transformer_identity": actual_identity,
        "action_migration_report": migration_report,
        "train_meta": train_meta,
    }


def _required_integer(value: object, *, label: str, minimum: int = 0) -> int:
    return strict_integer(value, label=label, minimum=minimum)


def validate_stage_a_parent_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_run_role: str,
    expected_track31_artifacts: Mapping[str, object] | None,
) -> dict[str, object]:
    """Validate the complete Stage A parent used by a fresh Stage B run.

    The returned runtime lineage is derived only from bytes audited in this
    call.  Callers must persist it instead of the launch-time config lineage.
    """

    root = Path(checkpoint_dir).resolve(strict=True)
    run_role = resolve_track31_run_role(expected_run_role, environ={})
    completion_snapshot = capture_stable_json_file(
        root / "checkpoint_complete.json",
        label="Stage A parent completion marker",
    )
    completion = completion_snapshot.json_object(
        label="Stage A parent completion marker"
    )
    completion_schema = _required_integer(
        completion.get("schema_version"),
        label="Stage A parent completion schema_version",
        minimum=1,
    )
    world_size = _required_integer(
        completion.get("world_size"),
        label="Stage A parent completion world_size",
        minimum=1,
    )
    if (
        completion_schema != STRICT_CHECKPOINT_SCHEMA_VERSION
        or completion.get("status") != "complete"
        or completion.get("action_schema") != "qpos8_next_step"
        or completion.get("optimizer_state_format") != OPTIMIZER_STATE_FORMAT
    ):
        raise ValueError("Stage A parent completion marker is incompatible")
    sidecar_snapshot = capture_sidecar_snapshot(
        root,
        completion.get("sidecar_inventory"),
        expected_sidecar_paths(
            world_size,
            include_transformer_config=True,
            include_action_migration=True,
        ),
    )
    sidecar_inventory = sidecar_snapshot.inventory
    transformer_config = sidecar_snapshot.json_object(
        "transformer/config.json",
        label="Stage A parent transformer config",
    )
    if (
        transformer_config.get("is_mot") is not True
        or transformer_config.get("action_dim") != 8
        or transformer_config.get("action_schema") != "qpos8_next_step"
    ):
        raise ValueError("Stage A parent transformer config is incompatible")
    training_state = sidecar_snapshot.json_object(
        "training_state.json",
        label="Stage A parent training state",
    )
    train_meta = sidecar_snapshot.json_object(
        "train_meta.json",
        label="Stage A parent train metadata",
    )
    if train_meta.get("training_profile_id") != MULTITASK_PRETRAIN_PROFILE:
        raise ValueError("parent training profile mismatch")
    if train_meta.get("action_schema") != "qpos8_next_step":
        raise ValueError("parent checkpoint action schema must be qpos8_next_step")
    actual_identity = audit_transformer_checkpoint(
        root / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
        expected_action_dim=8,
    )
    validate_transformer_identity_match(
        train_meta.get("transformer_identity"),
        actual_identity,
        expected_action_dim=8,
        label="parent checkpoint",
    )
    migration_report = validate_action_migration_report(
        sidecar_snapshot.json_object(
            "action_migration_report.json",
            label="action migration report",
        )
    )
    parent = {
        "training_profile_id": MULTITASK_PRETRAIN_PROFILE,
        "transformer_identity": actual_identity,
        "action_migration_report": migration_report,
        "train_meta": train_meta,
    }
    artifacts, profile_identity = validate_stage_a_artifact_identity(
        train_meta,
        run_role=run_role,
        expected_track31_artifacts=expected_track31_artifacts,
    )
    for label, payload in (
        ("Stage A parent train metadata", train_meta),
        ("Stage A parent training state", training_state),
        ("Stage A parent completion marker", completion),
    ):
        validate_transformer_identity_match(
            payload.get("transformer_identity"),
            actual_identity,
            expected_action_dim=8,
            label=label,
        )
    validate_checkpoint_latent_inventory_binding(
        expected=artifacts,
        payloads=(
            ("Stage A parent train metadata", train_meta),
            ("Stage A parent training state", training_state),
            ("Stage A parent completion marker", completion),
        ),
    )
    runtime_source_identity, checkpoint_invocation_identity = (
        validate_checkpoint_runtime_provenance(
            (
                ("Stage A parent train metadata", train_meta),
                ("Stage A parent training state", training_state),
                ("Stage A parent completion marker", completion),
            )
        )
    )

    step = _required_integer(
        training_state.get("step"),
        label="Stage A parent training step",
    )
    completion_step = _required_integer(
        completion.get("step"),
        label="Stage A parent completion step",
    )
    state_schema = _required_integer(
        training_state.get("schema_version"),
        label="Stage A parent training state schema_version",
        minimum=1,
    )
    state_world_size = _required_integer(
        training_state.get("world_size"),
        label="Stage A parent training state world_size",
        minimum=1,
    )
    optimizer_sha256 = validate_sha256(
        training_state.get("optimizer_inventory_sha256"),
        label="Stage A parent optimizer inventory SHA256",
    )
    if (
        state_schema != STRICT_CHECKPOINT_SCHEMA_VERSION
        or state_world_size != world_size
        or training_state.get("action_schema") != "qpos8_next_step"
        or training_state.get("optimizer_state_format") != OPTIMIZER_STATE_FORMAT
        or training_state.get("training_profile_identity") != profile_identity
        or completion_step != step
        or completion.get("optimizer_inventory_sha256") != optimizer_sha256
        or completion.get("runtime_signature")
        != training_state.get("runtime_signature")
        or completion.get("training_execution_contract")
        != training_state.get("training_execution_contract")
        or completion.get("training_profile_identity") != profile_identity
        or train_meta.get("training_execution_contract")
        != training_state.get("training_execution_contract")
    ):
        raise ValueError(
            "Stage A parent completion and training state are inconsistent"
        )

    optimizer_inventory = validate_optimizer_checkpoint(
        root / OPTIMIZER_DCP_DIRNAME,
        expected_inventory_sha256=optimizer_sha256,
    )

    sidecar_sha256 = validate_sha256(
        sidecar_inventory.get("inventory_sha256"),
        label="Stage A parent sidecar inventory SHA256",
    )
    runtime_lineage = {
        "schema_version": STAGE_A_RUNTIME_LINEAGE_SCHEMA_VERSION,
        "source_kind": "stage_a_checkpoint",
        "source_checkpoint": str(root),
        "parent_training_profile_id": MULTITASK_PRETRAIN_PROFILE,
        "parent_run_role": run_role,
        "parent_training_profile_identity": profile_identity,
        "parent_transformer_identity": dict(actual_identity),
        "parent_checkpoint_step": step,
        "parent_checkpoint_world_size": world_size,
        "parent_sidecar_inventory_sha256": sidecar_sha256,
        "parent_optimizer_inventory_sha256": optimizer_inventory.inventory_sha256,
        "parent_runtime_source_identity": runtime_source_identity,
        "parent_checkpoint_invocation_identity": checkpoint_invocation_identity,
    }
    return {
        **parent,
        "train_meta": dict(train_meta),
        "training_state": training_state,
        "completion": completion,
        "sidecar_inventory": sidecar_inventory,
        "sidecar_snapshot": sidecar_snapshot,
        "completion_snapshot": completion_snapshot,
        "optimizer_inventory": optimizer_inventory.to_json_dict(),
        "transformer_config": transformer_config,
        "runtime_training_lineage": runtime_lineage,
        "runtime_source_identity": runtime_source_identity,
        "checkpoint_invocation_identity": checkpoint_invocation_identity,
    }


__all__ = (
    "STAGE_A_RUNTIME_LINEAGE_SCHEMA_VERSION",
    "load_validated_action_migration_report",
    "load_validated_action_migration_report_for_contract",
    "validate_parent_training_checkpoint",
    "validate_action_migration_report",
    "validate_stage_a_parent_checkpoint",
)
