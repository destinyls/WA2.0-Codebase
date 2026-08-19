# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Checkpoint compatibility and resumable bundle contracts."""

from typing import TYPE_CHECKING

from .compatibility import (
    ACTION_PROJECTION_KEYS,
    ActionMigrationPlan,
    TensorShapeMismatch,
    build_action_migration_plan,
)
from .identity import (
    ACTION_TENSOR_KEYS,
    TRANSFORMER_ACTION_INNER_DIM,
    TRANSFORMER_IDENTITY_SCHEMA_VERSION,
    TRANSFORMER_SENTINEL_KEYS,
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
    validate_recorded_transformer_identity,
    validate_sha256,
    validate_transformer_identity_match,
)

if TYPE_CHECKING:
    from .training_lineage import (
        load_validated_action_migration_report,
        load_validated_action_migration_report_for_contract,
        validate_parent_training_checkpoint,
    )


def __getattr__(name: str) -> object:
    """Load lineage helpers lazily to keep data identity imports acyclic."""

    if name == "load_validated_action_migration_report":
        from .training_lineage import load_validated_action_migration_report

        return load_validated_action_migration_report
    if name == "load_validated_action_migration_report_for_contract":
        from .training_lineage import (
            load_validated_action_migration_report_for_contract,
        )

        return load_validated_action_migration_report_for_contract
    if name == "validate_parent_training_checkpoint":
        from .training_lineage import validate_parent_training_checkpoint

        return validate_parent_training_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "ACTION_PROJECTION_KEYS",
    "ActionMigrationPlan",
    "ACTION_TENSOR_KEYS",
    "TensorShapeMismatch",
    "TRANSFORMER_ACTION_INNER_DIM",
    "TRANSFORMER_IDENTITY_SCHEMA_VERSION",
    "TRANSFORMER_SENTINEL_KEYS",
    "TRANSFORMER_WEIGHTS_FILENAME",
    "audit_transformer_checkpoint",
    "build_action_migration_plan",
    "load_validated_action_migration_report",
    "load_validated_action_migration_report_for_contract",
    "validate_recorded_transformer_identity",
    "validate_parent_training_checkpoint",
    "validate_sha256",
    "validate_transformer_identity_match",
)
