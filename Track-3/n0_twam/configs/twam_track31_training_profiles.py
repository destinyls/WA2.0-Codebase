# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed Stage A/B training profiles for UniVTAC Track 3.1.

The profile contract deliberately separates same-profile continuation from a
Stage A to Stage B transition.  A continuation restores full training state;
the optional Stage B branch loads Stage A model weights only and starts a fresh
optimizer, scheduler, RNG stream, and data cursor.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

MULTITASK_PRETRAIN_PROFILE: Final = "multitask_pretrain_v1"
TARGET_FINETUNE_PROFILE: Final = "target_finetune_v1"
DEVELOPMENT_RUN_ROLE: Final = "development"
FINAL_REFIT_RUN_ROLE: Final = "final_refit"

TRAINING_PROFILE_IDS: Final = frozenset(
    {MULTITASK_PRETRAIN_PROFILE, TARGET_FINETUNE_PROFILE}
)
RUN_ROLE_IDS: Final = frozenset({DEVELOPMENT_RUN_ROLE, FINAL_REFIT_RUN_ROLE})


@dataclass(frozen=True)
class Track31ProfileSpec:
    """Immutable data and normalization identities for one training profile."""

    training_profile_id: str
    development_train_view_id: str
    development_validation_view_id: str
    final_train_view_id: str
    development_normalizer_id: str
    final_normalizer_id: str
    expected_parent_profile_id: str | None


PROFILE_SPECS: Final[dict[str, Track31ProfileSpec]] = {
    MULTITASK_PRETRAIN_PROFILE: Track31ProfileSpec(
        training_profile_id=MULTITASK_PRETRAIN_PROFILE,
        development_train_view_id="stage_a_dev719_v1",
        development_validation_view_id="internal_dev40_v1",
        final_train_view_id="stage_a_final759_v1",
        development_normalizer_id="qpos8_dev719_v1",
        final_normalizer_id="qpos8_final759_v1",
        expected_parent_profile_id=None,
    ),
    TARGET_FINETUNE_PROFILE: Track31ProfileSpec(
        training_profile_id=TARGET_FINETUNE_PROFILE,
        development_train_view_id="stage_b_dev180_v1",
        development_validation_view_id="internal_target_dev10_v1",
        final_train_view_id="stage_b_final190_v1",
        # Stage B must preserve the Stage A action representation.
        development_normalizer_id="qpos8_dev719_v1",
        final_normalizer_id="qpos8_final759_v1",
        expected_parent_profile_id=MULTITASK_PRETRAIN_PROFILE,
    ),
}


def _resolve_choice(
    *,
    value: str | None,
    env_name: str,
    default: str,
    choices: frozenset[str],
    environ: Mapping[str, str] | None,
) -> str:
    environment = os.environ if environ is None else environ
    resolved = value if value is not None else environment.get(env_name, default)
    if resolved not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{env_name} must be one of {{{allowed}}}, got {resolved!r}")
    return resolved


def resolve_track31_train_profile(
    value: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the mandatory Stage A or optional Stage B profile ID."""
    return _resolve_choice(
        value=value,
        env_name="N0_TRACK31_TRAIN_PROFILE",
        default=MULTITASK_PRETRAIN_PROFILE,
        choices=TRAINING_PROFILE_IDS,
        environ=environ,
    )


def resolve_track31_run_role(
    value: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve development model-selection or validation-free final refit."""
    return _resolve_choice(
        value=value,
        env_name="N0_TRACK31_RUN_ROLE",
        default=FINAL_REFIT_RUN_ROLE,
        choices=RUN_ROLE_IDS,
        environ=environ,
    )


def _checkpoint_value(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _parent_profile_id(parent_train_meta: Mapping[str, object] | None) -> str:
    if parent_train_meta is None:
        raise ValueError("parent train_meta is required for this initialization route")
    profile_id = parent_train_meta.get("training_profile_id")
    if not isinstance(profile_id, str) or profile_id not in TRAINING_PROFILE_IDS:
        raise ValueError(
            "parent train_meta must contain a recognized training_profile_id"
        )
    return profile_id


def build_track31_training_profile_contract(
    *,
    training_profile_id: str,
    run_role: str,
    train_view_id: str | None = None,
    validation_view_id: str | None = None,
    released_checkpoint: str | Path | None = None,
    init_from: str | Path | None = None,
    resume_from: str | Path | None = None,
    parent_train_meta: Mapping[str, object] | None = None,
    source_training_profile_id: str | None = None,
    parent_checkpoint_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the canonical profile, view, and initialization contract.

    ``resume_from`` is legal only when the parent has the same profile ID.
    ``init_from`` is reserved for a fresh Stage B branch and requires a Stage A
    parent.  A fresh Stage A always starts from the released EE20 checkpoint
    using the recorded 20D-to-8D action migration.
    """
    profile_id = resolve_track31_train_profile(training_profile_id, environ={})
    role = resolve_track31_run_role(run_role, environ={})
    spec = PROFILE_SPECS[profile_id]
    released_path = _checkpoint_value(released_checkpoint)
    init_path = _checkpoint_value(init_from)
    resume_path = _checkpoint_value(resume_from)

    if init_path is not None and resume_path is not None:
        raise ValueError("init_from and resume_from are mutually exclusive")

    if role == DEVELOPMENT_RUN_ROLE:
        expected_train_view_id = spec.development_train_view_id
        expected_validation_view_id: str | None = spec.development_validation_view_id
        normalizer_id = spec.development_normalizer_id
        normalizer_source_view_id = PROFILE_SPECS[
            MULTITASK_PRETRAIN_PROFILE
        ].development_train_view_id
    else:
        expected_train_view_id = spec.final_train_view_id
        expected_validation_view_id = None
        normalizer_id = spec.final_normalizer_id
        normalizer_source_view_id = PROFILE_SPECS[
            MULTITASK_PRETRAIN_PROFILE
        ].final_train_view_id

    resolved_train_view_id = train_view_id or expected_train_view_id
    if resolved_train_view_id != expected_train_view_id:
        raise ValueError(
            f"{profile_id}/{role} requires train view "
            f"{expected_train_view_id!r}, got {resolved_train_view_id!r}"
        )
    if validation_view_id is None:
        resolved_validation_view_id = expected_validation_view_id
    else:
        resolved_validation_view_id = validation_view_id
    if resolved_validation_view_id != expected_validation_view_id:
        if role == FINAL_REFIT_RUN_ROLE:
            raise ValueError("final_refit must not construct a validation view")
        raise ValueError(
            f"{profile_id}/{role} requires validation view "
            f"{expected_validation_view_id!r}, got "
            f"{resolved_validation_view_id!r}"
        )

    expected_parent = spec.expected_parent_profile_id
    parent_profile: str | None = None
    if source_training_profile_id is not None:
        if source_training_profile_id not in TRAINING_PROFILE_IDS:
            raise ValueError("source_training_profile_id is not recognized")
        parent_profile = source_training_profile_id
    elif parent_train_meta is not None:
        parent_profile = _parent_profile_id(parent_train_meta)
    resolved_parent_identity = parent_checkpoint_identity
    if resolved_parent_identity is None and parent_train_meta is not None:
        recorded_identity = parent_train_meta.get("transformer_identity")
        if recorded_identity is not None:
            if not isinstance(recorded_identity, Mapping):
                raise ValueError(
                    "parent transformer_identity must be a mapping when present"
                )
            resolved_parent_identity = recorded_identity
    if resume_path is not None:
        if parent_profile is None:
            parent_profile = _parent_profile_id(parent_train_meta)
        if parent_profile != profile_id:
            raise ValueError(
                "strict resume requires the same training profile; use a "
                "weights-only Stage A initialization for Stage B"
            )
        checkpoint_path = resume_path
        initialization_mode = "strict_full_state_resume"
        checkpoint_compatibility = "strict"
        source_action_dim = 8
        source_action_schema = "qpos8_next_step"
        source_kind = "same_profile_resume"
    elif profile_id == MULTITASK_PRETRAIN_PROFILE:
        if init_path is not None:
            raise ValueError(
                "fresh Stage A uses released_checkpoint; init_from is reserved "
                "for optional Stage B"
            )
        if released_path is None:
            raise ValueError("fresh Stage A requires released_checkpoint")
        checkpoint_path = released_path
        initialization_mode = "released_action_migration"
        checkpoint_compatibility = "migrate_action"
        source_action_dim = 20
        source_action_schema = "ee20_pi05"
        source_kind = "released_20d"
    else:
        if init_path is None:
            raise ValueError("fresh Stage B requires init_from Stage A checkpoint")
        if parent_profile is None:
            parent_profile = _parent_profile_id(parent_train_meta)
        if parent_profile != expected_parent:
            raise ValueError(
                "fresh Stage B requires a multitask_pretrain_v1 parent checkpoint"
            )
        checkpoint_path = init_path
        initialization_mode = "stage_a_weights_only"
        checkpoint_compatibility = "strict"
        source_action_dim = 8
        source_action_schema = "qpos8_next_step"
        source_kind = "stage_a_checkpoint"

    contract: dict[str, object] = {
        "schema_version": 1,
        "training_profile_id": profile_id,
        "run_role": role,
        "profile_spec": asdict(spec),
        "train_view_id": resolved_train_view_id,
        "validation_view_id": resolved_validation_view_id,
        "normalizer_id": normalizer_id,
        "normalizer_source_view_id": normalizer_source_view_id,
        "initialization_mode": initialization_mode,
        "checkpoint_path": checkpoint_path,
        "checkpoint_compatibility": checkpoint_compatibility,
        "checkpoint_source_action_dim": source_action_dim,
        "checkpoint_source_action_schema": source_action_schema,
        "expected_parent_profile_id": expected_parent,
        "parent_training_profile_id": parent_profile,
        "training_lineage": {
            "source_kind": source_kind,
            "source_checkpoint": checkpoint_path,
            "parent_training_profile_id": parent_profile,
            "parent_checkpoint_identity": (
                None
                if resolved_parent_identity is None
                else dict(resolved_parent_identity)
            ),
        },
        "reset_optimizer": initialization_mode != "strict_full_state_resume",
        "reset_scheduler": initialization_mode != "strict_full_state_resume",
        "reset_rng": initialization_mode != "strict_full_state_resume",
        "reset_data_cursor": initialization_mode != "strict_full_state_resume",
        "inherit_action_migration_report": (
            initialization_mode == "stage_a_weights_only"
        ),
        "has_validation_loader": resolved_validation_view_id is not None,
    }
    return contract
