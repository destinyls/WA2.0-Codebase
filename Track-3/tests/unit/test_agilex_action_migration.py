# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Generic action-migration lineage tests for AgileX qpos14."""

from __future__ import annotations

import pytest

from n0_twam.checkpointing.training_lineage import (
    validate_action_migration_report_for_contract,
)


def _source_identity() -> dict[str, object]:
    return {
        "schema_version": 1,
        "file_name": "diffusion_pytorch_model.safetensors",
        "sha256": "a" * 64,
        "size_bytes": 123,
        "tensor_count": 7,
        "action_dim": 20,
        "action_shapes": {
            "action_embedder.weight": [3072, 20],
            "action_embedder.bias": [3072],
            "action_proj_out.weight": [20, 3072],
            "action_proj_out.bias": [20],
        },
        "required_sentinel_keys": [
            "condition_embedder.text_embedder.linear_1.weight",
            "mot.experts.action.in_proj.weight",
            "mot.experts.tactile.in_proj.weight",
        ],
    }


def _report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "compatibility": "migrate_action",
        "source_action_dim": 20,
        "source_action_schema": "ee20_pi05",
        "target_action_dim": 14,
        "target_action_schema": "qpos14_joint_absolute_v1",
        "action_init_seed": 20260811,
        "source_checkpoint": "/immutable/base/transformer",
        "source_checkpoint_sha256": "a" * 64,
        "source_transformer_identity": _source_identity(),
        "plan": {
            "copied_keys": ["backbone.weight"],
            "reset_keys": [
                "action_embedder.bias",
                "action_embedder.weight",
                "action_proj_out.bias",
                "action_proj_out.weight",
            ],
            "missing_target_keys": ["agilex_wrench_conditioner.proj.weight"],
            "unexpected_source_keys": [],
            "shape_mismatches": [],
        },
    }


def test_qpos14_migration_report_validates_generic_contract() -> None:
    validated = validate_action_migration_report_for_contract(
        _report(),
        source_action_dim=20,
        source_action_schema="ee20_pi05",
        target_action_dim=14,
        target_action_schema="qpos14_joint_absolute_v1",
        initialized_target_only_prefixes=("agilex_wrench_conditioner.",),
    )

    assert validated["target_action_dim"] == 14
    assert validated["target_action_schema"] == "qpos14_joint_absolute_v1"


def test_qpos14_migration_report_rejects_wrong_target_contract() -> None:
    report = _report()
    report["target_action_dim"] = 8

    with pytest.raises(ValueError, match="action migration contract"):
        validate_action_migration_report_for_contract(
            report,
            source_action_dim=20,
            source_action_schema="ee20_pi05",
            target_action_dim=14,
            target_action_schema="qpos14_joint_absolute_v1",
            initialized_target_only_prefixes=("agilex_wrench_conditioner.",),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reset_keys", ["action_embedder.weight"], "reset key partition"),
        ("unexpected_source_keys", ["obsolete.weight"], "unexpected source"),
        (
            "missing_target_keys",
            ["unapproved_module.weight"],
            "target-only key",
        ),
    ],
)
def test_qpos14_migration_report_rejects_invalid_key_partition(
    field: str,
    value: list[str],
    message: str,
) -> None:
    report = _report()
    plan = dict(report["plan"])
    plan[field] = value
    report["plan"] = plan

    with pytest.raises(ValueError, match=message):
        validate_action_migration_report_for_contract(
            report,
            source_action_dim=20,
            source_action_schema="ee20_pi05",
            target_action_dim=14,
            target_action_schema="qpos14_joint_absolute_v1",
            initialized_target_only_prefixes=("agilex_wrench_conditioner.",),
        )
