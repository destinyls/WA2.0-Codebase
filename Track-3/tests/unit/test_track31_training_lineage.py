"""Tests for strict Stage A parent and inherited migration provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from n0_twam.checkpointing.identity import audit_transformer_checkpoint
from n0_twam.checkpointing.training_lineage import (
    load_validated_action_migration_report,
    validate_parent_training_checkpoint,
    validate_stage_a_parent_checkpoint,
)
from tests.unit.test_track31_preflight import (
    _TRACK31_ARTIFACT_IDENTITY,
    _write_completion_marker,
    _write_resume_sidecars,
    _write_training_state,
    _write_transformer_checkpoint,
)

STAGE_A = "multitask_pretrain_v1"


def _write_parent_checkpoint(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    weights = {
        "action_embedder.weight": torch.zeros(3072, 8),
        "action_embedder.bias": torch.zeros(3072),
        "action_proj_out.weight": torch.zeros(8, 3072),
        "action_proj_out.bias": torch.zeros(8),
        "condition_embedder.text_embedder.linear_1.weight": torch.zeros(1),
        "mot.experts.action.in_proj.weight": torch.zeros(1),
        "mot.experts.tactile.in_proj.weight": torch.zeros(1),
    }
    save_file(weights, transformer / "diffusion_pytorch_model.safetensors")

    from n0_twam.checkpointing.identity import audit_transformer_checkpoint

    parent_identity = audit_transformer_checkpoint(
        transformer / "diffusion_pytorch_model.safetensors",
        expected_action_dim=8,
    )
    source_identity = {
        **parent_identity,
        "sha256": "a" * 64,
        "action_dim": 20,
        "action_shapes": {
            "action_embedder.weight": [3072, 20],
            "action_embedder.bias": [3072],
            "action_proj_out.weight": [20, 3072],
            "action_proj_out.bias": [20],
        },
    }
    migration = {
        "source_action_dim": 20,
        "source_action_schema": "ee20_pi05",
        "target_action_dim": 8,
        "target_action_schema": "qpos8_next_step",
        "source_checkpoint_sha256": source_identity["sha256"],
        "source_transformer_identity": source_identity,
    }
    (root / "action_migration_report.json").write_text(
        json.dumps(migration), encoding="utf-8"
    )
    (root / "train_meta.json").write_text(
        json.dumps(
            {
                "training_profile_id": STAGE_A,
                "action_schema": "qpos8_next_step",
                "transformer_identity": parent_identity,
            }
        ),
        encoding="utf-8",
    )
    return parent_identity, migration


def test_stage_b_parent_validation_preserves_original_migration(tmp_path: Path) -> None:
    parent_identity, migration = _write_parent_checkpoint(tmp_path)

    contract = validate_parent_training_checkpoint(
        tmp_path,
        expected_training_profile_id=STAGE_A,
    )

    assert contract["transformer_identity"] == parent_identity
    assert contract["action_migration_report"] == migration
    assert contract["action_migration_report"]["source_action_dim"] == 20
    assert contract["action_migration_report"]["source_checkpoint_sha256"] == "a" * 64


def test_stage_b_parent_validation_rejects_wrong_profile(tmp_path: Path) -> None:
    _write_parent_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="training profile"):
        validate_parent_training_checkpoint(
            tmp_path,
            expected_training_profile_id="target_finetune_v1",
        )


def test_inherited_migration_rejects_rewritten_stage_a_source(tmp_path: Path) -> None:
    parent_identity, _ = _write_parent_checkpoint(tmp_path)
    migration_path = tmp_path / "action_migration_report.json"
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    migration["source_action_dim"] = 8
    migration["source_action_schema"] = "qpos8_next_step"
    migration["source_checkpoint_sha256"] = parent_identity["sha256"]
    migration["source_transformer_identity"] = parent_identity
    migration_path.write_text(json.dumps(migration), encoding="utf-8")

    with pytest.raises(ValueError, match="20D EE"):
        load_validated_action_migration_report(tmp_path)


def test_parent_validation_rejects_recorded_identity_tamper(tmp_path: Path) -> None:
    _write_parent_checkpoint(tmp_path)
    metadata_path = tmp_path / "train_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["transformer_identity"]["sha256"] = "b" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="transformer identity"):
        validate_parent_training_checkpoint(
            tmp_path,
            expected_training_profile_id=STAGE_A,
        )


def _write_formal_stage_a_parent(
    root: Path,
    *,
    run_role: str = "final_refit",
) -> dict[str, object]:
    _write_transformer_checkpoint(
        root,
        action_dim=8,
        action_schema="qpos8_next_step",
    )
    strict_fields = _write_resume_sidecars(
        root,
        world_size=1,
        training_profile_id=STAGE_A,
        run_role=run_role,
    )
    _write_training_state(root, strict_fields, world_size=1)
    return strict_fields


def _validate_formal_stage_a_parent(
    root: Path,
    *,
    run_role: str = "final_refit",
    expected_artifacts: dict[str, object] | None = None,
) -> dict[str, object]:
    return validate_stage_a_parent_checkpoint(
        root,
        expected_run_role=run_role,
        expected_track31_artifacts=(
            _TRACK31_ARTIFACT_IDENTITY
            if expected_artifacts is None
            else expected_artifacts
        ),
    )


def test_canonical_stage_a_parent_binds_runtime_lineage(tmp_path: Path) -> None:
    _write_formal_stage_a_parent(tmp_path)

    contract = _validate_formal_stage_a_parent(tmp_path)

    lineage = contract["runtime_training_lineage"]
    assert isinstance(lineage, dict)
    assert lineage["source_kind"] == "stage_a_checkpoint"
    assert lineage["parent_training_profile_id"] == STAGE_A
    assert lineage["parent_run_role"] == "final_refit"
    assert lineage["parent_transformer_identity"] == contract["transformer_identity"]
    assert (
        lineage["parent_sidecar_inventory_sha256"]
        == contract["sidecar_inventory"]["inventory_sha256"]
    )
    assert (
        lineage["parent_optimizer_inventory_sha256"]
        == contract["optimizer_inventory"]["inventory_sha256"]
    )
    assert (
        lineage["parent_runtime_source_identity"] == contract["runtime_source_identity"]
    )
    assert (
        lineage["parent_checkpoint_invocation_identity"]
        == contract["checkpoint_invocation_identity"]
    )


def test_canonical_stage_a_parent_rejects_well_formed_artifact_sha_tamper(
    tmp_path: Path,
) -> None:
    strict_fields = _write_formal_stage_a_parent(tmp_path)
    metadata_path = tmp_path / "train_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["track31_artifacts"]["manifest_sha256"] = "9" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_completion_marker(tmp_path, strict_fields, world_size=1, step=7)

    with pytest.raises(ValueError, match="artifact identity mismatch"):
        _validate_formal_stage_a_parent(tmp_path)


@pytest.mark.parametrize(
    "missing_relative_path",
    ("checkpoint_complete.json", "scheduler_state.json"),
)
def test_canonical_stage_a_parent_rejects_missing_completion_or_sidecar(
    tmp_path: Path,
    missing_relative_path: str,
) -> None:
    _write_formal_stage_a_parent(tmp_path)
    (tmp_path / missing_relative_path).unlink()

    with pytest.raises(FileNotFoundError, match="completion|sidecar|metadata"):
        _validate_formal_stage_a_parent(tmp_path)


def test_trainer_revalidation_rejects_parent_transformer_swap_after_preflight(
    tmp_path: Path,
) -> None:
    _write_formal_stage_a_parent(tmp_path)
    preflight_contract = _validate_formal_stage_a_parent(tmp_path)
    _write_transformer_checkpoint(
        tmp_path,
        action_dim=8,
        action_schema="qpos8_next_step",
        fill_value=1.0,
    )

    with pytest.raises(ValueError, match="transformer identity"):
        _validate_formal_stage_a_parent(tmp_path)
    assert preflight_contract["transformer_identity"]["sha256"] != (
        audit_transformer_checkpoint(
            tmp_path / "transformer" / "diffusion_pytorch_model.safetensors",
            expected_action_dim=8,
        )["sha256"]
    )


@pytest.mark.parametrize("missing", ("optimizer_dcp", "optimizer_dcp/__0_0.distcp"))
def test_canonical_stage_a_parent_rejects_missing_optimizer_payload(
    tmp_path: Path,
    missing: str,
) -> None:
    _write_formal_stage_a_parent(tmp_path)
    target = tmp_path / missing
    if target.is_dir():
        target.rename(tmp_path / "missing_optimizer_dcp")
    else:
        target.unlink()

    with pytest.raises((FileNotFoundError, ValueError), match="optimizer|DCP|distcp"):
        _validate_formal_stage_a_parent(tmp_path)


def test_canonical_stage_a_parent_rejects_optimizer_shard_drift(
    tmp_path: Path,
) -> None:
    _write_formal_stage_a_parent(tmp_path)
    (tmp_path / "optimizer_dcp" / "__0_0.distcp").write_bytes(b"drift")

    with pytest.raises(ValueError, match="DCP payload|optimizer"):
        _validate_formal_stage_a_parent(tmp_path)


def test_development_parent_binds_expected_validation_view(
    tmp_path: Path,
) -> None:
    _write_formal_stage_a_parent(tmp_path, run_role="development")
    expected = {
        **_TRACK31_ARTIFACT_IDENTITY,
        "normalizer_source_view_id": "stage_a_dev719_v1",
        "normalizer_source_view_sha256": "e" * 64,
        "parent_validation_view_id": "internal_dev40_v1",
        "parent_validation_view_sha256": "9" * 64,
    }

    with pytest.raises(ValueError, match="validation.*identity mismatch"):
        _validate_formal_stage_a_parent(
            tmp_path,
            run_role="development",
            expected_artifacts=expected,
        )
