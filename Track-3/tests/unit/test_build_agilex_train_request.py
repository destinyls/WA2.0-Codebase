# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Focused contracts for the immutable AgileX training request builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from n0_twam.integrations.worldarena.agilex_manifest import sha256_file
from script.track3_2 import build_agilex_train_request as builder


def _args(tmp_path: Path) -> argparse.Namespace:
    for name in ("raw", "dataset", "artifacts", "base"):
        (tmp_path / name).mkdir()
    artifacts = tmp_path / "artifacts"
    for name in (
        "source_manifest.json",
        "conversion_receipt.json",
        "latent_inventory.json",
        "repo_routes.json",
        "temporal_alignment.json",
        "qpos14_normalizer.json",
    ):
        (artifacts / name).write_text("{}", encoding="utf-8")
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"empty")
    checkpoint = tmp_path / "checkpoint"
    transformer = checkpoint / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(
        json.dumps(
            {
                "action_dim": 14,
                "action_schema": "qpos14_joint_absolute_v1",
            }
        ),
        encoding="utf-8",
    )
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")
    return argparse.Namespace(
        raw_root=tmp_path / "raw",
        dataset_root=tmp_path / "dataset",
        artifact_root=artifacts,
        base_model=tmp_path / "base",
        empty_embedding=empty,
        init_from=checkpoint,
        output_root=tmp_path / "new-output",
        output=tmp_path / "request.json",
        run_id="agilex-mixed-stage-b",
        devices=8,
        master_port=29653,
        network_interface="bond1",
        seed=20260812,
        num_steps=28500,
        stop_after_step=28500,
        save_interval=1500,
        val_interval=100,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_latent_frames=5,
    )


def test_builder_emits_requested_stage_b_recipe_and_audits_qpos14_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    artifacts = args.artifact_root
    expected_artifact_identity = {
        "schema_version": 1,
        "embodiment_profile_id": "agilex_dual_qpos14_v1",
        "action_schema": "qpos14_joint_absolute_v1",
        "tactile_profile": "mixed",
        "source_manifest_file_sha256": sha256_file(artifacts / "source_manifest.json"),
        "conversion_receipt_file_sha256": sha256_file(
            artifacts / "conversion_receipt.json"
        ),
        "latent_inventory_file_sha256": sha256_file(
            artifacts / "latent_inventory.json"
        ),
        "repo_route_manifest_file_sha256": sha256_file(artifacts / "repo_routes.json"),
        "temporal_alignment_file_sha256": sha256_file(
            artifacts / "temporal_alignment.json"
        ),
        "normalizer_file_sha256": sha256_file(artifacts / "qpos14_normalizer.json"),
    }
    common = {"track32_artifact_identity": expected_artifact_identity}
    snapshot = SimpleNamespace(
        world_size=8,
        transformer_config={
            "action_dim": 14,
            "action_schema": "qpos14_joint_absolute_v1",
        },
        train_meta={
            **common,
            "track32_profile_id": "agilex_track3_mixed_v1",
            "run_role": "final_refit",
        },
        training_state=dict(common),
        completion=dict(common),
        action_migration_report={},
    )
    observed_action_dims: list[int] = []

    def audit_transformer(*_args: object, **kwargs: object) -> dict[str, object]:
        observed_action_dims.append(int(kwargs["expected_action_dim"]))
        return {"sha256": "a" * 64, "action_dim": 14}

    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.capture_strict_checkpoint_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.build_strict_checkpoint_identity",
        lambda *_args, **_kwargs: {"identity_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.audit_transformer_checkpoint",
        audit_transformer,
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.validate_action_migration_report_for_contract",
        lambda report, **_kwargs: dict(report),
    )
    monkeypatch.setattr(builder, "audit_transformer_checkpoint", audit_transformer)

    builder.build(args)

    payload = json.loads(args.output.read_text(encoding="utf-8"))
    assert payload["train"] == {
        "run_role": "final_refit",
        "num_steps": 28500,
        "stop_after_step": 28500,
        "save_interval": 1500,
        "val_interval": 100,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_latent_frames": 5,
        "seed": 20260812,
    }
    assert payload["paths"]["init_transformer_sha256"] == "a" * 64
    assert payload["runtime"]["devices"] == list(range(8))
    assert observed_action_dims == [14]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("num_steps", 0, "positive integer"),
        ("stop_after_step", 28501, "cannot exceed"),
        ("batch_size", 2, "batch_size=1"),
    ),
)
def test_builder_rejects_invalid_typed_recipe_before_writing_request(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    args = _args(tmp_path)
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        builder.build(args)
    assert not args.output.exists()


@pytest.mark.parametrize("value", ("0", "-1", "1.5", "not-an-integer"))
def test_builder_cli_recipe_fields_require_positive_integers(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        builder._positive_int(value)
