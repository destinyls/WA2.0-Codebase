# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Physical-artifact completion gates for AgileX post-training."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from n0_twam.checkpointing.identity import audit_transformer_checkpoint
from n0_twam.checkpointing.runtime_provenance import LOCAL_EXECUTION_TIER
from n0_twam.checkpointing.strict_resume import (
    build_sidecar_inventory,
    expected_sidecar_paths,
)
from n0_twam.distributed.optimizer_checkpoint import (
    OPTIMIZER_STATE_FORMAT,
    STRICT_CHECKPOINT_SCHEMA_VERSION,
    build_training_execution_contract,
)
from n0_twam.embodiments import AGILEX_ACTION_SCHEMA
from n0_twam.configs.twam_track3_agilex_recipe import (
    build_agilex_resume_recipe_contract,
)
from n0_twam.track31.local_provenance import LocalProvenance
from n0_twam.track32_agilex.completion import verify_completed_checkpoint
from n0_twam.track32_agilex.preflight import (
    agilex_profile_id,
    build_artifact_identity,
)
from n0_twam.track32_agilex.request import (
    AgileXPathRequest,
    AgileXRuntimeRequest,
    AgileXTrainRecipe,
    AgileXTrainRequest,
)
from tests.unit.test_track31_checkpoint_review_regressions import _write_resume
from tests.unit.test_track31_preflight import _write_transformer_checkpoint


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _request(tmp_path: Path) -> AgileXTrainRequest:
    paths = AgileXPathRequest(
        source_root=tmp_path / "raw",
        source_manifest=tmp_path / "source.json",
        source_manifest_sha256="4" * 64,
        dataset_root=tmp_path / "dataset",
        artifact_root=tmp_path / "artifacts",
        conversion_receipt=tmp_path / "conversion.json",
        conversion_receipt_sha256="5" * 64,
        latent_inventory=tmp_path / "latents.json",
        latent_inventory_sha256="6" * 64,
        repo_route_manifest=tmp_path / "routes.json",
        repo_route_manifest_sha256="7" * 64,
        temporal_alignment=tmp_path / "temporal.json",
        temporal_alignment_sha256="8" * 64,
        normalizer=tmp_path / "normalizer.json",
        normalizer_sha256="9" * 64,
        base_model=tmp_path / "base",
        empty_embedding=tmp_path / "empty.pt",
        empty_embedding_sha256="3" * 64,
        init_from=tmp_path / "released",
        init_transformer_sha256="a" * 64,
        resume_from=None,
        resume_checkpoint_identity_sha256=None,
        output_root=tmp_path / "run",
    )
    return AgileXTrainRequest(
        source_path=tmp_path / "request.json",
        source_sha256="c" * 64,
        run_id="agilex-completion",
        profile="vision_only",
        runtime=AgileXRuntimeRequest(
            devices=(0,),
            master_port=29500,
            accelerator_profile="portable",
            collective_network_interface=None,
        ),
        paths=paths,
        train=AgileXTrainRecipe(
            run_role="development",
            num_steps=2000,
            stop_after_step=7,
            save_interval=300,
            val_interval=100,
            batch_size=1,
            gradient_accumulation_steps=4,
            max_latent_frames=5,
            seed=20260811,
        ),
    )


def _provenance(tmp_path: Path) -> LocalProvenance:
    runtime = {
        "schema_version": 2,
        "execution_tier": LOCAL_EXECUTION_TIER,
        "code_manifest_sha256": "1" * 64,
        "environment_manifest_sha256": "2" * 64,
        "empty_embedding_sha256": "3" * 64,
        "package_version": "0.1.0",
    }
    invocation = {
        "schema_version": 2,
        "execution_tier": LOCAL_EXECUTION_TIER,
        "invocation_id": "agilex-completion",
        "launch_receipt_sha256": "b" * 64,
    }
    return LocalProvenance(
        invocation_id="agilex-completion",
        code_manifest_path=tmp_path / "code.json",
        environment_manifest_path=tmp_path / "environment.json",
        launch_receipt_path=tmp_path / "launch.json",
        runtime_source_identity=runtime,
        checkpoint_invocation_identity=invocation,
    )


def _execution_contract(request: AgileXTrainRequest) -> dict[str, object]:
    return build_training_execution_contract(
        max_latent_frames=request.train.max_latent_frames,
        gradient_accumulation_steps=request.train.gradient_accumulation_steps,
        batch_size=request.train.batch_size,
        load_worker=0,
        num_steps=request.train.num_steps,
        lr_schedule="cosine",
        warmup_steps=20,
        lr_min_ratio=0.1,
        activation_checkpointing=True,
        attention_contract={
            "attention_backend": "grouped_sdpa",
            "grouped_sdpa_max_query_tokens": 16384,
            "mot_cross_attention_backend": "sdpa",
        },
    )


def _reseal(checkpoint: Path) -> None:
    completion_path = checkpoint / "checkpoint_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["sidecar_inventory"] = build_sidecar_inventory(
        checkpoint,
        expected_sidecar_paths(
            1,
            include_transformer_config=True,
            include_action_migration=True,
        ),
    )
    _write_json(completion_path, completion)


def _checkpoint_fixture(
    tmp_path: Path,
) -> tuple[AgileXTrainRequest, LocalProvenance, Path]:
    request = _request(tmp_path)
    provenance = _provenance(tmp_path)
    checkpoint = (
        request.paths.output_root
        / "checkpoints"
        / f"checkpoint_step_{request.train.stop_after_step}"
    )
    _write_resume(checkpoint)
    _write_transformer_checkpoint(
        checkpoint,
        action_dim=14,
        action_schema=AGILEX_ACTION_SCHEMA,
    )
    transformer_identity = audit_transformer_checkpoint(
        checkpoint / "transformer" / "diffusion_pytorch_model.safetensors",
        expected_action_dim=14,
    )
    execution = _execution_contract(request)
    artifacts = build_artifact_identity(request)
    lineage = {
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
    common = {
        "action_schema": AGILEX_ACTION_SCHEMA,
        "track32_artifact_identity": artifacts,
        "training_lineage": lineage,
        "training_execution_contract": execution,
        "training_profile_identity": None,
        "transformer_identity": transformer_identity,
        "runtime_source_identity": provenance.runtime_source_identity,
        "checkpoint_invocation_identity": (provenance.checkpoint_invocation_identity),
    }

    metadata_path = checkpoint / "train_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            **common,
            "action_dim": 14,
            "track32_profile_id": agilex_profile_id(request.profile),
            "tactile_profile": request.profile,
            "run_role": request.train.run_role,
            "accelerator_profile": request.runtime.accelerator_profile,
            "training_profile_id": None,
            "track31_artifacts": None,
        }
    )
    _write_json(metadata_path, metadata)

    state_path = checkpoint / "training_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            **common,
            "schema_version": STRICT_CHECKPOINT_SCHEMA_VERSION,
            "step": request.train.stop_after_step,
            "data_batches_consumed": (
                request.train.stop_after_step
                * request.train.gradient_accumulation_steps
            ),
            "world_size": 1,
            "gradient_accumulation_steps": (request.train.gradient_accumulation_steps),
            "optimizer_state_format": OPTIMIZER_STATE_FORMAT,
        }
    )
    _write_json(state_path, state)

    previous_report = json.loads(
        (checkpoint / "action_migration_report.json").read_text(encoding="utf-8")
    )
    migration = {
        "source_action_dim": 20,
        "source_action_schema": "ee20_pi05",
        "target_action_dim": 14,
        "target_action_schema": AGILEX_ACTION_SCHEMA,
        "source_checkpoint_sha256": "a" * 64,
        "source_transformer_identity": previous_report["source_transformer_identity"],
        "compatibility": "migrate_action",
        "plan": {
            "copied_keys": ["backbone.weight"],
            "reset_keys": [
                "action_embedder.weight",
                "action_embedder.bias",
                "action_proj_out.weight",
                "action_proj_out.bias",
            ],
            "missing_target_keys": [],
            "unexpected_source_keys": [],
            "shape_mismatches": [],
        },
    }
    _write_json(checkpoint / "action_migration_report.json", migration)

    completion = {
        **common,
        "schema_version": STRICT_CHECKPOINT_SCHEMA_VERSION,
        "step": request.train.stop_after_step,
        "world_size": 1,
        "optimizer_state_format": OPTIMIZER_STATE_FORMAT,
        "optimizer_inventory_sha256": state["optimizer_inventory_sha256"],
        "runtime_signature": state["runtime_signature"],
        "status": "complete",
    }
    _write_json(checkpoint / "checkpoint_complete.json", completion)
    _reseal(checkpoint)
    return request, provenance, checkpoint


def test_completion_accepts_exact_physical_checkpoint(tmp_path: Path) -> None:
    request, provenance, checkpoint = _checkpoint_fixture(tmp_path)

    verified, identity = verify_completed_checkpoint(request, provenance)

    assert verified == checkpoint.resolve(strict=True)
    assert identity["step"] == request.train.stop_after_step


def _stage_b_request(
    request: AgileXTrainRequest,
    checkpoint: Path,
    *,
    parent: Path,
) -> AgileXTrainRequest:
    shutil.copytree(checkpoint, parent)
    parent_identity = audit_transformer_checkpoint(
        parent / "transformer" / "diffusion_pytorch_model.safetensors",
        expected_action_dim=14,
    )
    return replace(
        request,
        paths=replace(
            request.paths,
            init_from=parent,
            init_transformer_sha256=str(parent_identity["sha256"]),
        ),
    )


def test_completion_accepts_stage_b_parent_sha_distinct_from_migration_source(
    tmp_path: Path,
) -> None:
    request, provenance, checkpoint = _checkpoint_fixture(tmp_path)
    stage_b_request = _stage_b_request(
        request,
        checkpoint,
        parent=tmp_path / "stage-b-parent",
    )

    verified, identity = verify_completed_checkpoint(stage_b_request, provenance)

    assert verified == checkpoint.resolve(strict=True)
    assert identity["step"] == request.train.stop_after_step


def test_completion_rejects_stage_b_migration_drift_from_parent(
    tmp_path: Path,
) -> None:
    request, provenance, checkpoint = _checkpoint_fixture(tmp_path)
    stage_b_request = _stage_b_request(
        request,
        checkpoint,
        parent=tmp_path / "stage-b-parent",
    )
    report_path = checkpoint / "action_migration_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["plan"]["copied_keys"].append("another_backbone.weight")
    _write_json(report_path, report)
    _reseal(checkpoint)

    with pytest.raises(ValueError, match="differs from Stage-B parent"):
        verify_completed_checkpoint(stage_b_request, provenance)


def test_completion_resume_keeps_validated_migration_without_init_sha_check(
    tmp_path: Path,
) -> None:
    request, provenance, checkpoint = _checkpoint_fixture(tmp_path)
    resume_request = replace(
        request,
        paths=replace(
            request.paths,
            init_from=None,
            init_transformer_sha256=None,
            resume_from=tmp_path / "resume-parent",
            resume_checkpoint_identity_sha256="d" * 64,
        ),
    )

    verified, _identity = verify_completed_checkpoint(resume_request, provenance)

    assert verified == checkpoint.resolve(strict=True)


def test_completion_rejects_transformer_byte_tamper(tmp_path: Path) -> None:
    request, provenance, checkpoint = _checkpoint_fixture(tmp_path)
    _write_transformer_checkpoint(
        checkpoint,
        action_dim=14,
        action_schema=AGILEX_ACTION_SCHEMA,
        fill_value=1.0,
    )

    with pytest.raises(ValueError, match="transformer identity"):
        verify_completed_checkpoint(request, provenance)


def test_completion_rejects_optimizer_shard_tamper(tmp_path: Path) -> None:
    request, provenance, checkpoint = _checkpoint_fixture(tmp_path)
    (checkpoint / "optimizer_dcp" / "__0_0.distcp").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="DCP payload"):
        verify_completed_checkpoint(request, provenance)


def test_completion_rejects_snapshot_accumulation_drift(tmp_path: Path) -> None:
    request, provenance, checkpoint = _checkpoint_fixture(tmp_path)
    state_path = checkpoint / "training_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["gradient_accumulation_steps"] = 2
    state["data_batches_consumed"] = request.train.stop_after_step * 2
    _write_json(state_path, state)
    _reseal(checkpoint)

    with pytest.raises(ValueError, match="gradient accumulation"):
        verify_completed_checkpoint(request, provenance)


@pytest.mark.parametrize("field", ("batch_size", "num_steps", "max_latent_frames"))
def test_completion_rejects_formal_recipe_drift(
    tmp_path: Path,
    field: str,
) -> None:
    request, provenance, checkpoint = _checkpoint_fixture(tmp_path)
    for filename in (
        "train_meta.json",
        "training_state.json",
        "checkpoint_complete.json",
    ):
        path = checkpoint / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        contract = dict(payload["training_execution_contract"])
        contract[field] = int(contract[field]) + 1
        payload["training_execution_contract"] = contract
        _write_json(path, payload)
    _reseal(checkpoint)

    with pytest.raises(ValueError, match="training execution contract mismatch"):
        verify_completed_checkpoint(request, provenance)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", 20260812),
        ("save_interval", 301),
        ("val_interval", 101),
    ),
)
def test_completion_rejects_resume_recipe_drift(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    request, provenance, _checkpoint = _checkpoint_fixture(tmp_path)
    changed = replace(
        request,
        train=replace(request.train, **{field: value}),
    )

    with pytest.raises(ValueError, match="training lineage differs from request"):
        verify_completed_checkpoint(changed, provenance)


def test_completion_rejects_run_role_drift(tmp_path: Path) -> None:
    request, provenance, _checkpoint = _checkpoint_fixture(tmp_path)
    changed = replace(
        request,
        train=replace(request.train, run_role="final_refit"),
    )

    with pytest.raises(ValueError, match="metadata differs from request"):
        verify_completed_checkpoint(changed, provenance)
