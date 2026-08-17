# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""End-to-end contracts for the single-command AgileX training entrypoint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from n0_twam.checkpointing.runtime_provenance import LOCAL_EXECUTION_TIER
from n0_twam.actions.qpos14 import CHANNEL_NAMES, CHANNEL_UNITS, GRIPPER_ENCODING
from n0_twam.cli import run_cli
from n0_twam.configs.twam_track3_agilex_recipe import (
    build_agilex_resume_recipe_contract,
)
from n0_twam.embodiments import build_agilex_repo_route_manifest
from n0_twam.integrations.worldarena.agilex_artifacts import (
    build_agilex_conversion_receipt,
    build_agilex_latent_inventory,
)
from n0_twam.integrations.worldarena.agilex_manifest import (
    build_repo_route,
    canonical_sha256,
    sha256_file,
)
from n0_twam.integrations.worldarena.agilex_normalizer import (
    AgileXQpos14Normalizer,
)
from n0_twam.track31.local_provenance import LocalProvenance, package_import_root
from n0_twam.track32_agilex.preflight import (
    _validate_initialization,
    agilex_profile_id,
    audit_agilex_inputs,
    build_artifact_identity,
)
from n0_twam.track32_agilex.request import (
    agilex_train_request_template,
    load_agilex_train_request,
)
from n0_twam.track32_agilex.runner import (
    _claim_output_root,
    build_launch_plan,
    build_training_command,
    build_training_environment,
    run_agilex_training,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _fixture_request(tmp_path: Path) -> Path:
    raw_root = tmp_path / "raw"
    raw_file = raw_root / "touch" / "episode.bin"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(b"agilex-official-fixture")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    dataset = tmp_path / "lerobot"
    _write_json(dataset / "touch" / "meta" / "info.json", {"robot": "agilex"})
    (dataset / "touch" / "data" / "chunk-000").mkdir(parents=True)
    (dataset / "touch" / "data" / "chunk-000" / "episode.parquet").write_bytes(b"table")
    (dataset / "touch" / "videos" / "chunk-000").mkdir(parents=True)
    (dataset / "touch" / "videos" / "chunk-000" / "episode.mp4").write_bytes(b"video")
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    empty = tmp_path / "empty_emb.pt"
    empty.write_bytes(b"empty")
    init_root = tmp_path / "released"
    transformer = init_root / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"fixture")
    _write_json(transformer / "config.json", {"action_dim": 20})

    temporal_identity = "b" * 64
    source_route_payload = {
        "embodiment": "agilex_dual_qpos14_v1",
        "action_schema": "qpos14_joint_absolute_v1",
        "action_label_source": "executed",
        "formal": True,
        "rgb_keys": [
            "observation.images.top",
            "observation.images.wrist_l",
            "observation.images.wrist_r",
        ],
        "tactile_keys": ["observation.images.tactile_l"],
        "wrench_keys": ["observation.wrench.left"],
        "channel_names": list(CHANNEL_NAMES),
        "channel_units": list(CHANNEL_UNITS),
        "gripper_encoding": GRIPPER_ENCODING,
        "temporal_alignment_identity": temporal_identity,
    }
    source_route = build_repo_route("touch", source_route_payload)
    records = [
        {
            "path": "touch/episode.bin",
            "size": raw_file.stat().st_size,
            "sha256": sha256_file(raw_file),
        }
    ]
    source_manifest = _write_json(
        artifacts / "source.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "dataset_id": "agilex-official-fixture",
            "revision": "fixture-v1",
            "records": records,
            "canonical_records_sha256": canonical_sha256(records),
            "repos": {"touch": source_route_payload},
        },
    )
    simple_routes = {
        "touch": {
            "embodiment": "agilex_dual_qpos14_v1",
            "action_schema": "qpos14_joint_absolute_v1",
            "rgb_keys": [
                "observation.images.top",
                "observation.images.wrist_l",
                "observation.images.wrist_r",
            ],
            "tactile_keys": ["observation.images.tactile_l"],
            "wrench_keys": ["observation.wrench.left"],
        }
    }
    route_contract = build_agilex_repo_route_manifest(simple_routes).to_json_dict()
    route_file = _write_json(artifacts / "routes.json", route_contract)
    temporal_core = {
        "schema_version": 1,
        "status": "complete",
        "repo_route_manifest_sha256": route_contract["contract_sha256"],
        "per_repo_bindings": {
            "touch": {
                "action_offsets_per_anchor": [0, 1, 2],
                "repo_route_identity": source_route.route_identity,
                "temporal_alignment_identity": temporal_identity,
            }
        },
    }
    temporal_payload = {
        **temporal_core,
        "contract_sha256": hashlib.sha256(
            json.dumps(
                temporal_core,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    temporal_file = _write_json(artifacts / "temporal.json", temporal_payload)
    source_sha = sha256_file(source_manifest)
    normalizer = AgileXQpos14Normalizer(
        q01=tuple([-1.0] * 14),
        q99=tuple([1.0] * 14),
        sample_count=10,
        source_manifest_sha256=source_sha,
        repo_route_manifest_sha256=str(route_contract["contract_sha256"]),
    )
    normalizer_file = _write_json(
        artifacts / "normalizer.json", normalizer.to_json_dict()
    )
    video_latent = (
        dataset / "touch" / "latents" / "chunk-000" / "observation.images.top"
    )
    video_latent.mkdir(parents=True)
    (video_latent / "episode_000000_0_4.pth").write_bytes(b"video-latent")
    tactile_latent = (
        dataset
        / "touch"
        / "latents_tactile"
        / "global"
        / "chunk-000"
        / "observation.images.tactile_l"
    )
    tactile_latent.mkdir(parents=True)
    (tactile_latent / "episode_000000_0_4.pth").write_bytes(b"touch-latent")
    conversion_payload = build_agilex_conversion_receipt(
        dataset_root=dataset,
        routes=(source_route,),
        source_manifest_sha256=source_sha,
        repo_route_manifest_sha256=str(route_contract["contract_sha256"]),
        temporal_alignment_contract_sha256=str(temporal_payload["contract_sha256"]),
    )
    conversion = _write_json(artifacts / "conversion.json", conversion_payload)
    latent_payload = build_agilex_latent_inventory(
        dataset_root=dataset,
        routes=(source_route,),
        conversion_receipt=conversion_payload,
        conversion_receipt_sha256=sha256_file(conversion),
    )
    latent = _write_json(
        artifacts / "latent.json",
        latent_payload,
    )

    payload = agilex_train_request_template()
    payload["run_id"] = "agilex-fixture"
    payload["profile"] = "vision_tactile"
    paths = payload["paths"]
    assert isinstance(paths, dict)
    paths.update(
        {
            "source_root": str(raw_root),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": source_sha,
            "dataset_root": str(dataset),
            "artifact_root": str(artifacts),
            "conversion_receipt": str(conversion),
            "conversion_receipt_sha256": sha256_file(conversion),
            "latent_inventory": str(latent),
            "latent_inventory_sha256": sha256_file(latent),
            "repo_route_manifest": str(route_file),
            "repo_route_manifest_sha256": sha256_file(route_file),
            "temporal_alignment": str(temporal_file),
            "temporal_alignment_sha256": sha256_file(temporal_file),
            "normalizer": str(normalizer_file),
            "normalizer_sha256": sha256_file(normalizer_file),
            "base_model": str(base_model),
            "empty_embedding": str(empty),
            "empty_embedding_sha256": sha256_file(empty),
            "init_from": str(init_root),
            "init_transformer_sha256": "a" * 64,
            "resume_from": None,
            "resume_checkpoint_identity_sha256": None,
            "output_root": str(tmp_path / "new-output"),
        }
    )
    return _write_json(tmp_path / "agilex.train.json", payload)


def _provenance(tmp_path: Path) -> LocalProvenance:
    return LocalProvenance(
        invocation_id="agilex-fixture",
        code_manifest_path=tmp_path / "code.json",
        environment_manifest_path=tmp_path / "env.json",
        launch_receipt_path=tmp_path / "launch.json",
        runtime_source_identity={
            "schema_version": 2,
            "execution_tier": LOCAL_EXECUTION_TIER,
            "code_manifest_sha256": "1" * 64,
            "environment_manifest_sha256": "2" * 64,
            "empty_embedding_sha256": sha256_file(tmp_path / "empty_emb.pt"),
            "package_version": "0.1.0",
        },
        checkpoint_invocation_identity={
            "schema_version": 2,
            "execution_tier": LOCAL_EXECUTION_TIER,
            "invocation_id": "agilex-fixture",
            "launch_receipt_sha256": "3" * 64,
        },
    )


def test_request_plan_and_command_select_exact_agilex_profile(tmp_path: Path) -> None:
    request = load_agilex_train_request(_fixture_request(tmp_path))
    plan = build_launch_plan(request)

    assert plan["profile"] == "agilex_track3_vision_tactile_v1"
    assert plan["world_size"] == 8
    assert plan["model_contract"]["model_action_schema"] == ("qpos14_joint_absolute_v1")
    command = build_training_command(request)
    assert command[-1] == "track3_agilex_vision_tactile"


def test_environment_is_clean_and_maps_every_formal_binding(tmp_path: Path) -> None:
    request = load_agilex_train_request(_fixture_request(tmp_path))
    environment = build_training_environment(
        request,
        _provenance(tmp_path),
        environ={
            "PATH": "/usr/bin",
            "PYTHONPATH": "/untrusted",
            "LD_PRELOAD": "/untrusted.so",
            "N0_ROGUE": "1",
            "CUDA_VISIBLE_DEVICES": "7",
        },
    )

    assert environment["PATH"] == "/usr/bin"
    assert environment["PYTHONPATH"] == str(package_import_root())
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert environment["N0_TRACK3_AGILEX_TACTILE_PROFILE"] == "vision_tactile"
    assert environment["N0_TRACK3_AGILEX_REQUEST_SHA256"] == request.source_sha256
    assert environment["N0_TRACK3_AGILEX_REPO_ROUTE_MANIFEST"].endswith("routes.json")
    assert environment["N0_RELEASED_TRANSFORMER_SHA256"] == "a" * 64
    assert "N0_ROGUE" not in environment
    assert "LD_PRELOAD" not in environment


def test_runner_rejects_request_drift_after_child_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _fixture_request(tmp_path)
    request = load_agilex_train_request(request_path)
    assert request.source_sha256 == sha256_file(request_path)

    def mutate_during_preflight(*_args: object, **_kwargs: object) -> str:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        payload["train"]["seed"] += 1
        _write_json(request_path, payload)
        return json.dumps({"status": "pass", "request_sha256": request.source_sha256})

    monkeypatch.setattr(
        "n0_twam.track32_agilex.runner._run_capture", mutate_during_preflight
    )
    with pytest.raises(ValueError, match="request changed after it was loaded"):
        run_agilex_training(request)
    assert not request.paths.output_root.exists()


def test_preflight_audits_records_routes_receipts_and_init20(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = load_agilex_train_request(_fixture_request(tmp_path))
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.audit_transformer_checkpoint",
        lambda *_args, **_kwargs: {"sha256": "a" * 64, "action_dim": 20},
    )

    audit = audit_agilex_inputs(request)

    assert audit["checkpoint"]["mode"] == "init20_migrate_qpos14"
    assert audit["artifact_identity"]["action_schema"] == ("qpos14_joint_absolute_v1")
    assert audit["training_lineage"]["tactile_profile"] == "vision_tactile"
    assert audit["training_lineage"]["resume_recipe_contract"]["seed"] == (
        request.train.seed
    )
    assert audit["training_lineage"]["resume_recipe_contract"]["run_role"] == (
        request.train.run_role
    )

    source_record = tmp_path / "raw" / "touch" / "episode.bin"
    source_record.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source identity mismatch"):
        audit_agilex_inputs(request)


def _qpos14_stage_b_snapshot(
    request: object,
    *,
    profile_id: str | None = None,
    run_role: str | None = None,
    artifact_identity: object | None = None,
) -> SimpleNamespace:
    expected_artifacts = build_artifact_identity(request)  # type: ignore[arg-type]
    selected_artifacts = (
        expected_artifacts if artifact_identity is None else artifact_identity
    )
    common = {
        "track32_artifact_identity": selected_artifacts,
        "training_lineage": {
            "resume_recipe_contract": {
                "run_role": "parent-role-is-not-used-for-recipe-resume",
                "seed": -1,
                "save_interval": 1,
                "val_interval": 1,
            }
        },
    }
    source_identity = {
        "schema_version": 1,
        "file_name": "diffusion_pytorch_model.safetensors",
        "size_bytes": 1,
        "sha256": "e" * 64,
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
    return SimpleNamespace(
        world_size=4,
        transformer_config={
            "action_dim": 14,
            "action_schema": "qpos14_joint_absolute_v1",
        },
        train_meta={
            **common,
            "track32_profile_id": (
                agilex_profile_id(request.profile)  # type: ignore[attr-defined]
                if profile_id is None
                else profile_id
            ),
            "run_role": (
                request.train.run_role  # type: ignore[attr-defined]
                if run_role is None
                else run_role
            ),
        },
        training_state=dict(common),
        completion=dict(common),
        action_migration_report={
            "source_action_dim": 20,
            "source_action_schema": "ee20_pi05",
            "target_action_dim": 14,
            "target_action_schema": "qpos14_joint_absolute_v1",
            "source_checkpoint_sha256": "e" * 64,
            "source_transformer_identity": source_identity,
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
        },
    )


def test_complete_qpos14_parent_is_stage_b_weights_only_without_resume_equality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = load_agilex_train_request(_fixture_request(tmp_path))
    _write_json(
        request.paths.init_from / "transformer" / "config.json",  # type: ignore[operator]
        {"action_dim": 14, "action_schema": "qpos14_joint_absolute_v1"},
    )
    observed_action_dims: list[int] = []

    def audit_transformer(*_args: object, **kwargs: object) -> dict[str, object]:
        observed_action_dims.append(int(kwargs["expected_action_dim"]))
        return {"sha256": "a" * 64, "action_dim": 14}

    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.audit_transformer_checkpoint",
        audit_transformer,
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.capture_strict_checkpoint_snapshot",
        lambda *_args, **_kwargs: _qpos14_stage_b_snapshot(request),
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.build_strict_checkpoint_identity",
        lambda *_args, **_kwargs: {"identity_sha256": "d" * 64},
    )

    checkpoint = audit_agilex_inputs(request)["checkpoint"]

    assert checkpoint["mode"] == "init14_weights_only_stage_b"
    assert checkpoint["transformer_identity"]["sha256"] == "a" * 64
    assert checkpoint["checkpoint_identity"]["identity_sha256"] == "d" * 64
    assert observed_action_dims == [14]
    assert not request.paths.output_root.exists()


@pytest.mark.parametrize("migration_state", ("missing", "tampered"))
def test_qpos14_stage_b_rejects_invalid_parent_migration_before_output_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    migration_state: str,
) -> None:
    request = load_agilex_train_request(_fixture_request(tmp_path))
    _write_json(
        request.paths.init_from / "transformer" / "config.json",  # type: ignore[operator]
        {"action_dim": 14, "action_schema": "qpos14_joint_absolute_v1"},
    )
    snapshot = _qpos14_stage_b_snapshot(request)
    if migration_state == "missing":
        snapshot.action_migration_report = None
    else:
        snapshot.action_migration_report["target_action_schema"] = "wrong"
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.audit_transformer_checkpoint",
        lambda *_args, **_kwargs: {"sha256": "a" * 64, "action_dim": 14},
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.capture_strict_checkpoint_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.build_strict_checkpoint_identity",
        lambda *_args, **_kwargs: {"identity_sha256": "d" * 64},
    )

    with pytest.raises(ValueError, match="migration"):
        _validate_initialization(request)
    assert not request.paths.output_root.exists()


@pytest.mark.parametrize(
    ("snapshot_changes", "message"),
    (
        ({"profile_id": "agilex_track3_vision_only_v1"}, "profile differs"),
        ({"run_role": "final_refit"}, "run_role differs"),
        ({"artifact_identity": {"wrong": "artifact"}}, "artifact identity differs"),
    ),
)
def test_qpos14_stage_b_rejects_parent_identity_drift_before_output_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_changes: dict[str, object],
    message: str,
) -> None:
    request = load_agilex_train_request(_fixture_request(tmp_path))
    _write_json(
        request.paths.init_from / "transformer" / "config.json",  # type: ignore[operator]
        {"action_dim": 14, "action_schema": "qpos14_joint_absolute_v1"},
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.audit_transformer_checkpoint",
        lambda *_args, **_kwargs: {"sha256": "a" * 64, "action_dim": 14},
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.capture_strict_checkpoint_snapshot",
        lambda *_args, **_kwargs: _qpos14_stage_b_snapshot(request, **snapshot_changes),
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.build_strict_checkpoint_identity",
        lambda *_args, **_kwargs: {"identity_sha256": "d" * 64},
    )

    with pytest.raises(ValueError, match=message):
        _validate_initialization(request)
    assert not request.paths.output_root.exists()


@pytest.mark.parametrize(
    "transformer_config",
    (
        {"action_dim": 12, "action_schema": "qpos14_joint_absolute_v1"},
        {"action_dim": 14, "action_schema": "ee20_pi05"},
        {"action_dim": 20, "action_schema": "qpos14_joint_absolute_v1"},
    ),
)
def test_init_rejects_unsupported_action_dimension_schema_pairs(
    tmp_path: Path,
    transformer_config: dict[str, object],
) -> None:
    request = load_agilex_train_request(_fixture_request(tmp_path))
    _write_json(
        request.paths.init_from / "transformer" / "config.json",  # type: ignore[operator]
        transformer_config,
    )

    with pytest.raises(ValueError, match="compatible 20D or qpos14"):
        _validate_initialization(request)
    assert not request.paths.output_root.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("seed", 20260812, "recipe differs from request"),
        ("run_role", "final_refit", "run_role differs from request"),
    ),
)
def test_resume_preflight_rejects_recipe_drift_before_output_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    initial = load_agilex_train_request(_fixture_request(tmp_path))
    resume_identity = "d" * 64
    resume = replace(
        initial,
        paths=replace(
            initial.paths,
            init_from=None,
            init_transformer_sha256=None,
            resume_from=tmp_path / "checkpoint-step-20",
            resume_checkpoint_identity_sha256=resume_identity,
        ),
    )
    artifact_identity = build_artifact_identity(resume)
    lineage = {
        "resume_recipe_contract": build_agilex_resume_recipe_contract(resume.train)
    }
    common = {
        "track32_artifact_identity": artifact_identity,
        "training_lineage": lineage,
    }
    snapshot = SimpleNamespace(
        world_size=len(resume.runtime.devices),
        transformer_config={"action_schema": "qpos14_joint_absolute_v1"},
        train_meta={
            **common,
            "track32_profile_id": agilex_profile_id(resume.profile),
            "run_role": resume.train.run_role,
        },
        training_state=dict(common),
        completion=dict(common),
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.capture_strict_checkpoint_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.build_strict_checkpoint_identity",
        lambda *_args, **_kwargs: {"identity_sha256": resume_identity},
    )
    changed = replace(resume, train=replace(resume.train, **{field: value}))

    with pytest.raises(ValueError, match=message):
        _validate_initialization(changed)
    assert not changed.paths.output_root.exists()


def test_preflight_rehashes_live_lerobot_and_latent_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "n0_twam.track32_agilex.preflight.audit_transformer_checkpoint",
        lambda *_args, **_kwargs: {"sha256": "a" * 64, "action_dim": 20},
    )
    request = load_agilex_train_request(_fixture_request(tmp_path / "table"))
    table = (
        request.paths.dataset_root / "touch" / "data" / "chunk-000" / "episode.parquet"
    )
    table.write_bytes(b"tampered-table")
    with pytest.raises(ValueError, match="live data/routes"):
        audit_agilex_inputs(request)

    request = load_agilex_train_request(_fixture_request(tmp_path / "latent"))
    latent = next((request.paths.dataset_root / "touch" / "latents").rglob("*.pth"))
    latent.write_bytes(b"tampered-latent")
    with pytest.raises(ValueError, match="live payload bytes"):
        audit_agilex_inputs(request)


def test_cli_exposes_agilex_template_and_dry_run(tmp_path: Path) -> None:
    template = run_cli(["track32", "agilex-template"])
    assert template["template"]["profile"] == "mixed"

    request_path = _fixture_request(tmp_path)
    result = run_cli(
        ["track32", "agilex-train", "--config", str(request_path), "--dry-run"]
    )
    assert result["status"] == "dry_run"
    assert result["plan"]["training_command"][-1] == ("track3_agilex_vision_tactile")


def test_request_rejects_unknown_profile_and_output_overlap(tmp_path: Path) -> None:
    request_path = _fixture_request(tmp_path)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    payload["profile"] = "automatic"
    _write_json(request_path, payload)
    with pytest.raises(ValueError, match="vision_tactile, mixed, or vision_only"):
        load_agilex_train_request(request_path)

    payload["profile"] = "vision_tactile"
    payload["paths"]["output_root"] = payload["paths"]["dataset_root"]
    _write_json(request_path, payload)
    with pytest.raises(ValueError, match="disjoint"):
        load_agilex_train_request(request_path)

    payload["paths"]["output_root"] = str(tmp_path / "new-output")
    payload["profile"] = "mixed"
    payload["train"]["batch_size"] = 2
    _write_json(request_path, payload)
    with pytest.raises(ValueError, match="mixed AgileX training requires batch_size=1"):
        load_agilex_train_request(request_path)


def test_request_rejects_input_and_output_leaf_symlinks(tmp_path: Path) -> None:
    request_path = _fixture_request(tmp_path)
    payload = json.loads(request_path.read_text(encoding="utf-8"))

    real_normalizer = Path(payload["paths"]["normalizer"])
    normalizer_link = tmp_path / "normalizer-link.json"
    normalizer_link.symlink_to(real_normalizer)
    payload["paths"]["normalizer"] = str(normalizer_link)
    _write_json(request_path, payload)
    with pytest.raises(ValueError, match="normalizer must not be a symbolic link"):
        load_agilex_train_request(request_path)

    payload["paths"]["normalizer"] = str(real_normalizer)
    dangling_output = tmp_path / "new-output-link"
    dangling_output.symlink_to(tmp_path / "not-created-output")
    payload["paths"]["output_root"] = str(dangling_output)
    _write_json(request_path, payload)
    with pytest.raises(ValueError, match="output_root must not be a symbolic link"):
        load_agilex_train_request(request_path)


def test_output_lineage_is_claimed_atomically(tmp_path: Path) -> None:
    output_root = tmp_path / "nested" / "new-output"

    assert _claim_output_root(output_root) == output_root.resolve(strict=True)
    with pytest.raises(FileExistsError):
        _claim_output_root(output_root)


def test_cli_exposes_agilex_serve_bundle_as_one_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_routes = _write_json(tmp_path / "tasks.json", {"wipe": {"task_id": "wipe"}})
    safety = _write_json(tmp_path / "safety.json", {"contract_sha256": "a" * 64})
    captured: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "complete", "bundle_root": str(kwargs["output"])}

    monkeypatch.setattr(
        "n0_twam.integrations.worldarena.agilex_serve_bundle."
        "build_agilex_serve_bundle",
        fake_builder,
    )
    digest = "a" * 64
    result = run_cli(
        [
            "track32",
            "agilex-serve-bundle",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--checkpoint-identity-sha256",
            digest,
            "--base-model",
            str(tmp_path / "base"),
            "--normalizer",
            str(tmp_path / "normalizer.json"),
            "--normalizer-file-sha256",
            digest,
            "--normalizer-contract-sha256",
            digest,
            "--source-manifest-sha256",
            digest,
            "--repo-route-manifest",
            str(tmp_path / "routes.json"),
            "--repo-route-manifest-file-sha256",
            digest,
            "--repo-route-manifest-sha256",
            digest,
            "--tactile-profile",
            "mixed",
            "--contact-profile-contract-sha256",
            digest,
            "--task-routes",
            str(task_routes),
            "--task-routes-sha256",
            digest,
            "--safety-contract",
            str(safety),
            "--safety-contract-sha256",
            digest,
            "--output",
            str(tmp_path / "bundle"),
        ]
    )

    assert result["status"] == "complete"
    assert captured["tactile_profile"] == "mixed"
    assert captured["task_routes"] == {"wipe": {"task_id": "wipe"}}
