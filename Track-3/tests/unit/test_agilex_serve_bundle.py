# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Immutable AgileX serve-bundle publication and re-audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from n0_twam.checkpointing.identity import audit_transformer_checkpoint
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.checkpointing.strict_resume import (
    build_sidecar_inventory,
    expected_sidecar_paths,
)
from n0_twam.configs.twam_track3_agilex_mixed_cfg import (
    build_track3_agilex_mixed_config,
)
from n0_twam.configs.twam_track3_agilex_vision_only_cfg import (
    build_track3_agilex_vision_only_config,
)
from n0_twam.configs.twam_track3_agilex_vision_tactile_cfg import (
    build_track3_agilex_vision_tactile_config,
)
from n0_twam.embodiments import build_agilex_repo_route_manifest
from n0_twam.integrations.worldarena.agilex_manifest import (
    canonical_sha256,
    sha256_file,
)
from n0_twam.integrations.worldarena.agilex_normalizer import (
    AgileXQpos14Normalizer,
)
from n0_twam.integrations.worldarena.agilex_serve_bundle import (
    build_agilex_serve_bundle,
    verify_agilex_serve_bundle,
)
from tests.unit.test_track31_checkpoint_review_regressions import _write_resume
from tests.unit.test_track31_preflight import _write_transformer_checkpoint

SOURCE_SHA = "1" * 64


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _routes(profile: str) -> dict[str, dict[str, object]]:
    rgb = [
        "observation.images.top",
        "observation.images.wrist_l",
        "observation.images.wrist_r",
    ]
    touch = {
        "embodiment": "agilex_dual_qpos14_v1",
        "action_schema": "qpos14_joint_absolute_v1",
        "rgb_keys": rgb,
        "tactile_keys": [
            "observation.images.tactile_l",
            "observation.images.tactile_r",
        ],
        "wrench_keys": [
            "observation.wrench.left",
            "observation.wrench.right",
        ],
    }
    vision = {
        "embodiment": "agilex_dual_qpos14_v1",
        "action_schema": "qpos14_joint_absolute_v1",
        "rgb_keys": rgb,
        "tactile_keys": [],
        "wrench_keys": [],
    }
    if profile == "vision_tactile":
        return {"official_touch": touch}
    if profile == "mixed":
        return {"official_touch": touch, "official_rgb": vision}
    return {"official_rgb": vision}


def _profile_config(profile: str, routes: dict[str, dict[str, object]]) -> object:
    builders = {
        "vision_tactile": build_track3_agilex_vision_tactile_config,
        "mixed": build_track3_agilex_mixed_config,
        "vision_only": build_track3_agilex_vision_only_config,
    }
    return builders[profile](repo_routes=routes)


def _rewrite_agilex_checkpoint(
    checkpoint: Path,
    *,
    profile: str,
    profile_contract: dict[str, object],
    contact_sha256: str,
    route_contract_sha256: str,
    route_file_sha256: str,
    normalizer_file_sha256: str,
) -> str:
    _write_resume(checkpoint)
    _write_transformer_checkpoint(
        checkpoint,
        action_dim=14,
        action_schema="qpos14_joint_absolute_v1",
    )
    transformer_config_path = checkpoint / "transformer" / "config.json"
    transformer_config = json.loads(transformer_config_path.read_text())
    transformer_config.update(
        {
            "tactile_profile": profile,
            "tactile_profile_contract_sha256": profile_contract["contract_sha256"],
        }
    )
    _write_json(transformer_config_path, transformer_config)
    transformer_identity = audit_transformer_checkpoint(
        checkpoint / "transformer" / "diffusion_pytorch_model.safetensors",
        expected_action_dim=14,
    )
    artifacts = {
        "schema_version": 1,
        "embodiment_profile_id": "agilex_dual_qpos14_v1",
        "action_schema": "qpos14_joint_absolute_v1",
        "tactile_profile": profile,
        "source_manifest_file_sha256": SOURCE_SHA,
        "conversion_receipt_file_sha256": "4" * 64,
        "latent_inventory_file_sha256": "5" * 64,
        "repo_route_manifest_file_sha256": route_file_sha256,
        "temporal_alignment_file_sha256": "6" * 64,
        "normalizer_file_sha256": normalizer_file_sha256,
    }
    lineage = {
        "schema_version": 1,
        "action_schema": "qpos14_joint_absolute_v1",
        "tactile_profile": profile,
        "contact_profile_contract_sha256": contact_sha256,
        "repo_route_manifest_sha256": route_contract_sha256,
        "normalizer_sha256": normalizer_file_sha256,
    }
    common = {
        "action_schema": "qpos14_joint_absolute_v1",
        "track32_artifact_identity": artifacts,
        "training_lineage": lineage,
        "tactile_profile_contract": profile_contract,
        "transformer_identity": transformer_identity,
    }
    for filename in ("train_meta.json", "training_state.json"):
        path = checkpoint / filename
        payload = json.loads(path.read_text())
        payload.update(common)
        if filename == "train_meta.json":
            payload.update(
                {
                    "action_dim": 14,
                    "track32_profile_id": f"agilex_track3_{profile}_v1",
                    "tactile_profile": profile,
                    "tactile_mode": (
                        "disabled" if profile == "vision_only" else "enabled"
                    ),
                    "training_profile_id": None,
                    "training_profile_identity": None,
                    "track31_artifacts": None,
                }
            )
        _write_json(path, payload)
    report_path = checkpoint / "action_migration_report.json"
    report = json.loads(report_path.read_text())
    report.update(
        {
            "schema_version": 1,
            "compatibility": "migrate_action",
            "target_action_dim": 14,
            "target_action_schema": "qpos14_joint_absolute_v1",
            "action_init_seed": 20260811,
            "source_checkpoint": "/immutable/released/base/transformer",
            "plan": {
                "copied_keys": ["backbone.weight"],
                "reset_keys": [
                    "action_embedder.bias",
                    "action_embedder.weight",
                    "action_proj_out.bias",
                    "action_proj_out.weight",
                ],
                "missing_target_keys": [],
                "unexpected_source_keys": [],
                "shape_mismatches": [],
            },
        }
    )
    _write_json(report_path, report)
    completion_path = checkpoint / "checkpoint_complete.json"
    completion = json.loads(completion_path.read_text())
    completion.update(common)
    completion["sidecar_inventory"] = build_sidecar_inventory(
        checkpoint,
        expected_sidecar_paths(1, include_action_migration=True),
    )
    _write_json(completion_path, completion)
    identity = build_strict_checkpoint_identity(
        capture_strict_checkpoint_snapshot(checkpoint)
    )
    return str(identity["identity_sha256"])


def _fixture(
    tmp_path: Path,
    profile: str,
    routes_override: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    routes = _routes(profile) if routes_override is None else routes_override
    cfg = _profile_config(profile, routes)
    route = build_agilex_repo_route_manifest(routes).to_json_dict()
    route_path = tmp_path / "repo_route_manifest.json"
    route_file_sha = _write_json(route_path, route)
    normalizer = AgileXQpos14Normalizer(
        q01=(-1.0,) * 14,
        q99=(1.0,) * 14,
        sample_count=64,
        source_manifest_sha256=SOURCE_SHA,
        repo_route_manifest_sha256=str(route["contract_sha256"]),
    )
    normalizer_path = tmp_path / "normalizer.json"
    normalizer_file_sha = _write_json(normalizer_path, normalizer.to_json_dict())
    base = tmp_path / "base"
    for component, filename in (
        ("vae", "config.json"),
        ("tokenizer", "tokenizer_config.json"),
        ("text_encoder", "config.json"),
        ("assets", "metadata.json"),
    ):
        _write_json(base / component / filename, {"component": component})
    checkpoint = tmp_path / "checkpoint"
    checkpoint_sha = _rewrite_agilex_checkpoint(
        checkpoint,
        profile=profile,
        profile_contract=dict(cfg.tactile_profile_contract),
        contact_sha256=str(cfg.contact_profile_contract_sha256),
        route_contract_sha256=str(route["contract_sha256"]),
        route_file_sha256=route_file_sha,
        normalizer_file_sha256=normalizer_file_sha,
    )
    contact = profile != "vision_only"
    task_core = {
        "task_id": "pick",
        "prompt": "pick the object",
        "tactile_required": contact,
        "wrench_required": contact,
        "tactile_keys": (
            ["observation.images.tactile_l", "observation.images.tactile_r"]
            if contact
            else []
        ),
        "wrench_keys": (
            ["observation.wrench.left", "observation.wrench.right"] if contact else []
        ),
    }
    task_routes = {
        "pick": {**task_core, "contract_sha256": canonical_sha256(task_core)}
    }
    safety_core = {
        "lower_bounds": [-2.0] * 14,
        "upper_bounds": [2.0] * 14,
        "max_step_per_second": [0.5] * 14,
        "min_execution_dt_s": 0.01,
        "max_execution_dt_s": 0.2,
        "max_state_age_s": 0.25,
        "max_inference_latency_s": 1.0,
    }
    safety = {**safety_core, "contract_sha256": canonical_sha256(safety_core)}
    return {
        "checkpoint": checkpoint,
        "checkpoint_identity_sha256": checkpoint_sha,
        "base_model": base,
        "normalizer": normalizer_path,
        "normalizer_file_sha256": normalizer_file_sha,
        "normalizer_contract_sha256": normalizer.contract_sha256,
        "source_manifest_sha256": SOURCE_SHA,
        "repo_route_manifest": route_path,
        "repo_route_manifest_file_sha256": route_file_sha,
        "repo_route_manifest_sha256": route["contract_sha256"],
        "tactile_profile": profile,
        "contact_profile_contract_sha256": cfg.contact_profile_contract_sha256,
        "task_routes": task_routes,
        "task_routes_sha256": canonical_sha256(task_routes),
        "safety_contract": safety,
        "safety_contract_sha256": safety["contract_sha256"],
        "output": tmp_path / "bundle",
    }


def _reseal_checkpoint(request: dict[str, object]) -> Path:
    checkpoint = request["checkpoint"]
    assert isinstance(checkpoint, Path)
    completion_path = checkpoint / "checkpoint_complete.json"
    completion = json.loads(completion_path.read_text())
    completion["sidecar_inventory"] = build_sidecar_inventory(
        checkpoint, expected_sidecar_paths(1, include_action_migration=True)
    )
    _write_json(completion_path, completion)
    request["checkpoint_identity_sha256"] = build_strict_checkpoint_identity(
        capture_strict_checkpoint_snapshot(checkpoint)
    )["identity_sha256"]
    return checkpoint


@pytest.mark.parametrize("profile", ["vision_tactile", "mixed", "vision_only"])
def test_build_and_verify_all_agilex_profiles(tmp_path: Path, profile: str) -> None:
    request = _fixture(tmp_path, profile)

    receipt = build_agilex_serve_bundle(**request)
    verified = verify_agilex_serve_bundle(
        request["output"],
        expected_receipt_file_sha256=receipt["receipt_file_sha256"],
    )

    assert verified["tactile_profile"] == profile
    assert verified["action_schema"] == "qpos14_joint_absolute_v1"
    assert verified["component_names"][-1] == "assets"


def test_builder_requires_new_disjoint_output(tmp_path: Path) -> None:
    request = _fixture(tmp_path, "vision_only")
    output = Path(request["output"])
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_agilex_serve_bundle(**request)

    request = _fixture(tmp_path / "nested", "vision_only")
    checkpoint = Path(request["checkpoint"])
    request["output"] = checkpoint / "bundle"
    with pytest.raises(ValueError, match="disjoint"):
        build_agilex_serve_bundle(**request)


def test_verifier_rejects_small_file_and_link_tampering(tmp_path: Path) -> None:
    request = _fixture(tmp_path, "mixed")
    receipt = build_agilex_serve_bundle(**request)
    output = Path(request["output"])

    normalizer_path = output / "normalizer.json"
    normalizer_path.chmod(0o644)
    normalizer_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="normalizer"):
        verify_agilex_serve_bundle(
            output,
            expected_receipt_file_sha256=receipt["receipt_file_sha256"],
        )

    request = _fixture(tmp_path / "retarget", "vision_only")
    receipt = build_agilex_serve_bundle(**request)
    output = Path(request["output"])
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (output / "transformer").unlink()
    (output / "transformer").symlink_to(replacement)
    with pytest.raises(ValueError, match="link changed"):
        verify_agilex_serve_bundle(
            output,
            expected_receipt_file_sha256=receipt["receipt_file_sha256"],
        )


def test_builder_rejects_wrong_action_dim_profile_and_placeholder(
    tmp_path: Path,
) -> None:
    request = _fixture(tmp_path, "vision_only")
    checkpoint = request["checkpoint"]
    assert isinstance(checkpoint, Path)
    config_path = checkpoint / "transformer" / "config.json"
    config = json.loads(config_path.read_text())
    config["action_dim"] = 8
    _write_json(config_path, config)
    _reseal_checkpoint(request)
    with pytest.raises(ValueError, match="action dimension"):
        build_agilex_serve_bundle(**request)

    request = _fixture(tmp_path / "profile", "mixed")
    request["tactile_profile"] = "vision_only"
    with pytest.raises(ValueError, match="profile"):
        build_agilex_serve_bundle(**request)

    request = _fixture(tmp_path / "placeholder", "vision_only")
    checkpoint = request["checkpoint"]
    assert isinstance(checkpoint, Path)
    for filename in ("train_meta.json", "training_state.json"):
        path = checkpoint / filename
        payload = json.loads(path.read_text())
        payload["track32_artifact_identity"] = None
        _write_json(path, payload)
    completion = json.loads((checkpoint / "checkpoint_complete.json").read_text())
    completion["track32_artifact_identity"] = None
    _write_json(checkpoint / "checkpoint_complete.json", completion)
    _reseal_checkpoint(request)
    with pytest.raises(ValueError, match="formal artifact identity"):
        build_agilex_serve_bundle(**request)
