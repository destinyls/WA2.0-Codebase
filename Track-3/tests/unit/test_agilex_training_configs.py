# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""AgileX qpos14 training and serving profile contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from easydict import EasyDict

from n0_twam.configs.twam_track3_agilex_base_cfg import (
    build_track3_agilex_base_config,
    load_agilex_repo_routes,
)
from n0_twam.configs.twam_track3_agilex_mixed_cfg import (
    build_track3_agilex_mixed_config,
)
from n0_twam.configs.twam_track3_agilex_server_cfg import (
    build_track3_agilex_server_config,
)
from n0_twam.configs.twam_track3_agilex_vision_only_cfg import (
    build_track3_agilex_vision_only_config,
)
from n0_twam.configs.twam_track3_agilex_vision_tactile_cfg import (
    build_track3_agilex_vision_tactile_config,
)
from n0_twam.configs import TWAM_CONFIGS
from n0_twam.embodiments import (
    AGILEX_ACTION_SCHEMA,
    AGILEX_EMBODIMENT_PROFILE_ID,
    build_agilex_repo_route_manifest,
    validate_action_route_contract,
    validate_embodiment_contract,
    validate_repo_route_manifest_contract,
)
from n0_twam.tactile_profiles import validate_tactile_profile_config

RGB_KEYS = [
    "observation.images.top",
    "observation.images.wrist_l",
    "observation.images.wrist_r",
]
TACTILE_KEYS = [
    "observation.images.tactile_l",
    "observation.images.tactile_r",
]
WRENCH_KEYS = [
    "observation.wrench.left",
    "observation.wrench.right",
]


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _route(*, tactile: bool) -> dict[str, object]:
    return {
        "embodiment": AGILEX_EMBODIMENT_PROFILE_ID,
        "action_schema": AGILEX_ACTION_SCHEMA,
        "rgb_keys": RGB_KEYS,
        "tactile_keys": TACTILE_KEYS if tactile else [],
        "wrench_keys": WRENCH_KEYS if tactile else [],
    }


def _vt_routes() -> dict[str, dict[str, object]]:
    return {
        "agilex_touch_a": _route(tactile=True),
        "agilex_touch_b": _route(tactile=True),
    }


def _mixed_routes() -> dict[str, dict[str, object]]:
    return {
        "agilex_touch": _route(tactile=True),
        "agilex_rgb": _route(tactile=False),
    }


def _vo_routes() -> dict[str, dict[str, object]]:
    return {"agilex_rgb": _route(tactile=False)}


def test_base_config_binds_native_qpos14_without_guessing_temporal_stride(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("N0_TRACK3_AGILEX_ACTIONS_PER_ANCHOR", raising=False)
    cfg = build_track3_agilex_base_config(repo_routes=_vo_routes())

    assert cfg.embodiment_profile == AGILEX_EMBODIMENT_PROFILE_ID
    assert cfg.action_schema == AGILEX_ACTION_SCHEMA
    assert cfg.dataset_adapter == "worldarena_agilex_qpos14"
    assert cfg.action_dim == 14
    assert cfg.state_dim == 14
    assert cfg.action_delta_mode == "none"
    assert cfg.used_action_channel_ids == list(range(14))
    assert cfg.inverse_used_action_channel_ids == list(range(14))
    assert cfg.action_per_frame is None
    assert cfg.temporal_alignment_contract["status"] == "import_safe_placeholder"
    assert cfg.temporal_alignment_is_placeholder is True
    assert validate_embodiment_contract(cfg.embodiment_contract) == (
        cfg.embodiment_contract
    )
    assert validate_action_route_contract(cfg.action_route_contract) == (
        cfg.action_route_contract
    )
    assert validate_repo_route_manifest_contract(cfg.repo_route_manifest) == (
        cfg.repo_route_manifest
    )


@pytest.mark.parametrize(
    ("transformer_config", "compatibility", "source_dim", "source_schema"),
    (
        (
            {"action_dim": 20, "action_schema": "ee20_pi05"},
            "migrate_action",
            20,
            "ee20_pi05",
        ),
        (
            {"action_dim": 14, "action_schema": AGILEX_ACTION_SCHEMA},
            "strict",
            14,
            AGILEX_ACTION_SCHEMA,
        ),
    ),
)
def test_weights_only_init_derives_compatibility_and_binds_request_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transformer_config: dict[str, object],
    compatibility: str,
    source_dim: int,
    source_schema: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    transformer = checkpoint / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(
        json.dumps(transformer_config), encoding="utf-8"
    )
    expected_sha256 = "a" * 64
    monkeypatch.setenv("N0_TRACK3_AGILEX_INIT_FROM", str(checkpoint))
    monkeypatch.setenv("N0_RELEASED_TRANSFORMER_SHA256", expected_sha256)

    cfg = build_track3_agilex_mixed_config(repo_routes=_mixed_routes())

    assert cfg.checkpoint_compatibility == compatibility
    assert cfg.checkpoint_source_action_dim == source_dim
    assert cfg.checkpoint_source_action_schema == source_schema
    assert cfg.expected_init_transformer_sha256 == expected_sha256
    assert cfg.strict_training_resume is False


@pytest.mark.parametrize(
    "transformer_config",
    (
        {"action_dim": 12, "action_schema": AGILEX_ACTION_SCHEMA},
        {"action_dim": 14, "action_schema": "ee20_pi05"},
        {"action_dim": 20, "action_schema": AGILEX_ACTION_SCHEMA},
    ),
)
def test_weights_only_init_rejects_unsupported_checkpoint_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transformer_config: dict[str, object],
) -> None:
    checkpoint = tmp_path / "checkpoint"
    transformer = checkpoint / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(
        json.dumps(transformer_config), encoding="utf-8"
    )
    monkeypatch.setenv("N0_TRACK3_AGILEX_INIT_FROM", str(checkpoint))

    with pytest.raises(ValueError, match="compatible 20D or qpos14"):
        build_track3_agilex_mixed_config(repo_routes=_mixed_routes())


def test_generic_route_manifest_only_applies_to_the_selected_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "routes.json"
    manifest.write_text(
        '{"routes":{"formal_touch":{"embodiment":"agilex_dual_qpos14_v1",'
        '"action_schema":"qpos14_joint_absolute_v1",'
        '"rgb_keys":["observation.images.top",'
        '"observation.images.wrist_l","observation.images.wrist_r"],'
        '"tactile_keys":["observation.images.tactile_l"],'
        '"wrench_keys":["observation.wrench.left"]}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("N0_TRACK3_AGILEX_TACTILE_PROFILE", "vision_tactile")
    monkeypatch.setenv("N0_TRACK3_AGILEX_REPO_ROUTE_MANIFEST", str(manifest))

    assert set(load_agilex_repo_routes("vision_tactile")) == {"formal_touch"}
    assert set(load_agilex_repo_routes("mixed")) == {
        "agilex_touch_repo",
        "agilex_rgb_repo",
    }


def test_generic_temporal_manifest_only_applies_to_selected_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    routes = _vo_routes()
    route_sha256 = build_agilex_repo_route_manifest(routes).contract_sha256
    raw = {
        "schema_version": 1,
        "status": "complete",
        "repo_route_manifest_sha256": route_sha256,
        "per_repo_bindings": {
            "agilex_rgb": {
                "action_offsets_per_anchor": [0, 1],
                "repo_route_identity": "a" * 64,
                "temporal_alignment_identity": "b" * 64,
            }
        },
    }
    temporal = {**raw, "contract_sha256": _canonical_sha256(raw)}
    temporal_path = tmp_path / "vision-only-temporal.json"
    temporal_path.write_text(json.dumps(temporal), encoding="utf-8")
    monkeypatch.setenv("N0_TRACK3_AGILEX_TACTILE_PROFILE", "vision_only")
    monkeypatch.setenv("N0_TRACK3_AGILEX_TEMPORAL_ALIGNMENT", str(temporal_path))

    selected = build_track3_agilex_vision_only_config(repo_routes=routes)
    unrelated = build_track3_agilex_mixed_config(repo_routes=_mixed_routes())

    assert selected.temporal_alignment_is_placeholder is False
    assert selected.action_per_frame == 2
    assert unrelated.temporal_alignment_is_placeholder is True
    assert unrelated.action_per_frame is None


def test_signed_repo_manifest_preserves_sensor_ids_and_original_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    routes = _vt_routes()
    signed = build_agilex_repo_route_manifest(
        routes,
        tactile_sensor_id_map={TACTILE_KEYS[0]: 7, TACTILE_KEYS[1]: 3},
        wrench_sensor_id_map={WRENCH_KEYS[0]: 9, WRENCH_KEYS[1]: 4},
    ).to_json_dict()
    manifest = tmp_path / "signed-routes.json"
    manifest.write_text(json.dumps(signed, indent=2), encoding="utf-8")
    source_file_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setenv(
        "N0_TRACK3_AGILEX_VISION_TACTILE_REPO_ROUTE_MANIFEST", str(manifest)
    )

    cfg = build_track3_agilex_vision_tactile_config()

    assert cfg.tactile_sensor_id_map == {
        TACTILE_KEYS[0]: 7,
        TACTILE_KEYS[1]: 3,
    }
    assert cfg.wrench_sensor_id_map == {
        WRENCH_KEYS[0]: 9,
        WRENCH_KEYS[1]: 4,
    }
    assert cfg.repo_route_manifest == signed
    assert cfg.repo_route_manifest_sha256 == signed["contract_sha256"]
    assert cfg.repo_route_manifest_source_file_sha256 == source_file_sha256
    assert cfg.repo_route_manifest_is_placeholder is False


def test_formal_temporal_binding_is_per_repo_self_hashed_and_runnable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    routes = _mixed_routes()
    signed_routes = build_agilex_repo_route_manifest(routes).to_json_dict()
    route_path = tmp_path / "signed-routes.json"
    route_path.write_text(json.dumps(signed_routes), encoding="utf-8")
    monkeypatch.setenv("N0_TRACK3_AGILEX_MIXED_REPO_ROUTE_MANIFEST", str(route_path))
    temporal_without_hash = {
        "schema_version": 1,
        "status": "complete",
        "repo_route_manifest_sha256": signed_routes["contract_sha256"],
        "per_repo_bindings": {
            repo: {
                "action_offsets_per_anchor": [0, 2, 5],
                "repo_route_identity": digest,
                "temporal_alignment_identity": temporal_digest,
            }
            for repo, digest, temporal_digest in (
                ("agilex_touch", "a" * 64, "b" * 64),
                ("agilex_rgb", "c" * 64, "d" * 64),
            )
        },
    }
    temporal = {
        **temporal_without_hash,
        "contract_sha256": _canonical_sha256(temporal_without_hash),
    }
    temporal_path = tmp_path / "temporal.json"
    temporal_path.write_text(json.dumps(temporal), encoding="utf-8")
    monkeypatch.setenv("N0_TRACK3_AGILEX_TACTILE_PROFILE", "mixed")
    monkeypatch.setenv("N0_TRACK3_AGILEX_TEMPORAL_ALIGNMENT", str(temporal_path))

    cfg = build_track3_agilex_mixed_config()

    assert cfg.temporal_alignment_contract == temporal
    assert cfg.temporal_alignment_contract_sha256 == temporal["contract_sha256"]
    assert cfg.temporal_alignment_is_placeholder is False
    assert cfg.action_per_frame == 3
    assert cfg.per_repo_action_offsets_per_anchor == {
        "agilex_rgb": [0, 2, 5],
        "agilex_touch": [0, 2, 5],
    }
    assert cfg.per_repo_route_identity == {
        "agilex_rgb": "c" * 64,
        "agilex_touch": "a" * 64,
    }
    assert cfg.per_repo_temporal_alignment_identity == {
        "agilex_rgb": "d" * 64,
        "agilex_touch": "b" * 64,
    }


def test_import_safe_temporal_placeholder_remains_content_addressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("N0_TRACK3_AGILEX_TEMPORAL_ALIGNMENT", raising=False)
    cfg = build_track3_agilex_base_config(repo_routes=_vo_routes())

    assert cfg.temporal_alignment_is_placeholder is True
    assert cfg.action_per_frame is None
    assert cfg.per_repo_action_offsets_per_anchor == {"agilex_rgb": []}
    assert set(cfg.per_repo_route_identity) == {"agilex_rgb"}
    assert set(cfg.per_repo_temporal_alignment_identity) == {"agilex_rgb"}
    assert len(cfg.temporal_alignment_contract_sha256) == 64
    payload = dict(cfg.temporal_alignment_contract)
    digest = payload.pop("contract_sha256")
    assert digest == _canonical_sha256(payload)


@pytest.mark.parametrize(
    ("builder", "routes", "profile"),
    [
        (build_track3_agilex_vision_tactile_config, _vt_routes, "vision_tactile"),
        (build_track3_agilex_mixed_config, _mixed_routes, "mixed"),
        (build_track3_agilex_vision_only_config, _vo_routes, "vision_only"),
    ],
)
def test_all_profiles_share_one_embodiment_and_qpos14_action_route(
    builder: Callable[..., EasyDict],
    routes: Callable[[], dict[str, dict[str, object]]],
    profile: str,
) -> None:
    cfg = builder(repo_routes=routes())

    assert cfg.tactile_profile == profile
    assert cfg.embodiment_profile == AGILEX_EMBODIMENT_PROFILE_ID
    assert cfg.action_schema == AGILEX_ACTION_SCHEMA
    assert cfg.action_dim == cfg.state_dim == 14
    assert cfg.server_action_output_format == "absolute"
    assert (
        validate_tactile_profile_config(
            cfg, repo_names=cfg.selected_repo_names
        ).to_json_dict()
        == cfg.tactile_profile_contract
    )
    assert cfg.instantiate_local_tactile is True
    assert cfg.instantiate_wrench_conditioner is True
    assert cfg.max_wrench_streams == 2
    assert cfg.wrench_arm_count == 2
    assert cfg.wrench_max_frames >= cfg.max_latent_frames
    assert set(cfg.initialized_target_only_prefixes) == {
        "local_tactile_",
        "agilex_wrench_",
    }
    contact = dict(cfg.contact_profile_contract)
    digest = contact.pop("contract_sha256")
    assert digest == cfg.contact_profile_contract_sha256
    assert digest == _canonical_sha256(contact)
    assert cfg.training_lineage["contact_profile_contract_sha256"] == digest
    assert cfg.training_lineage["temporal_alignment_contract_sha256"] == (
        cfg.temporal_alignment_contract_sha256
    )
    assert cfg.training_lineage["temporal_alignment_source_file_sha256"] == (
        cfg.temporal_alignment_source_file_sha256
    )
    assert cfg.training_lineage["normalizer_sha256"] == cfg.normalizer_sha256
    assert cfg.training_lineage["resume_recipe_contract"]["run_role"] == (cfg.run_role)
    assert cfg.training_lineage["resume_recipe_contract"]["seed"] == cfg.seed
    assert cfg.training_lineage["resume_recipe_contract"]["save_interval"] == (
        cfg.save_interval
    )
    assert cfg.training_lineage["resume_recipe_contract"]["val_interval"] == (
        cfg.val_interval
    )


def test_vision_tactile_requires_real_tactile_and_wrench_for_every_repo() -> None:
    cfg = build_track3_agilex_vision_tactile_config(repo_routes=_vt_routes())

    assert cfg.tactile_mode == "enabled"
    assert cfg.freeze_tactile_parameters is False
    assert cfg.tactile_cfg_prob == 0.0
    assert cfg.tactile_diffusion_loss_weight > 0.0
    assert all(cfg.per_repo_tactile_keys.values())
    assert all(cfg.per_repo_wrench_keys.values())
    assert cfg.instantiate_wrench_conditioner is True
    assert cfg.use_wrench_conditioner is True
    assert cfg.freeze_wrench_parameters is False
    assert cfg.contact_conditioning_bypass is False


def test_mixed_has_explicit_per_repo_routes_and_atomic_condition_drop() -> None:
    cfg = build_track3_agilex_mixed_config(repo_routes=_mixed_routes())

    assert set(cfg.per_repo_tactile_keys) == {"agilex_touch", "agilex_rgb"}
    assert cfg.per_repo_tactile_keys["agilex_touch"] == TACTILE_KEYS
    assert cfg.per_repo_tactile_keys["agilex_rgb"] == []
    assert cfg.per_repo_wrench_keys["agilex_touch"] == WRENCH_KEYS
    assert cfg.per_repo_wrench_keys["agilex_rgb"] == []
    assert 0.0 < cfg.tactile_cfg_prob < 1.0
    assert cfg.contact_cond_drop_strategy == "content_addressed_v1"
    assert cfg.contact_cond_drop_modalities == [
        "tactile",
        "local_tactile",
        "global_tactile",
        "wrench",
        "contact_gate",
    ]
    assert cfg.mask_contract["polarity"] == "true_is_valid_or_enabled"
    assert cfg.mask_contract["contact_cond_drop"] == "bool[B]"
    assert "tactile_cond_drop" not in cfg.mask_contract


def test_mixed_rejects_batch_size_above_one_until_drop_is_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("N0_TRACK3_AGILEX_TACTILE_PROFILE", "mixed")
    monkeypatch.setenv("N0_TRACK3_AGILEX_BATCH_SIZE", "2")

    with pytest.raises(ValueError, match="mixed.*batch_size=1"):
        build_track3_agilex_mixed_config(repo_routes=_mixed_routes())


def test_selected_non_mixed_profile_may_use_batch_size_above_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("N0_TRACK3_AGILEX_TACTILE_PROFILE", "vision_only")
    monkeypatch.setenv("N0_TRACK3_AGILEX_BATCH_SIZE", "2")

    cfg = build_track3_agilex_vision_only_config(repo_routes=_vo_routes())

    assert cfg.batch_size == 2


def test_vision_only_keeps_topology_but_freezes_and_bypasses_contact_modules() -> None:
    cfg = build_track3_agilex_vision_only_config(repo_routes=_vo_routes())

    assert cfg.tactile_mode == "disabled"
    assert cfg.tactile_keys == []
    assert cfg.wrench_keys == []
    assert cfg.freeze_tactile_parameters is True
    assert cfg.instantiate_local_tactile is True
    assert cfg.use_local_tactile is False
    assert cfg.bypass_local_tactile is True
    assert cfg.tactile_diffusion_loss_weight == 0.0
    assert cfg.instantiate_wrench_conditioner is True
    assert cfg.use_wrench_conditioner is False
    assert cfg.freeze_wrench_parameters is True
    assert cfg.contact_conditioning_bypass is True
    assert cfg.contact_cond_drop_fixed_value is True
    assert cfg.vision_only_ignore_valid_extra_tactile is True


def test_agilex_configs_are_registered_for_train_and_serve() -> None:
    assert {
        "track3_agilex_vision_tactile",
        "track3_agilex_mixed",
        "track3_agilex_vision_only",
        "track3_agilex_server",
    } <= set(TWAM_CONFIGS)


def test_profile_builders_reject_routes_incompatible_with_the_profile() -> None:
    with pytest.raises(ValueError, match="every repository"):
        build_track3_agilex_vision_tactile_config(repo_routes=_mixed_routes())

    with pytest.raises(ValueError, match="both tactile and no-tactile"):
        build_track3_agilex_mixed_config(repo_routes=_vt_routes())

    with pytest.raises(ValueError, match="cannot route tactile or wrench"):
        build_track3_agilex_vision_only_config(repo_routes=_vt_routes())


def test_server_config_preserves_profile_route_and_qpos14_wire_contract() -> None:
    cfg = build_track3_agilex_mixed_config(repo_routes=_mixed_routes())
    server = build_track3_agilex_server_config(training_config=cfg)

    assert server.infer_mode == "server"
    assert server.action_schema == AGILEX_ACTION_SCHEMA
    assert server.server_action_output_format == "absolute"
    assert server.server_return_action_channel_ids == list(range(14))
    assert server.embodiment_contract_sha256 == cfg.embodiment_contract_sha256
    assert server.action_route_contract_sha256 == cfg.action_route_contract_sha256
    assert server.repo_route_manifest_sha256 == cfg.repo_route_manifest_sha256
    assert server.tactile_profile_contract_sha256 == (
        cfg.tactile_profile_contract_sha256
    )
    assert server.require_signed_task_route is True
    assert server.policy_adapter == "worldarena_agilex_qpos14"


def test_selected_profile_binds_runner_metadata_and_exact_artifact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {
        "schema_version": 1,
        "embodiment_profile_id": AGILEX_EMBODIMENT_PROFILE_ID,
        "action_schema": AGILEX_ACTION_SCHEMA,
        "tactile_profile": "vision_only",
        "source_manifest_file_sha256": "1" * 64,
        "conversion_receipt_file_sha256": "2" * 64,
        "latent_inventory_file_sha256": "3" * 64,
        "repo_route_manifest_file_sha256": "4" * 64,
        "temporal_alignment_file_sha256": "5" * 64,
        "normalizer_file_sha256": "6" * 64,
    }
    monkeypatch.setenv("N0_TRACK3_AGILEX_TACTILE_PROFILE", "vision_only")
    monkeypatch.setenv("N0_TRACK3_AGILEX_PROFILE_ID", "agilex_track3_vision_only_v1")
    monkeypatch.setenv("N0_TRACK3_AGILEX_RUN_ROLE", "final_refit")
    monkeypatch.setenv("N0_TRACK3_AGILEX_ACCELERATOR_PROFILE", "hcu_performance")
    monkeypatch.setenv("N0_TRACK3_AGILEX_EXPECTED_WORLD_SIZE", "8")
    monkeypatch.setenv(
        "N0_TRACK3_AGILEX_ARTIFACT_IDENTITY_JSON",
        json.dumps(identity, sort_keys=True, separators=(",", ":")),
    )

    selected = build_track3_agilex_vision_only_config(repo_routes=_vo_routes())
    unrelated = build_track3_agilex_mixed_config(repo_routes=_mixed_routes())

    assert selected.run_role == "final_refit"
    assert selected.accelerator_profile == "hcu_performance"
    assert selected.expected_world_size == 8
    assert selected.track32_profile_id == "agilex_track3_vision_only_v1"
    assert selected.track32_artifact_identity == identity
    assert selected.capture_runtime_provenance is True
    assert unrelated.track32_artifact_identity is None


def test_selected_profile_rejects_incomplete_artifact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("N0_TRACK3_AGILEX_TACTILE_PROFILE", "vision_only")
    monkeypatch.setenv(
        "N0_TRACK3_AGILEX_ARTIFACT_IDENTITY_JSON",
        '{"schema_version":1}',
    )

    with pytest.raises(ValueError, match="artifact identity.*schema"):
        build_track3_agilex_vision_only_config(repo_routes=_vo_routes())
