# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from n0_twam.configs.twam_base_cfg import twam_base_cfg
from n0_twam.data.track31_training_identity import build_track31_profile_identity
from n0_twam.tactile_profiles import (
    MIXED,
    VISION_ONLY,
    VISION_TACTILE,
    legacy_checkpoint_matches_tactile_profile,
    resolve_absent_tactile_inference_route,
    resolve_repo_tactile_keys,
    validate_serving_tactile_binding,
    validate_tactile_profile_config,
    validate_tactile_profile_contract,
)


def test_registered_base_config_is_a_valid_vision_tactile_profile() -> None:
    contract = validate_tactile_profile_config(twam_base_cfg)

    assert contract.profile == VISION_TACTILE


def _config(profile: str) -> SimpleNamespace:
    common = {
        "tactile_profile": profile,
        "tactile_optional": False,
        "synthetic_tactile_data": False,
        "max_tactile_streams": 4,
        "use_contact_gate": False,
        "tactile_global_zero": False,
        "noisy_cond_prob_tactile": 0.0,
    }
    if profile == VISION_ONLY:
        return SimpleNamespace(
            **common,
            tactile_mode="disabled",
            freeze_tactile_parameters=True,
            tactile_keys=[],
            per_repo_tactile_keys={},
            tactile_sensor_id_map={},
            active_tactile_sensor_count=0,
            use_local_tactile=False,
            local_tactile_mode="current",
            tactile_cfg_prob=0.0,
            tactile_diffusion_loss_weight=0.0,
        )
    repo_map = {"touch": ["tactile_a"], "rgb": []} if profile == MIXED else {}
    return SimpleNamespace(
        **common,
        tactile_mode="enabled",
        freeze_tactile_parameters=False,
        tactile_keys=["tactile_a"],
        per_repo_tactile_keys=repo_map,
        tactile_sensor_id_map={"tactile_a": 0},
        active_tactile_sensor_count=1,
        use_local_tactile=True,
        local_tactile_mode="current",
        tactile_cfg_prob=0.1 if profile == MIXED else 0.0,
        tactile_diffusion_loss_weight=1.0,
    )


def _serve_semantics(config: SimpleNamespace) -> dict[str, object]:
    return {
        "live_tactile_mode": config.tactile_mode,
        "live_max_tactile_streams": config.max_tactile_streams,
        "live_use_local_tactile": config.use_local_tactile,
        "live_local_tactile_mode": config.local_tactile_mode,
        "live_use_contact_gate": config.use_contact_gate,
        "live_tactile_global_zero": config.tactile_global_zero,
    }


def _legacy_trainability(*, vision_only: bool = False) -> dict[str, object]:
    names = ["mot.experts.tactile.weight"] if vision_only else []
    canonical = {
        "schema_version": 1,
        "policy": "freeze_tactile_only_v1" if vision_only else "train_all_v1",
        "tactile_mode": "disabled" if vision_only else "enabled",
        "frozen_parameter_names": names,
        "frozen_parameter_count": len(names),
        "frozen_numel": 16 if vision_only else 0,
        "trainable_parameter_count": 4,
        "trainable_numel": 64,
    }
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {**canonical, "contract_sha256": digest}


def _legacy_track31_metadata() -> dict[str, object]:
    sha = "a" * 64
    profile = build_track31_profile_identity(
        {
            "training_profile_id": "multitask_pretrain_v1",
            "run_role": "development",
            "train_view_id": "stage_a_dev719_v1",
            "sampler_coverage_mode": "pad_global",
            "normalizer_source_view_id": "stage_a_dev719_v1",
            "validation_view_id": "internal_dev40_v1",
            "source_manifest_sha256": sha,
            "normalizer_sha256": sha,
            "train_view_sha256": sha,
            "validation_view_sha256": sha,
            "normalizer_source_view_sha256": sha,
            "video_inventory_sha256": sha,
            "tactile_inventory_sha256": sha,
        }
    )
    assert profile is not None
    artifacts = {
        "manifest_sha256": sha,
        "normalizer_sha256": sha,
        "train_episode_count": 719,
        "validation_episode_count": 40,
        "normalizer_sample_count": 719,
        "action_q01": [0.0] * 8,
        "action_q99": [1.0] * 8,
        "conversion_report_sha256": sha,
        "train_view_id": "stage_a_dev719_v1",
        "train_view_sha256": sha,
        "validation_view_id": "internal_dev40_v1",
        "validation_view_sha256": sha,
        "normalizer_source_view_id": "stage_a_dev719_v1",
        "normalizer_source_view_sha256": sha,
        "parent_validation_view_id": None,
        "parent_validation_view_sha256": None,
        "video_inventory_sha256": sha,
        "tactile_inventory_sha256": sha,
        "latent_segment_count": 719,
        "video_latent_artifact_count": 1438,
        "tactile_latent_artifact_count": 2876,
    }
    return {
        "training_profile_id": "multitask_pretrain_v1",
        "run_role": "development",
        "train_view_id": "stage_a_dev719_v1",
        "validation_view_id": "internal_dev40_v1",
        "training_profile_identity": profile,
        "track31_artifacts": artifacts,
        "tactile_mode": "enabled",
        "tactile_keys": ["tactile_a"],
        "local_tactile_mode": "current",
        "tactile_global_zero": False,
        "video_inventory_sha256": sha,
        "tactile_inventory_sha256": sha,
        "trainability_contract": _legacy_trainability(),
    }


@pytest.mark.parametrize(
    ("profile", "repositories"),
    [
        (VISION_TACTILE, ("touch_a", "touch_b")),
        (MIXED, ("touch", "rgb")),
        (VISION_ONLY, ("rgb_a", "rgb_b")),
    ],
)
def test_profiles_build_stable_contract_without_mutating_config(
    profile: str,
    repositories: tuple[str, ...],
) -> None:
    config = _config(profile)
    before = deepcopy(vars(config))

    contract = validate_tactile_profile_config(
        config, repo_names=repositories
    ).to_json_dict()

    assert contract["profile"] == profile
    assert len(contract["contract_sha256"]) == 64
    assert validate_tactile_profile_contract(contract) == contract
    assert vars(config) == before


@pytest.mark.parametrize("profile", ["", "optional", 7])
def test_unknown_profile_is_rejected(profile: object) -> None:
    config = _config(VISION_ONLY)
    config.tactile_profile = profile
    with pytest.raises(ValueError, match="tactile_profile"):
        validate_tactile_profile_config(config)


def test_legacy_unambiguous_vision_only_profile_is_inferred() -> None:
    config = _config(VISION_ONLY)
    del config.tactile_profile

    contract = validate_tactile_profile_config(config)

    assert contract.profile == VISION_ONLY


def test_vision_tactile_rejects_missing_or_dropped_tactile() -> None:
    config = _config(VISION_TACTILE)
    config.per_repo_tactile_keys = {"touch": []}
    with pytest.raises(ValueError, match="every repository"):
        validate_tactile_profile_config(config, repo_names=("touch",))

    config = _config(VISION_TACTILE)
    config.tactile_cfg_prob = 0.1
    with pytest.raises(ValueError, match="tactile_cfg_prob=0"):
        validate_tactile_profile_config(config)


@pytest.mark.parametrize(
    "repo_map",
    [
        {"touch": ["tactile_a"]},
        {"touch": [], "rgb": []},
        {"touch": ["tactile_a"], "other": ["tactile_a"]},
    ],
)
def test_mixed_requires_exact_tactile_and_no_tactile_roster(
    repo_map: dict[str, list[str]],
) -> None:
    config = _config(MIXED)
    config.per_repo_tactile_keys = repo_map
    with pytest.raises(ValueError, match="mixed"):
        validate_tactile_profile_config(config, repo_names=("touch", "rgb"))


def test_mixed_rejects_ambiguous_optional_fallback() -> None:
    config = _config(MIXED)
    config.tactile_optional = True
    with pytest.raises(ValueError, match="explicitly"):
        validate_tactile_profile_config(config, repo_names=("touch", "rgb"))


def test_mixed_keeps_a_nonzero_probability_of_real_tactile_training() -> None:
    config = _config(MIXED)
    config.tactile_cfg_prob = 1.0

    with pytest.raises(ValueError, match="< 1"):
        validate_tactile_profile_config(config, repo_names=("touch", "rgb"))


def test_mixed_rejects_unrouted_global_tactile_key() -> None:
    config = _config(MIXED)
    config.tactile_keys = ["tactile_a", "tactile_b"]
    config.tactile_sensor_id_map = {"tactile_a": 0, "tactile_b": 1}
    config.active_tactile_sensor_count = 2

    with pytest.raises(ValueError, match="exactly equal"):
        validate_tactile_profile_config(config, repo_names=("touch", "rgb"))


def test_mixed_repo_resolution_is_explicit_and_never_falls_back() -> None:
    config = _config(MIXED)

    assert resolve_repo_tactile_keys(config, "touch") == ("tactile_a",)
    assert resolve_repo_tactile_keys(config, "rgb") == ()
    with pytest.raises(ValueError, match="no tactile declaration"):
        resolve_repo_tactile_keys(config, "unknown")


def test_mixed_serving_binds_each_task_to_signed_keys_and_sensor_ids() -> None:
    config = _config(MIXED)
    payload = validate_tactile_profile_config(
        config, repo_names=("touch", "rgb")
    ).to_json_dict()

    _, touch_sensor_map = validate_serving_tactile_binding(
        payload,
        live_profile=MIXED,
        live_tactile_keys=["tactile_a"],
        serve_task="touch",
        **_serve_semantics(config),
    )
    _, rgb_sensor_map = validate_serving_tactile_binding(
        payload,
        live_profile=MIXED,
        live_tactile_keys=[],
        serve_task="rgb",
        **_serve_semantics(config),
    )

    assert touch_sensor_map == {"tactile_a": 0}
    assert rgb_sensor_map == {}


def test_mixed_serving_rejects_missing_task_or_wrong_live_profile() -> None:
    config = _config(MIXED)
    payload = validate_tactile_profile_config(
        config, repo_names=("touch", "rgb")
    ).to_json_dict()

    with pytest.raises(ValueError, match="serve_task"):
        validate_serving_tactile_binding(
            payload,
            live_profile=MIXED,
            live_tactile_keys=[],
            serve_task=None,
            **_serve_semantics(config),
        )
    with pytest.raises(ValueError, match="differs from checkpoint"):
        validate_serving_tactile_binding(
            payload,
            live_profile=VISION_TACTILE,
            live_tactile_keys=["tactile_a"],
            serve_task="touch",
            **_serve_semantics(config),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("live_tactile_mode", "disabled"),
        ("live_max_tactile_streams", 3),
        ("live_use_local_tactile", False),
        ("live_local_tactile_mode", "residual"),
        ("live_use_contact_gate", True),
        ("live_tactile_global_zero", True),
    ],
)
def test_serving_rejects_each_signed_tactile_semantic_mismatch(
    field: str,
    value: object,
) -> None:
    config = _config(MIXED)
    payload = validate_tactile_profile_config(
        config, repo_names=("touch", "rgb")
    ).to_json_dict()
    semantics = _serve_semantics(config)
    semantics[field] = value

    with pytest.raises(ValueError, match="differs from checkpoint"):
        validate_serving_tactile_binding(
            payload,
            live_profile=MIXED,
            live_tactile_keys=["tactile_a"],
            serve_task="touch",
            **semantics,
        )


def test_absent_tactile_inference_route_distinguishes_drop_from_disable() -> None:
    assert resolve_absent_tactile_inference_route(MIXED) == {"tactile_cond_drop": True}
    assert resolve_absent_tactile_inference_route(VISION_ONLY) == {
        "tactile_mode": "disabled"
    }
    with pytest.raises(ValueError, match="cannot omit"):
        resolve_absent_tactile_inference_route(VISION_TACTILE)


def test_selected_repository_basenames_must_be_unique() -> None:
    config = _config(MIXED)

    with pytest.raises(ValueError, match="unique"):
        validate_tactile_profile_config(
            config,
            repo_names=("touch", "rgb", "touch"),
        )


def test_legacy_track31_checkpoint_is_narrowly_inferable() -> None:
    contract = validate_tactile_profile_config(_config(VISION_TACTILE))
    metadata = _legacy_track31_metadata()

    assert legacy_checkpoint_matches_tactile_profile(contract, metadata)
    metadata.pop("track31_artifacts")
    assert not legacy_checkpoint_matches_tactile_profile(contract, metadata)


def test_legacy_track31_cannot_infer_new_per_repo_routes() -> None:
    config = _config(VISION_TACTILE)
    config.per_repo_tactile_keys = {
        "touch_a": ["tactile_a"],
        "touch_b": ["tactile_a"],
    }
    contract = validate_tactile_profile_config(
        config,
        repo_names=("touch_a", "touch_b"),
    )

    assert not legacy_checkpoint_matches_tactile_profile(
        contract,
        _legacy_track31_metadata(),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("local_tactile_mode", "residual"),
        ("tactile_global_zero", True),
        ("noisy_cond_prob_tactile", 0.5),
        ("tactile_diffusion_loss_weight", 2.0),
    ],
)
def test_legacy_track31_cannot_infer_changed_tactile_semantics(
    field: str,
    value: object,
) -> None:
    config = _config(VISION_TACTILE)
    setattr(config, field, value)
    contract = validate_tactile_profile_config(config)

    assert not legacy_checkpoint_matches_tactile_profile(
        contract,
        _legacy_track31_metadata(),
    )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("training_profile_identity", "source_manifest_sha256"),
        ("track31_artifacts", "video_inventory_sha256"),
        ("trainability_contract", "frozen_parameter_names"),
    ],
)
def test_legacy_track31_checkpoint_rejects_incomplete_identity(
    section: str,
    field: str,
) -> None:
    contract = validate_tactile_profile_config(_config(VISION_TACTILE))
    metadata = _legacy_track31_metadata()
    payload = metadata[section]
    assert isinstance(payload, dict)
    payload.pop(field)

    assert not legacy_checkpoint_matches_tactile_profile(contract, metadata)


def test_legacy_vision_only_requires_complete_hashed_trainability() -> None:
    contract = validate_tactile_profile_config(_config(VISION_ONLY))
    metadata = {
        "tactile_mode": "disabled",
        "tactile_keys": [],
        "trainability_contract": _legacy_trainability(vision_only=True),
    }

    assert legacy_checkpoint_matches_tactile_profile(contract, metadata)
    metadata["trainability_contract"]["frozen_numel"] = 17
    assert not legacy_checkpoint_matches_tactile_profile(contract, metadata)


def _rehash_legacy_trainability(payload: dict[str, object]) -> None:
    canonical = {
        key: value for key, value in payload.items() if key != "contract_sha256"
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def test_legacy_vision_only_rejects_non_tactile_frozen_inventory() -> None:
    contract = validate_tactile_profile_config(_config(VISION_ONLY))
    trainability = _legacy_trainability(vision_only=True)
    trainability["frozen_parameter_names"] = ["action_embedder.weight"]
    _rehash_legacy_trainability(trainability)

    assert not legacy_checkpoint_matches_tactile_profile(
        contract,
        {
            "tactile_mode": "disabled",
            "tactile_keys": [],
            "trainability_contract": trainability,
        },
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("frozen_parameter_count", True),
    ],
)
def test_legacy_trainability_rejects_boolean_integer_fields(
    field: str,
    value: object,
) -> None:
    contract = validate_tactile_profile_config(_config(VISION_ONLY))
    trainability = _legacy_trainability(vision_only=True)
    trainability[field] = value
    _rehash_legacy_trainability(trainability)

    assert not legacy_checkpoint_matches_tactile_profile(
        contract,
        {
            "tactile_mode": "disabled",
            "tactile_keys": [],
            "trainability_contract": trainability,
        },
    )


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("top", "video_inventory_sha256", "b" * 64),
        ("artifacts", "parent_validation_view_id", "tampered"),
        ("artifacts", "parent_validation_view_sha256", "not-a-sha"),
        ("artifacts", "action_q01", [9.0] * 8),
        ("artifacts", "action_q99", [-9.0] * 8),
        ("artifacts", "train_episode_count", 1),
        ("artifacts", "validation_episode_count", 999),
        ("artifacts", "train_episode_count", 719.0),
        ("artifacts", "validation_episode_count", False),
        ("profile", "run_role", "evil_role"),
        ("profile", "schema_version", 2.0),
        ("profile", "sampler_coverage_mode", "legacy"),
        ("profile", "train_view_id", "attacker_view"),
    ],
)
def test_legacy_track31_rejects_cross_identity_or_semantic_tamper(
    location: str,
    field: str,
    value: object,
) -> None:
    contract = validate_tactile_profile_config(_config(VISION_TACTILE))
    metadata = _legacy_track31_metadata()
    if location == "top":
        target = metadata
    elif location == "artifacts":
        target = metadata["track31_artifacts"]
    else:
        target = metadata["training_profile_identity"]
    assert isinstance(target, dict)
    target[field] = value

    assert not legacy_checkpoint_matches_tactile_profile(contract, metadata)


def test_legacy_mixed_checkpoint_is_never_inferred() -> None:
    contract = validate_tactile_profile_config(
        _config(MIXED), repo_names=("touch", "rgb")
    )
    metadata = _legacy_track31_metadata()

    assert not legacy_checkpoint_matches_tactile_profile(contract, metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tactile_keys", ["tactile_a"]),
        ("freeze_tactile_parameters", False),
        ("tactile_diffusion_loss_weight", 1.0),
        ("use_local_tactile", True),
        ("use_contact_gate", True),
        ("synthetic_tactile_data", True),
    ],
)
def test_vision_only_rejects_active_tactile_behavior(
    field: str,
    value: object,
) -> None:
    config = _config(VISION_ONLY)
    setattr(config, field, value)
    with pytest.raises(ValueError):
        validate_tactile_profile_config(config, repo_names=("rgb",))


def test_tactile_profile_contract_rejects_tampering() -> None:
    payload = validate_tactile_profile_config(
        _config(MIXED), repo_names=("touch", "rgb")
    ).to_json_dict()
    payload["tactile_cfg_prob"] = 0.5
    with pytest.raises(ValueError, match="SHA256"):
        validate_tactile_profile_contract(payload)


def test_local_tactile_mode_changes_profile_identity() -> None:
    current = _config(MIXED)
    residual = _config(MIXED)
    residual.local_tactile_mode = "residual"

    current_contract = validate_tactile_profile_config(current)
    residual_contract = validate_tactile_profile_config(residual)

    assert current_contract.contract_sha256 != residual_contract.contract_sha256


def test_tactile_profile_contract_rejects_semantically_resealed_payload() -> None:
    payload = validate_tactile_profile_config(_config(VISION_ONLY)).to_json_dict()
    payload["tactile_mode"] = "enabled"
    raw = dict(payload)
    raw.pop("contract_sha256")
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            raw,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="vision_only"):
        validate_tactile_profile_contract(payload)


@pytest.mark.parametrize(
    ("sensor_map", "active_count"),
    [
        ({}, 0),
        ({"tactile_a": 0, "unused": 1}, 2),
        ({"tactile_a": 0}, 2),
    ],
)
def test_enabled_profile_requires_exact_sensor_contract(
    sensor_map: dict[str, int],
    active_count: int,
) -> None:
    config = _config(VISION_TACTILE)
    config.tactile_sensor_id_map = sensor_map
    config.active_tactile_sensor_count = active_count

    with pytest.raises(ValueError, match="sensor"):
        validate_tactile_profile_config(config)
