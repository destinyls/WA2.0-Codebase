# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Immutable embodiment and action-route contracts for AgileX Track 3."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from n0_twam.embodiments import (
    AGILEX_ACTION_ROUTE_SPEC,
    AGILEX_ACTION_SCHEMA,
    AGILEX_EMBODIMENT_PROFILE_ID,
    AGILEX_EMBODIMENT_SPEC,
    build_agilex_repo_route_manifest,
    get_action_route_spec,
    get_embodiment_spec,
    list_action_route_specs,
    list_embodiment_specs,
    validate_action_route_contract,
    validate_embodiment_contract,
    validate_repo_route_manifest_contract,
)

RGB_KEYS = (
    "observation.images.top",
    "observation.images.wrist_l",
    "observation.images.wrist_r",
)
TACTILE_KEYS = (
    "observation.images.tactile_l",
    "observation.images.tactile_r",
)
WRENCH_KEYS = (
    "observation.wrench.left",
    "observation.wrench.right",
)


def _route(*, tactile: bool) -> dict[str, object]:
    return {
        "embodiment": AGILEX_EMBODIMENT_PROFILE_ID,
        "action_schema": AGILEX_ACTION_SCHEMA,
        "rgb_keys": list(RGB_KEYS),
        "tactile_keys": list(TACTILE_KEYS) if tactile else [],
        "wrench_keys": list(WRENCH_KEYS) if tactile else [],
    }


def test_agilex_embodiment_and_action_route_are_self_hashed_and_immutable() -> None:
    embodiment = AGILEX_EMBODIMENT_SPEC.to_json_dict()
    route = AGILEX_ACTION_ROUTE_SPEC.to_json_dict()

    assert embodiment["profile_id"] == "agilex_dual_qpos14_v1"
    assert embodiment["wire_action_schema"] == AGILEX_ACTION_SCHEMA
    assert embodiment["model_action_schema"] == AGILEX_ACTION_SCHEMA
    assert embodiment["dataset_adapter"] == "worldarena_agilex_qpos14"
    assert route["wire_action_spec"] == AGILEX_ACTION_SCHEMA
    assert route["model_action_spec"] == AGILEX_ACTION_SCHEMA
    assert route["active_model_channels"] == list(range(14))
    assert validate_embodiment_contract(embodiment) == embodiment
    assert validate_action_route_contract(route) == route
    assert len(embodiment["contract_sha256"]) == 64
    assert len(route["contract_sha256"]) == 64

    with pytest.raises(FrozenInstanceError):
        AGILEX_EMBODIMENT_SPEC.profile_id = "mutated"  # type: ignore[misc]


def test_tampering_with_self_hashed_contracts_is_rejected() -> None:
    embodiment = AGILEX_EMBODIMENT_SPEC.to_json_dict()
    embodiment["dataset_adapter"] = "dimension_guessed_adapter"
    with pytest.raises(ValueError, match="SHA256"):
        validate_embodiment_contract(embodiment)

    route = AGILEX_ACTION_ROUTE_SPEC.to_json_dict()
    route["active_model_channels"] = list(range(13))
    with pytest.raises(ValueError, match="SHA256"):
        validate_action_route_contract(route)


def test_registry_keeps_agilex_franka_and_univtac_explicitly_separate() -> None:
    assert set(list_embodiment_specs()) >= {
        "agilex_dual_qpos14_v1",
        "franka_pose8_ee20_v1",
        "univtac_panda_qpos8_v1",
    }
    assert set(list_action_route_specs()) >= {
        "agilex_qpos14_identity_v1",
        "franka_pose8_to_ee20_v1",
        "univtac_qpos8_identity_v1",
    }
    assert get_embodiment_spec("agilex_dual_qpos14_v1") is AGILEX_EMBODIMENT_SPEC
    assert get_action_route_spec("agilex_qpos14_identity_v1") is (
        AGILEX_ACTION_ROUTE_SPEC
    )
    assert get_embodiment_spec("franka_pose8_ee20_v1").model_action_schema == (
        "ee20_absee"
    )
    assert get_embodiment_spec("univtac_panda_qpos8_v1").model_action_schema == (
        "qpos8_next_step"
    )


def test_repo_route_manifest_is_canonical_and_has_strict_global_unions() -> None:
    routes_a = {
        "agilex_rgb": _route(tactile=False),
        "agilex_touch": _route(tactile=True),
    }
    routes_b = {
        "agilex_touch": _route(tactile=True),
        "agilex_rgb": _route(tactile=False),
    }

    manifest_a = build_agilex_repo_route_manifest(routes_a)
    manifest_b = build_agilex_repo_route_manifest(routes_b)
    payload = manifest_a.to_json_dict()

    assert manifest_a == manifest_b
    assert manifest_a.contract_sha256 == manifest_b.contract_sha256
    assert payload["global_rgb_keys"] == list(RGB_KEYS)
    assert payload["global_tactile_keys"] == list(TACTILE_KEYS)
    assert payload["global_wrench_keys"] == list(WRENCH_KEYS)
    assert payload["tactile_sensor_id_map"] == {
        TACTILE_KEYS[0]: 0,
        TACTILE_KEYS[1]: 1,
    }
    assert payload["wrench_sensor_id_map"] == {
        WRENCH_KEYS[0]: 0,
        WRENCH_KEYS[1]: 1,
    }
    assert validate_repo_route_manifest_contract(payload) == payload


def test_repo_route_manifest_rejects_schema_guessing_and_duplicate_sensor_ids() -> None:
    routes = {"agilex_touch": _route(tactile=True)}
    routes["agilex_touch"]["action_schema"] = "ee20_absee"
    with pytest.raises(ValueError, match="action schema"):
        build_agilex_repo_route_manifest(routes)

    routes = {"agilex_touch": _route(tactile=True)}
    with pytest.raises(ValueError, match="sensor IDs"):
        build_agilex_repo_route_manifest(
            routes,
            tactile_sensor_id_map={TACTILE_KEYS[0]: 0, TACTILE_KEYS[1]: 0},
        )

    routes = {"agilex_touch": _route(tactile=True)}
    routes["agilex_touch"]["rgb_keys"] = [RGB_KEYS[0]]
    with pytest.raises(ValueError, match="canonical AgileX order"):
        build_agilex_repo_route_manifest(routes)


def test_repo_route_manifest_hash_detects_route_tampering() -> None:
    payload = build_agilex_repo_route_manifest(
        {"agilex_touch": _route(tactile=True)}
    ).to_json_dict()
    payload["routes"]["agilex_touch"]["tactile_keys"] = []

    with pytest.raises(ValueError, match="SHA256"):
        validate_repo_route_manifest_contract(payload)
