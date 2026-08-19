# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Official AgileX dual-arm qpos14 embodiment and repository routing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from .registry import register_action_route_spec, register_embodiment_spec
from .spec import (
    CONTRACT_SCHEMA_VERSION,
    ActionRouteSpec,
    EmbodimentSpec,
    RepoObservationRouteSpec,
    RepoRouteManifestContract,
    build_action_route_spec,
    build_embodiment_spec,
    canonical_sha256,
    validate_sensor_id_map,
)

AGILEX_EMBODIMENT_PROFILE_ID = "agilex_dual_qpos14_v1"
AGILEX_ACTION_SCHEMA = "qpos14_joint_absolute_v1"
AGILEX_ACTION_ROUTE_NAME = "agilex_qpos14_identity_v1"
AGILEX_DATASET_ADAPTER = "worldarena_agilex_qpos14"
AGILEX_POLICY_ADAPTER = "worldarena_agilex_qpos14"

AGILEX_RGB_KEYS = (
    "observation.images.top",
    "observation.images.wrist_l",
    "observation.images.wrist_r",
)
AGILEX_TACTILE_KEYS = (
    "observation.images.tactile_l",
    "observation.images.tactile_r",
)
AGILEX_WRENCH_KEYS = (
    "observation.wrench.left",
    "observation.wrench.right",
)

AGILEX_ACTION_ROUTE_SPEC: ActionRouteSpec = register_action_route_spec(
    build_action_route_spec(
        name=AGILEX_ACTION_ROUTE_NAME,
        revision="1",
        wire_action_spec=AGILEX_ACTION_SCHEMA,
        model_action_spec=AGILEX_ACTION_SCHEMA,
        temporal_alignment_policy="manifest_index_map_v1",
        dataset_to_model_adapter="qpos14_identity_normalized_v1",
        model_to_wire_adapter="qpos14_identity_denormalized_v1",
        active_model_channels=tuple(range(14)),
    )
)

AGILEX_EMBODIMENT_SPEC: EmbodimentSpec = register_embodiment_spec(
    build_embodiment_spec(
        profile_id=AGILEX_EMBODIMENT_PROFILE_ID,
        revision="1",
        robot_family="agilex_dual_arm",
        state_schema="qpos14_measured_joint_v1",
        wire_action_schema=AGILEX_ACTION_SCHEMA,
        model_action_schema=AGILEX_ACTION_SCHEMA,
        action_route=AGILEX_ACTION_ROUTE_NAME,
        dataset_adapter=AGILEX_DATASET_ADAPTER,
        policy_adapter=AGILEX_POLICY_ADAPTER,
        safety_contract_id="agilex_signed_joint_safety_v1",
    )
)


def _route_from_mapping(
    repo_name: str,
    value: Mapping[str, object],
) -> RepoObservationRouteSpec:
    if not isinstance(repo_name, str) or not repo_name:
        raise ValueError("repository names must be non-empty strings")
    exact_fields = {
        "embodiment",
        "action_schema",
        "rgb_keys",
        "tactile_keys",
        "wrench_keys",
    }
    if set(value) != exact_fields:
        raise ValueError(f"route {repo_name!r} has an invalid field set")
    embodiment = value["embodiment"]
    action_schema = value["action_schema"]
    if embodiment != AGILEX_EMBODIMENT_PROFILE_ID:
        raise ValueError(f"route {repo_name!r} has an incompatible embodiment")
    if action_schema != AGILEX_ACTION_SCHEMA:
        raise ValueError(f"route {repo_name!r} has an incompatible action schema")

    def keys(field: str, *, required: bool) -> tuple[str, ...]:
        raw = value[field]
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f"route {repo_name!r} {field} must be a sequence")
        result = tuple(raw)
        if (required and not result) or any(
            not isinstance(key, str) or not key for key in result
        ):
            raise ValueError(f"route {repo_name!r} has invalid {field}")
        if len(result) != len(set(result)):
            raise ValueError(f"route {repo_name!r} has duplicate {field}")
        return cast(tuple[str, ...], result)

    rgb_keys = keys("rgb_keys", required=True)
    if rgb_keys != AGILEX_RGB_KEYS:
        raise ValueError(
            f"route {repo_name!r} rgb_keys must match the canonical AgileX order"
        )
    return RepoObservationRouteSpec(
        embodiment=AGILEX_EMBODIMENT_PROFILE_ID,
        action_schema=AGILEX_ACTION_SCHEMA,
        rgb_keys=rgb_keys,
        tactile_keys=keys("tactile_keys", required=False),
        wrench_keys=keys("wrench_keys", required=False),
    )


def _ordered_union(
    routes: tuple[RepoObservationRouteSpec, ...],
    field: str,
) -> tuple[str, ...]:
    return tuple(sorted({key for route in routes for key in getattr(route, field)}))


def build_agilex_repo_route_manifest(
    routes: Mapping[str, Mapping[str, object]],
    *,
    tactile_sensor_id_map: Mapping[str, int] | None = None,
    wrench_sensor_id_map: Mapping[str, int] | None = None,
) -> RepoRouteManifestContract:
    """Build a strict route manifest without inferring robot/action from tensors."""

    if not isinstance(routes, Mapping) or not routes:
        raise ValueError("AgileX repo routes must be a non-empty mapping")
    parsed = tuple(
        (repo, _route_from_mapping(repo, value))
        for repo, value in sorted(routes.items())
    )
    route_values = tuple(route for _, route in parsed)
    rgb_union = _ordered_union(route_values, "rgb_keys")
    tactile_union = _ordered_union(route_values, "tactile_keys")
    wrench_union = _ordered_union(route_values, "wrench_keys")
    tactile_map = validate_sensor_id_map(
        (
            {key: index for index, key in enumerate(tactile_union)}
            if tactile_sensor_id_map is None
            else tactile_sensor_id_map
        ),
        expected_keys=tactile_union,
        label="tactile_sensor_id_map",
    )
    wrench_map = validate_sensor_id_map(
        (
            {key: index for index, key in enumerate(wrench_union)}
            if wrench_sensor_id_map is None
            else wrench_sensor_id_map
        ),
        expected_keys=wrench_union,
        label="wrench_sensor_id_map",
    )
    payload: dict[str, object] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "embodiment_profile_id": AGILEX_EMBODIMENT_PROFILE_ID,
        "action_route": AGILEX_ACTION_ROUTE_NAME,
        "action_schema": AGILEX_ACTION_SCHEMA,
        "routes": {repo: route.to_json_dict() for repo, route in parsed},
        "global_rgb_keys": list(rgb_union),
        "global_tactile_keys": list(tactile_union),
        "global_wrench_keys": list(wrench_union),
        "tactile_sensor_id_map": dict(tactile_map),
        "wrench_sensor_id_map": dict(wrench_map),
    }
    return RepoRouteManifestContract(
        schema_version=CONTRACT_SCHEMA_VERSION,
        embodiment_profile_id=AGILEX_EMBODIMENT_PROFILE_ID,
        action_route=AGILEX_ACTION_ROUTE_NAME,
        action_schema=AGILEX_ACTION_SCHEMA,
        routes=parsed,
        global_rgb_keys=rgb_union,
        global_tactile_keys=tactile_union,
        global_wrench_keys=wrench_union,
        tactile_sensor_id_map=tactile_map,
        wrench_sensor_id_map=wrench_map,
        contract_sha256=canonical_sha256(payload),
    )
