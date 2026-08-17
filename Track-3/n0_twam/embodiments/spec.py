# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Immutable embodiment, action-route, and repository-route contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

CONTRACT_SCHEMA_VERSION = 1


def canonical_sha256(payload: Mapping[str, object]) -> str:
    """Hash a JSON contract with one canonical serialization."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_tuple(value: object, *, label: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a sequence of strings")
    result = tuple(value)
    if (not allow_empty and not result) or any(
        not isinstance(item, str) or not item for item in result
    ):
        raise ValueError(f"{label} contains an invalid string")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return cast(tuple[str, ...], result)


def _integer_tuple(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a sequence of integers")
    result = tuple(value)
    if any(type(item) is not int or item < 0 for item in result):
        raise ValueError(f"{label} must contain non-negative integers")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return cast(tuple[int, ...], result)


def _validate_self_hash(
    payload: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError(f"{label} has an invalid field set")
    raw = dict(payload)
    digest = raw.pop("contract_sha256")
    if raw.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported {label} schema")
    if digest != canonical_sha256(raw):
        raise ValueError(f"{label} SHA256 mismatch")
    return raw


@dataclass(frozen=True)
class EmbodimentSpec:
    """One robot embodiment identity, independent of its tactile profile."""

    schema_version: int
    profile_id: str
    revision: str
    robot_family: str
    state_schema: str
    wire_action_schema: str
    model_action_schema: str
    action_route: str
    dataset_adapter: str
    policy_adapter: str
    safety_contract_id: str
    contract_sha256: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "revision": self.revision,
            "robot_family": self.robot_family,
            "state_schema": self.state_schema,
            "wire_action_schema": self.wire_action_schema,
            "model_action_schema": self.model_action_schema,
            "action_route": self.action_route,
            "dataset_adapter": self.dataset_adapter,
            "policy_adapter": self.policy_adapter,
            "safety_contract_id": self.safety_contract_id,
            "contract_sha256": self.contract_sha256,
        }


@dataclass(frozen=True)
class ActionRouteSpec:
    """Explicit wire-to-model and model-to-wire semantic route."""

    schema_version: int
    name: str
    revision: str
    wire_action_spec: str
    model_action_spec: str
    temporal_alignment_policy: str
    dataset_to_model_adapter: str
    model_to_wire_adapter: str
    active_model_channels: tuple[int, ...]
    contract_sha256: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "revision": self.revision,
            "wire_action_spec": self.wire_action_spec,
            "model_action_spec": self.model_action_spec,
            "temporal_alignment_policy": self.temporal_alignment_policy,
            "dataset_to_model_adapter": self.dataset_to_model_adapter,
            "model_to_wire_adapter": self.model_to_wire_adapter,
            "active_model_channels": list(self.active_model_channels),
            "contract_sha256": self.contract_sha256,
        }


@dataclass(frozen=True)
class RepoObservationRouteSpec:
    """Canonical observation roster for one selected repository."""

    embodiment: str
    action_schema: str
    rgb_keys: tuple[str, ...]
    tactile_keys: tuple[str, ...]
    wrench_keys: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "embodiment": self.embodiment,
            "action_schema": self.action_schema,
            "rgb_keys": list(self.rgb_keys),
            "tactile_keys": list(self.tactile_keys),
            "wrench_keys": list(self.wrench_keys),
        }


@dataclass(frozen=True)
class RepoRouteManifestContract:
    """Content-addressed union of all selected per-repository routes."""

    schema_version: int
    embodiment_profile_id: str
    action_route: str
    action_schema: str
    routes: tuple[tuple[str, RepoObservationRouteSpec], ...]
    global_rgb_keys: tuple[str, ...]
    global_tactile_keys: tuple[str, ...]
    global_wrench_keys: tuple[str, ...]
    tactile_sensor_id_map: tuple[tuple[str, int], ...]
    wrench_sensor_id_map: tuple[tuple[str, int], ...]
    contract_sha256: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "embodiment_profile_id": self.embodiment_profile_id,
            "action_route": self.action_route,
            "action_schema": self.action_schema,
            "routes": {repo: route.to_json_dict() for repo, route in self.routes},
            "global_rgb_keys": list(self.global_rgb_keys),
            "global_tactile_keys": list(self.global_tactile_keys),
            "global_wrench_keys": list(self.global_wrench_keys),
            "tactile_sensor_id_map": dict(self.tactile_sensor_id_map),
            "wrench_sensor_id_map": dict(self.wrench_sensor_id_map),
            "contract_sha256": self.contract_sha256,
        }


def build_embodiment_spec(**values: object) -> EmbodimentSpec:
    """Validate and content-address an embodiment definition."""

    field_names = (
        "profile_id",
        "revision",
        "robot_family",
        "state_schema",
        "wire_action_schema",
        "model_action_schema",
        "action_route",
        "dataset_adapter",
        "policy_adapter",
        "safety_contract_id",
    )
    canonical = {
        name: _nonempty_string(values.get(name), label=f"EmbodimentSpec.{name}")
        for name in field_names
    }
    payload: dict[str, object] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        **canonical,
    }
    return EmbodimentSpec(
        schema_version=CONTRACT_SCHEMA_VERSION,
        **canonical,
        contract_sha256=canonical_sha256(payload),
    )


def build_action_route_spec(**values: object) -> ActionRouteSpec:
    """Validate and content-address one action route."""

    field_names = (
        "name",
        "revision",
        "wire_action_spec",
        "model_action_spec",
        "temporal_alignment_policy",
        "dataset_to_model_adapter",
        "model_to_wire_adapter",
    )
    canonical: dict[str, object] = {
        name: _nonempty_string(values.get(name), label=f"ActionRouteSpec.{name}")
        for name in field_names
    }
    active_channels = _integer_tuple(
        values.get("active_model_channels"),
        label="ActionRouteSpec.active_model_channels",
    )
    if not active_channels:
        raise ValueError("ActionRouteSpec.active_model_channels cannot be empty")
    canonical["active_model_channels"] = list(active_channels)
    payload: dict[str, object] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        **canonical,
    }
    return ActionRouteSpec(
        schema_version=CONTRACT_SCHEMA_VERSION,
        name=cast(str, canonical["name"]),
        revision=cast(str, canonical["revision"]),
        wire_action_spec=cast(str, canonical["wire_action_spec"]),
        model_action_spec=cast(str, canonical["model_action_spec"]),
        temporal_alignment_policy=cast(str, canonical["temporal_alignment_policy"]),
        dataset_to_model_adapter=cast(str, canonical["dataset_to_model_adapter"]),
        model_to_wire_adapter=cast(str, canonical["model_to_wire_adapter"]),
        active_model_channels=active_channels,
        contract_sha256=canonical_sha256(payload),
    )


_EMBODIMENT_FIELDS = frozenset(EmbodimentSpec.__dataclass_fields__)
_ACTION_ROUTE_FIELDS = frozenset(ActionRouteSpec.__dataclass_fields__)
_REPO_MANIFEST_FIELDS = frozenset(RepoRouteManifestContract.__dataclass_fields__)


def validate_embodiment_contract(payload: object) -> dict[str, object]:
    raw = _validate_self_hash(
        payload,
        fields=_EMBODIMENT_FIELDS,
        label="embodiment contract",
    )
    rebuilt = build_embodiment_spec(
        **{key: value for key, value in raw.items() if key != "schema_version"}
    ).to_json_dict()
    if rebuilt != dict(cast(Mapping[str, object], payload)):
        raise ValueError("embodiment contract is not canonical")
    return rebuilt


def validate_action_route_contract(payload: object) -> dict[str, object]:
    raw = _validate_self_hash(
        payload,
        fields=_ACTION_ROUTE_FIELDS,
        label="action route contract",
    )
    rebuilt = build_action_route_spec(
        **{key: value for key, value in raw.items() if key != "schema_version"}
    ).to_json_dict()
    if rebuilt != dict(cast(Mapping[str, object], payload)):
        raise ValueError("action route contract is not canonical")
    return rebuilt


def validate_sensor_id_map(
    value: object,
    *,
    expected_keys: Sequence[str],
    label: str,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    mapping = dict(value)
    if set(mapping) != set(expected_keys):
        raise ValueError(f"{label} must exactly cover the routed key union")
    ids = tuple(mapping[key] for key in sorted(mapping))
    if any(type(sensor_id) is not int or sensor_id < 0 for sensor_id in ids):
        raise ValueError(f"{label} sensor IDs must be non-negative integers")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} sensor IDs must be unique")
    return tuple((key, cast(int, mapping[key])) for key in sorted(mapping))


def validate_repo_route_manifest_contract(payload: object) -> dict[str, object]:
    """Validate self-hash and reconstruct all nested immutable routes."""

    raw = _validate_self_hash(
        payload,
        fields=_REPO_MANIFEST_FIELDS,
        label="repo route manifest contract",
    )
    from .agilex import build_agilex_repo_route_manifest

    routes = raw.get("routes")
    if not isinstance(routes, Mapping):
        raise ValueError("repo route manifest routes must be a mapping")
    rebuilt = build_agilex_repo_route_manifest(
        cast(Mapping[str, Mapping[str, object]], routes),
        tactile_sensor_id_map=cast(Mapping[str, int], raw.get("tactile_sensor_id_map")),
        wrench_sensor_id_map=cast(Mapping[str, int], raw.get("wrench_sensor_id_map")),
    ).to_json_dict()
    if rebuilt != dict(cast(Mapping[str, object], payload)):
        raise ValueError("repo route manifest contract is not canonical")
    return rebuilt
