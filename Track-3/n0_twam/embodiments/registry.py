# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Explicit registries for immutable embodiment and action-route specs."""

from __future__ import annotations

from .spec import ActionRouteSpec, EmbodimentSpec

_EMBODIMENT_SPECS: dict[str, EmbodimentSpec] = {}
_ACTION_ROUTE_SPECS: dict[str, ActionRouteSpec] = {}


def register_embodiment_spec(spec: EmbodimentSpec) -> EmbodimentSpec:
    current = _EMBODIMENT_SPECS.get(spec.profile_id)
    if current is not None and current != spec:
        raise ValueError(
            f"embodiment profile {spec.profile_id!r} is already registered"
        )
    _EMBODIMENT_SPECS[spec.profile_id] = spec
    return spec


def register_action_route_spec(spec: ActionRouteSpec) -> ActionRouteSpec:
    current = _ACTION_ROUTE_SPECS.get(spec.name)
    if current is not None and current != spec:
        raise ValueError(f"action route {spec.name!r} is already registered")
    _ACTION_ROUTE_SPECS[spec.name] = spec
    return spec


def get_embodiment_spec(profile_id: str) -> EmbodimentSpec:
    try:
        return _EMBODIMENT_SPECS[profile_id]
    except KeyError as error:
        available = ", ".join(sorted(_EMBODIMENT_SPECS))
        raise KeyError(
            f"unknown embodiment profile {profile_id!r}; available: {available}"
        ) from error


def get_action_route_spec(name: str) -> ActionRouteSpec:
    try:
        return _ACTION_ROUTE_SPECS[name]
    except KeyError as error:
        available = ", ".join(sorted(_ACTION_ROUTE_SPECS))
        raise KeyError(
            f"unknown action route {name!r}; available: {available}"
        ) from error


def list_embodiment_specs() -> tuple[str, ...]:
    return tuple(sorted(_EMBODIMENT_SPECS))


def list_action_route_specs() -> tuple[str, ...]:
    return tuple(sorted(_ACTION_ROUTE_SPECS))
