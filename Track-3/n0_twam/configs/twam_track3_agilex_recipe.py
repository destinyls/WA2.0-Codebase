# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict-resume recipe identity for AgileX training only."""

from __future__ import annotations

from collections.abc import Mapping

from .twam_track3_agilex_contracts import canonical_sha256

_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "run_role",
        "seed",
        "save_interval",
        "val_interval",
        "contract_sha256",
    }
)


def _field(source: object, name: str) -> object:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"AgileX {label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"AgileX {label} must be a non-negative integer")
    return value


def build_agilex_resume_recipe_contract(source: object) -> dict[str, object]:
    """Build the recipe subset that must remain fixed across strict resume."""

    run_role = _field(source, "run_role")
    if run_role not in {"development", "final_refit"}:
        raise ValueError("AgileX run_role must be development or final_refit")
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_role": run_role,
        "seed": _nonnegative_integer(_field(source, "seed"), label="seed"),
        "save_interval": _positive_integer(
            _field(source, "save_interval"), label="save_interval"
        ),
        "val_interval": _positive_integer(
            _field(source, "val_interval"), label="val_interval"
        ),
    }
    return {**payload, "contract_sha256": canonical_sha256(payload)}


def validate_agilex_resume_recipe_contract(
    value: object,
    *,
    current: object | None = None,
) -> dict[str, object]:
    """Validate one saved contract and optional current-recipe equality."""

    if not isinstance(value, Mapping):
        raise ValueError("AgileX resume recipe contract must be a mapping")
    if set(value) != _CONTRACT_FIELDS:
        raise ValueError("AgileX resume recipe contract fields are invalid")
    canonical = build_agilex_resume_recipe_contract(value)
    if dict(value) != canonical:
        raise ValueError("AgileX resume recipe contract identity is invalid")
    if current is not None:
        expected = build_agilex_resume_recipe_contract(current)
        if canonical != expected:
            raise ValueError("AgileX resume recipe contract differs from current run")
    return canonical


__all__ = (
    "build_agilex_resume_recipe_contract",
    "validate_agilex_resume_recipe_contract",
)
