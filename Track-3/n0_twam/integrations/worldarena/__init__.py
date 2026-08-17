# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""WorldArena-specific dataset and policy adapters."""

from __future__ import annotations

from typing import Any

from .agilex_actions import (
    QPOS14_ACTION_SCHEMA,
    AgileXActionLabelContract,
    derive_measured_next_qpos,
    official_action_contract,
    validate_qpos14,
)
from .franka_actions import (
    FRANKA_ACTION_SCHEMA,
    MODEL_ACTION_SCHEMA,
    ee10_to_end_pose8,
    embed_ee10_in_ee20,
    end_pose8_to_ee10,
    extract_ee10_from_ee20,
)
from .franka_manifest import OFFICIAL_REPO_ID, OFFICIAL_REVISION

__all__ = (
    "AgileXActionLabelContract",
    "build_agilex_serve_bundle",
    "FRANKA_ACTION_SCHEMA",
    "MODEL_ACTION_SCHEMA",
    "OFFICIAL_REPO_ID",
    "OFFICIAL_REVISION",
    "QPOS14_ACTION_SCHEMA",
    "derive_measured_next_qpos",
    "ee10_to_end_pose8",
    "embed_ee10_in_ee20",
    "end_pose8_to_ee10",
    "extract_ee10_from_ee20",
    "official_action_contract",
    "validate_qpos14",
    "verify_agilex_serve_bundle",
)


def __getattr__(name: str) -> Any:
    """Lazily expose serve-bundle helpers without checkpoint import cycles."""

    if name not in {"build_agilex_serve_bundle", "verify_agilex_serve_bundle"}:
        raise AttributeError(name)
    from .agilex_serve_bundle import (
        build_agilex_serve_bundle,
        verify_agilex_serve_bundle,
    )

    exports = {
        "build_agilex_serve_bundle": build_agilex_serve_bundle,
        "verify_agilex_serve_bundle": verify_agilex_serve_bundle,
    }
    return exports[name]
