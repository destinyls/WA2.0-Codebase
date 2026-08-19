# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""AgileX qpos14 mixed real-contact and RGB-only repository profile."""

from __future__ import annotations

from collections.abc import Mapping

from easydict import EasyDict

from n0_twam.tactile_profiles import MIXED

from .twam_track3_agilex_base_cfg import (
    apply_agilex_tactile_profile,
    build_track3_agilex_base_config,
    load_agilex_repo_route_binding,
)


def build_track3_agilex_mixed_config(
    *,
    repo_routes: Mapping[str, Mapping[str, object]] | None = None,
) -> EasyDict:
    binding = load_agilex_repo_route_binding(MIXED) if repo_routes is None else None
    routes = binding.routes if binding is not None else repo_routes
    assert routes is not None
    cfg = build_track3_agilex_base_config(
        repo_routes=routes,
        repo_route_binding=binding,
        tactile_profile=MIXED,
    )
    cfg.__name__ = "Config: N0-TWAM AgileX qpos14 mixed tactile/RGB-only"
    return apply_agilex_tactile_profile(cfg, MIXED)


twam_track3_agilex_mixed_cfg = build_track3_agilex_mixed_config()
