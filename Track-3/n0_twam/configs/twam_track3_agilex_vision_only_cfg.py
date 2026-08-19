# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""AgileX qpos14 vision-only profile with frozen, bypassed contact modules."""

from __future__ import annotations

from collections.abc import Mapping

from easydict import EasyDict

from n0_twam.tactile_profiles import VISION_ONLY

from .twam_track3_agilex_base_cfg import (
    apply_agilex_tactile_profile,
    build_track3_agilex_base_config,
    load_agilex_repo_route_binding,
)


def build_track3_agilex_vision_only_config(
    *,
    repo_routes: Mapping[str, Mapping[str, object]] | None = None,
) -> EasyDict:
    binding = (
        load_agilex_repo_route_binding(VISION_ONLY) if repo_routes is None else None
    )
    routes = binding.routes if binding is not None else repo_routes
    assert routes is not None
    cfg = build_track3_agilex_base_config(
        repo_routes=routes,
        repo_route_binding=binding,
        tactile_profile=VISION_ONLY,
    )
    cfg.__name__ = "Config: N0-TWAM AgileX qpos14 vision-only"
    return apply_agilex_tactile_profile(cfg, VISION_ONLY)


twam_track3_agilex_vision_only_cfg = build_track3_agilex_vision_only_config()
