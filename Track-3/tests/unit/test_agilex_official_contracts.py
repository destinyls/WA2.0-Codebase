# Copyright 2025-2026 NeoteAI Team. All rights reserved.
from __future__ import annotations

from n0_twam.integrations.worldarena.agilex_official_contracts import (
    build_official_routes,
    build_route_manifest,
    build_temporal_contract,
)
from n0_twam.integrations.worldarena.agilex_official_schema import (
    ACTION_OFFSETS_PER_LATENT_ANCHOR,
    VISION_ONLY_REPO_ID,
    VISION_TACTILE_REPO_ID,
)


def test_official_mixed_contract_has_two_route_homogeneous_repositories() -> None:
    routes = build_official_routes()
    assert set(routes) == {VISION_ONLY_REPO_ID, VISION_TACTILE_REPO_ID}
    assert routes[VISION_ONLY_REPO_ID].tactile_keys == ()
    assert len(routes[VISION_TACTILE_REPO_ID].tactile_keys) == 2
    manifest = build_route_manifest(routes)
    temporal = build_temporal_contract(
        routes, route_manifest_sha256=str(manifest["contract_sha256"])
    )
    assert set(temporal["per_repo_bindings"]) == set(routes)
    for value in temporal["per_repo_bindings"].values():
        assert value["action_offsets_per_anchor"] == list(
            ACTION_OFFSETS_PER_LATENT_ANCHOR
        )
