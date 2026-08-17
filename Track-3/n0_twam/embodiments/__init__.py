# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Robot embodiment and action-route contracts."""

from .spec import (
    ActionRouteSpec,
    EmbodimentSpec,
    RepoObservationRouteSpec,
    RepoRouteManifestContract,
    build_action_route_spec,
    build_embodiment_spec,
    validate_action_route_contract,
    validate_embodiment_contract,
    validate_repo_route_manifest_contract,
)
from .registry import (
    get_action_route_spec,
    get_embodiment_spec,
    list_action_route_specs,
    list_embodiment_specs,
)

# Importing the three explicit modules registers stable built-ins. There is no
# filesystem discovery, tensor-shape guessing, or environment-dependent route.
from .agilex import (  # noqa: E402
    AGILEX_ACTION_ROUTE_NAME,
    AGILEX_ACTION_ROUTE_SPEC,
    AGILEX_ACTION_SCHEMA,
    AGILEX_DATASET_ADAPTER,
    AGILEX_EMBODIMENT_PROFILE_ID,
    AGILEX_EMBODIMENT_SPEC,
    AGILEX_POLICY_ADAPTER,
    AGILEX_RGB_KEYS,
    AGILEX_TACTILE_KEYS,
    AGILEX_WRENCH_KEYS,
    build_agilex_repo_route_manifest,
)
from .franka import FRANKA_ACTION_ROUTE_SPEC, FRANKA_EMBODIMENT_SPEC  # noqa: E402
from .univtac import (  # noqa: E402
    UNIVTAC_ACTION_ROUTE_SPEC,
    UNIVTAC_EMBODIMENT_SPEC,
)

__all__ = (
    "AGILEX_ACTION_ROUTE_NAME",
    "AGILEX_ACTION_ROUTE_SPEC",
    "AGILEX_ACTION_SCHEMA",
    "AGILEX_DATASET_ADAPTER",
    "AGILEX_EMBODIMENT_PROFILE_ID",
    "AGILEX_EMBODIMENT_SPEC",
    "AGILEX_POLICY_ADAPTER",
    "AGILEX_RGB_KEYS",
    "AGILEX_TACTILE_KEYS",
    "AGILEX_WRENCH_KEYS",
    "ActionRouteSpec",
    "EmbodimentSpec",
    "FRANKA_ACTION_ROUTE_SPEC",
    "FRANKA_EMBODIMENT_SPEC",
    "RepoObservationRouteSpec",
    "RepoRouteManifestContract",
    "UNIVTAC_ACTION_ROUTE_SPEC",
    "UNIVTAC_EMBODIMENT_SPEC",
    "build_action_route_spec",
    "build_agilex_repo_route_manifest",
    "build_embodiment_spec",
    "get_action_route_spec",
    "get_embodiment_spec",
    "list_action_route_specs",
    "list_embodiment_specs",
    "validate_action_route_contract",
    "validate_embodiment_contract",
    "validate_repo_route_manifest_contract",
)
