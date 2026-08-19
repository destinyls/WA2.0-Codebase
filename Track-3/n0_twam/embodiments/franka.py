# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Legacy-compatible Franka embodiment identity."""

from n0_twam.integrations.worldarena.franka_actions import (
    FRANKA_ACTION_ROUTE_ID,
    FRANKA_ACTION_SCHEMA,
    FRANKA_EMBODIMENT_PROFILE_ID,
)

from .registry import register_action_route_spec, register_embodiment_spec
from .spec import build_action_route_spec, build_embodiment_spec

FRANKA_ACTION_ROUTE_SPEC = register_action_route_spec(
    build_action_route_spec(
        name=FRANKA_ACTION_ROUTE_ID,
        revision="2",
        wire_action_spec=FRANKA_ACTION_SCHEMA,
        model_action_spec="ee20_absee",
        temporal_alignment_policy="franka_pose8_xyzw_window_v2",
        dataset_to_model_adapter="franka_pose8_xyzw_to_ee10_to_ee20_v2",
        model_to_wire_adapter="franka_ee20_to_pose8_xyzw_v2",
        active_model_channels=tuple(range(10)),
    )
)

FRANKA_EMBODIMENT_SPEC = register_embodiment_spec(
    build_embodiment_spec(
        profile_id=FRANKA_EMBODIMENT_PROFILE_ID,
        revision="2",
        robot_family="franka_single_arm",
        state_schema=FRANKA_ACTION_SCHEMA,
        wire_action_schema=FRANKA_ACTION_SCHEMA,
        model_action_schema="ee20_absee",
        action_route=FRANKA_ACTION_ROUTE_ID,
        dataset_adapter="worldarena_franka_ee10",
        policy_adapter="worldarena_franka_pose8",
        safety_contract_id="franka_signed_pose_safety_v1",
    )
)
