# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Legacy-compatible Franka embodiment identity."""

from .registry import register_action_route_spec, register_embodiment_spec
from .spec import build_action_route_spec, build_embodiment_spec

FRANKA_ACTION_ROUTE_SPEC = register_action_route_spec(
    build_action_route_spec(
        name="franka_pose8_to_ee20_v1",
        revision="1",
        wire_action_spec="franka_end_pose_base_wxyz8_v1",
        model_action_spec="ee20_absee",
        temporal_alignment_policy="franka_pose8_window_v1",
        dataset_to_model_adapter="franka_pose8_to_ee10_to_ee20_v1",
        model_to_wire_adapter="franka_ee20_to_pose8_wxyz_v1",
        active_model_channels=tuple(range(10)),
    )
)

FRANKA_EMBODIMENT_SPEC = register_embodiment_spec(
    build_embodiment_spec(
        profile_id="franka_pose8_ee20_v1",
        revision="1",
        robot_family="franka_single_arm",
        state_schema="franka_end_pose_base_wxyz8_v1",
        wire_action_schema="franka_end_pose_base_wxyz8_v1",
        model_action_schema="ee20_absee",
        action_route="franka_pose8_to_ee20_v1",
        dataset_adapter="worldarena_franka_ee10",
        policy_adapter="worldarena_franka_pose8",
        safety_contract_id="franka_signed_pose_safety_v1",
    )
)
