# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Legacy-compatible UniVTAC Panda embodiment identity."""

from .registry import register_action_route_spec, register_embodiment_spec
from .spec import build_action_route_spec, build_embodiment_spec

UNIVTAC_ACTION_ROUTE_SPEC = register_action_route_spec(
    build_action_route_spec(
        name="univtac_qpos8_identity_v1",
        revision="1",
        wire_action_spec="qpos8_next_step",
        model_action_spec="qpos8_next_step",
        temporal_alignment_policy="univtac_qpos8_next_step_v1",
        dataset_to_model_adapter="qpos8_identity_normalized_v1",
        model_to_wire_adapter="qpos8_identity_denormalized_v1",
        active_model_channels=tuple(range(8)),
    )
)

UNIVTAC_EMBODIMENT_SPEC = register_embodiment_spec(
    build_embodiment_spec(
        profile_id="univtac_panda_qpos8_v1",
        revision="1",
        robot_family="panda_single_arm",
        state_schema="qpos8_measured_joint_v1",
        wire_action_schema="qpos8_next_step",
        model_action_schema="qpos8_next_step",
        action_route="univtac_qpos8_identity_v1",
        dataset_adapter="univtac_qpos8",
        policy_adapter="univtac_qpos8",
        safety_contract_id="univtac_qpos8_safety_v1",
    )
)
