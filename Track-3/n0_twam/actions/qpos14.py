# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Native AgileX dual-arm 14D absolute joint-position action contract."""

from .base import QuantileActionCodec
from .registry import register_action_codec
from .spec import ActionSpec

ACTION_DIM = 14
CHANNEL_NAMES = (
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_gripper",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_gripper",
)
CHANNEL_UNITS = (
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "normalized_0_1",
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "radian",
    "normalized_0_1",
)
GRIPPER_ENCODING = "0_closed_1_open"


@register_action_codec("qpos14_joint_absolute_v1")
def build_qpos14_codec(
    *,
    q01: tuple[float, ...],
    q99: tuple[float, ...],
    lower_bounds: tuple[float, ...] | None = None,
    upper_bounds: tuple[float, ...] | None = None,
) -> QuantileActionCodec:
    """Build the official left-arm-then-right-arm absolute-qpos codec.

    Quantile statistics only define model normalization. Optional physical
    bounds come from a separately audited robot calibration contract.
    """

    spec = ActionSpec(
        name="qpos14_joint_absolute_v1",
        revision="1",
        dim=ACTION_DIM,
        state_dim=ACTION_DIM,
        semantics="joint_absolute",
        normalization="q01q99",
        padding_policy="repeat_last_with_mask",
        wire_layout="HC",
        server_output_format="absolute",
        channel_names=CHANNEL_NAMES,
        active_channel_ids=tuple(range(ACTION_DIM)),
        gripper_indices=(6, 13),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    return QuantileActionCodec(spec, q01=q01, q99=q99)
