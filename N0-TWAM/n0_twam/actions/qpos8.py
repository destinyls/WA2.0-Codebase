# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Native UniVTAC 8D next-step joint-position action contract."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .base import FloatArray, QuantileActionCodec
from .registry import register_action_codec
from .spec import ActionSpec

ACTION_DIM = 8
CHANNEL_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
)


@dataclass(frozen=True)
class Qpos8Windows:
    """Current states and strictly next-step targets with terminal masks."""

    states: FloatArray
    actions: FloatArray
    is_pad: npt.NDArray[np.bool_]


@register_action_codec("qpos8_next_step")
def build_qpos8_codec(
    *,
    q01: tuple[float, ...],
    q99: tuple[float, ...],
    lower_bounds: tuple[float, ...] | None = None,
    upper_bounds: tuple[float, ...] | None = None,
) -> QuantileActionCodec:
    spec = ActionSpec(
        name="qpos8_next_step",
        revision="1",
        dim=ACTION_DIM,
        state_dim=ACTION_DIM,
        semantics="joint_absolute_next_step",
        normalization="q01q99",
        padding_policy="repeat_last_with_mask",
        wire_layout="HC",
        server_output_format="absolute",
        channel_names=CHANNEL_NAMES,
        active_channel_ids=tuple(range(ACTION_DIM)),
        gripper_indices=(7,),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    return QuantileActionCodec(spec, q01=q01, q99=q99)


def build_next_step_windows(
    joint: npt.ArrayLike,
    *,
    horizon: int,
) -> Qpos8Windows:
    """Create ``q[t] -> q[t+1:t+1+H]`` targets without double shifting.

    Terminal slots repeat the last valid joint target but are always marked in
    ``is_pad``. Losses and metrics must ignore every padded slot.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    joint_array = np.asarray(joint, dtype=np.float32)
    if joint_array.ndim != 2:
        raise ValueError("joint must have shape [timesteps, channels]")
    if joint_array.shape[0] < 2:
        raise ValueError("joint must contain at least two timesteps")
    if joint_array.shape[1] < ACTION_DIM:
        raise ValueError(f"joint must contain at least {ACTION_DIM} channels")
    if not np.isfinite(joint_array[:, :ACTION_DIM]).all():
        raise ValueError("joint contains non-finite qpos8 values")

    qpos = joint_array[:, :ACTION_DIM]
    num_samples = qpos.shape[0] - 1
    states = qpos[:-1].copy()
    actions = np.repeat(qpos[-1][None, None], num_samples * horizon, axis=0)
    actions = actions.reshape(num_samples, horizon, ACTION_DIM)
    is_pad = np.ones((num_samples, horizon), dtype=np.bool_)

    for sample_index in range(num_samples):
        valid_count = min(horizon, qpos.shape[0] - sample_index - 1)
        target_end = sample_index + 1 + valid_count
        actions[sample_index, :valid_count] = qpos[sample_index + 1 : target_end]
        is_pad[sample_index, :valid_count] = False

    return Qpos8Windows(states=states, actions=actions, is_pad=is_pad)
