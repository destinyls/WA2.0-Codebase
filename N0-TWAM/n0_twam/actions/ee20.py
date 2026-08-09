# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Legacy 20D dual-arm end-effector action contracts."""

import numpy as np
import numpy.typing as npt

from .base import FloatArray, QuantileActionCodec
from .registry import register_action_codec
from .spec import ActionSpec

_ARM_CHANNELS = (
    "x",
    "y",
    "z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper",
)
_CHANNEL_NAMES = tuple(
    f"{arm}_{channel}" for arm in ("left", "right") for channel in _ARM_CHANNELS
)


def _build_ee20_codec(
    *,
    name: str,
    semantics: str,
    q01: tuple[float, ...],
    q99: tuple[float, ...],
    lower_bounds: tuple[float, ...] | None = None,
    upper_bounds: tuple[float, ...] | None = None,
) -> QuantileActionCodec:
    spec = ActionSpec(
        name=name,
        revision="1",
        dim=20,
        state_dim=20,
        semantics=semantics,
        normalization="q01q99",
        padding_policy="legacy_dataset_mask",
        wire_layout="CFH",
        server_output_format="absolute",
        channel_names=_CHANNEL_NAMES,
        active_channel_ids=tuple(range(20)),
        gripper_indices=(9, 19),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    return QuantileActionCodec(spec, q01=q01, q99=q99)


@register_action_codec("ee20_absee")
def build_ee20_absee_codec(
    *,
    q01: tuple[float, ...],
    q99: tuple[float, ...],
    lower_bounds: tuple[float, ...] | None = None,
    upper_bounds: tuple[float, ...] | None = None,
) -> QuantileActionCodec:
    return _build_ee20_codec(
        name="ee20_absee",
        semantics="absolute_end_effector",
        q01=q01,
        q99=q99,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


PI05_DELTA_CHANNEL_IDS = tuple(range(0, 9)) + tuple(range(10, 19))


def _validate_pi05_inputs(
    values: npt.ArrayLike,
    current_state: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    action = np.asarray(values, dtype=np.float32)
    state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if action.ndim != 3 or action.shape[-1] != 20:
        raise ValueError("pi05 action must have shape [frames, slots, 20]")
    if state.shape[0] < 19:
        raise ValueError("pi05 current_state must contain at least 19 channels")
    if not np.isfinite(action).all() or not np.isfinite(state[:19]).all():
        raise ValueError("pi05 action/state contains non-finite values")
    return action.copy(), state


def absolute_to_pi05_delta(
    values: npt.ArrayLike,
    *,
    current_state: npt.ArrayLike,
) -> FloatArray:
    """Convert absolute EE targets to legacy per-frame pi0.5 deltas."""

    action, state = _validate_pi05_inputs(values, current_state)
    original = action.copy()
    for channel in PI05_DELTA_CHANNEL_IDS:
        for frame in range(action.shape[0]):
            anchor = state[channel] if frame == 0 else original[frame - 1, -1, channel]
            action[frame, :, channel] -= anchor
    return action


def pi05_delta_to_absolute(
    values: npt.ArrayLike,
    *,
    current_state: npt.ArrayLike,
) -> FloatArray:
    """Reconstruct absolute EE targets from legacy per-frame pi0.5 deltas."""

    action, state = _validate_pi05_inputs(values, current_state)
    for channel in PI05_DELTA_CHANNEL_IDS:
        anchor = state[channel]
        for frame in range(action.shape[0]):
            action[frame, :, channel] += anchor
            anchor = action[frame, -1, channel]
    return action


@register_action_codec("ee20_pi05")
def build_ee20_pi05_codec(
    *,
    q01: tuple[float, ...],
    q99: tuple[float, ...],
    lower_bounds: tuple[float, ...] | None = None,
    upper_bounds: tuple[float, ...] | None = None,
) -> QuantileActionCodec:
    return _build_ee20_codec(
        name="ee20_pi05",
        semantics="delta_end_effector",
        q01=q01,
        q99=q99,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
