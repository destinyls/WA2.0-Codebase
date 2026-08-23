# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Align native-15-Hz Franka EE10 targets to the 10-Hz Wan latent grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class FrankaEE10LatentTargets:
    actions: npt.NDArray[np.float32]
    valid_mask: npt.NDArray[np.bool_]
    anchor_frame_ids: npt.NDArray[np.int64]
    slots_per_frame: int


def _numpy(values: npt.ArrayLike) -> npt.NDArray[np.generic]:
    detach = getattr(values, "detach", None)
    if callable(detach):
        values = detach().cpu().numpy()
    return np.asarray(values)


def _anchors(latent_frame_ids: npt.ArrayLike) -> npt.NDArray[np.int64]:
    frame_ids = _numpy(latent_frame_ids)
    if frame_ids.ndim != 1 or frame_ids.size == 0:
        raise ValueError("latent_frame_ids must be a non-empty 1D sequence")
    if not np.issubdtype(frame_ids.dtype, np.integer):
        raise ValueError("latent_frame_ids must contain integers")
    frame_ids = frame_ids.astype(np.int64, copy=False)
    if frame_ids.size > 1 and not np.all(np.diff(frame_ids) > 0):
        raise ValueError("latent_frame_ids must be strictly increasing")
    if (frame_ids.size - 1) % 4 != 0:
        raise ValueError("latent_frame_ids count must follow the 4n+1 Wan grid")
    return np.ascontiguousarray(frame_ids[::4])


def build_franka_ee10_latent_targets(
    *,
    local_start_frame: int,
    local_end_frame: int,
    latent_frame_ids: npt.ArrayLike,
    converted_actions: npt.ArrayLike,
    converted_states: npt.ArrayLike,
    action_q01: npt.ArrayLike,
    action_q99: npt.ArrayLike,
    expected_slots_per_frame: int = 6,
) -> FrankaEE10LatentTargets:
    """Build EE20 targets with only the left/Franka half marked valid.

    Converted row ``t`` is already the next recorded absolute end pose for
    observation row ``t``. The 15→10 RGB sampler is non-uniform, but every four
    sampled RGB frames advances exactly six source rows, so Wan anchors are
    strictly ``0, 6, 12, ...`` within a segment. As in the released N0 action
    aligner, action frame zero is a repeated cold conditioning slot containing
    the actual end pose observed at the first video anchor. Action frame
    ``f>0`` contains the six targets following video anchor ``f-1``.
    """

    if local_start_frame < 0 or local_end_frame <= local_start_frame:
        raise ValueError("invalid local Franka segment bounds")
    if expected_slots_per_frame != 6:
        raise ValueError("Franka native-15-Hz contract requires six action slots")
    actions = np.asarray(_numpy(converted_actions), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 10 or actions.shape[0] == 0:
        raise ValueError("converted Franka actions must have shape [T,10]")
    if not np.isfinite(actions).all():
        raise ValueError("converted Franka actions contain non-finite values")
    if actions.shape[0] != local_end_frame - local_start_frame:
        raise ValueError("converted action count differs from segment bounds")
    states = np.asarray(_numpy(converted_states), dtype=np.float32)
    if states.shape != actions.shape:
        raise ValueError("converted Franka states must match action shape [T,10]")
    if not np.isfinite(states).all():
        raise ValueError("converted Franka states contain non-finite values")
    q01 = np.asarray(action_q01, dtype=np.float32)
    q99 = np.asarray(action_q99, dtype=np.float32)
    if q01.shape != (10,) or q99.shape != (10,) or not np.isfinite(q01).all():
        raise ValueError("Franka normalizer must contain finite EE10 q01/q99")
    if not np.isfinite(q99).all() or np.any(q99 < q01):
        raise ValueError("Franka normalizer channel ranges are invalid")

    anchors = _anchors(latent_frame_ids)
    if int(anchors[0]) < local_start_frame or int(anchors[-1]) >= local_end_frame:
        raise ValueError("Franka latent anchors must remain inside the segment")
    if anchors.size > 1 and not np.all(np.diff(anchors) == 6):
        raise ValueError("Franka latent anchors must advance six native 15-Hz rows")

    frame_count = int(anchors.size)
    first_anchor_offset = int(anchors[0]) - local_start_frame
    current_state = states[first_anchor_offset]
    raw = np.repeat(current_state[None, None, :], frame_count * 6, axis=0)
    raw = raw.reshape(frame_count, 6, 10)
    valid = np.ones_like(raw, dtype=np.bool_)
    for latent_index, anchor in enumerate(anchors[:-1], start=1):
        offset = int(anchor) - local_start_frame
        valid_count = min(6, actions.shape[0] - offset)
        if valid_count <= 0:
            raise ValueError("every predicted Franka action frame needs a target")
        if valid_count < 6:
            raw[latent_index] = actions[-1]
            valid[latent_index] = False
        raw[latent_index, :valid_count] = actions[offset : offset + valid_count]
        valid[latent_index, :valid_count] = True

    normalized = (raw - q01) / (q99 - q01 + np.float32(1e-6)) * 2.0 - 1.0
    normalized *= valid
    model_actions = np.zeros((frame_count, 6, 20), dtype=np.float32)
    model_mask = np.zeros((frame_count, 6, 20), dtype=np.bool_)
    model_actions[..., :10] = normalized
    model_mask[..., :10] = valid
    return FrankaEE10LatentTargets(
        actions=np.ascontiguousarray(
            np.transpose(model_actions, (2, 0, 1))[:, :, :, None]
        ),
        valid_mask=np.ascontiguousarray(
            np.transpose(model_mask, (2, 0, 1))[:, :, :, None]
        ),
        anchor_frame_ids=anchors,
        slots_per_frame=6,
    )


__all__ = ("FrankaEE10LatentTargets", "build_franka_ee10_latent_targets")
