# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Align converted qpos8 targets to N0's temporally compressed latent grid."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from n0_twam.actions.base import ActionCodec, FloatArray


@dataclass(frozen=True)
class Qpos8LatentTargets:
    """Normalized actions and valid-loss mask in N0 ``[C,F,H,1]`` layout."""

    actions: FloatArray
    valid_mask: npt.NDArray[np.bool_]
    anchor_frame_ids: npt.NDArray[np.int64]
    slots_per_frame: int


def _to_numpy(value: npt.ArrayLike) -> npt.NDArray[np.generic]:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return np.asarray(value)


def _latent_anchor_ids(latent_frame_ids: npt.ArrayLike) -> npt.NDArray[np.int64]:
    frame_ids = _to_numpy(latent_frame_ids)
    if frame_ids.ndim != 1 or frame_ids.size == 0:
        raise ValueError("latent_frame_ids must be a non-empty 1D sequence")
    if not np.issubdtype(frame_ids.dtype, np.integer):
        raise ValueError("latent_frame_ids must contain integers")
    frame_ids = frame_ids.astype(np.int64, copy=False)
    if frame_ids.size > 1 and not np.all(np.diff(frame_ids) > 0):
        raise ValueError("latent_frame_ids must be strictly increasing")
    anchors = np.ascontiguousarray(frame_ids[::4])
    anchors.setflags(write=False)
    return anchors


def build_qpos8_latent_targets(
    *,
    local_start_frame: int,
    local_end_frame: int,
    latent_frame_ids: npt.ArrayLike,
    converted_actions: npt.ArrayLike,
    codec: ActionCodec,
    expected_slots_per_frame: int,
) -> Qpos8LatentTargets:
    """Build next-step targets without the legacy leading-condition padding.

    ``converted_actions[t]`` is already ``joint[t+1]``. Each latent anchor at
    source frame ``t`` therefore starts directly from action row ``t``. Terminal
    slots repeat the last target for a stable tensor shape but are masked out.
    """

    if codec.spec.name != "qpos8_next_step" or codec.spec.dim != 8:
        raise ValueError("qpos8 latent alignment requires qpos8_next_step codec")
    if local_start_frame < 0:
        raise ValueError("local_start_frame must be non-negative")
    if local_end_frame <= local_start_frame:
        raise ValueError("local_end_frame must be greater than local_start_frame")
    if expected_slots_per_frame <= 0:
        raise ValueError("expected_slots_per_frame must be positive")

    actions = codec.validate(_to_numpy(converted_actions))
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError("converted_actions must have shape [timesteps, 8]")
    expected_action_count = int(local_end_frame) - int(local_start_frame)
    if actions.shape[0] != expected_action_count:
        raise ValueError(
            "converted action count does not match segment bounds: "
            f"{actions.shape[0]} vs {expected_action_count}"
        )
    anchors = _latent_anchor_ids(latent_frame_ids)
    if (len(_to_numpy(latent_frame_ids)) - 1) % 4 != 0:
        raise ValueError("latent_frame_ids count must follow the 4n+1 Wan grid")
    if int(anchors[0]) < local_start_frame or int(anchors[-1]) >= local_end_frame:
        raise ValueError("latent anchors must stay within segment bounds")
    if anchors.size > 1:
        anchor_strides = np.diff(anchors)
        if not np.all(anchor_strides == anchor_strides[0]):
            raise ValueError("latent anchor spacing must be uniform")
        derived_slots = int(anchor_strides[0])
        if derived_slots != expected_slots_per_frame:
            raise ValueError(
                "action_per_frame does not match latent anchor spacing: "
                f"{expected_slots_per_frame} vs {derived_slots}"
            )

    frame_count = int(anchors.size)
    horizon = int(expected_slots_per_frame)
    raw = np.repeat(actions[-1][None, None, :], frame_count * horizon, axis=0)
    raw = raw.reshape(frame_count, horizon, codec.spec.dim)
    valid = np.zeros_like(raw, dtype=np.bool_)

    for latent_index, anchor_frame in enumerate(anchors):
        action_offset = int(anchor_frame) - int(local_start_frame)
        if action_offset < 0:
            raise ValueError("latent anchor precedes the segment start")
        valid_count = min(horizon, max(0, actions.shape[0] - action_offset))
        if valid_count == 0:
            raise ValueError("every latent anchor must have a next-step action target")
        if valid_count < horizon and latent_index != frame_count - 1:
            raise ValueError("only the terminal latent anchor may contain padding")
        raw[latent_index, :valid_count] = actions[
            action_offset : action_offset + valid_count
        ]
        valid[latent_index, :valid_count] = True

    normalized = codec.normalize(raw)
    normalized *= valid
    model_actions = np.transpose(normalized, (2, 0, 1))[:, :, :, None]
    model_mask = np.transpose(valid, (2, 0, 1))[:, :, :, None]
    return Qpos8LatentTargets(
        actions=np.ascontiguousarray(model_actions, dtype=np.float32),
        valid_mask=np.ascontiguousarray(model_mask, dtype=np.bool_),
        anchor_frame_ids=anchors,
        slots_per_frame=horizon,
    )
