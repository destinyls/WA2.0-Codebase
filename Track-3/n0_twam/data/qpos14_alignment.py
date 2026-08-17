# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Map explicit AgileX action rows to N0's compressed latent grid."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from n0_twam.actions.base import ActionCodec, FloatArray

ACTION_DIM = 14
ACTION_SCHEMA = "qpos14_joint_absolute_v1"


@dataclass(frozen=True)
class Qpos14LatentTargets:
    """Normalized qpos14 targets plus canonical and model-layout masks."""

    actions: FloatArray
    action_valid_mask: npt.NDArray[np.bool_]
    model_valid_mask: npt.NDArray[np.bool_]
    latent_anchor_row_ids: npt.NDArray[np.int64]
    action_indices_per_anchor: npt.NDArray[np.int64]
    slots_per_frame: int


def _to_numpy(value: npt.ArrayLike) -> npt.NDArray[np.generic]:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return np.asarray(value)


def _immutable_copy(
    value: npt.NDArray[np.generic],
) -> npt.NDArray[np.generic]:
    output = np.ascontiguousarray(value).copy()
    output.setflags(write=False)
    return output


def _validate_integer_array(
    value: npt.ArrayLike,
    *,
    field_name: str,
    ndim: int,
) -> npt.NDArray[np.int64]:
    array = _to_numpy(value)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{field_name} must be a {ndim}D integer array")
    return array.astype(np.int64, copy=False)


def _validate_mapping_contract(
    *,
    action_indices: npt.NDArray[np.int64],
    valid_mask: npt.NDArray[np.bool_],
) -> None:
    valid_counts = valid_mask.sum(axis=1, dtype=np.int64)
    if np.any(valid_counts == 0):
        raise ValueError("every latent anchor must have at least one valid action")

    prefix_mask = np.arange(valid_mask.shape[1])[None, :] < valid_counts[:, None]
    if not np.array_equal(valid_mask, prefix_mask):
        raise ValueError("action_valid_mask must be a valid prefix in every row")
    if valid_mask.shape[0] > 1 and np.any(~valid_mask[:-1]):
        raise ValueError("only the terminal latent anchor may contain padding")

    for row_index, valid_count in enumerate(valid_counts.tolist()):
        if valid_count < valid_mask.shape[1]:
            last_valid_index = action_indices[row_index, valid_count - 1]
            if not np.all(action_indices[row_index, valid_count:] == last_valid_index):
                raise ValueError("padded slots must repeat the last valid action index")

    flattened_valid_indices = action_indices[valid_mask]
    if flattened_valid_indices.size > 1 and np.any(
        np.diff(flattened_valid_indices) <= 0
    ):
        raise ValueError("valid action indices must be strictly increasing")


def build_qpos14_latent_targets(
    *,
    converted_actions: npt.ArrayLike,
    codec: ActionCodec,
    action_indices_per_anchor: npt.ArrayLike,
    action_valid_mask: npt.ArrayLike,
    latent_anchor_row_ids: npt.ArrayLike,
    action_index_origin: int = 0,
) -> Qpos14LatentTargets:
    """Normalize explicitly mapped qpos14 targets without guessing rates.

    ``action_indices_per_anchor`` and ``action_valid_mask`` are emitted by the
    frozen temporal-alignment contract. Indices are consumed exactly as given;
    this function never applies an implicit next-step shift or derives a fixed
    action-per-frame ratio. A segment-local ``converted_actions`` array can use
    ``action_index_origin`` to retain the source-global action indices.
    """

    if codec.spec.name != ACTION_SCHEMA or codec.spec.dim != ACTION_DIM:
        raise ValueError(
            "qpos14 latent alignment requires qpos14_joint_absolute_v1 codec"
        )
    if (
        isinstance(action_index_origin, bool)
        or not isinstance(action_index_origin, (int, np.integer))
        or action_index_origin < 0
    ):
        raise ValueError("action_index_origin must be a non-negative integer")

    unvalidated_actions = _to_numpy(converted_actions)
    if (
        unvalidated_actions.ndim != 2
        or unvalidated_actions.shape[0] == 0
        or unvalidated_actions.shape[1] != ACTION_DIM
    ):
        raise ValueError("converted_actions must have shape [timesteps, 14]")
    actions = codec.validate(unvalidated_actions)

    action_indices = _validate_integer_array(
        action_indices_per_anchor,
        field_name="action_indices_per_anchor",
        ndim=2,
    )
    if action_indices.shape[0] == 0 or action_indices.shape[1] == 0:
        raise ValueError("action_indices_per_anchor must be non-empty")

    unvalidated_mask = _to_numpy(action_valid_mask)
    if unvalidated_mask.dtype != np.bool_:
        raise ValueError("action_valid_mask must have boolean dtype")
    if unvalidated_mask.shape != action_indices.shape:
        raise ValueError("action_valid_mask must match action_indices_per_anchor shape")
    valid_mask = unvalidated_mask.astype(np.bool_, copy=False)

    anchor_ids = _validate_integer_array(
        latent_anchor_row_ids,
        field_name="latent_anchor_row_ids",
        ndim=1,
    )
    if anchor_ids.shape != (action_indices.shape[0],):
        raise ValueError("latent_anchor_row_ids must contain one ID per mapping row")
    if np.any(anchor_ids < 0):
        raise ValueError("latent_anchor_row_ids must be non-negative")
    if anchor_ids.size > 1 and np.any(np.diff(anchor_ids) <= 0):
        raise ValueError("latent_anchor_row_ids must be strictly increasing")

    _validate_mapping_contract(
        action_indices=action_indices,
        valid_mask=valid_mask,
    )
    relative_indices = action_indices - int(action_index_origin)
    if np.any(relative_indices < 0) or np.any(relative_indices >= actions.shape[0]):
        raise ValueError("action_indices_per_anchor must stay within converted_actions")

    selected_actions = actions[relative_indices]
    normalized = codec.normalize(selected_actions)
    normalized = np.where(valid_mask[..., None], normalized, 0.0).astype(
        np.float32,
        copy=False,
    )
    model_actions = np.transpose(normalized, (2, 0, 1))[:, :, :, None]
    model_valid_mask = np.broadcast_to(
        valid_mask[None, :, :, None],
        model_actions.shape,
    )

    return Qpos14LatentTargets(
        actions=_immutable_copy(model_actions),
        action_valid_mask=_immutable_copy(valid_mask),
        model_valid_mask=_immutable_copy(model_valid_mask),
        latent_anchor_row_ids=_immutable_copy(anchor_ids),
        action_indices_per_anchor=_immutable_copy(action_indices),
        slots_per_frame=int(action_indices.shape[1]),
    )
