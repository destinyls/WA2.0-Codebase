# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Pure AgileX action-grid and contact-mask sample contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence, cast

import numpy as np
import numpy.typing as npt
import torch

from .sample_shape_signature import SampleShapeSignature, TensorShapeEntry

DATASET_ADAPTER = "worldarena_agilex_qpos14"
ACTION_SCHEMA = "qpos14_joint_absolute_v1"


def validate_digest(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer_vector(value: npt.ArrayLike, *, label: str) -> npt.NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{label} must be a 1D integer array")
    return array.astype(np.int64, copy=False)


def build_action_index_grid(
    *,
    latent_anchor_row_ids: npt.ArrayLike,
    action_offsets_per_anchor: Sequence[int],
    action_row_valid: npt.ArrayLike,
    action_row_count: int,
    action_index_origin: int = 0,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Build an explicit same-row index grid and repeat only padded tail slots."""

    anchors = _integer_vector(latent_anchor_row_ids, label="latent anchors")
    offsets = np.asarray(action_offsets_per_anchor)
    if (
        offsets.ndim != 1
        or offsets.size == 0
        or not np.issubdtype(offsets.dtype, np.integer)
        or np.any(offsets < 0)
        or np.any(np.diff(offsets) <= 0)
        or int(offsets[0]) != 0
    ):
        raise ValueError(
            "action_offsets_per_anchor must be increasing integers starting at 0"
        )
    if anchors.size == 0 or np.any(anchors < action_index_origin):
        raise ValueError("latent anchors must be non-empty and within the action range")
    if anchors.size > 1 and np.any(np.diff(anchors) <= 0):
        raise ValueError("latent anchors must be strictly increasing")
    if (
        isinstance(action_row_count, bool)
        or not isinstance(action_row_count, int)
        or action_row_count <= 0
    ):
        raise ValueError("action_row_count must be a positive integer")
    raw_valid = np.asarray(action_row_valid)
    if raw_valid.dtype != np.bool_ or raw_valid.shape != (action_row_count,):
        raise ValueError("action_row_valid must be boolean [action_row_count]")

    indices = anchors[:, None] + offsets.astype(np.int64)[None, :]
    relative = indices - int(action_index_origin)
    in_range = (relative >= 0) & (relative < action_row_count)
    valid = np.zeros(indices.shape, dtype=np.bool_)
    valid[in_range] = raw_valid[relative[in_range]]
    for row_index in range(indices.shape[0]):
        count = int(valid[row_index].sum())
        expected = np.arange(indices.shape[1]) < count
        if count == 0 or not np.array_equal(valid[row_index], expected):
            raise ValueError("each AgileX action grid row must have a valid prefix")
        if count < indices.shape[1]:
            indices[row_index, count:] = indices[row_index, count - 1]
    flattened = indices[valid]
    if flattened.size > 1 and np.any(np.diff(flattened) <= 0):
        raise ValueError("AgileX valid action grid indices must be strictly increasing")
    return np.ascontiguousarray(indices), np.ascontiguousarray(valid)


def content_addressed_contact_drop(
    *,
    seed: int,
    epoch: int,
    repo_id: str,
    episode_id: int,
    frame_anchor: int,
    probability: float,
) -> bool:
    """Make tactile CFG exposure invariant to rank and sampler ordering."""

    resolved_probability = float(probability)
    if not np.isfinite(resolved_probability) or not 0.0 <= resolved_probability <= 1.0:
        raise ValueError("contact drop probability must be within [0,1]")
    if resolved_probability in (0.0, 1.0):
        return bool(resolved_probability)
    payload = (int(seed), int(epoch), str(repo_id), int(episode_id), int(frame_anchor))
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    draw = int.from_bytes(hashlib.sha256(encoded).digest(), "big") / float(1 << 256)
    return draw < resolved_probability


def build_agilex_sample_shape_signature(
    base_signature: SampleShapeSignature,
    *,
    max_wrench_streams: int,
) -> SampleShapeSignature:
    """Translate one legacy latent signature to the exact AgileX sample roster."""

    if (
        isinstance(max_wrench_streams, bool)
        or not isinstance(max_wrench_streams, int)
        or max_wrench_streams <= 0
    ):
        raise ValueError("max_wrench_streams must be a positive integer")
    entries: dict[str, TensorShapeEntry] = {}
    for entry in base_signature:
        if (
            not isinstance(entry, tuple)
            or len(entry) != 3
            or not isinstance(entry[0], str)
            or not isinstance(entry[1], tuple)
            or not isinstance(entry[2], str)
        ):
            raise ValueError("legacy sample signature contains an invalid entry")
        if entry[0] in entries:
            raise ValueError(f"legacy sample signature repeats {entry[0]!r}")
        entries[entry[0]] = entry

    required = {"actions", "actions_mask", "latents", "tactile_cond_drop"}
    if not required <= set(entries):
        raise ValueError("legacy sample signature is missing required AgileX inputs")
    _, action_shape, action_dtype = entries["actions"]
    if (
        len(action_shape) != 4
        or action_shape[0] != 14
        or action_shape[-1] != 1
        or action_dtype != "torch.float32"
    ):
        raise ValueError("AgileX action signature must be float32 [14,F,H,1]")
    if entries["actions_mask"] != (
        "actions_mask",
        action_shape,
        "torch.bool",
    ):
        raise ValueError("legacy AgileX action-mask signature must match actions")
    if entries["tactile_cond_drop"] != (
        "tactile_cond_drop",
        (),
        "torch.bool",
    ):
        raise ValueError("legacy tactile_cond_drop signature must be scalar bool")
    frames, horizon = int(action_shape[1]), int(action_shape[2])
    latent_shape = entries["latents"][1]
    if len(latent_shape) != 4 or latent_shape[1] != frames:
        raise ValueError("AgileX action and video latent frame counts must match")

    tactile_fields = {
        "tactile_global_latent",
        "tactile_local_latent",
        "tactile_sensor_ids",
    }
    present = tactile_fields & set(entries)
    if present and present != tactile_fields:
        raise ValueError("AgileX tactile signature roster must be complete")
    sensors = 0
    if present:
        global_shape = entries["tactile_global_latent"][1]
        if (
            entries["tactile_global_latent"][1:] != entries["tactile_local_latent"][1:]
            or len(global_shape) != 5
            or global_shape[2] != frames
        ):
            raise ValueError("AgileX global/local tactile signatures must match")
        sensors = int(global_shape[0])
        if entries["tactile_sensor_ids"] != (
            "tactile_sensor_ids",
            (sensors,),
            "torch.int64",
        ):
            raise ValueError("AgileX tactile sensor ID signature is invalid")

    output = {
        name: entry
        for name, entry in entries.items()
        if name not in {"actions_mask", "tactile_cond_drop"}
    }
    canonical: tuple[TensorShapeEntry, ...] = (
        ("action_valid_mask", (frames, horizon), "torch.bool"),
        ("temporal_valid_mask", (frames,), "torch.bool"),
        ("tactile_available_mask", (frames, sensors), "torch.bool"),
        ("wrench", (frames, max_wrench_streams, 6), "torch.float32"),
        ("wrench_available_mask", (frames, max_wrench_streams), "torch.bool"),
        ("contact_cond_drop", (), "torch.bool"),
        ("tactile_condition_mask", (frames, sensors), "torch.bool"),
        ("tactile_target_mask", (frames, sensors), "torch.bool"),
        ("wrench_condition_mask", (frames, max_wrench_streams), "torch.bool"),
        ("repo_route_identity", (), "python.str"),
        ("temporal_alignment_identity", (), "python.str"),
    )
    for entry in canonical:
        if entry[0] in output:
            raise ValueError(f"legacy sample signature already contains {entry[0]!r}")
        output[entry[0]] = entry
    return tuple(sorted(output.values()))


def _canonical_action_mask(sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    actions = sample.get("actions")
    legacy = sample.get("actions_mask")
    if not torch.is_tensor(actions):
        raise ValueError("AgileX actions must have shape [14,F,H,1]")
    actions = cast(torch.Tensor, actions)
    if actions.ndim != 4 or actions.shape[0] != 14:
        raise ValueError("AgileX actions must have shape [14,F,H,1]")
    if actions.shape[-1] != 1 or not torch.isfinite(actions).all():
        raise ValueError("AgileX actions must be finite with singleton tail")
    if not torch.is_tensor(legacy):
        raise ValueError("legacy AgileX actions_mask must match actions")
    legacy = cast(torch.Tensor, legacy)
    if legacy.dtype != torch.bool or legacy.shape != actions.shape:
        raise ValueError("legacy AgileX actions_mask must match actions")
    reference = legacy[0:1].expand_as(legacy)
    if not torch.equal(legacy, reference):
        raise ValueError("AgileX action mask must be channel-invariant")
    return actions, legacy[0, :, :, 0].clone()


def canonicalize_agilex_sample(
    sample: dict[str, Any],
    *,
    tactile_available_mask: torch.Tensor,
    wrench: torch.Tensor,
    wrench_available_mask: torch.Tensor,
    contact_cond_drop: bool,
    repo_route_identity: str,
    temporal_alignment_identity: str,
    temporal_valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Translate the legacy base sample into the canonical AgileX mask contract."""

    if not isinstance(contact_cond_drop, bool):
        raise ValueError("contact_cond_drop must be boolean")
    actions, action_valid = _canonical_action_mask(sample)
    frames = int(actions.shape[1])
    if temporal_valid_mask is None:
        temporal = torch.ones(frames, dtype=torch.bool)
    else:
        temporal = torch.as_tensor(temporal_valid_mask)
        if temporal.dtype != torch.bool:
            raise ValueError("temporal_valid_mask must be boolean [F]")
        temporal = temporal.clone()
    if temporal.shape != (frames,):
        raise ValueError("temporal_valid_mask must have shape [F]")
    tactile_mask = torch.as_tensor(tactile_available_mask).clone()
    if (
        tactile_mask.dtype != torch.bool
        or tactile_mask.ndim != 2
        or tactile_mask.shape[0] != frames
    ):
        raise ValueError("tactile_available_mask must be boolean [F,S]")
    wrench_tensor = torch.as_tensor(wrench, dtype=torch.float32).clone()
    wrench_mask = torch.as_tensor(wrench_available_mask).clone()
    if (
        wrench_tensor.ndim != 3
        or wrench_tensor.shape[0] != frames
        or wrench_tensor.shape[-1] != 6
        or not torch.isfinite(wrench_tensor).all()
    ):
        raise ValueError("AgileX wrench must be finite [F,A,6]")
    if wrench_mask.dtype != torch.bool or wrench_mask.shape != wrench_tensor.shape[:2]:
        raise ValueError("wrench_available_mask must be boolean [F,A]")

    output = dict(sample)
    output.pop("actions_mask", None)
    output.pop("tactile_cond_drop", None)
    action_valid &= temporal[:, None]
    output["actions"] = actions * action_valid[None, :, :, None]
    output["action_valid_mask"] = action_valid
    output["temporal_valid_mask"] = temporal
    output["tactile_available_mask"] = tactile_mask & temporal[:, None]
    output["wrench_available_mask"] = wrench_mask & temporal[:, None]
    output["contact_cond_drop"] = torch.tensor(contact_cond_drop, dtype=torch.bool)
    output["wrench"] = wrench_tensor * output["wrench_available_mask"][..., None]
    output["repo_route_identity"] = validate_digest(
        repo_route_identity, label="repo route identity"
    )
    output["temporal_alignment_identity"] = validate_digest(
        temporal_alignment_identity,
        label="temporal alignment identity",
    )

    sensors = int(tactile_mask.shape[1])
    tactile_keys = ("tactile_global_latent", "tactile_local_latent")
    present = [key for key in tactile_keys if key in output]
    if present and len(present) != 2:
        raise ValueError("AgileX global/local tactile tensors must appear together")
    if present:
        availability = output["tactile_available_mask"].transpose(0, 1)
        for key in tactile_keys:
            tensor = output[key]
            if (
                not torch.is_tensor(tensor)
                or tensor.ndim != 5
                or tensor.shape[0] != sensors
                or tensor.shape[2] != frames
            ):
                raise ValueError(f"{key} must have shape [S,C,F,H,W]")
            output[key] = tensor * availability[:, None, :, None, None]
    elif sensors != 0:
        raise ValueError("declared tactile availability requires tactile tensors")
    dropped = output["contact_cond_drop"]
    output["tactile_condition_mask"] = output["tactile_available_mask"] & ~dropped
    output["wrench_condition_mask"] = output["wrench_available_mask"] & ~dropped
    output["tactile_target_mask"] = output["tactile_available_mask"].clone()
    return output


__all__ = (
    "ACTION_SCHEMA",
    "DATASET_ADAPTER",
    "build_action_index_grid",
    "build_agilex_sample_shape_signature",
    "canonicalize_agilex_sample",
    "content_addressed_contact_drop",
    "validate_digest",
)
