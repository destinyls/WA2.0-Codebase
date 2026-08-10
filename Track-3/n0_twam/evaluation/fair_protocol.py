# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed conditioning contract for fair future-tactile prediction."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Protocol

import torch

CAUSAL_FUTURE_ONLY_PROTOCOL = "causal_future_only_v1"

_MODEL_INPUT_KEYS = (
    "latents",
    "actions",
    "actions_mask",
    "text_emb",
    "tactile_global_latent",
    "tactile_local_latent",
    "tactile_sensor_ids",
)


class _Hash(Protocol):
    def update(self, data: bytes, /) -> object: ...


def _required_tensor(batch: Mapping[str, object], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} must be present as a tensor")
    return value


def _repeat_frame_zero(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if value.ndim < 3:
        raise ValueError(f"{label} must expose a temporal axis at dimension -3")
    if value.shape[-3] <= 1:
        raise ValueError(f"{label} must contain at least one future frame")
    return value[..., 0:1, :, :].expand_as(value).clone()


def sanitize_causal_future_only_batch(
    batch: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Build the only batch allowed by ``causal_future_only_v1``.

    The returned mapping is a strict allowlist. Video and both tactile streams
    contain frame 0 repeated over time; the entire action stream and its mask
    are zero. Consequently no future video, tactile, or action target can reach
    ``Trainer._prepare_input_dict``. ``text_emb`` and tactile sensor identities
    are retained because they contain task/sensor metadata rather than future
    observations.
    """

    tensors = {key: _required_tensor(batch, key) for key in _MODEL_INPUT_KEYS}
    video = tensors["latents"]
    actions = tensors["actions"]
    action_mask = tensors["actions_mask"]
    global_tactile = tensors["tactile_global_latent"]
    local_tactile = tensors["tactile_local_latent"]
    sensor_ids = tensors["tactile_sensor_ids"]
    text_emb = tensors["text_emb"]

    if video.ndim != 5:
        raise ValueError("latents must use [B,C,F,H,W] layout")
    if global_tactile.ndim != 6 or local_tactile.shape != global_tactile.shape:
        raise ValueError("global/local tactile tensors must share [B,S,C,F,H,W] layout")
    if actions.shape != action_mask.shape:
        raise ValueError("actions and actions_mask must have identical shapes")
    if actions.ndim < 3:
        raise ValueError("actions must expose a temporal axis")
    if action_mask.dtype is not torch.bool:
        raise ValueError("actions_mask must use bool dtype")
    if video.shape[0] != global_tactile.shape[0]:
        raise ValueError("video and tactile batch dimensions differ")
    if video.shape[-3] != global_tactile.shape[-3]:
        raise ValueError("video and tactile frame counts differ")
    if actions.shape[0] != video.shape[0]:
        raise ValueError("action and video batch dimensions differ")
    expected_sensor_ids = global_tactile.shape[0] * global_tactile.shape[1]
    if sensor_ids.numel() != expected_sensor_ids:
        raise ValueError("tactile_sensor_ids do not match tactile streams")
    if text_emb.numel() == 0 or text_emb.shape[0] != video.shape[0]:
        raise ValueError("text_emb must be non-empty and batch-aligned")

    return {
        "latents": _repeat_frame_zero(video, label="latents"),
        "actions": torch.zeros_like(actions),
        "actions_mask": torch.zeros_like(action_mask),
        "text_emb": text_emb.clone(),
        "tactile_global_latent": _repeat_frame_zero(
            global_tactile, label="tactile_global_latent"
        ),
        "tactile_local_latent": _repeat_frame_zero(
            local_tactile, label="tactile_local_latent"
        ),
        "tactile_sensor_ids": sensor_ids.clone(),
    }


def _update_payload_hash(digest: _Hash, value: object) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor:")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b":")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping{")
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("prediction payload mapping keys must be strings")
        for key in sorted(keys):
            digest.update(json.dumps(key, ensure_ascii=False).encode("utf-8"))
            digest.update(b":")
            _update_payload_hash(digest, value[key])
            digest.update(b",")
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence[")
        for item in value:
            _update_payload_hash(digest, item)
            digest.update(b",")
        digest.update(b"]")
        return
    if value is None or isinstance(value, (bool, int, str)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(b":")
        digest.update(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("prediction payload contains a non-finite float")
        digest.update(b"float:")
        digest.update(value.hex().encode("ascii"))
        return
    raise TypeError(f"unsupported prediction payload value: {type(value)!r}")


def causal_conditioning_sha256(payload: Mapping[str, object]) -> str:
    """Hash a sanitized or prepared model payload without lossy conversion."""

    digest = hashlib.sha256()
    _update_payload_hash(digest, payload)
    return digest.hexdigest()


def verify_future_gt_invariance(
    reference_batch: Mapping[str, object],
    future_perturbed_batch: Mapping[str, object],
) -> dict[str, object]:
    """Fail unless two raw samples reduce to the same causal conditioning."""

    first = sanitize_causal_future_only_batch(reference_batch)
    second = sanitize_causal_future_only_batch(future_perturbed_batch)
    first_hash = causal_conditioning_sha256(first)
    second_hash = causal_conditioning_sha256(second)
    if first_hash != second_hash:
        raise ValueError(
            "future-GT invariance failed: frame0/task/sensor conditioning differs"
        )
    return {
        "protocol_id": CAUSAL_FUTURE_ONLY_PROTOCOL,
        "conditioning_sha256": first_hash,
        "future_gt_invariant": True,
        "video_conditioning": "frame0_repeated",
        "tactile_conditioning": "frame0_repeated",
        "action_conditioning": "all_zero_with_zero_mask",
        "task_conditioning": "text_emb",
    }


__all__ = [
    "CAUSAL_FUTURE_ONLY_PROTOCOL",
    "causal_conditioning_sha256",
    "sanitize_causal_future_only_batch",
    "verify_future_gt_invariance",
]
