# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Request identity for the MoT Action Expert cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import torch


@dataclass(frozen=True)
class TensorContract:
    """Allocation-independent tensor contract used in request identity."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class ActionDenoiseIdentity:
    """Semantic and structural identity of one action-denoise request."""

    request_signature: str
    cache_name: str
    frame_grid_signature: tuple[object, ...]
    route_signature: tuple[tuple[str, object], ...]
    prompt_signature: tuple[object, ...]
    fixed_condition_signature: tuple[tuple[str, object], ...]


def _tensor_contract(value: torch.Tensor) -> TensorContract:
    return TensorContract(
        shape=tuple(int(size) for size in value.shape),
        stride=tuple(int(size) for size in value.stride()),
        dtype=str(value.dtype),
        device=str(value.device),
    )


def _stable_scalar(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_stable_scalar(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _stable_scalar(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not torch.is_tensor(item)
        )
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!s}"


def _field_signature(input_dict: Mapping[str, Any], key: str) -> object:
    value = input_dict.get(key)
    if torch.is_tensor(value):
        return _tensor_contract(value)
    return _stable_scalar(value)


def build_action_denoise_identity(
    input_dict: Mapping[str, Any],
    *,
    cache_name: str,
    request_signature: str,
) -> ActionDenoiseIdentity:
    """Bind a serving content digest to model-visible fixed conditions."""

    if not isinstance(request_signature, str) or not request_signature.strip():
        raise ValueError("request_signature must be a non-empty string")
    if not isinstance(cache_name, str) or not cache_name:
        raise ValueError("cache_name must be a non-empty string")

    grid = input_dict.get("grid_id")
    if not torch.is_tensor(grid):
        raise ValueError("action cache runtime requires tensor grid_id")
    grid = cast(torch.Tensor, grid)
    text = input_dict.get("text_emb")
    if not torch.is_tensor(text):
        raise ValueError("action cache runtime requires tensor text_emb")
    text = cast(torch.Tensor, text)

    route_keys = (
        "embodiment",
        "embodiment_id",
        "route_id",
        "tactile_mode",
        "tactile_profile",
        "action_profile",
        "action_representation",
    )
    prompt_keys = ("prompt_id", "prompt_signature", "instruction", "task")
    fixed_keys = (
        "tactile_global_latent",
        "tactile_local_latent",
        "tactile_sensor_ids",
        "wrench",
        "wrench_available_mask",
        "temporal_valid_mask",
        "contact_cond_drop",
        "tactile_cond_drop",
    )
    signature = request_signature.strip()
    frame_start_id = input_dict.get("frame_start_id")
    if frame_start_id is not None and (
        isinstance(frame_start_id, bool)
        or not isinstance(frame_start_id, int)
        or frame_start_id < 0
    ):
        raise ValueError("frame_start_id must be a non-negative integer when provided")
    return ActionDenoiseIdentity(
        request_signature=signature,
        cache_name=cache_name,
        frame_grid_signature=(
            signature,
            _tensor_contract(grid),
            frame_start_id,
        ),
        route_signature=tuple(
            (key, _field_signature(input_dict, key))
            for key in route_keys
            if key in input_dict
        ),
        prompt_signature=(
            signature,
            _tensor_contract(text),
            tuple(
                (key, _field_signature(input_dict, key))
                for key in prompt_keys
                if key in input_dict
            ),
        ),
        fixed_condition_signature=tuple(
            (key, (signature, _field_signature(input_dict, key)))
            for key in fixed_keys
            if key in input_dict
        ),
    )


def action_denoise_identity_mismatch(
    expected: ActionDenoiseIdentity,
    actual: ActionDenoiseIdentity,
) -> str | None:
    checks = (
        (
            expected.request_signature == actual.request_signature,
            "request signature mismatch",
        ),
        (expected.cache_name == actual.cache_name, "rolling cache name mismatch"),
        (
            expected.frame_grid_signature == actual.frame_grid_signature,
            "frame/grid identity mismatch",
        ),
        (expected.route_signature == actual.route_signature, "route identity mismatch"),
        (
            expected.prompt_signature == actual.prompt_signature,
            "prompt identity mismatch",
        ),
        (
            expected.fixed_condition_signature == actual.fixed_condition_signature,
            "fixed condition identity mismatch",
        ),
    )
    for compatible, reason in checks:
        if not compatible:
            return reason
    return None


__all__ = ["ActionDenoiseIdentity", "TensorContract"]
