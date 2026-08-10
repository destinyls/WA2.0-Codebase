# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Compatibility helpers for resolving codecs from existing EasyDict configs."""

from typing import Protocol

from .base import ActionCodec
from .registry import build_action_codec

_PI05_MODES = frozenset(("pi05_delta", "openpi_delta", "pi0.5_delta"))


class ActionConfig(Protocol):
    """Structural subset required to construct an action codec."""

    action_dim: int
    norm_stat: dict[str, list[float]]


def resolve_action_codec_name(config: object) -> str:
    """Resolve explicit schemas first, then preserve the legacy mode mapping."""

    explicit = getattr(config, "action_schema", None)
    if explicit:
        return str(explicit)
    delta_mode = str(getattr(config, "action_delta_mode", "")).lower()
    if delta_mode in _PI05_MODES:
        return "ee20_pi05"
    return "ee20_absee"


def _optional_tuple(config: object, field_name: str) -> tuple[float, ...] | None:
    values = getattr(config, field_name, None)
    if values is None:
        return None
    return tuple(float(value) for value in values)


def build_action_codec_from_config(config: ActionConfig) -> ActionCodec:
    """Build the selected codec from recorded normalization and physical bounds."""

    normalizer = config.norm_stat
    try:
        q01 = tuple(float(value) for value in normalizer["q01"])
        q99 = tuple(float(value) for value in normalizer["q99"])
    except (KeyError, TypeError) as exc:
        raise ValueError("norm_stat must contain q01 and q99 sequences") from exc
    codec = build_action_codec(
        resolve_action_codec_name(config),
        q01=q01,
        q99=q99,
        lower_bounds=_optional_tuple(config, "action_lower_bounds"),
        upper_bounds=_optional_tuple(config, "action_upper_bounds"),
    )
    if int(config.action_dim) != codec.spec.dim:
        raise ValueError(
            f"config action_dim={config.action_dim} does not match "
            f"action schema {codec.spec.name!r} dim={codec.spec.dim}"
        )
    return codec
