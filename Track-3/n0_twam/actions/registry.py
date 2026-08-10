# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Explicit action codec registry without auto-import side effects."""

from collections.abc import Callable
from typing import Any

from .base import ActionCodec

ActionCodecBuilder = Callable[..., ActionCodec]
_ACTION_CODEC_BUILDERS: dict[str, ActionCodecBuilder] = {}


def register_action_codec(
    name: str,
) -> Callable[[ActionCodecBuilder], ActionCodecBuilder]:
    """Register one codec builder under a stable schema name."""

    if not name:
        raise ValueError("action codec name must be non-empty")

    def decorator(builder: ActionCodecBuilder) -> ActionCodecBuilder:
        if name in _ACTION_CODEC_BUILDERS:
            raise ValueError(f"Action codec {name!r} is already registered")
        _ACTION_CODEC_BUILDERS[name] = builder
        return builder

    return decorator


def build_action_codec(name: str, **kwargs: Any) -> ActionCodec:
    """Build a registered action codec or fail with the available names."""

    builder = _ACTION_CODEC_BUILDERS.get(name)
    if builder is None:
        available = ", ".join(sorted(_ACTION_CODEC_BUILDERS))
        raise KeyError(f"Unknown action codec {name!r}; available: {available}")
    return builder(**kwargs)


def list_action_codecs() -> tuple[str, ...]:
    """Return registered codec names in deterministic order."""

    return tuple(sorted(_ACTION_CODEC_BUILDERS))
