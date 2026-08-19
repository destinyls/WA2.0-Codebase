# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Action schemas and codecs."""

from .base import ActionCodec, QuantileActionCodec
from .config import build_action_codec_from_config, resolve_action_codec_name
from .registry import build_action_codec, list_action_codecs, register_action_codec
from .spec import ActionSpec

# Explicit imports register built-ins only after the registry API is initialized.
from . import ee20 as _ee20  # noqa: F401,E402  # isort: skip
from . import qpos14 as _qpos14  # noqa: F401,E402  # isort: skip
from . import qpos8 as _qpos8  # noqa: F401,E402  # isort: skip

__all__ = (
    "ActionCodec",
    "ActionSpec",
    "QuantileActionCodec",
    "build_action_codec",
    "build_action_codec_from_config",
    "list_action_codecs",
    "register_action_codec",
    "resolve_action_codec_name",
)
