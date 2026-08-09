# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Lightweight data contracts shared by converters and training."""

from .timeline import (
    TemporalAlignment,
    TimestampAlignment,
    derive_temporal_alignment,
    derive_timestamp_alignment,
)

__all__ = (
    "TemporalAlignment",
    "TimestampAlignment",
    "derive_temporal_alignment",
    "derive_timestamp_alignment",
)
