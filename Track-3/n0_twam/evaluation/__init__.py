# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Internal offline evaluation contracts with a lightweight package root."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tactile_quality import evaluate_tactile_prediction_quality


def __getattr__(name: str) -> object:
    if name == "evaluate_tactile_prediction_quality":
        from .tactile_quality import evaluate_tactile_prediction_quality

        return evaluate_tactile_prediction_quality
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ("evaluate_tactile_prediction_quality",)
