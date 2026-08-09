# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Temporal shape helpers shared by training data and strict evaluation."""

from __future__ import annotations


def decoded_evaluation_frame_count(max_latent_frames: int) -> int:
    """Map a positive Wan temporal latent length to decoded frame count."""

    if max_latent_frames <= 0:
        raise ValueError("raw tactile evaluation requires max_latent_frames > 0")
    return (max_latent_frames - 1) * 4 + 1


__all__ = ("decoded_evaluation_frame_count",)
