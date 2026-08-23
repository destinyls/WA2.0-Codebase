# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Image validation helpers shared by Franka Policy components."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt


def image(value: object, *, label: str) -> npt.NDArray[np.uint8]:
    """Validate one uint8 HWC RGB image."""

    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{label} must be uint8 HWC RGB")
    return np.ascontiguousarray(array)


def training_aligned_video_history(
    history: tuple[Mapping[str, npt.NDArray[np.uint8]], ...],
    *,
    current_images: Mapping[str, npt.NDArray[np.uint8]],
) -> list[dict[str, npt.NDArray[np.uint8]]]:
    """Validate a complete Wan causal prefix ending at the current observation."""

    if not history or len(history) % 4 != 1:
        raise ValueError("training-aligned video history must contain 4*k+1 frames")
    expected_keys = set(current_images)
    if not expected_keys:
        raise ValueError("training-aligned video history requires camera images")
    current = {
        name: image(value, label=f"current_images.{name}")
        for name, value in current_images.items()
    }
    validated: list[dict[str, npt.NDArray[np.uint8]]] = []
    for index, row in enumerate(history):
        if set(row) != expected_keys:
            raise ValueError(
                "training-aligned video history camera keys changed at "
                f"frame {index}"
            )
        validated.append(
            {
                name: image(value, label=f"video_history[{index}].{name}")
                for name, value in row.items()
            }
        )
    if any(
        not np.array_equal(validated[-1][name], current[name])
        for name in expected_keys
    ):
        raise ValueError(
            "training-aligned video history must end at the current observation"
        )
    return validated


__all__ = ("image", "training_aligned_video_history")
