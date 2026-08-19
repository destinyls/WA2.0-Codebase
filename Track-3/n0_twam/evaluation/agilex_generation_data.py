# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Small tensor adapters for AgileX offline future prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import operator
from typing import cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
ImageArray = npt.NDArray[np.uint8]


def _numpy(value: object) -> npt.NDArray[np.generic]:
    detached = getattr(value, "detach", lambda: value)()
    cpu_value = getattr(detached, "cpu", lambda: detached)()
    return np.asarray(getattr(cpu_value, "numpy", lambda: cpu_value)())


def uint8_camera_tiles(video: ImageArray) -> ImageArray:
    """Split the three-camera tiled decoder output without dropping frame zero."""

    array = np.asarray(video)
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("decoded video must be uint8 [T,H,W,3]")
    if array.shape[2] % 3 != 0:
        raise ValueError("three-camera decoded video width must be divisible by three")
    tile_width = array.shape[2] // 3
    if tile_width < 7 or array.shape[1] < 7:
        raise ValueError("decoded video resolution is too small for SSIM")
    tiles = np.stack(
        tuple(
            array[:, :, index * tile_width : (index + 1) * tile_width]
            for index in range(3)
        ),
        axis=0,
    )
    return cast(ImageArray, np.ascontiguousarray(tiles))


def target_qpos14_rows(
    sample: Mapping[str, object],
    *,
    action_offsets: Sequence[int],
    q01: FloatArray,
    q99: FloatArray,
) -> tuple[FloatArray, BoolArray, tuple[int, ...]]:
    """Return future same-row qpos14 targets, explicitly excluding offset zero."""

    offsets = tuple(action_offsets)
    if (
        len(offsets) < 2
        or offsets[0] != 0
        or any(type(value) is not int for value in offsets)
        or any(right <= left for left, right in zip(offsets, offsets[1:]))
    ):
        raise ValueError("AgileX action offsets must increase from zero")
    actions = _numpy(sample.get("actions"))
    valid = _numpy(sample.get("action_valid_mask"))
    if (
        actions.dtype != np.float32
        or actions.ndim != 4
        or actions.shape[0] != 14
        or actions.shape[1] < 1
        or actions.shape[2] != len(offsets)
        or actions.shape[-1] != 1
        or valid.dtype != np.bool_
        or valid.shape != actions.shape[1:3]
    ):
        raise ValueError("AgileX evaluation action target contract mismatch")
    lower = np.asarray(q01, dtype=np.float32)
    upper = np.asarray(q99, dtype=np.float32)
    if lower.shape != (14,) or upper.shape != (14,) or np.any(upper <= lower):
        raise ValueError("AgileX evaluation q01/q99 are invalid")
    normalized = actions[:, 0, 1:, 0].T
    rows = (normalized + 1.0) * 0.5 * (upper - lower + 1e-6) + lower
    return (
        cast(FloatArray, np.ascontiguousarray(rows, dtype=np.float32)),
        cast(BoolArray, np.ascontiguousarray(valid[0, 1:])),
        offsets[1:],
    )


def current_qpos14(dataset: object, *, episode_id: int, row_id: int) -> FloatArray:
    """Read one converted observation state from its exact LeRobot row."""

    raw_index = dataset._get_global_idx(episode_id, row_id)  # noqa: SLF001
    shape = getattr(raw_index, "shape", None)
    if shape is not None:
        try:
            scalar_shape = tuple(operator.index(size) for size in shape)
        except (TypeError, ValueError) as exc:
            raise ValueError("LeRobot global index must be a scalar integer") from exc
        if scalar_shape:
            raise ValueError("LeRobot global index must be a scalar integer")
        item = getattr(raw_index, "item", None)
        if not callable(item):
            raise ValueError("LeRobot global index must be a scalar integer")
        raw_index = item()
    if isinstance(raw_index, (bool, np.bool_)):
        raise ValueError("LeRobot global index must be a scalar integer")
    try:
        global_index = operator.index(raw_index)
    except TypeError as exc:
        raise ValueError("LeRobot global index must be a scalar integer") from exc
    if global_index < 0:
        raise ValueError("LeRobot global index must be non-negative")
    values = dataset.hf_dataset[global_index]["observation.joint_qpos"]
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (14,) or not np.isfinite(array).all():
        raise ValueError("converted AgileX current qpos14 is invalid")
    return cast(FloatArray, np.ascontiguousarray(array))


def route_wrench(
    sample: Mapping[str, object],
    *,
    keys: tuple[str, ...],
    sensor_id_map: Mapping[str, int],
) -> dict[str, FloatArray] | None:
    """Extract current wrench vectors using the signed sensor-slot mapping."""

    if not keys:
        return None
    values = _numpy(sample.get("wrench"))
    available = _numpy(sample.get("wrench_available_mask"))
    if (
        values.dtype != np.float32
        or values.ndim != 3
        or values.shape[0] < 1
        or values.shape[2] != 6
        or available.dtype != np.bool_
        or available.shape != values.shape[:2]
    ):
        raise ValueError("AgileX evaluation wrench contract mismatch")
    result: dict[str, FloatArray] = {}
    for key in keys:
        slot = sensor_id_map.get(key)
        if type(slot) is not int or slot < 0 or slot >= values.shape[1]:
            raise ValueError("AgileX evaluation wrench sensor map is invalid")
        if not available[0, slot] or not np.isfinite(values[0, slot]).all():
            raise ValueError(f"AgileX evaluation wrench is unavailable: {key}")
        result[key] = cast(FloatArray, np.ascontiguousarray(values[0, slot]))
    return result


__all__ = (
    "current_qpos14",
    "route_wrench",
    "target_qpos14_rows",
    "uint8_camera_tiles",
)
