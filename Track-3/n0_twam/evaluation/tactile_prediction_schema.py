# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Typed schema and semantic validation for Stage-A tactile predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import numpy.typing as npt

from n0_twam.evaluation.sealed_artifact_io import nonnegative_integer

MODEL_TACTILE_SENSOR_CAPACITY = 4
ACTIVE_TACTILE_SENSOR_IDS = (0, 1)
TACTILE_FRAME_COUNT = 17
TACTILE_HEIGHT = 128
TACTILE_WIDTH = 128
TACTILE_CHANNELS = 3
PER_SAMPLE_RESIDUAL_SHAPE = (
    len(ACTIVE_TACTILE_SENSOR_IDS),
    TACTILE_FRAME_COUNT,
    TACTILE_HEIGHT,
    TACTILE_WIDTH,
    TACTILE_CHANNELS,
)

Int64Array = npt.NDArray[np.int64]
Float32Array = npt.NDArray[np.float32]


def strict_int64_vector(
    value: npt.ArrayLike,
    *,
    label: str,
    expected_length: int,
) -> Int64Array:
    """Validate integer provenance before converting it to canonical int64."""

    array = np.asarray(value)
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label} must have an integer dtype")
    if array.shape != (expected_length,):
        raise ValueError(f"{label} must contain {expected_length} values")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{label} exceeds the signed int64 range")
    converted = array.astype(np.int64, copy=False)
    if np.any(converted < 0) or (
        converted.size > 1 and np.any(np.diff(converted) <= 0)
    ):
        raise ValueError(f"{label} must be non-negative and strictly increasing")
    return np.ascontiguousarray(converted)


@dataclass(frozen=True)
class TactilePredictionSample:
    """One prediction with source coordinates but no embedded ground truth."""

    sample_id: str
    dataset_index: int
    lerobot_episode_index: int
    task: str
    source_relative_path: str
    source_row_ids: Int64Array
    source_step_ids: Int64Array
    active_sensor_ids: Int64Array
    signed_residual: Float32Array


def json_ready(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("generation provenance contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    raise TypeError(f"generation provenance is not JSON serializable: {type(value)!r}")


def _safe_component(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise ValueError(f"{label} must be one safe path component")
    return value


def _source_path(value: object, *, task: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("source_relative_path must be non-empty")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.suffix not in {".h5", ".hdf5"}:
        raise ValueError("source_relative_path must be a safe relative HDF5 path")
    if not path.parts or path.parts[0] != task:
        raise ValueError("source_relative_path task does not match sample task")
    return path.as_posix()


def _validate_int_vector(
    value: npt.NDArray[np.generic],
    *,
    label: str,
    expected_length: int,
    require_zero_start: bool,
) -> Int64Array:
    array = np.asarray(value)
    if array.dtype != np.int64 or array.shape != (expected_length,):
        raise ValueError(f"{label} must be int64 [{expected_length}]")
    if np.any(array < 0) or (array.size > 1 and np.any(np.diff(array) <= 0)):
        raise ValueError(f"{label} must be non-negative and strictly increasing")
    if require_zero_start and int(array[0]) != 0:
        raise ValueError(f"{label} must start at row zero for deterministic crop0")
    return np.ascontiguousarray(array)


def validate_sample(sample: TactilePredictionSample) -> TactilePredictionSample:
    """Return a contiguous, normalized sample after all semantic checks."""

    sample_id = _safe_component(sample.sample_id, label="sample_id")
    task = _safe_component(sample.task, label="task")
    source_relative_path = _source_path(sample.source_relative_path, task=task)
    dataset_index = nonnegative_integer(sample.dataset_index, label="dataset_index")
    episode_index = nonnegative_integer(
        sample.lerobot_episode_index, label="lerobot_episode_index"
    )
    source_row_ids = _validate_int_vector(
        sample.source_row_ids,
        label="source_row_ids",
        expected_length=TACTILE_FRAME_COUNT,
        require_zero_start=True,
    )
    source_step_ids = _validate_int_vector(
        sample.source_step_ids,
        label="source_step_ids",
        expected_length=TACTILE_FRAME_COUNT,
        require_zero_start=False,
    )
    active_sensor_ids = np.asarray(sample.active_sensor_ids)
    if active_sensor_ids.dtype != np.int64 or active_sensor_ids.tolist() != [0, 1]:
        raise ValueError("active tactile sensor IDs must be int64 [0, 1]")
    residual = np.asarray(sample.signed_residual)
    if residual.dtype != np.float32:
        raise ValueError("signed residual must use float32")
    if residual.shape != PER_SAMPLE_RESIDUAL_SHAPE:
        raise ValueError(
            "signed residual must have shape " + str(PER_SAMPLE_RESIDUAL_SHAPE)
        )
    if not np.isfinite(residual).all() or np.any(np.abs(residual) > 1.0):
        raise ValueError("signed residual must be finite and within [-1, 1]")
    return TactilePredictionSample(
        sample_id=sample_id,
        dataset_index=dataset_index,
        lerobot_episode_index=episode_index,
        task=task,
        source_relative_path=source_relative_path,
        source_row_ids=source_row_ids,
        source_step_ids=source_step_ids,
        active_sensor_ids=np.ascontiguousarray(active_sensor_ids),
        signed_residual=np.ascontiguousarray(residual),
    )
