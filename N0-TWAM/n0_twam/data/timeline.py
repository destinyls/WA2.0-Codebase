# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed source-frame to N0 latent/action timeline contracts."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class TemporalAlignment:
    """Explicit mapping from encoded observations to future action targets."""

    source_frame_count: int
    temporal_compression: int
    encoded_frame_stride: int
    action_offset: int
    action_per_latent_frame: int
    encoded_frame_ids: npt.NDArray[np.int64]
    latent_anchor_ids: npt.NDArray[np.int64]
    target_frame_ids: npt.NDArray[np.int64]
    is_pad: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class TimestampAlignment:
    """Timestamp-backed mapping from latent anchors to action samples."""

    temporal_compression: int
    action_per_latent_frame: int
    tolerance_seconds: float
    encoded_observation_timestamps: npt.NDArray[np.float64]
    latent_anchor_timestamps: npt.NDArray[np.float64]
    desired_action_timestamps: npt.NDArray[np.float64]
    target_action_indices: npt.NDArray[np.int64]
    target_action_timestamps: npt.NDArray[np.float64]
    is_pad: npt.NDArray[np.bool_]


def _immutable_copy(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    output = array.copy()
    output.setflags(write=False)
    return output


def _validate_timestamps(
    values: npt.ArrayLike,
    *,
    name: str,
    minimum_count: int,
) -> npt.NDArray[np.float64]:
    timestamps = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if timestamps.shape[0] < minimum_count:
        raise ValueError(f"{name} must contain at least {minimum_count} values")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return timestamps


def _nearest_timestamp_indices(
    action_timestamps: npt.NDArray[np.float64],
    desired_timestamps: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    right = np.searchsorted(action_timestamps, desired_timestamps, side="left")
    right = np.clip(right, 0, action_timestamps.shape[0] - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(action_timestamps[left] - desired_timestamps) <= np.abs(
        action_timestamps[right] - desired_timestamps
    )
    return np.where(choose_left, left, right).astype(np.int64)


def derive_timestamp_alignment(
    *,
    encoded_observation_timestamps: npt.ArrayLike,
    action_timestamps: npt.ArrayLike,
    latent_frame_count: int,
    temporal_compression: int = 4,
    action_per_latent_frame: int | None = None,
    tolerance_seconds: float | None = None,
) -> TimestampAlignment:
    """Align jittered observation/action streams on physical timestamps.

    The desired targets uniformly divide each latent-anchor interval and exclude
    the anchor itself, so the first target always represents the next action.
    Missing interior samples fail closed. Only targets beyond the recorded action
    stream may use repeat-last padding.
    """

    if latent_frame_count <= 1:
        raise ValueError("latent_frame_count must be greater than one")
    if temporal_compression <= 0:
        raise ValueError("temporal_compression must be positive")

    expected_count = (latent_frame_count - 1) * temporal_compression + 1
    observation_times = _validate_timestamps(
        encoded_observation_timestamps,
        name="encoded_observation_timestamps",
        minimum_count=expected_count,
    )
    if observation_times.shape[0] != expected_count:
        raise ValueError(
            f"expected {expected_count} encoded observation timestamps for "
            f"{latent_frame_count} latent frames, got {observation_times.shape[0]}"
        )
    action_times = _validate_timestamps(
        action_timestamps,
        name="action_timestamps",
        minimum_count=2,
    )

    anchor_times = observation_times[::temporal_compression]
    anchor_intervals = np.diff(anchor_times)
    action_interval = float(np.median(np.diff(action_times)))
    median_anchor_interval = float(np.median(anchor_intervals))
    if action_per_latent_frame is None:
        action_per_latent_frame = int(round(median_anchor_interval / action_interval))
    if action_per_latent_frame <= 0:
        raise ValueError("action_per_latent_frame must be positive")

    if tolerance_seconds is None:
        tolerance_seconds = max(1e-6, action_interval * 0.25)
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0.0:
        raise ValueError("tolerance_seconds must be finite and non-negative")

    interval_per_anchor = np.concatenate((anchor_intervals, (median_anchor_interval,)))
    fractions = (
        np.arange(1, action_per_latent_frame + 1, dtype=np.float64)
        / action_per_latent_frame
    )
    desired_times = anchor_times[:, None] + interval_per_anchor[:, None] * fractions

    terminal_pad = desired_times > action_times[-1] + tolerance_seconds
    nearest_indices = _nearest_timestamp_indices(action_times, desired_times)
    nearest_indices[terminal_pad] = action_times.shape[0] - 1
    selected_times = action_times[nearest_indices]

    valid = ~terminal_pad
    timing_error = np.abs(selected_times - desired_times)
    if np.any(valid & (timing_error > tolerance_seconds)):
        row, column = np.argwhere(valid & (timing_error > tolerance_seconds))[0]
        raise ValueError(
            "no action sample within tolerance for target "
            f"[{int(row)}, {int(column)}]"
        )
    if np.any(valid & (selected_times <= anchor_times[:, None])):
        raise ValueError("timestamp alignment selected a non-future action")
    if np.any(np.all(terminal_pad, axis=1)):
        raise ValueError("a latent anchor has no future action target")
    for row_indices, row_valid in zip(nearest_indices, valid, strict=True):
        valid_indices = row_indices[row_valid]
        if valid_indices.shape[0] > 1 and np.any(np.diff(valid_indices) <= 0):
            raise ValueError("multiple targets map to the same action sample")

    return TimestampAlignment(
        temporal_compression=temporal_compression,
        action_per_latent_frame=action_per_latent_frame,
        tolerance_seconds=float(tolerance_seconds),
        encoded_observation_timestamps=_immutable_copy(observation_times),
        latent_anchor_timestamps=_immutable_copy(anchor_times),
        desired_action_timestamps=_immutable_copy(desired_times),
        target_action_indices=_immutable_copy(nearest_indices),
        target_action_timestamps=_immutable_copy(selected_times),
        is_pad=_immutable_copy(terminal_pad),
    )


def derive_temporal_alignment(
    *,
    encoded_frame_ids: npt.ArrayLike,
    latent_frame_count: int,
    source_frame_count: int,
    temporal_compression: int = 4,
    action_offset: int = 1,
) -> TemporalAlignment:
    """Derive future target indices without assuming a 12- or 50-step horizon.

    ``encoded_frame_ids`` contains raw source indices retained by the RGB/tactile
    encoder. N0's temporal VAE maps every ``temporal_compression`` encoded frames
    to the next latent anchor. The resulting raw-frame distance is the number of
    action targets associated with each latent frame.
    """

    if latent_frame_count <= 0:
        raise ValueError("latent_frame_count must be positive")
    if source_frame_count <= 1:
        raise ValueError("source_frame_count must be greater than one")
    if temporal_compression <= 0:
        raise ValueError("temporal_compression must be positive")
    if action_offset <= 0:
        raise ValueError("action_offset must be positive")

    frame_ids = np.asarray(encoded_frame_ids, dtype=np.int64)
    if frame_ids.ndim != 1:
        raise ValueError("encoded_frame_ids must be one-dimensional")
    expected_count = (latent_frame_count - 1) * temporal_compression + 1
    if frame_ids.shape[0] != expected_count:
        raise ValueError(
            f"expected {expected_count} encoded frame IDs for "
            f"{latent_frame_count} latent frames, got {frame_ids.shape[0]}"
        )
    if np.any(frame_ids < 0) or np.any(frame_ids >= source_frame_count):
        raise ValueError("encoded_frame_ids must be within the source episode")

    if frame_ids.shape[0] == 1:
        raise ValueError("at least two encoded frames are required to derive stride")
    differences = np.diff(frame_ids)
    if np.any(differences <= 0) or not np.all(differences == differences[0]):
        raise ValueError("encoded_frame_ids must be strictly and uniformly spaced")

    encoded_stride = int(differences[0])
    action_per_latent = encoded_stride * temporal_compression
    anchor_ids = frame_ids[::temporal_compression]
    if anchor_ids.shape[0] != latent_frame_count:
        raise RuntimeError("internal latent anchor derivation mismatch")

    offsets = action_offset + np.arange(action_per_latent, dtype=np.int64)
    target_ids = anchor_ids[:, None] + offsets[None, :]
    is_pad = target_ids >= source_frame_count
    if np.any(np.all(is_pad, axis=1)):
        raise ValueError("a latent anchor has no future action target")
    target_ids = np.minimum(target_ids, source_frame_count - 1)

    return TemporalAlignment(
        source_frame_count=source_frame_count,
        temporal_compression=temporal_compression,
        encoded_frame_stride=encoded_stride,
        action_offset=action_offset,
        action_per_latent_frame=action_per_latent,
        encoded_frame_ids=_immutable_copy(frame_ids),
        latent_anchor_ids=_immutable_copy(anchor_ids),
        target_frame_ids=_immutable_copy(target_ids),
        is_pad=_immutable_copy(is_pad),
    )
