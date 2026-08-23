# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed observation contract for training-aligned Franka deployment."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

LIVE_CONDITIONING_MODE = "training_aligned_rgb_replan_v1"
TRAINING_ACTION_HZ = 15
TRAINING_ACTIONS_PER_REPLAN = 6
TRAINING_FUTURE_START_INDEX = 6

_OBSERVATION_FIELDS = {
    "conditioning_mode",
    "target_fps",
    "frame_interval_tolerance_ms",
    "max_state_image_skew_ms",
    "max_history_frames",
}
_EXECUTION_FIELDS = {
    "action_hz",
    "actions_per_replan",
    "future_start_index",
    "require_dual_camera_history",
    "require_fresh_observation_after_chunk",
}


@dataclass(frozen=True)
class FrankaLiveContractConfig:
    """Signed timing limits for one training-aligned live observation."""

    conditioning_mode: str
    target_fps: int
    frame_interval_tolerance_ms: float
    max_state_image_skew_ms: float
    max_history_frames: int
    action_hz: int = TRAINING_ACTION_HZ
    actions_per_replan: int = TRAINING_ACTIONS_PER_REPLAN
    future_start_index: int = TRAINING_FUTURE_START_INDEX
    require_dual_camera_history: bool = True
    require_fresh_observation_after_chunk: bool = True
    execution_contract_explicit: bool = False


@dataclass(frozen=True)
class ValidatedLiveObservation:
    """Validated monotonic live-observation identity."""

    observation_sequence_id: int
    image_timestamp_ns: int
    state_timestamp_ns: int
    report: Mapping[str, object]


def parse_live_contract_config(
    value: object,
    *,
    require_execution_fields: bool = False,
) -> FrankaLiveContractConfig:
    """Parse a training-aligned observation and execution contract."""

    if not isinstance(value, Mapping):
        raise ValueError("live_contract must be a JSON object")
    fields = set(value)
    legacy = fields == _OBSERVATION_FIELDS
    expected = _OBSERVATION_FIELDS | _EXECUTION_FIELDS
    if fields != expected and not legacy:
        raise ValueError(
            "invalid live_contract: "
            f"missing={sorted(expected - fields)}, "
            f"unexpected={sorted(fields - expected)}"
        )
    if require_execution_fields and legacy:
        raise ValueError(
            "live_contract must explicitly bind action cadence and future slots"
        )
    mode = value.get("conditioning_mode")
    if mode != LIVE_CONDITIONING_MODE:
        raise ValueError(f"conditioning_mode must be {LIVE_CONDITIONING_MODE}")
    target_fps = value.get("target_fps")
    history_frames = value.get("max_history_frames")
    if type(target_fps) is not int or target_fps != 10:
        raise ValueError("live target_fps must be exactly 10")
    if (
        type(history_frames) is not int
        or history_frames <= 0
        or history_frames % 4 != 1
    ):
        raise ValueError("max_history_frames must be a positive 4*k+1 integer")
    interval_tolerance = _positive_finite(
        value.get("frame_interval_tolerance_ms"),
        label="frame_interval_tolerance_ms",
    )
    state_skew = _positive_finite(
        value.get("max_state_image_skew_ms"),
        label="max_state_image_skew_ms",
    )
    if interval_tolerance >= 1000.0 / target_fps:
        raise ValueError("frame interval tolerance must be below one frame interval")
    action_hz = TRAINING_ACTION_HZ if legacy else value.get("action_hz")
    actions_per_replan = (
        TRAINING_ACTIONS_PER_REPLAN if legacy else value.get("actions_per_replan")
    )
    future_start_index = (
        TRAINING_FUTURE_START_INDEX if legacy else value.get("future_start_index")
    )
    for field_value, label, expected_value in (
        (action_hz, "action_hz", TRAINING_ACTION_HZ),
        (
            actions_per_replan,
            "actions_per_replan",
            TRAINING_ACTIONS_PER_REPLAN,
        ),
        (
            future_start_index,
            "future_start_index",
            TRAINING_FUTURE_START_INDEX,
        ),
    ):
        if type(field_value) is not int or field_value != expected_value:
            raise ValueError(f"{label} must be exactly {expected_value}")
    require_dual_camera_history = (
        True if legacy else value.get("require_dual_camera_history")
    )
    require_fresh_observation = (
        True
        if legacy
        else value.get("require_fresh_observation_after_chunk")
    )
    if require_dual_camera_history is not True:
        raise ValueError("require_dual_camera_history must be true")
    if require_fresh_observation is not True:
        raise ValueError("require_fresh_observation_after_chunk must be true")
    return FrankaLiveContractConfig(
        conditioning_mode=mode,
        target_fps=target_fps,
        frame_interval_tolerance_ms=interval_tolerance,
        max_state_image_skew_ms=state_skew,
        max_history_frames=history_frames,
        action_hz=int(action_hz),
        actions_per_replan=int(actions_per_replan),
        future_start_index=int(future_start_index),
        require_dual_camera_history=True,
        require_fresh_observation_after_chunk=True,
        execution_contract_explicit=not legacy,
    )


def validate_live_observation(
    observation: Mapping[str, object],
    *,
    config: FrankaLiveContractConfig,
    history: Sequence[Mapping[str, npt.NDArray[np.uint8]]],
    current_state: npt.NDArray[np.float32],
    previous_sequence_id: int | None,
    previous_image_timestamp_ns: int | None,
) -> ValidatedLiveObservation:
    """Validate causality, cadence, state binding, and monotonic delivery."""

    if not history or len(history) > config.max_history_frames:
        raise ValueError(
            "strict live RGB history must contain between 1 and "
            f"{config.max_history_frames} frames"
        )
    if len(history) % 4 != 1:
        raise ValueError("strict live RGB history must contain 4*k+1 frames")
    timestamps = _timestamps(
        observation.get("training_aligned_video_timestamps_ns"),
        expected_length=len(history),
    )
    image_timestamp_ns = _non_negative_int(
        observation.get("image_timestamp_ns"), label="image_timestamp_ns"
    )
    state_timestamp_ns = _non_negative_int(
        observation.get("state_timestamp_ns"), label="state_timestamp_ns"
    )
    sequence_id = _non_negative_int(
        observation.get("observation_sequence_id"),
        label="observation_sequence_id",
    )
    if timestamps[-1] != image_timestamp_ns:
        raise ValueError("image_timestamp_ns must equal the final RGB timestamp")
    expected_interval_ns = int(round(1_000_000_000 / config.target_fps))
    tolerance_ns = int(round(config.frame_interval_tolerance_ms * 1_000_000))
    intervals = np.diff(np.asarray(timestamps, dtype=np.int64))
    if len(intervals) and np.any(
        np.abs(intervals - expected_interval_ns) > tolerance_ns
    ):
        raise ValueError("RGB history is not on the signed 10 Hz cadence")
    if state_timestamp_ns < image_timestamp_ns:
        raise ValueError("state timestamp precedes the causal RGB observation")
    state_skew_ns = state_timestamp_ns - image_timestamp_ns
    if state_skew_ns > int(round(config.max_state_image_skew_ms * 1_000_000)):
        raise ValueError("state and RGB timestamps exceed the signed skew limit")
    if previous_sequence_id is not None and sequence_id <= previous_sequence_id:
        raise ValueError("observation_sequence_id must increase monotonically")
    if (
        previous_image_timestamp_ns is not None
        and image_timestamp_ns <= previous_image_timestamp_ns
    ):
        raise ValueError("live image timestamp must increase monotonically")
    state = np.asarray(current_state, dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("live current state must be a finite pose8 vector")
    report: dict[str, object] = {
        "status": "verified",
        "conditioning_mode": config.conditioning_mode,
        "observation_sequence_id": sequence_id,
        "history_frame_count": len(history),
        "history_start_timestamp_ns": timestamps[0],
        "image_timestamp_ns": image_timestamp_ns,
        "state_timestamp_ns": state_timestamp_ns,
        "state_image_skew_ms": state_skew_ns / 1_000_000.0,
        "target_fps": config.target_fps,
        "action_hz": config.action_hz,
        "actions_per_replan": config.actions_per_replan,
        "future_start_index": config.future_start_index,
        "require_dual_camera_history": config.require_dual_camera_history,
        "require_fresh_observation_after_chunk": (
            config.require_fresh_observation_after_chunk
        ),
        "execution_contract_explicit": config.execution_contract_explicit,
        "rgb_history_sha256": _history_sha256(history, timestamps),
        "current_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
    }
    return ValidatedLiveObservation(
        observation_sequence_id=sequence_id,
        image_timestamp_ns=image_timestamp_ns,
        state_timestamp_ns=state_timestamp_ns,
        report=report,
    )


def _timestamps(value: object, *, expected_length: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise ValueError("training_aligned_video_timestamps_ns length mismatch")
    timestamps = tuple(
        _non_negative_int(item, label="training_aligned_video_timestamps_ns")
        for item in value
    )
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("RGB timestamps must increase strictly")
    return timestamps


def _history_sha256(
    history: Sequence[Mapping[str, npt.NDArray[np.uint8]]],
    timestamps: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    for timestamp, row in zip(timestamps, history, strict=True):
        digest.update(int(timestamp).to_bytes(8, "big", signed=False))
        for name in sorted(row):
            image = np.asarray(row[name])
            digest.update(name.encode("utf-8"))
            digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
            digest.update(image.tobytes())
    return digest.hexdigest()


def _positive_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _non_negative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


__all__ = (
    "FrankaLiveContractConfig",
    "LIVE_CONDITIONING_MODE",
    "TRAINING_ACTION_HZ",
    "TRAINING_ACTIONS_PER_REPLAN",
    "TRAINING_FUTURE_START_INDEX",
    "ValidatedLiveObservation",
    "parse_live_contract_config",
    "validate_live_observation",
)
