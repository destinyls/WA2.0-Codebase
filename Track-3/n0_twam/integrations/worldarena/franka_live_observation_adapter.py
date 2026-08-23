# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Adapt canonical WorldArena observations to the strict Franka RGB contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
import numpy.typing as npt

from .franka_live_contract import FrankaLiveContractConfig

_MODEL_CAMERA_KEYS = ("cam_high", "cam_left_wrist")


@dataclass(frozen=True)
class _TimedFrame:
    timestamp_ns: int
    frame: npt.NDArray[np.uint8]


class FrankaLiveObservationAdapter:
    """Build a causal, synchronized 10 Hz RGB prefix from Hub camera history."""

    def __init__(
        self,
        config: FrankaLiveContractConfig,
        *,
        gripper_max_width_m: float = 0.08,
    ) -> None:
        if not np.isfinite(gripper_max_width_m) or gripper_max_width_m <= 0.0:
            raise ValueError("gripper_max_width_m must be positive and finite")
        self._config = config
        self._gripper_max_width_m = float(gripper_max_width_m)
        self._frame_interval_ns = int(round(1_000_000_000 / config.target_fps))
        self._frame_tolerance_ns = int(
            round(config.frame_interval_tolerance_ms * 1_000_000)
        )
        self._buffers: dict[str, list[_TimedFrame]] = {
            key: [] for key in _MODEL_CAMERA_KEYS
        }
        self._sequence_id = 0
        self._last_packet_timestamp_ns: int | None = None
        self._lock = Lock()

    def reset(self) -> None:
        """Clear episode-local frame history and sequencing state."""

        with self._lock:
            for frames in self._buffers.values():
                frames.clear()
            self._sequence_id = 0
            self._last_packet_timestamp_ns = None

    def augment(
        self,
        packet: object,
        new_obs: Mapping[str, object],
        *,
        decode_camera_frames: Callable[[object], Sequence[npt.NDArray[np.uint8]]],
        role_to_model_key: Mapping[str, str],
    ) -> dict[str, object]:
        """Return ``new_obs`` with strict training-aligned live RGB fields."""

        with self._lock:
            packet_timestamp_ns = _positive_timestamp(
                getattr(packet, "observation_timestamp_ns", None),
                label="observation_timestamp_ns",
            )
            if (
                self._last_packet_timestamp_ns is not None
                and packet_timestamp_ns <= self._last_packet_timestamp_ns
            ):
                raise ValueError(
                    "WorldArena observation timestamps must increase monotonically"
                )

            camera_observations = getattr(packet, "camera_observations", None)
            if not isinstance(camera_observations, Sequence):
                raise ValueError("ObservationPacket has no camera observations")
            selected: dict[str, object] = {}
            for camera in camera_observations:
                role = str(getattr(camera, "camera_role", ""))
                model_key = role_to_model_key.get(role)
                if model_key not in _MODEL_CAMERA_KEYS:
                    continue
                if model_key in selected:
                    raise ValueError(
                        f"multiple live cameras map to model key {model_key!r}"
                    )
                selected[model_key] = camera
            missing = tuple(key for key in _MODEL_CAMERA_KEYS if key not in selected)
            if missing:
                raise ValueError(
                    "strict Franka live input is missing camera roles for "
                    + ", ".join(missing)
                )

            for model_key, camera in selected.items():
                frames = tuple(decode_camera_frames(camera))
                timestamps = _camera_timestamps(camera)
                if len(frames) != len(timestamps):
                    raise ValueError(
                        f"{model_key} frame-history timestamps do not match frames"
                    )
                for timestamp_ns, frame in zip(timestamps, frames):
                    if timestamp_ns > packet_timestamp_ns:
                        raise ValueError(
                            f"{model_key} contains an RGB frame from the future"
                        )
                    self._insert_frame(
                        model_key,
                        _TimedFrame(
                            timestamp_ns=timestamp_ns,
                            frame=_rgb_frame(frame, label=model_key),
                        ),
                    )

            history, target_timestamps, source_timestamps = self._aligned_history(
                packet_timestamp_ns
            )
            output = dict(new_obs)
            images_raw = output.get("images")
            if not isinstance(images_raw, Mapping):
                raise ValueError("legacy WorldArena observation has no images mapping")
            images = dict(images_raw)
            current = history[-1]
            images["cam_high"] = current["cam_high"]
            images["cam_left_wrist"] = current["cam_left_wrist"]
            images["cam_wrist"] = current["cam_left_wrist"]
            output["images"] = images
            output["first_frame"] = current["cam_high"]
            output["training_aligned_video_history"] = history
            output["training_aligned_video_timestamps_ns"] = target_timestamps
            output["training_aligned_video_source_timestamps_ns"] = source_timestamps
            output["image_timestamp_ns"] = target_timestamps[-1]
            output["state_timestamp_ns"] = packet_timestamp_ns
            output["observation_sequence_id"] = self._sequence_id
            output["observation_packet_timestamp_ns"] = packet_timestamp_ns
            output["state_timestamp_provenance"] = (
                "observation_packet.observation_timestamp_ns"
            )
            output["camera_latest_timestamps_ns"] = {
                key: values[-1] for key, values in source_timestamps.items()
            }
            output["history_coverage_start_ns"] = target_timestamps[0]
            output["history_coverage_end_ns"] = target_timestamps[-1]
            output["history_coverage_duration_ms"] = (
                target_timestamps[-1] - target_timestamps[0]
            ) / 1_000_000.0
            packet_step_index = _optional_non_negative_int(
                getattr(packet, "step_index", None), label="step_index"
            )
            if packet_step_index is not None:
                output["observation_packet_step_index"] = packet_step_index
            output.update(
                _single_arm_proprioception(
                    packet,
                    gripper_max_width_m=self._gripper_max_width_m,
                )
            )
            self._sequence_id += 1
            self._last_packet_timestamp_ns = packet_timestamp_ns
            return output

    def _insert_frame(self, key: str, value: _TimedFrame) -> None:
        by_timestamp = {
            frame.timestamp_ns: frame for frame in self._buffers[key]
        }
        by_timestamp[value.timestamp_ns] = value
        self._buffers[key] = [
            by_timestamp[timestamp_ns] for timestamp_ns in sorted(by_timestamp)
        ]

    def _aligned_history(
        self, packet_timestamp_ns: int
    ) -> tuple[
        tuple[dict[str, npt.NDArray[np.uint8]], ...],
        tuple[int, ...],
        dict[str, tuple[int, ...]],
    ]:
        high_frames = self._buffers["cam_high"]
        wrist_frames = self._buffers["cam_left_wrist"]
        if not high_frames or not wrist_frames:
            raise ValueError("strict Franka live RGB buffers are incomplete")

        endpoint_ns = high_frames[-1].timestamp_ns
        if endpoint_ns > packet_timestamp_ns:
            raise ValueError("latest RGB timestamp exceeds the robot-state timestamp")
        state_skew_ns = packet_timestamp_ns - endpoint_ns
        max_state_skew_ns = int(
            round(self._config.max_state_image_skew_ms * 1_000_000)
        )
        if state_skew_ns > max_state_skew_ns:
            raise ValueError("live robot state is too far from the latest RGB frame")

        common_start_ns = max(high_frames[0].timestamp_ns, wrist_frames[0].timestamp_ns)
        available_intervals = max(
            0,
            (endpoint_ns - common_start_ns + self._frame_tolerance_ns)
            // self._frame_interval_ns,
        )
        available_count = min(
            self._config.max_history_frames,
            int(available_intervals) + 1,
        )
        history_count = 1 + 4 * ((available_count - 1) // 4)
        targets = tuple(
            endpoint_ns - index * self._frame_interval_ns
            for index in reversed(range(history_count))
        )

        selected_by_key: dict[str, tuple[_TimedFrame, ...]] = {}
        for key, frames in self._buffers.items():
            selected_by_key[key] = tuple(
                self._nearest_frame(frames, target_ns=target, label=key)
                for target in targets
            )
        history = tuple(
            {
                key: np.ascontiguousarray(selected_by_key[key][index].frame)
                for key in _MODEL_CAMERA_KEYS
            }
            for index in range(history_count)
        )
        sources = {
            key: tuple(frame.timestamp_ns for frame in selected_by_key[key])
            for key in _MODEL_CAMERA_KEYS
        }
        self._prune(endpoint_ns)
        return history, targets, sources

    def _nearest_frame(
        self,
        frames: Sequence[_TimedFrame],
        *,
        target_ns: int,
        label: str,
    ) -> _TimedFrame:
        candidates = tuple(
            frame
            for frame in frames
            if abs(frame.timestamp_ns - target_ns) <= self._frame_tolerance_ns
        )
        if not candidates:
            raise ValueError(
                f"{label} has no RGB frame on the required 10 Hz causal grid"
            )
        return min(
            candidates,
            key=lambda frame: (abs(frame.timestamp_ns - target_ns), frame.timestamp_ns),
        )

    def _prune(self, endpoint_ns: int) -> None:
        retention_ns = (
            self._config.max_history_frames + 2
        ) * self._frame_interval_ns
        threshold = endpoint_ns - retention_ns
        for key, frames in self._buffers.items():
            self._buffers[key] = [
                frame for frame in frames if frame.timestamp_ns >= threshold
            ]


class LiveAdaptedPolicy:
    """Reset the live RGB adapter with the wrapped episode Policy."""

    def __init__(self, policy: object, adapter: FrankaLiveObservationAdapter) -> None:
        self._policy = policy
        self._adapter = adapter

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        self._adapter.reset()
        reset = getattr(self._policy, "reset")
        reset(reset_info)

    def infer(self, new_obs: dict[str, Any]) -> dict[str, Any]:
        infer = getattr(self._policy, "infer")
        return infer(new_obs)


def _positive_timestamp(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer timestamp")
    return value


def _optional_non_negative_int(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _camera_timestamps(camera: object) -> tuple[int, ...]:
    history = getattr(camera, "frame_history_timestamps_ns", None)
    history_values = () if history is None else tuple(history)
    timestamps = tuple(
        _positive_timestamp(value, label="camera history timestamp")
        for value in history_values
    )
    current = _positive_timestamp(
        getattr(camera, "timestamp_ns", None), label="camera timestamp"
    )
    combined = (*timestamps, current)
    if any(right <= left for left, right in zip(combined, combined[1:])):
        raise ValueError("camera timestamps must increase strictly")
    return combined


def _rgb_frame(value: object, *, label: str) -> npt.NDArray[np.uint8]:
    frame = np.asarray(value)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{label} must decode to an HWC uint8 RGB frame")
    return np.ascontiguousarray(frame)


def _single_arm_proprioception(
    packet: object,
    *,
    gripper_max_width_m: float,
) -> dict[str, object]:
    """Normalize canonical single-arm state to the Franka training contract."""

    robot_state = getattr(packet, "robot_state", None)
    arms_raw = getattr(robot_state, "arms", None)
    if not isinstance(arms_raw, Sequence) or not arms_raw:
        raise ValueError("strict Franka live input has no canonical arm state")
    arms = tuple(arms_raw)
    if len(arms) == 1:
        arm = arms[0]
    else:
        right = tuple(
            candidate
            for candidate in arms
            if str(getattr(candidate, "arm_id", "")).lower() == "right"
        )
        if len(right) != 1:
            raise ValueError("strict Franka live input cannot resolve one active arm")
        arm = right[0]

    pose = getattr(arm, "ee_pose_base", None)
    if str(getattr(pose, "frame", "")) != "base":
        raise ValueError("strict Franka end-effector pose must use the base frame")
    position = getattr(pose, "position_m", None)
    quaternion = getattr(pose, "orientation_xyzw", None)
    pose7 = np.asarray(
        (
            getattr(position, "x", np.nan),
            getattr(position, "y", np.nan),
            getattr(position, "z", np.nan),
            getattr(quaternion, "x", np.nan),
            getattr(quaternion, "y", np.nan),
            getattr(quaternion, "z", np.nan),
            getattr(quaternion, "w", np.nan),
        ),
        dtype=np.float32,
    )
    if not np.isfinite(pose7).all():
        raise ValueError("strict Franka end-effector pose must be finite")

    joint_state = getattr(arm, "joint_state", None)
    joints = np.asarray(getattr(joint_state, "position_rad", ()), dtype=np.float32)
    if joints.ndim != 1 or joints.size not in (7, 8) or not np.isfinite(joints).all():
        raise ValueError("strict Franka joint state must contain 7 or 8 finite values")
    gripper = getattr(arm, "gripper", None)
    width_m = getattr(gripper, "width_m", None)
    if width_m is None:
        open_ratio = getattr(gripper, "open_ratio", None)
        if open_ratio is None or not np.isfinite(open_ratio):
            raise ValueError("strict Franka live input has no finite gripper state")
        ratio = float(open_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("strict Franka gripper.open_ratio must be within [0, 1]")
        width = ratio * gripper_max_width_m
        gripper_source = "open_ratio_calibrated"
    else:
        if not np.isfinite(width_m):
            raise ValueError("strict Franka gripper.width_m must be finite")
        width = float(width_m)
        gripper_source = "width_m"
    if not 0.0 <= width <= gripper_max_width_m:
        raise ValueError("strict Franka gripper width is outside calibrated limits")
    qpos8 = np.concatenate((joints[:7], np.asarray((width,), dtype=np.float32)))
    return {
        "left_end_pose": pose7,
        "joint_qpos": qpos8,
        "active_arm_id": str(getattr(arm, "arm_id", "")),
        "gripper_width_m": width,
        "gripper_observation_unit": "width_m",
        "gripper_observation_source": gripper_source,
    }


__all__ = (
    "FrankaLiveObservationAdapter",
    "LiveAdaptedPolicy",
)
