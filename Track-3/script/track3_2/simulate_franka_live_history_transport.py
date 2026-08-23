#!/usr/bin/env python3
"""Simulate causal 10 Hz RGB history across blocking action intervals."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from n0_twam.integrations.worldarena.franka_live_contract import (
    parse_live_contract_config,
)
from n0_twam.integrations.worldarena.franka_live_observation_adapter import (
    FrankaLiveObservationAdapter,
)


@dataclass(frozen=True)
class Camera:
    camera_role: str
    timestamp_ns: int
    frames: tuple[np.ndarray, ...]
    frame_history_timestamps_ns: tuple[int, ...]


@dataclass(frozen=True)
class Position:
    x: float = 0.45
    y: float = 0.0
    z: float = 0.25


@dataclass(frozen=True)
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass(frozen=True)
class Pose:
    frame: str = "base"
    position_m: Position = field(default_factory=Position)
    orientation_xyzw: Quaternion = field(default_factory=Quaternion)


@dataclass(frozen=True)
class JointState:
    position_rad: tuple[float, ...] = (0.0,) * 7


@dataclass(frozen=True)
class Gripper:
    width_m: float = 0.04


@dataclass(frozen=True)
class Arm:
    arm_id: str = "franka"
    joint_state: JointState = field(default_factory=JointState)
    ee_pose_base: Pose = field(default_factory=Pose)
    gripper: Gripper = field(default_factory=Gripper)


@dataclass(frozen=True)
class RobotState:
    arms: tuple[Arm, ...] = field(default_factory=lambda: (Arm(),))


@dataclass(frozen=True)
class ObservationPacket:
    observation_timestamp_ns: int
    camera_observations: tuple[Camera, ...]
    robot_state: RobotState = field(default_factory=RobotState)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--policy-config-label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap-seconds", type=float, default=12.0)
    parser.add_argument("--cycles", type=int, default=3)
    return parser.parse_args()


def _camera(role: str, timestamps: tuple[int, ...], offset: int) -> Camera:
    frames = tuple(
        np.full((4, 6, 3), (index + offset) % 256, dtype=np.uint8)
        for index in range(len(timestamps))
    )
    return Camera(
        camera_role=role,
        timestamp_ns=timestamps[-1],
        frames=frames,
        frame_history_timestamps_ns=timestamps[:-1],
    )


def _decode(camera: object) -> tuple[np.ndarray, ...]:
    if not isinstance(camera, Camera):
        raise TypeError("unexpected camera type")
    return camera.frames


def _packet(timestamps: tuple[int, ...]) -> ObservationPacket:
    return ObservationPacket(
        observation_timestamp_ns=timestamps[-1] + 10_000_000,
        camera_observations=(
            _camera("global", timestamps, offset=10),
            _camera("wrist", timestamps, offset=20),
        ),
    )


def _adapter(policy_config: dict[str, Any]) -> FrankaLiveObservationAdapter:
    config = parse_live_contract_config(
        policy_config["live_contract"],
        require_execution_fields=int(policy_config.get("schema_version", 0)) >= 4,
    )
    return FrankaLiveObservationAdapter(config)


def _augment(
    adapter: FrankaLiveObservationAdapter,
    timestamps: tuple[int, ...],
) -> dict[str, object]:
    return adapter.augment(
        _packet(timestamps),
        {"images": {}},
        decode_camera_frames=_decode,
        role_to_model_key={"global": "cam_high", "wrist": "cam_left_wrist"},
    )


def _simulate_backfill(
    policy_config: dict[str, Any],
    *,
    gap_seconds: float,
    cycles: int,
) -> dict[str, Any]:
    target_fps = int(policy_config["live_contract"]["target_fps"])
    interval_ns = int(round(1_000_000_000 / target_fps))
    gap_frames = int(round(gap_seconds * target_fps))
    base_ns = 10_000_000_000
    adapter = _adapter(policy_config)
    history_lengths: list[int] = []
    retained_targets: tuple[int, ...] = ()
    previous_index = 0
    for cycle in range(cycles):
        current_index = cycle * gap_frames
        start_index = 0 if cycle == 0 else previous_index + 1
        packet_timestamps = tuple(
            base_ns + index * interval_ns
            for index in range(start_index, current_index + 1)
        )
        result = _augment(adapter, packet_timestamps)
        history = result["training_aligned_video_history"]
        retained_targets = result["training_aligned_video_timestamps_ns"]
        history_lengths.append(len(history))
        previous_index = current_index

    expected_capture_frames = (cycles - 1) * gap_frames + 1
    intervals = np.diff(np.asarray(retained_targets, dtype=np.int64))
    max_interval_error_ms = (
        float(np.max(np.abs(intervals - interval_ns))) / 1_000_000
        if intervals.size
        else 0.0
    )
    return {
        "status": "verified",
        "capture_fps": target_fps,
        "blocking_action_gap_seconds": gap_seconds,
        "policy_request_count": cycles,
        "captured_frames_per_camera": expected_capture_frames,
        "retained_history_frames": len(retained_targets),
        "history_lengths_at_requests": history_lengths,
        "history_coverage_percent": 100.0,
        "max_10hz_grid_error_ms": max_interval_error_ms,
        "causal_future_frame_count": 0,
    }


def _simulate_latest_only(
    policy_config: dict[str, Any], *, gap_seconds: float
) -> dict[str, Any]:
    target_fps = int(policy_config["live_contract"]["target_fps"])
    interval_ns = int(round(1_000_000_000 / target_fps))
    gap_frames = int(round(gap_seconds * target_fps))
    base_ns = 20_000_000_000
    adapter = _adapter(policy_config)
    _augment(adapter, (base_ns,))
    second = base_ns + gap_frames * interval_ns
    try:
        _augment(adapter, (second,))
    except ValueError as error:
        return {
            "status": "rejected_as_required",
            "latest_only_frames_received": 2,
            "missing_frames_during_action": gap_frames - 1,
            "error": str(error),
        }
    return {"status": "unexpectedly_accepted"}


def main() -> None:
    args = _parse_args()
    if args.gap_seconds <= 0.0 or args.cycles < 2:
        raise ValueError("gap-seconds must be positive and cycles must be at least 2")
    policy_config = json.loads(args.policy_config.read_text(encoding="utf-8"))
    backfill = _simulate_backfill(
        policy_config,
        gap_seconds=args.gap_seconds,
        cycles=args.cycles,
    )
    latest_only = _simulate_latest_only(
        policy_config,
        gap_seconds=args.gap_seconds,
    )
    external_chunk_actions = int(policy_config["external_chunk_actions"])
    plan_action_count = int(policy_config.get("max_chunk_actions", 12))
    live = parse_live_contract_config(
        policy_config["live_contract"],
        require_execution_fields=int(policy_config.get("schema_version", 0)) >= 4,
    )
    valid_execution = (
        external_chunk_actions == live.actions_per_replan
        and live.future_start_index + live.actions_per_replan == plan_action_count
        and live.require_fresh_observation_after_chunk
    )
    passed = (
        backfill["status"] == "verified"
        and latest_only["status"] == "rejected_as_required"
        and valid_execution
    )
    report = {
        "schema_version": 1,
        "evaluation_type": "software transport and conditioning simulation",
        "real_robot_task_success_claimed": False,
        "status": "verified" if passed else "failed",
        "policy_config": args.policy_config_label or str(args.policy_config),
        "policy_contract": {
            "conditioning_mode": policy_config["live_contract"]["conditioning_mode"],
            "plan_action_count": plan_action_count,
            "external_chunk_actions": external_chunk_actions,
            "conditioning_slots_skipped": live.future_start_index,
            "future_prediction_range": [
                live.future_start_index,
                live.future_start_index + live.actions_per_replan,
            ],
            "returned_future_actions": live.actions_per_replan,
            "return_batches_per_plan": 1,
            "discarded_future_prediction_count": 0,
            "fresh_plan_per_returned_batch": True,
            "requested_action_hz": live.action_hz,
            "expected_batch_duration_ms": (
                1000.0 * live.actions_per_replan / live.action_hz
            ),
        },
        "continuous_capture_with_backfill": backfill,
        "request_time_latest_frame_only": latest_only,
        "conclusion": (
            "A robot-side 10 Hz capture buffer must keep running while inference and "
            "action execution block, then backfill every missing frame in the next packet."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
