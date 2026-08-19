# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Live sensor snapshots map to existing AgileX and Franka Policy wires."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from n0_twam.integrations.worldarena.realtime_sensors import (
    AgileXObservationSpec,
    BoundedLatestRing,
    ClockStamp,
    CoherentSnapshot,
    FrankaObservationSpec,
    LatestCoherentSynchronizer,
    SensorPacket,
    SynchronizerConfig,
    build_agilex_live_observation,
    build_franka_live_observation,
)


def _packet(
    stream_id: str,
    modality: str,
    payload: NDArray[Any],
    *,
    capture_ns: int = 1_000_000_000,
) -> SensorPacket:
    return SensorPacket(
        stream_id=stream_id,
        modality=modality,
        payload=payload,
        sequence_id=7,
        generation=3,
        clock=ClockStamp(
            device_ns=None,
            arrival_mono_ns=capture_ns + 1_000,
            capture_mono_ns=capture_ns,
            capture_wall_ns=1_800_000_000_000_000_000 + capture_ns,
            clock_domain_id="rig-clock",
            quality="host_correlated",
            uncertainty_ns=1_000,
            correlation_generation=3,
        ),
    )


def _snapshot(packets: tuple[SensorPacket, ...]) -> CoherentSnapshot:
    buffers = {packet.stream_id: BoundedLatestRing(capacity=4) for packet in packets}
    for packet in packets:
        buffers[packet.stream_id].push(packet)
    synchronizer = LatestCoherentSynchronizer(
        buffers=buffers,
        config=SynchronizerConfig(
            required_streams=tuple(buffers),
            max_skew_ns=1_000_000,
            max_age_ns=10_000_000,
        ),
    )
    result = synchronizer.build_latest(now_mono_ns=1_000_000_000)
    assert result is not None
    return result


def _rgb(value: int) -> NDArray[np.uint8]:
    return cast(NDArray[np.uint8], np.full((4, 5, 3), value, dtype=np.uint8))


def _agilex_packets(*, contact: bool = True) -> tuple[SensorPacket, ...]:
    packets = [
        _packet("rgb.top", "rgb", _rgb(1)),
        _packet("rgb.wrist_l", "rgb", _rgb(2)),
        _packet("rgb.wrist_r", "rgb", _rgb(3)),
        _packet("qpos14", "qpos", np.arange(14, dtype=np.float32)),
    ]
    if contact:
        packets.extend(
            (
                _packet("tactile.left", "tactile", _rgb(4)),
                _packet("tactile.right", "tactile", _rgb(5)),
                _packet("wrench.left", "wrench", np.arange(6, dtype=np.float32)),
                _packet(
                    "wrench.right",
                    "wrench",
                    np.arange(6, dtype=np.float32) + 10,
                ),
            )
        )
    return tuple(packets)


def test_agilex_bridge_uses_capture_time_and_exact_contact_wire() -> None:
    observation = build_agilex_live_observation(
        _snapshot(_agilex_packets()),
        spec=AgileXObservationSpec(),
        task_id="insert",
        prompt="insert the object",
        tactile_profile="vision_tactile",
        execution_dt_s=0.1,
        tactile_required=True,
        wrench_required=True,
    )

    images = observation["images"]
    assert isinstance(images, Mapping)
    assert tuple(images) == (
        "cam_high",
        "cam_wrist_left",
        "cam_wrist_right",
    )
    qpos = np.asarray(observation["joint_qpos"])
    np.testing.assert_array_equal(observation["state"], qpos)
    np.testing.assert_array_equal(observation["left_arm_joint_state"], qpos[:7])
    np.testing.assert_array_equal(observation["right_arm_joint_state"], qpos[7:])
    assert observation["timestamp"] == pytest.approx(1_800_000_001.0)
    assert observation["execution_dt_s"] == pytest.approx(0.1)
    contact = observation["tactile"]
    assert isinstance(contact, Mapping)
    assert tuple(contact) == (
        "left_gripper",
        "right_gripper",
        "left_wrist_force",
        "right_wrist_force",
    )


def test_agilex_bridge_drops_contact_for_vision_only_route() -> None:
    observation = build_agilex_live_observation(
        _snapshot(_agilex_packets()),
        spec=AgileXObservationSpec(),
        task_id="wipe",
        prompt="wipe the table",
        tactile_profile="vision_only",
        execution_dt_s=0.1,
        tactile_required=False,
        wrench_required=False,
    )

    assert "tactile" not in observation


def test_agilex_bridge_rejects_missing_required_contact() -> None:
    with pytest.raises(ValueError, match="tactile.*required|missing.*tactile"):
        build_agilex_live_observation(
            _snapshot(_agilex_packets(contact=False)),
            spec=AgileXObservationSpec(),
            task_id="insert",
            prompt="insert the object",
            tactile_profile="vision_tactile",
            execution_dt_s=0.1,
            tactile_required=True,
            wrench_required=True,
        )


def test_franka_bridge_preserves_two_rgb_pose7_qpos8_and_excludes_contact() -> None:
    packets = _agilex_packets()
    franka_packets = (
        packets[0],
        packets[1],
        _packet("pose7", "pose", np.arange(7, dtype=np.float32)),
        _packet("qpos8", "qpos", np.arange(8, dtype=np.float32)),
        *packets[4:],
    )

    observation = build_franka_live_observation(
        _snapshot(franka_packets),
        spec=FrankaObservationSpec(),
        task_id="clear_up",
        prompt="clear the table",
    )

    images = observation["images"]
    assert isinstance(images, Mapping)
    assert tuple(images) == ("cam_high", "cam_left_wrist")
    np.testing.assert_array_equal(
        observation["left_end_pose"], np.arange(7, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        observation["joint_qpos"], np.arange(8, dtype=np.float32)
    )
    assert "tactile" not in observation
    assert "wrench" not in observation
    assert "timestamp" not in observation
