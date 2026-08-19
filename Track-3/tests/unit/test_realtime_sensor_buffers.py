# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed contracts for bounded latest-only sensor buffers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from n0_twam.integrations.worldarena.realtime_sensors import (
    BoundedLatestRing,
    ClockStamp,
    SensorPacket,
)


def _clock(capture_mono_ns: int, *, generation: int = 1) -> ClockStamp:
    return ClockStamp(
        device_ns=None,
        arrival_mono_ns=capture_mono_ns + 1_000,
        capture_mono_ns=capture_mono_ns,
        capture_wall_ns=1_800_000_000_000_000_000 + capture_mono_ns,
        clock_domain_id="host-monotonic",
        quality="host_arrival",
        uncertainty_ns=1_000_000,
        correlation_generation=generation,
    )


def _packet(
    sequence_id: int,
    capture_mono_ns: int,
    *,
    generation: int = 1,
    stream_id: str = "rgb.top",
    payload: object | None = None,
) -> SensorPacket:
    return SensorPacket(
        stream_id=stream_id,
        modality="rgb",
        payload=(
            np.full((2, 3, 3), sequence_id, dtype=np.uint8)
            if payload is None
            else payload
        ),
        sequence_id=sequence_id,
        generation=generation,
        clock=_clock(capture_mono_ns, generation=generation),
    )


def test_sensor_packet_owns_a_read_only_contiguous_payload_copy() -> None:
    source = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)[:, ::-1]
    packet = _packet(1, 100, payload=source)

    source[...] = 255

    stored = np.asarray(packet.payload)
    np.testing.assert_array_equal(
        stored,
        np.array(
            [
                [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                [[15, 16, 17], [12, 13, 14], [9, 10, 11]],
            ],
            dtype=np.uint8,
        ),
    )
    assert stored.flags.c_contiguous
    assert not stored.flags.writeable
    with pytest.raises(ValueError, match="read-only|assignment destination"):
        stored[0, 0, 0] = 0


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "capacity",
    (1, 5, True),
)
def test_ring_rejects_capacity_outside_two_to_four(capacity: object) -> None:
    with pytest.raises((TypeError, ValueError), match="capacity"):
        BoundedLatestRing(capacity=cast(int, capacity))


def test_ring_drops_oldest_and_keeps_only_latest_four_of_thirty_four() -> None:
    ring = BoundedLatestRing(capacity=4)

    for sequence_id in range(1, 35):
        ring.push(_packet(sequence_id, sequence_id * 100_000_000))

    view = ring.view()
    assert len(view) == 4
    assert [packet.sequence_id for packet in view] == [31, 32, 33, 34]
    assert ring.overwrite_count == 30


def test_ring_rejects_duplicate_or_regressed_sequence_without_mutation() -> None:
    ring = BoundedLatestRing(capacity=4)
    accepted = _packet(7, 700)
    ring.push(accepted)

    with pytest.raises(ValueError, match="sequence"):
        ring.push(_packet(7, 800))
    with pytest.raises(ValueError, match="sequence"):
        ring.push(_packet(6, 900))

    assert ring.view() == (accepted,)
    assert ring.overwrite_count == 0


def test_ring_rejects_capture_time_regression_without_mutation() -> None:
    ring = BoundedLatestRing(capacity=4)
    accepted = _packet(7, 700)
    ring.push(accepted)

    with pytest.raises(ValueError, match="capture_mono_ns|timestamp|time"):
        ring.push(_packet(8, 699))

    assert ring.view() == (accepted,)


def test_ring_accepts_generation_advance_then_rejects_old_generation() -> None:
    ring = BoundedLatestRing(capacity=4)
    ring.push(_packet(9, 900, generation=3))
    current = _packet(0, 1_000, generation=4)
    ring.push(current)

    with pytest.raises(ValueError, match="generation"):
        ring.push(_packet(10, 1_100, generation=3))

    assert ring.view()[-1] is current
    assert ring.view()[-1].generation == 4
