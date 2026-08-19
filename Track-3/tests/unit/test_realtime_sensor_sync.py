# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Latest coherent snapshot and planner-result freshness contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pytest

from n0_twam.integrations.worldarena.realtime_sensors import (
    BoundedLatestRing,
    ClockStamp,
    CoherentSnapshot,
    GatedPlannerResultSlot,
    LatestCoherentSynchronizer,
    LatestSnapshotSlot,
    PlannerRequest,
    PlannerResult,
    PlannerResultGate,
    SensorPacket,
    SynchronizerConfig,
)

STREAMS = ("rgb", "tactile", "qpos", "wrench")


def _packet(
    stream_id: str,
    sequence_id: int,
    capture_mono_ns: int,
    *,
    generation: int = 1,
    clock_domain_id: str = "host-monotonic",
) -> SensorPacket:
    clock = ClockStamp(
        device_ns=None,
        arrival_mono_ns=capture_mono_ns + 1,
        capture_mono_ns=capture_mono_ns,
        capture_wall_ns=1_800_000_000_000_000_000 + capture_mono_ns,
        clock_domain_id=clock_domain_id,
        quality="host_arrival",
        uncertainty_ns=1,
        correlation_generation=generation,
    )
    return SensorPacket(
        stream_id=stream_id,
        modality=stream_id,
        payload=np.array([sequence_id], dtype=np.float32),
        sequence_id=sequence_id,
        generation=generation,
        clock=clock,
    )


def _buffers() -> dict[str, BoundedLatestRing]:
    return {stream_id: BoundedLatestRing(capacity=4) for stream_id in STREAMS}


def _synchronizer(
    buffers: Mapping[str, BoundedLatestRing],
    *,
    max_skew_ns: int = 5,
    max_age_ns: int = 50,
) -> LatestCoherentSynchronizer:
    return LatestCoherentSynchronizer(
        buffers=buffers,
        config=SynchronizerConfig(
            required_streams=STREAMS,
            max_skew_ns=max_skew_ns,
            max_age_ns=max_age_ns,
        ),
    )


def _push_roster(
    buffers: Mapping[str, BoundedLatestRing],
    *,
    sequence_id: int,
    captures: tuple[int, int, int, int],
    generation: int = 1,
) -> None:
    for stream_id, capture in zip(STREAMS, captures, strict=True):
        buffers[stream_id].push(
            _packet(
                stream_id,
                sequence_id,
                capture,
                generation=generation,
            )
        )


def test_synchronizer_selects_newest_full_coherent_roster_not_independent_latest() -> (
    None
):
    buffers = _buffers()
    _push_roster(buffers, sequence_id=1, captures=(100, 102, 101, 103))
    buffers["rgb"].push(_packet("rgb", 2, 140))
    buffers["tactile"].push(_packet("tactile", 2, 141))

    snapshot = _synchronizer(buffers).build_latest(now_mono_ns=145)

    assert snapshot is not None
    assert tuple(snapshot.packets) == STREAMS
    assert [snapshot.packets[key].sequence_id for key in STREAMS] == [1, 1, 1, 1]
    assert snapshot.reference_mono_ns == 103
    assert snapshot.max_skew_ns == 3
    assert snapshot.age_ns == 42
    assert snapshot.generation == 1


def test_synchronizer_never_publishes_a_roster_invalidated_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb = BoundedLatestRing(capacity=4)
    qpos = BoundedLatestRing(capacity=4)
    rgb.push(_packet("rgb", 1, 100))
    qpos.push(_packet("qpos", 1, 100))
    original_snapshot: Callable[[], tuple[int, tuple[SensorPacket, ...]]] = rgb.snapshot
    advanced = False

    def snapshot_then_advance() -> tuple[int, tuple[SensorPacket, ...]]:
        nonlocal advanced
        captured = original_snapshot()
        if not advanced:
            advanced = True
            rgb.push(_packet("rgb", 1, 200, generation=2))
        return captured

    monkeypatch.setattr(rgb, "snapshot", snapshot_then_advance)
    synchronizer = LatestCoherentSynchronizer(
        buffers={"rgb": rgb, "qpos": qpos},
        config=SynchronizerConfig(
            required_streams=("rgb", "qpos"),
            max_skew_ns=0,
            max_age_ns=200,
        ),
    )

    assert synchronizer.build_latest(now_mono_ns=210) is None


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "failure",
    ("missing", "skew", "stale", "mixed_generation", "mixed_clock_domain"),
)
def test_synchronizer_rejects_incoherent_or_unsafe_roster(failure: str) -> None:
    buffers = _buffers()
    captures = (100, 101, 102, 103)
    for index, (stream_id, capture) in enumerate(zip(STREAMS, captures, strict=True)):
        if failure == "missing" and stream_id == "wrench":
            continue
        if failure == "skew" and stream_id == "wrench":
            capture = 120
        generation = 2 if failure == "mixed_generation" and index == 3 else 1
        clock_domain_id = (
            "foreign-clock"
            if failure == "mixed_clock_domain" and index == 3
            else "host-monotonic"
        )
        buffers[stream_id].push(
            _packet(
                stream_id,
                1,
                capture,
                generation=generation,
                clock_domain_id=clock_domain_id,
            )
        )

    now = 200 if failure == "stale" else 110
    assert _synchronizer(buffers).build_latest(now_mono_ns=now) is None


def test_latest_snapshot_slot_overwrites_one_through_thirty_four_without_fifo() -> None:
    buffers = _buffers()
    synchronizer = _synchronizer(buffers, max_skew_ns=0, max_age_ns=1)
    slot = LatestSnapshotSlot(max_age_ns=1)
    first_cursor = 0
    last_cursor = 0

    for sequence_id in range(1, 35):
        capture = sequence_id * 100
        _push_roster(
            buffers,
            sequence_id=sequence_id,
            captures=(capture, capture, capture, capture),
        )
        snapshot = synchronizer.build_latest(now_mono_ns=capture)
        assert snapshot is not None
        last_cursor = slot.publish(snapshot)
        if sequence_id == 1:
            first_cursor = last_cursor

    latest = slot.take_newer_than(first_cursor, now_mono_ns=3_400)
    assert latest is not None
    assert [latest.packets[key].sequence_id for key in STREAMS] == [34, 34, 34, 34]
    assert slot.take_newer_than(last_cursor, now_mono_ns=3_400) is None


def test_latest_snapshot_slot_drops_a_snapshot_that_expires_before_take() -> None:
    buffers = _buffers()
    _push_roster(buffers, sequence_id=1, captures=(100, 100, 100, 100))
    snapshot = _synchronizer(
        buffers,
        max_skew_ns=0,
        max_age_ns=10,
    ).build_latest(now_mono_ns=100)
    assert snapshot is not None
    slot = LatestSnapshotSlot(max_age_ns=10)
    slot.publish(snapshot)

    assert slot.take_newer_than(0, now_mono_ns=111) is None


def _request(*, request_id: int = 7, generation: int = 3) -> PlannerRequest:
    return PlannerRequest(
        request_id=request_id,
        snapshot_id=34,
        generation=generation,
        created_mono_ns=1_000,
        deadline_mono_ns=5_000,
        payload={"qpos": np.zeros(14, dtype=np.float32)},
    )


def _result(*, request_id: int = 7, generation: int = 3) -> PlannerResult:
    return PlannerResult(
        request_id=request_id,
        snapshot_id=34,
        generation=generation,
        started_mono_ns=1_100,
        finished_mono_ns=4_000,
        payload={"actions": np.zeros((12, 14), dtype=np.float32)},
    )


def test_planner_gate_accepts_matching_result_even_when_new_sensor_frames_exist() -> (
    None
):
    decision = PlannerResultGate().evaluate(
        request=_request(),
        result=_result(),
        now_mono_ns=4_100,
        latest_request_id=7,
        active_generation=3,
    )

    assert decision.accepted is True
    assert decision.reason == "accepted"


def _published_snapshot(*, generation: int = 3) -> CoherentSnapshot:
    ring = BoundedLatestRing(capacity=4)
    ring.push(_packet("rgb", 34, 900, generation=generation))
    synchronizer = LatestCoherentSynchronizer(
        buffers={"rgb": ring},
        config=SynchronizerConfig(
            required_streams=("rgb",),
            max_skew_ns=0,
            max_age_ns=10,
        ),
    )
    snapshot = synchronizer.build_latest(now_mono_ns=900)
    assert snapshot is not None
    slot = LatestSnapshotSlot(max_age_ns=10)
    slot.publish(snapshot)
    published = slot.take_newer_than(0, now_mono_ns=900)
    assert published is not None
    return published


def test_gated_result_slot_drops_expired_and_superseded_results() -> None:
    slot = GatedPlannerResultSlot()
    snapshot = _published_snapshot()
    first = slot.submit(
        snapshot=snapshot,
        created_mono_ns=1_000,
        deadline_mono_ns=5_000,
        payload={"qpos": np.zeros(14, dtype=np.float32)},
    )
    second = slot.submit(
        snapshot=snapshot,
        created_mono_ns=2_000,
        deadline_mono_ns=5_000,
        payload={"qpos": np.ones(14, dtype=np.float32)},
    )
    stale = PlannerResult(
        request_id=first.request_id,
        snapshot_id=first.snapshot_id,
        generation=first.generation,
        started_mono_ns=1_100,
        finished_mono_ns=4_000,
        payload={"actions": np.zeros((12, 14), dtype=np.float32)},
    )
    expired = PlannerResult(
        request_id=second.request_id,
        snapshot_id=second.snapshot_id,
        generation=second.generation,
        started_mono_ns=2_100,
        finished_mono_ns=5_001,
        payload={"actions": np.zeros((12, 14), dtype=np.float32)},
    )

    assert "superseded" in slot.publish(stale, now_mono_ns=4_100).reason
    assert "deadline" in slot.publish(expired, now_mono_ns=5_001).reason
    assert slot.take_newer_than(0, now_mono_ns=5_001) is None


def test_gated_result_slot_only_exposes_a_matching_snapshot_bound_result() -> None:
    slot = GatedPlannerResultSlot()
    snapshot = _published_snapshot()
    request = slot.submit(
        snapshot=snapshot,
        created_mono_ns=1_000,
        deadline_mono_ns=5_000,
        payload={"qpos": np.zeros(14, dtype=np.float32)},
    )
    result = PlannerResult(
        request_id=request.request_id,
        snapshot_id=request.snapshot_id,
        generation=request.generation,
        started_mono_ns=1_100,
        finished_mono_ns=4_000,
        payload={"actions": np.zeros((12, 14), dtype=np.float32)},
    )

    assert slot.publish(result, now_mono_ns=4_100).accepted is True
    publication = slot.take_newer_than(0, now_mono_ns=4_100)
    assert publication is not None
    assert publication.request.snapshot_id == snapshot.snapshot_id
    assert publication.result is result
    assert "duplicate" in slot.publish(result, now_mono_ns=4_100).reason


def test_gated_result_slot_drops_a_result_that_expires_before_consumption() -> None:
    slot = GatedPlannerResultSlot()
    snapshot = _published_snapshot()
    request = slot.submit(
        snapshot=snapshot,
        created_mono_ns=1_000,
        deadline_mono_ns=5_000,
        payload={"qpos": np.zeros(14, dtype=np.float32)},
    )
    result = PlannerResult(
        request_id=request.request_id,
        snapshot_id=request.snapshot_id,
        generation=request.generation,
        started_mono_ns=1_100,
        finished_mono_ns=4_900,
        payload={"actions": np.zeros((12, 14), dtype=np.float32)},
    )

    assert slot.publish(result, now_mono_ns=4_900).accepted is True
    assert slot.take_newer_than(0, now_mono_ns=5_100) is None


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    (
        "request_obj",
        "result",
        "now",
        "latest_request_id",
        "generation",
        "reason",
    ),
    (
        (_request(), _result(), 5_001, 7, 3, "deadline"),
        (_request(), _result(generation=2), 4_100, 7, 3, "generation"),
        (_request(), _result(request_id=6), 4_100, 7, 3, "request"),
        (_request(), _result(), 4_100, 8, 3, "superseded"),
    ),
)
def test_planner_gate_rejects_expired_mismatched_or_superseded_result(
    request_obj: PlannerRequest,
    result: PlannerResult,
    now: int,
    latest_request_id: int,
    generation: int,
    reason: str,
) -> None:
    decision = PlannerResultGate().evaluate(
        request=request_obj,
        result=result,
        now_mono_ns=now,
        latest_request_id=latest_request_id,
        active_generation=generation,
    )

    assert decision.accepted is False
    assert reason in decision.reason
