# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Threaded sensor acquisition stays independent from slow planning."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import pytest

from n0_twam.integrations.worldarena.realtime_sensors import (
    BoundedLatestRing,
    ClockStamp,
    LatestCoherentSynchronizer,
    PlannerResult,
    SensorPacket,
    SynchronizerConfig,
    ThreadedSensorRuntime,
)

STREAMS = ("rgb", "tactile", "qpos", "wrench")
TICK_NS = 100_000_000


def _packet(
    stream_id: str,
    sequence_id: int,
    *,
    generation: int = 1,
) -> SensorPacket:
    capture_ns = sequence_id * TICK_NS
    return SensorPacket(
        stream_id=stream_id,
        modality=stream_id,
        payload=np.array([sequence_id], dtype=np.float32),
        sequence_id=sequence_id,
        generation=generation,
        clock=ClockStamp(
            device_ns=None,
            arrival_mono_ns=capture_ns + 1,
            capture_mono_ns=capture_ns,
            capture_wall_ns=1_800_000_000_000_000_000 + capture_ns,
            clock_domain_id="accelerated-10hz",
            quality="synthetic",
            uncertainty_ns=0,
            correlation_generation=generation,
        ),
    )


class _LogicalClock:
    def __init__(self, initial_ns: int) -> None:
        self._value = initial_ns
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._value

    def set(self, value: int) -> None:
        with self._lock:
            self._value = value


class _QueueSource:
    def __init__(self, *, failure: BaseException | None = None) -> None:
        self._items: queue.Queue[SensorPacket | object] = queue.Queue(maxsize=1)
        self._closed_sentinel = object()
        self._failure = failure
        self._last_polled_sequence = 0
        self._poll_lock = threading.Lock()
        self.failure_raised = threading.Event()
        self.closed = False

    def feed(self, packet: SensorPacket) -> None:
        self._items.put(packet, timeout=1.0)

    @property
    def last_polled_sequence(self) -> int:
        with self._poll_lock:
            return self._last_polled_sequence

    @property
    def pending_count(self) -> int:
        return self._items.qsize()

    def poll(self, timeout_s: float) -> SensorPacket | None:
        if self._failure is not None:
            failure, self._failure = self._failure, None
            self.failure_raised.set()
            raise failure
        try:
            item = self._items.get(timeout=timeout_s)
        except queue.Empty:
            return None
        if item is self._closed_sentinel:
            return None
        packet = cast(SensorPacket, item)
        with self._poll_lock:
            self._last_polled_sequence = packet.sequence_id
        return packet

    def close(self) -> None:
        self.closed = True
        try:
            self._items.put_nowait(self._closed_sentinel)
        except queue.Full:
            self._items.get_nowait()
            self._items.put_nowait(self._closed_sentinel)


class _BlockingCloseSource(_QueueSource):
    def __init__(self) -> None:
        super().__init__()
        self.release_close = threading.Event()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        assert self.release_close.wait(timeout=1.0)
        super().close()


def _wait_until(predicate: Callable[[], bool], *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.001)
    raise AssertionError("condition was not satisfied before the test deadline")


def _runtime(
    sources: Mapping[str, _QueueSource],
    *,
    logical_clock: _LogicalClock,
) -> ThreadedSensorRuntime:
    buffers = {stream_id: BoundedLatestRing(capacity=4) for stream_id in sources}
    synchronizer = LatestCoherentSynchronizer(
        buffers=buffers,
        config=SynchronizerConfig(
            required_streams=tuple(sources),
            max_skew_ns=0,
            max_age_ns=TICK_NS,
        ),
    )
    return ThreadedSensorRuntime(
        sources=sources,
        buffers=buffers,
        synchronizer=synchronizer,
        monotonic_ns=logical_clock,
    )


def _latest_sequence(runtime: ThreadedSensorRuntime, cursor: int) -> int | None:
    snapshot = runtime.take_latest_snapshot(cursor)
    if snapshot is None:
        return None
    return int(min(packet.sequence_id for packet in snapshot.packets.values()))


def test_accelerated_10hz_capture_skips_history_during_3_4s_fake_planner() -> None:
    clock = _LogicalClock(TICK_NS)
    sources = {stream_id: _QueueSource() for stream_id in STREAMS}
    runtime = _runtime(sources, logical_clock=clock)
    planner_started = threading.Event()
    release_planner = threading.Event()
    planner_finished = threading.Event()
    runtime.start()

    try:
        for source_id, source in sources.items():
            source.feed(_packet(source_id, 1))
        _wait_until(lambda: _latest_sequence(runtime, 0) == 1)
        first = runtime.take_latest_snapshot(0)
        assert first is not None
        assert runtime.planner_results.active_generation == first.generation

        def fake_3_4s_planner() -> None:
            planner_started.set()
            assert release_planner.wait(timeout=1.0)
            planner_finished.set()

        planner = threading.Thread(target=fake_3_4s_planner, daemon=True)
        planner.start()
        assert planner_started.wait(timeout=1.0)

        # Thirty-four logical 10 Hz ticks are delivered without a 3.4 s wall sleep.
        for sequence_id in range(2, 35):
            clock.set(sequence_id * TICK_NS)
            for source_id, source in sources.items():
                source.feed(_packet(source_id, sequence_id))
            _wait_until(
                lambda: all(
                    source.last_polled_sequence == sequence_id
                    for source in sources.values()
                )
            )

        _wait_until(lambda: _latest_sequence(runtime, first.snapshot_id) == 34)
        assert planner_finished.is_set() is False
        assert all(length <= 4 for length in runtime.buffer_lengths.values())
        assert runtime.buffer_lengths == dict.fromkeys(STREAMS, 4)
        assert runtime.overwrite_counts == dict.fromkeys(STREAMS, 30)
        assert all(source.pending_count == 0 for source in sources.values())

        release_planner.set()
        assert planner_finished.wait(timeout=1.0)
        next_snapshot = runtime.take_latest_snapshot(first.snapshot_id)
        assert next_snapshot is not None
        assert {packet.sequence_id for packet in next_snapshot.packets.values()} == {34}
        assert all(packet.sequence_id != 2 for packet in next_snapshot.packets.values())
        runtime.raise_if_failed()
    finally:
        release_planner.set()
        runtime.stop(timeout_s=1.0)

    assert all(source.closed for source in sources.values())


def test_worker_exception_is_propagated_and_stop_closes_every_source() -> None:
    clock = _LogicalClock(TICK_NS)
    sources = {
        "rgb": _QueueSource(failure=RuntimeError("camera transport failed")),
        "qpos": _QueueSource(),
    }
    runtime = _runtime(sources, logical_clock=clock)
    runtime.start()

    try:
        assert sources["rgb"].failure_raised.wait(timeout=1.0)

        def failure_is_visible() -> bool:
            try:
                runtime.raise_if_failed()
            except RuntimeError:
                return True
            return False

        _wait_until(failure_is_visible)
        with pytest.raises(RuntimeError, match="rgb.*camera transport failed"):
            runtime.raise_if_failed()
    finally:
        runtime.stop(timeout_s=1.0)

    assert all(source.closed for source in sources.values())
    runtime.stop(timeout_s=1.0)


def test_runtime_rejects_same_names_backed_by_different_rings() -> None:
    sources = {"rgb": _QueueSource()}
    runtime_buffers = {"rgb": BoundedLatestRing(capacity=4)}
    synchronizer = LatestCoherentSynchronizer(
        buffers={"rgb": BoundedLatestRing(capacity=4)},
        config=SynchronizerConfig(
            required_streams=("rgb",),
            max_skew_ns=0,
            max_age_ns=TICK_NS,
        ),
    )

    with pytest.raises(ValueError, match="same rings"):
        ThreadedSensorRuntime(
            sources=sources,
            buffers=runtime_buffers,
            synchronizer=synchronizer,
        )


def test_stop_timeout_can_be_retried_without_closing_a_source_twice() -> None:
    source = _BlockingCloseSource()
    clock = _LogicalClock(TICK_NS)
    runtime = _runtime({"rgb": source}, logical_clock=clock)
    runtime.start()

    with pytest.raises(TimeoutError, match="sensor-close-rgb"):
        runtime.stop(timeout_s=0.01)
    source.release_close.set()
    runtime.stop(timeout_s=1.0)

    assert source.close_calls == 1
    assert source.closed is True


def test_one_stream_generation_advance_immediately_invalidates_old_result() -> None:
    clock = _LogicalClock(TICK_NS)
    sources = {stream_id: _QueueSource() for stream_id in ("rgb", "qpos")}
    runtime = _runtime(sources, logical_clock=clock)
    runtime.start()

    try:
        for stream_id, source in sources.items():
            source.feed(_packet(stream_id, 1))
        _wait_until(lambda: _latest_sequence(runtime, 0) == 1)
        snapshot = runtime.take_latest_snapshot(0)
        assert snapshot is not None
        request = runtime.planner_results.submit(
            snapshot=snapshot,
            created_mono_ns=TICK_NS + 1,
            deadline_mono_ns=5 * TICK_NS,
            payload={"qpos": np.zeros(14, dtype=np.float32)},
        )

        clock.set(2 * TICK_NS)
        sources["rgb"].feed(_packet("rgb", 2, generation=2))
        _wait_until(lambda: runtime.planner_results.active_generation == 2)
        result = PlannerResult(
            request_id=request.request_id,
            snapshot_id=request.snapshot_id,
            generation=request.generation,
            started_mono_ns=TICK_NS + 2,
            finished_mono_ns=2 * TICK_NS,
            payload={"actions": np.zeros((12, 14), dtype=np.float32)},
        )

        decision = runtime.planner_results.publish(
            result,
            now_mono_ns=2 * TICK_NS,
        )
        assert decision.accepted is False
        assert (
            runtime.planner_results.take_newer_than(
                0,
                now_mono_ns=2 * TICK_NS,
            )
            is None
        )
    finally:
        runtime.stop(timeout_s=1.0)
