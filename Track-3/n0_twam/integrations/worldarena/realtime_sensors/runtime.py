# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Threaded source acquisition independent from slow model planning."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Protocol

from .buffers import BoundedLatestRing
from .contracts import CoherentSnapshot, SensorPacket
from .planner import GatedPlannerResultSlot
from .synchronizer import LatestCoherentSynchronizer, LatestSnapshotSlot


class SensorSource(Protocol):
    """Minimal transport-neutral source boundary used by the runtime."""

    def poll(self, timeout_s: float) -> SensorPacket | None:
        """Return the next available sample or None at the timeout."""

        ...

    def close(self) -> None:
        """Unblock any pending poll and release the source."""

        ...


class ThreadedSensorRuntime:
    """Run every source independently and publish only the latest snapshot."""

    def __init__(
        self,
        *,
        sources: Mapping[str, SensorSource],
        buffers: Mapping[str, BoundedLatestRing],
        synchronizer: LatestCoherentSynchronizer,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        poll_timeout_s: float = 0.01,
    ) -> None:
        copied_sources = dict(sources)
        copied_buffers = dict(buffers)
        if not copied_sources:
            raise ValueError("sources must not be empty")
        if set(copied_sources) != set(copied_buffers):
            raise ValueError("sources and buffers must have the same stream roster")
        if set(copied_sources) != set(synchronizer.config.required_streams):
            raise ValueError("runtime roster must match synchronizer required_streams")
        synchronized_buffers = synchronizer.buffers
        if any(
            synchronized_buffers[stream_id] is not copied_buffers[stream_id]
            for stream_id in copied_buffers
        ):
            raise ValueError("runtime and synchronizer must share the same rings")
        if any(
            not isinstance(value, BoundedLatestRing)
            for value in copied_buffers.values()
        ):
            raise TypeError("buffers must contain BoundedLatestRing values")
        if isinstance(poll_timeout_s, bool) or not isinstance(
            poll_timeout_s, (int, float)
        ):
            raise TypeError("poll_timeout_s must be a finite positive number")
        checked_timeout = float(poll_timeout_s)
        if not math.isfinite(checked_timeout) or checked_timeout <= 0.0:
            raise ValueError("poll_timeout_s must be a finite positive number")
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable")

        self._sources = copied_sources
        self._buffers = copied_buffers
        self._synchronizer = synchronizer
        self._monotonic_ns = monotonic_ns
        self._poll_timeout_s = checked_timeout
        self._lifecycle_lock = threading.RLock()
        self._failure_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sample_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._close_threads: list[threading.Thread] = []
        self._failure: tuple[str, BaseException] | None = None
        self._started = False
        self._stop_requested = False
        self._stopped = False
        self._last_published: tuple[tuple[str, int, int, int], ...] | None = None
        self.latest_slot = LatestSnapshotSlot(max_age_ns=synchronizer.config.max_age_ns)
        self.planner_results = GatedPlannerResultSlot()

    @property
    def buffer_lengths(self) -> dict[str, int]:
        return {stream_id: len(buffer) for stream_id, buffer in self._buffers.items()}

    @property
    def overwrite_counts(self) -> dict[str, int]:
        return {
            stream_id: buffer.overwrite_count
            for stream_id, buffer in self._buffers.items()
        }

    def take_latest_snapshot(self, cursor: int) -> CoherentSnapshot | None:
        """Take only a still-fresh snapshot using the runtime monotonic clock."""

        return self.latest_slot.take_newer_than(
            cursor,
            now_mono_ns=self._monotonic_ns(),
        )

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._stopped or self._stop_requested:
                raise RuntimeError("a stopped sensor runtime cannot be restarted")
            if self._started:
                raise RuntimeError("sensor runtime is already started")
            self._started = True
            for stream_id, source in self._sources.items():
                thread = threading.Thread(
                    target=self._source_worker,
                    args=(stream_id, source),
                    name=f"sensor-{stream_id}",
                    daemon=True,
                )
                self._threads.append(thread)
            self._threads.append(
                threading.Thread(
                    target=self._synchronizer_worker,
                    name="sensor-synchronizer",
                    daemon=True,
                )
            )
            for thread in self._threads:
                thread.start()

    def stop(self, *, timeout_s: float) -> None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("timeout_s must be a finite non-negative number")
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("timeout_s must be a finite non-negative number")
        with self._lifecycle_lock:
            if self._stopped:
                return
            if not self._stop_requested:
                self._stop_requested = True
                self._stop_event.set()
                self._sample_event.set()
                for stream_id, source in self._sources.items():
                    thread = threading.Thread(
                        target=self._close_source,
                        args=(stream_id, source),
                        name=f"sensor-close-{stream_id}",
                        daemon=True,
                    )
                    self._close_threads.append(thread)
                    thread.start()
            threads = tuple((*self._close_threads, *self._threads))

        deadline = time.monotonic() + timeout
        for thread in threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(remaining)
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            raise TimeoutError(f"sensor runtime workers did not stop: {alive}")
        with self._lifecycle_lock:
            self._stopped = True

    def raise_if_failed(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is None:
            return
        worker, cause = failure
        raise RuntimeError(f"{worker}: {cause}") from cause

    def _source_worker(self, stream_id: str, source: SensorSource) -> None:
        try:
            while not self._stop_event.is_set():
                packet = source.poll(self._poll_timeout_s)
                if packet is None:
                    continue
                if not isinstance(packet, SensorPacket):
                    raise TypeError("source poll must return SensorPacket or None")
                if packet.stream_id != stream_id:
                    raise ValueError(
                        f"source {stream_id!r} returned packet for {packet.stream_id!r}"
                    )
                self.planner_results.activate_generation(packet.generation)
                self._buffers[stream_id].push(packet)
                self._sample_event.set()
        except BaseException as exc:
            self._record_failure(f"sensor worker {stream_id}", exc)
            self._stop_event.set()
            self._sample_event.set()

    def _close_source(self, stream_id: str, source: SensorSource) -> None:
        try:
            source.close()
        except BaseException as exc:
            self._record_failure(f"{stream_id} close", exc)

    def _synchronizer_worker(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._sample_event.wait(self._poll_timeout_s)
                self._sample_event.clear()
                snapshot = self._synchronizer.build_latest(
                    now_mono_ns=self._monotonic_ns()
                )
                if snapshot is None:
                    continue
                fingerprint = self._fingerprint(snapshot)
                if fingerprint == self._last_published:
                    continue
                self.planner_results.activate_generation(snapshot.generation)
                self.latest_slot.publish(snapshot)
                self._last_published = fingerprint
        except BaseException as exc:
            self._record_failure("synchronizer worker", exc)
            self._stop_event.set()

    @staticmethod
    def _fingerprint(
        snapshot: CoherentSnapshot,
    ) -> tuple[tuple[str, int, int, int], ...]:
        return tuple(
            (
                stream_id,
                packet.generation,
                packet.sequence_id,
                packet.clock.capture_mono_ns,
            )
            for stream_id, packet in snapshot.packets.items()
        )

    def _record_failure(self, worker: str, cause: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = (worker, cause)
