# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Small thread-safe rings that always favor the newest sensor samples."""

from __future__ import annotations

import threading
from collections import deque

from .contracts import SensorPacket, require_int


class BoundedLatestRing:
    """A capacity-two-to-four drop-oldest ring for one logical stream."""

    def __init__(self, *, capacity: int) -> None:
        checked_capacity = require_int("capacity", capacity)
        if checked_capacity < 2 or checked_capacity > 4:
            raise ValueError("capacity must be between 2 and 4 inclusive")
        self._capacity = checked_capacity
        self._packets: deque[SensorPacket] = deque(maxlen=checked_capacity)
        self._lock = threading.RLock()
        self._overwrite_count = 0
        self._revision = 0
        self._stream_id: str | None = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def overwrite_count(self) -> int:
        with self._lock:
            return self._overwrite_count

    def __len__(self) -> int:
        with self._lock:
            return len(self._packets)

    def push(self, packet: SensorPacket) -> None:
        """Append a fresh packet, rejecting timeline regressions atomically."""

        if not isinstance(packet, SensorPacket):
            raise TypeError("packet must be a SensorPacket")
        with self._lock:
            if self._stream_id is not None and packet.stream_id != self._stream_id:
                raise ValueError("packet stream_id does not match this ring")
            previous = self._packets[-1] if self._packets else None
            if previous is not None:
                self._validate_advance(previous, packet)
            if previous is not None and packet.generation > previous.generation:
                self._packets.clear()
            if len(self._packets) == self._capacity:
                self._overwrite_count += 1
            self._packets.append(packet)
            self._revision += 1
            if self._stream_id is None:
                self._stream_id = packet.stream_id

    @staticmethod
    def _validate_advance(previous: SensorPacket, packet: SensorPacket) -> None:
        if packet.generation < previous.generation:
            raise ValueError("packet generation regressed")
        previous_capture = previous.clock.capture_mono_ns
        packet_capture = packet.clock.capture_mono_ns
        if packet_capture <= previous_capture:
            raise ValueError("packet capture_mono_ns must increase")
        if (
            packet.generation == previous.generation
            and packet.sequence_id <= previous.sequence_id
        ):
            raise ValueError("packet sequence_id must increase within a generation")

    def view(self) -> tuple[SensorPacket, ...]:
        """Return a stable immutable view without exposing the deque."""

        return self.snapshot()[1]

    def snapshot(self) -> tuple[int, tuple[SensorPacket, ...]]:
        """Return one atomic revision and immutable packet view."""

        with self._lock:
            return self._revision, tuple(self._packets)

    def revision_is(self, revision: int) -> bool:
        """Check that no packet was pushed since an atomic snapshot."""

        checked_revision = require_int("revision", revision)
        with self._lock:
            return self._revision == checked_revision
