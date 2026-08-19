# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Latest-coherent snapshot selection and single-slot publication."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace

from .buffers import BoundedLatestRing
from .contracts import (
    CoherentSnapshot,
    SensorPacket,
    require_int,
    require_text,
)


@dataclass(frozen=True)
class SynchronizerConfig:
    """Signed limits for selecting a complete same-generation roster."""

    required_streams: tuple[str, ...]
    max_skew_ns: int
    max_age_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.required_streams, tuple):
            raise TypeError("required_streams must be a tuple")
        if not self.required_streams:
            raise ValueError("required_streams must not be empty")
        for stream_id in self.required_streams:
            require_text("required stream", stream_id)
        if len(set(self.required_streams)) != len(self.required_streams):
            raise ValueError("required_streams must not contain duplicates")
        require_int("max_skew_ns", self.max_skew_ns)
        require_int("max_age_ns", self.max_age_ns)


class LatestCoherentSynchronizer:
    """Choose the newest full roster, never independent per-stream latests."""

    _MAX_SNAPSHOT_RETRIES = 3

    def __init__(
        self,
        *,
        buffers: Mapping[str, BoundedLatestRing],
        config: SynchronizerConfig,
    ) -> None:
        if not isinstance(config, SynchronizerConfig):
            raise TypeError("config must be a SynchronizerConfig")
        copied = dict(buffers)
        if set(copied) != set(config.required_streams):
            raise ValueError("buffers must exactly match required_streams")
        if any(not isinstance(value, BoundedLatestRing) for value in copied.values()):
            raise TypeError("buffers must contain BoundedLatestRing values")
        self._buffers = copied
        self._config = config

    @property
    def buffers(self) -> Mapping[str, BoundedLatestRing]:
        return self._buffers.copy()

    @property
    def config(self) -> SynchronizerConfig:
        return self._config

    def build_latest(self, *, now_mono_ns: int) -> CoherentSnapshot | None:
        """Return the newest matchable full roster within skew and age bounds."""

        now_ns = require_int("now_mono_ns", now_mono_ns)
        for _ in range(self._MAX_SNAPSHOT_RETRIES):
            captured = {
                stream_id: self._buffers[stream_id].snapshot()
                for stream_id in self._config.required_streams
            }
            revisions = {
                stream_id: revision for stream_id, (revision, _) in captured.items()
            }
            views = {stream_id: packets for stream_id, (_, packets) in captured.items()}
            result = self._build_from_views(views, now_ns=now_ns)
            if all(
                self._buffers[stream_id].revision_is(revision)
                for stream_id, revision in revisions.items()
            ):
                return result
        return None

    def _build_from_views(
        self,
        views: Mapping[str, tuple[SensorPacket, ...]],
        *,
        now_ns: int,
    ) -> CoherentSnapshot | None:
        if any(not packets for packets in views.values()):
            return None
        common_generations = self._common_generations(views)
        candidates = sorted(
            {
                packet.clock.capture_mono_ns
                for packets in views.values()
                for packet in packets
                if packet.generation in common_generations
                and packet.clock.capture_mono_ns <= now_ns
            },
            reverse=True,
        )
        for reference_ns in candidates:
            age_ns = now_ns - reference_ns
            if age_ns > self._config.max_age_ns:
                break
            for generation in sorted(common_generations, reverse=True):
                selected = self._select_at_reference(
                    views,
                    generation=generation,
                    reference_ns=reference_ns,
                )
                if selected is None:
                    continue
                captures = [
                    packet.clock.capture_mono_ns for packet in selected.values()
                ]
                max_skew_ns = reference_ns - min(captures)
                if max_skew_ns > self._config.max_skew_ns:
                    continue
                return CoherentSnapshot(
                    packets=selected,
                    reference_mono_ns=reference_ns,
                    max_skew_ns=max_skew_ns,
                    age_ns=age_ns,
                    generation=generation,
                )
        return None

    @staticmethod
    def _common_generations(
        views: Mapping[str, tuple[SensorPacket, ...]],
    ) -> set[int]:
        generations: set[int] | None = None
        for packets in views.values():
            packet_generations = {packet.generation for packet in packets}
            generations = (
                packet_generations
                if generations is None
                else generations & packet_generations
            )
        return generations or set()

    def _select_at_reference(
        self,
        views: Mapping[str, tuple[SensorPacket, ...]],
        *,
        generation: int,
        reference_ns: int,
    ) -> dict[str, SensorPacket] | None:
        selected: dict[str, SensorPacket] = {}
        for stream_id in self._config.required_streams:
            eligible = (
                packet
                for packet in views[stream_id]
                if packet.generation == generation
                and packet.clock.capture_mono_ns <= reference_ns
            )
            packet = max(
                eligible,
                key=lambda item: (
                    item.clock.capture_mono_ns,
                    item.sequence_id,
                ),
                default=None,
            )
            if packet is None:
                return None
            selected[stream_id] = packet
        if (
            max(packet.clock.capture_mono_ns for packet in selected.values())
            != reference_ns
        ):
            return None
        if len({packet.clock.clock_domain_id for packet in selected.values()}) != 1:
            return None
        return selected


class LatestSnapshotSlot:
    """One atomic latest-overwrite slot with a monotonically increasing cursor."""

    def __init__(self, *, max_age_ns: int) -> None:
        self._max_age_ns = require_int("max_age_ns", max_age_ns)
        self._lock = threading.Lock()
        self._cursor = 0
        self._snapshot: CoherentSnapshot | None = None

    @property
    def cursor(self) -> int:
        with self._lock:
            return self._cursor

    def publish(self, snapshot: CoherentSnapshot) -> int:
        if not isinstance(snapshot, CoherentSnapshot):
            raise TypeError("snapshot must be a CoherentSnapshot")
        with self._lock:
            self._cursor += 1
            self._snapshot = replace(snapshot, snapshot_id=self._cursor)
            return self._cursor

    def take_newer_than(
        self,
        cursor: int,
        *,
        now_mono_ns: int,
    ) -> CoherentSnapshot | None:
        checked_cursor = require_int("cursor", cursor)
        now_ns = require_int("now_mono_ns", now_mono_ns)
        with self._lock:
            if self._snapshot is None or self._cursor <= checked_cursor:
                return None
            age_ns = now_ns - self._snapshot.reference_mono_ns
            if age_ns < 0 or age_ns > self._max_age_ns:
                self._snapshot = None
                return None
            return self._snapshot
