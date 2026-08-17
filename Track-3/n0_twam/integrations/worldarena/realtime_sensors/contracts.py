# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Immutable contracts shared by the real-time sensor runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray


def require_int(
    name: str,
    value: object,
    *,
    minimum: int = 0,
) -> int:
    """Return a strict integer, rejecting bool and values below ``minimum``."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, not bool or another numeric type")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def require_text(name: str, value: object) -> str:
    """Return a non-empty string without silently coercing another type."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def owned_read_only_array(value: object, *, name: str) -> NDArray[Any]:
    """Take an owned, C-contiguous copy and expose it as read-only."""

    try:
        result = np.array(value, copy=True, order="C", subok=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be array-like") from exc
    if result.dtype.hasobject:
        raise TypeError(f"{name} must not use an object dtype")
    result.setflags(write=False)
    return result


def _freeze_payload_value(value: object, *, name: str) -> object:
    if isinstance(value, np.ndarray):
        return owned_read_only_array(value, name=name)
    if isinstance(value, Mapping):
        return freeze_payload(value, name=name)
    if isinstance(value, tuple):
        return tuple(
            _freeze_payload_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return tuple(
            _freeze_payload_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise TypeError(f"{name} contains unsupported mutable value {type(value)!r}")


def freeze_payload(
    payload: Mapping[str, object],
    *,
    name: str,
) -> MappingProxyType[str, object]:
    """Copy a string-keyed payload into an immutable mapping."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen: dict[str, object] = {}
    for key, value in payload.items():
        require_text(f"{name} key", key)
        frozen[key] = _freeze_payload_value(value, name=f"{name}[{key!r}]")
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class ClockStamp:
    """One sensor timestamp correlated into the host monotonic clock domain."""

    device_ns: int | None
    arrival_mono_ns: int
    capture_mono_ns: int
    capture_wall_ns: int
    clock_domain_id: str
    quality: str
    uncertainty_ns: int
    correlation_generation: int

    def __post_init__(self) -> None:
        if self.device_ns is not None:
            require_int("device_ns", self.device_ns)
        arrival_ns = require_int("arrival_mono_ns", self.arrival_mono_ns)
        capture_ns = require_int("capture_mono_ns", self.capture_mono_ns)
        require_int("capture_wall_ns", self.capture_wall_ns)
        require_text("clock_domain_id", self.clock_domain_id)
        require_text("quality", self.quality)
        require_int("uncertainty_ns", self.uncertainty_ns)
        require_int("correlation_generation", self.correlation_generation)
        if arrival_ns < capture_ns:
            raise ValueError("arrival_mono_ns must not precede capture_mono_ns")


@dataclass(frozen=True)
class SensorPacket:
    """One immutable sample emitted by exactly one sensor source."""

    stream_id: str
    modality: str
    payload: object
    sequence_id: int
    generation: int
    clock: ClockStamp

    def __post_init__(self) -> None:
        require_text("stream_id", self.stream_id)
        require_text("modality", self.modality)
        require_int("sequence_id", self.sequence_id)
        generation = require_int("generation", self.generation)
        if not isinstance(self.clock, ClockStamp):
            raise TypeError("clock must be a ClockStamp")
        if generation != self.clock.correlation_generation:
            raise ValueError("generation must match clock.correlation_generation")
        object.__setattr__(
            self,
            "payload",
            owned_read_only_array(self.payload, name="payload"),
        )


@dataclass(frozen=True)
class CoherentSnapshot:
    """A full, same-generation roster selected on one monotonic timeline."""

    packets: Mapping[str, SensorPacket]
    reference_mono_ns: int
    max_skew_ns: int
    age_ns: int
    generation: int
    snapshot_id: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.packets, Mapping):
            raise TypeError("packets must be a mapping")
        copied: dict[str, SensorPacket] = {}
        for key, packet in self.packets.items():
            require_text("packet key", key)
            if not isinstance(packet, SensorPacket):
                raise TypeError("packets must contain SensorPacket values")
            if key != packet.stream_id:
                raise ValueError("packet key must match packet.stream_id")
            copied[key] = packet
        if not copied:
            raise ValueError("packets must not be empty")
        generation = require_int("generation", self.generation)
        if any(packet.generation != generation for packet in copied.values()):
            raise ValueError("snapshot packets must have one generation")
        require_int("reference_mono_ns", self.reference_mono_ns)
        require_int("max_skew_ns", self.max_skew_ns)
        require_int("age_ns", self.age_ns)
        require_int("snapshot_id", self.snapshot_id)
        object.__setattr__(self, "packets", MappingProxyType(copied))
