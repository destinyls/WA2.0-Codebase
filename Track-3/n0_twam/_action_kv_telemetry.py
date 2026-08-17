# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Small accelerator-aware timing utility for Action KV serving receipts."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import torch

_T = TypeVar("_T")

ACTION_KV_TIMING_PHASES = (
    "prepare",
    "cached",
    "terminal",
    "full_reference",
    "fallback_replay",
)


def _synchronize_default_accelerator() -> None:
    """Synchronize the active eager accelerator before wall-clock sampling."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    hpu = getattr(torch, "hpu", None)
    is_available = getattr(hpu, "is_available", None)
    synchronize = getattr(hpu, "synchronize", None)
    if callable(is_available) and callable(synchronize) and bool(is_available()):
        synchronize()


class ActionKVPhaseTimer:
    """Collect JSON-safe per-phase latencies with injectable clock and sync."""

    def __init__(
        self,
        *,
        accelerator_sync: Callable[[], None] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._accelerator_sync = (
            accelerator_sync
            if accelerator_sync is not None
            else _synchronize_default_accelerator
        )
        self._clock = (
            monotonic_clock if monotonic_clock is not None else time.perf_counter
        )
        self._samples: dict[str, list[float]] = {
            phase: [] for phase in ACTION_KV_TIMING_PHASES
        }

    def measure(
        self,
        phase: str,
        operation: Callable[..., _T],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Run one operation and record synchronized elapsed milliseconds."""

        if phase not in self._samples:
            raise ValueError(f"unknown Action KV timing phase {phase!r}")
        self._accelerator_sync()
        started = self._clock()
        try:
            return operation(*args, **kwargs)
        finally:
            self._accelerator_sync()
            elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
            self._samples[phase].append(float(elapsed_ms))

    @staticmethod
    def _summary(samples: list[float]) -> dict[str, int | float | None]:
        if not samples:
            return {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "max": None,
            }
        values = np.asarray(samples, dtype=np.float64)
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }

    def receipt(self) -> dict[str, object]:
        """Return immutable-by-copy samples and aggregate statistics."""

        samples = {phase: list(values) for phase, values in self._samples.items()}
        return {
            "unit": "ms",
            "samples": samples,
            "summary": {
                phase: self._summary(values) for phase, values in samples.items()
            },
        }
