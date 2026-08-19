# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Deterministic latency accounting for the Track 3.2 Franka Policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

REALTIME_ASSESSMENT_SCHEMA_VERSION: Final[int] = 1
OFFICIAL_FRANKA_CONTROL_HZ: Final[float] = 15.0
_TIMING_KINDS: Final[tuple[str, ...]] = (
    "cold_generation",
    "queue_hit",
    "grounding_refill",
)


def _positive_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _timing_value(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Policy timing {key} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"Policy timing {key} must be finite and non-negative")
    return result


def _percentile_summary(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("latency summary requires finite non-empty samples")
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
    }


def summarize_franka_policy_latency(
    timings: Sequence[Mapping[str, object]],
    *,
    control_hz: float = OFFICIAL_FRANKA_CONTROL_HZ,
    minimum_refill_samples: int = 2,
) -> dict[str, object]:
    """Summarize cold, queue-hit, and synchronous refill latency.

    A refill is currently executed inside ``Policy.infer``.  It therefore has to
    finish within one control period; the 12-action prediction horizon is not a
    valid deadline unless generation is actually overlapped with action execution.
    """

    rate_hz = _positive_number(control_hz, label="control_hz")
    if (
        isinstance(minimum_refill_samples, bool)
        or not isinstance(minimum_refill_samples, int)
        or minimum_refill_samples < 1
    ):
        raise ValueError("minimum_refill_samples must be a positive integer")
    if not timings:
        raise ValueError("latency assessment requires at least one Policy timing")

    grouped: dict[str, list[float]] = {kind: [] for kind in _TIMING_KINDS}
    all_latencies: list[float] = []
    phase_totals = {
        "input_ms": [],
        "grounding_ms": [],
        "generation_ms": [],
        "postprocess_ms": [],
    }
    for row in timings:
        kind = row.get("kind")
        if kind not in grouped:
            raise ValueError(f"unsupported Policy timing kind: {kind!r}")
        infer_ms = _timing_value(row, "infer_ms")
        grouped[str(kind)].append(infer_ms)
        all_latencies.append(infer_ms)
        for phase in phase_totals:
            phase_totals[phase].append(_timing_value(row, phase))

    deadline_ms = 1000.0 / rate_hz
    classes: dict[str, object] = {}
    for kind, values in grouped.items():
        if not values:
            classes[kind] = {"count": 0}
            continue
        summary = _percentile_summary(values)
        summary["control_deadline_ms"] = deadline_ms
        summary["p99_meets_control_deadline"] = bool(
            float(summary["p99_ms"]) <= deadline_ms
        )
        classes[kind] = summary

    queue = classes["queue_hit"]
    refill = classes["grounding_refill"]
    queue_count = int(queue.get("count", 0)) if isinstance(queue, Mapping) else 0
    refill_count = int(refill.get("count", 0)) if isinstance(refill, Mapping) else 0
    enough_samples = queue_count >= 1 and refill_count >= minimum_refill_samples
    realtime_pass = bool(
        enough_samples
        and isinstance(queue, Mapping)
        and isinstance(refill, Mapping)
        and queue.get("p99_meets_control_deadline") is True
        and refill.get("p99_meets_control_deadline") is True
    )
    reasons: list[str] = []
    if queue_count < 1:
        reasons.append("missing_queue_hit_sample")
    if refill_count < minimum_refill_samples:
        reasons.append("insufficient_grounding_refill_samples")
    if isinstance(queue, Mapping) and queue.get("p99_meets_control_deadline") is False:
        reasons.append("queue_hit_p99_exceeds_control_deadline")
    if (
        isinstance(refill, Mapping)
        and refill.get("p99_meets_control_deadline") is False
    ):
        reasons.append("grounding_refill_p99_exceeds_control_deadline")

    return {
        "schema_version": REALTIME_ASSESSMENT_SCHEMA_VERSION,
        "control_hz": rate_hz,
        "control_period_ms": deadline_ms,
        "minimum_refill_samples": minimum_refill_samples,
        "sample_count": len(timings),
        "classes": classes,
        "overall": _percentile_summary(all_latencies),
        "phase_mean_ms": {
            phase: float(np.mean(values)) for phase, values in phase_totals.items()
        },
        "realtime_pass": realtime_pass,
        "failure_reasons": reasons,
    }


def require_franka_realtime_pass(assessment: Mapping[str, object]) -> None:
    """Fail closed unless a complete latency assessment meets the deadline."""

    if assessment.get("schema_version") != REALTIME_ASSESSMENT_SCHEMA_VERSION:
        raise ValueError("unsupported Franka realtime assessment schema")
    if assessment.get("realtime_pass") is not True:
        reasons = assessment.get("failure_reasons")
        raise RuntimeError(f"Franka realtime latency gate failed: {reasons}")


__all__ = (
    "OFFICIAL_FRANKA_CONTROL_HZ",
    "REALTIME_ASSESSMENT_SCHEMA_VERSION",
    "require_franka_realtime_pass",
    "summarize_franka_policy_latency",
)
