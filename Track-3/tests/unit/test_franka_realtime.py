# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import pytest

from n0_twam.integrations.worldarena.franka_realtime import (
    require_franka_realtime_pass,
    summarize_franka_policy_latency,
)


def _timing(kind: str, infer_ms: float) -> dict[str, object]:
    return {
        "kind": kind,
        "infer_ms": infer_ms,
        "input_ms": 0.1,
        "grounding_ms": 0.2 if kind == "grounding_refill" else 0.0,
        "generation_ms": 0.3 if kind != "queue_hit" else 0.0,
        "postprocess_ms": 0.1,
        "grounded": kind == "grounding_refill",
        "generated": kind != "queue_hit",
        "queue_depth_after": 5,
    }


def test_realtime_assessment_passes_only_with_queue_and_refill_samples() -> None:
    assessment = summarize_franka_policy_latency(
        [
            _timing("cold_generation", 100.0),
            _timing("queue_hit", 1.0),
            _timing("queue_hit", 2.0),
            _timing("grounding_refill", 40.0),
            _timing("grounding_refill", 50.0),
        ],
        control_hz=15.0,
    )

    assert assessment["realtime_pass"] is True
    assert assessment["classes"]["queue_hit"]["p99_ms"] < 3.0
    assert assessment["classes"]["grounding_refill"]["p99_ms"] < 51.0
    require_franka_realtime_pass(assessment)


def test_realtime_assessment_fails_when_synchronous_refill_misses_period() -> None:
    assessment = summarize_franka_policy_latency(
        [
            _timing("cold_generation", 500.0),
            _timing("queue_hit", 1.0),
            _timing("grounding_refill", 80.0),
            _timing("grounding_refill", 90.0),
        ],
        control_hz=15.0,
    )

    assert assessment["realtime_pass"] is False
    assert assessment["failure_reasons"] == [
        "grounding_refill_p99_exceeds_control_deadline"
    ]
    with pytest.raises(RuntimeError, match="latency gate failed"):
        require_franka_realtime_pass(assessment)


def test_realtime_assessment_rejects_unknown_timing_kind() -> None:
    with pytest.raises(ValueError, match="unsupported Policy timing kind"):
        summarize_franka_policy_latency([_timing("prefetch", 1.0)])
