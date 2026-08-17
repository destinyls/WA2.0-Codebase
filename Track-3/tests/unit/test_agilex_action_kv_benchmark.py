# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Small receipt contracts for the AgileX Action-KV HCU canary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from n0_twam.integrations.worldarena.agilex_manifest import canonical_sha256
from script.track3_2.benchmark_agilex_action_kv import (
    _phase_summary,
    _summary,
    _validate_output,
)
from script.track3_2.action_kv_paired_parity import run_paired_action_kv_parity
from script.track3_2.compare_action_kv_benchmarks import compare


def test_latency_summary_is_deterministic() -> None:
    assert _summary([1.0, 2.0, 3.0, 4.0]) == {
        "count": 4,
        "mean": 2.5,
        "p50": 2.5,
        "p95": pytest.approx(3.85),
        "max": 4.0,
    }


def test_phase_summary_aggregates_request_receipts() -> None:
    receipts = [
        {"phase_latency_ms": {"samples": {"cached": [1.0, 2.0]}}},
        {"phase_latency_ms": {"samples": {"cached": [3.0], "terminal": [5.0]}}},
    ]

    summary = _phase_summary(receipts)

    assert summary["cached"] == {
        "count": 3,
        "mean": 2.0,
        "p50": 2.0,
        "p95": pytest.approx(2.9),
        "max": 3.0,
    }
    assert summary["terminal"]["count"] == 1
    assert summary["terminal"]["mean"] == 5.0


class _PairedBackend:
    def __init__(self, *, drift: bool) -> None:
        self._server = SimpleNamespace(
            job_config=SimpleNamespace(action_denoise_kv_reuse=True),
            _last_action_kv_reuse_receipt=None,
        )
        self._drift = drift

    def reset(self, **_: object) -> None:
        return None

    def infer_prediction_latent_chunk(
        self, **_: object
    ) -> tuple[np.ndarray, torch.Tensor]:
        enabled = bool(self._server.job_config.action_denoise_kv_reuse)
        value = 0.1 if enabled and self._drift else 0.0
        self._server._last_action_kv_reuse_receipt = {
            "requested": enabled,
            "used_fast_path": enabled,
            "cached_steps": 3 if enabled else 0,
        }
        return (
            np.full((12, 14), value, dtype=np.float32),
            torch.full((1, 2), value, dtype=torch.float32),
        )


@pytest.mark.parametrize(("drift", "passed"), ((False, True), (True, False)))
def test_paired_semantic_probe_detects_drift(drift: bool, passed: bool) -> None:
    backend = _PairedBackend(drift=drift)

    result = run_paired_action_kv_parity(
        backend=backend,
        job_config=backend._server.job_config,
        task_id="insert",
        prompt="insert",
        profile="mixed",
        seed=7,
        images={},
        qpos=np.zeros(14, dtype=np.float32),
        tactile={},
        wrench={},
        synchronize=lambda: None,
    )

    assert result["passed"] is passed
    assert backend._server.job_config.action_denoise_kv_reuse is True


def test_receipt_validator_rejects_identity_tampering() -> None:
    core = {"schema_version": 1, "status": "complete", "kv_reuse": True}
    payload = {**core, "benchmark_identity_sha256": canonical_sha256(core)}
    raw = json.dumps(payload).encode("utf-8")
    assert _validate_output(raw)["kv_reuse"] is True

    payload["kv_reuse"] = False
    with pytest.raises(ValueError, match="self hash mismatch"):
        _validate_output(json.dumps(payload).encode("utf-8"))


def _write_arm(path: Path, *, enabled: bool, p50: float) -> None:
    runtime_core = {
        "schema_version": 1,
        "profile": "mixed",
        "action_denoise_kv_reuse": enabled,
    }
    runtime_contract = {
        **runtime_core,
        "contract_sha256": canonical_sha256(runtime_core),
    }
    core = {
        "schema_version": 1,
        "status": "complete",
        "kv_reuse": enabled,
        "checkpoint_identity_sha256": "a" * 64,
        "train_request_file_sha256": "b" * 64,
        "policy_config_file_sha256": "c" * 64,
        "serve_output": str(path.parent / f"serve-{enabled}"),
        "evaluation_view_file_sha256": "d" * 64,
        "source_manifest_file_sha256": "f" * 64,
        "evaluation_view_sha256": "e" * 64,
        "runtime_environment": {"torch_version": "test"},
        "server_runtime_contract": runtime_contract,
        "server_runtime_contract_sha256": runtime_contract["contract_sha256"],
        "sample": {"repo_id": "touch", "episode_id": 0, "task_id": "insert"},
        "seed": 7,
        "warmup": 2,
        "iterations": 3,
        "total_latency_ms": {"mean": p50, "p50": p50, "p95": p50},
        "action_phase_latency_ms": {"cached": {"mean": 1.0}},
        "action_digests": ["same"] * 3,
        "action_outputs": [[[0.0] * 14 for _ in range(12)] for _ in range(3)],
        "latent_digests": ["latent"] * 3,
    }
    payload = {**core, "benchmark_identity_sha256": canonical_sha256(core)}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rewrite(path: Path, key: str, value: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = value
    core = {
        field: field_value
        for field, field_value in payload.items()
        if field != "benchmark_identity_sha256"
    }
    payload["benchmark_identity_sha256"] = canonical_sha256(core)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_requires_identical_arms_and_reports_speedup(tmp_path: Path) -> None:
    off = tmp_path / "off.json"
    on = tmp_path / "on.json"
    _write_arm(off, enabled=False, p50=10.0)
    _write_arm(on, enabled=True, p50=5.0)

    result = compare(off, on)

    p50 = result["total_latency_comparison"]["p50"]
    assert p50["speedup"] == 2.0
    assert p50["reduction_percent"] == 50.0
    assert result["off_serve_output"].endswith("serve-False")
    assert result["on_serve_output"].endswith("serve-True")


@pytest.mark.parametrize(
    ("field", "value"),
    (("seed", 8), ("runtime_environment", {"torch_version": "changed"})),
)
def test_compare_rejects_mismatched_execution_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    off = tmp_path / "off.json"
    on = tmp_path / "on.json"
    _write_arm(off, enabled=False, p50=10.0)
    _write_arm(on, enabled=True, p50=5.0)
    _rewrite(on, field, value)

    with pytest.raises(ValueError, match="identities differ"):
        compare(off, on)


def test_compare_rejects_action_semantic_drift(tmp_path: Path) -> None:
    off = tmp_path / "off.json"
    on = tmp_path / "on.json"
    _write_arm(off, enabled=False, p50=10.0)
    _write_arm(on, enabled=True, p50=5.0)
    drifted = [[[0.0] * 14 for _ in range(12)] for _ in range(3)]
    drifted[0][0][0] = 0.1
    _rewrite(on, "action_outputs", drifted)

    with pytest.raises(ValueError, match="paired semantic probe"):
        compare(off, on)


def _write_paired(path: Path, *, passed: bool) -> None:
    _write_arm(path, enabled=True, p50=5.0)
    probe = {
        "schema_version": 1,
        "status": "complete",
        "execution_tier": "same_process_same_seed_semantic_probe",
        "seed": 1_000_007,
        "passed": passed,
    }
    _rewrite(path, "paired_semantic_parity", probe)
    _rewrite(path, "source_manifest_file_sha256", "9" * 64)


def test_paired_probe_closes_stochastic_cross_process_comparison(
    tmp_path: Path,
) -> None:
    off = tmp_path / "off.json"
    on = tmp_path / "on.json"
    paired = tmp_path / "paired.json"
    _write_arm(off, enabled=False, p50=10.0)
    _write_arm(on, enabled=True, p50=5.0)
    _write_paired(paired, passed=True)
    drifted = [[[0.0] * 14 for _ in range(12)] for _ in range(3)]
    drifted[0][0][0] = 0.1
    _rewrite(on, "action_outputs", drifted)
    _rewrite(on, "latent_digests", ["different"] * 3)

    result = compare(off, on, paired)

    assert result["paired_semantic_parity"]["probe"]["passed"] is True
    assert result["separate_process_action_delta"]["passed"] is False
    assert result["separate_process_latent_digest_match_count"] == 0


def test_compare_rejects_failed_paired_probe(tmp_path: Path) -> None:
    off = tmp_path / "off.json"
    on = tmp_path / "on.json"
    paired = tmp_path / "paired.json"
    _write_arm(off, enabled=False, p50=10.0)
    _write_arm(on, enabled=True, p50=5.0)
    _write_paired(paired, passed=False)

    with pytest.raises(ValueError, match="did not pass"):
        compare(off, on, paired)
