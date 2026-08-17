# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Seal an OFF/ON comparison for AgileX Action-KV HCU canaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import BinaryIO, Mapping, cast

import numpy as np

from n0_twam.evaluation.franka_atomic_io import publish_atomic_file
from n0_twam.integrations.worldarena.agilex_manifest import (
    canonical_sha256,
    sha256_file,
)


def _load(path: Path, *, expected_enabled: bool) -> dict[str, object]:
    raw_source = path.expanduser()
    if raw_source.is_symlink() or not raw_source.is_file():
        raise ValueError("benchmark input must be a regular file")
    source = raw_source.resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError("benchmark input is incomplete")
    identity = payload.get("benchmark_identity_sha256")
    core = {
        key: value
        for key, value in payload.items()
        if key != "benchmark_identity_sha256"
    }
    if identity != canonical_sha256(core):
        raise ValueError("benchmark input self hash mismatch")
    if payload.get("kv_reuse") is not expected_enabled:
        raise ValueError("benchmark input is assigned to the wrong A/B arm")
    return cast(dict[str, object], payload)


def _latency(payload: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"benchmark {field} is invalid")
    return value


def _positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"benchmark {label} is invalid")
    result = float(value)
    if result <= 0:
        raise ValueError(f"benchmark {label} must be positive")
    return result


def _runtime_core(
    payload: Mapping[str, object], *, expected_enabled: bool
) -> dict[str, object]:
    raw_contract = payload.get("server_runtime_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("benchmark server runtime contract is invalid")
    contract = dict(raw_contract)
    identity = contract.pop("contract_sha256", None)
    if identity != canonical_sha256(contract):
        raise ValueError("benchmark server runtime contract self hash mismatch")
    if payload.get("server_runtime_contract_sha256") != identity:
        raise ValueError("benchmark server runtime contract identity mismatch")
    if contract.get("action_denoise_kv_reuse") is not expected_enabled:
        raise ValueError("benchmark server runtime contract has the wrong A/B flag")
    return contract


def _action_delta(
    off: Mapping[str, object], on: Mapping[str, object]
) -> dict[str, object]:
    off_actions = np.asarray(off.get("action_outputs"), dtype=np.float64)
    on_actions = np.asarray(on.get("action_outputs"), dtype=np.float64)
    iterations = off.get("iterations")
    expected = (iterations, 12, 14)
    if (
        type(iterations) is not int
        or off_actions.shape != expected
        or on_actions.shape != expected
        or not np.isfinite(off_actions).all()
        or not np.isfinite(on_actions).all()
    ):
        raise ValueError("OFF/ON action output payloads are invalid")
    absolute = np.abs(off_actions - on_actions)
    atol = 1e-4
    rtol = 1e-3
    passed = bool(np.allclose(off_actions, on_actions, atol=atol, rtol=rtol))
    result: dict[str, object] = {
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(absolute.max()),
        "mean_abs": float(absolute.mean()),
        "digest_match": off.get("action_digests") == on.get("action_digests"),
    }
    return result


def _paired_parity(
    path: Path,
    *,
    off: Mapping[str, object],
    on: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    paired = _load(path, expected_enabled=True)
    exact_fields = (
        "checkpoint_identity_sha256",
        "train_request_file_sha256",
        "policy_config_file_sha256",
        "evaluation_view_file_sha256",
        "evaluation_view_sha256",
        "runtime_environment",
        "sample",
        "seed",
    )
    if any(paired.get(field) != off.get(field) for field in exact_fields):
        raise ValueError("paired semantic probe identity differs from performance A/B")
    paired_runtime = _runtime_core(paired, expected_enabled=True)
    on_runtime = _runtime_core(on, expected_enabled=True)
    if paired_runtime != on_runtime:
        raise ValueError("paired semantic probe server runtime contract differs")
    raw_probe = paired.get("paired_semantic_parity")
    if not isinstance(raw_probe, Mapping):
        raise ValueError("paired semantic probe payload is missing")
    probe = dict(raw_probe)
    if (
        probe.get("status") != "complete"
        or probe.get("execution_tier") != "same_process_same_seed_semantic_probe"
        or probe.get("passed") is not True
    ):
        raise ValueError("paired semantic probe did not pass")
    binding = {
        "receipt_file_sha256": sha256_file(path),
        "source_manifest_file_sha256": paired["source_manifest_file_sha256"],
        "probe": probe,
    }
    return binding, paired_runtime


def compare(
    off_path: Path, on_path: Path, paired_parity_path: Path | None = None
) -> dict[str, object]:
    off = _load(off_path, expected_enabled=False)
    on = _load(on_path, expected_enabled=True)
    exact_fields = (
        "checkpoint_identity_sha256",
        "train_request_file_sha256",
        "policy_config_file_sha256",
        "evaluation_view_file_sha256",
        "source_manifest_file_sha256",
        "evaluation_view_sha256",
        "runtime_environment",
        "sample",
        "seed",
        "warmup",
        "iterations",
    )
    if any(off.get(field) != on.get(field) for field in exact_fields):
        raise ValueError("OFF/ON benchmark identities differ")
    off_runtime = _runtime_core(off, expected_enabled=False)
    on_runtime = _runtime_core(on, expected_enabled=True)
    off_runtime.pop("action_denoise_kv_reuse")
    on_runtime.pop("action_denoise_kv_reuse")
    if off_runtime != on_runtime:
        raise ValueError("OFF/ON server runtime contracts differ beyond the KV flag")
    action_delta = _action_delta(off, on)
    latent_digests_off = off.get("latent_digests")
    latent_digests_on = on.get("latent_digests")
    if not isinstance(latent_digests_off, list) or not isinstance(
        latent_digests_on, list
    ):
        raise ValueError("OFF/ON latent digest payloads are invalid")
    latent_digest_match_count = sum(
        left == right for left, right in zip(latent_digests_off, latent_digests_on)
    )
    paired_binding: dict[str, object] | None = None
    if paired_parity_path is None:
        if not bool(action_delta["passed"]) or latent_digests_off != latent_digests_on:
            raise ValueError(
                "separate-process outputs differ; a paired semantic probe is required"
            )
    else:
        paired_binding, _ = _paired_parity(
            paired_parity_path,
            off=off,
            on=on,
        )
    off_total = _latency(off, "total_latency_ms")
    on_total = _latency(on, "total_latency_ms")
    comparison: dict[str, object] = {}
    for percentile in ("mean", "p50", "p95"):
        baseline = _positive(off_total.get(percentile), label=f"OFF {percentile}")
        cached = _positive(on_total.get(percentile), label=f"ON {percentile}")
        comparison[percentile] = {
            "off_ms": baseline,
            "on_ms": cached,
            "speedup": baseline / cached,
            "reduction_percent": (baseline - cached) / baseline * 100.0,
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "execution_tier": "offline_hcu_action_kv_latency_ab",
        "real_robot_evaluation_completed": False,
        "organizer_evaluation_completed": False,
        "checkpoint_identity_sha256": off["checkpoint_identity_sha256"],
        "sample": off["sample"],
        "warmup": off["warmup"],
        "iterations": off["iterations"],
        "off_receipt_file_sha256": sha256_file(off_path),
        "on_receipt_file_sha256": sha256_file(on_path),
        "off_serve_output": off["serve_output"],
        "on_serve_output": on["serve_output"],
        "total_latency_comparison": comparison,
        "separate_process_action_delta": action_delta,
        "separate_process_latent_digest_match_count": latent_digest_match_count,
        "on_action_phase_latency_ms": on["action_phase_latency_ms"],
    }
    if paired_binding is not None:
        result["paired_semantic_parity"] = paired_binding
    return result


def _validate(raw: bytes) -> dict[str, object]:
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError("Action KV comparison is incomplete")
    identity = payload.get("comparison_identity_sha256")
    core = {
        key: value
        for key, value in payload.items()
        if key != "comparison_identity_sha256"
    }
    if identity != canonical_sha256(core):
        raise ValueError("Action KV comparison self hash mismatch")
    return cast(dict[str, object], payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--paired-parity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    core = compare(args.off, args.on, args.paired_parity)
    payload = {**core, "comparison_identity_sha256": canonical_sha256(core)}
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )

    def writer(handle: BinaryIO) -> None:
        handle.write(raw)

    publish_atomic_file(
        output=args.output,
        writer=writer,
        validator=_validate,
        label="AgileX Action KV benchmark comparison",
    )


if __name__ == "__main__":
    main()
