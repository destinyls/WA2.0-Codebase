#!/usr/bin/env python3
"""Strict schema and identity validation for Stage A collective reports."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn

from script.track3_1.track31_collective_smoke_contract import (
    HCU_PER_NODE,
    REPORT_STATUS,
    SCHEMA_VERSION,
    SmokeContract,
    broadcast_sentinel,
    format_timestamp_utc,
    validate_rank_observations,
)
from script.track3_1.track31_collective_smoke_io import (
    CollectiveSmokeValidationError,
    read_immutable_report,
)

_FUTURE_CLOCK_SKEW_NS = 5_000_000_000
_MTIME_GENERATION_SKEW_NS = 5_000_000_000


def _fail(message: str) -> NoReturn:
    raise CollectiveSmokeValidationError(message)


def _plain_integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum} (bool is forbidden)")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} keys differ: missing={sorted(expected - actual)!r} "
            f"extra={sorted(actual - expected)!r}"
        )


def _validate_top_level_schema(report: Mapping[str, object]) -> None:
    _exact_keys(
        report,
        {
            "schema_version",
            "status",
            "smoke_id",
            "created_at_utc",
            "created_at_unix_ns",
            "backend",
            "world_size",
            "topology",
            "identities",
            "nccl_environment",
            "training_execution_contract",
            "collective_evidence",
            "ranks",
        },
        "report",
    )
    if _plain_integer(report.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        _fail("collective smoke report schema_version is not 2")
    if report.get("status") != REPORT_STATUS:
        _fail("collective smoke report status is not PASS")


def _validate_topology_and_identities(
    report: Mapping[str, object],
    contract: SmokeContract,
) -> None:
    if report.get("smoke_id") != contract.smoke_id:
        _fail("collective smoke report smoke_id does not match")
    if report.get("backend") != contract.backend:
        _fail("collective smoke report backend does not match")
    if _plain_integer(report.get("world_size"), "world_size", minimum=1) != (
        contract.world_size
    ):
        _fail("collective smoke report world_size does not match")
    topology = report.get("topology")
    if not isinstance(topology, Mapping):
        _fail("collective smoke report topology is missing")
    _exact_keys(
        topology,
        {"nodes", "node_count", "hcu_per_node", "world_size", "hcu_order"},
        "topology",
    )
    for label in ("node_count", "hcu_per_node", "world_size"):
        _plain_integer(topology.get(label), f"topology.{label}", minimum=1)
    expected_topology = {
        "nodes": list(contract.nodes),
        "node_count": len(contract.nodes),
        "hcu_per_node": HCU_PER_NODE,
        "world_size": contract.world_size,
        "hcu_order": list(contract.hcu_order),
    }
    if topology != expected_topology:
        _fail("collective smoke report topology identities do not match")
    identities = report.get("identities")
    if not isinstance(identities, Mapping):
        _fail("collective smoke report identities are missing")
    _exact_keys(
        identities,
        {
            "image_id",
            "source_manifest_sha256",
            "overlay_manifest_sha256",
            "run_role",
        },
        "identities",
    )
    expected_identities = {
        "image_id": contract.image_id,
        "source_manifest_sha256": contract.source_manifest_sha256,
        "overlay_manifest_sha256": contract.overlay_manifest_sha256,
        "run_role": contract.run_role,
    }
    if identities != expected_identities:
        _fail("collective smoke report exact identities do not match")
    if report.get("nccl_environment") != dict(contract.nccl_environment):
        _fail("collective smoke report nccl_environment does not match")
    expected_execution = {
        "batch_size": contract.batch_size,
        "gradient_accumulation_steps": contract.gradient_accumulation_steps,
    }
    execution = report.get("training_execution_contract")
    if not isinstance(execution, Mapping):
        _fail("collective smoke report training_execution_contract is malformed")
    _exact_keys(
        execution,
        {"batch_size", "gradient_accumulation_steps"},
        "training_execution_contract",
    )
    _plain_integer(execution.get("batch_size"), "batch_size", minimum=1)
    _plain_integer(
        execution.get("gradient_accumulation_steps"),
        "gradient_accumulation_steps",
        minimum=1,
    )
    if execution != expected_execution:
        _fail("collective smoke report training_execution_contract does not match")


def _validate_rank_schema(
    ranks: object,
    contract: SmokeContract,
) -> list[Mapping[str, object]]:
    if not isinstance(ranks, list):
        _fail("collective smoke report ranks must be a list")
    normalized: list[Mapping[str, object]] = []
    rank_keys = {
        "rank",
        "local_rank",
        "node_rank",
        "node_address",
        "hostname",
        "container_id",
        "backend",
        "hcu_order",
        "nccl_environment",
        "bindings",
        "device",
        "collectives",
    }
    for index, rank in enumerate(ranks):
        if not isinstance(rank, Mapping):
            _fail(f"rank record {index} is not an object")
        _exact_keys(rank, rank_keys, f"rank record {index}")
        collectives = rank.get("collectives")
        device = rank.get("device")
        if not isinstance(collectives, Mapping) or not isinstance(device, Mapping):
            _fail(f"rank record {index} has malformed nested evidence")
        _exact_keys(collectives, {"all_reduce", "broadcast"}, "rank collectives")
        all_reduce = collectives.get("all_reduce")
        broadcast = collectives.get("broadcast")
        if not isinstance(all_reduce, Mapping) or not isinstance(broadcast, Mapping):
            _fail(f"rank record {index} has malformed collective values")
        _exact_keys(
            all_reduce, {"dtype", "input", "expected", "actual"}, "rank all_reduce"
        )
        _exact_keys(
            broadcast,
            {"dtype", "source_rank", "expected", "actual"},
            "rank broadcast",
        )
        normalized.append(rank)
    try:
        validate_rank_observations(contract, normalized)
    except ValueError as error:
        raise CollectiveSmokeValidationError(
            f"collective smoke rank roster is invalid: {error}"
        ) from error
    return normalized


def _validate_collective_summary(
    evidence: object,
    ranks: Sequence[Mapping[str, object]],
    contract: SmokeContract,
) -> None:
    if not isinstance(evidence, Mapping):
        _fail("collective_evidence is missing")
    _exact_keys(evidence, {"all_reduce", "broadcast"}, "collective_evidence")
    all_reduce = evidence.get("all_reduce")
    broadcast = evidence.get("broadcast")
    if not isinstance(all_reduce, Mapping) or not isinstance(broadcast, Mapping):
        _fail("collective_evidence entries are malformed")
    _exact_keys(
        all_reduce,
        {"dtype", "operation", "input_formula", "expected", "actual_by_rank", "passed"},
        "collective all_reduce",
    )
    _exact_keys(
        broadcast,
        {
            "source_rank",
            "dtype",
            "value_derivation",
            "expected",
            "actual_by_rank",
            "passed",
        },
        "collective broadcast",
    )
    expected_sum = contract.world_size * (contract.world_size + 1) // 2
    sentinel = broadcast_sentinel(contract.smoke_id)
    expected_all_reduce = {
        "dtype": "torch.int64",
        "operation": "SUM",
        "input_formula": "rank+1",
        "expected": expected_sum,
        "actual_by_rank": [expected_sum] * contract.world_size,
        "passed": True,
    }
    expected_broadcast = {
        "dtype": "torch.int64",
        "source_rank": 0,
        "value_derivation": "sha256(smoke_id)[:15]+1",
        "expected": sentinel,
        "actual_by_rank": [sentinel] * contract.world_size,
        "passed": True,
    }
    if all_reduce.get("passed") is not True or broadcast.get("passed") is not True:
        _fail("collective passed flags must be literal booleans")
    for label, value in (
        ("all_reduce.expected", all_reduce.get("expected")),
        ("broadcast.source_rank", broadcast.get("source_rank")),
        ("broadcast.expected", broadcast.get("expected")),
    ):
        _plain_integer(value, label)
    for label, values in (
        ("all_reduce.actual_by_rank", all_reduce.get("actual_by_rank")),
        ("broadcast.actual_by_rank", broadcast.get("actual_by_rank")),
    ):
        if not isinstance(values, list) or len(values) != contract.world_size:
            _fail(f"{label} must be an exact world-size list")
        for index, value in enumerate(values):
            _plain_integer(value, f"{label}[{index}]")
    if all_reduce != expected_all_reduce:
        _fail("collective all_reduce expected/actual evidence does not match")
    if broadcast != expected_broadcast:
        _fail("collective broadcast expected/actual evidence does not match")
    if len(ranks) != contract.world_size:
        _fail("collective evidence does not cover the exact rank roster")


def _validate_freshness(
    report: Mapping[str, object],
    metadata: os.stat_result,
    *,
    max_age_seconds: int,
    now_ns: int,
) -> None:
    _plain_integer(max_age_seconds, "max_age_seconds", minimum=1)
    _plain_integer(now_ns, "now_ns", minimum=1)
    created_at_ns = _plain_integer(
        report.get("created_at_unix_ns"),
        "created_at_unix_ns",
        minimum=1,
    )
    if report.get("created_at_utc") != format_timestamp_utc(created_at_ns):
        _fail("created_at_utc does not exactly encode created_at_unix_ns")
    age_ns = now_ns - created_at_ns
    if age_ns < -_FUTURE_CLOCK_SKEW_NS:
        _fail("collective smoke report is implausibly future-dated")
    if age_ns > max_age_seconds * 1_000_000_000:
        _fail("collective smoke report is stale")
    if abs(metadata.st_mtime_ns - created_at_ns) > _MTIME_GENERATION_SKEW_NS:
        _fail("collective smoke report mtime does not match report generation")


def _validate_expected_node_identities(
    ranks: Sequence[Mapping[str, object]],
    contract: SmokeContract,
    *,
    expected_container_ids: Mapping[str, str] | None,
    expected_hostnames: Mapping[str, str] | None,
) -> None:
    for label, expected, field in (
        ("expected_container_ids", expected_container_ids, "container_id"),
        ("expected_hostnames", expected_hostnames, "hostname"),
    ):
        if expected is None:
            continue
        unknown = set(expected) - set(contract.nodes)
        if unknown:
            _fail(f"{label} contains unknown nodes: {sorted(unknown)!r}")
        for node, expected_value in expected.items():
            if not isinstance(expected_value, str) or not expected_value:
                _fail(f"{label} contains an invalid value for {node}")
            node_rank = contract.nodes.index(node)
            start = node_rank * HCU_PER_NODE
            for rank in ranks[start : start + HCU_PER_NODE]:
                if rank.get(field) != expected_value:
                    _fail(f"rank {rank.get('rank')} {field} does not match {node}")


def validate_collective_smoke_report(
    report_path: Path,
    *,
    expected_sha256: str,
    smoke_id: str,
    expected_nodes: Sequence[str],
    expected_hcu_order: Sequence[str],
    expected_nccl_environment: Mapping[str, str],
    image_id: str,
    source_manifest_sha256: str,
    overlay_manifest_sha256: str,
    run_role: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    expected_container_ids: Mapping[str, str] | None = None,
    expected_hostnames: Mapping[str, str] | None = None,
    expected_world_size: int | None = None,
    expected_node_count: int | None = None,
    expected_hcu_per_node: int = HCU_PER_NODE,
    max_age_seconds: int = 300,
    now_ns: int | None = None,
) -> dict[str, object]:
    """Validate file safety, exact identities, freshness, ranks, and collectives."""
    contract = SmokeContract(
        smoke_id=smoke_id,
        nodes=tuple(expected_nodes),
        hcu_order=tuple(expected_hcu_order),
        backend="nccl",
        nccl_environment=expected_nccl_environment,
        image_id=image_id,
        source_manifest_sha256=source_manifest_sha256,
        overlay_manifest_sha256=overlay_manifest_sha256,
        run_role=run_role,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    for value, expected, label in (
        (expected_world_size, contract.world_size, "expected_world_size"),
        (expected_node_count, len(contract.nodes), "expected_node_count"),
        (expected_hcu_per_node, HCU_PER_NODE, "expected_hcu_per_node"),
    ):
        if value is not None and _plain_integer(value, label, minimum=1) != expected:
            _fail(f"{label} conflicts with exact node/HCU roster")
    report, raw_payload, metadata = read_immutable_report(report_path)
    if report_path.name != f"collective_smoke.{contract.smoke_id}.json":
        _fail("collective smoke report filename does not bind smoke_id")
    actual_sha256 = hashlib.sha256(raw_payload).hexdigest()
    if actual_sha256 != expected_sha256:
        _fail("collective smoke report SHA256 does not match expected SHA256")
    _validate_top_level_schema(report)
    _validate_topology_and_identities(report, contract)
    ranks = _validate_rank_schema(report.get("ranks"), contract)
    _validate_collective_summary(report.get("collective_evidence"), ranks, contract)
    _validate_expected_node_identities(
        ranks,
        contract,
        expected_container_ids=expected_container_ids,
        expected_hostnames=expected_hostnames,
    )
    _validate_freshness(
        report,
        metadata,
        max_age_seconds=max_age_seconds,
        now_ns=time.time_ns() if now_ns is None else now_ns,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "smoke_id": contract.smoke_id,
        "backend": contract.backend,
        "world_size": contract.world_size,
        "report_sha256": actual_sha256,
    }
