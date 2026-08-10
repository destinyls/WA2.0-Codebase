#!/usr/bin/env python3
"""Pure-stdlib schema and publication contract for Stage A collective evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType

SCHEMA_VERSION = 2
REPORT_STATUS = "PASS"
HCU_PER_NODE = 8
FORMAL_NODE_COUNTS = (2, 4, 6)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SMOKE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ENVIRONMENT_KEY_PATTERN = re.compile(r"^(?:NCCL|TORCH_NCCL)_[A-Z0-9_]+$")


def _plain_integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def parse_csv(value: str, label: str) -> tuple[str, ...]:
    """Parse a non-empty, duplicate-free comma-separated identity sequence."""
    items = tuple(item.strip() for item in value.split(","))
    if not items or any(
        not item or any(char.isspace() for char in item) for item in items
    ):
        raise ValueError(f"{label} contains an empty or whitespace-bearing item")
    if len(set(items)) != len(items):
        raise ValueError(f"{label} contains duplicate items")
    return items


def parse_assignments(values: Sequence[str], label: str) -> dict[str, str]:
    """Parse repeated NAME=VALUE arguments without silently replacing duplicates."""
    assignments: dict[str, str] = {}
    for raw_value in values:
        name, separator, value = raw_value.partition("=")
        if not separator or not name or not value:
            raise ValueError(f"{label} entries must use non-empty NAME=VALUE")
        if name in assignments:
            raise ValueError(f"{label} contains duplicate key {name!r}")
        assignments[name] = value
    return assignments


@dataclass(frozen=True)
class SmokeContract:
    """Immutable identity inputs that every participating rank must bind."""

    smoke_id: str
    nodes: tuple[str, ...]
    hcu_order: tuple[str, ...]
    backend: str
    nccl_environment: Mapping[str, str]
    image_id: str
    source_manifest_sha256: str
    overlay_manifest_sha256: str
    run_role: str
    batch_size: int
    gradient_accumulation_steps: int

    def __post_init__(self) -> None:
        if not _SMOKE_ID_PATTERN.fullmatch(self.smoke_id):
            raise ValueError("smoke_id is not a safe invocation identity")
        if len(self.nodes) not in FORMAL_NODE_COUNTS or len(set(self.nodes)) != len(
            self.nodes
        ):
            raise ValueError("nodes must contain exactly 2, 4, or 6 unique entries")
        if any(not node or any(char.isspace() for char in node) for node in self.nodes):
            raise ValueError("nodes contain an empty or whitespace-bearing identity")
        expected_hcus = {str(index) for index in range(HCU_PER_NODE)}
        if len(self.hcu_order) != HCU_PER_NODE or set(self.hcu_order) != expected_hcus:
            raise ValueError("hcu_order must be an exact permutation of HCU 0..7")
        if self.backend != "nccl":
            raise ValueError("formal collective smoke backend must be nccl")
        environment = dict(self.nccl_environment)
        if not environment:
            raise ValueError("nccl_environment must not be empty")
        for name, value in environment.items():
            if (
                not isinstance(name, str)
                or not _ENVIRONMENT_KEY_PATTERN.fullmatch(name)
                or not isinstance(value, str)
                or not value
            ):
                raise ValueError(f"invalid NCCL environment binding: {name!r}")
        object.__setattr__(
            self,
            "nccl_environment",
            MappingProxyType(dict(sorted(environment.items()))),
        )
        if not _IMAGE_ID_PATTERN.fullmatch(self.image_id):
            raise ValueError("image_id must be a full lowercase sha256 image ID")
        for label, digest in (
            ("source_manifest_sha256", self.source_manifest_sha256),
            ("overlay_manifest_sha256", self.overlay_manifest_sha256),
        ):
            if not _SHA256_PATTERN.fullmatch(digest):
                raise ValueError(f"{label} must be a lowercase SHA256")
        if self.run_role not in ("development", "final_refit"):
            raise ValueError("run_role must be development or final_refit")
        _plain_integer(self.batch_size, "batch_size", minimum=1)
        _plain_integer(
            self.gradient_accumulation_steps,
            "gradient_accumulation_steps",
            minimum=1,
        )

    @property
    def world_size(self) -> int:
        return len(self.nodes) * HCU_PER_NODE

    def rank_binding(self) -> dict[str, object]:
        """Return the canonical identity object duplicated into every rank record."""
        return {
            "smoke_id": self.smoke_id,
            "nodes": list(self.nodes),
            "hcu_order": list(self.hcu_order),
            "backend": self.backend,
            "nccl_environment": dict(self.nccl_environment),
            "image_id": self.image_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "overlay_manifest_sha256": self.overlay_manifest_sha256,
            "run_role": self.run_role,
            "training_execution_contract": {
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
            },
        }


def broadcast_sentinel(smoke_id: str) -> int:
    """Derive a deterministic positive int64 sentinel from the smoke identity."""
    digest = hashlib.sha256(smoke_id.encode("utf-8")).hexdigest()
    return int(digest[:15], 16) + 1


def format_timestamp_utc(timestamp_ns: int) -> str:
    _plain_integer(timestamp_ns, "created_at_unix_ns", minimum=1)
    timestamp = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc)
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_report_bytes(report: Mapping[str, object]) -> bytes:
    payload = json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (payload + "\n").encode("utf-8")


def canonical_report_sha256(report: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_report_bytes(report)).hexdigest()


def _validate_rank_observation(
    observation: Mapping[str, object],
    *,
    expected_rank: int,
    contract: SmokeContract,
) -> None:
    node_rank, local_rank = divmod(expected_rank, HCU_PER_NODE)
    if _plain_integer(observation.get("rank"), "rank") != expected_rank:
        raise ValueError(f"rank {expected_rank} has a non-canonical rank")
    if _plain_integer(observation.get("local_rank"), "local_rank") != local_rank:
        raise ValueError(f"rank {expected_rank} has an invalid local_rank")
    if _plain_integer(observation.get("node_rank"), "node_rank") != node_rank:
        raise ValueError(f"rank {expected_rank} has an invalid node_rank")
    if observation.get("node_address") != contract.nodes[node_rank]:
        raise ValueError(f"rank {expected_rank} has an invalid node_address")
    for label in ("hostname", "container_id"):
        value = observation.get(label)
        if (
            not isinstance(value, str)
            or not value
            or any(char.isspace() for char in value)
        ):
            raise ValueError(f"rank {expected_rank} has an invalid {label}")
    if observation.get("backend") != contract.backend:
        raise ValueError(f"rank {expected_rank} has an invalid backend")
    if observation.get("hcu_order") != list(contract.hcu_order):
        raise ValueError(f"rank {expected_rank} has an invalid hcu_order")
    if observation.get("nccl_environment") != dict(contract.nccl_environment):
        raise ValueError(f"rank {expected_rank} has an invalid nccl_environment")
    bindings = observation.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError(f"rank {expected_rank} has malformed bindings")
    execution = bindings.get("training_execution_contract")
    if not isinstance(execution, Mapping):
        raise ValueError(f"rank {expected_rank} has malformed bindings execution")
    _plain_integer(execution.get("batch_size"), "bindings.batch_size", minimum=1)
    _plain_integer(
        execution.get("gradient_accumulation_steps"),
        "bindings.gradient_accumulation_steps",
        minimum=1,
    )
    if bindings != contract.rank_binding():
        raise ValueError(f"rank {expected_rank} has invalid bindings")

    device = observation.get("device")
    if not isinstance(device, Mapping) or set(device) != {
        "type",
        "local_index",
        "physical_id",
        "name",
        "total_memory_bytes",
    }:
        raise ValueError(f"rank {expected_rank} has an invalid device record")
    if (
        device.get("type") != "cuda"
        or device.get("physical_id") != contract.hcu_order[local_rank]
    ):
        raise ValueError(f"rank {expected_rank} has an invalid device identity")
    if _plain_integer(device.get("local_index"), "device.local_index") != local_rank:
        raise ValueError(f"rank {expected_rank} has an invalid device local_index")
    if not isinstance(device.get("name"), str) or not device.get("name"):
        raise ValueError(f"rank {expected_rank} has an invalid device name")
    _plain_integer(device.get("total_memory_bytes"), "device memory", minimum=1)

    expected_sum = contract.world_size * (contract.world_size + 1) // 2
    expected_broadcast = broadcast_sentinel(contract.smoke_id)
    collectives = observation.get("collectives")
    if not isinstance(collectives, Mapping):
        raise ValueError(f"rank {expected_rank} has no collective evidence")
    all_reduce = collectives.get("all_reduce")
    broadcast = collectives.get("broadcast")
    if not isinstance(all_reduce, Mapping) or not isinstance(broadcast, Mapping):
        raise ValueError(f"rank {expected_rank} has malformed collective evidence")
    for label, value in (
        ("all_reduce.input", all_reduce.get("input")),
        ("all_reduce.expected", all_reduce.get("expected")),
        ("all_reduce.actual", all_reduce.get("actual")),
        ("broadcast.source_rank", broadcast.get("source_rank")),
        ("broadcast.expected", broadcast.get("expected")),
        ("broadcast.actual", broadcast.get("actual")),
    ):
        _plain_integer(value, label)
    if all_reduce != {
        "dtype": "torch.int64",
        "input": expected_rank + 1,
        "expected": expected_sum,
        "actual": expected_sum,
    }:
        raise ValueError(f"rank {expected_rank} has invalid all_reduce evidence")
    if broadcast != {
        "dtype": "torch.int64",
        "source_rank": 0,
        "expected": expected_broadcast,
        "actual": expected_broadcast,
    }:
        raise ValueError(f"rank {expected_rank} has invalid broadcast evidence")


def validate_rank_observations(
    contract: SmokeContract,
    observations: Sequence[Mapping[str, object]],
) -> None:
    """Validate exact global/local/node rank coverage and live node identities."""
    if len(observations) != contract.world_size:
        raise ValueError("rank roster does not contain the exact world_size")
    node_hostnames: dict[int, str] = {}
    node_containers: dict[int, str] = {}
    for expected_rank, observation in enumerate(observations):
        _validate_rank_observation(
            observation,
            expected_rank=expected_rank,
            contract=contract,
        )
        node_rank = expected_rank // HCU_PER_NODE
        hostname = str(observation["hostname"])
        container_id = str(observation["container_id"])
        previous_hostname = node_hostnames.setdefault(node_rank, hostname)
        previous_container = node_containers.setdefault(node_rank, container_id)
        if previous_hostname != hostname:
            raise ValueError(f"node_rank {node_rank} has inconsistent hostname")
        if previous_container != container_id:
            raise ValueError(f"node_rank {node_rank} has inconsistent container_id")
    if len(set(node_hostnames.values())) != len(contract.nodes):
        raise ValueError("hostname identities are not unique across nodes")
    if len(set(node_containers.values())) != len(contract.nodes):
        raise ValueError("container_id identities are not unique across nodes")


def build_report(
    contract: SmokeContract,
    observations: Sequence[Mapping[str, object]],
    *,
    created_at_unix_ns: int,
) -> dict[str, object]:
    """Build schema v2 only after rank zero validates the full gathered roster."""
    validate_rank_observations(contract, observations)
    created_at = format_timestamp_utc(created_at_unix_ns)
    expected_sum = contract.world_size * (contract.world_size + 1) // 2
    sentinel = broadcast_sentinel(contract.smoke_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": REPORT_STATUS,
        "smoke_id": contract.smoke_id,
        "created_at_utc": created_at,
        "created_at_unix_ns": created_at_unix_ns,
        "backend": contract.backend,
        "world_size": contract.world_size,
        "topology": {
            "nodes": list(contract.nodes),
            "node_count": len(contract.nodes),
            "hcu_per_node": HCU_PER_NODE,
            "world_size": contract.world_size,
            "hcu_order": list(contract.hcu_order),
        },
        "identities": {
            "image_id": contract.image_id,
            "source_manifest_sha256": contract.source_manifest_sha256,
            "overlay_manifest_sha256": contract.overlay_manifest_sha256,
            "run_role": contract.run_role,
        },
        "nccl_environment": dict(contract.nccl_environment),
        "training_execution_contract": {
            "batch_size": contract.batch_size,
            "gradient_accumulation_steps": contract.gradient_accumulation_steps,
        },
        "collective_evidence": {
            "all_reduce": {
                "dtype": "torch.int64",
                "operation": "SUM",
                "input_formula": "rank+1",
                "expected": expected_sum,
                "actual_by_rank": [expected_sum] * contract.world_size,
                "passed": True,
            },
            "broadcast": {
                "dtype": "torch.int64",
                "source_rank": 0,
                "value_derivation": "sha256(smoke_id)[:15]+1",
                "expected": sentinel,
                "actual_by_rank": [sentinel] * contract.world_size,
                "passed": True,
            },
        },
        "ranks": copy.deepcopy(list(observations)),
    }
