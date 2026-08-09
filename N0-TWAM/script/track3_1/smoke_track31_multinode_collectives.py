#!/usr/bin/env python3
"""Produce strong schema-v2 NCCL evidence for a formal Stage A topology."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.track3_1.track31_collective_smoke_contract import (  # noqa: E402
    HCU_PER_NODE,
    SmokeContract,
    broadcast_sentinel,
    build_report,
    parse_assignments,
    parse_csv,
)
from script.track3_1.track31_collective_smoke_io import (  # noqa: E402
    ensure_report_target_available,
    publish_report_atomic,
)

_NCCL_PREFIXES = ("NCCL_", "TORCH_NCCL_")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--smoke-id", required=True)
    parser.add_argument(
        "--expected-nodes",
        required=True,
        help="Exact ordered comma-separated node-address roster.",
    )
    parser.add_argument(
        "--expected-nccl-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Optional exact NCCL/TORCH_NCCL environment; repeat once per key. "
            "When omitted, the complete live NCCL environment is still recorded."
        ),
    )
    parser.add_argument("--backend", choices=("nccl",), default="nccl")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args(argv)


def _environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value or any(char.isspace() for char in value):
        raise ValueError(f"environment is missing a safe non-empty {name}")
    return value


def _environment_integer(name: str, *, minimum: int = 0) -> int:
    raw_value = _environment(name)
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"environment has invalid {name}: {raw_value!r}") from error
    if value < minimum:
        raise ValueError(f"environment has {name} below {minimum}: {value}")
    return value


def _capture_nccl_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.startswith(_NCCL_PREFIXES)
    }
    if not environment:
        raise ValueError("live NCCL/TORCH_NCCL environment is empty")
    return dict(sorted(environment.items()))


def _contract_from_environment(args: argparse.Namespace) -> SmokeContract:
    nodes = parse_csv(str(args.expected_nodes), "--expected-nodes")
    hcu_order = parse_csv(_environment("HIP_VISIBLE_DEVICES"), "HIP_VISIBLE_DEVICES")
    live_nccl_environment = _capture_nccl_environment()
    requested_environment = parse_assignments(
        list(args.expected_nccl_env),
        "--expected-nccl-env",
    )
    if requested_environment and requested_environment != live_nccl_environment:
        raise ValueError(
            "live NCCL environment does not exactly match --expected-nccl-env"
        )
    contract = SmokeContract(
        smoke_id=str(args.smoke_id),
        nodes=nodes,
        hcu_order=hcu_order,
        backend=str(args.backend),
        nccl_environment=live_nccl_environment,
        image_id=_environment("N0_TRACK31_IMAGE_ID"),
        source_manifest_sha256=_environment("N0_TRACK31_SOURCE_MANIFEST_SHA256"),
        overlay_manifest_sha256=_environment("N0_TRACK31_OVERLAY_MANIFEST_SHA256"),
        run_role=_environment("N0_TRACK31_RUN_ROLE"),
        batch_size=_environment_integer("N0_TRACK31_BATCH_SIZE", minimum=1),
        gradient_accumulation_steps=_environment_integer(
            "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS",
            minimum=1,
        ),
    )
    if _environment("N0_TRACK31_INVOCATION_ID") != contract.smoke_id:
        raise ValueError("--smoke-id must equal N0_TRACK31_INVOCATION_ID")
    return contract


def _validate_torchrun_environment(contract: SmokeContract) -> tuple[int, int, int]:
    rank = _environment_integer("RANK")
    local_rank = _environment_integer("LOCAL_RANK")
    world_size = _environment_integer("WORLD_SIZE", minimum=1)
    node_rank = _environment_integer("NODE_RANK")
    if world_size != contract.world_size:
        raise ValueError("WORLD_SIZE does not match the exact expected node roster")
    if rank >= world_size:
        raise ValueError("RANK is outside WORLD_SIZE")
    expected_node_rank, expected_local_rank = divmod(rank, HCU_PER_NODE)
    if local_rank != expected_local_rank or node_rank != expected_node_rank:
        raise ValueError("RANK/LOCAL_RANK/NODE_RANK do not form the canonical roster")
    if _environment("N0_TRACK31_NODE_ADDRESS") != contract.nodes[node_rank]:
        raise ValueError("N0_TRACK31_NODE_ADDRESS does not match --expected-nodes")
    _environment("N0_TRACK31_CONTAINER_ID")
    optional_integer_contracts = {
        "LOCAL_WORLD_SIZE": HCU_PER_NODE,
        "GROUP_RANK": node_rank,
        "GROUP_WORLD_SIZE": len(contract.nodes),
    }
    for name, expected in optional_integer_contracts.items():
        if name in os.environ and _environment_integer(name) != expected:
            raise ValueError(f"{name} does not match the canonical topology")
    return rank, local_rank, node_rank


def _initialize_process_group(
    contract: SmokeContract,
    *,
    timeout_seconds: int,
) -> tuple[int, int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("formal NCCL smoke requires the vendor CUDA-compatible API")
    rank, local_rank, node_rank = _validate_torchrun_environment(contract)
    if torch.cuda.device_count() != HCU_PER_NODE:
        raise RuntimeError("formal collective smoke requires exactly 8 visible HCUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend=contract.backend,
        init_method="env://",
        rank=rank,
        world_size=contract.world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    if str(dist.get_backend()) != contract.backend:
        raise RuntimeError("initialized distributed backend is not exact NCCL")
    return rank, local_rank, node_rank, device


def _broadcast_rank_zero_result(
    message: Mapping[str, object] | None,
    *,
    rank: int,
    device: torch.device,
) -> Mapping[str, object]:
    messages: list[object] = [dict(message) if rank == 0 and message else None]
    dist.broadcast_object_list(messages, src=0, device=device)
    received = messages[0]
    if not isinstance(received, Mapping):
        raise RuntimeError("rank-zero control broadcast returned an invalid message")
    return received


def _preflight_report_target(
    path: Path,
    *,
    rank: int,
    device: torch.device,
) -> None:
    message: dict[str, object] | None = None
    if rank == 0:
        try:
            ensure_report_target_available(path)
            message = {"ok": True}
        except Exception as error:
            message = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    received = _broadcast_rank_zero_result(message, rank=rank, device=device)
    if received.get("ok") is not True:
        raise RuntimeError(f"rank-zero report target preflight failed: {received!r}")


def _run_collectives(
    contract: SmokeContract,
    *,
    rank: int,
    device: torch.device,
) -> tuple[int, int, int, int, str, str]:
    all_reduce_input = rank + 1
    all_reduce_expected = contract.world_size * (contract.world_size + 1) // 2
    reduced = torch.tensor(all_reduce_input, dtype=torch.int64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    all_reduce_actual = int(reduced.item())

    broadcast_expected = broadcast_sentinel(contract.smoke_id)
    broadcast_value = broadcast_expected if rank == 0 else -1
    broadcast_tensor = torch.tensor(broadcast_value, dtype=torch.int64, device=device)
    dist.broadcast(broadcast_tensor, src=0)
    broadcast_actual = int(broadcast_tensor.item())
    torch.cuda.synchronize(device)
    return (
        all_reduce_input,
        all_reduce_expected,
        all_reduce_actual,
        broadcast_actual,
        str(reduced.dtype),
        str(broadcast_tensor.dtype),
    )


def _local_observation(
    contract: SmokeContract,
    *,
    rank: int,
    local_rank: int,
    node_rank: int,
    device: torch.device,
    collective_values: tuple[int, int, int, int, str, str],
) -> dict[str, object]:
    (
        all_reduce_input,
        all_reduce_expected,
        all_reduce_actual,
        broadcast_actual,
        all_reduce_dtype,
        broadcast_dtype,
    ) = collective_values
    properties = torch.cuda.get_device_properties(device)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "node_rank": node_rank,
        "node_address": _environment("N0_TRACK31_NODE_ADDRESS"),
        "hostname": socket.gethostname(),
        "container_id": _environment("N0_TRACK31_CONTAINER_ID"),
        "backend": str(dist.get_backend()),
        "hcu_order": list(contract.hcu_order),
        "nccl_environment": dict(contract.nccl_environment),
        "bindings": contract.rank_binding(),
        "device": {
            "type": "cuda",
            "local_index": local_rank,
            "physical_id": contract.hcu_order[local_rank],
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
        },
        "collectives": {
            "all_reduce": {
                "dtype": all_reduce_dtype,
                "input": all_reduce_input,
                "expected": all_reduce_expected,
                "actual": all_reduce_actual,
            },
            "broadcast": {
                "dtype": broadcast_dtype,
                "source_rank": 0,
                "expected": broadcast_sentinel(contract.smoke_id),
                "actual": broadcast_actual,
            },
        },
    }


def _gather_rank_observations(
    observation: Mapping[str, object],
    *,
    world_size: int,
) -> list[Mapping[str, object]]:
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, dict(observation))
    observations: list[Mapping[str, object]] = []
    for index, item in enumerate(gathered):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"rank {index} returned a malformed observation")
        observations.append(item)
    return observations


def _publish_rank_zero_report(
    path: Path,
    contract: SmokeContract,
    observations: Sequence[Mapping[str, object]],
    *,
    rank: int,
    device: torch.device,
) -> str:
    message: dict[str, object] | None = None
    if rank == 0:
        try:
            report = build_report(
                contract,
                observations,
                created_at_unix_ns=time.time_ns(),
            )
            digest = publish_report_atomic(path, report)
            message = {"ok": True, "report_sha256": digest}
        except Exception as error:
            message = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    received = _broadcast_rank_zero_result(message, rank=rank, device=device)
    if received.get("ok") is not True or not isinstance(
        received.get("report_sha256"), str
    ):
        raise RuntimeError(f"rank-zero report publication failed: {received!r}")
    return str(received["report_sha256"])


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if isinstance(args.timeout_seconds, bool) or not 1 <= args.timeout_seconds <= 300:
        raise ValueError("--timeout-seconds must be an integer in [1, 300]")
    contract = _contract_from_environment(args)
    expected_report_name = f"collective_smoke.{contract.smoke_id}.json"
    if Path(args.report_path).name != expected_report_name:
        raise ValueError(
            f"--report-path must end with the unique name {expected_report_name}"
        )
    initialized = False
    rank = -1
    try:
        rank, local_rank, node_rank, device = _initialize_process_group(
            contract,
            timeout_seconds=int(args.timeout_seconds),
        )
        initialized = True
        _preflight_report_target(Path(args.report_path), rank=rank, device=device)
        collective_values = _run_collectives(contract, rank=rank, device=device)
        observation = _local_observation(
            contract,
            rank=rank,
            local_rank=local_rank,
            node_rank=node_rank,
            device=device,
            collective_values=collective_values,
        )
        observations = _gather_rank_observations(
            observation,
            world_size=contract.world_size,
        )
        digest = _publish_rank_zero_report(
            Path(args.report_path),
            contract,
            observations,
            rank=rank,
            device=device,
        )
        dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "smoke_id": contract.smoke_id,
                        "report_path": str(args.report_path),
                        "report_sha256": digest,
                    },
                    sort_keys=True,
                )
            )
        return 0
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
