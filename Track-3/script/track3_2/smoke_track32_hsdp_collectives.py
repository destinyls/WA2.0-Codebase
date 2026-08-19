#!/usr/bin/env python3
"""Run a bounded multi-rank HSDP/NCCL collective smoke on vendor HCUs."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from n0_twam.distributed.fsdp import (
    _build_fsdp_device_mesh,
    _resolve_fsdp_topology,
    capture_fsdp_execution_contract,
)


def _positive_integer(name: str) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        raise ValueError(f"torchrun environment is missing {name}")
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--elements", type=int, default=8_388_608)
    return parser.parse_args()


def _write_new_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = _arguments()
    world_size = _positive_integer("WORLD_SIZE")
    local_world_size = _positive_integer("LOCAL_WORLD_SIZE")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_world_size != 8 or world_size % local_world_size:
        raise ValueError(
            "Track 3.2 HSDP smoke requires LOCAL_WORLD_SIZE=8 and a "
            "WORLD_SIZE divisible by 8"
        )
    replicate_size = world_size // local_world_size
    if replicate_size < 2:
        raise ValueError("Track 3.2 HSDP smoke requires at least two nodes")
    if torch.cuda.device_count() != local_world_size:
        raise ValueError("visible HCU count differs from LOCAL_WORLD_SIZE")
    if args.warmup < 1 or args.iterations < 1 or args.elements < 1:
        raise ValueError("smoke sizes must be positive")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=5),
    )
    try:
        contract = capture_fsdp_execution_contract(
            world_size=world_size,
            local_world_size=local_world_size,
        )
        expected_contract = {
            "schema_version": 1,
            "topology": "hsdp",
            "mesh_shape": [replicate_size, local_world_size],
            "replicate_size": replicate_size,
            "shard_size": local_world_size,
            "expert_reshard_policy": "after_layer",
            "reduce_dtype": "bfloat16",
        }
        if contract != expected_contract:
            raise ValueError(f"unexpected HSDP contract: {contract!r}")

        topology = _resolve_fsdp_topology(
            world_size=world_size,
            local_world_size=local_world_size,
        )
        device_mesh = _build_fsdp_device_mesh(topology)
        if device_mesh is None:
            raise ValueError("HSDP smoke did not construct a two-dimensional mesh")
        shard_value = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(shard_value, group=device_mesh.get_group("shard"))
        node_index = rank // local_world_size
        shard_expected = sum(
            node_index * local_world_size + offset + 1
            for offset in range(local_world_size)
        )
        if float(shard_value.item()) != float(shard_expected):
            raise ValueError("node-local HSDP shard collective is incorrect")
        replicate_value = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(replicate_value, group=device_mesh.get_group("replicate"))
        replicate_expected = sum(
            node_index * local_world_size + local_rank + 1
            for node_index in range(replicate_size)
        )
        if float(replicate_value.item()) != float(replicate_expected):
            raise ValueError("cross-node HSDP replica collective is incorrect")

        tensor = torch.empty(args.elements, dtype=torch.bfloat16, device=device)
        # Use an exactly representable BF16 operand for the bandwidth tensor.
        # Rank-sensitive membership is already verified above with FP32 shard
        # and replica collectives; summing rank+1 in BF16 can round differently
        # across valid RCCL reduction trees once the result exceeds 1024.
        expected_sum = world_size
        for _ in range(args.warmup):
            tensor.fill_(1.0)
            dist.all_reduce(tensor)
        torch.cuda.synchronize(device)
        dist.barrier()
        started = time.perf_counter()
        for _ in range(args.iterations):
            tensor.fill_(1.0)
            dist.all_reduce(tensor)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        observed_sum = float(tensor[0].item())
        if observed_sum != float(expected_sum):
            raise ValueError("all-reduce result is incorrect")

        elapsed_tensor = torch.tensor(elapsed, dtype=torch.float32, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        max_elapsed = float(elapsed_tensor.item())
        observations: list[object] = [None for _ in range(world_size)]
        dist.all_gather_object(
            observations,
            {
                "rank": rank,
                "local_rank": local_rank,
                "host": socket.gethostname(),
                "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
                "nccl_net_gdr_level": os.environ.get("NCCL_NET_GDR_LEVEL"),
                "nccl_dmabuf_enable": os.environ.get("NCCL_DMABUF_ENABLE"),
                "nccl_net_plugin": os.environ.get("NCCL_NET_PLUGIN"),
                "rccl_plugin_sha256": os.environ.get("N0_TRACK32_RCCL_PLUGIN_SHA256"),
            },
        )
        if rank == 0:
            payload_bytes = tensor.numel() * tensor.element_size()
            seconds_per_collective = max_elapsed / args.iterations
            algorithm_gbps = payload_bytes / seconds_per_collective / 1e9
            bus_gbps = algorithm_gbps * 2.0 * (world_size - 1) / world_size
            _write_new_report(
                args.report,
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "backend": "nccl",
                    "world_size": world_size,
                    "local_world_size": local_world_size,
                    "fsdp_execution_contract": contract,
                    "mesh_collectives": {
                        "shard_group_size": local_world_size,
                        "replicate_group_size": world_size // local_world_size,
                        "status": "PASS",
                    },
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "payload_bytes": payload_bytes,
                    "collective_input_value": 1.0,
                    "collective_expected_value": expected_sum,
                    "collective_observed_value": observed_sum,
                    "max_elapsed_seconds": max_elapsed,
                    "seconds_per_collective": seconds_per_collective,
                    "algorithm_bandwidth_gbps": algorithm_gbps,
                    "bus_bandwidth_gbps": bus_gbps,
                    "rank_observations": observations,
                },
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
