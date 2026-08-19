# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed multi-node orchestration helpers for Track 3.2 training."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from n0_twam.checkpointing.runtime_provenance import (
    LOCAL_CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION,
    LOCAL_EXECUTION_TIER,
    validate_checkpoint_invocation_identity,
)
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.integrations.worldarena.franka_actions import TRACK32_PROFILE_ID
from n0_twam.track31.local_provenance import (
    prepare_local_provenance,
    sha256_file,
    write_immutable_json,
)

from .request import Track32TrainRequest, load_track32_train_request
from .runner import (
    _artifact_identity,
    build_launch_plan,
    build_training_environment,
)

_NCCL_DEBUG_SUBSYSTEMS = "INIT,NET,GRAPH"
_RCCL_NET_PLUGIN = "shca"
_RCCL_PLUGIN_HOST_DIRECTORY = "/opt/hpc/software/app/rccl/shca_rdma_plugins/v8/lib"
_RCCL_PLUGIN_DIRECTORY = "/opt/n0_twam/rccl_plugin"
_RCCL_PLUGIN_FILENAME = "librccl-net-shca.so.0.0.0"
_RCCL_PLUGIN_SHA256 = "20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"
_IB_HCA_ROSTER_PATTERN = re.compile(
    r"[A-Za-z0-9_.-]+(?::[1-9][0-9]*)?(?:,[A-Za-z0-9_.-]+(?::[1-9][0-9]*)?)*"
)
_SENSITIVE_ENV_MARKERS = (
    "ACCESS_KEY",
    "API_KEY",
    "AUTH",
    "COOKIE",
    "CREDENTIAL",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)
_SENSITIVE_ENV_NAMES = frozenset(
    {"DBUS_SESSION_BUS_ADDRESS", "KRB5CCNAME", "SSH_AUTH_SOCK"}
)


def _positive(value: int, *, label: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    source = Path(path).resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _validated_ib_hca_roster(value: str) -> str:
    if not isinstance(value, str) or _IB_HCA_ROSTER_PATTERN.fullmatch(value) is None:
        raise ValueError("nccl_ib_hca must be a comma-separated HCA[:port] roster")
    if len(value.split(",")) != len(set(value.split(","))):
        raise ValueError("nccl_ib_hca may not contain duplicate HCA ports")
    return value


def _secret_free_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Drop credentials and user-session handles before persisting launch env."""

    return {
        name: value
        for name, value in source.items()
        if name.upper() not in _SENSITIVE_ENV_NAMES
        and not any(marker in name.upper() for marker in _SENSITIVE_ENV_MARKERS)
    }


def _multinode_execution_contract(
    *,
    node_count: int,
    local_world_size: int,
    nccl_ib_hca: str,
) -> tuple[dict[str, str], dict[str, object]]:
    node_count = _positive(node_count, label="node_count")
    local_world_size = _positive(local_world_size, label="local_world_size")
    if node_count < 2:
        raise ValueError("multinode HSDP requires at least two nodes")
    ib_hca = _validated_ib_hca_roster(nccl_ib_hca)
    environment = {
        "HSA_FORCE_FINE_GRAIN_PCIE": "1",
        "N0_FSDP_TOPOLOGY": "hsdp",
        "N0_FSDP_SHARD_SIZE": str(local_world_size),
        "N0_TRACK32_EXPECTED_IB_UVERBS": str(len(ib_hca.split(","))),
        "N0_TRACK32_RCCL_PLUGIN_DIR": _RCCL_PLUGIN_DIRECTORY,
        "N0_TRACK32_RCCL_PLUGIN_HOST_DIR": _RCCL_PLUGIN_HOST_DIRECTORY,
        "N0_TRACK32_RCCL_PLUGIN_FILENAME": _RCCL_PLUGIN_FILENAME,
        "N0_TRACK32_RCCL_PLUGIN_SHA256": _RCCL_PLUGIN_SHA256,
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": _NCCL_DEBUG_SUBSYSTEMS,
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": ib_hca,
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_IB_QPS_PER_CONNECTION": "4",
        "NCCL_IB_TC": "160",
        "NCCL_IB_TIMEOUT": "22",
        "NCCL_NET_PLUGIN": _RCCL_NET_PLUGIN,
        "NCCL_NET_GDR_LEVEL": "PHB",
    }
    contract: dict[str, object] = {
        "fsdp_topology": "hsdp",
        "mesh_shape": [node_count, local_world_size],
        "mesh_dim_names": ["replicate", "shard"],
        "replicate_size": node_count,
        "shard_size": local_world_size,
        "nccl_backend": "nccl",
        "nccl_debug": "INFO",
        "nccl_debug_subsystems": _NCCL_DEBUG_SUBSYSTEMS.split(","),
        "nccl_net_plugin": _RCCL_NET_PLUGIN,
        "nccl_ib_hca": ib_hca.split(","),
        "nccl_ib_gid_index": 3,
        "nccl_ib_qps_per_connection": 4,
        "nccl_ib_traffic_class": 160,
        "nccl_ib_timeout": 22,
        "nccl_net_gdr_level": "PHB",
        "nccl_dmabuf_enable": None,
        "rccl_plugin_container_directory": _RCCL_PLUGIN_DIRECTORY,
        "rccl_plugin_host_directory": _RCCL_PLUGIN_HOST_DIRECTORY,
        "rccl_plugin_filename": _RCCL_PLUGIN_FILENAME,
        "rccl_plugin_sha256": _RCCL_PLUGIN_SHA256,
        "required_hca_port_state": "ACTIVE/LinkUp",
        "required_hca_rate_gbps": 400,
        "required_rdma_devices": [
            "rdma_cm",
            *(f"uverbs{index}" for index in range(len(ib_hca.split(",")))),
        ],
    }
    return environment, contract


def _distributed_plan(
    request: Track32TrainRequest,
    *,
    nodes: tuple[str, ...],
    local_world_size: int,
    master_addr: str,
    master_port: int,
    execution_contract: dict[str, object],
) -> dict[str, object]:
    world_size = len(nodes) * local_world_size
    plan = build_launch_plan(request)
    plan.update(
        {
            "execution_tier": "local_package_multinode",
            "world_size": world_size,
            "nnodes": len(nodes),
            "local_world_size": local_world_size,
            "nodes": list(nodes),
            "master_addr": master_addr,
            "master_port": master_port,
            "distributed_execution_contract": execution_contract,
            "training_command": [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nnodes={len(nodes)}",
                f"--nproc-per-node={local_world_size}",
                "--node-rank=<node_rank>",
                f"--master-addr={master_addr}",
                f"--master-port={master_port}",
                "--tee",
                "3",
                "-m",
                "n0_twam.train",
                "--config-name",
                "track32_franka",
            ],
        }
    )
    return plan


def prepare_multinode_launch(
    *,
    request_path: Path,
    nodes: tuple[str, ...],
    local_world_size: int,
    master_addr: str,
    master_port: int,
    nccl_ib_hca: str,
    environment_path: Path,
    descriptor_path: Path,
) -> dict[str, object]:
    """Create one shared provenance identity and immutable launch environment."""

    if len(nodes) < 2 or len(nodes) != len(set(nodes)):
        raise ValueError("multinode launch requires at least two unique nodes")
    local_world_size = _positive(local_world_size, label="local_world_size")
    master_port = _positive(master_port, label="master_port")
    request = load_track32_train_request(request_path)
    if len(request.runtime.devices) != local_world_size:
        raise ValueError("request device count differs from local world size")
    if request.runtime.master_port != master_port:
        raise ValueError("request and multinode master ports differ")
    execution_environment, execution_contract = _multinode_execution_contract(
        node_count=len(nodes),
        local_world_size=local_world_size,
        nccl_ib_hca=nccl_ib_hca,
    )
    output_root = request.paths.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    plan = _distributed_plan(
        request,
        nodes=nodes,
        local_world_size=local_world_size,
        master_addr=master_addr,
        master_port=master_port,
        execution_contract=execution_contract,
    )
    provenance = prepare_local_provenance(
        output_root=output_root,
        run_id=request.run_id,
        request_sha256=sha256_file(request.source_path),
        launch_plan=plan,
    )
    environment = build_training_environment(
        request,
        provenance,
        environ=_secret_free_environment(os.environ),
    )
    existing_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        f"{_RCCL_PLUGIN_DIRECTORY}:{existing_library_path}"
        if existing_library_path
        else _RCCL_PLUGIN_DIRECTORY
    )
    world_size = len(nodes) * local_world_size
    environment.update(
        {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "NNODES": str(len(nodes)),
            "N0_TRACK32_EXPECTED_WORLD_SIZE": str(world_size),
            "N0_TRACK32_EXPECTED_NNODES": str(len(nodes)),
            "N0_TRACK32_EXPECTED_LOCAL_WORLD_SIZE": str(local_world_size),
            **execution_environment,
        }
    )
    environment_file_sha256 = write_immutable_json(environment_path, environment)
    descriptor: dict[str, object] = {
        "schema_version": 1,
        "status": "ready",
        "request": str(request.source_path),
        "request_sha256": sha256_file(request.source_path),
        "output_root": str(output_root),
        "invocation_id": provenance.invocation_id,
        "nodes": list(nodes),
        "nnodes": len(nodes),
        "local_world_size": local_world_size,
        "world_size": world_size,
        "master_addr": master_addr,
        "master_port": master_port,
        "distributed_execution_contract": execution_contract,
        "environment": str(environment_path),
        "environment_file_sha256": environment_file_sha256,
        "launch_receipt": str(provenance.launch_receipt_path),
        "launch_receipt_sha256": sha256_file(provenance.launch_receipt_path),
    }
    write_immutable_json(descriptor_path, descriptor)
    return descriptor


def _load_environment(path: Path, *, node_rank: int) -> dict[str, str]:
    payload = _load_json(path, label="multinode environment")
    environment: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("multinode environment must map strings to strings")
        environment[key] = value
    nnodes = int(environment["N0_TRACK32_EXPECTED_NNODES"])
    if node_rank < 0 or node_rank >= nnodes:
        raise ValueError("node_rank is outside the declared cluster")
    environment["NODE_RANK"] = str(node_rank)
    return environment


def run_child(*, environment_path: Path, node_rank: int, mode: str) -> int:
    """Run one node-local preflight or replace the process with torchrun."""

    environment = _load_environment(environment_path, node_rank=node_rank)
    if mode == "preflight":
        completed = subprocess.run(
            [sys.executable, "-m", "n0_twam.track32.preflight"],
            check=False,
            env=environment,
        )
        return completed.returncode
    if mode != "train":
        raise ValueError("mode must be preflight or train")
    nnodes = int(environment["N0_TRACK32_EXPECTED_NNODES"])
    local_world_size = int(environment["N0_TRACK32_EXPECTED_LOCAL_WORLD_SIZE"])
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc-per-node={local_world_size}",
        f"--node-rank={node_rank}",
        f"--master-addr={environment['MASTER_ADDR']}",
        f"--master-port={environment['MASTER_PORT']}",
        "--tee",
        "3",
        "-m",
        "n0_twam.train",
        "--config-name",
        "track32_franka",
    ]
    os.execvpe(sys.executable, command, environment)
    raise AssertionError("os.execvpe unexpectedly returned")


def run_collective_smoke(
    *,
    environment_path: Path,
    node_rank: int,
    master_port: int,
    report_path: Path,
) -> int:
    """Replace this process with one node of the bounded multi-rank smoke."""

    environment = _load_environment(environment_path, node_rank=node_rank)
    environment["MASTER_PORT"] = str(_positive(master_port, label="master_port"))
    nnodes = int(environment["N0_TRACK32_EXPECTED_NNODES"])
    local_world_size = int(environment["N0_TRACK32_EXPECTED_LOCAL_WORLD_SIZE"])
    script_path = (
        Path(__file__).resolve().parents[2]
        / "script"
        / "track3_2"
        / "smoke_track32_hsdp_collectives.py"
    ).resolve(strict=True)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--nproc-per-node={local_world_size}",
        f"--node-rank={node_rank}",
        f"--master-addr={environment['MASTER_ADDR']}",
        f"--master-port={environment['MASTER_PORT']}",
        "--tee",
        "3",
        str(script_path),
        "--report",
        str(report_path),
    ]
    os.execvpe(sys.executable, command, environment)
    raise AssertionError("os.execvpe unexpectedly returned")


def verify_multinode_completion(
    *, request_path: Path, descriptor_path: Path
) -> dict[str, object]:
    """Verify and publish the completed multi-node checkpoint receipt."""

    request = load_track32_train_request(request_path)
    descriptor = _load_json(descriptor_path, label="multinode descriptor")
    expected_world_size = int(descriptor["world_size"])
    checkpoint = (
        request.paths.output_root
        / "checkpoints"
        / f"checkpoint_step_{request.train.stop_after_step}"
    )
    snapshot = capture_strict_checkpoint_snapshot(checkpoint)
    if snapshot.step != request.train.stop_after_step:
        raise ValueError("completed checkpoint step differs from request")
    if snapshot.world_size != expected_world_size:
        raise ValueError("completed checkpoint world size differs from cluster")
    launch_receipt = _load_json(
        Path(str(descriptor["launch_receipt"])), label="launch receipt"
    )
    if snapshot.runtime_source_identity != launch_receipt["runtime_source_identity"]:
        raise ValueError("checkpoint source identity differs from launch receipt")
    expected_invocation = validate_checkpoint_invocation_identity(
        {
            "schema_version": LOCAL_CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION,
            "execution_tier": LOCAL_EXECUTION_TIER,
            "invocation_id": descriptor["invocation_id"],
            "launch_receipt_sha256": descriptor["launch_receipt_sha256"],
        }
    )
    if snapshot.checkpoint_invocation_identity != expected_invocation:
        raise ValueError("checkpoint invocation identity differs from launch")
    if snapshot.transformer_config.get("action_schema") != "ee20_absee":
        raise ValueError("completed checkpoint action schema is not ee20_absee")
    meta = snapshot.train_meta
    if meta.get("track32_artifact_identity") != _artifact_identity(request):
        raise ValueError("checkpoint artifact identity differs from request")
    expected_meta: dict[str, object] = {
        "track32_profile_id": TRACK32_PROFILE_ID,
        "tactile_profile": "vision_only",
        "tactile_mode": "disabled",
        "tactile_keys": [],
        "used_action_channel_ids": list(range(10)),
        "accelerator_profile": request.runtime.accelerator_profile,
        "fsdp_topology": "hsdp",
        "fsdp_shard_size": int(descriptor["local_world_size"]),
        "fsdp_replicate_size": int(descriptor["nnodes"]),
        "nccl_ib_hca": ",".join(
            descriptor["distributed_execution_contract"]["nccl_ib_hca"]
        ),
        "nccl_net_gdr_level": descriptor["distributed_execution_contract"][
            "nccl_net_gdr_level"
        ],
        "nccl_dmabuf_enable": descriptor["distributed_execution_contract"][
            "nccl_dmabuf_enable"
        ],
        "nccl_net_plugin": descriptor["distributed_execution_contract"][
            "nccl_net_plugin"
        ],
        "rccl_plugin_sha256": descriptor["distributed_execution_contract"][
            "rccl_plugin_sha256"
        ],
    }
    if any(meta.get(key) != value for key, value in expected_meta.items()):
        raise ValueError("checkpoint Franka metadata differs from request")
    checkpoint_identity = build_strict_checkpoint_identity(snapshot)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "execution_tier": "local_package_multinode",
        "formal_track32_training_completed": (
            request.train.run_role == "final_refit"
            and request.train.stop_after_step == request.train.num_steps
        ),
        "leaderboard_evaluation_completed": False,
        "invocation_id": descriptor["invocation_id"],
        "request_sha256": descriptor["request_sha256"],
        "world_size": expected_world_size,
        "nodes": descriptor["nodes"],
        "distributed_execution_contract": descriptor["distributed_execution_contract"],
        "checkpoint": str(checkpoint.resolve(strict=True)),
        "checkpoint_identity": checkpoint_identity,
        "launch_receipt": descriptor["launch_receipt"],
    }
    receipt_path = (
        request.paths.output_root
        / "training_receipts"
        / f"training_receipt.{descriptor['invocation_id']}.json"
    )
    receipt_sha256 = write_immutable_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": receipt_sha256}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--nodes", required=True)
    prepare.add_argument("--local-world-size", type=int, required=True)
    prepare.add_argument("--master-addr", required=True)
    prepare.add_argument("--master-port", type=int, required=True)
    prepare.add_argument("--nccl-ib-hca", required=True)
    prepare.add_argument("--environment", type=Path, required=True)
    prepare.add_argument("--descriptor", type=Path, required=True)
    child = subparsers.add_parser("run-child")
    child.add_argument("--environment", type=Path, required=True)
    child.add_argument("--node-rank", type=int, required=True)
    child.add_argument("--mode", choices=("preflight", "train"), required=True)
    smoke = subparsers.add_parser("run-smoke")
    smoke.add_argument("--environment", type=Path, required=True)
    smoke.add_argument("--node-rank", type=int, required=True)
    smoke.add_argument("--master-port", type=int, required=True)
    smoke.add_argument("--report", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--request", type=Path, required=True)
    verify.add_argument("--descriptor", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        nodes = tuple(value for value in args.nodes.split(",") if value)
        payload = prepare_multinode_launch(
            request_path=args.request,
            nodes=nodes,
            local_world_size=args.local_world_size,
            master_addr=args.master_addr,
            master_port=args.master_port,
            nccl_ib_hca=args.nccl_ib_hca,
            environment_path=args.environment,
            descriptor_path=args.descriptor,
        )
    elif args.command == "run-child":
        return run_child(
            environment_path=args.environment,
            node_rank=args.node_rank,
            mode=args.mode,
        )
    elif args.command == "run-smoke":
        return run_collective_smoke(
            environment_path=args.environment,
            node_rank=args.node_rank,
            master_port=args.master_port,
            report_path=args.report,
        )
    else:
        payload = verify_multinode_completion(
            request_path=args.request, descriptor_path=args.descriptor
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
