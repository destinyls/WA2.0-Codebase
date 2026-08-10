"""Fail-closed receipt tests for the formal Stage A orchestrator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.unit.test_track31_launcher import (
    LAUNCHER,
    _base_environment,
)
from script.track3_1.track31_collective_smoke_contract import (
    SmokeContract,
    broadcast_sentinel,
    build_report,
)
from script.track3_1.track31_collective_smoke_io import publish_report_atomic

HCU_LIBRARY = LAUNCHER.parent / "script/track3_1/hcu_stage_a_lib.sh"
IDENTITY_LIBRARY = LAUNCHER.parent / "script/track3_1/hcu_stage_a_identity_lib.sh"
HCU_ORDER = ("0", "1", "5", "4", "2", "3", "7", "6")
NCCL_ENVIRONMENT = {
    "NCCL_DEBUG": "INFO",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_QPS_PER_CONNECTION": "4",
    "NCCL_IB_TC": "160",
    "NCCL_IB_TIMEOUT": "22",
    "NCCL_NET_GDR_LEVEL": "2",
    "NCCL_ROCE_SRC_PORT_LIST": "60000,60051,57663,57804",
    "NCCL_SOCKET_IFNAME": "bond1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "1",
}


def _collective_rank_observations(
    contract: SmokeContract,
) -> list[dict[str, object]]:
    expected_sum = contract.world_size * (contract.world_size + 1) // 2
    sentinel = broadcast_sentinel(contract.smoke_id)
    observations: list[dict[str, object]] = []
    container_id_digits = ("e", "f", "a", "b", "c", "d")
    for rank in range(contract.world_size):
        node_rank, local_rank = divmod(rank, 8)
        observations.append(
            {
                "rank": rank,
                "local_rank": local_rank,
                "node_rank": node_rank,
                "node_address": contract.nodes[node_rank],
                "hostname": f"container-{node_rank}",
                "container_id": container_id_digits[node_rank] * 64,
                "backend": "nccl",
                "hcu_order": list(HCU_ORDER),
                "nccl_environment": dict(NCCL_ENVIRONMENT),
                "bindings": contract.rank_binding(),
                "device": {
                    "type": "cuda",
                    "local_index": local_rank,
                    "physical_id": HCU_ORDER[local_rank],
                    "name": "Vendor HCU",
                    "total_memory_bytes": 64 * 1024**3,
                },
                "collectives": {
                    "all_reduce": {
                        "dtype": "torch.int64",
                        "input": rank + 1,
                        "expected": expected_sum,
                        "actual": expected_sum,
                    },
                    "broadcast": {
                        "dtype": "torch.int64",
                        "source_rank": 0,
                        "expected": sentinel,
                        "actual": sentinel,
                    },
                },
            }
        )
    return observations


def _run_roster(root: Path) -> subprocess.CompletedProcess[str]:
    command = (
        'source "$1"; roster=$(source_roster_audit_script); '
        'bash -lc "$roster" _ "$2"'
    )
    return subprocess.run(
        ["bash", "-c", command, "_", str(HCU_LIBRARY), str(root)],
        text=True,
        capture_output=True,
        check=False,
    )


def _add_valid_receipt(
    environment: dict[str, str],
    tmp_path: Path,
    *,
    node_count: int = 2,
) -> Path:
    invocation = "phase20-001"
    nodes = ("192.0.2.10",) + tuple(
        f"n{node_rank + 1}" for node_rank in range(1, node_count)
    )
    source_manifest = tmp_path / ".source_manifest.sha256"
    source_manifest.write_text("source\n", encoding="utf-8")
    source_sha = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    overlay_manifest = Path(environment["N0_TRACK31_PYTHON_OVERLAY"]) / (
        ".overlay_manifest.sha256"
    )
    overlay_manifest.write_text("overlay\n", encoding="utf-8")
    overlay_sha = hashlib.sha256(overlay_manifest.read_bytes()).hexdigest()
    empty_embedding = Path(environment["N0_EMPTY_EMBEDDING"])
    empty_embedding.write_bytes(b"empty")
    empty_sha = hashlib.sha256(empty_embedding.read_bytes()).hexdigest()
    save_root = tmp_path / "formal_run"
    environment_manifest = (
        save_root / "environment/environment_manifest.phase20-001.json"
    )
    environment_manifest.parent.mkdir(parents=True)
    environment_manifest.write_text("{}\n", encoding="utf-8")
    environment_sha = hashlib.sha256(environment_manifest.read_bytes()).hexdigest()
    collective_report = save_root / f"preflight/collective_smoke.{invocation}.json"
    collective_report.parent.mkdir(parents=True)
    smoke_contract = SmokeContract(
        smoke_id=invocation,
        nodes=nodes,
        hcu_order=HCU_ORDER,
        backend="nccl",
        nccl_environment=NCCL_ENVIRONMENT,
        image_id="sha256:" + "a" * 64,
        source_manifest_sha256=source_sha,
        overlay_manifest_sha256=overlay_sha,
        run_role="final_refit",
        batch_size=1,
        gradient_accumulation_steps=1,
    )
    collective_sha = publish_report_atomic(
        collective_report,
        build_report(
            smoke_contract,
            _collective_rank_observations(smoke_contract),
            created_at_unix_ns=time.time_ns(),
        ),
    )
    receipt = save_root / "launch_manifests" / f"launch_manifest.{invocation}.json"
    receipt.parent.mkdir(parents=True)
    payload = {
        "schema_version": 2,
        "invocation_id": invocation,
        "nodes": list(nodes),
        "run_role": "final_refit",
        "dataset_contract": {
            "physical_split": "train759",
            "train_view_id": "stage_a_final759_v1",
            "train_episode_count": 759,
            "validation_view_id": None,
            "normalizer_id": "qpos8_final759_v1",
        },
        "topology": {
            "node_count": node_count,
            "hcu_per_node": 8,
            "world_size": node_count * 8,
        },
        "collective_smoke": {
            "path": str(collective_report),
            "sha256": collective_sha,
        },
        "source_manifest_sha256": source_sha,
        "overlay_manifest_sha256": overlay_sha,
        "image_id": "sha256:" + "a" * 64,
        "empty_embedding_sha256": empty_sha,
        "ssh_port": 36000,
        "known_hosts_sha256": "c" * 64,
        "environment_manifest": {
            "path": str(environment_manifest),
            "sha256": environment_sha,
        },
        "rank0_foreground_preflight": {
            "node": "192.0.2.10",
            "node_rank": 0,
            "status": "passed",
        },
        "hcu": {"per_node": 8, "order": "0,1,5,4,2,3,7,6"},
        "nccl": NCCL_ENVIRONMENT,
        "multinode_hcu_preflight": "passed",
        "recipe": {
            "num_steps": 5000,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "save_interval": 500,
            "val_interval": 100,
            "max_latent_frames": 5,
            "load_worker": 0,
            "train_seed": 20260801,
            "action_init_seed": 0,
            "pythonhashseed": 20260801,
        },
        "stop_after_step": 20,
        "resume_from": None,
        "resume_parent_identity": None,
        "provenance_trust_boundary": "trusted-operator provenance only",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    receipt.write_bytes(raw)
    receipt.chmod(0o444)
    environment.update(
        {
            "NGPU": "8",
            "NNODES": str(node_count),
            "NODE_RANK": "1",
            "MASTER_ADDR": "192.0.2.10",
            "N0_TRACK31_TRAIN_PROFILE": "multitask_pretrain_v1",
            "N0_TRACK31_RUN_ROLE": "final_refit",
            "N0_TRACK31_NODES": ",".join(nodes),
            "N0_TRACK31_NODE_ADDRESS": "n2",
            "N0_TRACK31_CONTAINER_ID": "f" * 64,
            "N0_TRACK31_SAVE_ROOT": str(save_root),
            "N0_TRACK31_NUM_STEPS": "5000",
            "N0_TRACK31_SAVE_INTERVAL": "500",
            "N0_TRACK31_VAL_INTERVAL": "100",
            "N0_TRACK31_BATCH_SIZE": "1",
            "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS": "1",
            "N0_TRACK31_STOP_AFTER_STEP": "20",
            "N0_TRACK31_ORCHESTRATED_PREFLIGHT": "track31-stage-a-v2",
            "N0_TRACK31_INVOCATION_ID": invocation,
            "N0_TRACK31_LAUNCH_RECEIPT": str(receipt),
            "N0_TRACK31_LAUNCH_RECEIPT_SHA256": hashlib.sha256(raw).hexdigest(),
            "N0_TRACK31_LAUNCH_MANIFEST_SHA256": hashlib.sha256(raw).hexdigest(),
            "N0_TRACK31_SOURCE_MANIFEST_SHA256": source_sha,
            "N0_TRACK31_OVERLAY_MANIFEST_SHA256": overlay_sha,
            "N0_TRACK31_IMAGE_ID": "sha256:" + "a" * 64,
            "N0_EMPTY_EMBEDDING_SHA256": empty_sha,
            "N0_TRACK31_SSH_PORT": "36000",
            "N0_TRACK31_KNOWN_HOSTS_SHA256": "c" * 64,
            "N0_TRACK31_ENVIRONMENT_MANIFEST": str(environment_manifest),
            "N0_TRACK31_ENVIRONMENT_MANIFEST_SHA256": environment_sha,
            "N0_TRACK31_COLLECTIVE_SMOKE_REPORT": str(collective_report),
            "N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256": collective_sha,
            "N0_TRACK31_PYTHON_BIN": sys.executable,
            "HIP_VISIBLE_DEVICES": ",".join(HCU_ORDER),
            **NCCL_ENVIRONMENT,
        }
    )
    return receipt


def _rewrite_receipt(
    receipt: Path,
    environment: dict[str, str],
    *,
    failure: str,
) -> None:
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if failure == "schema":
        payload["schema_version"] = 1
    elif failure == "role":
        payload["run_role"] = "development"
    elif failure == "topology":
        payload["topology"]["world_size"] = 15
    elif failure == "nodes":
        payload["nodes"] = ["192.0.2.10"]
    elif failure == "dataset":
        payload["dataset_contract"]["train_episode_count"] = 758
    elif failure == "recipe":
        payload["recipe"]["gradient_accumulation_steps"] = 12
    elif failure == "collective":
        report = Path(environment["N0_TRACK31_COLLECTIVE_SMOKE_REPORT"])
        smoke = json.loads(report.read_text(encoding="utf-8"))
        smoke["world_size"] = 15
        report.chmod(0o644)
        report.write_text(
            json.dumps(smoke, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        report.chmod(0o444)
        smoke_sha = hashlib.sha256(report.read_bytes()).hexdigest()
        environment["N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256"] = smoke_sha
        payload["collective_smoke"]["sha256"] = smoke_sha
    else:
        raise ValueError(f"unknown receipt failure: {failure}")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    receipt.chmod(0o644)
    receipt.write_bytes(raw)
    receipt.chmod(0o444)
    digest = hashlib.sha256(raw).hexdigest()
    environment["N0_TRACK31_LAUNCH_RECEIPT_SHA256"] = digest
    environment["N0_TRACK31_LAUNCH_MANIFEST_SHA256"] = digest


def test_launcher_accepts_valid_orchestrated_preflight_receipt(tmp_path: Path) -> None:
    environment, call_log = _base_environment(tmp_path)
    _add_valid_receipt(environment, tmp_path)
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert calls[0].startswith("torchrun expected_world_size=16")
    assert "validated trusted-operator launch provenance" in result.stderr


def test_launcher_accepts_valid_four_node_orchestrated_receipt(
    tmp_path: Path,
) -> None:
    environment, call_log = _base_environment(tmp_path)
    _add_valid_receipt(environment, tmp_path, node_count=4)
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert calls[0].startswith("torchrun expected_world_size=32")
    assert "--nnodes=4" in calls[0]
    assert "validated trusted-operator launch provenance" in result.stderr


@pytest.mark.parametrize("failure", ("sha", "writable", "partial"))
def test_launcher_rejects_invalid_orchestrated_preflight_receipt(
    tmp_path: Path,
    failure: str,
) -> None:
    environment, call_log = _base_environment(tmp_path)
    receipt = _add_valid_receipt(environment, tmp_path)
    if failure == "sha":
        environment["N0_TRACK31_LAUNCH_RECEIPT_SHA256"] = "0" * 64
    elif failure == "writable":
        receipt.chmod(0o644)
    else:
        environment.pop("N0_TRACK31_ORCHESTRATED_PREFLIGHT")
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not call_log.exists()


@pytest.mark.parametrize(
    "failure",
    (
        "schema",
        "role",
        "topology",
        "nodes",
        "dataset",
        "recipe",
        "collective",
        "protocol_v1",
    ),
)
def test_launcher_rejects_tampered_topology_and_dataset_contract(
    tmp_path: Path,
    failure: str,
) -> None:
    environment, call_log = _base_environment(tmp_path)
    receipt = _add_valid_receipt(environment, tmp_path)
    if failure == "protocol_v1":
        environment["N0_TRACK31_ORCHESTRATED_PREFLIGHT"] = "track31-stage-a-v1"
    else:
        _rewrite_receipt(receipt, environment, failure=failure)
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not call_log.exists()


def test_roster_requires_exact_canonical_path_set(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    inside = root / "inside.py"
    omitted = root / "omitted.py"
    inside.write_text("safe", encoding="utf-8")
    omitted.write_text("bound", encoding="utf-8")
    inside_sha = hashlib.sha256(inside.read_bytes()).hexdigest()
    omitted_sha = hashlib.sha256(omitted.read_bytes()).hexdigest()
    manifest = root / ".source_manifest.sha256"
    manifest.write_text(
        f"{inside_sha}  inside.py\n{omitted_sha}  omitted.py\n", encoding="utf-8"
    )
    assert _run_roster(root).returncode == 0

    manifest.write_text(
        f"{inside_sha}  inside.py\n{inside_sha}  ./inside.py\n", encoding="utf-8"
    )
    assert _run_roster(root).returncode != 0

    outside = tmp_path / "outside.py"
    outside.write_text("unsafe", encoding="utf-8")
    outside_sha = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest.write_text(
        f"{inside_sha}  inside.py\n{outside_sha}  ../outside.py\n",
        encoding="utf-8",
    )
    assert _run_roster(root).returncode != 0


def test_process_count_reports_total_and_direct_workers_separately() -> None:
    rows = [
        "PID PPID PGID COMMAND",
        "100 1 100 /usr/bin/python3 /usr/local/bin/torchrun --nnodes=6 -m n0_twam.train",
        *(
            f"{101 + rank} 100 {101 + rank} /usr/bin/python3 -m n0_twam.train"
            for rank in range(8)
        ),
        "999 1 999 /usr/bin/python3 -m n0_twam.train --unowned",
    ]
    command = (
        'source "$1"; remote_exec() { printf "%s\\n" "$PROCESS_ROWS"; }; '
        "training_process_state ignored ignored"
    )
    result = subprocess.run(
        ["bash", "-c", command, "_", str(HCU_LIBRARY)],
        env={"PROCESS_ROWS": "\n".join(rows)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1 9 100 100 8"


def test_resume_parent_identity_requires_exact_step20_for_stop25(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run/checkpoints/checkpoint_step_20"
    checkpoint.mkdir(parents=True)
    transformer = {"sha256": "a" * 64}
    state = {"step": 20, "transformer_identity": transformer}
    completion = {
        "step": 20,
        "status": "complete",
        "transformer_identity": transformer,
    }
    (checkpoint / "training_state.json").write_text(json.dumps(state))
    (checkpoint / "checkpoint_complete.json").write_text(json.dumps(completion))
    (checkpoint / "train_meta.json").write_text("{}")
    command = (
        'source "$1"; source "$2"; remote_exec() { shift; "$@"; }; '
        "NODES=(local); CONTAINER_RUN_ROOT=/formal/run; N0_TRACK31_STOP_AFTER_STEP=25; "
        'N0_TRACK31_RUN_ROOT_HOST="$3/run"; '
        "N0_TRACK31_RESUME_FROM=/formal/run/checkpoints/checkpoint_step_20; "
        "resolve_resume_parent_identity || exit; "
        'printf "%s" "$RESUME_PARENT_IDENTITY_JSON"'
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "_",
            str(HCU_LIBRARY),
            str(IDENTITY_LIBRARY),
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["step"] == 20

    state["step"] = 19
    (checkpoint / "training_state.json").write_text(json.dumps(state))
    assert (
        subprocess.run(
            [
                "bash",
                "-c",
                command,
                "_",
                str(HCU_LIBRARY),
                str(IDENTITY_LIBRARY),
                str(tmp_path),
            ],
            check=False,
        ).returncode
        != 0
    )
