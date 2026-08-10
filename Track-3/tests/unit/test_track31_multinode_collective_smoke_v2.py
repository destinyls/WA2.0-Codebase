# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Schema-v2 evidence tests for the formal Stage A collective smoke."""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import pytest

from script.track3_1.smoke_track31_multinode_collectives import (
    SmokeContract,
    _contract_from_environment,
    _parse_args as parse_producer_args,
    _validate_torchrun_environment,
    broadcast_sentinel,
    build_report,
    publish_report_atomic,
)
from script.track3_1.track31_collective_smoke_contract import (
    canonical_report_sha256,
)
from script.track3_1.validate_track31_collective_smoke import (
    CollectiveSmokeValidationError,
    main as validator_main,
    validate_collective_smoke_report,
)

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
NOW_NS = 1_786_000_000_000_000_000


def _contract(node_count: int) -> SmokeContract:
    return SmokeContract(
        smoke_id=f"stage-a-final759-{node_count}n",
        nodes=tuple(f"10.0.0.{index + 1}" for index in range(node_count)),
        hcu_order=HCU_ORDER,
        backend="nccl",
        nccl_environment=NCCL_ENVIRONMENT,
        image_id="sha256:" + "a" * 64,
        source_manifest_sha256="b" * 64,
        overlay_manifest_sha256="c" * 64,
        run_role="final_refit",
        batch_size=1,
        gradient_accumulation_steps=1,
    )


def _rank_observations(contract: SmokeContract) -> list[dict[str, object]]:
    expected_sum = contract.world_size * (contract.world_size + 1) // 2
    sentinel = broadcast_sentinel(contract.smoke_id)
    binding = contract.rank_binding()
    observations: list[dict[str, object]] = []
    for rank in range(contract.world_size):
        node_rank, local_rank = divmod(rank, 8)
        observations.append(
            {
                "rank": rank,
                "local_rank": local_rank,
                "node_rank": node_rank,
                "node_address": contract.nodes[node_rank],
                "hostname": f"node-{node_rank}",
                "container_id": f"container-{node_rank}",
                "backend": "nccl",
                "hcu_order": list(HCU_ORDER),
                "nccl_environment": dict(NCCL_ENVIRONMENT),
                "bindings": copy.deepcopy(binding),
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


def _write_report(
    tmp_path: Path,
    contract: SmokeContract,
    *,
    created_at_unix_ns: int = NOW_NS,
    mutate: object | None = None,
) -> tuple[Path, str]:
    report = build_report(
        contract,
        _rank_observations(contract),
        created_at_unix_ns=created_at_unix_ns,
    )
    if mutate is not None:
        assert callable(mutate)
        mutate(report)
    path = tmp_path / f"collective_smoke.{contract.smoke_id}.json"
    expected_sha256 = canonical_report_sha256(report)
    assert publish_report_atomic(path, report) == expected_sha256
    os.utime(
        path,
        ns=(created_at_unix_ns, created_at_unix_ns),
        follow_symlinks=False,
    )
    return path, expected_sha256


def _validate(
    path: Path,
    expected_sha256: str,
    contract: SmokeContract,
    *,
    now_ns: int = NOW_NS,
    expected_container_ids: dict[str, str] | None = None,
) -> dict[str, object]:
    return validate_collective_smoke_report(
        path,
        expected_sha256=expected_sha256,
        smoke_id=contract.smoke_id,
        expected_nodes=contract.nodes,
        expected_hcu_order=contract.hcu_order,
        expected_nccl_environment=contract.nccl_environment,
        image_id=contract.image_id,
        source_manifest_sha256=contract.source_manifest_sha256,
        overlay_manifest_sha256=contract.overlay_manifest_sha256,
        run_role=contract.run_role,
        batch_size=contract.batch_size,
        gradient_accumulation_steps=contract.gradient_accumulation_steps,
        expected_container_ids=expected_container_ids,
        max_age_seconds=300,
        now_ns=now_ns,
    )


@pytest.mark.parametrize("node_count", (2, 4, 6))
def test_schema_v2_report_validates_all_ranks_for_formal_topologies(
    tmp_path: Path,
    node_count: int,
) -> None:
    contract = _contract(node_count)
    path, digest = _write_report(tmp_path, contract)
    container_ids = {
        node: f"container-{node_rank}" for node_rank, node in enumerate(contract.nodes)
    }

    summary = _validate(
        path,
        digest,
        contract,
        expected_container_ids=container_ids,
    )

    assert summary == {
        "schema_version": 2,
        "status": "PASS",
        "smoke_id": contract.smoke_id,
        "backend": "nccl",
        "world_size": node_count * 8,
        "report_sha256": digest,
    }
    assert path.stat().st_mode & 0o777 == 0o444
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_atomic_publication_refuses_preexisting_and_symlink_targets(
    tmp_path: Path,
) -> None:
    contract = _contract(2)
    report = build_report(
        contract,
        _rank_observations(contract),
        created_at_unix_ns=NOW_NS,
    )
    path = tmp_path / "evidence.json"
    publish_report_atomic(path, report)

    with pytest.raises(FileExistsError, match="already exists"):
        publish_report_atomic(path, report)

    target = tmp_path / "unrelated.json"
    target.write_text("{}", encoding="utf-8")
    symlink_path = tmp_path / "dangling-safe-name.json"
    symlink_path.symlink_to(target)
    with pytest.raises(FileExistsError, match="already exists"):
        publish_report_atomic(symlink_path, report)
    assert target.read_text(encoding="utf-8") == "{}"


def test_validator_rejects_non_readonly_and_symlink_reports(tmp_path: Path) -> None:
    contract = _contract(2)
    path, digest = _write_report(tmp_path, contract)
    path.chmod(0o644)
    with pytest.raises(CollectiveSmokeValidationError, match="mode 0444"):
        _validate(path, digest, contract)

    path.chmod(0o444)
    symlink_path = tmp_path / "linked.json"
    symlink_path.symlink_to(path)
    with pytest.raises(CollectiveSmokeValidationError, match="symlink"):
        _validate(symlink_path, digest, contract)

    hardlink_path = tmp_path / "hardlinked.json"
    os.link(path, hardlink_path)
    with pytest.raises(CollectiveSmokeValidationError, match="hard link"):
        _validate(path, digest, contract)


def test_validator_rejects_wrong_sha_stale_or_wrong_expected_container(
    tmp_path: Path,
) -> None:
    contract = _contract(2)
    path, digest = _write_report(tmp_path, contract)
    with pytest.raises(CollectiveSmokeValidationError, match="SHA256"):
        _validate(path, "0" * 64, contract)
    with pytest.raises(CollectiveSmokeValidationError, match="stale"):
        _validate(path, digest, contract, now_ns=NOW_NS + 301_000_000_000)
    with pytest.raises(CollectiveSmokeValidationError, match="container_id"):
        _validate(
            path,
            digest,
            contract,
            expected_container_ids={
                contract.nodes[0]: "wrong",
                contract.nodes[1]: "container-1",
            },
        )
    old_ns = NOW_NS - 30_000_000_000
    os.utime(path, ns=(old_ns, old_ns), follow_symlinks=False)
    with pytest.raises(CollectiveSmokeValidationError, match="mtime"):
        _validate(path, digest, contract)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("smoke_id", "smoke_id"),
        ("rank_binding", "bindings"),
        ("rank_layout", "local_rank"),
        ("all_reduce", "all_reduce"),
        ("duplicate_container", "container_id"),
        ("nccl_environment", "nccl_environment"),
        ("bool_as_int", "broadcast.source_rank"),
        ("topology_float", "topology.node_count"),
        ("collective_float", "all_reduce.expected"),
        ("collective_list_float", "all_reduce.actual_by_rank"),
        ("execution_float", "batch_size"),
        ("binding_float", "bindings.batch_size"),
    ),
)
def test_validator_fails_closed_on_identity_rank_and_collective_mutations(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    contract = _contract(2)

    def mutate(report: dict[str, object]) -> None:
        ranks = report["ranks"]
        assert isinstance(ranks, list)
        if mutation == "smoke_id":
            report["smoke_id"] = "other-smoke"
        elif mutation == "rank_binding":
            ranks[0]["bindings"]["image_id"] = "sha256:" + "d" * 64
        elif mutation == "rank_layout":
            ranks[1]["local_rank"] = 0
        elif mutation == "all_reduce":
            ranks[0]["collectives"]["all_reduce"]["actual"] += 1
        elif mutation == "duplicate_container":
            for rank in ranks[8:]:
                rank["container_id"] = "container-0"
        elif mutation == "nccl_environment":
            ranks[0]["nccl_environment"]["NCCL_IB_DISABLE"] = "1"
        elif mutation == "bool_as_int":
            ranks[0]["collectives"]["broadcast"]["source_rank"] = False
        elif mutation == "topology_float":
            report["topology"]["node_count"] = 2.0
        elif mutation == "collective_float":
            report["collective_evidence"]["all_reduce"]["expected"] = 136.0
        elif mutation == "collective_list_float":
            report["collective_evidence"]["all_reduce"]["actual_by_rank"][0] = 136.0
        elif mutation == "execution_float":
            report["training_execution_contract"]["batch_size"] = 2.0
        elif mutation == "binding_float":
            ranks[0]["bindings"]["training_execution_contract"]["batch_size"] = 2.0

    path, digest = _write_report(
        tmp_path,
        contract,
        mutate=mutate,
    )
    with pytest.raises(CollectiveSmokeValidationError, match=message):
        _validate(path, digest, contract)


def test_validator_cli_binds_full_formal_identity_surface(tmp_path: Path) -> None:
    contract = _contract(2)
    path, digest = _write_report(
        tmp_path,
        contract,
        created_at_unix_ns=time.time_ns(),
    )
    arguments = [
        f"--report-path={path}",
        f"--expected-report-sha256={digest}",
        f"--expected-smoke-id={contract.smoke_id}",
        f"--expected-nodes={','.join(contract.nodes)}",
        f"--expected-hcu-order={','.join(contract.hcu_order)}",
        f"--expected-image-id={contract.image_id}",
        f"--expected-source-manifest-sha256={contract.source_manifest_sha256}",
        f"--expected-overlay-manifest-sha256={contract.overlay_manifest_sha256}",
        f"--expected-run-role={contract.run_role}",
        f"--expected-batch-size={contract.batch_size}",
        "--expected-gradient-accumulation-steps=1",
        "--expected-world-size=16",
        "--expected-node-count=2",
        "--expected-hcu-per-node=8",
    ]
    arguments.extend(
        f"--expected-nccl-env={name}={value}"
        for name, value in NCCL_ENVIRONMENT.items()
    )
    arguments.extend(
        f"--expected-container-id={node}=container-{node_rank}"
        for node_rank, node in enumerate(contract.nodes)
    )

    assert validator_main(arguments) == 0


def test_producer_binds_exact_torchrun_and_live_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(os.environ):
        if name.startswith(("NCCL_", "TORCH_NCCL_")):
            monkeypatch.delenv(name)
    environment = {
        **NCCL_ENVIRONMENT,
        "HIP_VISIBLE_DEVICES": ",".join(HCU_ORDER),
        "N0_TRACK31_IMAGE_ID": "sha256:" + "a" * 64,
        "N0_TRACK31_SOURCE_MANIFEST_SHA256": "b" * 64,
        "N0_TRACK31_OVERLAY_MANIFEST_SHA256": "c" * 64,
        "N0_TRACK31_RUN_ROLE": "final_refit",
        "N0_TRACK31_BATCH_SIZE": "1",
        "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS": "1",
        "N0_TRACK31_INVOCATION_ID": "stage-a-final759-2n",
        "N0_TRACK31_NODE_ADDRESS": "10.0.0.2",
        "N0_TRACK31_CONTAINER_ID": "container-1",
        "RANK": "9",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "16",
        "NODE_RANK": "1",
        "LOCAL_WORLD_SIZE": "8",
        "GROUP_RANK": "1",
        "GROUP_WORLD_SIZE": "2",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    arguments = parse_producer_args(
        [
            "--report-path=/tmp/collective_smoke.stage-a-final759-2n.json",
            "--smoke-id=stage-a-final759-2n",
            "--expected-nodes=10.0.0.1,10.0.0.2",
            *(
                f"--expected-nccl-env={key}={value}"
                for key, value in NCCL_ENVIRONMENT.items()
            ),
        ]
    )

    contract = _contract_from_environment(arguments)

    assert _validate_torchrun_environment(contract) == (9, 1, 1)
