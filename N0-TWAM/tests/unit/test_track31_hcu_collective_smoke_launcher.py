"""Launcher integration tests for invocation-scoped multi-node NCCL smoke."""

from __future__ import annotations

from pathlib import Path

from tests.unit.test_track31_hcu_cluster_launcher import _calls, _run


def test_two_node_collective_smoke_runs_bounded_foreground_jobs(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        "collective-smoke",
        N0_TRACK31_NODES="n1,n2",
        N0_TRACK31_RUN_ROLE="final_refit",
        N0_TRACK31_INVOCATION_ID="phase20-001",
    )

    assert result.returncode == 0, result.stderr
    smoke_calls = [
        call
        for call in _calls(tmp_path)
        if call["cmd"][:2] == ["docker", "exec"]
        and "smoke_track31_multinode_collectives.py" in " ".join(call["cmd"])
    ]
    assert {call["host"] for call in smoke_calls} == {"n1", "n2"}
    assert all("-d" not in call["cmd"] for call in smoke_calls)
    assert all("/usr/bin/timeout" in call["cmd"] for call in smoke_calls)
    assert all("N0_TRACK31_NODES=n1,n2" in call["cmd"] for call in smoke_calls)
    assert any("NODE_RANK=0" in call["cmd"] for call in smoke_calls)
    assert any("NODE_RANK=1" in call["cmd"] for call in smoke_calls)
    validator_calls = [
        call
        for call in _calls(tmp_path)
        if "validate_track31_collective_smoke.py" in " ".join(call["cmd"])
    ]
    assert len(validator_calls) == 1
    assert validator_calls[0]["cmd"][:2] == ["/usr/bin/python3", "-B"]
    assert (
        "N0_TRACK31_COLLECTIVE_SMOKE_REPORT="
        "/formal/run/preflight/collective_smoke.phase20-001.json"
    ) in result.stdout
    assert ("N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256=" + "d" * 64) in result.stdout


def test_four_node_collective_smoke_launches_all_32_ranks(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "collective-smoke",
        N0_TRACK31_NODES="n1,n2,n3,n4",
        N0_TRACK31_RUN_ROLE="final_refit",
        N0_TRACK31_INVOCATION_ID="phase20-4n32",
        N0_TRACK31_COLLECTIVE_SMOKE_REPORT=(
            "/formal/run/preflight/collective_smoke.phase20-4n32.json"
        ),
    )

    assert result.returncode == 0, result.stderr
    smoke_calls = [
        call
        for call in _calls(tmp_path)
        if call["cmd"][:2] == ["docker", "exec"]
        and "smoke_track31_multinode_collectives.py" in " ".join(call["cmd"])
    ]
    assert {call["host"] for call in smoke_calls} == {"n1", "n2", "n3", "n4"}
    assert all("NNODES=4" in call["cmd"] for call in smoke_calls)


def test_collective_smoke_rejects_training_port_collision(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "collective-smoke",
        N0_TRACK31_NODES="n1,n2",
        N0_TRACK31_RUN_ROLE="final_refit",
        N0_TRACK31_SMOKE_MASTER_PORT="29660",
    )

    assert result.returncode != 0
    assert "distinct master ports" in result.stderr
    assert not any(
        "smoke_track31_multinode_collectives.py" in " ".join(call["cmd"])
        for call in _calls(tmp_path)
    )
