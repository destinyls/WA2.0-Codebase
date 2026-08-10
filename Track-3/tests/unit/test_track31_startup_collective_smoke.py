# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Real two-rank Gloo coverage for the Track 3.1 startup smoke CLI."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pytest
import torch.multiprocessing as mp

from script.track3_1.smoke_track31_startup_collectives import (
    _artifact_failure_config,
    _pickle_payload,
    _run_expected_failure,
)
from script.track3_1.smoke_track31_startup_collectives import main as smoke_main


def _require_gloo_loopback() -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
    except OSError as error:
        pytest.skip(f"sandbox does not allow a Gloo loopback socket: {error}")


def _run_gloo_rank(
    rank: int,
    mode: str,
    report_path: str,
    missing_artifact_root: str,
    rendezvous_uri: str,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    result = smoke_main(
        [
            "--backend=gloo",
            f"--mode={mode}",
            f"--report-path={report_path}",
            f"--missing-artifact-root={missing_artifact_root}",
            f"--init-method={rendezvous_uri}",
            "--timeout-seconds=30",
        ]
    )
    if result != 0:
        raise RuntimeError(f"rank {rank} returned {result}")


def _run_two_rank_smoke(
    tmp_path: Path,
    *,
    mode: str,
) -> dict[str, object]:
    report_path = tmp_path / f"{mode}.json"
    missing_artifact_root = tmp_path / "intentionally_missing_artifacts"
    rendezvous_path = tmp_path / f"{mode}.rendezvous"
    _require_gloo_loopback()
    mp.spawn(
        _run_gloo_rank,
        args=(
            mode,
            str(report_path),
            str(missing_artifact_root),
            rendezvous_path.as_uri(),
        ),
        nprocs=2,
        join=True,
    )
    assert report_path.is_file()
    assert list(tmp_path.glob(f".{report_path.name}.*.tmp")) == []
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_failure_configs_only_corrupt_rank_one_in_rank_mismatch(
    tmp_path: Path,
) -> None:
    mismatch_rank_zero = _artifact_failure_config(
        rank=0,
        mode="rank_mismatch",
        missing_root=tmp_path / "missing",
    )
    mismatch_rank_one = _artifact_failure_config(
        rank=1,
        mode="rank_mismatch",
        missing_root=tmp_path / "missing",
    )
    audit_rank_one = _artifact_failure_config(
        rank=1,
        mode="rank0_audit_failure",
        missing_root=tmp_path / "missing",
    )

    assert mismatch_rank_zero.rank == 0
    assert mismatch_rank_one.rank == 0
    assert audit_rank_one.rank == 1
    assert mismatch_rank_one.dataset_manifest_path == (
        tmp_path / "missing" / "dataset_manifest.json"
    )


@pytest.mark.parametrize(
    ("mode", "error_message", "startup_collectives"),
    (
        (
            "rank_mismatch",
            "distributed Track 3.1 rank contract failed: rank 1: configured rank 0",
            1,
        ),
        (
            "rank0_audit_failure",
            "rank-0 Track 3.1 artifact verification failed: {missing_root}",
            2,
        ),
    ),
)
def test_expected_failure_control_flow_without_distributed_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    error_message: str,
    startup_collectives: int,
) -> None:
    missing_root = tmp_path / "missing"
    canonical_error = error_message.format(missing_root=missing_root)

    def fake_verify(config: object, *, device: object) -> None:
        del config, device
        raise ValueError(canonical_error)

    def fake_gather(
        observation: dict[str, object],
        *,
        world_size: int,
    ) -> list[dict[str, object]]:
        assert world_size == 2
        return [
            {**observation, "rank": 0},
            {**observation, "rank": 1},
        ]

    monkeypatch.setattr(
        "script.track3_1.smoke_track31_startup_collectives."
        "verify_track31_training_startup",
        fake_verify,
    )
    monkeypatch.setattr(
        "script.track3_1.smoke_track31_startup_collectives._gather_observations",
        fake_gather,
    )

    report = _run_expected_failure(
        rank=1,
        world_size=2,
        device=None,
        mode=mode,
        missing_root=missing_root,
    )

    assert report["all_ranks_agree"] is True
    assert report["startup_collectives_completed"] == startup_collectives


def test_object_payload_contains_only_pickle_safe_primitive_tuples() -> None:
    def assert_safe(value: object) -> None:
        assert isinstance(value, (str, int, float, tuple)) or value is None
        if isinstance(value, tuple):
            for item in value:
                assert_safe(item)

    payload = _pickle_payload()

    assert len(payload) >= 10
    assert_safe(payload)


@pytest.mark.parametrize("mode", ("rank_mismatch", "rank0_audit_failure"))
def test_expected_startup_failures_are_identical_on_two_real_ranks(
    tmp_path: Path,
    mode: str,
) -> None:
    report = _run_two_rank_smoke(tmp_path, mode=mode)

    assert report["schema_version"] == 1
    assert report["status"] == "PASS"
    assert report["mode"] == mode
    assert report["backend"] == "gloo"
    assert report["world_size"] == 2
    assert report["all_ranks_agree"] is True
    observations = report["rank_observations"]
    assert isinstance(observations, list)
    assert [observation["rank"] for observation in observations] == [0, 1]
    assert {observation["outcome"] for observation in observations} == {
        "expected_failure"
    }
    assert len({observation["error"] for observation in observations}) == 1
    error = observations[0]["error"]
    if mode == "rank_mismatch":
        assert "rank 1" in error
        assert "configured rank 0" in error
        assert report["startup_collectives_completed"] == 1
    else:
        assert "rank-0 Track 3.1 artifact verification failed" in error
        assert "intentionally_missing_artifacts" in error
        assert report["startup_collectives_completed"] == 2


def test_object_broadcast_uses_pickle_safe_payload_on_two_real_ranks(
    tmp_path: Path,
) -> None:
    report = _run_two_rank_smoke(tmp_path, mode="object_broadcast_success")

    assert report["schema_version"] == 1
    assert report["status"] == "PASS"
    assert report["mode"] == "object_broadcast_success"
    assert report["backend"] == "gloo"
    assert report["world_size"] == 2
    assert report["all_ranks_agree"] is True
    digest = report["payload_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    observations = report["rank_observations"]
    assert isinstance(observations, list)
    assert [observation["rank"] for observation in observations] == [0, 1]
    assert {observation["outcome"] for observation in observations} == {"success"}
    assert {observation["payload_sha256"] for observation in observations} == {digest}
