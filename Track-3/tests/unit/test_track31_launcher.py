"""Shell-launch contract tests for Track 3.1 distributed training."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "run_track31_univtac.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _base_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    call_log = tmp_path / "calls.log"
    _write_executable(
        command_dir / "python",
        "#!/bin/bash\n"
        "printf 'python expected_world_size=%s profile=%s role=%s "
        "save_root=%s args=%s\\n' "
        '"$N0_TRACK31_EXPECTED_WORLD_SIZE" "$N0_TRACK31_TRAIN_PROFILE" '
        '"$N0_TRACK31_RUN_ROLE" "$N0_TRACK31_SAVE_ROOT" "$*" '
        '>> "$CALL_LOG"\n',
    )
    _write_executable(
        command_dir / "torchrun",
        "#!/bin/bash\n"
        "printf 'torchrun expected_world_size=%s args=%s\\n' "
        '"$N0_TRACK31_EXPECTED_WORLD_SIZE" "$*" >> "$CALL_LOG"\n',
    )
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{command_dir}:{environment['PATH']}",
            "CALL_LOG": str(call_log),
            "N0_TRACK31_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "N0_TRACK31_LEROBOT_ROOT": str(tmp_path / "lerobot"),
            "N0_BASE_MODEL": str(tmp_path / "base_model"),
            "N0_EMPTY_EMBEDDING": str(tmp_path / "empty_emb.pt"),
            "N0_EMPTY_EMBEDDING_SHA256": "d" * 64,
            "N0_RELEASED_CHECKPOINT": str(tmp_path / "released"),
            "N0_RELEASED_TRANSFORMER_SHA256": "a" * 64,
            "N0_TRACK31_PYTHON_OVERLAY": str(overlay),
            "N0_TRACK31_PYTHON_BIN": str(command_dir / "python"),
            "N0_TRACK31_TORCHRUN_BIN": str(command_dir / "torchrun"),
        }
    )
    for name in (
        "N0_TRACK31_INIT_FROM",
        "N0_TRACK31_RESUME_FROM",
        "N0_TRACK31_SAVE_ROOT",
        "N0_TRACK31_TRAIN_PROFILE",
        "N0_TRACK31_RUN_ROLE",
        "N0_TRACK31_VALIDATION_VIEW_PATH",
        "N0_TRACK31_ORCHESTRATED_PREFLIGHT",
        "N0_TRACK31_INVOCATION_ID",
        "N0_TRACK31_LAUNCH_RECEIPT",
        "N0_TRACK31_LAUNCH_RECEIPT_SHA256",
        "N0_TRACK31_SOURCE_MANIFEST_SHA256",
        "N0_TRACK31_OVERLAY_MANIFEST_SHA256",
        "N0_TRACK31_IMAGE_ID",
        "N0_TRACK31_LAUNCH_MANIFEST_SHA256",
    ):
        environment.pop(name, None)
    return environment, call_log


def test_launcher_builds_four_node_32_rank_torchrun_contract(
    tmp_path: Path,
) -> None:
    environment, call_log = _base_environment(tmp_path)
    environment.update(
        {
            "NGPU": "8",
            "NNODES": "4",
            "NODE_RANK": "2",
            "MASTER_ADDR": "192.0.2.10",
            "MASTER_PORT": "29631",
        }
    )

    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert calls[0].startswith("python expected_world_size=32")
    assert "profile=multitask_pretrain_v1" in calls[0]
    assert "role=final_refit" in calls[0]
    assert (
        f"save_root={tmp_path}/artifacts/runs/" "multitask_pretrain_v1/final_refit"
    ) in calls[0]
    assert calls[1].startswith("torchrun expected_world_size=32")
    assert "--nnodes=4" in calls[1]
    assert "--nproc-per-node=8" in calls[1]
    assert "--node-rank=2" in calls[1]
    assert "--master-addr=192.0.2.10" in calls[1]
    assert "--master-port=29631" in calls[1]


def test_launcher_preserves_single_node_port_compatibility(tmp_path: Path) -> None:
    environment, call_log = _base_environment(tmp_path)
    environment.update({"NGPU": "4", "PORT": "29999"})
    environment.pop("MASTER_PORT", None)

    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert calls[0].startswith("python expected_world_size=4")
    assert "--nnodes=1" in calls[1]
    assert "--nproc-per-node=4" in calls[1]
    assert "--node-rank=0" in calls[1]
    assert "--master-addr=127.0.0.1" in calls[1]
    assert "--master-port=29999" in calls[1]


def test_launcher_rejects_node_rank_outside_cluster(tmp_path: Path) -> None:
    environment, call_log = _base_environment(tmp_path)
    environment.update(
        {
            "NGPU": "8",
            "NNODES": "4",
            "NODE_RANK": "4",
            "MASTER_ADDR": "192.0.2.10",
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "NODE_RANK must be an integer in [0, NNODES)" in result.stderr
    assert not call_log.exists()


def test_launcher_stage_b_uses_weights_only_parent_without_released_prior(
    tmp_path: Path,
) -> None:
    environment, call_log = _base_environment(tmp_path)
    environment.update(
        {
            "N0_TRACK31_TRAIN_PROFILE": "target_finetune_v1",
            "N0_TRACK31_RUN_ROLE": "development",
            "N0_TRACK31_INIT_FROM": str(tmp_path / "stage_a_parent"),
        }
    )
    environment.pop("N0_RELEASED_CHECKPOINT")
    environment.pop("N0_RELEASED_TRANSFORMER_SHA256")

    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    call = call_log.read_text(encoding="utf-8").splitlines()[0]
    assert "profile=target_finetune_v1" in call
    assert "role=development" in call
    assert (
        f"save_root={tmp_path}/artifacts/runs/" "target_finetune_v1/development"
    ) in call


def test_launcher_rejects_stage_b_without_parent(tmp_path: Path) -> None:
    environment, call_log = _base_environment(tmp_path)
    environment["N0_TRACK31_TRAIN_PROFILE"] = "target_finetune_v1"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "N0_TRACK31_INIT_FROM" in result.stderr
    assert not call_log.exists()


def test_launcher_rejects_stage_a_init_from(tmp_path: Path) -> None:
    environment, call_log = _base_environment(tmp_path)
    environment["N0_TRACK31_INIT_FROM"] = str(tmp_path / "wrong_parent")

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "reserved for target_finetune_v1" in result.stderr
    assert not call_log.exists()


def test_launcher_final_refit_rejects_validation_view_override(
    tmp_path: Path,
) -> None:
    environment, call_log = _base_environment(tmp_path)
    environment["N0_TRACK31_VALIDATION_VIEW_PATH"] = str(tmp_path / "frozen_view.json")

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "final_refit must not use a validation view" in result.stderr
    assert not call_log.exists()


def test_launcher_rejects_loopback_master_for_multinode(tmp_path: Path) -> None:
    environment, call_log = _base_environment(tmp_path)
    environment.update(
        {
            "NGPU": "8",
            "NNODES": "4",
            "NODE_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "MASTER_ADDR must be reachable" in result.stderr
    assert not call_log.exists()


@pytest.mark.parametrize(
    "extra_arguments",
    (
        ("--config-name", "posttrain"),
        ("--save-root", "/tmp/unchecked-track31-output"),
        ("unexpected-positional-argument",),
    ),
)
def test_launcher_rejects_all_cli_arguments_before_preflight(
    tmp_path: Path,
    extra_arguments: tuple[str, ...],
) -> None:
    environment, call_log = _base_environment(tmp_path)

    result = subprocess.run(
        ["bash", str(LAUNCHER), *extra_arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not accept CLI arguments" in result.stderr
    assert not call_log.exists()
