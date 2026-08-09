# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Synchronous public runner for N0-TWAM Track 3.1 training."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)

from .local_provenance import (
    LocalProvenance,
    package_import_root,
    prepare_local_provenance,
    sha256_file,
    write_immutable_json,
)
from .request import Track31TrainRequest

_VISIBILITY_ENV = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NGPU",
        "NNODES",
        "NODE_RANK",
        "PORT",
    }
)
_INJECTION_ENV = frozenset(
    {
        "DYLD_INSERT_LIBRARIES",
        "LD_AUDIT",
        "LD_PRELOAD",
    }
)


def _base_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {
        key: value
        for key, value in source.items()
        if not key.startswith("N0_")
        and not key.startswith("PYTHON")
        and key not in _VISIBILITY_ENV
        and key not in _INJECTION_ENV
    }


def build_preflight_command() -> tuple[str, ...]:
    return (sys.executable, "-m", "n0_twam.track31.preflight")


def build_training_command(request: Track31TrainRequest) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc-per-node={len(request.runtime.devices)}",
        "--node-rank=0",
        "--master-addr=127.0.0.1",
        f"--master-port={request.runtime.master_port}",
        "--tee",
        "3",
        "-m",
        "n0_twam.train",
        "--config-name",
        "track31_univtac",
    )


def build_launch_plan(request: Track31TrainRequest) -> dict[str, object]:
    """Build the public, secret-free execution plan shown by ``--dry-run``."""

    return {
        "schema_version": 1,
        "execution_tier": "local_package",
        "formal_track31": False,
        "leaderboard_eligible": False,
        "run_id": request.run_id,
        "world_size": len(request.runtime.devices),
        "devices": list(request.runtime.devices),
        "request": str(request.source_path),
        "paths": {
            "artifact_root": str(request.paths.artifact_root),
            "lerobot_root": str(request.paths.lerobot_root),
            "base_model": str(request.paths.base_model),
            "empty_embedding": str(request.paths.empty_embedding),
            "released_checkpoint": (
                None
                if request.paths.released_checkpoint is None
                else str(request.paths.released_checkpoint)
            ),
            "output_root": str(request.paths.output_root),
            "resume_from": (
                None
                if request.paths.resume_from is None
                else str(request.paths.resume_from)
            ),
            "init_from": (
                None
                if request.paths.init_from is None
                else str(request.paths.init_from)
            ),
        },
        "empty_embedding_sha256": request.paths.empty_embedding_sha256,
        "released_transformer_sha256": (request.paths.released_transformer_sha256),
        "preflight_command": list(build_preflight_command()),
        "training_command": list(build_training_command(request)),
        "recipe": {
            "profile": request.train.profile,
            "run_role": request.train.run_role,
            "num_steps": request.train.num_steps,
            "stop_after_step": request.train.stop_after_step,
            "save_interval": request.train.save_interval,
            "val_interval": request.train.val_interval,
            "batch_size": request.train.batch_size,
            "gradient_accumulation_steps": (request.train.gradient_accumulation_steps),
            "max_latent_frames": request.train.max_latent_frames,
            "seed": request.train.seed,
            "action_init_seed": request.train.action_init_seed,
        },
    }


def build_training_environment(
    request: Track31TrainRequest,
    provenance: LocalProvenance,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Map a strict request onto a clean child environment."""

    environment = _base_environment(environ)
    devices = ",".join(str(device) for device in request.runtime.devices)
    paths = request.paths
    recipe = request.train
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": devices,
            "HIP_VISIBLE_DEVICES": devices,
            "PYTHONHASHSEED": str(recipe.seed),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(package_import_root()),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "N0_TRACK31_ARTIFACT_ROOT": str(paths.artifact_root),
            "N0_TRACK31_LEROBOT_ROOT": str(paths.lerobot_root),
            "N0_BASE_MODEL": str(paths.base_model),
            "N0_EMPTY_EMBEDDING": str(paths.empty_embedding),
            "N0_EMPTY_EMBEDDING_SHA256": paths.empty_embedding_sha256,
            "N0_TRACK31_SAVE_ROOT": str(paths.output_root),
            "N0_TRACK31_TRAIN_PROFILE": recipe.profile,
            "N0_TRACK31_RUN_ROLE": recipe.run_role,
            "N0_TRACK31_EXPECTED_WORLD_SIZE": str(len(request.runtime.devices)),
            "N0_TRACK31_NUM_STEPS": str(recipe.num_steps),
            "N0_TRACK31_STOP_AFTER_STEP": str(recipe.stop_after_step),
            "N0_TRACK31_SAVE_INTERVAL": str(recipe.save_interval),
            "N0_TRACK31_VAL_INTERVAL": str(recipe.val_interval),
            "N0_TRACK31_BATCH_SIZE": str(recipe.batch_size),
            "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS": str(
                recipe.gradient_accumulation_steps
            ),
            "N0_TRACK31_MAX_LATENT_FRAMES": str(recipe.max_latent_frames),
            "N0_TRACK31_LOAD_WORKER": "0",
            "N0_TRAIN_SEED": str(recipe.seed),
            "N0_ACTION_INIT_SEED": str(recipe.action_init_seed),
            **provenance.environment(),
        }
    )
    if paths.released_checkpoint is not None:
        environment["N0_RELEASED_CHECKPOINT"] = str(paths.released_checkpoint)
    if paths.released_transformer_sha256 is not None:
        environment["N0_RELEASED_TRANSFORMER_SHA256"] = (
            paths.released_transformer_sha256
        )
    if paths.resume_from is not None:
        environment["N0_TRACK31_RESUME_FROM"] = str(paths.resume_from)
    if paths.init_from is not None:
        environment["N0_TRACK31_INIT_FROM"] = str(paths.init_from)
    return environment


def _run_preflight(
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
    working_directory: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(environment),
            cwd=working_directory,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            f"Track 3.1 preflight failed ({process.returncode}); see {log_path}"
        )


def _run_training_process(
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
    working_directory: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(environment),
            cwd=working_directory,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stderr.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Track 3.1 training failed ({return_code}); see {log_path}")


def _verify_completed_checkpoint(
    request: Track31TrainRequest,
    provenance: LocalProvenance,
) -> tuple[Path, dict[str, object]]:
    checkpoint = (
        request.paths.output_root
        / "checkpoints"
        / f"checkpoint_step_{request.train.stop_after_step}"
    )
    snapshot = capture_strict_checkpoint_snapshot(checkpoint)
    if snapshot.step != request.train.stop_after_step:
        raise ValueError("completed checkpoint step differs from the request")
    if snapshot.world_size != len(request.runtime.devices):
        raise ValueError("completed checkpoint world_size differs from the request")
    if snapshot.runtime_source_identity != provenance.runtime_source_identity:
        raise ValueError("completed checkpoint runtime source differs from this launch")
    if (
        snapshot.checkpoint_invocation_identity
        != provenance.checkpoint_invocation_identity
    ):
        raise ValueError("completed checkpoint invocation differs from this launch")
    return checkpoint.resolve(strict=True), build_strict_checkpoint_identity(snapshot)


def _prepare_isolated_working_directory(provenance: LocalProvenance) -> Path:
    """Create an empty, non-writable CWD for both public child processes."""

    working_directory = provenance.launch_receipt_path.parent / "workdir"
    working_directory.mkdir(mode=0o500)
    if working_directory.is_symlink() or not working_directory.is_dir():
        raise RuntimeError("public child working directory is not a real directory")
    if any(working_directory.iterdir()):
        raise RuntimeError("public child working directory must start empty")
    working_directory.chmod(0o500)
    mode = stat.S_IMODE(working_directory.stat().st_mode)
    if mode != 0o500:
        raise RuntimeError("unable to make public child working directory read-only")
    return working_directory.resolve(strict=True)


def _verify_requested_empty_embedding(request: Track31TrainRequest) -> None:
    path = request.paths.empty_embedding
    if path.is_symlink() or not path.is_file():
        raise ValueError("paths.empty_embedding must be a regular non-symlink file")
    if sha256_file(path) != request.paths.empty_embedding_sha256:
        raise ValueError("paths.empty_embedding_sha256 does not match the file")


def run_track31_training(
    request: Track31TrainRequest,
    *,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run preflight, training, and strict completion verification synchronously."""

    plan = build_launch_plan(request)
    if dry_run:
        return {"status": "dry_run", "plan": plan}
    _verify_requested_empty_embedding(request)
    output_root = request.paths.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    request_sha = sha256_file(request.source_path)
    provenance = prepare_local_provenance(
        output_root=output_root,
        run_id=request.run_id,
        request_sha256=request_sha,
        launch_plan=plan,
    )
    environment = build_training_environment(request, provenance, environ=environ)
    working_directory = _prepare_isolated_working_directory(provenance)
    logs_root = output_root / "logs"
    preflight_log = logs_root / f"preflight.{provenance.invocation_id}.log"
    training_log = logs_root / f"train.{provenance.invocation_id}.log"
    _run_preflight(
        command=build_preflight_command(),
        environment=environment,
        log_path=preflight_log,
        working_directory=working_directory,
    )
    _run_training_process(
        command=build_training_command(request),
        environment=environment,
        log_path=training_log,
        working_directory=working_directory,
    )
    checkpoint, checkpoint_identity = _verify_completed_checkpoint(request, provenance)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "execution_tier": "local_package",
        "formal_track31": False,
        "leaderboard_eligible": False,
        "invocation_id": provenance.invocation_id,
        "request_sha256": request_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_identity": checkpoint_identity,
        "preflight_log": str(preflight_log.resolve(strict=True)),
        "training_log": str(training_log.resolve(strict=True)),
        "launch_receipt": str(provenance.launch_receipt_path.resolve(strict=True)),
    }
    receipt_path = (
        output_root
        / "training_receipts"
        / f"training_receipt.{provenance.invocation_id}.json"
    )
    receipt_sha = write_immutable_json(receipt_path, receipt)
    return {
        **receipt,
        "receipt": str(receipt_path.resolve(strict=True)),
        "receipt_sha256": receipt_sha,
    }


def format_result(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


__all__ = (
    "build_launch_plan",
    "build_preflight_command",
    "build_training_command",
    "build_training_environment",
    "format_result",
    "run_track31_training",
)
