# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Single-command synchronous runner for Franka Track 3.2 post-training."""

from __future__ import annotations

import os
import socket
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.integrations.worldarena.franka_manifest import (
    OFFICIAL_RECORDS_SHA256,
)
from n0_twam.track31.local_provenance import (
    LocalProvenance,
    package_import_root,
    prepare_local_provenance,
    sha256_file,
    write_immutable_json,
)

from .request import Track32TrainRequest

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
_INJECTION_ENV = frozenset({"DYLD_INSERT_LIBRARIES", "LD_AUDIT", "LD_PRELOAD"})
_DISTRIBUTED_INTERFACE_ENV = frozenset({"GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME"})


def _base_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {
        key: value
        for key, value in source.items()
        if not key.startswith("N0_")
        and not key.startswith("PYTHON")
        and key not in _VISIBILITY_ENV
        and key not in _INJECTION_ENV
        and key not in _DISTRIBUTED_INTERFACE_ENV
    }


def _available_network_interfaces() -> frozenset[str]:
    try:
        return frozenset(name for _, name in socket.if_nameindex())
    except OSError as error:
        raise RuntimeError(
            "unable to enumerate collective network interfaces"
        ) from error


def _require_collective_network_interface(
    request: Track32TrainRequest,
) -> str | None:
    interface = request.runtime.collective_network_interface
    if request.runtime.accelerator_profile == "portable":
        if interface is not None:
            raise ValueError("portable training may not bind an HCU interface")
        return None
    if interface is None:
        raise ValueError("hcu_performance requires a collective network interface")
    available = _available_network_interfaces()
    if interface not in available:
        raise ValueError(
            "runtime.collective_network_interface is unavailable in this "
            f"container: {interface}; available={sorted(available)}"
        )
    return interface


def build_preflight_command() -> tuple[str, ...]:
    return (sys.executable, "-m", "n0_twam.track32.preflight")


def build_training_command(request: Track32TrainRequest) -> tuple[str, ...]:
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
        "track32_franka",
    )


def _artifact_identity(request: Track32TrainRequest) -> dict[str, object]:
    validation_id = (
        "franka_dev_validation60_v1"
        if request.train.run_role == "development"
        else None
    )
    return {
        "schema_version": 1,
        "profile": "franka_track32_vision_only_v1",
        "run_role": request.train.run_role,
        "source_records_sha256": OFFICIAL_RECORDS_SHA256,
        "train_view_id": (
            "franka_dev_train540_v1"
            if request.train.run_role == "development"
            else "franka_final_refit600_v1"
        ),
        "validation_view_id": validation_id,
        "validation_view_sha256": request.artifacts.validation_view_sha256,
        "prepare_receipt_file_sha256": (request.artifacts.prepare_receipt_sha256),
        "conversion_report_file_sha256": (request.artifacts.conversion_report_sha256),
        "latent_inventory_file_sha256": (
            request.artifacts.latent_inventory_file_sha256
        ),
        "train_view_sha256": request.artifacts.train_view_sha256,
        "normalizer_sha256": request.artifacts.normalizer_sha256,
    }


def build_launch_plan(request: Track32TrainRequest) -> dict[str, object]:
    """Build the secret-free plan emitted by ``--dry-run``."""

    paths = request.paths
    recipe = request.train
    return {
        "schema_version": 1,
        "execution_tier": "local_package",
        "formal_track32_training_requested": (
            recipe.run_role == "final_refit"
            and recipe.stop_after_step == recipe.num_steps
        ),
        "leaderboard_evaluation_completed": False,
        "run_id": request.run_id,
        "world_size": len(request.runtime.devices),
        "devices": list(request.runtime.devices),
        "accelerator_profile": request.runtime.accelerator_profile,
        "collective_network_interface": (request.runtime.collective_network_interface),
        "request": str(request.source_path),
        "paths": {
            "artifact_root": str(paths.artifact_root),
            "lerobot_root": str(paths.lerobot_root),
            "base_model": str(paths.base_model),
            "empty_embedding": str(paths.empty_embedding),
            "init_from": None if paths.init_from is None else str(paths.init_from),
            "resume_from": (
                None if paths.resume_from is None else str(paths.resume_from)
            ),
            "output_root": str(paths.output_root),
        },
        "empty_embedding_sha256": paths.empty_embedding_sha256,
        "init_transformer_sha256": paths.init_transformer_sha256,
        "resume_checkpoint_identity_sha256": (paths.resume_checkpoint_identity_sha256),
        "artifact_identity": _artifact_identity(request),
        "preflight_command": list(build_preflight_command()),
        "training_command": list(build_training_command(request)),
        "recipe": {
            "run_role": recipe.run_role,
            "num_steps": recipe.num_steps,
            "stop_after_step": recipe.stop_after_step,
            "save_interval": recipe.save_interval,
            "val_interval": recipe.val_interval,
            "batch_size": recipe.batch_size,
            "gradient_accumulation_steps": recipe.gradient_accumulation_steps,
            "max_latent_frames": recipe.max_latent_frames,
            "seed": recipe.seed,
        },
        "model_contract": {
            "wire_action_schema": "franka_end_pose_base_wxyz8_v1",
            "derived_action_schema": "franka_ee10_rot6d_columns_v1",
            "model_action_schema": "ee20_absee",
            "active_action_channels": list(range(10)),
            "tactile_profile": "vision_only",
            "tactile_mode": "disabled",
            "frozen_tactile_parameters": True,
        },
    }


def build_training_environment(
    request: Track32TrainRequest,
    provenance: LocalProvenance,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Map a request to the sole clean environment used by both children."""

    environment = _base_environment(environ)
    collective_interface = _require_collective_network_interface(request)
    devices = ",".join(str(device) for device in request.runtime.devices)
    paths = request.paths
    artifacts = request.artifacts
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
            "N0_TRACK32_ARTIFACT_ROOT": str(paths.artifact_root),
            "N0_TRACK32_LEROBOT_ROOT": str(paths.lerobot_root),
            "N0_BASE_MODEL": str(paths.base_model),
            "N0_EMPTY_EMBEDDING": str(paths.empty_embedding),
            "N0_EMPTY_EMBEDDING_SHA256": paths.empty_embedding_sha256,
            "N0_TRACK32_SAVE_ROOT": str(paths.output_root),
            "N0_TRACK32_RUN_ROLE": recipe.run_role,
            "N0_TRACK32_ACCELERATOR_PROFILE": (request.runtime.accelerator_profile),
            "N0_TRACK32_EXPECTED_WORLD_SIZE": str(len(request.runtime.devices)),
            "N0_TRACK32_NUM_STEPS": str(recipe.num_steps),
            "N0_TRACK32_STOP_AFTER_STEP": str(recipe.stop_after_step),
            "N0_TRACK32_SAVE_INTERVAL": str(recipe.save_interval),
            "N0_TRACK32_VAL_INTERVAL": str(recipe.val_interval),
            "N0_TRACK32_BATCH_SIZE": str(recipe.batch_size),
            "N0_TRACK32_GRADIENT_ACCUMULATION_STEPS": str(
                recipe.gradient_accumulation_steps
            ),
            "N0_TRACK32_MAX_LATENT_FRAMES": str(recipe.max_latent_frames),
            "N0_TRACK32_LOAD_WORKER": "0",
            "N0_TRACK32_SEED": str(recipe.seed),
            "N0_TRACK32_PREPARE_RECEIPT_SHA256": (artifacts.prepare_receipt_sha256),
            "N0_TRACK32_CONVERSION_REPORT_SHA256": (artifacts.conversion_report_sha256),
            "N0_TRACK32_LATENT_INVENTORY_FILE_SHA256": (
                artifacts.latent_inventory_file_sha256
            ),
            "N0_TRACK32_TRAIN_VIEW_SHA256": artifacts.train_view_sha256,
            "N0_TRACK32_NORMALIZER_SHA256": artifacts.normalizer_sha256,
            **provenance.environment(),
        }
    )
    if request.runtime.accelerator_profile == "hcu_performance":
        assert collective_interface is not None
        environment.update(
            {
                "GLOO_SOCKET_IFNAME": collective_interface,
                "NCCL_SOCKET_IFNAME": collective_interface,
                "N0_TRACK32_COLLECTIVE_NETWORK_INTERFACE": collective_interface,
                "N0_FLEX_ATTENTION_BACKEND": "grouped_flash_attn",
                "N0_MOT_CROSS_ATTENTION_BACKEND": "flash_attn",
                "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "1",
                "N0_FSDP_REDUCE_DTYPE": "bfloat16",
                "N0_FSDP_EXPERT_RESHARD_POLICY": "after_layer",
                "N0_MOT_ACTIVATION_CHECKPOINTING": "0",
                "N0_SYNC_ATTENTION_WINDOW": "1",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                "TORCH_NCCL_BLOCKING_WAIT": "1",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
            }
        )
    else:
        environment.update(
            {
                "N0_FLEX_ATTENTION_BACKEND": "grouped_sdpa",
                "N0_MOT_CROSS_ATTENTION_BACKEND": "sdpa",
                "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "0",
                "N0_FSDP_REDUCE_DTYPE": "float32",
                "N0_FSDP_EXPERT_RESHARD_POLICY": "after_call",
                "N0_MOT_ACTIVATION_CHECKPOINTING": "1",
                "N0_SYNC_ATTENTION_WINDOW": "1",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
            }
        )
    if artifacts.validation_view_sha256 is not None:
        environment["N0_TRACK32_VALIDATION_VIEW_SHA256"] = (
            artifacts.validation_view_sha256
        )
    if paths.init_from is not None:
        environment["N0_TRACK32_INIT_FROM"] = str(paths.init_from)
        environment["N0_TRACK32_INIT_TRANSFORMER_SHA256"] = str(
            paths.init_transformer_sha256
        )
    if paths.resume_from is not None:
        environment["N0_TRACK32_RESUME_FROM"] = str(paths.resume_from)
        environment["N0_TRACK32_RESUME_CHECKPOINT_IDENTITY_SHA256"] = str(
            paths.resume_checkpoint_identity_sha256
        )
    return environment


def _prepare_isolated_working_directory(provenance: LocalProvenance) -> Path:
    working_directory = provenance.launch_receipt_path.parent / "workdir"
    working_directory.mkdir(mode=0o500)
    if working_directory.is_symlink() or not working_directory.is_dir():
        raise RuntimeError("child working directory must be a real directory")
    if any(working_directory.iterdir()):
        raise RuntimeError("child working directory must start empty")
    working_directory.chmod(0o500)
    if stat.S_IMODE(working_directory.stat().st_mode) != 0o500:
        raise RuntimeError("unable to make child working directory read-only")
    return working_directory.resolve(strict=True)


def _run_command(
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
    working_directory: Path,
    stream: bool,
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
            if stream:
                sys.stderr.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Track 3.2 command failed ({return_code}); see {log_path}")


def _verify_requested_empty_embedding(request: Track32TrainRequest) -> None:
    path = request.paths.empty_embedding
    if path.is_symlink() or not path.is_file():
        raise ValueError("paths.empty_embedding must be a regular non-symlink file")
    if sha256_file(path) != request.paths.empty_embedding_sha256:
        raise ValueError("paths.empty_embedding_sha256 does not match the file")


def _verify_completed_checkpoint(
    request: Track32TrainRequest,
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
    if snapshot.transformer_config.get("action_schema") != "ee20_absee":
        raise ValueError("completed checkpoint action schema is not ee20_absee")
    meta = snapshot.train_meta
    if meta.get("track32_artifact_identity") != _artifact_identity(request):
        raise ValueError("completed checkpoint artifact identity differs from request")
    expected_meta = {
        "track32_profile_id": "franka_track32_vision_only_v1",
        "source_action_schema": "franka_end_pose_base_wxyz8_v1",
        "derived_action_schema": "franka_ee10_rot6d_columns_v1",
        "tactile_profile": "vision_only",
        "tactile_mode": "disabled",
        "tactile_keys": [],
        "used_action_channel_ids": list(range(10)),
        "accelerator_profile": request.runtime.accelerator_profile,
    }
    if any(meta.get(field) != value for field, value in expected_meta.items()):
        raise ValueError("completed checkpoint Franka metadata is incompatible")
    trainability = meta.get("trainability_contract")
    if not isinstance(trainability, Mapping) or (
        trainability.get("policy") != "freeze_tactile_only_v1"
        or trainability.get("tactile_mode") != "disabled"
        or trainability.get("tactile_profile", "vision_only") != "vision_only"
    ):
        raise ValueError("completed checkpoint lacks the tactile freeze contract")
    return checkpoint.resolve(strict=True), build_strict_checkpoint_identity(snapshot)


def run_track32_training(
    request: Track32TrainRequest,
    *,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run preflight, post-training, and strict completion verification."""

    plan = build_launch_plan(request)
    if dry_run:
        return {"status": "dry_run", "plan": plan}
    _require_collective_network_interface(request)
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
    _run_command(
        command=build_preflight_command(),
        environment=environment,
        log_path=preflight_log,
        working_directory=working_directory,
        stream=False,
    )
    _run_command(
        command=build_training_command(request),
        environment=environment,
        log_path=training_log,
        working_directory=working_directory,
        stream=True,
    )
    checkpoint, checkpoint_identity = _verify_completed_checkpoint(request, provenance)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "execution_tier": "local_package",
        "formal_track32_training_completed": (
            request.train.run_role == "final_refit"
            and request.train.stop_after_step == request.train.num_steps
        ),
        "leaderboard_evaluation_completed": False,
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


__all__ = (
    "build_launch_plan",
    "build_preflight_command",
    "build_training_command",
    "build_training_environment",
    "run_track32_training",
)
