# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Single-command synchronous runner for AgileX qpos14 post-training."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from n0_twam.embodiments import AGILEX_ACTION_SCHEMA
from n0_twam.track31.local_provenance import (
    LocalProvenance,
    package_import_root,
    prepare_local_provenance,
    write_immutable_json,
)

from .request import AgileXTrainRequest, require_agilex_request_unchanged

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
_INTERFACE_ENV = frozenset({"GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME"})


def _profile_id(profile: str) -> str:
    return f"agilex_track3_{profile}_v1"


def _artifact_identity(request: AgileXTrainRequest) -> dict[str, object]:
    paths = request.paths
    return {
        "schema_version": 1,
        "embodiment_profile_id": "agilex_dual_qpos14_v1",
        "action_schema": AGILEX_ACTION_SCHEMA,
        "tactile_profile": request.profile,
        "source_manifest_file_sha256": paths.source_manifest_sha256,
        "conversion_receipt_file_sha256": paths.conversion_receipt_sha256,
        "latent_inventory_file_sha256": paths.latent_inventory_sha256,
        "repo_route_manifest_file_sha256": paths.repo_route_manifest_sha256,
        "temporal_alignment_file_sha256": paths.temporal_alignment_sha256,
        "normalizer_file_sha256": paths.normalizer_sha256,
    }


def _base_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {
        key: value
        for key, value in source.items()
        if not key.startswith("N0_")
        and not key.startswith("PYTHON")
        and key not in _VISIBILITY_ENV
        and key not in _INJECTION_ENV
        and key not in _INTERFACE_ENV
    }


def _available_network_interfaces() -> frozenset[str]:
    try:
        return frozenset(name for _, name in socket.if_nameindex())
    except OSError as error:
        raise RuntimeError("unable to enumerate collective interfaces") from error


def _collective_interface(request: AgileXTrainRequest) -> str | None:
    interface = request.runtime.collective_network_interface
    if request.runtime.accelerator_profile == "portable":
        if interface is not None:
            raise ValueError("portable training may not bind an HCU interface")
        return None
    if interface is None or interface not in _available_network_interfaces():
        raise ValueError(f"collective interface is unavailable: {interface}")
    return interface


def build_preflight_command() -> tuple[str, ...]:
    return (sys.executable, "-m", "n0_twam.track32_agilex.preflight")


def build_training_command(request: AgileXTrainRequest) -> tuple[str, ...]:
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
        f"track3_agilex_{request.profile}",
    )


def _recipe(request: AgileXTrainRequest) -> dict[str, object]:
    recipe = request.train
    return {
        "run_role": recipe.run_role,
        "num_steps": recipe.num_steps,
        "stop_after_step": recipe.stop_after_step,
        "save_interval": recipe.save_interval,
        "val_interval": recipe.val_interval,
        "batch_size": recipe.batch_size,
        "gradient_accumulation_steps": recipe.gradient_accumulation_steps,
        "max_latent_frames": recipe.max_latent_frames,
        "seed": recipe.seed,
    }


def build_launch_plan(request: AgileXTrainRequest) -> dict[str, object]:
    if request.profile == "mixed" and request.train.batch_size != 1:
        raise ValueError("mixed AgileX training requires batch_size=1")
    paths = request.paths
    return {
        "schema_version": 1,
        "execution_tier": "local_package",
        "leaderboard_evaluation_completed": False,
        "run_id": request.run_id,
        "profile": _profile_id(request.profile),
        "world_size": len(request.runtime.devices),
        "devices": list(request.runtime.devices),
        "accelerator_profile": request.runtime.accelerator_profile,
        "collective_network_interface": request.runtime.collective_network_interface,
        "request": str(request.source_path),
        "request_sha256": request.source_sha256,
        "empty_embedding_sha256": paths.empty_embedding_sha256,
        "init_transformer_sha256": paths.init_transformer_sha256,
        "resume_checkpoint_identity_sha256": (paths.resume_checkpoint_identity_sha256),
        "paths": {
            field: None if value is None else str(value)
            for field, value in vars(paths).items()
            if not field.endswith("sha256")
        },
        "artifact_identity": _artifact_identity(request),
        "preflight_command": list(build_preflight_command()),
        "training_command": list(build_training_command(request)),
        "recipe": _recipe(request),
        "model_contract": {
            "embodiment": "agilex_dual_qpos14_v1",
            "wire_action_schema": AGILEX_ACTION_SCHEMA,
            "model_action_schema": AGILEX_ACTION_SCHEMA,
            "active_action_channels": list(range(14)),
            "tactile_profile": request.profile,
            "contact_topology_preserved": True,
        },
        "formal_training_requested": (
            request.train.run_role == "final_refit"
            and request.train.stop_after_step == request.train.num_steps
        ),
    }


def _request_environment(
    request: AgileXTrainRequest,
    *,
    environ: Mapping[str, str] | None,
) -> dict[str, str]:
    environment = _base_environment(environ)
    interface = _collective_interface(request)
    paths, recipe = request.paths, request.train
    artifact_json = json.dumps(
        _artifact_identity(request), sort_keys=True, separators=(",", ":")
    )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, request.runtime.devices)),
            "HIP_VISIBLE_DEVICES": ",".join(map(str, request.runtime.devices)),
            "PYTHONHASHSEED": str(recipe.seed),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(package_import_root()),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "N0_TRACK3_AGILEX_REQUEST": str(request.source_path),
            "N0_TRACK3_AGILEX_REQUEST_SHA256": request.source_sha256,
            "N0_TRACK3_AGILEX_TACTILE_PROFILE": request.profile,
            "N0_TRACK3_AGILEX_PROFILE_ID": _profile_id(request.profile),
            "N0_TRACK3_AGILEX_ARTIFACT_IDENTITY_JSON": artifact_json,
            "N0_TRACK3_AGILEX_SOURCE_ROOT": str(paths.source_root),
            "N0_TRACK3_AGILEX_SOURCE_MANIFEST": str(paths.source_manifest),
            "N0_TRACK3_AGILEX_SOURCE_MANIFEST_SHA256": paths.source_manifest_sha256,
            "N0_TRACK3_AGILEX_DATASET_ROOT": str(paths.dataset_root),
            "N0_TRACK3_AGILEX_ARTIFACT_ROOT": str(paths.artifact_root),
            "N0_TRACK3_AGILEX_CONVERSION_RECEIPT": str(paths.conversion_receipt),
            "N0_TRACK3_AGILEX_LATENT_INVENTORY": str(paths.latent_inventory),
            "N0_TRACK3_AGILEX_REPO_ROUTE_MANIFEST": str(paths.repo_route_manifest),
            "N0_TRACK3_AGILEX_TEMPORAL_ALIGNMENT": str(paths.temporal_alignment),
            "N0_TRACK3_AGILEX_NORMALIZER": str(paths.normalizer),
            "N0_BASE_MODEL": str(paths.base_model),
            "N0_EMPTY_EMBEDDING": str(paths.empty_embedding),
            "N0_EMPTY_EMBEDDING_SHA256": paths.empty_embedding_sha256,
            "N0_TRACK3_AGILEX_SAVE_ROOT": str(paths.output_root),
            "N0_TRACK3_AGILEX_RUN_ROLE": recipe.run_role,
            "N0_TRACK3_AGILEX_ACCELERATOR_PROFILE": (
                request.runtime.accelerator_profile
            ),
            "N0_TRACK3_AGILEX_EXPECTED_WORLD_SIZE": str(len(request.runtime.devices)),
            "N0_TRACK3_AGILEX_NUM_STEPS": str(recipe.num_steps),
            "N0_TRACK3_AGILEX_STOP_AFTER_STEP": str(recipe.stop_after_step),
            "N0_TRACK3_AGILEX_SAVE_INTERVAL": str(recipe.save_interval),
            "N0_TRACK3_AGILEX_VAL_INTERVAL": str(recipe.val_interval),
            "N0_TRACK3_AGILEX_BATCH_SIZE": str(recipe.batch_size),
            "N0_TRACK3_AGILEX_GRADIENT_ACCUMULATION_STEPS": str(
                recipe.gradient_accumulation_steps
            ),
            "N0_TRACK3_AGILEX_MAX_LATENT_FRAMES": str(recipe.max_latent_frames),
            "N0_TRACK3_AGILEX_LOAD_WORKER": "0",
            "N0_TRACK3_AGILEX_SEED": str(recipe.seed),
            "N0_SYNC_ATTENTION_WINDOW": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
    if paths.init_from is not None:
        environment["N0_TRACK3_AGILEX_INIT_FROM"] = str(paths.init_from)
        environment["N0_RELEASED_TRANSFORMER_SHA256"] = str(
            paths.init_transformer_sha256
        )
    else:
        environment["N0_TRACK3_AGILEX_RESUME_FROM"] = str(paths.resume_from)
        environment["N0_TRACK3_AGILEX_RESUME_CHECKPOINT_IDENTITY_SHA256"] = str(
            paths.resume_checkpoint_identity_sha256
        )
    if request.runtime.accelerator_profile == "hcu_performance":
        assert interface is not None
        environment.update(
            {
                "GLOO_SOCKET_IFNAME": interface,
                "NCCL_SOCKET_IFNAME": interface,
                "N0_TRACK3_AGILEX_COLLECTIVE_NETWORK_INTERFACE": interface,
                "N0_FLEX_ATTENTION_BACKEND": "grouped_flash_attn",
                "N0_MOT_CROSS_ATTENTION_BACKEND": "flash_attn",
                "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "1",
                "N0_FSDP_REDUCE_DTYPE": "bfloat16",
                "N0_FSDP_EXPERT_RESHARD_POLICY": "after_layer",
                "N0_MOT_ACTIVATION_CHECKPOINTING": "0",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                "TORCH_NCCL_BLOCKING_WAIT": "1",
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
            }
        )
    return environment


def build_training_environment(
    request: AgileXTrainRequest,
    provenance: LocalProvenance,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = _request_environment(request, environ=environ)
    environment.update(provenance.environment())
    return environment


def build_agilex_request_environment(
    request: AgileXTrainRequest,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Expose the canonical request environment to isolated evaluation CLIs."""

    return _request_environment(request, environ=environ)


def _run_capture(command: Sequence[str], environment: Mapping[str, str]) -> str:
    result = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        env=dict(environment),
        cwd=package_import_root(),
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"AgileX preflight failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def _run_training(
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
    working_directory: Path,
) -> None:
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
        raise RuntimeError(f"AgileX training failed ({return_code}); see {log_path}")


def _working_directory(provenance: LocalProvenance) -> Path:
    path = provenance.launch_receipt_path.parent / "workdir"
    path.mkdir(mode=0o500)
    path.chmod(0o500)
    if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
        raise RuntimeError("AgileX child workdir must be a new empty directory")
    return path.resolve(strict=True)


def _claim_output_root(path: Path) -> Path:
    """Atomically claim a new output lineage after preflight succeeds."""

    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
        raise RuntimeError("AgileX output_root claim is not a new empty directory")
    return path.resolve(strict=True)


def run_agilex_training(
    request: AgileXTrainRequest,
    *,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    require_agilex_request_unchanged(request)
    plan = build_launch_plan(request)
    if dry_run:
        return {"status": "dry_run", "plan": plan}
    if request.paths.output_root.exists():
        raise FileExistsError(
            f"AgileX output_root must be new: {request.paths.output_root}"
        )
    preflight_environment = _request_environment(request, environ=environ)
    preflight_output = _run_capture(build_preflight_command(), preflight_environment)
    preflight_payload = json.loads(preflight_output)
    if (
        not isinstance(preflight_payload, dict)
        or preflight_payload.get("status") != "pass"
        or preflight_payload.get("request_sha256") != request.source_sha256
    ):
        raise RuntimeError("AgileX preflight did not return a pass receipt")
    require_agilex_request_unchanged(request)
    _claim_output_root(request.paths.output_root)
    provenance = prepare_local_provenance(
        output_root=request.paths.output_root,
        run_id=request.run_id,
        request_sha256=request.source_sha256,
        launch_plan=plan,
    )
    logs = request.paths.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    preflight_log = logs / f"preflight.{provenance.invocation_id}.json"
    write_immutable_json(preflight_log, preflight_payload)
    training_log = logs / f"train.{provenance.invocation_id}.log"
    environment = build_training_environment(request, provenance, environ=environ)
    require_agilex_request_unchanged(request)
    _run_training(
        build_training_command(request),
        environment,
        training_log,
        _working_directory(provenance),
    )
    require_agilex_request_unchanged(request)
    from .completion import verify_completed_checkpoint

    checkpoint, identity = verify_completed_checkpoint(request, provenance)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "formal_training_completed": (
            request.train.run_role == "final_refit"
            and request.train.stop_after_step == request.train.num_steps
        ),
        "leaderboard_evaluation_completed": False,
        "invocation_id": provenance.invocation_id,
        "checkpoint": str(checkpoint),
        "checkpoint_identity": identity,
        "preflight_log": str(preflight_log),
        "training_log": str(training_log.resolve(strict=True)),
        "launch_receipt": str(provenance.launch_receipt_path),
    }
    receipt_path = (
        request.paths.output_root
        / "training_receipts"
        / f"{provenance.invocation_id}.json"
    )
    receipt_sha256 = write_immutable_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": receipt_sha256}


__all__ = (
    "build_agilex_request_environment",
    "build_launch_plan",
    "build_preflight_command",
    "build_training_command",
    "build_training_environment",
    "run_agilex_training",
)
