"""Strict sharded optimizer checkpoint helpers shared by training and smoke."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.optim import Optimizer

STRICT_CHECKPOINT_SCHEMA_VERSION = 6
OPTIMIZER_DCP_DIRNAME = "optimizer_dcp"
OPTIMIZER_INVENTORY_FILENAME = "optimizer_inventory.json"
OPTIMIZER_INVENTORY_SCHEMA_VERSION = 1
OPTIMIZER_STATE_FORMAT = "dcp_sharded_v1"
RUNTIME_SIGNATURE_SCHEMA_VERSION = 2
TRAINING_EXECUTION_CONTRACT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class OptimizerCheckpointFile:
    """One immutable DCP payload entry."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class OptimizerCheckpointInventory:
    """Exact inventory binding every DCP byte to a completion marker."""

    schema_version: int
    state_format: str
    files: tuple[OptimizerCheckpointFile, ...]
    inventory_sha256: str

    def to_json_dict(self) -> dict[str, object]:
        """Return the stable JSON representation stored on disk."""
        return {
            "schema_version": self.schema_version,
            "state_format": self.state_format,
            "files": [asdict(file_entry) for file_entry in self.files],
            "inventory_sha256": self.inventory_sha256,
        }


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_core(
    files: tuple[OptimizerCheckpointFile, ...],
) -> dict[str, object]:
    return {
        "schema_version": OPTIMIZER_INVENTORY_SCHEMA_VERSION,
        "state_format": OPTIMIZER_STATE_FORMAT,
        "files": [asdict(file_entry) for file_entry in files],
    }


def _discover_dcp_payload(
    checkpoint_dir: Path,
) -> tuple[OptimizerCheckpointFile, ...]:
    metadata_path = checkpoint_dir / ".metadata"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise FileNotFoundError(
            f"optimizer checkpoint is missing regular DCP metadata: {metadata_path}"
        )
    shard_paths = sorted(checkpoint_dir.glob("*.distcp"))
    if not shard_paths:
        raise FileNotFoundError(
            f"optimizer checkpoint has no DCP shards: {checkpoint_dir}"
        )
    payload_paths = (metadata_path, *shard_paths)
    entries: list[OptimizerCheckpointFile] = []
    for path in payload_paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"DCP payload is not a regular file: {path}")
        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            raise ValueError(f"DCP payload is empty: {path}")
        entries.append(
            OptimizerCheckpointFile(
                path=path.name,
                size_bytes=size_bytes,
                sha256=_sha256_file(path),
            )
        )
    return tuple(entries)


def write_optimizer_checkpoint_inventory(
    checkpoint_dir: Path,
) -> OptimizerCheckpointInventory:
    """Hash all DCP payloads and atomically write their exact inventory."""
    checkpoint_dir = Path(checkpoint_dir)
    files = _discover_dcp_payload(checkpoint_dir)
    payload_names = {file_entry.path for file_entry in files}
    actual_names = {path.name for path in checkpoint_dir.iterdir()}
    if actual_names != payload_names:
        raise ValueError(
            "optimizer checkpoint contains files outside the DCP payload: "
            f"{sorted(actual_names - payload_names)}"
        )
    inventory_sha256 = hashlib.sha256(
        _canonical_json_bytes(_inventory_core(files))
    ).hexdigest()
    inventory = OptimizerCheckpointInventory(
        schema_version=OPTIMIZER_INVENTORY_SCHEMA_VERSION,
        state_format=OPTIMIZER_STATE_FORMAT,
        files=files,
        inventory_sha256=inventory_sha256,
    )
    inventory_path = checkpoint_dir / OPTIMIZER_INVENTORY_FILENAME
    temporary_path = checkpoint_dir / f".{OPTIMIZER_INVENTORY_FILENAME}.tmp"
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(inventory.to_json_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, inventory_path)
    return inventory


def _parse_inventory(payload: object) -> OptimizerCheckpointInventory:
    if not isinstance(payload, dict):
        raise ValueError("optimizer inventory must be a JSON object")
    if int(payload.get("schema_version", 0)) != OPTIMIZER_INVENTORY_SCHEMA_VERSION:
        raise ValueError("unsupported optimizer inventory schema")
    if payload.get("state_format") != OPTIMIZER_STATE_FORMAT:
        raise ValueError("unsupported optimizer inventory state format")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("optimizer inventory has no files")
    files: list[OptimizerCheckpointFile] = []
    seen_paths: set[str] = set()
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            raise ValueError("optimizer inventory file entry must be an object")
        path = str(raw_file.get("path", ""))
        if not path or Path(path).name != path or path in seen_paths:
            raise ValueError(f"invalid optimizer inventory path: {path!r}")
        seen_paths.add(path)
        size_bytes = int(raw_file.get("size_bytes", 0))
        sha256 = str(raw_file.get("sha256", ""))
        if (
            size_bytes <= 0
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"invalid optimizer inventory entry: {path}")
        files.append(
            OptimizerCheckpointFile(
                path=path,
                size_bytes=size_bytes,
                sha256=sha256,
            )
        )
    file_tuple = tuple(files)
    expected_digest = hashlib.sha256(
        _canonical_json_bytes(_inventory_core(file_tuple))
    ).hexdigest()
    recorded_digest = str(payload.get("inventory_sha256", ""))
    if recorded_digest != expected_digest:
        raise ValueError("optimizer inventory digest is inconsistent")
    return OptimizerCheckpointInventory(
        schema_version=OPTIMIZER_INVENTORY_SCHEMA_VERSION,
        state_format=OPTIMIZER_STATE_FORMAT,
        files=file_tuple,
        inventory_sha256=recorded_digest,
    )


def validate_optimizer_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_inventory_sha256: str | None = None,
) -> OptimizerCheckpointInventory:
    """Verify the exact file set, sizes, hashes, and marker-bound digest."""
    checkpoint_dir = Path(checkpoint_dir)
    inventory_path = checkpoint_dir / OPTIMIZER_INVENTORY_FILENAME
    if not inventory_path.is_file() or inventory_path.is_symlink():
        raise FileNotFoundError(
            f"optimizer checkpoint is missing inventory: {inventory_path}"
        )
    inventory = _parse_inventory(json.loads(inventory_path.read_text(encoding="utf-8")))
    if (
        expected_inventory_sha256 is not None
        and inventory.inventory_sha256 != expected_inventory_sha256
    ):
        raise ValueError("optimizer inventory does not match completion marker")

    expected_names = {
        OPTIMIZER_INVENTORY_FILENAME,
        *(file_entry.path for file_entry in inventory.files),
    }
    actual_names = {path.name for path in checkpoint_dir.iterdir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            "optimizer checkpoint file inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if sum(file_entry.path == ".metadata" for file_entry in inventory.files) != 1:
        raise ValueError("optimizer inventory must contain exactly one .metadata")
    if not any(file_entry.path.endswith(".distcp") for file_entry in inventory.files):
        raise ValueError("optimizer inventory has no DCP shards")
    if any(
        file_entry.path != ".metadata" and not file_entry.path.endswith(".distcp")
        for file_entry in inventory.files
    ):
        raise ValueError("optimizer inventory contains a non-DCP payload")

    for file_entry in inventory.files:
        path = checkpoint_dir / file_entry.path
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"DCP inventory entry is not a regular file: {path}")
        if path.stat().st_size != file_entry.size_bytes:
            raise ValueError(f"DCP payload size mismatch: {path}")
        if _sha256_file(path) != file_entry.sha256:
            raise ValueError(f"DCP payload SHA256 mismatch: {path}")
    return inventory


def capture_runtime_signature() -> dict[str, object]:
    """Capture runtime fields that affect FSDP2/DCP checkpoint compatibility."""
    from n0_twam.distributed.fsdp import (
        FSDP2_API_SOURCE,
        capture_fsdp_execution_contract,
    )

    cuda_available = torch.cuda.is_available()
    vendor_device = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if cuda_available
        else "cpu"
    )
    return {
        "schema_version": RUNTIME_SIGNATURE_SCHEMA_VERSION,
        "torch_version": str(torch.__version__),
        "fsdp2_api_source": FSDP2_API_SOURCE,
        "accelerator_api": "cuda" if cuda_available else "cpu",
        "vendor_device": str(vendor_device),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "torch_hip_version": str(getattr(torch.version, "hip", None) or ""),
        "fsdp_execution_contract": capture_fsdp_execution_contract(),
    }


def build_training_execution_contract(
    *,
    max_latent_frames: int,
    gradient_accumulation_steps: int,
    batch_size: int,
    load_worker: int,
    num_steps: int,
    lr_schedule: str,
    warmup_steps: int,
    lr_min_ratio: float,
    activation_checkpointing: bool,
    attention_contract: Mapping[str, object],
) -> dict[str, object]:
    """Build the strict sequence-length and attention execution contract."""
    if isinstance(max_latent_frames, bool) or not isinstance(max_latent_frames, int):
        raise TypeError("max_latent_frames must be an integer")
    if max_latent_frames < 0:
        raise ValueError("max_latent_frames must be non-negative")
    if (
        isinstance(gradient_accumulation_steps, bool)
        or not isinstance(gradient_accumulation_steps, int)
        or gradient_accumulation_steps <= 0
    ):
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    for label, value in (("batch_size", batch_size), ("num_steps", num_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if (
        isinstance(load_worker, bool)
        or not isinstance(load_worker, int)
        or load_worker < 0
    ):
        raise ValueError("load_worker must be a non-negative integer")
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"unsupported lr_schedule: {lr_schedule!r}")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 0
    ):
        raise ValueError("warmup_steps must be a non-negative integer")
    if (
        isinstance(lr_min_ratio, bool)
        or not isinstance(lr_min_ratio, (int, float))
        or not math.isfinite(float(lr_min_ratio))
        or not 0.0 <= float(lr_min_ratio) <= 1.0
    ):
        raise ValueError("lr_min_ratio must be finite and in [0, 1]")
    if not isinstance(attention_contract, Mapping):
        raise TypeError("attention execution contract must be a mapping")
    if not isinstance(activation_checkpointing, bool):
        raise TypeError("activation_checkpointing must be a boolean")
    expected_attention_keys = {
        "attention_backend",
        "grouped_sdpa_max_query_tokens",
        "mot_cross_attention_backend",
    }
    if set(attention_contract) != expected_attention_keys:
        raise ValueError(
            "attention execution contract fields are incompatible: "
            f"{sorted(attention_contract)}"
        )
    attention_backend = attention_contract.get("attention_backend")
    if attention_backend not in ("flex", "grouped_sdpa", "grouped_flash_attn"):
        raise ValueError(
            f"unsupported attention execution backend: {attention_backend!r}"
        )
    grouped_token_limit = attention_contract.get("grouped_sdpa_max_query_tokens")
    mot_cross_attention_backend = attention_contract.get(
        "mot_cross_attention_backend"
    )
    if mot_cross_attention_backend not in ("sdpa", "flash_attn"):
        raise ValueError(
            "unsupported MoT cross-attention backend: "
            f"{mot_cross_attention_backend!r}"
        )
    if attention_backend in ("grouped_sdpa", "grouped_flash_attn"):
        if (
            isinstance(grouped_token_limit, bool)
            or not isinstance(grouped_token_limit, int)
            or grouped_token_limit <= 0
        ):
            raise ValueError(
                "grouped attention requires a positive grouped query-token limit"
            )
    elif grouped_token_limit is not None:
        raise ValueError("flex execution must not record a grouped query-token limit")
    return {
        "schema_version": TRAINING_EXECUTION_CONTRACT_SCHEMA_VERSION,
        "max_latent_frames": max_latent_frames,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "batch_size": batch_size,
        "load_worker": load_worker,
        "num_steps": num_steps,
        "lr_schedule": lr_schedule,
        "warmup_steps": warmup_steps,
        "lr_min_ratio": float(lr_min_ratio),
        "activation_checkpointing": activation_checkpointing,
        "attention_backend": attention_backend,
        "grouped_sdpa_max_query_tokens": grouped_token_limit,
        "mot_cross_attention_backend": mot_cross_attention_backend,
    }


def _parse_training_execution_contract(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError("training execution contract must be a mapping")
    expected_keys = {
        "schema_version",
        "max_latent_frames",
        "gradient_accumulation_steps",
        "batch_size",
        "load_worker",
        "num_steps",
        "lr_schedule",
        "warmup_steps",
        "lr_min_ratio",
        "activation_checkpointing",
        "attention_backend",
        "grouped_sdpa_max_query_tokens",
        "mot_cross_attention_backend",
    }
    if set(payload) != expected_keys:
        raise ValueError("training execution contract fields are incompatible")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != TRAINING_EXECUTION_CONTRACT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported training execution contract schema")
    return build_training_execution_contract(
        max_latent_frames=payload["max_latent_frames"],
        gradient_accumulation_steps=payload["gradient_accumulation_steps"],
        batch_size=payload["batch_size"],
        load_worker=payload["load_worker"],
        num_steps=payload["num_steps"],
        lr_schedule=payload["lr_schedule"],
        warmup_steps=payload["warmup_steps"],
        lr_min_ratio=payload["lr_min_ratio"],
        activation_checkpointing=payload["activation_checkpointing"],
        attention_contract={
            "attention_backend": payload["attention_backend"],
            "grouped_sdpa_max_query_tokens": payload["grouped_sdpa_max_query_tokens"],
            "mot_cross_attention_backend": payload[
                "mot_cross_attention_backend"
            ],
        },
    )


def validate_training_execution_contract(
    saved_contract: object,
    *,
    current_contract: Mapping[str, object],
) -> dict[str, object]:
    """Fail every rank when sequence or attention execution settings drift."""
    current = _parse_training_execution_contract(current_contract)
    saved: dict[str, object] | None
    try:
        saved = _parse_training_execution_contract(saved_contract)
    except (TypeError, ValueError):
        saved = None
    local_mismatch = saved != current
    if dist.is_initialized():
        rank_zero_contract: list[object] = [current if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(rank_zero_contract, src=0)
        local_mismatch = local_mismatch or current != rank_zero_contract[0]
        mismatch = torch.tensor(
            int(local_mismatch), dtype=torch.int32, device=_collective_device()
        )
        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
        local_mismatch = bool(mismatch.item())
    if local_mismatch:
        raise ValueError(
            "training execution contract mismatch: "
            f"saved={saved_contract!r}, current={current!r}"
        )
    return current


def _collective_device() -> torch.device:
    if dist.is_initialized() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def validate_runtime_signature(saved_signature: object) -> dict[str, object]:
    """Fail every rank if any current runtime differs from the saved signature."""
    current_signature = capture_runtime_signature()
    local_mismatch = saved_signature != current_signature
    if dist.is_initialized():
        rank_zero_signature: list[object] = [
            current_signature if dist.get_rank() == 0 else None
        ]
        dist.broadcast_object_list(rank_zero_signature, src=0)
        local_mismatch = local_mismatch or current_signature != rank_zero_signature[0]
        mismatch = torch.tensor(
            int(local_mismatch), dtype=torch.int32, device=_collective_device()
        )
        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
        local_mismatch = bool(mismatch.item())
    if local_mismatch:
        raise ValueError(
            "runtime signature mismatch: "
            f"saved={saved_signature!r}, current={current_signature!r}"
        )
    return current_signature


def validate_rng_state(rng_state: object) -> dict[str, object]:
    """Validate every RNG state with isolated generators before collectives."""
    required_keys = {"python", "numpy", "torch", "cuda"}
    if not isinstance(rng_state, dict) or not required_keys.issubset(rng_state):
        raise ValueError("resume RNG sidecar is incompatible")

    python_rng = random.Random()
    python_rng.setstate(rng_state["python"])
    numpy_rng = np.random.RandomState()
    numpy_rng.set_state(rng_state["numpy"])

    torch_state = rng_state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        raise ValueError("resume CPU torch RNG state is not a tensor")
    torch.Generator(device="cpu").set_state(torch_state)

    cuda_states = rng_state["cuda"]
    if not isinstance(cuda_states, (list, tuple)):
        raise ValueError("resume CUDA RNG states must be a sequence")
    expected_cuda_states = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_states) != expected_cuda_states:
        raise ValueError(
            "resume CUDA RNG state count mismatch: "
            f"{len(cuda_states)} vs {expected_cuda_states}"
        )
    for device_index, cuda_state in enumerate(cuda_states):
        if not isinstance(cuda_state, torch.Tensor):
            raise ValueError(f"resume CUDA RNG state {device_index} is not a tensor")
        torch.Generator(device=torch.device("cuda", device_index)).set_state(cuda_state)
    return rng_state


def _root_validate_optimizer_checkpoint(
    checkpoint_dir: Path,
    expected_inventory_sha256: str | None,
) -> OptimizerCheckpointInventory:
    if not dist.is_initialized():
        return validate_optimizer_checkpoint(
            checkpoint_dir,
            expected_inventory_sha256=expected_inventory_sha256,
        )
    rank = dist.get_rank()
    inventory: OptimizerCheckpointInventory | None = None
    stage_error: Exception | None = None
    if rank == 0:
        try:
            inventory = validate_optimizer_checkpoint(
                checkpoint_dir,
                expected_inventory_sha256=expected_inventory_sha256,
            )
        except Exception as error:  # propagated collectively below
            stage_error = error
    failed = torch.tensor(
        int(stage_error is not None), dtype=torch.int32, device=_collective_device()
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed.item()):
        raise RuntimeError(
            "rank-zero optimizer checkpoint validation failed"
        ) from stage_error
    payload: list[object] = [
        inventory.to_json_dict() if inventory is not None else None
    ]
    dist.broadcast_object_list(payload, src=0)
    return _parse_inventory(payload[0])


def _root_write_optimizer_checkpoint_inventory(
    checkpoint_dir: Path,
) -> OptimizerCheckpointInventory:
    if not dist.is_initialized():
        return write_optimizer_checkpoint_inventory(checkpoint_dir)
    rank = dist.get_rank()
    inventory: OptimizerCheckpointInventory | None = None
    stage_error: Exception | None = None
    if rank == 0:
        try:
            inventory = write_optimizer_checkpoint_inventory(checkpoint_dir)
        except Exception as error:  # propagated collectively below
            stage_error = error
    failed = torch.tensor(
        int(stage_error is not None), dtype=torch.int32, device=_collective_device()
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed.item()):
        raise RuntimeError(
            "rank-zero optimizer inventory creation failed"
        ) from stage_error
    payload: list[object] = [
        inventory.to_json_dict() if inventory is not None else None
    ]
    dist.broadcast_object_list(payload, src=0)
    return _parse_inventory(payload[0])


def save_optimizer_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    checkpoint_dir: Path,
) -> OptimizerCheckpointInventory:
    """Save DTensor optimizer shards and finalize a hash-bound inventory."""
    optimizer_state = get_optimizer_state_dict(model, optimizer)
    dcp.save(
        {"optimizer": optimizer_state},
        checkpoint_id=str(checkpoint_dir),
    )
    return _root_write_optimizer_checkpoint_inventory(checkpoint_dir)


def _prune_optimizer_state_without_checkpoint_payload(
    optimizer_state: dict[str, object],
    *,
    metadata_keys: tuple[str, ...],
) -> tuple[str, ...]:
    """Remove DCP's synthetic Adam states for parameters absent at save time.

    Torch 2.5 initializes every parameter in a fresh optimizer when
    ``get_optimizer_state_dict`` is called. A live optimizer checkpoint is
    legitimately sparse when some trainable branches have not received a
    gradient yet. Pruning the synthetic target entries preserves the exact
    lazy-Adam semantics and lets the default strict DCP planner validate every
    state tensor that was actually saved.
    """

    raw_state = optimizer_state.get("state")
    if not isinstance(raw_state, dict):
        raise TypeError("optimizer state_dict has no mutable state mapping")
    if any(not isinstance(key, str) for key in raw_state):
        raise TypeError("optimizer state_dict parameter names must be strings")
    if any(not isinstance(key, str) for key in metadata_keys):
        raise TypeError("DCP metadata keys must be strings")

    checkpoint_state_keys = {
        key for key in metadata_keys if key.startswith("optimizer.state.")
    }
    expected_checkpoint_keys: set[str] = set()
    saved_fqns: set[str] = set()
    for fqn, parameter_state in raw_state.items():
        if not isinstance(parameter_state, dict):
            raise TypeError(f"optimizer state for {fqn} must be a mapping")
        if any(not isinstance(key, str) for key in parameter_state):
            raise TypeError(f"optimizer state field names for {fqn} must be strings")
        expected_keys = {
            f"optimizer.state.{fqn}.{state_name}" for state_name in parameter_state
        }
        expected_checkpoint_keys.update(expected_keys)
        present_keys = expected_keys & checkpoint_state_keys
        if not present_keys:
            parameter_state.clear()
        elif present_keys != expected_keys:
            missing = sorted(expected_keys - present_keys)
            raise ValueError(
                f"optimizer checkpoint has partial state for {fqn}: {missing}"
            )
        else:
            saved_fqns.add(fqn)
    unmatched_checkpoint_keys = checkpoint_state_keys - expected_checkpoint_keys
    if unmatched_checkpoint_keys:
        preview = sorted(unmatched_checkpoint_keys)[:8]
        raise ValueError(
            "optimizer checkpoint contains state for unknown parameters: " f"{preview}"
        )

    return tuple(sorted(saved_fqns))


def _optimizer_param_group_fqns(
    optimizer_state: Mapping[str, object],
) -> tuple[tuple[str, ...], ...]:
    raw_groups = optimizer_state.get("param_groups")
    if not isinstance(raw_groups, list):
        raise TypeError("optimizer state_dict has no param_groups list")
    groups = []
    for group in raw_groups:
        if not isinstance(group, Mapping):
            raise TypeError("optimizer param group must be a mapping")
        params = group.get("params")
        if not isinstance(params, list) or any(
            not isinstance(param, str) for param in params
        ):
            raise TypeError("optimizer param group params must be FQN strings")
        groups.append(tuple(params))
    return tuple(groups)


def _validate_optimizer_param_group_metadata(
    optimizer_state: Mapping[str, object],
    *,
    planner_keys: tuple[str, ...],
) -> None:
    """Reject missing, extra, or renamed DCP optimizer-group fields."""

    raw_groups = optimizer_state.get("param_groups")
    if not isinstance(raw_groups, list):
        raise TypeError("optimizer state_dict has no param_groups list")
    expected_keys: set[str] = set()
    for group_index, group in enumerate(raw_groups):
        if not isinstance(group, Mapping) or any(
            not isinstance(field_name, str) for field_name in group
        ):
            raise TypeError("optimizer param group fields must be strings")
        expected_keys.update(
            f"optimizer.param_groups.{group_index}.{field_name}" for field_name in group
        )
    checkpoint_keys = {
        key for key in planner_keys if key.startswith("optimizer.param_groups.")
    }
    if checkpoint_keys != expected_keys:
        missing = sorted(expected_keys - checkpoint_keys)[:8]
        unexpected = sorted(checkpoint_keys - expected_keys)[:8]
        raise ValueError(
            "optimizer checkpoint param-group metadata differs from runtime: "
            f"missing={missing} unexpected={unexpected}"
        )


def load_optimizer_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    checkpoint_dir: Path,
    *,
    expected_inventory_sha256: str | None = None,
) -> OptimizerCheckpointInventory:
    """Verify and restore DTensor optimizer shards into a fresh optimizer."""
    inventory = _root_validate_optimizer_checkpoint(
        checkpoint_dir,
        expected_inventory_sha256,
    )
    optimizer_state = get_optimizer_state_dict(model, optimizer)
    storage_reader = dcp.FileSystemReader(str(checkpoint_dir))
    metadata = storage_reader.read_metadata()
    metadata_mapping = getattr(metadata, "state_dict_metadata", None)
    if not isinstance(metadata_mapping, Mapping):
        raise TypeError("optimizer DCP metadata has no state_dict mapping")
    planner_mapping = getattr(metadata, "planner_data", None)
    if not isinstance(planner_mapping, Mapping) or any(
        not isinstance(key, str) for key in planner_mapping
    ):
        raise TypeError("optimizer DCP metadata has no planner mapping")
    saved_state_fqns = _prune_optimizer_state_without_checkpoint_payload(
        optimizer_state,
        metadata_keys=tuple(metadata_mapping),
    )
    _validate_optimizer_param_group_metadata(
        optimizer_state,
        planner_keys=tuple(planner_mapping),
    )
    expected_param_groups = _optimizer_param_group_fqns(optimizer_state)
    checkpoint_state = {"optimizer": optimizer_state}
    dcp.load(checkpoint_state, storage_reader=storage_reader)
    if _optimizer_param_group_fqns(checkpoint_state["optimizer"]) != (
        expected_param_groups
    ):
        raise ValueError("optimizer checkpoint parameter groups differ from runtime")
    set_optimizer_state_dict(
        model,
        optimizer,
        checkpoint_state["optimizer"],
    )
    for parameter in tuple(optimizer.state):
        if not optimizer.state[parameter]:
            del optimizer.state[parameter]
    restored_state = get_optimizer_state_dict(model, optimizer).get("state")
    if not isinstance(restored_state, Mapping):
        raise TypeError("restored optimizer has no state mapping")
    saved_state_set = set(saved_state_fqns)
    if set(restored_state) != saved_state_set:
        raise RuntimeError(
            "restored optimizer state parameters differ from checkpoint metadata"
        )
    if any(not parameter_state for parameter_state in restored_state.values()):
        raise RuntimeError(
            "restored optimizer lazy-state coverage differs from checkpoint metadata"
        )
    return inventory
