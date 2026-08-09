# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    FSDP2_API_SOURCE = "torch.distributed.fsdp"
except ImportError:
    # PyTorch 2.4/2.5 exposes the same composable FSDP2 API from the private
    # namespace.  Some accelerator vendor builds remain on that layout even
    # though their fully_shard and MixedPrecisionPolicy signatures match the
    # later public API.  Keep this fallback isolated so it can be removed once
    # those runtimes move to the public namespace.
    from torch.distributed._composable.fsdp import (  # type: ignore[no-redef]
        MixedPrecisionPolicy,
        fully_shard,
    )

    FSDP2_API_SOURCE = "torch.distributed._composable.fsdp"

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)


FSDP_EXECUTION_CONTRACT_SCHEMA_VERSION = 1
_FSDP_TOPOLOGIES = ("global_shard", "hsdp")
_EXPERT_RESHARD_POLICIES = ("after_call", "after_layer", "after_backward")


@dataclass(frozen=True)
class FSDPTopologyConfig:
    """Resolved FSDP2 topology with node-local HSDP invariants."""

    mode: str
    world_size: int
    local_world_size: int
    shard_size: int
    replicate_size: int
    mesh_shape: tuple[int, ...]


def _mot_expert_stacks(model):
    """For a MoT model, yield each per-expert block ModuleList; else None."""
    if hasattr(model, "mot"):
        return [model.mot.experts[name].blocks for name in model.mot.expert_names]
    return None


def apply_ac(model):
    """Apply activation checkpointing to the model's transformer blocks."""
    if not _activation_checkpointing_enabled():
        if hasattr(model, "mot"):
            model.mot.gradient_checkpointing = False
        return
    stacks = _mot_expert_stacks(model)
    if stacks is None:
        for layer_id, transformer_block in enumerate(model.blocks):
            model.blocks[layer_id] = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        return
    # MoT: do NOT per-block checkpoint. The block is split into pre/post around the
    # shared attention, so a per-block wrapper leaves q/k/v/attn (~10GB at S~15k)
    # resident -> 80GB OOM. Instead MoTBackbone.forward checkpoints the WHOLE layer
    # body (legacy granularity), recomputing q/k/v/attn/ffn in backward.
    if hasattr(model, "mot"):
        model.mot.gradient_checkpointing = True


def _resolve_fsdp_reduce_dtype() -> torch.dtype:
    """Resolve the collective dtype while preserving the legacy default."""
    value = os.environ.get("N0_FSDP_REDUCE_DTYPE", "float32")
    supported = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if value not in supported:
        raise ValueError(
            "N0_FSDP_REDUCE_DTYPE must be one of "
            f"{tuple(supported)}, got {value!r}"
        )
    return supported[value]


def _keep_expert_params_between_pre_post() -> bool:
    """Return whether one MoT layer should reuse its gathered expert weights."""
    value = os.environ.get(
        "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST",
        "0",
    )
    if value not in ("0", "1"):
        raise ValueError(
            "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST must be 0 or 1"
        )
    return value == "1"


def _resolve_expert_reshard_policy() -> str:
    """Resolve when gathered MoT expert parameters are reshared.

    The legacy controls remain the default source of truth: no parameter reuse
    maps to ``after_call`` and pre/post reuse maps to ``after_layer``.  The new
    explicit policy adds ``after_backward`` for runs with enough memory to keep
    expert parameters gathered through backward and avoid a second all-gather.
    """
    keep_expert_params = _keep_expert_params_between_pre_post()
    default_policy = "after_layer" if keep_expert_params else "after_call"
    policy = os.environ.get(
        "N0_FSDP_EXPERT_RESHARD_POLICY",
        default_policy,
    )
    if policy not in _EXPERT_RESHARD_POLICIES:
        raise ValueError(
            "N0_FSDP_EXPERT_RESHARD_POLICY must be one of "
            f"{_EXPERT_RESHARD_POLICIES}, got {policy!r}"
        )
    requires_reuse = policy in ("after_layer", "after_backward")
    if requires_reuse and not keep_expert_params:
        raise ValueError(
            f"N0_FSDP_EXPERT_RESHARD_POLICY={policy!r} requires "
            "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST=1"
        )
    if policy == "after_call" and keep_expert_params:
        raise ValueError(
            "N0_FSDP_EXPERT_RESHARD_POLICY='after_call' requires "
            "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST=0"
        )
    return policy


def _parse_positive_int(value: str, *, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _resolve_fsdp_topology(
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
) -> FSDPTopologyConfig:
    """Resolve a global FSDP or node-local two-dimensional HSDP mesh."""
    mode = os.environ.get("N0_FSDP_TOPOLOGY", "global_shard")
    if mode not in _FSDP_TOPOLOGIES:
        raise ValueError(
            f"N0_FSDP_TOPOLOGY must be one of {_FSDP_TOPOLOGIES}, "
            f"got {mode!r}"
        )
    resolved_world_size = world_size
    if resolved_world_size is None:
        resolved_world_size = (
            dist.get_world_size()
            if dist.is_initialized()
            else _parse_positive_int(
                os.environ.get("WORLD_SIZE", "1"),
                label="WORLD_SIZE",
            )
        )
    resolved_local_world_size = local_world_size
    if resolved_local_world_size is None:
        resolved_local_world_size = _parse_positive_int(
            os.environ.get("LOCAL_WORLD_SIZE", str(resolved_world_size)),
            label="LOCAL_WORLD_SIZE",
        )
    if resolved_world_size <= 0 or resolved_local_world_size <= 0:
        raise ValueError("world sizes must be positive integers")

    shard_size_value = os.environ.get("N0_FSDP_SHARD_SIZE")
    if shard_size_value is not None:
        configured_shard_size = _parse_positive_int(
            shard_size_value,
            label="N0_FSDP_SHARD_SIZE",
        )
    else:
        configured_shard_size = resolved_local_world_size

    if mode == "global_shard":
        return FSDPTopologyConfig(
            mode=mode,
            world_size=resolved_world_size,
            local_world_size=resolved_local_world_size,
            shard_size=resolved_world_size,
            replicate_size=1,
            mesh_shape=(resolved_world_size,),
        )
    if configured_shard_size != resolved_local_world_size:
        raise ValueError(
            "HSDP shard size must equal LOCAL_WORLD_SIZE so every shard group "
            "stays within one physical node"
        )
    if resolved_world_size % configured_shard_size:
        raise ValueError(
            "WORLD_SIZE must be divisible by the node-local HSDP shard size"
        )
    replicate_size = resolved_world_size // configured_shard_size
    if replicate_size < 2:
        raise ValueError("HSDP requires at least two replica groups")
    return FSDPTopologyConfig(
        mode=mode,
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        shard_size=configured_shard_size,
        replicate_size=replicate_size,
        mesh_shape=(replicate_size, configured_shard_size),
    )


def _build_fsdp_device_mesh(
    topology: FSDPTopologyConfig,
) -> DeviceMesh | None:
    """Build the shared HSDP mesh, or preserve legacy implicit global sharding."""
    if topology.mode == "global_shard":
        return None
    return init_device_mesh(
        "cuda",
        topology.mesh_shape,
        mesh_dim_names=("replicate", "shard"),
    )


def capture_fsdp_execution_contract(
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
) -> dict[str, Any]:
    """Capture performance settings that affect FSDP execution and resume."""
    topology = _resolve_fsdp_topology(
        world_size=world_size,
        local_world_size=local_world_size,
    )
    reduce_dtype = _resolve_fsdp_reduce_dtype()
    return {
        "schema_version": FSDP_EXECUTION_CONTRACT_SCHEMA_VERSION,
        "topology": topology.mode,
        "mesh_shape": list(topology.mesh_shape),
        "replicate_size": topology.replicate_size,
        "shard_size": topology.shard_size,
        "expert_reshard_policy": _resolve_expert_reshard_policy(),
        "reduce_dtype": "bfloat16" if reduce_dtype is torch.bfloat16 else "float32",
    }


def _activation_checkpointing_enabled() -> bool:
    """Resolve whether transformer activation checkpointing is enabled."""
    value = os.environ.get("N0_MOT_ACTIVATION_CHECKPOINTING", "1")
    if value not in ("0", "1"):
        raise ValueError("N0_MOT_ACTIVATION_CHECKPOINTING must be 0 or 1")
    return value == "1"


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=None):
    if reduce_dtype is None:
        reduce_dtype = _resolve_fsdp_reduce_dtype()
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    topology = _resolve_fsdp_topology()
    device_mesh = _build_fsdp_device_mesh(topology)
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}
    if device_mesh is not None:
        fsdp_config["mesh"] = device_mesh

    stacks = _mot_expert_stacks(model)
    if stacks is None:
        for block in model.blocks:
            fully_shard(block.attn1, **fsdp_config)
            fully_shard(block.attn2, **fsdp_config)
            fully_shard(block.ffn, **fsdp_config)
            fully_shard(block, **fsdp_config)
    else:
        expert_reshard_policy = _resolve_expert_reshard_policy()
        expert_fsdp_config = {
            **fsdp_config,
            "reshard_after_forward": expert_reshard_policy == "after_call",
        }
        # MoT: shard each expert block as ONE FSDP unit (it IS run via forward, so
        # the all-gather hook fires and the block reshards after each pre/post call).
        # We do NOT separately shard attn1/attn2/ffn: those are invoked via
        # project_qkv/merge_out (not their own forward), so a nested fully_shard
        # would bypass their hook. Block-level sharding gathers them together.
        for stack in stacks:
            for block in stack:
                fully_shard(block, **expert_fsdp_config)
        model.mot.manual_reshard_experts = (
            expert_reshard_policy == "after_layer"
        )

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
