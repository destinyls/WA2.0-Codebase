#!/usr/bin/env python3
"""Exercise FSDP2, strict DCP resume, and numerical continuation parity."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)

from n0_twam.distributed.fsdp import (
    FSDP2_API_SOURCE,
    MixedPrecisionPolicy,
    _build_fsdp_device_mesh,
    _resolve_fsdp_topology,
    fully_shard,
)
from n0_twam.distributed.optimizer_checkpoint import (
    OPTIMIZER_STATE_FORMAT,
    STRICT_CHECKPOINT_SCHEMA_VERSION,
    build_training_execution_contract,
    capture_runtime_signature,
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
    validate_runtime_signature,
    validate_training_execution_contract,
)
from n0_twam.models.model import capture_attention_execution_contract


@dataclass(frozen=True)
class SmokeResult:
    """Machine-readable outcome emitted by rank zero."""

    schema_version: int
    status: str
    torch_version: str
    fsdp2_api_source: str
    world_size: int
    device_name: str
    runtime_signature: dict[str, object]
    training_execution_contract: dict[str, object]
    loss_first_step: float
    loss_uninterrupted_second_step: float
    loss_resumed_second_step: float
    restore_max_abs_error: float
    parameter_second_step_max_abs_error: float
    adam_state_second_step_max_abs_error: float
    optimizer_state_restored: bool
    optimizer_inventory_sha256: str
    checkpoint_path: str
    resumed_checkpoint_path: str


class TinyBlock(nn.Module):  # type: ignore[misc]
    """Small block that is sharded independently before the root module."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(inputs))


class TinyModel(nn.Module):  # type: ignore[misc]
    """Minimal model following the same bottom-up FSDP2 pattern as N0-TWAM."""

    def __init__(self, width: int, block_count: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(width) for _ in range(block_count)])
        self.output = nn.Linear(width, width)
        # Deliberately unused: production MoT contains trainable branches that
        # may receive no gradient before an early staged checkpoint. This keeps
        # the DCP smoke faithful to sparse lazy-Adam state.
        self.unused = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="New or empty directory used for checkpoint artifacts.",
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument(
        "--blocks",
        type=int,
        default=8,
        help=(
            "Number of independently sharded tiny blocks. The default keeps "
            "optimizer DCP payloads non-empty in high-world-size smokes."
        ),
    )
    parser.add_argument("--max-latent-frames", type=int, default=5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-worker", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument(
        "--lr-schedule", choices=("constant", "cosine"), default="cosine"
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    return parser.parse_args(argv)


def _initialize_distributed() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError("runtime smoke requires at least two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    return rank, world_size, device


def _shard_model(model: TinyModel) -> TinyModel:
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    shard_options: dict[str, Any] = {
        "mp_policy": policy,
        "reshard_after_forward": True,
    }
    topology = _resolve_fsdp_topology()
    device_mesh = _build_fsdp_device_mesh(topology)
    if device_mesh is not None:
        shard_options["mesh"] = device_mesh
    for block in model.blocks:
        fully_shard(block, **shard_options)
    fully_shard(model, **shard_options)
    return model


def _full_state_options() -> StateDictOptions:
    return StateDictOptions(full_state_dict=True, cpu_offload=True)


def _collective_raise_if_error(
    error: Exception | None,
    stage: str,
    device: torch.device,
) -> None:
    """Make a rank-local stage failure stop every rank before the next stage."""
    failed = torch.tensor(int(error is not None), dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed.item()):
        raise RuntimeError(
            f"runtime smoke {stage} failed on at least one rank"
        ) from error


def _write_completion_marker(
    checkpoint_dir: Path,
    optimizer_inventory_sha256: str,
    runtime_signature: Mapping[str, object],
    training_execution_contract: Mapping[str, object],
) -> None:
    marker = {
        "schema_version": STRICT_CHECKPOINT_SCHEMA_VERSION,
        "status": "complete",
        "optimizer_state_format": OPTIMIZER_STATE_FORMAT,
        "optimizer_inventory_sha256": optimizer_inventory_sha256,
        "runtime_signature": dict(runtime_signature),
        "training_execution_contract": dict(training_execution_contract),
    }
    temporary_path = checkpoint_dir / ".checkpoint_complete.tmp"
    temporary_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, checkpoint_dir / "checkpoint_complete.json")


def _save_checkpoint(
    model: TinyModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    rank: int,
    device: torch.device,
    runtime_signature: Mapping[str, object],
    training_execution_contract: Mapping[str, object],
) -> str:
    stage_error: Exception | None = None
    try:
        model_state = get_model_state_dict(model, options=_full_state_options())
        if rank == 0:
            if checkpoint_dir.exists():
                raise FileExistsError(
                    f"refusing to overwrite smoke checkpoint: {checkpoint_dir}"
                )
            checkpoint_dir.mkdir(parents=True)
            torch.save(model_state, checkpoint_dir / "model.pt")
    except Exception as error:
        stage_error = error
    _collective_raise_if_error(stage_error, "model checkpoint stage", device)

    inventory = save_optimizer_checkpoint(
        model,
        optimizer,
        checkpoint_dir / "optimizer_dcp",
    )
    inventory_sha256 = inventory.inventory_sha256
    if not isinstance(inventory_sha256, str):
        raise TypeError("smoke optimizer inventory digest must be a string")

    stage_error = None
    try:
        if rank == 0:
            _write_completion_marker(
                checkpoint_dir,
                inventory_sha256,
                runtime_signature,
                training_execution_contract,
            )
    except Exception as error:
        stage_error = error
    _collective_raise_if_error(stage_error, "completion marker stage", device)
    return inventory_sha256


def _build_resumed_state(
    checkpoint_dir: Path,
    device: torch.device,
    training_execution_contract: Mapping[str, object],
) -> tuple[TinyModel, torch.optim.Optimizer, str]:
    stage_error: Exception | None = None
    model: TinyModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    marker: dict[str, Any] | None = None
    inventory_sha256: str | None = None
    try:
        marker = json.loads(
            (checkpoint_dir / "checkpoint_complete.json").read_text(encoding="utf-8")
        )
        if (
            int(marker.get("schema_version", 0)) != STRICT_CHECKPOINT_SCHEMA_VERSION
            or marker.get("status") != "complete"
            or marker.get("optimizer_state_format") != OPTIMIZER_STATE_FORMAT
        ):
            raise ValueError("smoke checkpoint completion marker is incompatible")
        inventory_sha256 = marker.get("optimizer_inventory_sha256")
        if not isinstance(inventory_sha256, str) or len(inventory_sha256) != 64:
            raise ValueError("smoke optimizer inventory digest is invalid")
        model_state = torch.load(
            checkpoint_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        block_indices = {
            int(name.split(".")[1])
            for name in model_state
            if name.startswith("blocks.")
        }
        if not block_indices or block_indices != set(range(max(block_indices) + 1)):
            raise ValueError("smoke checkpoint has a non-contiguous block inventory")
        model = TinyModel(
            width=model_state["output.weight"].shape[1],
            block_count=len(block_indices),
        )
        model.load_state_dict(model_state)
        model = _shard_model(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, fused=True, foreach=False
        )
    except Exception as error:
        stage_error = error
    _collective_raise_if_error(stage_error, "resume construction stage", device)
    assert (
        model is not None
        and optimizer is not None
        and marker is not None
        and inventory_sha256 is not None
    )

    validate_runtime_signature(marker.get("runtime_signature"))
    validate_training_execution_contract(
        marker.get("training_execution_contract"),
        current_contract=training_execution_contract,
    )
    load_optimizer_checkpoint(
        model,
        optimizer,
        checkpoint_dir / "optimizer_dcp",
        expected_inventory_sha256=inventory_sha256,
    )
    return model, optimizer, inventory_sha256


def _optimizer_step(
    model: TinyModel,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    predictions = model(inputs)
    loss = torch.nn.functional.mse_loss(predictions.float(), targets.float())
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite optimizer-step loss")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def _clone_local_tensor(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "to_local"):
        value = value.to_local()
    return value.detach().cpu().clone()


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _clone_local_tensor(value)
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return value


def _snapshot_parameters(model: TinyModel) -> dict[str, torch.Tensor]:
    return {
        name: _clone_local_tensor(parameter)
        for name, parameter in model.named_parameters()
    }


def _snapshot_adam_state(
    model: TinyModel,
    optimizer: torch.optim.Optimizer,
) -> Any:
    optimizer_state = get_optimizer_state_dict(model, optimizer)
    return _snapshot(optimizer_state["state"])


def _snapshot_max_abs_error(expected: Any, actual: Any) -> float:
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            return float("inf")
        if expected.numel() == 0:
            return 0.0
        if expected.is_floating_point() or expected.is_complex():
            if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                return float("inf")
            return float((expected - actual).abs().max().item())
        return 0.0 if torch.equal(expected, actual) else float("inf")
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            return float("inf")
        return max(
            (_snapshot_max_abs_error(expected[key], actual[key]) for key in expected),
            default=0.0,
        )
    if isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        if len(expected) != len(actual):
            return float("inf")
        return max(
            (
                _snapshot_max_abs_error(left, right)
                for left, right in zip(expected, actual, strict=True)
            ),
            default=0.0,
        )
    return 0.0 if expected == actual else float("inf")


def _global_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_latent_frames <= 0:
        raise ValueError("--max-latent-frames must be greater than zero")
    if args.blocks <= 0:
        raise ValueError("--blocks must be greater than zero")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be greater than zero")
    if args.batch_size <= 0 or args.num_steps <= 0:
        raise ValueError("--batch-size and --num-steps must be greater than zero")
    if args.load_worker != 0:
        raise ValueError("--load-worker must be 0 for strict resume parity")
    rank, world_size, device = _initialize_distributed()
    try:
        torch.manual_seed(20260801)
        torch.cuda.manual_seed_all(20260801)
        runtime_signature = capture_runtime_signature()
        validate_runtime_signature(runtime_signature)
        training_execution_contract: dict[str, object] | None = None
        stage_error: Exception | None = None
        try:
            training_execution_contract = build_training_execution_contract(
                max_latent_frames=args.max_latent_frames,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                batch_size=args.batch_size,
                load_worker=args.load_worker,
                num_steps=args.num_steps,
                lr_schedule=args.lr_schedule,
                warmup_steps=args.warmup_steps,
                lr_min_ratio=args.lr_min_ratio,
                activation_checkpointing=True,
                attention_contract=capture_attention_execution_contract(),
            )
        except Exception as error:
            stage_error = error
        _collective_raise_if_error(
            stage_error,
            "training execution contract capture stage",
            device,
        )
        assert training_execution_contract is not None
        validate_training_execution_contract(
            training_execution_contract,
            current_contract=training_execution_contract,
        )
        model = _shard_model(TinyModel(args.width, block_count=args.blocks))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, fused=True, foreach=False
        )
        inputs = torch.linspace(
            -1.0,
            1.0,
            steps=4 * args.width,
            device=device,
            dtype=torch.bfloat16,
        ).reshape(4, args.width)
        targets = torch.zeros_like(inputs)

        first_loss = _optimizer_step(model, optimizer, inputs, targets)
        with torch.no_grad():
            expected_after_first_step = model(inputs).float().detach().clone()
        checkpoint_path = args.work_dir / "checkpoint"
        optimizer_inventory_sha256 = _save_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            rank,
            device,
            runtime_signature,
            training_execution_contract,
        )

        uninterrupted_second_loss = _optimizer_step(model, optimizer, inputs, targets)
        expected_parameters = _snapshot_parameters(model)
        expected_adam_state = _snapshot_adam_state(model, optimizer)

        del model
        del optimizer
        torch.cuda.empty_cache()
        model, optimizer, restored_inventory_sha256 = _build_resumed_state(
            checkpoint_path,
            device,
            training_execution_contract,
        )
        if restored_inventory_sha256 != optimizer_inventory_sha256:
            raise RuntimeError("restored optimizer inventory digest changed")

        with torch.no_grad():
            restored = model(inputs).float()
        restore_error = _global_max(
            float((restored - expected_after_first_step).abs().max().item()),
            device,
        )
        if not math.isfinite(restore_error) or restore_error > 1e-5:
            raise RuntimeError(
                f"checkpoint restore mismatch: max_abs_error={restore_error}"
            )

        optimizer_state_restored = bool(optimizer.state)
        restored_flag = torch.tensor(
            int(optimizer_state_restored), device=device, dtype=torch.int32
        )
        dist.all_reduce(restored_flag, op=dist.ReduceOp.MIN)
        if int(restored_flag.item()) != 1:
            raise RuntimeError("optimizer state was not restored on every rank")

        resumed_second_loss = _optimizer_step(model, optimizer, inputs, targets)
        parameter_error = _global_max(
            _snapshot_max_abs_error(expected_parameters, _snapshot_parameters(model)),
            device,
        )
        adam_state_error = _global_max(
            _snapshot_max_abs_error(
                expected_adam_state, _snapshot_adam_state(model, optimizer)
            ),
            device,
        )
        if (
            not math.isfinite(parameter_error)
            or not math.isfinite(adam_state_error)
            or parameter_error > 1e-7
            or adam_state_error > 1e-7
        ):
            raise RuntimeError(
                "resumed second step diverged from uninterrupted execution: "
                f"parameter_error={parameter_error}, "
                f"adam_state_error={adam_state_error}"
            )

        resumed_checkpoint_path = args.work_dir / "checkpoint_resumed"
        _save_checkpoint(
            model,
            optimizer,
            resumed_checkpoint_path,
            rank,
            device,
            runtime_signature,
            training_execution_contract,
        )

        result_stage_error: Exception | None = None
        try:
            if rank == 0:
                result = SmokeResult(
                    schema_version=3,
                    status="ok",
                    torch_version=torch.__version__,
                    fsdp2_api_source=FSDP2_API_SOURCE,
                    world_size=world_size,
                    device_name=torch.cuda.get_device_name(device),
                    runtime_signature=runtime_signature,
                    training_execution_contract=training_execution_contract,
                    loss_first_step=float(first_loss.item()),
                    loss_uninterrupted_second_step=float(
                        uninterrupted_second_loss.item()
                    ),
                    loss_resumed_second_step=float(resumed_second_loss.item()),
                    restore_max_abs_error=restore_error,
                    parameter_second_step_max_abs_error=parameter_error,
                    adam_state_second_step_max_abs_error=adam_state_error,
                    optimizer_state_restored=True,
                    optimizer_inventory_sha256=optimizer_inventory_sha256,
                    checkpoint_path=str(checkpoint_path.resolve()),
                    resumed_checkpoint_path=str(resumed_checkpoint_path.resolve()),
                )
                result_json = json.dumps(asdict(result), sort_keys=True)
                (args.work_dir / "result.json").write_text(
                    result_json + "\n", encoding="utf-8"
                )
                print(result_json)
        except Exception as error:
            result_stage_error = error
        _collective_raise_if_error(result_stage_error, "result stage", device)
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
