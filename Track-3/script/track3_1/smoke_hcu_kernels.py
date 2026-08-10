#!/usr/bin/env python3
"""Run the exact attention/RoPE kernels required before real N0 HCU loading."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.distributed.fsdp import FSDP2_API_SOURCE
from n0_twam.models.model import (
    FlexAttnFunc,
    GroupedAttentionMask,
    WanAttention,
    WanRotaryPosEmbed,
    _uses_eager_block_mask_creation,
    custom_sdpa,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _require_finite(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{label} contains non-finite values")


def _flex_smoke(device: torch.device) -> dict[str, object]:
    base_length = 2 * (1 * 8 * 8) + 2 * 4
    padded_length = (-base_length) % 128
    block_mask, cross_mask = FlexAttnFunc.init_mask(
        latent_shape=(1, 48, 1, 16, 16),
        action_shape=(1, 8, 4, 1, 1),
        padded_length=padded_length,
        chunk_size=1,
        window_size=4,
        patch_size=(1, 2, 2),
        device=device,
        text_token_length=512,
    )
    sequence_length = base_length + padded_length
    values = torch.linspace(
        -1.0,
        1.0,
        steps=sequence_length * 2 * 64,
        dtype=torch.bfloat16,
        device=device,
    ).reshape(1, sequence_length, 2, 64)
    query = values.clone().requires_grad_(True)
    key = values.flip(1).clone().requires_grad_(True)
    value = values.roll(1, dims=1).clone().requires_grad_(True)
    operation = FlexAttnFunc().to(device)
    operation.set_block_mask(block_mask)
    output = operation(query, key, value)
    if torch.count_nonzero(output[:, base_length:]).item() != 0:
        raise RuntimeError("self-attention padding queries must produce zero output")
    self_loss = output.float().square().mean()
    self_loss.backward()
    for name, tensor in (
        ("flex output", output),
        ("flex query grad", query.grad),
        ("flex key grad", key.grad),
        ("flex value grad", value.grad),
    ):
        if tensor is None:
            raise RuntimeError(f"{name} is missing")
        _require_finite(tensor, name)

    cross_query = values.detach().clone().requires_grad_(True)
    cross_values = torch.linspace(
        -0.75,
        0.75,
        steps=512 * 2 * 64,
        dtype=torch.bfloat16,
        device=device,
    ).reshape(1, 512, 2, 64)
    cross_key = cross_values.flip(1).clone().requires_grad_(True)
    cross_value = cross_values.roll(1, dims=1).clone().requires_grad_(True)
    cross_operation = FlexAttnFunc(is_cross=True).to(device)
    cross_operation.set_block_mask(cross_mask)
    cross_output = cross_operation(cross_query, cross_key, cross_value)
    if torch.count_nonzero(cross_output[:, base_length:]).item() != 0:
        raise RuntimeError("cross-attention padding queries must produce zero output")
    cross_loss = cross_output.float().square().mean()
    cross_loss.backward()
    for name, tensor in (
        ("cross-flex output", cross_output),
        ("cross-flex query grad", cross_query.grad),
        ("cross-flex key grad", cross_key.grad),
        ("cross-flex value grad", cross_value.grad),
    ):
        if tensor is None:
            raise RuntimeError(f"{name} is missing")
        _require_finite(tensor, name)

    return {
        "backend": operation.attention_backend,
        "mask_creation_mode": (
            "grouped_sdpa"
            if isinstance(block_mask, GroupedAttentionMask)
            else (
                "eager_torch_2_5"
                if _uses_eager_block_mask_creation(torch.__version__)
                else "compiled"
            )
        ),
        "sequence_length": sequence_length,
        "self_mask_shape": list(block_mask.shape),
        "cross_mask_shape": list(cross_mask.shape),
        "self_loss": float(self_loss.item()),
        "cross_loss": float(cross_loss.item()),
    }


def _rope_smoke(device: torch.device) -> dict[str, object]:
    sequence_length = 8
    grid_ids = torch.zeros((1, 3, sequence_length), dtype=torch.float64, device=device)
    grid_ids[:, 0, :] = torch.arange(sequence_length, device=device)
    rotary = WanRotaryPosEmbed(128, (1, 2, 2), 1024).to(device)
    rotary_emb = rotary(grid_ids)[:, :, None]
    attention = WanAttention(
        dim=128,
        heads=1,
        dim_head=128,
        attn_mode="torch",
    ).to(device=device, dtype=torch.bfloat16)
    inputs = torch.linspace(
        -0.5,
        0.5,
        steps=sequence_length * 128,
        dtype=torch.bfloat16,
        device=device,
    ).reshape(1, sequence_length, 128)
    query = inputs.clone().requires_grad_(True)
    key = inputs.flip(1).clone().requires_grad_(True)
    value = inputs.roll(1, dims=1).clone().requires_grad_(True)
    output = attention(query, key, value, rotary_emb)
    loss = output.float().square().mean()
    loss.backward()
    _require_finite(output, "RoPE attention output")
    for name, tensor in (
        ("RoPE query grad", query.grad),
        ("RoPE key grad", key.grad),
        ("RoPE value grad", value.grad),
    ):
        if tensor is None:
            raise RuntimeError(f"{name} is missing")
        _require_finite(tensor, name)
    return {
        "sequence_length": sequence_length,
        "rotary_dtype": str(rotary_emb.dtype),
        "loss": float(loss.item()),
    }


def _sdpa_and_optimizer_smoke(device: torch.device) -> dict[str, float]:
    query = torch.randn(
        (1, 8, 2, 32), dtype=torch.bfloat16, device=device, requires_grad=True
    )
    key = torch.randn(
        (1, 12, 2, 32), dtype=torch.bfloat16, device=device, requires_grad=True
    )
    value = torch.randn(
        (1, 12, 2, 32), dtype=torch.bfloat16, device=device, requires_grad=True
    )
    output = custom_sdpa(query, key, value)
    sdpa_loss = output.float().square().mean()
    sdpa_loss.backward()
    _require_finite(output, "SDPA output")

    layer = torch.nn.Linear(32, 32, device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(
        layer.parameters(), lr=1e-3, fused=True, foreach=False
    )
    inputs = torch.ones((2, 32), device=device, dtype=torch.bfloat16)
    optimizer_loss = layer(inputs).float().square().mean()
    optimizer_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _require_finite(optimizer_loss, "fused AdamW loss")
    return {
        "sdpa_loss": float(sdpa_loss.item()),
        "fused_adamw_loss": float(optimizer_loss.item()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("HCU kernel smoke requires the vendor CUDA-compatible API")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(20260801)
    torch.cuda.manual_seed_all(20260801)
    if not hasattr(torch._inductor.config, "realize_opcount_threshold"):
        raise RuntimeError("vendor Torch lacks realize_opcount_threshold")

    report = {
        "schema_version": 1,
        "status": "ok",
        "torch_version": str(torch.__version__),
        "device": torch.cuda.get_device_name(device),
        "fsdp2_api_source": FSDP2_API_SOURCE,
        "flex": _flex_smoke(device),
        "rope": _rope_smoke(device),
        "sdpa_optimizer": _sdpa_and_optimizer_smoke(device),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
