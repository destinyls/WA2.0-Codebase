#!/usr/bin/env python3
"""Verify FSDP2 parameter reuse across split pre/post expert forwards."""

from __future__ import annotations

import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from n0_twam.distributed.fsdp import MixedPrecisionPolicy, fully_shard


class SplitExpert(nn.Module):  # type: ignore[misc]
    """Tiny expert called twice, matching MoT's split pre/post pattern."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.pre = nn.Linear(width, width)
        self.post = nn.Linear(width, width)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mode: str,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if mode == "pre":
            return self.pre(inputs), inputs
        if mode == "post" and residual is not None:
            return F.gelu(self.post(inputs)) + residual
        raise ValueError("mode must be pre or post with a residual")


class SplitModel(nn.Module):  # type: ignore[misc]
    """Model that explicitly reshards experts after every split layer."""

    def __init__(self, width: int, expert_count: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [SplitExpert(width) for _ in range(expert_count)]
        )
        self.output = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pre_outputs: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        for expert in self.experts:
            pre_output, residual = expert(inputs, mode="pre")
            pre_outputs.append(pre_output)
            residuals.append(residual)

        merged = torch.stack(pre_outputs).mean(dim=0)
        outputs: list[torch.Tensor] = []
        for expert, residual in zip(self.experts, residuals, strict=True):
            outputs.append(expert(merged, mode="post", residual=residual))

        for expert in self.experts:
            expert.reshard()  # type: ignore[attr-defined]
        return self.output(torch.stack(outputs).mean(dim=0))


def main() -> None:
    """Run one finite distributed optimizer update and emit a rank-zero report."""
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)

    torch.manual_seed(20260804)
    model = SplitModel(width=128, expert_count=3).to(device=device)
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        cast_forward_inputs=False,
    )
    for expert in model.experts:
        fully_shard(expert, mp_policy=policy, reshard_after_forward=False)
    fully_shard(model, mp_policy=policy, reshard_after_forward=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randn(2, 32, 128, device=device, dtype=torch.bfloat16)
    target = torch.randn_like(inputs)
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    loss = (output - target).float().square().mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    finite = torch.tensor(
        int(math.isfinite(float(loss)) and math.isfinite(float(grad_norm))),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not int(finite.item()):
        raise RuntimeError("non-finite loss or gradient norm")

    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "world_size": world_size,
                    "loss": float(loss),
                    "grad_norm": float(grad_norm),
                    "reduce_dtype": "bfloat16",
                    "reshard_after_forward": False,
                    "manual_reshard": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
