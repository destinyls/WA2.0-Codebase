"""Distributed sampling utilities for attention-mask schedules."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def resolve_sync_attention_window() -> bool:
    """Return whether all ranks must share one chunk/window schedule."""
    value = os.environ.get("N0_SYNC_ATTENTION_WINDOW", "0")
    if value not in ("0", "1"):
        raise ValueError("N0_SYNC_ATTENTION_WINDOW must be 0 or 1")
    return value == "1"


@torch.no_grad()
def sample_attention_mask_schedule(
    device: torch.device | str,
) -> tuple[int, int]:
    """Sample a chunk/window pair and optionally broadcast rank zero's values."""
    if not (resolve_sync_attention_window() and dist.is_initialized()):
        return (
            int(torch.randint(1, 5, (1,)).item()),
            int(torch.randint(4, 65, (1,)).item()),
        )

    sampled = torch.zeros(2, dtype=torch.int64, device=device)
    if dist.get_rank() == 0:
        sampled[0] = torch.randint(1, 5, (1,), device=device)
        sampled[1] = torch.randint(4, 65, (1,), device=device)
    dist.broadcast(sampled, src=0)
    return int(sampled[0].item()), int(sampled[1].item())
