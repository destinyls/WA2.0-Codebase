#!/usr/bin/env python3
"""Profile the exact Track 3.1 F-token grouped attention contract on one HCU."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.models.model import FlexAttnFunc, GroupedAttentionMask


@dataclass(frozen=True)
class AttentionRun:
    kind: str
    query_tokens: int
    key_value_tokens: int
    group_count: int
    cumulative_key_value_indices: int
    max_query_group_tokens: int
    max_key_value_group_tokens: int
    loss: float
    elapsed_seconds: float
    elapsed_samples_seconds: tuple[float, ...]
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    repeat_count: int


@dataclass(frozen=True)
class AttentionSample:
    loss: float
    elapsed_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--latent-frames", type=int, default=5)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, choices=range(1, 5), default=None)
    parser.add_argument("--window-size", type=int, choices=range(4, 65), default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args(argv)


def _require_finite(tensor: torch.Tensor | None, label: str) -> None:
    if tensor is None or not torch.isfinite(tensor).all():
        raise RuntimeError(f"{label} is missing or non-finite")


def _expected_attended_queries(
    *,
    kind: str,
    latent_frames: int,
    padded_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    video_tokens = 2 * latent_frames * 8 * 16
    action_tokens = 2 * latent_frames * 4
    tactile_tokens = 4 * latent_frames * 4 * 4
    non_padding_tokens = video_tokens + action_tokens + tactile_tokens
    if kind == "self":
        attended_tokens = non_padding_tokens
    elif kind == "cross":
        attended_tokens = video_tokens + action_tokens
    else:
        raise ValueError(f"Unsupported attention kind: {kind}")
    sequence_tokens = non_padding_tokens + padded_tokens
    if attended_tokens > sequence_tokens:
        raise RuntimeError("Expected attended queries exceed the sequence length")
    return torch.arange(attended_tokens, dtype=torch.long, device=device)


def _validate_mask_contract(
    *,
    kind: str,
    mask: GroupedAttentionMask,
    latent_frames: int,
    padded_tokens: int,
    query_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    attended_queries = torch.cat(
        [query_indices for query_indices, _ in mask.groups],
        dim=0,
    )
    expected_attended = _expected_attended_queries(
        kind=kind,
        latent_frames=latent_frames,
        padded_tokens=padded_tokens,
        device=device,
    )
    sorted_attended = torch.sort(attended_queries).values
    if not torch.equal(sorted_attended, expected_attended):
        raise RuntimeError(
            f"{kind} mask attended-query contract mismatch: "
            f"actual={sorted_attended.numel()}, expected={expected_attended.numel()}"
        )
    if torch.unique(attended_queries).numel() != attended_queries.numel():
        raise RuntimeError(f"{kind} mask assigns a query to more than one group")
    expected_unassigned = query_tokens - expected_attended.numel()
    if kind == "self" and expected_unassigned != padded_tokens:
        raise RuntimeError("Self-attention unassigned-query contract mismatch")
    if kind == "cross":
        expected_tactile_tokens = 4 * latent_frames * 4 * 4
        if expected_unassigned != expected_tactile_tokens + padded_tokens:
            raise RuntimeError("Cross-attention unassigned-query contract mismatch")
    unassigned = torch.ones(query_tokens, dtype=torch.bool, device=device)
    unassigned[expected_attended] = False
    return unassigned


def _run_operation_once(
    *,
    kind: str,
    mask: GroupedAttentionMask,
    unassigned: torch.Tensor,
    query_tokens: int,
    key_value_tokens: int,
    heads: int,
    head_dim: int,
    device: torch.device,
) -> AttentionSample:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    query = torch.randn(
        (1, query_tokens, heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    key = torch.randn(
        (1, key_value_tokens, heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    value = torch.randn_like(key, requires_grad=True)
    operation = FlexAttnFunc(is_cross=kind == "cross").to(device)
    if operation.attention_backend not in ("grouped_sdpa", "grouped_flash_attn"):
        raise RuntimeError(
            "Track 3.1 HCU profile requires a grouped backend, got "
            f"{operation.attention_backend}"
        )
    operation.set_block_mask(mask)

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = operation(query, key, value)
    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    # Cache the operation peak before validation creates temporary tensors.
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    _require_finite(output, f"{kind} output")
    _require_finite(query.grad, f"{kind} query gradient")
    _require_finite(key.grad, f"{kind} key gradient")
    _require_finite(value.grad, f"{kind} value gradient")
    if unassigned.any() and torch.count_nonzero(output[:, unassigned]).item() != 0:
        raise RuntimeError(f"{kind} fully masked queries must produce zero output")

    return AttentionSample(
        loss=float(loss.item()),
        elapsed_seconds=elapsed,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )


def _profile_operation(
    *,
    kind: str,
    mask: GroupedAttentionMask,
    latent_frames: int,
    padded_tokens: int,
    query_tokens: int,
    key_value_tokens: int,
    heads: int,
    head_dim: int,
    device: torch.device,
    warmup_runs: int,
    repeats: int,
) -> AttentionRun:
    unassigned = _validate_mask_contract(
        kind=kind,
        mask=mask,
        latent_frames=latent_frames,
        padded_tokens=padded_tokens,
        query_tokens=query_tokens,
        device=device,
    )
    for _ in range(warmup_runs):
        _run_operation_once(
            kind=kind,
            mask=mask,
            unassigned=unassigned,
            query_tokens=query_tokens,
            key_value_tokens=key_value_tokens,
            heads=heads,
            head_dim=head_dim,
            device=device,
        )
    samples = tuple(
        _run_operation_once(
            kind=kind,
            mask=mask,
            unassigned=unassigned,
            query_tokens=query_tokens,
            key_value_tokens=key_value_tokens,
            heads=heads,
            head_dim=head_dim,
            device=device,
        )
        for _ in range(repeats)
    )
    elapsed_samples = tuple(sample.elapsed_seconds for sample in samples)
    return AttentionRun(
        kind=kind,
        query_tokens=query_tokens,
        key_value_tokens=key_value_tokens,
        group_count=len(mask.groups),
        cumulative_key_value_indices=sum(
            int(key_value_indices.numel()) for _, key_value_indices in mask.groups
        ),
        max_query_group_tokens=max(
            int(query_indices.numel()) for query_indices, _ in mask.groups
        ),
        max_key_value_group_tokens=max(
            int(key_value_indices.numel()) for _, key_value_indices in mask.groups
        ),
        loss=float(statistics.median(sample.loss for sample in samples)),
        elapsed_seconds=float(statistics.median(elapsed_samples)),
        elapsed_samples_seconds=elapsed_samples,
        peak_allocated_bytes=max(sample.peak_allocated_bytes for sample in samples),
        peak_reserved_bytes=max(sample.peak_reserved_bytes for sample in samples),
        repeat_count=repeats,
    )


def _mask_cases(args: argparse.Namespace) -> tuple[tuple[int, int], ...]:
    if (args.chunk_size is None) != (args.window_size is None):
        raise ValueError("--chunk-size and --window-size must be provided together")
    if args.chunk_size is not None:
        return ((args.chunk_size, args.window_size),)
    return ((1, 4), (1, 64), (4, 4), (4, 64))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if (
        args.latent_frames <= 0
        or args.heads <= 0
        or args.head_dim <= 0
        or args.warmup_runs < 0
        or args.repeats <= 0
    ):
        raise ValueError(
            "latent frames, heads, head dim, and repeats must be positive; "
            "warmup runs must be non-negative"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("HCU profile requires the vendor CUDA-compatible API")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(20260801)
    torch.cuda.manual_seed_all(20260801)

    tactile_half_tokens = 2 * args.latent_frames * 4 * 4
    base_tokens = (
        2 * args.latent_frames * 8 * 16
        + 2 * args.latent_frames * 4
        + 2 * tactile_half_tokens
    )
    padded_tokens = (-base_tokens) % 128
    sequence_tokens = base_tokens + padded_tokens
    cases = []
    for chunk_size, window_size in _mask_cases(args):
        self_mask, cross_mask = FlexAttnFunc.init_mask(
            latent_shape=(1, 48, args.latent_frames, 16, 32),
            action_shape=(1, 8, args.latent_frames, 4, 1),
            padded_length=padded_tokens,
            chunk_size=chunk_size,
            window_size=window_size,
            patch_size=(1, 2, 2),
            device=device,
            text_token_length=512,
            tactile_token_length=2 * tactile_half_tokens,
            tactile_noisy_token_length=tactile_half_tokens,
            tactile_grid_shape=(1, 2, args.latent_frames, 4, 4),
        )
        if not isinstance(self_mask, GroupedAttentionMask) or not isinstance(
            cross_mask, GroupedAttentionMask
        ):
            raise RuntimeError("HCU profile did not construct grouped attention masks")
        cases.append(
            {
                "chunk_size": chunk_size,
                "window_size": window_size,
                "self_attention": asdict(
                    _profile_operation(
                        kind="self",
                        mask=self_mask,
                        latent_frames=args.latent_frames,
                        padded_tokens=padded_tokens,
                        query_tokens=sequence_tokens,
                        key_value_tokens=sequence_tokens,
                        heads=args.heads,
                        head_dim=args.head_dim,
                        device=device,
                        warmup_runs=args.warmup_runs,
                        repeats=args.repeats,
                    )
                ),
                "cross_attention": asdict(
                    _profile_operation(
                        kind="cross",
                        mask=cross_mask,
                        latent_frames=args.latent_frames,
                        padded_tokens=padded_tokens,
                        query_tokens=sequence_tokens,
                        key_value_tokens=512,
                        heads=args.heads,
                        head_dim=args.head_dim,
                        device=device,
                        warmup_runs=args.warmup_runs,
                        repeats=args.repeats,
                    )
                ),
            }
        )
    report = {
        "schema_version": 2,
        "status": "ok",
        "measurement_scope": "isolated_grouped_attention_forward_backward",
        "torch_version": str(torch.__version__),
        "device": torch.cuda.get_device_name(device),
        "latent_frames": args.latent_frames,
        "video_latent_shape": [1, 48, args.latent_frames, 16, 32],
        "tactile_grid_shape": [1, 2, args.latent_frames, 4, 4],
        "sequence_tokens": sequence_tokens,
        "padding_tokens": padded_tokens,
        "warmup_runs": args.warmup_runs,
        "repeat_count": args.repeats,
        "cases": cases,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
