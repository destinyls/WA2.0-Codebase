# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import torch
import pytest

import n0_twam.models.mot as mot_module
from n0_twam.models.mot import MoTBackbone, SharedSelfAttention


def test_streaming_attention_uses_vendor_flash_backend(monkeypatch) -> None:
    attention = SharedSelfAttention()
    attention.flex.attention_backend = "grouped_flash_attn"
    attention.init_kv_cache(
        "episode",
        total_tolen=8,
        num_head=2,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )
    calls: list[tuple[torch.Size, torch.Size, torch.Size]] = []

    def _fake_flash(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        dropout_p: float,
        causal: bool,
    ) -> torch.Tensor:
        assert dropout_p == 0.0
        assert causal is False
        calls.append((q.shape, k.shape, v.shape))
        return torch.zeros_like(q)

    monkeypatch.setattr(mot_module, "flash_attn_func", _fake_flash)
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 2, 2, 4)
    value = torch.randn(1, 2, 2, 4)

    output = attention(query, key, value, cache_name="episode")

    assert output.shape == query.shape
    assert calls == [(query.shape, key.shape, value.shape)]


def test_streaming_attention_keeps_sdpa_fallback(monkeypatch) -> None:
    attention = SharedSelfAttention()
    attention.flex.attention_backend = "grouped_sdpa"
    attention.init_kv_cache(
        "episode",
        total_tolen=8,
        num_head=2,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )
    monkeypatch.setattr(mot_module, "flash_attn_func", None)
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 2, 2, 4)
    value = torch.randn(1, 2, 2, 4)

    output = attention(query, key, value, cache_name="episode")

    assert output.shape == query.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("active_name", ["video", "action", "tactile"])
def test_mot_backbone_skips_every_empty_expert(
    monkeypatch: pytest.MonkeyPatch,
    active_name: str,
) -> None:
    backbone = MoTBackbone(
        num_layers=1,
        dim=8,
        ffn_dim=16,
        num_heads=2,
        attn_mode="torch",
    ).eval()

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("empty expert executed")

    for name in {"video", "action", "tactile"} - {active_name}:
        expert = backbone.experts[name]
        monkeypatch.setattr(expert, "embed_in", unexpected)
        monkeypatch.setattr(expert, "modulation", unexpected)
        monkeypatch.setattr(expert, "text_kv", unexpected)
        monkeypatch.setattr(expert, "embed_out", unexpected)
        monkeypatch.setattr(expert.blocks[0], "forward", unexpected)

    slices = []
    cursor = 0
    for name in ("video", "action", "tactile"):
        end = cursor + (2 if name == active_name else 0)
        slices.append((name, cursor, end))
        cursor = end

    hidden = torch.randn(1, 2, 8)
    with torch.no_grad():
        output = backbone(
            hidden,
            torch.randn(1, 3, 8),
            torch.randn(1, 2, 6, 8),
            None,
            torch.ones(1, 2, 1, 2, dtype=torch.complex64),
            tuple(slices),
        )

    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
