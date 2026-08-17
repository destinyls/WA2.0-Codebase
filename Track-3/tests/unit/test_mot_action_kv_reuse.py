# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""CPU contracts for the MoT action-denoise KV-reuse dispatcher."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import pytest
import torch

import n0_twam.models.mot as mot_module
from n0_twam.models.mot import MoTBackbone

ACTION_COLS = slice(2, 4)
SLICES = (
    ("video", 0, 2),
    ("action", 2, 4),
    ("tactile", 4, 6),
)


def _fixed_context_mask() -> torch.Tensor:
    """Let action read all context while fixed rows cannot read noisy action."""
    mask = torch.ones(6, 6, dtype=torch.bool)
    mask[:2, ACTION_COLS] = False
    mask[4:, ACTION_COLS] = False
    return mask


def _backbone() -> MoTBackbone:
    torch.manual_seed(23)
    backbone = MoTBackbone(
        num_layers=2,
        dim=8,
        ffn_dim=16,
        num_heads=2,
        attn_mode="torch",
    ).eval()
    backbone.set_masks(dense_self_mask=_fixed_context_mask())
    return backbone


def _inputs(*, batch_size: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "hidden_states": torch.randn(batch_size, 6, 8, generator=generator),
        "encoder_hidden_states": torch.randn(batch_size, 3, 8, generator=generator),
        "timestep_proj": torch.randn(batch_size, 6, 6, 8, generator=generator),
        "temb": torch.randn(batch_size, 6, 8, generator=generator),
        "rotary_emb": torch.ones(batch_size, 6, 1, 2, dtype=torch.complex64),
    }


def _replace_action_step(
    fixed: dict[str, torch.Tensor], *, seed: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    step = {name: value.clone() for name, value in fixed.items()}
    batch_size = fixed["hidden_states"].shape[0]
    step["hidden_states"][:, ACTION_COLS] = torch.randn(
        batch_size, 2, 8, generator=generator
    )
    step["timestep_proj"][:, ACTION_COLS] = torch.randn(
        batch_size, 2, 6, 8, generator=generator
    )
    step["temb"][:, ACTION_COLS] = torch.randn(batch_size, 2, 8, generator=generator)
    return step


def _full_action(
    backbone: MoTBackbone, inputs: dict[str, torch.Tensor]
) -> torch.Tensor:
    with torch.no_grad():
        output = backbone(**inputs, slices=SLICES)
    return output[:, ACTION_COLS]


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _snapshot(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return value


def _assert_snapshot_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if isinstance(expected, dict):
        actual_values = (
            {field.name: getattr(actual, field.name) for field in fields(actual)}
            if is_dataclass(actual) and not isinstance(actual, type)
            else actual
        )
        assert actual_values.keys() == expected.keys()
        for key in expected:
            _assert_snapshot_equal(actual_values[key], expected[key])
        return
    if isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_snapshot_equal(actual_item, expected_item)
        return
    assert actual == expected


def _prefill(
    backbone: MoTBackbone,
    inputs: dict[str, torch.Tensor],
    *,
    cache_name: str | None = None,
) -> Any:
    return backbone.prefill_action_denoise_cache(
        **inputs,
        slices=SLICES,
        action_cols=ACTION_COLS,
        action_rows_mask=_fixed_context_mask()[ACTION_COLS],
        cache_name=cache_name,
    )


@pytest.mark.parametrize("step_seed", [101, 202])
def test_action_cache_matches_full_cpu_for_two_denoise_steps(
    step_seed: int,
) -> None:
    backbone = _backbone()
    fixed = _inputs(batch_size=1, seed=17)
    cache = _prefill(backbone, fixed)
    step = _replace_action_step(fixed, seed=step_seed)

    expected = _full_action(backbone, step)
    with torch.no_grad():
        actual, execution = backbone.forward_action_cached_or_full(
            **step,
            slices=SLICES,
            action_cols=ACTION_COLS,
            action_rows_mask=_fixed_context_mask()[ACTION_COLS],
            action_cache=cache,
            cache_name=None,
        )

    assert execution.used_cache is True
    assert execution.fallback_reason is None
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_cached_action_step_does_not_mutate_prefill_or_rolling_cache() -> None:
    backbone = _backbone()
    fixed = _inputs(batch_size=1, seed=31)

    for layer, attention in enumerate(backbone.shared_attn):
        attention.init_kv_cache(
            "episode",
            total_tolen=12,
            num_head=2,
            head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=1,
        )
        generator = torch.Generator().manual_seed(300 + layer)
        attention.update_cache(
            "episode",
            torch.randn(1, 2, 2, 4, generator=generator),
            torch.randn(1, 2, 2, 4, generator=generator),
            is_pred=False,
        )
    cache = _prefill(backbone, fixed, cache_name="episode")
    cache_before = _snapshot(cache)
    rolling_before = [
        _snapshot(attn.attn_caches["episode"]) for attn in backbone.shared_attn
    ]

    for step_seed in (401, 402):
        step = _replace_action_step(fixed, seed=step_seed)
        with torch.no_grad():
            _, execution = backbone.forward_action_cached_or_full(
                **step,
                slices=SLICES,
                action_cols=ACTION_COLS,
                action_rows_mask=_fixed_context_mask()[ACTION_COLS],
                action_cache=cache,
                cache_name="episode",
            )
        assert execution.used_cache is True

    _assert_snapshot_equal(cache, cache_before)
    for attention, expected in zip(backbone.shared_attn, rolling_before, strict=True):
        _assert_snapshot_equal(attention.attn_caches["episode"], expected)


def test_incompatible_action_cache_falls_back_to_full_path() -> None:
    backbone = _backbone()
    cache = _prefill(backbone, _inputs(batch_size=1, seed=51))
    incompatible_step = _inputs(batch_size=2, seed=52)
    expected = _full_action(backbone, incompatible_step)

    with torch.no_grad():
        actual, execution = backbone.forward_action_cached_or_full(
            **incompatible_step,
            slices=SLICES,
            action_cols=ACTION_COLS,
            action_rows_mask=_fixed_context_mask()[ACTION_COLS],
            action_cache=cache,
            cache_name=None,
        )

    assert execution.used_cache is False
    assert "batch" in execution.fallback_reason.lower()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_prefill_rejects_bidirectional_fixed_context() -> None:
    backbone = _backbone()
    backbone.set_masks(dense_self_mask=torch.ones(6, 6, dtype=torch.bool))

    with pytest.raises(
        ValueError,
        match="fixed-context rows must not attend noisy action columns",
    ):
        _prefill(backbone, _inputs(batch_size=1, seed=71))


def test_prefill_does_not_evict_a_full_rolling_pool() -> None:
    backbone = _backbone()
    fixed = _inputs(batch_size=1, seed=81)
    for layer, attention in enumerate(backbone.shared_attn):
        attention.init_kv_cache(
            "full",
            total_tolen=8,
            num_head=2,
            head_dim=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=1,
        )
        generator = torch.Generator().manual_seed(900 + layer)
        attention.update_cache(
            "full",
            torch.randn(1, 8, 2, 4, generator=generator),
            torch.randn(1, 8, 2, 4, generator=generator),
            is_pred=False,
        )
    before = [
        _snapshot(attention.attn_caches["full"]) for attention in backbone.shared_attn
    ]

    _prefill(backbone, fixed, cache_name="full")

    for attention, expected in zip(backbone.shared_attn, before, strict=True):
        _assert_snapshot_equal(attention.attn_caches["full"], expected)


def test_cached_action_step_reshards_each_action_block_after_layer() -> None:
    backbone = _backbone()
    fixed = _inputs(batch_size=1, seed=91)
    cache = _prefill(backbone, fixed)
    calls: list[int] = []
    for layer, block in enumerate(backbone.experts["action"].blocks):
        block.reshard = lambda layer=layer: calls.append(layer)  # type: ignore[attr-defined]
    backbone.manual_reshard_experts = True

    step = _replace_action_step(fixed, seed=92)
    with torch.no_grad():
        _, execution = backbone.forward_action_cached_or_full(
            **step,
            slices=SLICES,
            action_cols=ACTION_COLS,
            action_rows_mask=_fixed_context_mask()[ACTION_COLS],
            action_cache=cache,
            cache_name=None,
        )

    assert execution.used_cache is True
    assert calls == [0, 1]


def test_cached_action_keeps_full_reference_attention_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-visible action row must not silently switch to vendor flash."""

    backbone = _backbone()
    for attention in backbone.shared_attn:
        attention.flex.attention_backend = "grouped_flash_attn"

    def incompatible_flash(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        dropout_p: float,
        causal: bool,
    ) -> torch.Tensor:
        del key, value, dropout_p, causal
        return torch.zeros_like(query)

    monkeypatch.setattr(mot_module, "flash_attn_func", incompatible_flash)
    fixed = _inputs(batch_size=1, seed=101)
    cache = _prefill(backbone, fixed)
    step = _replace_action_step(fixed, seed=102)
    expected = _full_action(backbone, step)

    with torch.no_grad():
        actual, execution = backbone.forward_action_cached_or_full(
            **step,
            slices=SLICES,
            action_cols=ACTION_COLS,
            action_rows_mask=_fixed_context_mask()[ACTION_COLS],
            action_cache=cache,
            cache_name=None,
        )

    assert execution.used_cache is True
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_cached_action_preserves_full_reference_query_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDPA kernel selection must see the same query shape in both paths."""

    original_sdpa = mot_module.custom_sdpa

    def query_extent_sensitive_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = original_sdpa(query, key, value, attn_mask=attn_mask)
        return output + output.new_tensor(float(query.shape[1]) / 100.0)

    monkeypatch.setattr(mot_module, "custom_sdpa", query_extent_sensitive_sdpa)
    backbone = _backbone()
    fixed = _inputs(batch_size=1, seed=111)
    cache = _prefill(backbone, fixed)
    step = _replace_action_step(fixed, seed=112)
    expected = _full_action(backbone, step)

    with torch.no_grad():
        actual, execution = backbone.forward_action_cached_or_full(
            **step,
            slices=SLICES,
            action_cols=ACTION_COLS,
            action_rows_mask=_fixed_context_mask()[ACTION_COLS],
            action_cache=cache,
            cache_name=None,
        )

    assert execution.used_cache is True
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
