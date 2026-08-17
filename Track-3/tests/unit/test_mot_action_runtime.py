# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Model-level contracts for request-local Action Expert KV reuse."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from n0_twam.models.mot import WanMoTTransformer3DModel
from n0_twam.models.mot_action_cache import ActionCacheExecution
from n0_twam.models.mot_action_runtime import ActionDenoiseModelCache


def _model(*, full_tail: bool = False) -> WanMoTTransformer3DModel:
    torch.manual_seed(7)
    return WanMoTTransformer3DModel(
        patch_size=[1, 1, 1],
        num_attention_heads=2,
        attention_head_dim=6,
        in_channels=2,
        out_channels=2,
        action_dim=3,
        text_dim=8,
        freq_dim=8,
        ffn_dim=16,
        num_layers=2,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=32,
        attn_mode="torch",
        use_local_tactile=full_tail,
        use_contact_gate=full_tail,
        contact_gate_heads=2,
        use_wrench_conditioner=full_tail,
        wrench_arm_count=2,
        wrench_max_frames=4,
        mot_expert_ffn_dim={"video": 16, "action": 16, "tactile": 16},
        mot_expert_hidden_dim={"video": 12, "action": 12, "tactile": 12},
    ).eval()


def _action_input(*, seed: int, frame_start: int = 0) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "noisy_latents": torch.randn(1, 3, 2, 1, 1, generator=generator),
        "text_emb": torch.randn(1, 4, 8, generator=generator),
        "grid_id": torch.tensor(
            [[[frame_start, frame_start + 2], [0, 0], [0, 0], [0, 0]]]
        ),
        "frame_start_id": frame_start,
        "timesteps": torch.full((1, 2), float(seed % 10 + 1)),
        "tactile_global_latent": torch.randn(1, 1, 48, 1, 1, 1, generator=generator),
        "tactile_sensor_ids": torch.tensor([[0]]),
        "tactile_profile": "vision_tactile",
        "route_id": "agilex_touch",
        "prompt_id": "pick-cube",
    }


def _next_step(fixed: dict[str, Any], *, seed: int) -> dict[str, Any]:
    step = copy.copy(fixed)
    generator = torch.Generator().manual_seed(seed)
    step["noisy_latents"] = torch.randn(1, 3, 2, 1, 1, generator=generator)
    step["timesteps"] = torch.full((1, 2), float(seed % 10 + 1))
    return step


def test_model_cached_action_matches_fixed_context_full_tail() -> None:
    model = _model()
    first = _action_input(seed=11)
    step = _next_step(first, seed=22)

    with torch.no_grad():
        _, action_cache = model.prepare_action_denoise_cache(
            first,
            cache_name="episode",
            request_signature="episode:chunk:4",
        )
        cached, execution = model.forward_action_denoise_cached(
            step,
            cache_name="episode",
            action_cache=action_cache,
            request_signature="episode:chunk:4",
        )
        expected = model.forward_action_denoise_terminal(
            step,
            cache_name="unused",
            request_signature="episode:chunk:4",
        )

    assert isinstance(action_cache, ActionDenoiseModelCache)
    assert isinstance(execution, ActionCacheExecution)
    assert execution.used_cache is True
    assert execution.fallback_reason is None
    torch.testing.assert_close(cached, expected, rtol=1e-5, atol=1e-5)


def test_full_reference_matches_prepare_cached_and_terminal_causal_semantics() -> None:
    model = _model()
    first = _action_input(seed=23)
    step = _next_step(first, seed=24)
    signature = "episode:chunk:causal-reference"

    with torch.no_grad():
        full_first = model.forward_action_denoise_full(
            first,
            cache_name="full-first",
            request_signature=signature,
            update_cache=0,
        )
        prepared, action_cache = model.prepare_action_denoise_cache(
            first,
            cache_name="episode",
            request_signature=signature,
        )
        full_step = model.forward_action_denoise_full(
            step,
            cache_name="full-step",
            request_signature=signature,
            update_cache=0,
        )
        cached, execution = model.forward_action_denoise_cached(
            step,
            cache_name="episode",
            action_cache=action_cache,
            request_signature=signature,
        )
        terminal = model.forward_action_denoise_terminal(
            step,
            cache_name="terminal",
            request_signature=signature,
        )

    assert execution.used_cache is True
    torch.testing.assert_close(prepared, full_first, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cached, full_step, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(terminal, full_step, rtol=1e-5, atol=1e-5)


def test_model_identity_mismatch_runs_safe_full_fallback() -> None:
    model = _model()
    first = _action_input(seed=31)
    step = _next_step(first, seed=32)

    with torch.no_grad():
        _, action_cache = model.prepare_action_denoise_cache(
            first,
            cache_name="episode",
            request_signature="episode:chunk:7",
        )
        actual, execution = model.forward_action_denoise_cached(
            step,
            cache_name="episode",
            action_cache=action_cache,
            request_signature="episode:chunk:8",
        )
        expected = model.forward_action_denoise_terminal(
            step,
            cache_name="unused",
            request_signature="episode:chunk:8",
        )

    assert execution.used_cache is False
    assert "request signature" in execution.fallback_reason
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_cached_path_preserves_local_tactile_wrench_contact_and_head_tail() -> None:
    model = _model(full_tail=True)
    first = _action_input(seed=35)
    first.update(
        {
            "tactile_local_latent": torch.randn(1, 1, 48, 1, 1, 1),
            "wrench": torch.randn(1, 1, 2, 6),
            "wrench_available_mask": torch.ones(1, 1, 2, dtype=torch.bool),
            "temporal_valid_mask": torch.ones(1, 1, dtype=torch.bool),
            "contact_cond_drop": torch.zeros(1, dtype=torch.bool),
        }
    )
    step = _next_step(first, seed=36)
    calls = {"local": 0, "wrench": 0, "contact": 0}

    def _count(name: str):
        def hook(_module: object, _inputs: object, _output: object) -> None:
            calls[name] += 1

        return hook

    handles = (
        model.local_tactile_cross_attn.register_forward_hook(_count("local")),
        model.agilex_wrench_cross_attn.register_forward_hook(_count("wrench")),
        model.contact_gate.register_forward_hook(_count("contact")),
    )
    try:
        with torch.no_grad():
            _, action_cache = model.prepare_action_denoise_cache(
                first,
                cache_name="episode",
                request_signature="episode:chunk:tail",
            )
            cached, execution = model.forward_action_denoise_cached(
                step,
                cache_name="episode",
                action_cache=action_cache,
                request_signature="episode:chunk:tail",
            )
            expected = model.forward_action_denoise_terminal(
                step,
                cache_name="unused",
                request_signature="episode:chunk:tail",
            )
    finally:
        for handle in handles:
            handle.remove()

    assert execution.used_cache is True
    assert calls == {"local": 3, "wrench": 3, "contact": 3}
    torch.testing.assert_close(cached, expected, rtol=1e-5, atol=1e-5)


def test_model_identity_binds_frame_route_prompt_and_fixed_condition() -> None:
    model = _model()
    input_dict = _action_input(seed=41, frame_start=12)

    with torch.no_grad():
        _, action_cache = model.prepare_action_denoise_cache(
            input_dict,
            cache_name="episode",
            request_signature="episode:chunk:12",
        )

    identity = action_cache.identity
    assert identity.request_signature == "episode:chunk:12"
    assert identity.frame_grid_signature
    assert identity.route_signature
    assert identity.prompt_signature
    assert identity.fixed_condition_signature

    changes: tuple[tuple[str, dict[str, object]], ...] = (
        ("frame/grid", {"frame_start_id": 32}),
        ("route", {"route_id": "franka_touch"}),
        ("prompt", {"prompt_id": "place-cube"}),
        (
            "fixed condition",
            {
                "tactile_global_latent": input_dict["tactile_global_latent"].repeat(
                    1, 2, 1, 1, 1, 1
                ),
                "tactile_sensor_ids": torch.tensor([[0, 1]]),
            },
        ),
    )
    for expected_reason, updates in changes:
        changed = copy.copy(input_dict)
        changed.update(updates)
        with torch.no_grad():
            _, execution = model.forward_action_denoise_cached(
                changed,
                cache_name="episode",
                action_cache=action_cache,
                request_signature="episode:chunk:12",
            )
        assert execution.used_cache is False
        assert expected_reason in execution.fallback_reason


def test_terminal_full_pass_commits_once_and_clears_transient_state() -> None:
    model = _model()
    input_dict = _action_input(seed=51)
    model.create_empty_cache(
        "episode",
        attn_window=2,
        latent_token_per_chunk=1,
        action_token_per_chunk=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )
    before = tuple(sa.cache_revision("episode") for sa in model.mot.shared_attn)

    with torch.no_grad():
        output = model.forward_action_denoise_terminal(
            input_dict,
            cache_name="episode",
            request_signature="episode:chunk:terminal",
        )

    after = tuple(sa.cache_revision("episode") for sa in model.mot.shared_attn)
    assert output.shape == (1, 2, 3)
    assert all(a[1] == b[1] + 1 for a, b in zip(after, before, strict=True))
    assert not hasattr(model, "_mot_action_runtime_request")
    assert not hasattr(model, "_mot_action_runtime_result")


def test_model_advertises_action_denoise_kv_capability() -> None:
    assert _model().supports_action_denoise_kv_reuse is True


def test_custom_runtime_uses_module_call_so_root_fsdp_hooks_fire() -> None:
    model = _model()
    first = _action_input(seed=61)
    step = _next_step(first, seed=62)
    hook_calls = {"pre": 0, "post": 0}

    def pre_hook(_module: object, _inputs: object) -> None:
        hook_calls["pre"] += 1

    def post_hook(_module: object, _inputs: object, _output: object) -> None:
        hook_calls["post"] += 1

    handles = (
        model.register_forward_pre_hook(pre_hook),
        model.register_forward_hook(post_hook),
    )
    try:
        _, action_cache = model.prepare_action_denoise_cache(
            first,
            cache_name="episode",
            request_signature="episode:chunk:hooks",
        )
        model.forward_action_denoise_cached(
            step,
            cache_name="episode",
            action_cache=action_cache,
            request_signature="episode:chunk:hooks",
        )
        model.forward_action_denoise_terminal(
            step,
            cache_name="unused",
            request_signature="episode:chunk:hooks",
        )
    finally:
        for handle in handles:
            handle.remove()

    assert hook_calls == {"pre": 3, "post": 3}


def test_frame_start_cpu_scalar_avoids_grid_device_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    input_dict = _action_input(seed=71, frame_start=8)

    def fail_min(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("grid min would synchronize the accelerator")

    monkeypatch.setattr(torch.Tensor, "min", fail_min)
    with torch.no_grad():
        output = model.forward_action_denoise_terminal(
            input_dict,
            cache_name="unused",
            request_signature="episode:chunk:no-device-sync",
        )

    assert output.shape == (1, 2, 3)
