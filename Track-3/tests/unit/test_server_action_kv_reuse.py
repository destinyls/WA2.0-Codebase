# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Server-side contracts for request-local Action Expert KV reuse."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from n0_twam.configs.twam_track32_franka_server_cfg import (
    build_track32_franka_server_config,
    twam_track32_franka_server_cfg,
)
from n0_twam.configs.twam_track3_agilex_mixed_cfg import (
    twam_track3_agilex_mixed_cfg,
)
from n0_twam.configs.twam_track3_agilex_server_cfg import (
    build_track3_agilex_server_config,
)
from n0_twam.n0_twam_server import TWAM_Server
from n0_twam.server_action_kv import _ActionKVReuseDispatcher


class _ActionScheduler:
    def step(
        self,
        prediction: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        return_dict: bool,
    ) -> torch.Tensor:
        assert return_dict is False
        del prediction, timestep
        return sample + 1


class _FastTransformer:
    supports_action_denoise_kv_reuse = True

    def __init__(self, *, incompatible_cached_step: int | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._cached_step = 0
        self._incompatible_cached_step = incompatible_cached_step

    @staticmethod
    def _prediction() -> torch.Tensor:
        return torch.ones(1, 1, 1)

    def prepare_action_denoise_cache(
        self,
        input_dict: dict[str, object],
        *,
        cache_name: str,
        request_signature: str,
    ) -> tuple[torch.Tensor, object]:
        self.calls.append(("prepare", (input_dict, cache_name, request_signature)))
        return self._prediction(), object()

    def forward_action_denoise_cached(
        self,
        input_dict: dict[str, object],
        *,
        cache_name: str,
        action_cache: object,
        request_signature: str,
    ) -> tuple[torch.Tensor, SimpleNamespace]:
        del action_cache
        self._cached_step += 1
        self.calls.append(("cached", (input_dict, cache_name, request_signature)))
        incompatible = self._cached_step == self._incompatible_cached_step
        return self._prediction(), SimpleNamespace(
            used_cache=not incompatible,
            fallback_reason="fixed_context_identity_mismatch" if incompatible else None,
        )

    def forward_action_denoise_terminal(
        self,
        input_dict: dict[str, object],
        *,
        cache_name: str,
        request_signature: str,
    ) -> torch.Tensor:
        self.calls.append(("terminal", (input_dict, cache_name, request_signature)))
        return self._prediction()

    def forward_action_denoise_full(
        self,
        input_dict: dict[str, object],
        *,
        cache_name: str,
        request_signature: str,
        update_cache: int,
    ) -> torch.Tensor:
        self.calls.append(
            (
                "causal_full",
                (input_dict, cache_name, request_signature, update_cache),
            )
        )
        return self._prediction()

    def __call__(
        self,
        input_dict: dict[str, object],
        *,
        update_cache: int,
        cache_name: str,
        action_mode: bool,
    ) -> torch.Tensor:
        self.calls.append(("full", (input_dict, update_cache, cache_name, action_mode)))
        return self._prediction()


class _LegacyTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def __call__(
        self,
        input_dict: dict[str, object],
        *,
        update_cache: int,
        cache_name: str,
        action_mode: bool,
    ) -> torch.Tensor:
        del input_dict
        assert cache_name == "pos"
        self.calls.append((update_cache, action_mode))
        return torch.ones(1, 1, 1)


def _server(transformer: object, *, with_tactile: bool = True) -> TWAM_Server:
    server = object.__new__(TWAM_Server)
    server.transformer = transformer
    server.cache_name = "pos"
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.action_per_frame = 1
    server.action_scheduler = _ActionScheduler()
    server.job_config = SimpleNamespace(
        action_dim=1,
        action_guidance_scale=1,
        action_denoise_kv_reuse=True,
        action_denoise_kv_fallback_policy="rerun_full_v1",
        cold_seed_mode="free",
        show_inference_progress=False,
    )
    server._uses_pi05_delta_actions = lambda: False

    def prepare_input(*args: object, **kwargs: object) -> dict[str, object]:
        del kwargs
        action_input: dict[str, object] = {"step": int(args[2])}
        if with_tactile:
            action_input["tactile_global_latent"] = torch.ones(1, 1, 1)
        return {"action_res_lst": action_input}

    server._prepare_latent_input = prepare_input
    server._repeat_input_for_cfg = lambda value: value
    return server


def _run(server: TWAM_Server) -> torch.Tensor:
    return server._denoise_actions(
        obs={},
        frame_st_id=12,
        frame_chunk_size=1,
        action_timesteps=torch.tensor([3, 2, 1, 0]),
        actions=torch.zeros(1, 1, 1, 1, 1),
        tactile_latents=None,
        wrench_condition=None,
    )


def test_fast_path_runs_prepare_cached_then_terminal_without_rolling_writes() -> None:
    transformer = _FastTransformer()
    server = _server(transformer)

    actions = _run(server)

    assert actions.item() == 3
    assert [name for name, _ in transformer.calls] == [
        "prepare",
        "cached",
        "cached",
        "terminal",
    ]
    receipt = server._last_action_kv_reuse_receipt
    assert receipt["used_fast_path"] is True
    assert receipt["cached_steps"] == 2
    assert receipt["reran_full"] is False
    assert receipt["fallback_reason"] is None


def test_missing_capability_uses_original_full_path_for_the_whole_round() -> None:
    transformer = _LegacyTransformer()
    server = _server(transformer)

    actions = _run(server)

    assert actions.item() == 3
    assert transformer.calls == [(0, True), (0, True), (0, True), (1, True)]
    receipt = server._last_action_kv_reuse_receipt
    assert receipt["used_fast_path"] is False
    assert receipt["fallback_reason"] == "capability_unavailable"
    assert receipt["reran_full"] is False


def test_vision_only_route_skips_cache_even_when_model_supports_it() -> None:
    transformer = _FastTransformer()
    server = _server(transformer, with_tactile=False)

    actions = _run(server)

    assert actions.item() == 3
    assert [name for name, _ in transformer.calls] == ["causal_full"] * 4
    assert [payload[-1] for _, payload in transformer.calls] == [0, 0, 0, 1]
    receipt = server._last_action_kv_reuse_receipt
    assert receipt["used_fast_path"] is False
    assert receipt["fallback_reason"] == "no_fixed_context_tokens"
    assert receipt["reran_full"] is False


def test_cache_identity_fallback_restarts_the_whole_round_from_initial_noise() -> None:
    transformer = _FastTransformer(incompatible_cached_step=1)
    server = _server(transformer)
    clock_values = iter(
        [
            0.000,
            0.001,
            0.010,
            0.012,
            0.020,
            0.021,
            0.022,
            0.023,
            0.024,
            0.025,
            0.026,
            0.027,
            0.028,
            0.030,
        ]
    )
    sync_calls: list[None] = []
    server._action_kv_monotonic_clock = lambda: next(clock_values)
    server._action_kv_accelerator_sync = lambda: sync_calls.append(None)

    actions = _run(server)

    assert actions.item() == 3
    assert [name for name, _ in transformer.calls] == [
        "prepare",
        "cached",
        "causal_full",
        "causal_full",
        "causal_full",
        "causal_full",
    ]
    assert [call[1][-1] for call in transformer.calls[-4:]] == [0, 0, 0, 1]
    receipt = server._last_action_kv_reuse_receipt
    assert receipt["used_fast_path"] is False
    assert receipt["fallback_reason"] == "fixed_context_identity_mismatch"
    assert receipt["reran_full"] is True
    assert receipt["attempted_cached_steps"] == 1
    assert receipt["phase_latency_ms"]["samples"]["full_reference"] == [
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(1.0),
    ]
    assert receipt["phase_latency_ms"]["samples"]["fallback_replay"] == [
        pytest.approx(10.0)
    ]
    assert receipt["phase_latency_ms"]["summary"]["fallback_replay"] == {
        "count": 1,
        "mean": pytest.approx(10.0),
        "p50": pytest.approx(10.0),
        "p95": pytest.approx(10.0),
        "max": pytest.approx(10.0),
    }
    assert len(sync_calls) == 14


def test_dispatcher_uses_one_request_signature_for_the_entire_chunk() -> None:
    transformer = _FastTransformer()
    dispatcher = _ActionKVReuseDispatcher(
        transformer=transformer,
        cache_name="pos",
        requested=True,
        fallback_policy="rerun_full_v1",
        request_signature="pos:frame=12:request=7",
    )

    for index in range(4):
        dispatcher.predict(
            {"step": index, "tactile_global_latent": torch.ones(1)},
            last_step=index == 3,
        )

    signatures = [payload[-1] for _, payload in transformer.calls]
    assert signatures == ["pos:frame=12:request=7"] * 4


def test_dispatcher_records_accelerator_synchronized_phase_latency() -> None:
    transformer = _FastTransformer()
    clock_values = iter([0.000, 0.001, 0.010, 0.014, 0.020, 0.022, 0.030, 0.035])
    sync_calls: list[None] = []
    dispatcher = _ActionKVReuseDispatcher(
        transformer=transformer,
        cache_name="pos",
        requested=True,
        fallback_policy="rerun_full_v1",
        request_signature="timed-request",
        accelerator_sync=lambda: sync_calls.append(None),
        monotonic_clock=lambda: next(clock_values),
    )

    for index in range(4):
        dispatcher.predict(
            {"step": index, "tactile_global_latent": torch.ones(1)},
            last_step=index == 3,
        )

    timing = dispatcher.receipt()["phase_latency_ms"]
    assert timing["samples"] == {
        "prepare": [pytest.approx(1.0)],
        "cached": [pytest.approx(4.0), pytest.approx(2.0)],
        "terminal": [pytest.approx(5.0)],
        "full_reference": [],
        "fallback_replay": [],
    }
    assert timing["summary"]["cached"] == {
        "count": 2,
        "mean": pytest.approx(3.0),
        "p50": pytest.approx(3.0),
        "p95": pytest.approx(3.9),
        "max": pytest.approx(4.0),
    }
    assert len(sync_calls) == 8


def test_disabled_dispatcher_receipt_is_explicit_and_has_no_fast_samples() -> None:
    transformer = _FastTransformer()
    server = _server(transformer)
    server.job_config.action_denoise_kv_reuse = False

    _run(server)

    assert [name for name, _ in transformer.calls] == ["causal_full"] * 4
    assert [payload[-1] for _, payload in transformer.calls] == [0, 0, 0, 1]
    receipt = server._last_action_kv_reuse_receipt
    assert receipt["requested"] is False
    assert receipt["fallback_reason"] == "disabled_by_config"
    assert receipt["full_reference_steps"] == 4
    samples = receipt["phase_latency_ms"]["samples"]
    assert samples["prepare"] == []
    assert samples["cached"] == []
    assert samples["terminal"] == []
    assert samples["fallback_replay"] == []
    assert len(samples["full_reference"]) == 4
    assert receipt["phase_latency_ms"]["summary"]["full_reference"]["count"] == 4
    assert {
        name: values for name, values in samples.items() if name != "full_reference"
    } == {
        "prepare": [],
        "cached": [],
        "terminal": [],
        "fallback_replay": [],
    }


def test_prepare_inputs_expose_cpu_frame_start_identity_without_hcu_sync() -> None:
    server = object.__new__(TWAM_Server)
    server.job_config = SimpleNamespace(tactile_profile="vision_only")
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.prompt_embeds = torch.zeros(1, 2, 4)
    server.action_mask = torch.ones(1, dtype=torch.bool)

    prepared = server._prepare_latent_input(
        torch.zeros(1, 1, 1, 2, 2),
        torch.zeros(1, 1, 1, 1, 1),
        frame_st_id=17,
    )

    assert prepared["latent_res_lst"]["frame_start_id"] == 17
    assert prepared["action_res_lst"]["frame_start_id"] == 17
    assert isinstance(prepared["action_res_lst"]["frame_start_id"], int)


def test_agilex_and_franka_configs_enable_the_same_fail_safe_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("N0_ACTION_DENOISE_KV_REUSE", raising=False)
    agilex = build_track3_agilex_server_config(
        training_config=twam_track3_agilex_mixed_cfg
    )
    franka = twam_track32_franka_server_cfg

    for config in (agilex, franka):
        assert config.action_denoise_kv_reuse is True
        assert config.action_denoise_kv_contract == "fixed_context_causal_v1"
        assert config.action_denoise_kv_activation_policy == (
            "requires_tactile_global_tokens_v1"
        )
        assert config.action_denoise_kv_fallback_policy == "rerun_full_v1"

    assert agilex.server_runtime_contract["action_denoise_kv_reuse"] is True
    assert agilex.server_runtime_contract["action_denoise_kv_contract"] == (
        "fixed_context_causal_v1"
    )
    assert franka.server_runtime_contract["action_denoise_kv_reuse"] is True
    assert franka.cold_seed_mode == "current_state"
    assert franka.server_runtime_contract["cold_seed_mode"] == "current_state"
    assert franka.server_runtime_contract["contract_sha256"] == (
        franka.server_runtime_contract_sha256
    )


@pytest.mark.parametrize(("raw_value", "expected"), [("0", False), ("1", True)])
def test_action_kv_env_switch_is_bound_into_both_runtime_contracts(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("N0_ACTION_DENOISE_KV_REUSE", raw_value)

    agilex = build_track3_agilex_server_config(
        training_config=twam_track3_agilex_mixed_cfg
    )
    franka = build_track32_franka_server_config()

    for config in (agilex, franka):
        assert config.action_denoise_kv_reuse is expected
        assert config.server_runtime_contract["action_denoise_kv_reuse"] is expected
        assert config.server_runtime_contract["contract_sha256"] == (
            config.server_runtime_contract_sha256
        )


@pytest.mark.parametrize("raw_value", ["", "true", "2", " 1"])
def test_action_kv_env_switch_rejects_non_binary_values(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    monkeypatch.setenv("N0_ACTION_DENOISE_KV_REUSE", raw_value)

    with pytest.raises(ValueError, match="N0_ACTION_DENOISE_KV_REUSE.*0.*1"):
        build_track3_agilex_server_config(training_config=twam_track3_agilex_mixed_cfg)
    with pytest.raises(ValueError, match="N0_ACTION_DENOISE_KV_REUSE.*0.*1"):
        build_track32_franka_server_config()
