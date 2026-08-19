# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Request-local Action Expert KV-reuse orchestration for serving."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Protocol, cast

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm  # type: ignore[import-untyped]

from n0_twam._action_kv_telemetry import ActionKVPhaseTimer

logger = logging.getLogger(__name__)


class _ActionScheduler(Protocol):
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor: ...


class _ActionKVReuseFallback(RuntimeError):
    """Signal that a cache attempt must be replayed from original noise."""


class _ActionKVReuseDispatcher:
    """Dispatch a chunk through cached or original action forwards."""

    _REQUIRED_METHODS = (
        "prepare_action_denoise_cache",
        "forward_action_denoise_cached",
        "forward_action_denoise_full",
        "forward_action_denoise_terminal",
    )
    _FALLBACK_POLICIES = {"rerun_full_v1", "fail_closed_v1"}

    def __init__(
        self,
        *,
        transformer: Any,
        cache_name: str,
        requested: bool,
        fallback_policy: str,
        request_signature: str,
        accelerator_sync: Callable[[], None] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        if fallback_policy not in self._FALLBACK_POLICIES:
            raise ValueError(
                "action_denoise_kv_fallback_policy must be one of "
                f"{sorted(self._FALLBACK_POLICIES)}, got {fallback_policy!r}"
            )
        self.transformer = transformer
        self.cache_name = str(cache_name)
        self.requested = bool(requested)
        self.fallback_policy = str(fallback_policy)
        self.request_signature = str(request_signature)
        self._phase_timer = ActionKVPhaseTimer(
            accelerator_sync=accelerator_sync,
            monotonic_clock=monotonic_clock,
        )
        self.supported = bool(
            getattr(transformer, "supports_action_denoise_kv_reuse", False)
        ) and all(
            callable(getattr(transformer, method_name, None))
            for method_name in self._REQUIRED_METHODS
        )
        self.active = self.requested and self.supported
        self.prepared = False
        self.action_cache: Any = None
        self.cached_steps = 0
        self.attempted_cached_steps = 0
        self.full_reference_steps = 0
        self.reran_full = False
        if not self.requested:
            self.fallback_reason: str | None = "disabled_by_config"
        elif not self.supported:
            self.fallback_reason = "capability_unavailable"
        else:
            self.fallback_reason = None

    def disable(self, reason: str) -> None:
        """Select the original path before a cached step is used."""

        if self.prepared or self.attempted_cached_steps:
            raise RuntimeError("cannot disable Action KV reuse after prepare")
        self.active = False
        self.fallback_reason = str(reason)

    def restart_full(self, reason: str) -> None:
        """Discard the request-local cache and mark a full replay."""

        self.active = False
        self.prepared = False
        self.action_cache = None
        self.reran_full = True
        self.fallback_reason = str(reason)

    def _full(self, input_dict: dict[str, Any], *, last_step: bool) -> Any:
        update_cache = 1 if last_step else 0
        if self.supported:
            self.full_reference_steps += 1
            return self._phase_timer.measure(
                "full_reference",
                self.transformer.forward_action_denoise_full,
                input_dict,
                cache_name=self.cache_name,
                request_signature=self.request_signature,
                update_cache=update_cache,
            )
        return self.transformer(
            input_dict,
            update_cache=update_cache,
            cache_name=self.cache_name,
            action_mode=True,
        )

    def predict(self, input_dict: dict[str, Any], *, last_step: bool) -> Any:
        """Predict one step without exposing an incompatible cache result."""

        if not self.active:
            return self._full(input_dict, last_step=last_step)
        if last_step:
            if not self.prepared:
                return self._full(input_dict, last_step=True)
            return self._phase_timer.measure(
                "terminal",
                self.transformer.forward_action_denoise_terminal,
                input_dict,
                cache_name=self.cache_name,
                request_signature=self.request_signature,
            )
        if not self.prepared:
            fixed_context = input_dict.get("tactile_global_latent")
            if not torch.is_tensor(fixed_context):
                self.disable("no_fixed_context_tokens")
                return self._full(input_dict, last_step=False)
            fixed_context = cast(torch.Tensor, fixed_context)
            if fixed_context.numel() == 0:
                self.disable("no_fixed_context_tokens")
                return self._full(input_dict, last_step=False)
            prediction, self.action_cache = self._phase_timer.measure(
                "prepare",
                self.transformer.prepare_action_denoise_cache,
                input_dict,
                cache_name=self.cache_name,
                request_signature=self.request_signature,
            )
            self.prepared = True
            return prediction

        self.attempted_cached_steps += 1
        prediction, execution = self._phase_timer.measure(
            "cached",
            self.transformer.forward_action_denoise_cached,
            input_dict,
            cache_name=self.cache_name,
            action_cache=self.action_cache,
            request_signature=self.request_signature,
        )
        if bool(getattr(execution, "used_cache", False)):
            self.cached_steps += 1
            return prediction

        reason = str(
            getattr(execution, "fallback_reason", None) or "model_rejected_action_cache"
        )
        self.fallback_reason = reason
        if self.fallback_policy == "rerun_full_v1":
            raise _ActionKVReuseFallback(reason)
        raise RuntimeError(
            "Action Expert KV cache became incompatible and fail-closed policy "
            f"is active: {reason}"
        )

    def receipt(self) -> dict[str, Any]:
        """Return a JSON-safe per-chunk benchmark/fallback receipt."""

        return {
            "schema_version": 1,
            "requested": self.requested,
            "supported": self.supported,
            "attempted": self.prepared or self.attempted_cached_steps > 0,
            "used_fast_path": self.cached_steps > 0 and not self.reran_full,
            "cached_steps": self.cached_steps,
            "attempted_cached_steps": self.attempted_cached_steps,
            "full_reference_steps": self.full_reference_steps,
            "fallback_reason": self.fallback_reason,
            "reran_full": self.reran_full,
            "request_signature": self.request_signature,
            "phase_latency_ms": self._phase_timer.receipt(),
        }

    def replay_full(
        self,
        operation: Callable[[torch.Tensor], torch.Tensor],
        initial_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self._phase_timer.measure("fallback_replay", operation, initial_actions)


class ActionKVServerMixin:
    """Shared AgileX/Franka action-denoise transaction implementation."""

    cache_name: str
    job_config: Any
    transformer: Any
    device: torch.device
    dtype: torch.dtype
    action_per_frame: int
    action_scheduler: _ActionScheduler
    _action_kv_request_sequence: int
    _action_kv_episode_sequence: int
    _action_kv_prompt_sha256: str
    _last_action_kv_reuse_receipt: dict[str, Any] | None
    _uses_pi05_delta_actions: Callable[[], bool]
    preprocess_action: Callable[..., torch.Tensor]
    _prepare_latent_input: Callable[..., dict[str, Any]]
    _repeat_input_for_cfg: Callable[[dict[str, Any]], dict[str, Any]]

    def _next_action_kv_request_signature(self, frame_st_id: int) -> str:
        sequence = int(getattr(self, "_action_kv_request_sequence", 0)) + 1
        self._action_kv_request_sequence = sequence
        config = self.job_config
        route = getattr(
            config, "serve_task", getattr(config, "action_schema", "unspecified")
        )
        identity = {
            "schema_version": 1,
            "cache_name": str(self.cache_name),
            "episode_sequence": int(getattr(self, "_action_kv_episode_sequence", 0)),
            "frame_start": int(frame_st_id),
            "request_sequence": sequence,
            "tactile_profile": str(getattr(config, "tactile_profile", "unspecified")),
            "route": str(route),
            "prompt_sha256": str(
                getattr(self, "_action_kv_prompt_sha256", "uninitialized")
            ),
            "runtime_contract_sha256": str(
                getattr(config, "server_runtime_contract_sha256", "unbound")
            ),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"action-kv-v1:{digest}"

    def _build_action_condition(
        self,
        obs: dict[str, Any],
        frame_st_id: int,
    ) -> torch.Tensor | None:
        if frame_st_id != 0:
            return None
        cold_seed_mode = (
            "zeros"
            if self._uses_pi05_delta_actions()
            else str(getattr(self.job_config, "cold_seed_mode", "free")).lower()
        )
        current_state = obs.get("current_state")
        if cold_seed_mode == "free":
            return None
        if cold_seed_mode == "current_state" and current_state is not None:
            state = np.asarray(current_state, dtype=np.float32).reshape(-1)
            action_dim = int(self.job_config.action_dim)
            if state.shape[0] < action_dim:
                state = np.pad(state, (0, action_dim - state.shape[0]))
            chunk = np.repeat(
                state[:action_dim].reshape(-1, 1, 1),
                self.action_per_frame,
                axis=2,
            )
            return self.preprocess_action(chunk).to(
                device=self.device,
                dtype=self.dtype,
            )
        return torch.zeros(
            [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
            device=self.device,
            dtype=self.dtype,
        )

    def _denoise_actions_pass(
        self,
        *,
        obs: dict[str, Any],
        frame_st_id: int,
        frame_chunk_size: int,
        action_timesteps: torch.Tensor,
        actions: torch.Tensor,
        tactile_latents: dict[str, torch.Tensor] | None,
        wrench_condition: dict[str, torch.Tensor] | None,
        dispatcher: _ActionKVReuseDispatcher,
    ) -> torch.Tensor:
        for index, timestep in enumerate(
            tqdm(
                action_timesteps,
                disable=not bool(
                    getattr(self.job_config, "show_inference_progress", True)
                ),
            )
        ):
            last_step = index == len(action_timesteps) - 1
            action_cond = self._build_action_condition(obs, frame_st_id)
            input_dict = self._prepare_latent_input(
                None,
                actions,
                timestep,
                timestep,
                None,
                action_cond,
                frame_st_id=frame_st_id,
                tactile_latents=tactile_latents,
                wrench_condition=wrench_condition,
            )
            action_noise_pred = dispatcher.predict(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                last_step=last_step,
            )
            if not last_step:
                action_noise_pred = rearrange(
                    action_noise_pred,
                    "b (f n) c -> b c f n 1",
                    f=frame_chunk_size,
                )
                if self.job_config.action_guidance_scale > 1:
                    action_noise_pred = action_noise_pred[
                        1:
                    ] + self.job_config.action_guidance_scale * (
                        action_noise_pred[:1] - action_noise_pred[1:]
                    )
                else:
                    action_noise_pred = action_noise_pred[:1]
                actions = self.action_scheduler.step(
                    action_noise_pred,
                    timestep,
                    actions,
                    return_dict=False,
                )
            if action_cond is not None:
                actions[:, :, 0:1] = action_cond
        return actions

    def _denoise_actions(
        self,
        *,
        obs: dict[str, Any],
        frame_st_id: int,
        frame_chunk_size: int,
        action_timesteps: torch.Tensor,
        actions: torch.Tensor,
        tactile_latents: dict[str, torch.Tensor] | None,
        wrench_condition: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Denoise a chunk with transactional request-local KV reuse."""

        dispatcher = _ActionKVReuseDispatcher(
            transformer=self.transformer,
            cache_name=self.cache_name,
            requested=bool(getattr(self.job_config, "action_denoise_kv_reuse", False)),
            fallback_policy=str(
                getattr(
                    self.job_config,
                    "action_denoise_kv_fallback_policy",
                    "rerun_full_v1",
                )
            ),
            request_signature=self._next_action_kv_request_signature(frame_st_id),
            accelerator_sync=getattr(self, "_action_kv_accelerator_sync", None),
            monotonic_clock=getattr(self, "_action_kv_monotonic_clock", None),
        )
        if dispatcher.active and len(action_timesteps) <= 2:
            dispatcher.disable("no_reusable_middle_step")

        initial_actions = actions.detach().clone()

        def run_pass(sample: torch.Tensor) -> torch.Tensor:
            return self._denoise_actions_pass(
                obs=obs,
                frame_st_id=frame_st_id,
                frame_chunk_size=frame_chunk_size,
                action_timesteps=action_timesteps,
                actions=sample,
                tactile_latents=tactile_latents,
                wrench_condition=wrench_condition,
                dispatcher=dispatcher,
            )

        try:
            actions = run_pass(actions)
        except _ActionKVReuseFallback as error:
            dispatcher.restart_full(str(error))
            actions = dispatcher.replay_full(run_pass, initial_actions.clone())

        receipt = dispatcher.receipt()
        self._last_action_kv_reuse_receipt = receipt
        logger.info("[action-kv-reuse] %s", json.dumps(receipt, sort_keys=True))
        return actions
