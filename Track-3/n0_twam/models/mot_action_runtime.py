# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Request-local model runtime for MoT Action Expert KV reuse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast

import torch

from .mot_action_cache import ActionCacheExecution, ActionDenoiseCache
from .mot_action_identity import (
    ActionDenoiseIdentity,
    action_denoise_identity_mismatch,
    build_action_denoise_identity,
)

_RuntimeMode = Literal["prepare", "cached", "full", "terminal"]


@dataclass(frozen=True)
class ActionDenoiseModelCache:
    """Model-level cache carrying backbone K/V and request identity."""

    backbone_cache: ActionDenoiseCache
    identity: ActionDenoiseIdentity


@dataclass(frozen=True)
class _ActionRuntimeRequest:
    mode: _RuntimeMode
    identity: ActionDenoiseIdentity
    action_cache: ActionDenoiseModelCache | None = None


@dataclass(frozen=True)
class _ActionRuntimeResult:
    backbone_cache: ActionDenoiseCache | None = None
    execution: ActionCacheExecution | None = None


class MoTActionRuntimeMixin:
    """Public model API and backbone dispatch for cached action denoising."""

    supports_action_denoise_kv_reuse = True
    mot: Any
    __call__: Any

    def _execute_action_runtime(
        self,
        input_dict: Mapping[str, Any],
        *,
        request: _ActionRuntimeRequest,
        update_cache: int,
        cache_name: str,
    ) -> tuple[torch.Tensor, _ActionRuntimeResult]:
        if hasattr(self, "_mot_action_runtime_request"):
            raise RuntimeError("action cache runtime is not re-entrant")
        self._mot_action_runtime_request = request
        try:
            # FSDP2 owns root parameters via forward pre/post hooks, so custom
            # serving methods must enter through nn.Module.__call__, not forward.
            output = self(
                input_dict,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=True,
            )
            result = getattr(self, "_mot_action_runtime_result", None)
            if not isinstance(result, _ActionRuntimeResult):
                raise RuntimeError("MoT action runtime did not publish a result")
            return output, result
        finally:
            if hasattr(self, "_mot_action_runtime_result"):
                del self._mot_action_runtime_result
            if hasattr(self, "_mot_action_runtime_request"):
                del self._mot_action_runtime_request

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def prepare_action_denoise_cache(
        self,
        input_dict: Mapping[str, Any],
        *,
        cache_name: str,
        request_signature: str,
    ) -> tuple[torch.Tensor, ActionDenoiseModelCache]:
        identity = build_action_denoise_identity(
            input_dict,
            cache_name=cache_name,
            request_signature=request_signature,
        )
        output, result = self._execute_action_runtime(
            input_dict,
            request=_ActionRuntimeRequest("prepare", identity),
            update_cache=0,
            cache_name=cache_name,
        )
        if result.backbone_cache is None:
            raise RuntimeError("action cache prefill did not return backbone K/V")
        return output, ActionDenoiseModelCache(result.backbone_cache, identity)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def forward_action_denoise_cached(
        self,
        input_dict: Mapping[str, Any],
        *,
        cache_name: str,
        action_cache: ActionDenoiseModelCache,
        request_signature: str,
    ) -> tuple[torch.Tensor, ActionCacheExecution]:
        if not isinstance(action_cache, ActionDenoiseModelCache):
            raise TypeError("action_cache must be an ActionDenoiseModelCache")
        identity = build_action_denoise_identity(
            input_dict,
            cache_name=cache_name,
            request_signature=request_signature,
        )
        reason = action_denoise_identity_mismatch(action_cache.identity, identity)
        mode: _RuntimeMode = "cached" if reason is None else "full"
        output, result = self._execute_action_runtime(
            input_dict,
            request=_ActionRuntimeRequest(mode, identity, action_cache),
            update_cache=0,
            cache_name=cache_name,
        )
        execution = result.execution
        if reason is not None:
            execution = ActionCacheExecution(False, reason)
        if execution is None:
            raise RuntimeError(
                "cached action runtime did not return execution metadata"
            )
        return output, execution

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def forward_action_denoise_full(
        self,
        input_dict: Mapping[str, Any],
        *,
        cache_name: str,
        request_signature: str,
        update_cache: int,
    ) -> torch.Tensor:
        """Run the causal full reference with an explicit cache-write policy."""

        if isinstance(update_cache, bool) or update_cache not in (0, 1):
            raise ValueError("update_cache must be the integer 0 or 1")
        identity = build_action_denoise_identity(
            input_dict,
            cache_name=cache_name,
            request_signature=request_signature,
        )
        output, _ = self._execute_action_runtime(
            input_dict,
            request=_ActionRuntimeRequest("full", identity),
            update_cache=update_cache,
            cache_name=cache_name,
        )
        return output

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def forward_action_denoise_terminal(
        self,
        input_dict: Mapping[str, Any],
        *,
        cache_name: str,
        request_signature: str,
    ) -> torch.Tensor:
        identity = build_action_denoise_identity(
            input_dict,
            cache_name=cache_name,
            request_signature=request_signature,
        )
        output, _ = self._execute_action_runtime(
            input_dict,
            request=_ActionRuntimeRequest("terminal", identity),
            update_cache=1,
            cache_name=cache_name,
        )
        return output

    def _run_action_runtime_backbone(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep_proj: torch.Tensor,
        temb: torch.Tensor | None,
        rotary_emb: torch.Tensor,
        slices: tuple[tuple[str, int, int], ...],
        *,
        update_cache: int,
        cache_name: str,
    ) -> torch.Tensor:
        request = self._mot_action_runtime_request
        action_stop = slices[1][2]
        sequence_length = int(hidden_states.shape[1])
        if action_stop <= 0 or action_stop > sequence_length:
            raise ValueError("action runtime requires a non-empty action token block")
        action_cols = slice(0, action_stop)
        dense_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        dense_mask[action_stop:, action_cols] = False
        action_rows_mask = dense_mask[action_cols]
        self.mot.set_masks(dense_self_mask=dense_mask, cross_masks={})

        if request.mode == "prepare":
            backbone_cache = self.mot.prefill_action_denoise_cache(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                temb,
                rotary_emb,
                slices,
                action_cols=action_cols,
                action_rows_mask=action_rows_mask,
                cache_name=cache_name,
            )
            self._mot_action_runtime_result = _ActionRuntimeResult(
                backbone_cache=backbone_cache
            )
            return cast(torch.Tensor, backbone_cache.fixed_output)

        if request.mode == "cached":
            if request.action_cache is None:
                raise RuntimeError("cached action runtime requires action_cache")
            action_output, execution = self.mot.forward_action_cached_or_full(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                temb,
                rotary_emb,
                slices,
                action_cols=action_cols,
                action_rows_mask=action_rows_mask,
                action_cache=request.action_cache.backbone_cache,
                cache_name=cache_name,
            )
            if execution.used_cache:
                output = request.action_cache.backbone_cache.fixed_output.clone()
                output[:, action_cols] = action_output
            else:
                # Re-run full: low-level fallback returns action rows only.
                output = self.mot(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    temb,
                    rotary_emb,
                    slices,
                    update_cache=0,
                    cache_name=cache_name,
                )
            self._mot_action_runtime_result = _ActionRuntimeResult(execution=execution)
            return output
        output = self.mot(
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            temb,
            rotary_emb,
            slices,
            update_cache=update_cache,
            cache_name=cache_name,
        )
        self._mot_action_runtime_result = _ActionRuntimeResult(
            execution=(
                ActionCacheExecution(False, "model cache identity mismatch")
                if request.mode == "full"
                else None
            )
        )
        return cast(torch.Tensor, output)
