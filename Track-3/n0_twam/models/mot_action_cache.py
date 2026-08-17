# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Typed action-denoise cache orchestration for the MoT backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ActionCacheExecution:
    """Result metadata for one action-denoise dispatch."""

    used_cache: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class ActionDenoiseCache:
    """Immutable per-layer K/V snapshot used by action-only denoising."""

    per_layer_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    fixed_output: torch.Tensor
    action_start: int
    action_stop: int
    cached_action_start: int
    cached_action_stop: int
    action_rows_mask: torch.Tensor
    full_rows_mask: torch.Tensor
    action_rows_unmasked: bool
    batch_size: int
    sequence_length: int
    hidden_dim: int
    dtype: torch.dtype
    device: torch.device
    slices: tuple[tuple[str, int, int], ...]
    cache_name: str | None
    rolling_revisions: tuple[tuple[int, int], ...]


class MoTActionCacheMixin:
    """Mixin implementing prefill, validation, and safe full fallback."""

    __call__: Any
    forward_action_cached: Any
    shared_attn: Any
    _cross_masks: dict[str, torch.Tensor]

    @staticmethod
    def _validated_action_slice(
        action_cols: slice, sequence_length: int
    ) -> tuple[int, int]:
        if not isinstance(action_cols, slice) or action_cols.step not in (None, 1):
            raise ValueError("action_cols must be a contiguous slice")
        start = 0 if action_cols.start is None else int(action_cols.start)
        stop = sequence_length if action_cols.stop is None else int(action_cols.stop)
        if start < 0 or stop <= start or stop > sequence_length:
            raise ValueError(
                "action_cols must select a non-empty in-range token interval"
            )
        return start, stop

    def prefill_action_denoise_cache(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep_proj: torch.Tensor,
        temb: torch.Tensor | None,
        rotary_emb: torch.Tensor,
        slices: tuple[tuple[str, int, int], ...],
        *,
        action_cols: slice,
        action_rows_mask: torch.Tensor,
        cache_name: str | None = None,
    ) -> ActionDenoiseCache:
        """Prefill fixed-context K/V without modifying the rolling cache."""

        sequence_length = int(hidden_states.shape[1])
        action_start, action_stop = self._validated_action_slice(
            action_cols, sequence_length
        )
        action_length = action_stop - action_start
        if action_rows_mask.dtype is not torch.bool:
            raise ValueError("action_rows_mask must have boolean dtype")
        if tuple(action_rows_mask.shape) != (action_length, sequence_length):
            raise ValueError(
                "action_rows_mask must have shape "
                f"({action_length}, {sequence_length})"
            )
        non_action_rows = torch.cat(
            [
                torch.arange(0, action_start, device=action_rows_mask.device),
                torch.arange(
                    action_stop,
                    sequence_length,
                    device=action_rows_mask.device,
                ),
            ]
        )
        full_mask = self.shared_attn[0]._dense_mask
        if full_mask is None:
            raise ValueError("action cache prefill requires a dense fixed-context mask")
        if full_mask.dtype is not torch.bool or tuple(full_mask.shape) != (
            sequence_length,
            sequence_length,
        ):
            raise ValueError(
                "action cache fixed-context mask must be boolean with shape "
                f"({sequence_length}, {sequence_length})"
            )
        for layer, attention in enumerate(self.shared_attn[1:], start=1):
            if attention._dense_mask is not full_mask:
                raise ValueError(
                    f"MoT layer {layer} does not share the action cache mask"
                )
        full_mask = full_mask.to(device=action_rows_mask.device)
        if not torch.equal(full_mask[action_start:action_stop], action_rows_mask):
            raise ValueError("action cache row mask differs from the MoT mask")
        if non_action_rows.numel() and bool(
            full_mask[non_action_rows, action_start:action_stop].any().item()
        ):
            raise ValueError("fixed-context rows must not attend noisy action columns")
        action_rows_unmasked = bool(action_rows_mask.all().item())
        output, current_kv = self(
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            temb,
            rotary_emb,
            slices,
            collect_cache=True,
            update_cache=0,
            cache_name=cache_name,
        )
        combined: list[tuple[torch.Tensor, torch.Tensor]] = []
        revisions: list[tuple[int, int]] = []
        prefix_length: int | None = None
        for attention, (current_k, current_v) in zip(
            self.shared_attn, current_kv, strict=True
        ):
            prefix_k, prefix_v, revision = attention.snapshot_valid_kv(cache_name)
            if prefix_k is None or prefix_v is None:
                layer_prefix_length = 0
                combined_k = current_k.detach().clone()
                combined_v = current_v.detach().clone()
            else:
                layer_prefix_length = int(prefix_k.shape[1])
                combined_k = torch.cat([prefix_k, current_k.detach()], dim=1).clone()
                combined_v = torch.cat([prefix_v, current_v.detach()], dim=1).clone()
            if prefix_length is None:
                prefix_length = layer_prefix_length
            elif prefix_length != layer_prefix_length:
                raise RuntimeError(
                    "rolling action cache token counts differ between MoT layers"
                )
            combined.append((combined_k, combined_v))
            revisions.append(revision)

        prefix_length = 0 if prefix_length is None else prefix_length
        prefix_mask = torch.ones(
            action_length,
            prefix_length,
            dtype=torch.bool,
            device=action_rows_mask.device,
        )
        combined_rows_mask = torch.cat(
            [prefix_mask, action_rows_mask.detach().clone()], dim=1
        )
        full_prefix_mask = torch.ones(
            sequence_length,
            prefix_length,
            dtype=torch.bool,
            device=action_rows_mask.device,
        )
        combined_full_rows_mask = torch.cat(
            [full_prefix_mask, full_mask.detach().clone()], dim=1
        )
        return ActionDenoiseCache(
            per_layer_kv=tuple(combined),
            fixed_output=output.detach().clone(),
            action_start=action_start,
            action_stop=action_stop,
            cached_action_start=prefix_length + action_start,
            cached_action_stop=prefix_length + action_stop,
            action_rows_mask=combined_rows_mask,
            full_rows_mask=combined_full_rows_mask,
            action_rows_unmasked=action_rows_unmasked,
            batch_size=int(hidden_states.shape[0]),
            sequence_length=sequence_length,
            hidden_dim=int(hidden_states.shape[2]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            slices=tuple(slices),
            cache_name=cache_name,
            rolling_revisions=tuple(revisions),
        )

    def _action_cache_incompatibility(
        self,
        *,
        hidden_states: torch.Tensor,
        slices: tuple[tuple[str, int, int], ...],
        action_cols: slice,
        action_rows_mask: torch.Tensor,
        action_cache: ActionDenoiseCache,
        cache_name: str | None,
    ) -> str | None:
        action_start, action_stop = self._validated_action_slice(
            action_cols, int(hidden_states.shape[1])
        )
        checks = (
            (int(hidden_states.shape[0]) == action_cache.batch_size, "batch mismatch"),
            (
                int(hidden_states.shape[1]) == action_cache.sequence_length,
                "sequence mismatch",
            ),
            (int(hidden_states.shape[2]) == action_cache.hidden_dim, "hidden mismatch"),
            (hidden_states.dtype == action_cache.dtype, "dtype mismatch"),
            (hidden_states.device == action_cache.device, "device mismatch"),
            (tuple(slices) == action_cache.slices, "slice layout mismatch"),
            (
                (action_start, action_stop)
                == (action_cache.action_start, action_cache.action_stop),
                "action columns mismatch",
            ),
            (cache_name == action_cache.cache_name, "rolling cache name mismatch"),
        )
        for compatible, reason in checks:
            if not compatible:
                return reason
        expected_mask = action_cache.action_rows_mask[
            :, -action_cache.sequence_length :
        ]
        if action_rows_mask.dtype is not torch.bool or tuple(
            action_rows_mask.shape
        ) != tuple(expected_mask.shape):
            return "action rows mask mismatch"
        # ``action_rows_unmasked`` was proven once during prefill.  The model
        # runtime rebuilds the same fully-visible action-row contract each
        # denoise step, so avoid a device-synchronising ``torch.equal`` here.
        if not action_cache.action_rows_unmasked and not torch.equal(
            action_rows_mask, expected_mask
        ):
            return "action rows mask mismatch"
        revisions = tuple(
            attention.cache_revision(cache_name) for attention in self.shared_attn
        )
        if revisions != action_cache.rolling_revisions:
            return "rolling cache revision mismatch"
        return None

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def forward_action_cached_or_full(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep_proj: torch.Tensor,
        temb: torch.Tensor | None,
        rotary_emb: torch.Tensor,
        slices: tuple[tuple[str, int, int], ...],
        *,
        action_cols: slice,
        action_rows_mask: torch.Tensor,
        action_cache: ActionDenoiseCache,
        cache_name: str | None = None,
    ) -> tuple[torch.Tensor, ActionCacheExecution]:
        """Run the action-only fast path, or a safe full-forward fallback."""

        reason = self._action_cache_incompatibility(
            hidden_states=hidden_states,
            slices=slices,
            action_cols=action_cols,
            action_rows_mask=action_rows_mask,
            action_cache=action_cache,
            cache_name=cache_name,
        )
        if reason is not None:
            full = self(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                temb,
                rotary_emb,
                slices,
                update_cache=0,
                cache_name=cache_name,
            )
            start, stop = self._validated_action_slice(
                action_cols, int(hidden_states.shape[1])
            )
            return full[:, start:stop], ActionCacheExecution(False, reason)

        action_temb = (
            None
            if temb is None
            else temb[:, action_cache.action_start : action_cache.action_stop]
        )
        output = self.forward_action_cached(
            hidden_states[:, action_cache.action_start : action_cache.action_stop],
            timestep_proj[:, action_cache.action_start : action_cache.action_stop],
            action_temb,
            rotary_emb[:, action_cache.action_start : action_cache.action_stop],
            action_cache.per_layer_kv,
            slice(
                action_cache.cached_action_start,
                action_cache.cached_action_stop,
            ),
            action_cache.action_rows_mask,
            action_cache.full_rows_mask,
            slice(action_cache.action_start, action_cache.action_stop),
            encoder_hidden_states=encoder_hidden_states,
            cross_attn_mask=self._cross_masks.get("action"),
        )
        return output, ActionCacheExecution(True, None)
