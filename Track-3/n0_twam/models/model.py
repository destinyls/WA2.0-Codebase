# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)
from torch.utils.checkpoint import checkpoint as _checkpoint

from n0_twam.utils.utils import get_mesh_id

from .wrench_conditioner import WrenchConditioner

try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        # flash-attn is optional: serving uses attn_mode='torch' (SDPA) and
        # training uses 'flex'. Only attn_mode='flashattn' needs the package.
        flash_attn_func = None

__all__ = ["WanTransformer3DModel", "capture_attention_execution_contract"]


def custom_sdpa(q, k, v, attn_mask=None):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask
    )
    return out.transpose(1, 2)


def _uses_eager_block_mask_creation(torch_version: str) -> bool:
    """Return whether Torch needs the 2.5 FlexAttention mask workaround."""
    release = torch_version.split("+", maxsplit=1)[0]
    version_parts = release.split(".")
    return len(version_parts) >= 2 and version_parts[:2] == ["2", "5"]


def _vmap_safe_index_1d(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Read one value without Tensor.__getitem__ converting a BatchedTensor to int."""
    # FlexAttention's compiled HOP uses int32 q/kv indices on the HCU vendor
    # Torch build, while ``gather`` requires int64 indices.  The eager block-mask
    # builder happens to provide int64 indices, so keep the conversion here where
    # both execution paths share the exact same lookup semantics.
    # The vendor Flex HOP can expose q/kv indices as CPU FakeTensors while the
    # captured lookup table is placed on the accelerator.  Normalize both dtype
    # and device before ``gather`` so FakeTensor propagation sees one device.
    gather_index = index.to(device=values.device, dtype=torch.long).reshape(1)
    return torch.gather(values, dim=0, index=gather_index).squeeze(0)


def _resolve_flex_attention_backend() -> str:
    """Resolve the sparse-attention backend without silently accepting typos."""
    override = os.environ.get("N0_FLEX_ATTENTION_BACKEND")
    supported_backends = ("flex", "grouped_sdpa", "grouped_flash_attn")
    if override is not None:
        if override not in supported_backends:
            raise ValueError(
                "N0_FLEX_ATTENTION_BACKEND must be one of "
                f"{supported_backends}, got {override!r}"
            )
        if override == "grouped_flash_attn" and flash_attn_func is None:
            raise RuntimeError(
                "grouped_flash_attn requires the vendor flash-attn package"
            )
        return override
    hip_version = getattr(torch.version, "hip", None)
    if _uses_eager_block_mask_creation(torch.__version__) and hip_version is not None:
        return "grouped_flash_attn" if flash_attn_func is not None else "grouped_sdpa"
    return "flex"


def _grouped_sdpa_query_token_limit() -> int:
    """Return the explicitly bounded grouped-SDPA query length."""
    raw_limit = os.environ.get("N0_GROUPED_SDPA_MAX_QUERY_TOKENS", "16384")
    if not raw_limit or any(character not in "0123456789" for character in raw_limit):
        raise ValueError(
            "N0_GROUPED_SDPA_MAX_QUERY_TOKENS must be a positive base-10 integer, "
            f"got {raw_limit!r}"
        )
    limit = int(raw_limit)
    if limit <= 0:
        raise ValueError("N0_GROUPED_SDPA_MAX_QUERY_TOKENS must be greater than zero")
    return limit


def _resolve_mot_cross_attention_backend() -> str:
    """Resolve the MoT text cross-attention kernel without silent fallback."""
    backend = os.environ.get("N0_MOT_CROSS_ATTENTION_BACKEND", "sdpa")
    if backend not in ("sdpa", "flash_attn"):
        raise ValueError("N0_MOT_CROSS_ATTENTION_BACKEND must be sdpa or flash_attn")
    if backend == "flash_attn" and flash_attn_func is None:
        raise RuntimeError(
            "N0_MOT_CROSS_ATTENTION_BACKEND=flash_attn requires the vendor "
            "flash-attn package"
        )
    return backend


def capture_attention_execution_contract() -> dict[str, object]:
    """Return the resolved sparse-attention settings used by new modules."""
    backend = _resolve_flex_attention_backend()
    return {
        "attention_backend": backend,
        "grouped_sdpa_max_query_tokens": (
            _grouped_sdpa_query_token_limit()
            if backend in ("grouped_sdpa", "grouped_flash_attn")
            else None
        ),
        "mot_cross_attention_backend": _resolve_mot_cross_attention_backend(),
    }


@dataclass(frozen=True)
class GroupedAttentionMask:
    """Sparse attention mask represented by disjoint query/key index groups."""

    shape: tuple[int, int, int, int]
    groups: tuple[tuple[torch.Tensor, torch.Tensor], ...]


class FlexAttnFunc(nn.Module):
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention,
        dynamic=True,
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)

    def __init__(
        self,
        is_cross=False,
    ) -> None:
        super().__init__()
        self.is_cross = is_cross
        self.attention_backend = _resolve_flex_attention_backend()
        self.block_mask: BlockMask | GroupedAttentionMask | None = None

    def set_block_mask(
        self,
        block_mask: BlockMask | GroupedAttentionMask | None,
    ) -> None:
        self.block_mask = block_mask

    @staticmethod
    def _validate_grouped_mask(
        mask: GroupedAttentionMask,
        query_length: int,
        key_value_length: int,
        device: torch.device,
    ) -> None:
        query_groups = []
        key_value_groups = []
        for group_index, (query_indices, key_value_indices) in enumerate(mask.groups):
            for label, indices in (
                ("query", query_indices),
                ("key/value", key_value_indices),
            ):
                if indices.ndim != 1:
                    raise ValueError(
                        f"group {group_index} {label} indices must be one-dimensional"
                    )
                if indices.dtype != torch.long:
                    raise TypeError(
                        f"group {group_index} {label} indices must use torch.long"
                    )
                if indices.device != device:
                    raise ValueError(
                        f"group {group_index} {label} indices must be on "
                        f"{device}, got {indices.device}"
                    )
            if query_indices.numel() > 0:
                query_groups.append(query_indices)
            if key_value_indices.numel() > 0:
                key_value_groups.append(key_value_indices)
        if query_groups:
            all_query_indices = torch.cat(query_groups)
            if bool(torch.any(all_query_indices < 0)) or bool(
                torch.any(all_query_indices >= query_length)
            ):
                raise IndexError(
                    "Grouped SDPA query indices are out of bounds for length "
                    f"{query_length}"
                )
            if torch.unique(all_query_indices).numel() != all_query_indices.numel():
                raise ValueError("Grouped SDPA query groups must be mutually exclusive")
        if key_value_groups:
            all_key_value_indices = torch.cat(key_value_groups)
            if bool(torch.any(all_key_value_indices < 0)) or bool(
                torch.any(all_key_value_indices >= key_value_length)
            ):
                raise IndexError(
                    "Grouped SDPA key/value indices are out of bounds for length "
                    f"{key_value_length}"
                )

    @staticmethod
    def _run_grouped_sdpa_group(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_indices: torch.Tensor,
        key_value_indices: torch.Tensor,
    ) -> torch.Tensor:
        grouped_query = query.index_select(2, query_indices)
        grouped_key = key.index_select(2, key_value_indices)
        grouped_value = value.index_select(2, key_value_indices)
        return F.scaled_dot_product_attention(
            grouped_query,
            grouped_key,
            grouped_value,
        )

    @staticmethod
    def _run_grouped_flash_attention_group(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_indices: torch.Tensor,
        key_value_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run one exact grouped mask partition with the vendor fused kernel."""
        if flash_attn_func is None:
            raise RuntimeError(
                "grouped_flash_attn requires the vendor flash-attn package"
            )
        grouped_query = query.index_select(2, query_indices).transpose(1, 2)
        grouped_key = key.index_select(2, key_value_indices).transpose(1, 2)
        grouped_value = value.index_select(2, key_value_indices).transpose(1, 2)
        output = flash_attn_func(
            grouped_query.contiguous(),
            grouped_key.contiguous(),
            grouped_value.contiguous(),
            dropout_p=0.0,
            causal=False,
        )
        return output.transpose(1, 2).contiguous()

    @staticmethod
    def _grouped_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: GroupedAttentionMask,
        *,
        backend: str = "grouped_sdpa",
    ) -> torch.Tensor:
        if query.device != key.device or key.device != value.device:
            raise ValueError("Grouped SDPA query, key, and value devices must match")
        if key.shape[2] != value.shape[2]:
            raise ValueError("Grouped SDPA key and value lengths must match")
        query_length = query.shape[2]
        key_value_length = key.shape[2]
        query_limit = _grouped_sdpa_query_token_limit()
        if query_length > query_limit:
            raise ValueError(
                "Grouped SDPA query length exceeds the fail-closed safety limit: "
                f"query_length={query_length}, limit={query_limit}. Raise "
                "N0_GROUPED_SDPA_MAX_QUERY_TOKENS only after a measured HCU "
                "memory and step-time gate."
            )
        FlexAttnFunc._validate_grouped_mask(
            mask,
            query_length,
            key_value_length,
            query.device,
        )

        if backend == "grouped_sdpa":
            group_operation = FlexAttnFunc._run_grouped_sdpa_group
        elif backend == "grouped_flash_attn":
            group_operation = FlexAttnFunc._run_grouped_flash_attention_group
        else:
            raise ValueError(f"unsupported grouped attention backend: {backend!r}")

        output = torch.zeros_like(query)
        attended_query_groups = []
        output_groups = []
        requires_grad = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (query, key, value)
        )
        for query_indices, key_value_indices in mask.groups:
            if query_indices.numel() == 0 or key_value_indices.numel() == 0:
                continue
            # The vendor FlashAttention kernel owns its fused backward and this
            # grouped operation already runs inside the whole-layer MoT
            # checkpoint.  Nesting another checkpoint here only repeats index
            # selection and kernel work.  Keep the legacy checkpoint solely for
            # the math-SDPA fallback, whose peak memory still needs that guard.
            if requires_grad and backend == "grouped_sdpa":
                grouped_output = _checkpoint(
                    group_operation,
                    query,
                    key,
                    value,
                    query_indices,
                    key_value_indices,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                grouped_output = group_operation(
                    query,
                    key,
                    value,
                    query_indices,
                    key_value_indices,
                )
            attended_query_groups.append(query_indices)
            output_groups.append(grouped_output)
        if not output_groups:
            return output + (query.sum() + key.sum() + value.sum()) * 0
        return torch.index_copy(
            output,
            2,
            torch.cat(attended_query_groups),
            torch.cat(output_groups, dim=2),
        )

    @staticmethod
    def _create_block_mask(
        mask_mod: _mask_mod_signature,
        query_length: int,
        key_value_length: int,
        device: str | torch.device,
    ) -> BlockMask:
        """Create a BlockMask with the vendor Torch 2.5 compatibility path."""
        if _uses_eager_block_mask_creation(torch.__version__):
            # Torch 2.5 disables TransformGetItemToIndex inside its compiled
            # create_mask path. Keep only mask construction eager; the actual
            # FlexAttention forward/backward kernel remains compiled.
            return create_block_mask(
                mask_mod,
                1,
                1,
                query_length,
                key_value_length,
                device=device,
                _compile=False,
            )
        return FlexAttnFunc.compiled_create_block_mask(
            mask_mod,
            1,
            1,
            query_length,
            key_value_length,
            device=device,
            _compile=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = self.block_mask
        q_len = q_varlen.shape[2]
        kv_len = k_varlen.shape[2]
        if block_mask is not None:
            mask_shape = tuple(getattr(block_mask, "shape", ()))
            if len(mask_shape) < 4:
                raise ValueError("Attention mask must expose a four-dimensional shape")
            mask_q_len = mask_shape[-2]
            mask_kv_len = mask_shape[-1]
            if mask_q_len != q_len or mask_kv_len != kv_len:
                raise ValueError(
                    "FlexAttention mask length mismatch: "
                    f"mask=({mask_q_len}, {mask_kv_len}) qkv=({q_len}, {kv_len}) "
                    f"is_cross={self.is_cross}"
                )

        if block_mask is None:
            # Inference does not need the train-time sparse block mask. Falling
            # back to SDPA also avoids flex decoding compilation issues when
            # q/k lengths vary because of KV cache.
            x_out = F.scaled_dot_product_attention(q_varlen, k_varlen, v_varlen)
        elif self.attention_backend in ("grouped_sdpa", "grouped_flash_attn"):
            if not isinstance(block_mask, GroupedAttentionMask):
                raise TypeError(
                    f"{self.attention_backend} requires GroupedAttentionMask, got "
                    f"{type(block_mask).__name__}"
                )
            x_out = FlexAttnFunc._grouped_sdpa(
                q_varlen,
                k_varlen,
                v_varlen,
                block_mask,
                backend=self.attention_backend,
            )
        else:
            if isinstance(block_mask, GroupedAttentionMask):
                raise TypeError("flex backend requires a BlockMask")
            x_out = FlexAttnFunc.flex_attn(
                q_varlen,
                k_varlen,
                v_varlen,
                block_mask=block_mask,
                kernel_options={
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 64,
                    "BLOCK_M2": 64,
                    "BLOCK_N2": 32,
                },
            )

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        latent_shape,
        action_shape,
        padded_length,
        chunk_size,
        window_size,
        patch_size,
        device,
        text_token_length=512,
        tactile_token_length=0,  # total (noisy + clean) for backward-compat
        tactile_noisy_token_length=0,  # symdiff-tactile only: noisy half
        include_action_tokens=True,
        tactile_grid_shape=None,  # (B, S, Fp, Hp, Wp) → frame-aligned tactile
    ):
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        latent_seq_id = (
            torch.arange(B)[:, None, None, None]
            .expand(
                -1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2]
            )
            .flatten()
        )
        # latent_frame_id must be ONE entry per latent TOKEN (temporal token count =
        # L_F // patch_size[0]), to match latent_seq_id's length. arange(L_F) only
        # matched because patch_size[0]==1; use the patched temporal count so a future
        # temporal patch (patch_size[0]>1) can't desync the id-tensor lengths.
        latent_frame_id = (
            torch.arange(L_F // patch_size[0])[None, :, None, None]
            .expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None]
            .flatten()
        )
        seq_ids = torch.cat([latent_seq_id] * 2)
        frame_ids = torch.cat([latent_frame_id // chunk_size * 2] * 2)
        noise_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_id),
                torch.ones_like(latent_frame_id),
            ]
        )
        # modality: 0=video, 1=action, 2=tactile
        modality_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_id),
                torch.zeros_like(latent_frame_id),
            ]
        )

        if include_action_tokens:
            action_seq_id = (
                torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
            )
            action_frame_id = (
                torch.arange(A_F)[None, :, None, None]
                .expand(B, -1, A_H, A_W)[None]
                .flatten()
            )
            seq_ids = torch.cat([seq_ids, action_seq_id, action_seq_id])
            frame_ids = torch.cat(
                [
                    frame_ids,
                    action_frame_id // chunk_size * 2 + 1,
                    action_frame_id // chunk_size * 2 + 1,
                ]
            )
            noise_ids = torch.cat(
                [
                    noise_ids,
                    torch.zeros_like(action_frame_id),
                    torch.ones_like(action_frame_id),
                ]
            )
            modality_ids = torch.cat(
                [
                    modality_ids,
                    torch.ones_like(action_frame_id),
                    torch.ones_like(action_frame_id),
                ]
            )

        if tactile_token_length < 0:
            raise ValueError("tactile_token_length must be non-negative.")
        if tactile_token_length > 0 and tactile_token_length % B != 0:
            raise ValueError(
                "tactile_token_length must be divisible by batch size: "
                f"tactile_token_length={tactile_token_length}, batch_size={B}"
            )

        # Add tactile tokens to self-attention sequences.
        # Symdiff tactile: tactile_token_length = tactile_noisy_len + tactile_clean_len.
        # First the noisy half (noise_id=0), then the clean half (noise_id=1).
        # Legacy clean-only tactile: tactile_noisy_token_length=0 → all tactile is clean (legacy).
        tactile_clean_token_length = tactile_token_length - tactile_noisy_token_length

        # Frame-aligned tactile (Option 3): give each tactile token the frame_id of
        # its corresponding VIDEO frame, so tactile co-generates WITH video at the
        # same frame and obeys the same causal + window rules. Token order per batch
        # is (sensor, frame, hp, wp); frame_id mirrors video = (f // chunk_size) * 2.
        # (Legacy fallback when grid shape is unknown: all-zero blob, old behaviour.)
        def _tactile_frame_ids(token_length):
            if tactile_grid_shape is None:
                return torch.zeros(token_length, dtype=torch.long)
            _, S_t, Fp_t, Hp_t, Wp_t = tactile_grid_shape
            per_sensor = torch.arange(Fp_t).repeat_interleave(Hp_t * Wp_t)
            pattern = per_sensor.repeat(S_t)  # (S*Fp*Hp*Wp,) order (s,f,hp,wp)
            pattern = (pattern // chunk_size) * 2  # align to video frame_id
            return pattern.repeat(B).to(torch.long)  # (B*S*Fp*Hp*Wp,)

        if tactile_noisy_token_length > 0:
            per_batch_n = tactile_noisy_token_length // B
            tn_seq_id = torch.arange(B)[:, None].expand(-1, per_batch_n).flatten()
            tn_frame_id = _tactile_frame_ids(tactile_noisy_token_length)
            tn_noise_id = torch.zeros(
                tactile_noisy_token_length, dtype=torch.long
            )  # noisy
            tn_modality_id = torch.full(
                (tactile_noisy_token_length,), 2, dtype=torch.long
            )
            seq_ids = torch.cat([seq_ids, tn_seq_id])
            frame_ids = torch.cat([frame_ids, tn_frame_id])
            noise_ids = torch.cat([noise_ids, tn_noise_id])
            modality_ids = torch.cat([modality_ids, tn_modality_id])
        if tactile_clean_token_length > 0:
            per_batch_c = tactile_clean_token_length // B
            tc_seq_id = torch.arange(B)[:, None].expand(-1, per_batch_c).flatten()
            tc_frame_id = _tactile_frame_ids(tactile_clean_token_length)
            tc_noise_id = torch.ones(
                tactile_clean_token_length, dtype=torch.long
            )  # clean
            tc_modality_id = torch.full(
                (tactile_clean_token_length,), 2, dtype=torch.long
            )
            seq_ids = torch.cat([seq_ids, tc_seq_id])
            frame_ids = torch.cat([frame_ids, tc_frame_id])
            noise_ids = torch.cat([noise_ids, tc_noise_id])
            modality_ids = torch.cat([modality_ids, tc_modality_id])

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)
        modality_ids = F.pad(modality_ids, (0, padded_length), value=-1)

        attention_backend = _resolve_flex_attention_backend()
        seq_ids_device = seq_ids.long().to(device)
        frame_ids_device = frame_ids.long().to(device)
        noise_ids_device = noise_ids.long().to(device)
        modality_ids_device = modality_ids.long().to(device)
        if attention_backend in ("grouped_sdpa", "grouped_flash_attn"):
            query_limit = _grouped_sdpa_query_token_limit()
            if len(seq_ids) > query_limit:
                raise ValueError(
                    "Grouped SDPA query length exceeds the fail-closed safety limit: "
                    f"query_length={len(seq_ids)}, limit={query_limit}"
                )
            block_mask = FlexAttnFunc._create_grouped_self_mask(
                seq_ids_device,
                frame_ids_device,
                noise_ids_device,
                window_size,
            )
        else:
            mask_mod = FlexAttnFunc._get_mask_mod(
                seq_ids_device,
                frame_ids_device,
                noise_ids_device,
                modality_ids_device,
                window_size,
            )
            block_mask = FlexAttnFunc._create_block_mask(
                mask_mod,
                len(seq_ids),
                len(seq_ids),
                device,
            )

        # Cross-attention: text only (tactile moved to self-attention)
        text_seq_ids = torch.arange(B)[:, None].expand(-1, text_token_length).flatten()
        text_type_ids = torch.zeros_like(text_seq_ids)

        # query_modality_ids for cross-attn (same as modality_ids but padded)
        text_seq_ids_device = text_seq_ids.long().to(device)
        if attention_backend in ("grouped_sdpa", "grouped_flash_attn"):
            block_mask_cross = FlexAttnFunc._create_grouped_cross_mask(
                seq_ids_device,
                modality_ids_device,
                text_seq_ids_device,
            )
        else:
            mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(
                seq_ids_device,
                modality_ids_device,
                text_seq_ids_device,
                text_type_ids.long().to(device),
            )
            block_mask_cross = FlexAttnFunc._create_block_mask(
                mask_mod_cross,
                len(seq_ids),
                len(text_seq_ids),
                device,
            )
        return block_mask, block_mask_cross

    @staticmethod
    @torch.no_grad()
    def _create_grouped_self_mask(
        seq_ids: torch.Tensor,
        frame_ids: torch.Tensor,
        noise_ids: torch.Tensor,
        window_size: int,
    ) -> GroupedAttentionMask:
        if not (seq_ids.shape == frame_ids.shape == noise_ids.shape):
            raise ValueError("Self-attention identifier tensors must have equal shape")
        if seq_ids.ndim != 1:
            raise ValueError(
                "Self-attention identifier tensors must be one-dimensional"
            )
        descriptor = torch.stack((seq_ids, frame_ids, noise_ids), dim=1)
        valid_descriptor = (seq_ids >= 0) & ((noise_ids == 0) | (noise_ids == 1))
        unique_descriptors = torch.unique(
            descriptor[valid_descriptor],
            dim=0,
            sorted=True,
        )
        groups = []
        for query_seq_id, query_frame_id, query_noise_id in unique_descriptors.unbind(
            dim=0
        ):
            query_match = (
                (seq_ids == query_seq_id)
                & (frame_ids == query_frame_id)
                & (noise_ids == query_noise_id)
            )
            same_valid_sequence = (seq_ids == query_seq_id) & (seq_ids >= 0)
            within_window = (query_frame_id - frame_ids).abs() <= window_size
            clean_to_clean = (
                (query_noise_id == 1) & (noise_ids == 1) & (frame_ids <= query_frame_id)
            )
            noise_to_clean = (
                (query_noise_id == 0) & (noise_ids == 1) & (frame_ids < query_frame_id)
            )
            noise_to_noise = (
                (query_noise_id == 0) & (noise_ids == 0) & (frame_ids == query_frame_id)
            )
            key_value_match = (
                (clean_to_clean | noise_to_clean | noise_to_noise)
                & same_valid_sequence
                & within_window
            )
            query_indices = torch.nonzero(query_match, as_tuple=False).flatten().long()
            key_value_indices = (
                torch.nonzero(key_value_match, as_tuple=False).flatten().long()
            )
            if query_indices.numel() > 0 and key_value_indices.numel() > 0:
                groups.append((query_indices, key_value_indices))
        sequence_length = seq_ids.numel()
        return GroupedAttentionMask(
            shape=(1, 1, sequence_length, sequence_length),
            groups=tuple(groups),
        )

    @staticmethod
    @torch.no_grad()
    def _create_grouped_cross_mask(
        query_seq_ids: torch.Tensor,
        query_modality_ids: torch.Tensor,
        cond_seq_ids: torch.Tensor,
    ) -> GroupedAttentionMask:
        if query_seq_ids.shape != query_modality_ids.shape:
            raise ValueError("Cross-attention query identifier shapes must match")
        if query_seq_ids.ndim != 1 or cond_seq_ids.ndim != 1:
            raise ValueError(
                "Cross-attention identifier tensors must be one-dimensional"
            )
        valid_query = (query_seq_ids >= 0) & (query_modality_ids != 2)
        unique_sequences = torch.unique(query_seq_ids[valid_query], sorted=True)
        groups = []
        for sequence_id in unique_sequences.unbind(dim=0):
            query_match = (
                (query_seq_ids == sequence_id)
                & (query_seq_ids >= 0)
                & (query_modality_ids != 2)
            )
            key_value_match = (cond_seq_ids == sequence_id) & (cond_seq_ids >= 0)
            query_indices = torch.nonzero(query_match, as_tuple=False).flatten().long()
            key_value_indices = (
                torch.nonzero(key_value_match, as_tuple=False).flatten().long()
            )
            if query_indices.numel() > 0 and key_value_indices.numel() > 0:
                groups.append((query_indices, key_value_indices))
        return GroupedAttentionMask(
            shape=(1, 1, query_seq_ids.numel(), cond_seq_ids.numel()),
            groups=tuple(groups),
        )

    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(
        query_seq_ids, query_modality_ids, cond_seq_ids, cond_type_ids
    ):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            query_seq_id = _vmap_safe_index_1d(query_seq_ids, q_idx)
            cond_seq_id = _vmap_safe_index_1d(cond_seq_ids, kv_idx)
            same_seq = (
                (query_seq_id == cond_seq_id) & (query_seq_id >= 0) & (cond_seq_id >= 0)
            )
            # Tactile tokens (modality=2) don't need cross-attention to text
            q_not_tactile = _vmap_safe_index_1d(query_modality_ids, q_idx) != 2
            return same_seq & q_not_tactile

        return seq_mask

    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, modality_ids, window_size):
        window_size_tensor = torch.as_tensor(
            window_size,
            dtype=frame_ids.dtype,
            device=frame_ids.device,
        )

        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            query_seq_id = _vmap_safe_index_1d(seq_ids, q_idx)
            key_value_seq_id = _vmap_safe_index_1d(seq_ids, kv_idx)
            return (
                (query_seq_id == key_value_seq_id)
                & (query_seq_id >= 0)
                & (key_value_seq_id >= 0)
            )

        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            query_frame_id = _vmap_safe_index_1d(frame_ids, q_idx)
            key_value_frame_id = _vmap_safe_index_1d(frame_ids, kv_idx)
            return key_value_frame_id <= query_frame_id

        def block_causal_mask_exclude_self(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            query_frame_id = _vmap_safe_index_1d(frame_ids, q_idx)
            key_value_frame_id = _vmap_safe_index_1d(frame_ids, kv_idx)
            return key_value_frame_id < query_frame_id

        def block_self_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            query_frame_id = _vmap_safe_index_1d(frame_ids, q_idx)
            key_value_frame_id = _vmap_safe_index_1d(frame_ids, kv_idx)
            return key_value_frame_id == query_frame_id

        def clean2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            query_noise_id = _vmap_safe_index_1d(noise_ids, q_idx)
            key_value_noise_id = _vmap_safe_index_1d(noise_ids, kv_idx)
            return (query_noise_id == 1) & (key_value_noise_id == 1)

        def noise2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            query_noise_id = _vmap_safe_index_1d(noise_ids, q_idx)
            key_value_noise_id = _vmap_safe_index_1d(noise_ids, kv_idx)
            return (query_noise_id == 0) & (key_value_noise_id == 1)

        def noise2noise_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            query_noise_id = _vmap_safe_index_1d(noise_ids, q_idx)
            key_value_noise_id = _vmap_safe_index_1d(noise_ids, kv_idx)
            return (query_noise_id == 0) & (key_value_noise_id == 0)

        def block_window_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
            window_size: torch.Tensor,
        ):
            query_frame_id = _vmap_safe_index_1d(frame_ids, q_idx)
            key_value_frame_id = _vmap_safe_index_1d(frame_ids, kv_idx)
            return (query_frame_id - key_value_frame_id).abs() <= window_size

        # Option 3: tactile is now a NORMAL frame-aligned modality (its frame_id is
        # set to the video frame it co-generates with). So it flows through exactly
        # the same causal + window rules as video/action — no tactile-specific
        # bypass. Consequences:
        #   • tactile_noisy(2k) ↔ video_noisy(2k): co-generate at the same frame
        #     via noise2noise ∧ block_self (mutual V↔T conditioning).
        #   • tactile_noisy(2k) → clean(<2k) only (noise2clean ∧ exclude_self): it
        #     conditions on PAST clean video/tactile/action but NOT the same-frame
        #     clean — so causality itself prevents the leak the old
        #     no_tactile_query_side_channel rule used to guard against.
        #   • action_noisy(2k+1) → tactile_clean(2k): step-2 action conditions on
        #     the (clean) tactile produced in step 1.
        #   • everything is clipped to the sliding window like video/action, so the
        #     streaming KV-cache inference stays consistent with training.
        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask = or_masks(*mask_list)
        mask = and_masks(mask, seq_mask)
        mask = and_masks(
            mask,
            partial(block_window_mask, window_size=window_size_tensor),
        )
        return mask


class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.f_dim, 2)[: (self.f_dim // 2)].double()
                / self.f_dim
            )
        )
        h_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.h_dim, 2)[: (self.h_dim // 2)].double()
                / self.h_dim
            )
        )
        w_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.w_dim, 2)[: (self.w_dim // 2)].double()
                / self.w_dim
            )
        )
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(
                grid_ids.device
            )
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(
                grid_ids.device
            )
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(
                grid_ids.device
            )
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class WanAttention(torch.nn.Module):

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode="torch",
    ):
        super().__init__()
        self.attn_mode = attn_mode
        if attn_mode == "torch":
            self.attn_op = custom_sdpa
        elif attn_mode == "flashattn":
            self.attn_op = flash_attn_func
        elif attn_mode == "flex":
            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support torch, flashattn and flex"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim
            if cross_attention_dim_head is None
            else cross_attention_dim_head * heads
        )

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )
        self.norm_k = torch.nn.RMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )
        self.attn_caches = {} if cross_attention_dim_head is None else None

    def set_flex_attention_mask(
        self,
        block_mask: BlockMask | GroupedAttentionMask | None,
    ) -> None:
        if isinstance(self.attn_op, FlexAttnFunc):
            self.attn_op.set_block_mask(block_mask)

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        is_pred = cache["is_pred"]
        cache["mask"][is_pred] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(
        self, cache_name, total_tolen, num_head, head_dim, device, dtype, batch_size
    ):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            "k": torch.empty(
                [batch_size, total_tolen, num_head, head_dim],
                device=device,
                dtype=dtype,
            ),
            "v": torch.empty(
                [batch_size, total_tolen, num_head, head_dim],
                device=device,
                dtype=dtype,
            ),
            "id": torch.full((total_tolen,), -1, device=device),
            "mask": torch.zeros((total_tolen,), dtype=torch.bool, device=device),
            "is_pred": torch.zeros((total_tolen,), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]

            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]["id"]
        mask = self.attn_caches[cache_name]["mask"]

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache["k"][:, slots] = key
        cache["v"][:, slots] = value
        cache["mask"][slots] = True
        cache["id"][slots] = new_id
        cache["is_pred"][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]["mask"][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name="pos",
        attn_mask=None,
    ):
        kv_cache = (
            self.attn_caches[cache_name]
            if (self.attn_caches is not None) and (cache_name in self.attn_caches)
            else None
        )

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                if x.shape[1] == 0:  # empty modality (0 tokens) -> RoPE is a no-op
                    return x
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(
                        x.shape[0], x.shape[1], x.shape[2], -1, 2
                    )
                )
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        slots = None
        if kv_cache is not None and kv_cache["k"] is not None:
            slots = self.update_cache(
                cache_name, key, value, is_pred=(update_cache == 1)
            )
            key_pool = self.attn_caches[cache_name]["k"]
            value_pool = self.attn_caches[cache_name]["v"]
            mask = self.attn_caches[cache_name]["mask"]
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        if attn_mask is not None:
            # Vendor FlashAttention has no arbitrary dense-mask argument. MoT
            # training uses per-device batch size one (no text-isolation mask),
            # while larger packed batches take this exact SDPA fallback.
            attention_operation = (
                custom_sdpa if self.attn_mode == "flashattn" else self.attn_op
            )
            hidden_states = attention_operation(
                query,
                key,
                value,
                attn_mask=attn_mask,
            )
        else:
            hidden_states = self.attn_op(query, key, value)

        if update_cache == 0:
            if kv_cache is not None and kv_cache["k"] is not None:
                self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    # ───────────────────── MoT split (Mixture-of-Transformers) ─────────────────────
    # `forward` runs the whole self-attention internally. For MoT we need to compute
    # each expert's q/k/v, CONCATENATE them across experts, run ONE shared attention,
    # then split back. So we expose the projection half (`project_qkv`) and the
    # output half (`merge_out`) separately. Composing
    #     merge_out(attn_op(*project_qkv(q,k,v,rope)))
    # is byte-for-byte identical to `forward(q,k,v,rope)` on the train path (no
    # KV-cache, no attn_mask).
    def project_qkv(self, q, k, v, rotary_emb):
        """First half of `forward`: q/k/v proj + RMSNorm(q,k) + head-reshape + RoPE.
        Returns query/key/value each shaped (B, S, heads, dim_head)."""
        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query).unflatten(2, (self.heads, -1))
        key = self.norm_k(key).unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                if x.shape[1] == 0:  # empty modality (0 tokens) -> RoPE is a no-op
                    return x
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(
                        x.shape[0], x.shape[1], x.shape[2], -1, 2
                    )
                )
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        return query, key, value

    def merge_out(self, attn_output, ref=None):
        """Second half of `forward`: flatten heads + output projection.
        `attn_output` is THIS expert's slice of the shared attention, shaped
        (B, S, heads, dim_head). `ref` (optional) gives the dtype to cast to,
        mirroring `forward`'s `type_as(query)`."""
        hidden_states = attn_output.flatten(2, 3)
        if ref is not None:
            hidden_states = hidden_states.type_as(ref)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
        attn_head_dim=None,
    ):
        super().__init__()
        self.attn_mode = attn_mode
        # MoT slim experts decouple the residual width (`dim`) from the attention
        # width (`num_heads * attn_head_dim`): q/k/v project dim -> attn_inner,
        # the output proj maps attn_inner -> dim, and FFN/norms run at `dim`. This
        # lets an action/tactile expert keep a narrow residual (e.g. 1024) while its
        # q/k/v stay 3072-wide so they can be concatenated with the video expert in
        # the shared attention. Default = dim//num_heads (legacy: attn width == dim).
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=head_dim,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=head_dim,
            eps=eps,
            cross_attention_dim_head=head_dim,
            attn_mode=attn_mode,
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def set_flex_attention_masks(
        self,
        self_attention_mask: BlockMask | GroupedAttentionMask | None,
        cross_attention_mask: BlockMask | GroupedAttentionMask | None,
    ) -> None:
        self.attn1.set_flex_attention_mask(self_attention_mask)
        self.attn2.set_flex_attention_mask(cross_attention_mask)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        temb=None,
        rotary_emb=None,
        update_cache=0,
        cache_name="pos",
        mot_mode=None,
        mot_kwargs=None,
    ) -> torch.Tensor:
        # MoT dispatch: the orchestrator runs each expert's pre/post halves THROUGH
        # this forward (not by calling pre_attn/post_attn directly) so that FSDP2's
        # all-gather hook and activation-checkpoint wrappers — which hook __call__/
        # forward, not arbitrary methods — fire correctly per block.
        if mot_mode == "pre":
            return self.pre_attn(hidden_states, temb, rotary_emb)
        if mot_mode == "post":
            return self.post_attn(hidden_states, **(mot_kwargs or {}))
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(6, dim=1)
        )
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            rotary_emb,
            update_cache=update_cache,
            cache_name=cache_name,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            None,
            update_cache=0,
            cache_name=cache_name,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)
        return hidden_states

    # ───────────────────── MoT split (Mixture-of-Transformers) ─────────────────────
    # `forward` is split so a MoT orchestrator can interleave a single cross-expert
    # shared attention between the two halves:
    #   q,k,v, residual, mods = block.pre_attn(h, temb, rope)   # per expert
    #   attn = shared_attention(cat over experts)               # ONE call, joint mask
    #   h = block.post_attn(residual, attn_slice, *mods, enc, do_cross_attn)
    # Composing post_attn(pre_attn(...)) with the block's own attn_op reproduces
    # `forward` exactly on the train path.
    def pre_attn(self, hidden_states, temb, rotary_emb):
        """First half of `forward`: AdaLN modulation + norm1 + attn1 q/k/v.
        Returns (query, key, value [B,S,heads,dh]), the residual input, and the
        modulation tensors (gate_msa + the 3 FFN-branch mods) for `post_attn`."""
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(6, dim=1)
        )
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        query, key, value = self.attn1.project_qkv(
            norm_hidden_states, norm_hidden_states, norm_hidden_states, rotary_emb
        )
        return (
            query,
            key,
            value,
            hidden_states,
            gate_msa,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
        )

    def post_attn(
        self,
        hidden_states,
        attn_output,
        gate_msa,
        c_shift_msa,
        c_scale_msa,
        c_gate_msa,
        encoder_hidden_states,
        do_cross_attn=True,
        cross_attn_mask=None,
    ):
        """Second half of `forward`: merge shared-attn output + residual gate,
        optional text cross-attention, then FFN. `attn_output` is THIS expert's
        slice of the shared attention, shaped (B, S, heads, dh).
        `cross_attn_mask` is an optional dense bool mask (B|1, 1, S_q, S_text;
        True=attend) for the text cross-attention — used by the MoT orchestrator
        for per-expert batch isolation (the experts run attn2 in 'torch'/SDPA mode
        since the heavy self-attention is done externally by the shared attention).
        `do_cross_attn=False` skips text cross-attn (tactile expert)."""
        attn_output = self.attn1.merge_out(attn_output, ref=hidden_states)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )
        # 2. Cross-attention (text)
        if do_cross_attn:
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(
                hidden_states
            )
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states,
                encoder_hidden_states,
                None,
                update_cache=0,
                attn_mask=cross_attn_mask,
            )
            hidden_states = hidden_states + attn_output
        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)
        return hidden_states


class ContactStateGate(nn.Module):
    r"""Predictive-contact gate (symdiff add-on).

    Reads the model's (predicted) GlobalTactile hidden state, distils a per-sample
    "contact state" with a small Transformer, and FiLM-modulates the action stream.
    Aligns with the design flow "predict visual+tactile -> predict action": the
    action is gated by the model's own *anticipated* touch (coarse in/out-of-contact
    state), complementing the existing LocalTactile cross-attn (fine spatial contact).

    Safety: the FiLM output projection is ZERO-initialised, so at start gamma=0,
    beta=0  =>  action_hidden * (1+0) + 0 = action_hidden (exact identity / no-op).
    The pretrained model is therefore byte-for-byte unchanged until this module is
    trained. `stop_grad` detaches the tactile input so the action loss never
    perturbs GlobalTactile prediction quality.
    """

    def __init__(self, inner_dim, n_layers=2, n_heads=8, ffn_mult=1, stop_grad=True):
        super().__init__()
        self.stop_grad = stop_grad
        if inner_dim % n_heads != 0:
            n_heads = 8 if inner_dim % 8 == 0 else 4
        layer = nn.TransformerEncoderLayer(
            d_model=inner_dim,
            nhead=n_heads,
            dim_feedforward=int(inner_dim * ffn_mult),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.state_norm = nn.LayerNorm(inner_dim)
        self.to_film = nn.Linear(inner_dim, 2 * inner_dim)
        self.reset_film()

    def reset_film(self):
        # zero -> identity gate (gamma=0 => scale 1, beta=0)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)

    def forward(self, action_hidden, tactile_hidden, batch_size):
        # CFG-drop / no GlobalTactile this step -> identity (also keeps the module
        # gradient-free on dropped steps, matching the existing tactile path).
        if tactile_hidden is None or tactile_hidden.shape[1] == 0:
            return action_hidden
        t = tactile_hidden.detach() if self.stop_grad else tactile_hidden
        t = t.to(self.to_film.weight.dtype)  # robust to bf16/fp32 mix
        if t.shape[0] == 1 and batch_size > 1:  # packed "1 (b k) c"
            t = rearrange(t, "1 (b k) c -> b k c", b=batch_size)
        s = self.encoder(t)  # (B, K, C)
        s = self.state_norm(s.mean(dim=1))  # (B, C)
        gamma, beta = self.to_film(s).chunk(2, dim=-1)  # (B, C), (B, C)
        gamma = gamma.to(action_hidden.dtype)
        beta = beta.to(action_hidden.dtype)
        packed = action_hidden.shape[0] == 1 and batch_size > 1
        a = (
            rearrange(action_hidden, "1 (b l) c -> b l c", b=batch_size)
            if packed
            else action_hidden
        )
        a = a * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        if packed:
            a = rearrange(a, "b l c -> 1 (b l) c")
        return a


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    Diffusion backbone: a single shared transformer stack over the joint
    video / tactile / action token sequence, trained under a rectified-flow /
    flow-matching objective. The Mixture-of-Transformers variant that splits this
    into per-modality experts is ``WanMoTTransformer3DModel`` (see mot.py).
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        # "patch_embedding",
        "patch_embedding_mlp",
        "condition_embedder",
        "condition_embedder_action",
        "norm",
    ]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "scale_shift_table_action",
        "norm1",
        "action_norm1",
        "text_norm1",
        "norm2",
        "action_norm2",
        "text_norm2",
        "norm3",
        "action_norm3",
        "text_norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
        tactile_in_channels=3,
        tactile_num_tokens=4,
        tactile_encoder_dim=256,
        max_tactile_streams=4,
        use_local_tactile=True,
        instantiate_local_tactile=None,
        use_contact_gate=False,
        contact_gate_layers=2,
        contact_gate_heads=8,
        contact_gate_stop_grad=True,
        use_wrench_conditioner=False,
        instantiate_wrench_conditioner=None,
        wrench_arm_count=2,
        wrench_max_frames=64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.tactile_num_tokens = tactile_num_tokens
        self.max_tactile_streams = max_tactile_streams
        # use_local_tactile: build + use the LocalTactile cross-attn branch into the
        # action head. Default True (existing ckpts have it). Set False for pretrain
        # (GlobalTactile alone) — then post-train flips it on and the branch is
        # zero-init so warm-starting a no-local pretrain ckpt is identity at first.
        self.use_local_tactile = bool(use_local_tactile)
        self.instantiate_local_tactile = (
            self.use_local_tactile
            if instantiate_local_tactile is None
            else bool(instantiate_local_tactile)
        )
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2], inner_dim
        )
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        # ─────────────── Tactile (VAE-encoded latent) modules ───────────────
        # GlobalTactile latent: (B, S_sensors, tactile_latent_channels=48, F_lat,
        #                        tactile_latent_height, tactile_latent_width)
        # — preprocessed by script/encode_tactile_latent.py
        #   (RGB residual vs first frame) → 128x128 → Wan VAE → 48ch × 8×8 latent
        # Patchified with the same patch_size=(1,2,2) as video, sharing patch
        # embedding dimension but with its own Linear so it can specialize.
        tactile_latent_channels = 48
        self.tactile_latent_channels = tactile_latent_channels
        # RoPE width-coordinate offset for the whole GlobalTactile block. Video
        # views are tiled along W into [0, N_views*Wp); tactile tokens are a
        # separate stream whose grid would otherwise start at w=0 and overlap the
        # video band. Pushing tactile to w∈[offset, offset+Wp) makes it disjoint
        # from video/action in RoPE space (like an extra tile far to the right),
        # WITHOUT touching tactile-internal relative positions (a constant offset
        # cancels in relative position, so each sensor keeps its intra-pad spatial
        # structure). Sensors stay distinguished by the additive sensor_id_embed.
        # Must be a FIXED constant (not derived from the video width) so the
        # tactile grid is identical between joint training and per-stage inference.
        # 10000 clears any realistic video width (≤ a few hundred patches) and is
        # within the lowest RoPE frequency's wavelength (~2*pi*theta), so it reads
        # as an unambiguous "far" position rather than aliasing back onto video.
        self.tactile_pos_offset = 10000
        self.tactile_patch_embed = nn.Linear(
            tactile_latent_channels * math.prod(patch_size),
            inner_dim,
        )
        # sensor_id embedding (token added per-sensor, common to all frames+patches
        # within that sensor's slice). max_tactile_streams covers up to 4 sensors
        # (e.g. 2 hands × 2 fingers).
        self.sensor_id_embed = nn.Embedding(max_tactile_streams, inner_dim)
        self.tactile_norm = FP32LayerNorm(inner_dim, eps, elementwise_affine=True)

        # ─────── LocalTactile cross-attention (action head condition) ───────
        # LocalTactile cross-attn branch into the action head (NOT in the self-attn
        # sequence). Only built when use_local_tactile=True. Pretrain sets it False
        # (no local), post-train flips it True (warm-start: branch zero-init at the
        # output proj -> identity, so a no-local pretrain ckpt is unaffected at step 0).
        if self.instantiate_local_tactile:
            self.local_tactile_patch_embed = nn.Linear(
                tactile_latent_channels * math.prod(patch_size),
                inner_dim,
            )
            self.local_tactile_sensor_embed = nn.Embedding(
                max_tactile_streams, inner_dim
            )
            self.local_tactile_frame_embed = nn.Embedding(rope_max_seq_len, inner_dim)
            self.local_tactile_h_embed = nn.Embedding(rope_max_seq_len, inner_dim)
            self.local_tactile_w_embed = nn.Embedding(rope_max_seq_len, inner_dim)
            self.local_tactile_norm = FP32LayerNorm(
                inner_dim, eps, elementwise_affine=True
            )
            self.local_tactile_pre_norm = FP32LayerNorm(
                inner_dim, eps, elementwise_affine=False
            )
            self.local_tactile_cross_attn = WanAttention(
                dim=inner_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                eps=eps,
                cross_attention_dim_head=attention_head_dim,  # cross-attn mode (no KV cache)
                attn_mode="torch",  # use SDPA for inference compatibility
            )
            # ZERO-INIT the cross-attn output projection -> attn_out == 0 at step 0, so
            # `action_hidden + attn_out == action_hidden` (identity). Critical for the
            # post-train warm-start: a freshly-built LocalTactile branch added on top of
            # a no-local pretrain ckpt must NOT inject random noise into the trained
            # action stream — it ramps in gradually as to_out learns. Same trick as the
            # contact gate's zero-init FiLM (ContactStateGate) and ControlNet/adapters.
            nn.init.zeros_(self.local_tactile_cross_attn.to_out[0].weight)
            nn.init.zeros_(self.local_tactile_cross_attn.to_out[0].bias)

        # ─────── Predictive-contact gate (opt-in predictive-contact-gate add-on) ───────
        # Default OFF: existing checkpoints' config.json has no use_contact_gate
        # key -> defaults to False here -> module not instantiated -> the model is
        # byte-for-byte identical to before. When ON, the FiLM is zero-init
        # (identity) so it is still a no-op until trained. See ContactStateGate.
        self.use_contact_gate = bool(use_contact_gate)
        if self.use_contact_gate:
            self.contact_gate = ContactStateGate(
                inner_dim,
                n_layers=int(contact_gate_layers),
                n_heads=int(contact_gate_heads),
                stop_grad=bool(contact_gate_stop_grad),
            )

        # AgileX-only wrist-wrench branch. Default OFF keeps every legacy
        # Franka/UniVTAC/native checkpoint state-dict key unchanged.
        self.use_wrench_conditioner = bool(use_wrench_conditioner)
        self.instantiate_wrench_conditioner = (
            self.use_wrench_conditioner
            if instantiate_wrench_conditioner is None
            else bool(instantiate_wrench_conditioner)
        )
        if self.instantiate_wrench_conditioner:
            self.agilex_wrench_conditioner = WrenchConditioner(
                hidden_dim=inner_dim,
                max_frames=int(wrench_max_frames),
                arm_count=int(wrench_arm_count),
            )
            self.agilex_wrench_pre_norm = FP32LayerNorm(
                inner_dim, eps, elementwise_affine=False
            )
            self.agilex_wrench_cross_attn = WanAttention(
                dim=inner_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                eps=eps,
                cross_attention_dim_head=attention_head_dim,
                attn_mode="torch",
            )
            nn.init.zeros_(self.agilex_wrench_cross_attn.to_out[0].weight)
            nn.init.zeros_(self.agilex_wrench_cross_attn.to_out[0].bias)

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    cross_attn_norm,
                    eps,
                    attn_mode=attn_mode,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        # Symdiff tactile: tactile diffusion head — predicts GlobalTactile
        # velocity (= ε - x_0) at the per-token granularity, mirroring proj_out.
        self.tactile_proj_out = nn.Linear(
            inner_dim,
            tactile_latent_channels * math.prod(patch_size),
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

    def clear_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(
        self,
        cache_name,
        attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        device,
        dtype,
        batch_size,
    ):
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2
        ) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(
                cache_name,
                total_tolen,
                self.num_attention_heads,
                self.attention_head_dim,
                device,
                dtype,
                batch_size,
            )

    def _input_embed(self, latents, input_type="latent"):
        if input_type == "latent":
            hidden_states = rearrange(
                latents,
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == "action":
            hidden_states = rearrange(latents, "b c f h w -> b (f h w) c")
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == "text":
            hidden_states = self.condition_embedder.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _encode_text_condition(self, text_emb):
        return self._input_embed(text_emb, input_type="text")

    def _encode_tactile_condition(self, tactile_latent, sensor_ids=None):
        """Encode GlobalTactile VAE latent into tokens for self-attention.

        Input:
          tactile_latent: (B, S_sensors, C=48, F_lat, H_lat, W_lat) — VAE-encoded
                          residual latent from script/encode_tactile_latent.py
          sensor_ids:     (B, S_sensors) or (S_sensors,) — sensor_id index for each
                          sensor slot.

        Output:
          tokens: (B, S * F_lat * (H_lat/p_h) * (W_lat/p_w), inner_dim)
        """
        if tactile_latent is None:
            raise ValueError(
                "tactile_global_latent is required in tactile cond branch."
            )
        if tactile_latent.dim() == 5:  # (S, C, F, H, W) → add batch
            tactile_latent = tactile_latent.unsqueeze(0)

        B, S, C, F_lat, H_lat, W_lat = tactile_latent.shape
        if S > self.sensor_id_embed.num_embeddings:
            raise ValueError(
                f"Got {S} tactile sensors but model supports max "
                f"{self.sensor_id_embed.num_embeddings}"
            )

        # Patchify each sensor's latent the same way as video latent
        # (B, S, C, F, H, W) → (B*S, C, F, H, W) → patchify → (B*S, F*hp*wp, C*pf*ph*pw)
        p_f, p_h, p_w = self.patch_size
        x = rearrange(tactile_latent, "b s c f h w -> (b s) c f h w")
        x = rearrange(
            x,
            "n c (f pf) (h ph) (w pw) -> n (f h w) (c pf ph pw)",
            pf=p_f,
            ph=p_h,
            pw=p_w,
        )
        # Linear patch embed → (B*S, num_patches, inner_dim)
        tokens = self.tactile_patch_embed(x)
        tokens = rearrange(tokens, "(b s) n d -> b s n d", b=B, s=S)

        # Add sensor_id embedding (broadcast across all patches of that sensor)
        if sensor_ids is None:
            raise ValueError("tactile_sensor_ids is required in tactile cond branch.")
        if sensor_ids.dim() == 1:
            sensor_ids = sensor_ids[None].expand(B, S)
        elif sensor_ids.dim() == 2 and sensor_ids.shape[0] == 1:
            sensor_ids = sensor_ids.expand(B, S)
        sensor_ids = sensor_ids.to(device=tokens.device, dtype=torch.long)
        if sensor_ids.shape != (B, S):
            raise ValueError(
                "tactile_sensor_ids must have shape "
                f"({B}, {S}), got {tuple(sensor_ids.shape)}."
            )
        sensor_emb = self.sensor_id_embed(sensor_ids)  # (B, S, inner_dim)
        tokens = tokens + sensor_emb[:, :, None, :]  # broadcast over patches

        # FP32 norm
        tokens = self.tactile_norm(tokens.float()).type_as(tactile_latent)

        # Flatten S × N into a single sequence
        return rearrange(tokens, "b s n d -> b (s n) d")

    def _encode_local_tactile(self, local_tactile_latent, sensor_ids=None):
        """Encode LocalTactile VAE latent for action-head cross-attention.

        Independent module from _encode_tactile_condition: uses its own
        patch_embed + sensor embed + norm so the cross-attn condition has a
        specialised representation (LocalTactile is high-frequency, GlobalTactile
        is low-frequency).

        Input/output shape: same as _encode_tactile_condition.
        """
        if local_tactile_latent is None:
            raise ValueError("tactile_local_latent is required in tactile cond branch.")
        if local_tactile_latent.dim() == 5:
            local_tactile_latent = local_tactile_latent.unsqueeze(0)

        B, S, C, F_lat, H_lat, W_lat = local_tactile_latent.shape
        if S > self.local_tactile_sensor_embed.num_embeddings:
            raise ValueError(
                f"Got {S} local tactile sensors but model supports max "
                f"{self.local_tactile_sensor_embed.num_embeddings}. Increase "
                "max_tactile_streams in the config."
            )
        p_f, p_h, p_w = self.patch_size
        x = rearrange(local_tactile_latent, "b s c f h w -> (b s) c f h w")
        x = rearrange(
            x,
            "n c (f pf) (h ph) (w pw) -> n (f h w) (c pf ph pw)",
            pf=p_f,
            ph=p_h,
            pw=p_w,
        )
        tokens = self.local_tactile_patch_embed(x)
        tokens = rearrange(tokens, "(b s) n d -> b s n d", b=B, s=S)

        pos_emb = self._build_local_tactile_position_embedding(
            B,
            S,
            F_lat // p_f,
            H_lat // p_h,
            W_lat // p_w,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = tokens + pos_emb

        if sensor_ids is None:
            raise ValueError("tactile_sensor_ids is required in tactile cond branch.")
        if sensor_ids.dim() == 1:
            sensor_ids = sensor_ids[None].expand(B, S)
        elif sensor_ids.dim() == 2 and sensor_ids.shape[0] == 1:
            sensor_ids = sensor_ids.expand(B, S)
        sensor_ids = sensor_ids.to(device=tokens.device, dtype=torch.long)
        if sensor_ids.shape != (B, S):
            raise ValueError(
                "tactile_sensor_ids must have shape "
                f"({B}, {S}), got {tuple(sensor_ids.shape)}."
            )
        tokens = tokens + self.local_tactile_sensor_embed(sensor_ids)[:, :, None, :]

        tokens = self.local_tactile_norm(tokens.float()).type_as(local_tactile_latent)
        return rearrange(tokens, "b s n d -> b (s n) d")

    def _local_tactile_causal_mask(
        self, action_latent_shape, tactile_grid_shape, chunk_size, device
    ):
        """Chunk-causal cross-attn mask so an action token only attends LocalTactile
        of frames up to (and including) its own chunk — never future tactile.

        Uses the SAME chunk-causal convention as the self-attention stage:
        action chunk c → frame_id 2c+1, local tactile chunk c → frame_id 2c, so
        the condition `local_fid <= action_fid` means action chunk c sees local
        chunks <= c. Returns a bool mask (1, 1, L_action, S*N_local); True = attend.
        Token orders match the encoders: action = (frame, A, W), local = (sensor,
        frame, hp, wp).
        """
        _, _, F_a, A_a, W_a = action_latent_shape
        tokens_per_frame = A_a * W_a
        af = torch.arange(F_a, device=device).repeat_interleave(tokens_per_frame)
        action_fid = (af // chunk_size) * 2 + 1  # (L_action,)
        _, S_t, Fp_t, Hp_t, Wp_t = tactile_grid_shape
        lf = (
            torch.arange(Fp_t, device=device).repeat_interleave(Hp_t * Wp_t).repeat(S_t)
        )
        local_fid = (lf // chunk_size) * 2  # (S*Fp*Hp*Wp,)
        mask = local_fid[None, :] <= action_fid[:, None]  # (L_action, S*N)
        return mask[None, None]  # (1, 1, L_a, S*N)

    def _apply_local_tactile_cross_attn(
        self, action_hidden_states, local_tactile_tokens, attn_mask=None
    ):
        """Mix LocalTactile information into action_hidden_states via cross-attn.

        Called after the backbone (after _apply_output_norm + split for training,
        or after backbone for inference), right before action_proj_out.

        Two calling conventions supported:
          - Training: action_hs (1, B*L_action, inner_dim) — batch flattened
                       local_tactile_tokens (B, S*N_patches, inner_dim)
                       → Q is reshaped back to B samples before attention.
          - Inference: action_hs (B, L_action, inner_dim) — standard batch.

        attn_mask: optional (1, 1, L_action, S*N_local) bool mask (True = attend),
        broadcast over batch & heads — used to make the cross-attn chunk-causal.
        """
        if local_tactile_tokens is None:
            raise ValueError("local_tactile_tokens is required in tactile cond branch.")
        if local_tactile_tokens.dim() != 3:
            raise ValueError(
                "local_tactile_tokens must have shape (B, L, C), got "
                f"{tuple(local_tactile_tokens.shape)}."
            )

        # Training flattens batch into the token axis. Undo that before
        # cross-attention so samples cannot attend each other's tactile tokens.
        flattened_training_batch = (
            action_hidden_states.shape[0] == 1 and local_tactile_tokens.shape[0] > 1
        )
        if flattened_training_batch:
            batch_size = local_tactile_tokens.shape[0]
            if action_hidden_states.shape[1] % batch_size != 0:
                raise ValueError(
                    "Flattened action token length must be divisible by tactile "
                    f"batch size: action_tokens={action_hidden_states.shape[1]}, "
                    f"tactile_batch={batch_size}."
                )
            q = rearrange(action_hidden_states, "1 (b l) c -> b l c", b=batch_size)
            kv = local_tactile_tokens
        else:
            if action_hidden_states.shape[0] != local_tactile_tokens.shape[0]:
                raise ValueError(
                    "Action/tactile batch mismatch: "
                    f"action={action_hidden_states.shape[0]}, "
                    f"tactile={local_tactile_tokens.shape[0]}."
                )
            q = action_hidden_states
            kv = local_tactile_tokens

        q_norm = self.local_tactile_pre_norm(q.float()).type_as(q)
        attn_out = self.local_tactile_cross_attn(
            q_norm,
            kv,
            kv,
            rotary_emb=None,
            update_cache=0,
            cache_name="local_tactile_cross",
            attn_mask=attn_mask,
        ).type_as(q)
        if flattened_training_batch:
            attn_out = rearrange(attn_out, "b l c -> 1 (b l) c")
        return action_hidden_states + attn_out

    def _encode_agilex_wrench_condition(
        self,
        wrench,
        *,
        wrench_available_mask,
        temporal_valid_mask,
        contact_cond_drop,
    ):
        if not self.use_wrench_conditioner:
            raise ValueError("AgileX wrench conditioning is disabled")
        return self.agilex_wrench_conditioner(
            wrench,
            wrench_available_mask=wrench_available_mask,
            temporal_valid_mask=temporal_valid_mask,
            contact_cond_drop=contact_cond_drop,
        )

    def _apply_agilex_wrench_cross_attn(
        self,
        action_hidden_states,
        wrench_tokens,
        attn_mask=None,
    ):
        if not self.use_wrench_conditioner:
            raise ValueError("AgileX wrench conditioning is disabled")
        if wrench_tokens is None or wrench_tokens.dim() != 3:
            raise ValueError("wrench_tokens must have shape (B, L, C)")
        flattened_training_batch = (
            action_hidden_states.shape[0] == 1 and wrench_tokens.shape[0] > 1
        )
        if flattened_training_batch:
            batch_size = wrench_tokens.shape[0]
            if action_hidden_states.shape[1] % batch_size != 0:
                raise ValueError(
                    "Flattened action token length must be divisible by wrench batch"
                )
            query = rearrange(action_hidden_states, "1 (b l) c -> b l c", b=batch_size)
        else:
            if action_hidden_states.shape[0] != wrench_tokens.shape[0]:
                raise ValueError("action and wrench batch sizes must match")
            query = action_hidden_states
        query_norm = self.agilex_wrench_pre_norm(query.float()).type_as(query)
        attention_output = self.agilex_wrench_cross_attn(
            query_norm,
            wrench_tokens,
            wrench_tokens,
            rotary_emb=None,
            update_cache=0,
            cache_name="agilex_wrench_cross",
            attn_mask=attn_mask,
        ).type_as(query)
        if flattened_training_batch:
            attention_output = rearrange(attention_output, "b l c -> 1 (b l) c")
        return action_hidden_states + attention_output

    @staticmethod
    def _agilex_wrench_causal_mask(
        action_latent_shape,
        *,
        wrench_frames,
        wrench_arms,
        chunk_size,
        device,
    ):
        _, _, action_frames, action_height, action_width = action_latent_shape
        action_tokens_per_frame = action_height * action_width
        action_frame_ids = (
            torch.arange(action_frames, device=device)
            .repeat_interleave(action_tokens_per_frame)
            .div(chunk_size, rounding_mode="floor")
            .mul(2)
            .add(1)
        )
        wrench_frame_ids = (
            torch.arange(wrench_frames, device=device)
            .mul(2)
            .repeat_interleave(wrench_arms)
        )
        return (wrench_frame_ids[None, :] <= action_frame_ids[:, None])[None, None]

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        pach_scale_h, pach_scale_w = (
            (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        )
        latent_time_steps = torch.repeat_interleave(
            timesteps, (H // pach_scale_h) * (W // pach_scale_w), dim=1
        )  # L
        current_condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        temb, timestep_proj = current_condition_embedder(latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    @staticmethod
    def _require_tactile_entry(container, key, owner):
        if key not in container:
            raise KeyError(f"{owner} must contain '{key}' in tactile cond branch.")
        value = container[key]
        if value is None:
            raise ValueError(f"{owner}['{key}'] is None in tactile cond branch.")
        return value

    @staticmethod
    def _should_drop_tactile_condition(action_dict):
        value = action_dict.get(
            "contact_cond_drop",
            action_dict.get("tactile_cond_drop", False),
        )
        if torch.is_tensor(value):
            value = value.detach().bool().flatten()
            return bool(value.numel() > 0 and value.any().item())
        return bool(value)

    def _zero_tactile_parameter_anchor(self):
        # FSDP requires root-level params in one reduce group to have a
        # consistent grad dtype. During tactile CFG drop, tactile tokens are not
        # inserted and tactile modules should receive zero effective gradient;
        # this anchor keeps those params present in autograd without updates.
        modules = [
            self.tactile_patch_embed,
            self.sensor_id_embed,
            self.tactile_norm,
            # symdiff tactile: tactile diffusion output head — anchored here so
            # tactile_proj_out also participates in FSDP grad sync under CFG drop.
            self.tactile_proj_out,
        ]
        # LocalTactile modules only exist when use_local_tactile=True; anchor them
        # for FSDP grad-dtype consistency only when present (pretrain has none).
        if getattr(self, "use_local_tactile", True):
            modules += [
                self.local_tactile_patch_embed,
                self.local_tactile_sensor_embed,
                self.local_tactile_frame_embed,
                self.local_tactile_h_embed,
                self.local_tactile_w_embed,
                self.local_tactile_norm,
                self.local_tactile_pre_norm,
                self.local_tactile_cross_attn,
            ]
        # predictive-contact gate: on CFG-drop steps the gate is a no-op (no
        # tactile to read), so its params would otherwise get no gradient and trip
        # FSDP's "must use every param each step" / grad-dtype-consistency check.
        # Anchor them here too (zero effective gradient) like the tactile modules.
        if getattr(self, "use_contact_gate", False):
            modules.append(self.contact_gate)
        anchor = None
        for module in modules:
            for param in module.parameters(recurse=True):
                term = param.float().sum() * 0.0
                anchor = term if anchor is None else anchor + term
        if anchor is None:
            return self.scale_shift_table.sum() * 0.0
        return anchor

    def _zero_agilex_wrench_parameter_anchor(self):
        if not self.use_wrench_conditioner:
            return self.scale_shift_table.sum() * 0.0
        anchor = None
        for module in (
            self.agilex_wrench_conditioner,
            self.agilex_wrench_pre_norm,
            self.agilex_wrench_cross_attn,
        ):
            for parameter in module.parameters(recurse=True):
                term = parameter.float().sum() * 0.0
                anchor = term if anchor is None else anchor + term
        return anchor

    def _tactile_patch_grid_shape(self, tactile_latent):
        if tactile_latent.dim() == 5:
            B = 1
            S, _, F_lat, H_lat, W_lat = tactile_latent.shape
        elif tactile_latent.dim() == 6:
            B, S, _, F_lat, H_lat, W_lat = tactile_latent.shape
        else:
            raise ValueError(
                "tactile_global_latent must have shape (B, S, C, F, H, W) "
                f"or (S, C, F, H, W), got {tuple(tactile_latent.shape)}."
            )

        p_f, p_h, p_w = self.patch_size
        if F_lat % p_f != 0 or H_lat % p_h != 0 or W_lat % p_w != 0:
            raise ValueError(
                "tactile_global_latent spatial/temporal shape must be divisible "
                f"by patch_size={self.patch_size}, got "
                f"(F,H,W)=({F_lat},{H_lat},{W_lat})."
            )
        return B, S, F_lat // p_f, H_lat // p_h, W_lat // p_w

    def _build_tactile_grid_id(
        self,
        batch_size,
        num_sensors,
        frames,
        height,
        width,
        device,
        dtype,
        frame_start=0,
    ):
        if min(batch_size, num_sensors, frames, height, width) <= 0:
            raise ValueError(
                "tactile grid dimensions must be positive, got "
                f"B={batch_size}, S={num_sensors}, F={frames}, "
                f"H={height}, W={width}."
            )

        # frame_start = the video's f_shift for this chunk: tactile RoPE frames
        # must advance with the video's, or the same-frame V/T coupling drifts
        # in streaming inference (training is a single segment, both 0).
        base_grid = get_mesh_id(
            frames,
            height,
            width,
            t=2,
            f_w=1,
            f_shift=int(frame_start),
            action=False,
        ).to(device=device, dtype=dtype)

        # Shift the tactile block past the video's width band so tactile never
        # shares a RoPE position with video/action; sensors are told apart by
        # the additive sensor_id embedding, not by position.
        base_grid[2] = base_grid[2] + self.tactile_pos_offset

        # Token order must match _encode_tactile_condition:
        # sensor-major, then frame-major patch order inside each sensor.
        grid_id = base_grid.repeat(1, num_sensors)
        return grid_id[None].expand(batch_size, -1, -1).contiguous()

    def _build_local_tactile_position_embedding(
        self, batch_size, num_sensors, frames, height, width, device, dtype
    ):
        limits = {
            "frames": (frames, self.local_tactile_frame_embed.num_embeddings),
            "height": (height, self.local_tactile_h_embed.num_embeddings),
            "width": (width, self.local_tactile_w_embed.num_embeddings),
        }
        for name, (size, limit) in limits.items():
            if size > limit:
                raise ValueError(
                    f"local tactile {name}={size} exceeds embedding capacity {limit}."
                )

        f_idx = torch.arange(frames, device=device, dtype=torch.long)
        h_idx = torch.arange(height, device=device, dtype=torch.long)
        w_idx = torch.arange(width, device=device, dtype=torch.long)
        ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
        pos = (
            self.local_tactile_frame_embed(ff.flatten())
            + self.local_tactile_h_embed(hh.flatten())
            + self.local_tactile_w_embed(ww.flatten())
        ).to(dtype=dtype)
        return (
            pos[None, None]
            .expand(
                batch_size,
                num_sensors,
                -1,
                -1,
            )
            .contiguous()
        )

    def _prepare_train_inputs(self, input_dict):
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        batch_size = latent_dict["noisy_latents"].shape[0]

        latent_hidden_states = self._input_embed(
            latent_dict["noisy_latents"], input_type="latent"
        ).flatten(0, 1)[None]
        action_hidden_states = self._input_embed(
            action_dict["noisy_latents"], input_type="action"
        ).flatten(0, 1)[None]
        text_hidden_states = self._encode_text_condition(latent_dict["text_emb"])
        encoder_hidden_states = text_hidden_states.flatten(0, 1)[None]

        condition_latent_hidden_states = self._input_embed(
            latent_dict["latent"], input_type="latent"
        ).flatten(0, 1)[None]
        condition_action_hidden_states = self._input_embed(
            action_dict["latent"], input_type="action"
        ).flatten(0, 1)[None]

        tactile_disabled = action_dict.get("tactile_mode") == "disabled"
        if tactile_disabled:
            forbidden = {
                "tactile_global_latent",
                "tactile_global_noisy_latent",
                "tactile_global_clean_latent",
                "tactile_local_latent",
                "tactile_sensor_ids",
                "wrench",
                "wrench_available_mask",
            }
            present = sorted(forbidden.intersection(action_dict))
            if present:
                raise ValueError(
                    "disabled tactile training received tactile tensors: " f"{present}"
                )
            if self.use_local_tactile or self.use_contact_gate:
                raise ValueError(
                    "disabled tactile training requires local tactile and "
                    "contact gate off"
                )
        contact_condition_dropped = self._should_drop_tactile_condition(action_dict)
        has_tactile_target = any(
            key in action_dict
            for key in (
                "tactile_global_noisy_latent",
                "tactile_global_targets",
            )
        )
        drop_tactile = tactile_disabled or (
            contact_condition_dropped and not has_tactile_target
        )
        if drop_tactile:
            # CFG tactile drop: no tactile tokens enter the sequence. The
            # tactile_zero_anchor keeps every tactile module (incl. the symdiff
            # tactile_proj_out) in the autograd graph with zero gradient so FSDP
            # sees a uniform grad dtype.
            inner_dim = latent_hidden_states.shape[-1]
            tactile_hidden_states = latent_hidden_states.new_zeros(1, 0, inner_dim)
            tactile_token_length = 0
            tactile_noisy_hidden_states = None
            tactile_noisy_token_length = 0
            tactile_clean_hidden_states = None
            tactile_clean_token_length = 0
            local_tactile_tokens = None
            # A true vision-only route freezes tactile-only parameters and does
            # not even reduce them into a zero anchor. The anchor remains needed
            # for trainable tactile modules on stochastic CFG-drop steps.
            tactile_zero_anchor = (
                None if tactile_disabled else self._zero_tactile_parameter_anchor()
            )
        else:
            tactile_zero_anchor = None
            local_tactile_tokens = (
                None  # set below iff use_local_tactile (else stays None)
            )
            # ─── Symdiff tactile: GlobalTactile noisy + clean both into seq ───
            # GlobalTactile is a diffusion target like video/action. Dataset
            # provides both:
            #   action_dict['tactile_global_noisy_latent'] : (B, S, 48, F_lat, H, W)
            #     (noisy version after train_scheduler.add_noise)
            #   action_dict['tactile_global_clean_latent'] : same shape (clean cond)
            #   action_dict['tactile_sensor_ids']          : (B, S) sensor index
            # Fallback to 'tactile_global_latent' for backward compatibility
            # (treated as clean condition with no noisy counterpart).
            tactile_noisy_latent = action_dict.get("tactile_global_noisy_latent")
            tactile_clean_latent = action_dict.get("tactile_global_clean_latent")
            if tactile_clean_latent is None:
                tactile_clean_latent = self._require_tactile_entry(
                    action_dict,
                    "tactile_global_latent",
                    "action_dict",
                )
            tactile_sensor_ids = self._require_tactile_entry(
                action_dict,
                "tactile_sensor_ids",
                "action_dict",
            )

            tactile_clean_latent = tactile_clean_latent.to(torch.bfloat16)
            if tactile_noisy_latent is not None:
                tactile_noisy_latent = tactile_noisy_latent.to(torch.bfloat16)

            tactile_grid_shape = self._tactile_patch_grid_shape(tactile_clean_latent)

            # Encode both noisy and clean tactile via the SAME tactile_patch_embed
            # (so they share representation, just different noise levels).
            tactile_clean_hidden_states = self._encode_tactile_condition(
                tactile_clean_latent,
                sensor_ids=tactile_sensor_ids,
            )
            tactile_clean_hidden_states = tactile_clean_hidden_states.flatten(0, 1)[
                None
            ]
            tactile_clean_token_length = tactile_clean_hidden_states.shape[1]

            if tactile_noisy_latent is not None:
                tactile_noisy_hidden_states = self._encode_tactile_condition(
                    tactile_noisy_latent,
                    sensor_ids=tactile_sensor_ids,
                )
                tactile_noisy_hidden_states = tactile_noisy_hidden_states.flatten(0, 1)[
                    None
                ]
                tactile_noisy_token_length = tactile_noisy_hidden_states.shape[1]
            else:
                tactile_noisy_hidden_states = None
                tactile_noisy_token_length = 0

            tactile_hidden_states = tactile_clean_hidden_states  # back-compat alias
            tactile_token_length = tactile_clean_token_length

            # ─── LocalTactile (cross-attn condition for action head) ───
            # Encoded but kept separate from self-attn sequence. Skipped entirely when
            # use_local_tactile=False (pretrain): local_tactile_tokens stays None and
            # forward_train's `if local_tactile_tokens is not None` guard no-ops it.
            if self.use_local_tactile and not contact_condition_dropped:
                local_tactile_latent = self._require_tactile_entry(
                    action_dict,
                    "tactile_local_latent",
                    "action_dict",
                ).to(torch.bfloat16)
                local_tactile_tokens = self._encode_local_tactile(
                    local_tactile_latent,
                    sensor_ids=tactile_sensor_ids,
                )
        if drop_tactile:
            tactile_grid_shape = None

        wrench_tokens = None
        wrench_zero_anchor = None
        wrench_frames = 0
        wrench_arms = 0
        if self.use_wrench_conditioner and not tactile_disabled:
            wrench = action_dict.get("wrench")
            if wrench is None:
                wrench_zero_anchor = self._zero_agilex_wrench_parameter_anchor()
            else:
                required_wrench_fields = {
                    "wrench_available_mask",
                    "temporal_valid_mask",
                    "contact_cond_drop",
                }
                missing_wrench_fields = sorted(
                    required_wrench_fields - set(action_dict)
                )
                if missing_wrench_fields:
                    raise ValueError(
                        "wrench conditioning is missing fields: "
                        + ", ".join(missing_wrench_fields)
                    )
                wrench = wrench.to(latent_hidden_states.dtype)
                wrench_tokens = self._encode_agilex_wrench_condition(
                    wrench,
                    wrench_available_mask=action_dict["wrench_available_mask"],
                    temporal_valid_mask=action_dict["temporal_valid_mask"],
                    contact_cond_drop=action_dict["contact_cond_drop"],
                )
                wrench_frames = int(wrench.shape[1])
                wrench_arms = int(wrench.shape[2])

        return dict(
            batch_size=batch_size,
            latent_hidden_states=latent_hidden_states,
            condition_latent_hidden_states=condition_latent_hidden_states,
            action_hidden_states=action_hidden_states,
            condition_action_hidden_states=condition_action_hidden_states,
            tactile_hidden_states=tactile_hidden_states,  # alias = clean
            tactile_token_length=tactile_token_length,
            tactile_noisy_hidden_states=tactile_noisy_hidden_states,  # symdiff tactile
            tactile_noisy_token_length=tactile_noisy_token_length,
            tactile_clean_hidden_states=tactile_clean_hidden_states,
            tactile_clean_token_length=tactile_clean_token_length,
            tactile_grid_shape=tactile_grid_shape,
            local_tactile_tokens=local_tactile_tokens,
            tactile_zero_anchor=tactile_zero_anchor,
            wrench_tokens=wrench_tokens,
            wrench_zero_anchor=wrench_zero_anchor,
            wrench_frames=wrench_frames,
            wrench_arms=wrench_arms,
            contact_condition_dropped=contact_condition_dropped,
            text_hidden_states=text_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

    def _build_stage_position_inputs(
        self,
        latent_dict,
        action_dict,
        dtype,
        include_action_tokens=True,
        tactile_token_length=0,
        tactile_noisy_token_length=0,
        tactile_clean_token_length=0,
        tactile_grid_shape=None,
    ):
        latent_grid_id = latent_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([latent_grid_id] * 2, dim=2)

        latent_time_steps = torch.cat(
            [
                latent_dict["timesteps"].flatten(0, 1),
                latent_dict["cond_timesteps"].flatten(0, 1),
            ]
        )[None]
        temb, timestep_proj = self._time_embed(
            latent_time_steps,
            latent_dict["noisy_latents"].shape[-2],
            latent_dict["noisy_latents"].shape[-1],
            dtype=dtype,
            action_mode=False,
        )

        if include_action_tokens:
            action_grid_id = action_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
            full_grid_id = torch.cat(
                [full_grid_id, action_grid_id, action_grid_id], dim=2
            )
            action_time_steps = torch.cat(
                [
                    action_dict["timesteps"].flatten(0, 1),
                    action_dict["cond_timesteps"].flatten(0, 1),
                ]
            )[None]
            action_temb, action_timestep_proj = self._time_embed(
                action_time_steps,
                action_dict["noisy_latents"].shape[-2],
                action_dict["noisy_latents"].shape[-1],
                dtype=dtype,
                action_mode=True,
            )
            temb = torch.cat([temb, action_temb], dim=1)
            timestep_proj = torch.cat([timestep_proj, action_timestep_proj], dim=1)

        if tactile_token_length < 0:
            raise ValueError("tactile_token_length must be non-negative.")
        if tactile_token_length > 0:
            # Symdiff tactile: give tactile tokens REAL spatial/temporal RoPE
            # positions (_build_tactile_grid_id) and REAL diffusion-timestep
            # embeddings — exactly like video/action tokens, instead of the old
            # all-zero placeholders that left the tactile patches permutation-
            # invariant and noise-level-agnostic. The noisy and clean halves share
            # the same spatial grid (as video's noisy/clean halves do) but each
            # carries its own per-frame timestep (noisy: tactile_global_timesteps;
            # clean: tactile_global_cond_timesteps).
            if tactile_grid_shape is None:
                raise ValueError(
                    "tactile_grid_shape is required when tactile tokens are present."
                )
            B_t, S_t, Fp_t, Hp_t, Wp_t = tactile_grid_shape
            p_h, p_w = self.patch_size[1], self.patch_size[2]
            H_lat_t, W_lat_t = Hp_t * p_h, Wp_t * p_w

            # Per-half spatial grid, batch-flattened to match the (1, L, ...) layout
            # used throughout this stage. Token order is sensor-major then frame /
            # spatial, identical to _encode_tactile_condition.
            tactile_grid_id = (
                self._build_tactile_grid_id(
                    B_t,
                    S_t,
                    Fp_t,
                    Hp_t,
                    Wp_t,
                    device=full_grid_id.device,
                    dtype=full_grid_id.dtype,
                )
                .permute(1, 0, 2)
                .flatten(1)[None]
            )  # (1, 4, B*S*Fp*Hp*Wp)

            def _tactile_half_temb(timesteps_bf):
                # timesteps_bf: (B, Fp) per-frame diffusion timesteps, shared
                # across sensors and spatial patches. Expand to (1, B*S*Fp) in
                # (b, s, f) order, then _time_embed replicates over the Hp*Wp
                # spatial patches → (1, B*S*Fp*Hp*Wp), matching the token layout.
                ts = timesteps_bf[:, None, :].expand(B_t, S_t, -1).reshape(1, -1)
                return self._time_embed(
                    ts, H_lat_t, W_lat_t, dtype=dtype, action_mode=False
                )

            if tactile_noisy_token_length > 0:
                noisy_temb, noisy_ts_proj = _tactile_half_temb(
                    action_dict["tactile_global_timesteps"]
                )
                full_grid_id = torch.cat([full_grid_id, tactile_grid_id], dim=2)
                temb = torch.cat([temb, noisy_temb], dim=1)
                timestep_proj = torch.cat([timestep_proj, noisy_ts_proj], dim=1)

            if tactile_clean_token_length > 0:
                clean_ts = action_dict.get("tactile_global_cond_timesteps")
                if clean_ts is None:
                    # legacy fallback: clean condition with no noisy counterpart
                    # is fully clean → timestep 0.
                    clean_ts = torch.zeros(B_t, Fp_t, device=temb.device)
                clean_temb, clean_ts_proj = _tactile_half_temb(clean_ts)
                full_grid_id = torch.cat([full_grid_id, tactile_grid_id], dim=2)
                temb = torch.cat([temb, clean_temb], dim=1)
                timestep_proj = torch.cat([timestep_proj, clean_ts_proj], dim=1)

        rotary_emb = self.rope(full_grid_id)[:, :, None]
        return rotary_emb, temb, timestep_proj

    def _pad_stage_tensors(self, hidden_states, rotary_emb, temb, timestep_proj):
        padded_length = (128 - hidden_states.shape[1] % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))
        return hidden_states, rotary_emb, temb, timestep_proj, padded_length

    def _apply_output_norm(self, hidden_states, temb):
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(
            2, dim=1
        )
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        return (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(
            hidden_states
        )

    def _run_block_slice(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep_proj,
        rotary_emb,
        block_slice,
    ):
        for block in block_slice:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=False,
            )
        return hidden_states

    def _run_main_blocks(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep_proj,
        temb,
        rotary_emb,
        update_cache,
        cache_name,
        action_mode,
        main_token_count,
        tactile_token_count,
    ):
        """Streaming-inference block loop, extracted as an overridable hook so the
        MoT variant swaps the single shared stack for per-modality experts (mirrors
        _run_backbone for the training path). Default = legacy shared stack."""
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )
        return hidden_states

    def _set_block_slice_masks(
        self,
        block_slice,
        self_attention_mask: BlockMask | GroupedAttentionMask | None,
        cross_attention_mask: BlockMask | GroupedAttentionMask | None,
    ):
        for block in block_slice:
            block.set_flex_attention_masks(self_attention_mask, cross_attention_mask)

    def _run_backbone(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep_proj,
        rotary_emb,
        self_attention_mask,
        cross_attention_mask,
        split_list,
        batch_size,
        temb=None,
    ):
        """Run the transformer backbone over the assembled sequence. Extracted as
        an overridable hook so the Mixture-of-Transformers variant
        (WanMoTTransformer3DModel) can swap the single shared stack for per-modality
        experts WITHOUT duplicating forward_train. Default = the legacy shared
        stack; behaviour-identical to the previous inline two lines. `split_list`,
        `batch_size` and `temb` (the time VECTOR, for MoT narrow experts) are unused
        here but consumed by the MoT override."""
        self._set_block_slice_masks(
            self.blocks, self_attention_mask, cross_attention_mask
        )
        return self._run_block_slice(
            hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self.blocks
        )

    def _project_latent_tokens(self, latent_hidden_states, batch_size):
        latent_hidden_states = self.proj_out(latent_hidden_states)
        return rearrange(
            latent_hidden_states,
            "1 (b l) (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
            b=batch_size,
        )

    def _project_action_tokens(self, action_hidden_states, batch_size):
        action_hidden_states = self.action_proj_out(action_hidden_states)
        return rearrange(action_hidden_states, "1 (b l) c -> b l c", b=batch_size)

    def forward_train(self, input_dict):
        input_dict["latent_dict"]["noisy_latents"] = input_dict["latent_dict"][
            "noisy_latents"
        ].to(torch.bfloat16)
        input_dict["latent_dict"]["latent"] = input_dict["latent_dict"]["latent"].to(
            torch.bfloat16
        )
        input_dict["action_dict"]["noisy_latents"] = input_dict["action_dict"][
            "noisy_latents"
        ].to(torch.bfloat16)
        input_dict["action_dict"]["latent"] = input_dict["action_dict"]["latent"].to(
            torch.bfloat16
        )

        prepared = self._prepare_train_inputs(input_dict)
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        batch_size = prepared["batch_size"]

        # Symdiff tactile: sequence has 6 segments
        #   [v_noisy, v_clean, a_noisy, a_clean, gt_noisy, gt_clean]
        parts = [
            prepared["latent_hidden_states"],
            prepared["condition_latent_hidden_states"],
            prepared["action_hidden_states"],
            prepared["condition_action_hidden_states"],
        ]
        if prepared.get("tactile_noisy_hidden_states") is not None:
            parts.append(prepared["tactile_noisy_hidden_states"])
        if prepared.get("tactile_clean_hidden_states") is not None:
            parts.append(prepared["tactile_clean_hidden_states"])
        hidden_states = torch.cat(parts, dim=1)

        # Combined tactile token length (noisy + clean) drives mask sizing
        total_tactile_token_length = prepared.get(
            "tactile_noisy_token_length", 0
        ) + prepared.get("tactile_clean_token_length", 0)

        rotary_emb, temb, timestep_proj = self._build_stage_position_inputs(
            latent_dict,
            action_dict,
            hidden_states.dtype,
            include_action_tokens=True,
            tactile_token_length=total_tactile_token_length,
            tactile_noisy_token_length=prepared.get("tactile_noisy_token_length", 0),
            tactile_clean_token_length=prepared.get("tactile_clean_token_length", 0),
            tactile_grid_shape=prepared.get("tactile_grid_shape"),
        )
        hidden_states, rotary_emb, temb, timestep_proj, padded_length = (
            self._pad_stage_tensors(hidden_states, rotary_emb, temb, timestep_proj)
        )

        split_list = [
            prepared["latent_hidden_states"].shape[1],
            prepared["condition_latent_hidden_states"].shape[1],
            prepared["action_hidden_states"].shape[1],
            prepared["condition_action_hidden_states"].shape[1],
            prepared.get("tactile_noisy_token_length", 0),
            prepared.get("tactile_clean_token_length", 0),
            padded_length,
        ]

        self_attention_mask, cross_attention_mask = FlexAttnFunc.init_mask(
            latent_dict["noisy_latents"].shape,
            action_dict["noisy_latents"].shape,
            padded_length,
            input_dict["chunk_size"],
            window_size=input_dict["window_size"],
            patch_size=self.patch_size,
            device=hidden_states.device,
            text_token_length=prepared["text_hidden_states"].shape[1],
            tactile_token_length=total_tactile_token_length,
            tactile_noisy_token_length=prepared.get("tactile_noisy_token_length", 0),
            include_action_tokens=True,
            tactile_grid_shape=prepared.get("tactile_grid_shape"),
        )
        hidden_states = self._run_backbone(
            hidden_states,
            prepared["encoder_hidden_states"],
            timestep_proj,
            rotary_emb,
            self_attention_mask,
            cross_attention_mask,
            split_list,
            batch_size,
            temb,
        )
        hidden_states = self._apply_output_norm(hidden_states, temb)
        # split: v_noisy, v_clean, a_noisy, a_clean, gt_noisy, gt_clean, pad
        # tactile_clean_out (gt_clean) is the GlobalTactile the action conditions
        # on in the attention design (action_noisy(2k+1) -> tactile_clean(2k));
        # the contact gate reads the SAME source for consistency.
        (
            latent_hidden_states,
            _,
            action_hidden_states,
            _,
            tactile_noisy_out,
            tactile_clean_out,
            _,
        ) = torch.split(hidden_states, split_list, dim=1)

        # LocalTactile cross-attention BEFORE action head. CFG tactile drop
        # skips this path entirely, so tactile modules receive no gradient.
        # Chunk-causal mask: action only attends LocalTactile up to its own chunk
        # (no future-tactile leak), matching the self-attention causal structure.
        if prepared["local_tactile_tokens"] is not None:
            local_attn_mask = None
            if prepared.get("tactile_grid_shape") is not None:
                local_attn_mask = self._local_tactile_causal_mask(
                    action_dict["noisy_latents"].shape,
                    prepared["tactile_grid_shape"],
                    input_dict["chunk_size"],
                    device=action_hidden_states.device,
                )
            action_hidden_states = self._apply_local_tactile_cross_attn(
                action_hidden_states,
                prepared["local_tactile_tokens"],
                attn_mask=local_attn_mask,
            )

        if prepared["wrench_tokens"] is not None:
            wrench_attn_mask = self._agilex_wrench_causal_mask(
                action_dict["noisy_latents"].shape,
                wrench_frames=prepared["wrench_frames"],
                wrench_arms=prepared["wrench_arms"],
                chunk_size=input_dict["chunk_size"],
                device=action_hidden_states.device,
            )
            action_hidden_states = self._apply_agilex_wrench_cross_attn(
                action_hidden_states,
                prepared["wrench_tokens"],
                attn_mask=wrench_attn_mask,
            )

        if prepared["tactile_zero_anchor"] is not None:
            action_hidden_states = action_hidden_states + prepared[
                "tactile_zero_anchor"
            ].to(action_hidden_states.dtype)
        if prepared["wrench_zero_anchor"] is not None:
            action_hidden_states = action_hidden_states + prepared[
                "wrench_zero_anchor"
            ].to(action_hidden_states.dtype)

        # Predictive-contact gate: gate the action stream by the GlobalTactile the
        # action conditions on — tactile_CLEAN (gt_clean), i.e. the step-1-produced
        # tactile (GT at train / denoised prediction at inference), matching the
        # attention rule action_noisy(2k+1) -> tactile_clean(2k). NOT the gt_noisy
        # prediction-in-progress (which the action does not attend to and would mix
        # same-frame noise into the action). Identity at init (zero FiLM) -> no-op
        # until trained. Gradient flows (stop_grad=False by default): the action
        # loss co-shapes the CLEAN-tactile *conditioning* representation (joint
        # optimisation), WITHOUT directly touching the gt_noisy prediction target,
        # so tactile prediction stays protected by construction. CFG-drop
        # (tactile_clean_out empty) -> no-op. inference uses the gt clean hidden.
        if self.use_contact_gate and not prepared["contact_condition_dropped"]:
            action_hidden_states = self.contact_gate(
                action_hidden_states, tactile_clean_out, batch_size
            )

        # Symdiff tactile: predict tactile velocity from gt_noisy hidden.
        # tactile_noisy_out has shape (1, B*S*F_lat*patch_n, inner_dim);
        # tactile_proj_out outputs (1, B*S*F_lat*patch_n, C*prod(patch_size)),
        # then we rearrange to (B, S*F_lat*spatial, C) for caller.
        tactile_pred = None
        if tactile_noisy_out is not None and tactile_noisy_out.shape[1] > 0:
            tactile_pred_raw = self.tactile_proj_out(tactile_noisy_out)
            tactile_pred = rearrange(
                tactile_pred_raw,
                "1 (b l) (n c) -> b (l n) c",
                n=math.prod(self.patch_size),
                b=batch_size,
            )

        return (
            self._project_latent_tokens(latent_hidden_states, batch_size),
            self._project_action_tokens(action_hidden_states, batch_size),
            tactile_pred,
        )

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        r"""Run the diffusion backbone.

        Args:
            input_dict: packed latents and conditions (video / tactile / action
                tokens, text embeddings, timesteps and masks) built by the caller.
            update_cache: KV-cache mode (0 = no cache, 1 = commit predicted tokens).
            cache_name: name of the streaming KV-cache pool to use.
            action_mode: if True, run only the action expert / head.
            train_mode: if True, dispatch to ``forward_train`` (returns the
                per-modality velocity predictions for the flow-matching loss).

        Returns:
            The denoised velocity prediction(s) for the active modalities.
        """
        if train_mode:
            return self.forward_train(input_dict)

        # Training installs length-specific FlexAttention masks on the blocks.
        # Inference uses its own KV-cache masking, so clear any stale train mask.
        # MoT deletes self.blocks (per-modality experts instead); a pure serve only
        # ever infers, so there is no stale train mask to clear -> skip for MoT.
        if hasattr(self, "blocks"):
            self._set_block_slice_masks(self.blocks, None, None)

        if action_mode:  # action input emb
            latent_hidden_states = rearrange(
                input_dict["noisy_latents"], "b c f h w -> b (f h w) c"
            )
            latent_hidden_states = self.action_embedder(latent_hidden_states)  # B L1 C
        else:  # latent input emb
            latent_hidden_states = rearrange(
                input_dict["noisy_latents"],
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            latent_hidden_states = self.patch_embedding_mlp(latent_hidden_states)
        text_hidden_states = self._encode_text_condition(input_dict["text_emb"])

        # ─── Inference-time tactile injection ───
        # GlobalTactile latent → token, concat to self-attn sequence for both
        # video and action inference. This mirrors training, where video/action
        # tokens and GlobalTactile clean condition tokens share the backbone.
        # LocalTactile remains action-only and is consumed by the action head.
        #
        # TACTILE DENOISE (video path only): when input_dict carries a
        # 'tactile_noisy_latent' (the server's per-step noisy GlobalTactile), we
        # inject a NOISY tactile token block that co-generates with the video noisy
        # latent (same FlowMatch schedule, frame-aligned) and predict its velocity
        # via tactile_proj_out (mirrors forward_train).
        # Sequence layout: DENOISE -> [video_noisy | gt_noisy] (NO same-frame clean —
        # training's noise2clean is exclude-self, and a mask-less streaming attention
        # would otherwise let the noisy tactile copy the observed same-frame clean;
        # past clean tactile comes from the KV-cache, symmetric to video). Non-denoise
        # (legacy) -> [video | gt_clean]. action path is UNCHANGED (single return) so
        # existing action inference is untouched.
        main_token_count = latent_hidden_states.shape[1]
        tactile_disabled = input_dict.get("tactile_mode") == "disabled"
        tactile_dropped = self._should_drop_tactile_condition(input_dict)
        skip_tactile = tactile_disabled or tactile_dropped
        tactile_keys = {
            "tactile_global_latent",
            "tactile_local_latent",
            "tactile_sensor_ids",
            "tactile_noisy_latent",
            "tactile_timesteps",
            "wrench",
            "wrench_available_mask",
        }
        if tactile_disabled and any(key in input_dict for key in tactile_keys):
            raise ValueError("disabled tactile mode cannot carry tactile tensors")
        if tactile_disabled and (self.use_local_tactile or self.use_contact_gate):
            raise ValueError(
                "disabled tactile mode requires local tactile and contact gate off"
            )
        global_tactile_latent = None
        tactile_sensor_ids = None
        if not skip_tactile:
            global_tactile_latent = self._require_tactile_entry(
                input_dict,
                "tactile_global_latent",
                "input_dict",
            )
            tactile_sensor_ids = self._require_tactile_entry(
                input_dict,
                "tactile_sensor_ids",
                "input_dict",
            )
            if global_tactile_latent.dtype != latent_hidden_states.dtype:
                global_tactile_latent = global_tactile_latent.to(
                    latent_hidden_states.dtype
                )

        tactile_noisy_latent = input_dict.get("tactile_noisy_latent", None)
        denoise_tactile = (
            not skip_tactile
            and (not action_mode)
            and (tactile_noisy_latent is not None)
        )
        gt_noisy_tokens = None
        gt_noisy_token_count = 0
        gt_tokens = None
        tactile_clean_token_count = 0
        if skip_tactile:
            tactile_token_count = 0
        elif denoise_tactile:
            # only the noisy tactile block is in the sequence -> skip encoding the
            # (unused) clean tokens entirely.
            if tactile_noisy_latent.dtype != latent_hidden_states.dtype:
                tactile_noisy_latent = tactile_noisy_latent.to(
                    latent_hidden_states.dtype
                )
            gt_noisy_tokens = self._encode_tactile_condition(
                tactile_noisy_latent,
                sensor_ids=tactile_sensor_ids,
            )
            gt_noisy_token_count = gt_noisy_tokens.shape[1]
            if gt_noisy_token_count <= 0:
                raise ValueError(
                    "forward requires positive tactile_token_count in tactile cond branch."
                )
        else:
            gt_tokens = self._encode_tactile_condition(
                global_tactile_latent,
                sensor_ids=tactile_sensor_ids,
            )
            tactile_clean_token_count = gt_tokens.shape[1]
            if tactile_clean_token_count <= 0:
                raise ValueError(
                    "forward requires positive tactile_token_count in tactile cond branch."
                )
        # total tactile tokens appended after the main (video/action) block.
        # DENOISE (video pass): sequence = [video_noisy | gt_noisy] ONLY — do NOT
        # append the SAME-FRAME clean tactile. Training's mask gives tactile_noisy
        # `noise2clean ∧ block_causal_EXCLUDE_self` (model.py mask_list): it may
        # condition on PAST clean tactile (which here lives in the KV-cache from prior
        # chunks' compute_kv_cache real observations), but NEVER the same-frame clean.
        # Appending same-frame gt_clean here (the observed tactile) let the mask-less
        # streaming attention copy the answer -> defeats "predict tactile". Same-frame
        # coupling that IS wanted (video_noisy ↔ tactile_noisy) is already present via
        # the video_noisy block in the sequence. Non-denoise path keeps the legacy
        # clean-condition behaviour unchanged.
        if skip_tactile:
            pass
        elif denoise_tactile:
            tactile_token_count = gt_noisy_token_count  # [video | gt_noisy]
            latent_hidden_states = torch.cat(
                [latent_hidden_states, gt_noisy_tokens],
                dim=1,
            )
        else:
            tactile_token_count = tactile_clean_token_count
            latent_hidden_states = torch.cat(
                [latent_hidden_states, gt_tokens],
                dim=1,
            )

        local_tactile_tokens = None
        if action_mode and self.use_local_tactile and not skip_tactile:
            local_tactile_latent = self._require_tactile_entry(
                input_dict,
                "tactile_local_latent",
                "input_dict",
            )
            if local_tactile_latent.dtype != latent_hidden_states.dtype:
                local_tactile_latent = local_tactile_latent.to(
                    latent_hidden_states.dtype
                )
            local_tactile_tokens = self._encode_local_tactile(
                local_tactile_latent,
                sensor_ids=tactile_sensor_ids,
            )

        wrench_tokens = None
        if (
            action_mode
            and self.use_wrench_conditioner
            and not tactile_disabled
            and input_dict.get("wrench") is not None
        ):
            required_wrench_fields = {
                "wrench_available_mask",
                "temporal_valid_mask",
                "contact_cond_drop",
            }
            missing_wrench_fields = sorted(required_wrench_fields - set(input_dict))
            if missing_wrench_fields:
                raise ValueError(
                    "wrench conditioning is missing fields: "
                    + ", ".join(missing_wrench_fields)
                )
            wrench_tokens = self._encode_agilex_wrench_condition(
                input_dict["wrench"].to(latent_hidden_states.dtype),
                wrench_available_mask=input_dict["wrench_available_mask"],
                temporal_valid_mask=input_dict["temporal_valid_mask"],
                contact_cond_drop=input_dict["contact_cond_drop"],
            )

        # Symdiff tactile: tactile tokens carry REAL spatial/temporal RoPE
        # positions (matching forward_train), so the model localises which sensor
        # / frame / spatial patch each tactile token belongs to.
        latent_grid_id = input_dict["grid_id"]
        B_g = latent_grid_id.shape[0]
        if skip_tactile:
            S_t = Fp_t = Hp_t = Wp_t = 0
            full_grid_id = latent_grid_id
        else:
            _, S_t, Fp_t, Hp_t, Wp_t = self._tactile_patch_grid_shape(
                global_tactile_latent
            )
            # Serving supplies this CPU scalar so the denoise hot path does not
            # synchronize the accelerator merely to recover a known frame id.
            _vid_frame_start_value = input_dict.get("frame_start_id")
            if _vid_frame_start_value is None:
                _vid_frame_start = int(latent_grid_id[0, 0].min().item())
            elif (
                isinstance(_vid_frame_start_value, bool)
                or not isinstance(_vid_frame_start_value, int)
                or _vid_frame_start_value < 0
            ):
                raise ValueError("frame_start_id must be a non-negative integer")
            else:
                _vid_frame_start = _vid_frame_start_value
            tactile_grid_id = self._build_tactile_grid_id(
                B_g,
                S_t,
                Fp_t,
                Hp_t,
                Wp_t,
                device=latent_grid_id.device,
                dtype=latent_grid_id.dtype,
                frame_start=_vid_frame_start,
            )
            full_grid_id = torch.cat([latent_grid_id, tactile_grid_id], dim=2)
        rotary_emb = self.rope(full_grid_id)[:, :, None]  # 1 L 1 C
        pach_scale_h, pach_scale_w = (
            (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        )

        latent_time_steps = torch.repeat_interleave(
            input_dict["timesteps"],
            (input_dict["noisy_latents"].shape[-2] // pach_scale_h)
            * (input_dict["noisy_latents"].shape[-1] // pach_scale_w),
            dim=1,
        )  # L
        current_condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C

        p_h, p_w = self.patch_size[1], self.patch_size[2]
        if skip_tactile:
            pass
        elif denoise_tactile:
            # tactile block = gt_noisy: per-frame DIFFUSION timestep (mirrors training's
            # tactile_global_timesteps noisy half). server passes 'tactile_timesteps'
            # as (B,F) (or S*F) at the current denoise step.
            t_noisy = input_dict.get("tactile_timesteps", None)
            if t_noisy is None:
                t_noisy = (
                    input_dict["timesteps"].reshape(-1)[:1].expand(B_g, S_t * Fp_t)
                )
            else:
                t_noisy = t_noisy.to(temb.device).reshape(B_g, -1)
                if t_noisy.shape[1] != S_t * Fp_t:
                    t_noisy = (
                        t_noisy[:, None, :]
                        .expand(B_g, S_t, Fp_t)
                        .reshape(B_g, S_t * Fp_t)
                    )
            tactile_noisy_temb, tactile_noisy_ts_proj = self._time_embed(
                t_noisy,
                Hp_t * p_h,
                Wp_t * p_w,
                dtype=latent_hidden_states.dtype,
                action_mode=False,
            )
            temb = torch.cat([temb, tactile_noisy_temb], dim=1)  # [video | gt_noisy]
            timestep_proj = torch.cat([timestep_proj, tactile_noisy_ts_proj], dim=1)
        else:
            # tactile block = gt_clean condition -> timestep 0 (legacy behaviour).
            tactile_clean_ts = torch.zeros(B_g, S_t * Fp_t, device=temb.device)
            tactile_temb, tactile_ts_proj = self._time_embed(
                tactile_clean_ts,
                Hp_t * p_h,
                Wp_t * p_w,
                dtype=latent_hidden_states.dtype,
                action_mode=False,
            )
            temb = torch.cat([temb, tactile_temb], dim=1)
            timestep_proj = torch.cat([timestep_proj, tactile_ts_proj], dim=1)

        latent_hidden_states = self._run_main_blocks(
            latent_hidden_states,
            text_hidden_states,
            timestep_proj,
            temb,
            rotary_emb,
            update_cache,
            cache_name,
            action_mode,
            main_token_count,
            tactile_token_count,
        )
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(
            2, dim=1
        )
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (
            self.norm_out(latent_hidden_states.float()) * (1.0 + scale) + shift
        ).type_as(latent_hidden_states)

        # TACTILE DENOISE (video path): the whole tactile tail IS the gt_noisy block
        # now (no same-frame clean appended). Predict tactile velocity from it,
        # mirroring forward_train's tactile_proj_out(tactile_noisy_out).
        tactile_pred = None
        if denoise_tactile and gt_noisy_token_count > 0:
            tactile_noisy_out = latent_hidden_states[
                :, main_token_count : main_token_count + gt_noisy_token_count, :
            ]
            tactile_pred_raw = self.tactile_proj_out(tactile_noisy_out)
            tactile_pred = rearrange(
                tactile_pred_raw, "b l (n c) -> b (l n) c", n=math.prod(self.patch_size)
            )

        # contact gate uses the clean tactile tail (action pass only; non-denoise, so
        # the tail is gt_clean). In the denoise video pass action_mode is False ->
        # gt_hidden_for_gate is None anyway.
        gt_hidden_for_gate = (
            latent_hidden_states[:, main_token_count:, :]
            if (action_mode and self.use_contact_gate and not skip_tactile)
            else None
        )
        # Drop the tactile tail before the modality output head.
        latent_hidden_states = latent_hidden_states[:, :main_token_count, :]

        if action_mode:
            # LocalTactile cross-attn before action head (skipped when
            # use_local_tactile=False -> tokens None -> action head sees video/global only)
            if local_tactile_tokens is not None:
                latent_hidden_states = self._apply_local_tactile_cross_attn(
                    latent_hidden_states,
                    local_tactile_tokens,
                )
            if wrench_tokens is not None:
                latent_hidden_states = self._apply_agilex_wrench_cross_attn(
                    latent_hidden_states,
                    wrench_tokens,
                )
            if self.use_contact_gate and not skip_tactile:
                latent_hidden_states = self.contact_gate(
                    latent_hidden_states,
                    gt_hidden_for_gate,
                    batch_size=latent_hidden_states.shape[0],
                )
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(
                latent_hidden_states,
                "b l (n c) -> b (l n) c",
                n=math.prod(self.patch_size),
            )  #

        # video path co-generating tactile -> return (video_pred, tactile_pred);
        # all other paths (action, or video without tactile denoise) keep the
        # legacy SINGLE return so existing callers are untouched.
        if tactile_pred is not None:
            return latent_hidden_states, tactile_pred
        return latent_hidden_states


if __name__ == "__main__":
    model = WanTransformer3DModel(
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
    )
    print(model)
