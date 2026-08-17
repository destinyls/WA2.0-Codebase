# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Masked wrist-wrench conditioning for the AgileX action route."""

from __future__ import annotations

import torch
import torch.nn as nn


class WrenchConditioner(nn.Module):
    """Encode per-arm 6D wrench observations into zero-init action tokens.

    Availability, temporal validity, and sample-level condition drop are applied
    both before feature extraction and after projection. Consequently placeholder
    bytes cannot affect action conditioning through MLP biases or embeddings.
    """

    def __init__(self, *, hidden_dim: int, max_frames: int, arm_count: int = 2) -> None:
        super().__init__()
        if hidden_dim <= 0 or max_frames <= 0 or arm_count <= 0:
            raise ValueError("hidden_dim, max_frames and arm_count must be positive")
        self.hidden_dim = int(hidden_dim)
        self.max_frames = int(max_frames)
        self.arm_count = int(arm_count)
        self.wrench_mlp = nn.Sequential(
            nn.Linear(6, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.arm_embedding = nn.Embedding(self.arm_count, self.hidden_dim)
        self.frame_embedding = nn.Embedding(self.max_frames, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _require_bool_mask(
        value: torch.Tensor,
        *,
        shape: tuple[int, ...],
        label: str,
        device: torch.device,
    ) -> torch.Tensor:
        if value.shape != shape or value.dtype is not torch.bool:
            raise ValueError(f"{label} must have bool shape {shape}")
        if value.device != device:
            raise ValueError(f"{label} must be on {device}, got {value.device}")
        return value

    def forward(
        self,
        wrench: torch.Tensor,
        *,
        wrench_available_mask: torch.Tensor,
        temporal_valid_mask: torch.Tensor,
        contact_cond_drop: torch.Tensor,
    ) -> torch.Tensor:
        if wrench.ndim != 4 or wrench.shape[-1] != 6:
            raise ValueError("wrench must have shape [B,F,A,6]")
        if not wrench.is_floating_point():
            raise TypeError("wrench must use a floating-point dtype")
        if not torch.isfinite(wrench).all():
            raise ValueError("wrench contains non-finite values")
        batch, frames, arms, _ = wrench.shape
        if frames > self.max_frames:
            raise ValueError(
                f"wrench frame count {frames} exceeds max_frames={self.max_frames}"
            )
        if arms != self.arm_count:
            raise ValueError(
                f"wrench arm count {arms} does not match arm_count={self.arm_count}"
            )
        device = wrench.device
        available = self._require_bool_mask(
            wrench_available_mask,
            shape=(batch, frames, arms),
            label="wrench_available_mask",
            device=device,
        )
        temporal = self._require_bool_mask(
            temporal_valid_mask,
            shape=(batch, frames),
            label="temporal_valid_mask",
            device=device,
        )
        cond_drop = self._require_bool_mask(
            contact_cond_drop,
            shape=(batch,),
            label="contact_cond_drop",
            device=device,
        )
        condition_mask = available & temporal[:, :, None] & ~cond_drop[:, None, None]
        clean_wrench = wrench.masked_fill(~condition_mask[..., None], 0.0)
        hidden = self.wrench_mlp(clean_wrench)
        arm_ids = torch.arange(arms, device=device)
        frame_ids = torch.arange(frames, device=device)
        hidden = (
            hidden
            + self.arm_embedding(arm_ids)[None, None, :, :]
            + self.frame_embedding(frame_ids)[None, :, None, :]
        )
        hidden = hidden.masked_fill(~condition_mask[..., None], 0.0)
        hidden = self.norm(hidden)
        output = self.output_projection(hidden)
        output = output.masked_fill(~condition_mask[..., None], 0.0)
        return output.reshape(batch, frames * arms, self.hidden_dim)


__all__ = ["WrenchConditioner"]
