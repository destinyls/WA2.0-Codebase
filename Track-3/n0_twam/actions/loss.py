# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Weighted action flow-matching objectives for trajectory-focused training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ActionLossProfile:
    """Immutable action-loss weighting contract recorded with each run."""

    name: str
    scale: float
    channel_weights: tuple[float, ...]
    horizon_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("action loss profile name must be canonical")
        if not 0.0 < self.scale < float("inf"):
            raise ValueError("action loss scale must be positive and finite")
        for label, weights in (
            ("channel", self.channel_weights),
            ("horizon", self.horizon_weights),
        ):
            if not weights or any(
                not 0.0 < weight < float("inf") for weight in weights
            ):
                raise ValueError(f"{label} loss weights must be positive and finite")


def build_action_loss_profile(
    name: str,
    *,
    action_dim: int,
    action_horizon: int,
) -> ActionLossProfile:
    """Resolve one reproducible action-loss recipe by explicit name."""

    if action_dim <= 0 or action_horizon <= 0:
        raise ValueError("action loss dimensions must be positive")
    if name == "legacy_v1":
        return ActionLossProfile(
            name=name,
            scale=1.0,
            channel_weights=(1.0,) * action_dim,
            horizon_weights=(1.0,) * action_horizon,
        )
    if name == "franka_trajectory_fit_v1":
        if action_dim != 20 or action_horizon != 6:
            raise ValueError(
                "franka_trajectory_fit_v1 requires EE20 with six actions per frame"
            )
        return ActionLossProfile(
            name=name,
            scale=25.0,
            channel_weights=(
                4.0,
                4.0,
                8.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                4.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            horizon_weights=(4.0, 3.0, 2.0, 1.5, 1.0, 1.0),
        )
    raise ValueError(f"unsupported action loss profile: {name!r}")


def _weight_vector(
    values: Sequence[float] | torch.Tensor | None,
    *,
    length: int,
    label: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    if values is None:
        return torch.ones(length, dtype=torch.float32, device=reference.device)
    weights = torch.as_tensor(values, dtype=torch.float32, device=reference.device)
    if weights.shape != (length,):
        raise ValueError(f"{label} weights must have shape [{length}]")
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError(f"{label} weights must be positive and finite")
    return weights


def weighted_action_flow_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    frame_weights: torch.Tensor,
    *,
    channel_weights: Sequence[float] | torch.Tensor | None = None,
    horizon_weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute frame-normalized MSE with channel and horizon importance weights.

    Dense tensors follow ``[B,C,F,N,1]``. Scheduler frame weights preserve the
    existing flow-matching weighting, while channel/horizon weights redistribute
    supervision within each valid frame. All-one weights reproduce the legacy
    objective exactly.
    """

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask shapes must match")
    if prediction.ndim != 5 or prediction.shape[-1] != 1:
        raise ValueError("dense action tensors must have shape [B,C,F,N,1]")
    batch, channels, frames, horizon, _ = prediction.shape
    if frame_weights.shape != (batch, frames):
        raise ValueError("frame weights must have shape [B,F]")
    if not bool(torch.isfinite(frame_weights).all()):
        raise ValueError("frame weights must be finite")

    channel = _weight_vector(
        channel_weights,
        length=channels,
        label="channel",
        reference=prediction,
    ).reshape(1, channels, 1, 1, 1)
    future = _weight_vector(
        horizon_weights,
        length=horizon,
        label="horizon",
        reference=prediction,
    ).reshape(1, 1, 1, horizon, 1)
    valid = mask.float()
    element_weights = valid * channel * future
    squared_error = F.mse_loss(
        prediction.float(), target.float().detach(), reduction="none"
    )
    weighted = (
        squared_error
        * element_weights
        * frame_weights.float().reshape(batch, 1, frames, 1, 1)
    )

    weighted = weighted.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    element_weights = (
        element_weights.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    )
    numerator = weighted.sum(dim=1)
    denominator = element_weights.sum(dim=1)
    valid_frames = denominator > 0
    if not bool(valid_frames.any()):
        return prediction.float().sum() * 0.0
    return (numerator[valid_frames] / denominator[valid_frames]).mean()


__all__ = (
    "ActionLossProfile",
    "build_action_loss_profile",
    "weighted_action_flow_mse",
)
