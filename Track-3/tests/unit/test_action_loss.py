# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import pytest
import torch

from n0_twam.actions.loss import (
    build_action_loss_profile,
    weighted_action_flow_mse,
)


def test_legacy_profile_reproduces_unweighted_mse() -> None:
    prediction = torch.tensor([[[[[1.0], [0.0]]], [[[0.0], [0.0]]]]])
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    loss = weighted_action_flow_mse(
        prediction,
        target,
        mask,
        torch.ones(1, 1),
    )
    assert loss.item() == pytest.approx(0.25)


def test_channel_and_near_horizon_weights_emphasize_important_error() -> None:
    prediction = torch.tensor([[[[[1.0], [0.0]]], [[[0.0], [0.0]]]]])
    target = torch.zeros_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    loss = weighted_action_flow_mse(
        prediction,
        target,
        mask,
        torch.ones(1, 1),
        channel_weights=(2.0, 1.0),
        horizon_weights=(3.0, 1.0),
    )
    assert loss.item() == pytest.approx(0.5)


def test_empty_action_mask_returns_differentiable_zero() -> None:
    prediction = torch.ones(1, 2, 1, 2, 1, requires_grad=True)
    loss = weighted_action_flow_mse(
        prediction,
        torch.zeros_like(prediction),
        torch.zeros_like(prediction, dtype=torch.bool),
        torch.ones(1, 1),
    )
    loss.backward()
    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_franka_trajectory_profile_targets_xyz_z_gripper_and_near_actions() -> None:
    profile = build_action_loss_profile(
        "franka_trajectory_fit_v1",
        action_dim=20,
        action_horizon=6,
    )
    assert profile.scale == 25.0
    assert profile.channel_weights[:3] == (4.0, 4.0, 8.0)
    assert profile.channel_weights[9] == 4.0
    assert profile.horizon_weights == (4.0, 3.0, 2.0, 1.5, 1.0, 1.0)


def test_franka_trajectory_profile_rejects_wrong_action_contract() -> None:
    with pytest.raises(ValueError, match="requires EE20"):
        build_action_loss_profile(
            "franka_trajectory_fit_v1",
            action_dim=8,
            action_horizon=1,
        )
