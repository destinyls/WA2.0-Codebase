# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import numpy as np
import pytest

from n0_twam.evaluation.franka_training_fit import summarize_franka_training_fit


def _poses() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.zeros((2, 2, 8), dtype=np.float32)
    target[..., 6] = 1.0
    predicted = target.copy()
    valid = np.ones((2, 2), dtype=np.bool_)
    return predicted, target, valid


def test_perfect_training_trajectory_passes_all_gates() -> None:
    predicted, target, valid = _poses()
    result = summarize_franka_training_fit(
        predicted=predicted,
        target=target,
        valid=valid,
        action_offsets=(1, 2),
    )
    assert result["all_gates_passed"] is True
    assert result["position_cm"]["p95"] == 0.0
    assert result["rotation_deg"]["p95"] == 0.0


def test_z_bias_and_tail_fail_explicit_training_gates() -> None:
    predicted, target, valid = _poses()
    predicted[..., 2] = 0.01
    predicted[1, 1, 2] = 0.04
    result = summarize_franka_training_fit(
        predicted=predicted,
        target=target,
        valid=valid,
        action_offsets=(1, 2),
    )
    assert result["all_gates_passed"] is False
    assert result["per_axis"]["z"]["signed_bias_cm"] == pytest.approx(1.75)
    assert result["gates"]["absolute_z_bias_cm"]["passed"] is False
    assert result["gates"]["z_p95_absolute_cm"]["passed"] is False


def test_quaternion_sign_flip_is_the_same_rotation() -> None:
    predicted, target, valid = _poses()
    predicted[..., 6] = -1.0
    result = summarize_franka_training_fit(
        predicted=predicted,
        target=target,
        valid=valid,
        action_offsets=(1, 2),
    )
    assert result["rotation_deg"]["max"] == 0.0


def test_training_fit_rejects_invalid_action_offsets() -> None:
    predicted, target, valid = _poses()
    with pytest.raises(ValueError, match="strictly increasing"):
        summarize_franka_training_fit(
            predicted=predicted,
            target=target,
            valid=valid,
            action_offsets=(1, 1),
        )
