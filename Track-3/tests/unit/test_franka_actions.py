# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import numpy as np
import pytest

from n0_twam.integrations.worldarena.franka_actions import (
    ee10_to_end_pose8,
    embed_ee10_in_ee20,
    end_pose8_to_ee10,
    extract_ee10_from_ee20,
    quaternion_wxyz_to_matrix,
)


def _golden_poses() -> np.ndarray:
    return np.asarray(
        (
            (0.31, -0.12, 0.48, 1.0, 0.0, 0.0, 0.0, 0.02),
            (0.22, 0.08, 0.37, 0.35, -0.2, 0.71, 0.57, 0.06),
            (-0.10, 0.14, 0.61, 0.5, 0.5, -0.5, 0.5, 0.01),
        ),
        dtype=np.float32,
    )


def test_end_pose8_ee10_round_trip_preserves_pose_and_wxyz_order() -> None:
    poses = _golden_poses()

    ee10 = end_pose8_to_ee10(poses)
    decoded = ee10_to_end_pose8(ee10, quaternion_reference=poses[:, 3:7])

    assert ee10.shape == (3, 10)
    np.testing.assert_allclose(decoded[:, :3], poses[:, :3], atol=1e-6)
    np.testing.assert_allclose(decoded[:, 7], poses[:, 7], atol=1e-6)
    np.testing.assert_allclose(
        quaternion_wxyz_to_matrix(decoded[:, 3:7]),
        quaternion_wxyz_to_matrix(poses[:, 3:7]),
        atol=1e-6,
    )
    assert not np.allclose(decoded[1, 3:7], poses[1, [4, 5, 6, 3]])


def test_franka_ee10_embeds_only_in_first_half_of_released_head() -> None:
    ee10 = end_pose8_to_ee10(_golden_poses())

    ee20 = embed_ee10_in_ee20(ee10)

    assert ee20.shape == (3, 20)
    np.testing.assert_array_equal(ee20[:, :10], ee10)
    np.testing.assert_array_equal(ee20[:, 10:], np.zeros((3, 10), np.float32))
    np.testing.assert_array_equal(extract_ee10_from_ee20(ee20), ee10)


def test_franka_action_conversion_rejects_joint8_guess_and_bad_rotation() -> None:
    with pytest.raises(ValueError, match="last dimension must be 8"):
        end_pose8_to_ee10(np.zeros((2, 7), dtype=np.float32))

    degenerate = np.zeros((1, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="norm is too small"):
        end_pose8_to_ee10(degenerate)

    malformed = np.zeros((1, 10), dtype=np.float32)
    malformed[:, 3] = 1.0
    malformed[:, 6] = 1.0
    with pytest.raises(ValueError, match="collinear"):
        ee10_to_end_pose8(malformed)
