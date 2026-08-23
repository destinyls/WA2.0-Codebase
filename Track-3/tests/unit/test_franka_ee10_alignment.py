# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import numpy as np
import pytest

from n0_twam.data.ee10_alignment import build_franka_ee10_latent_targets

SAMPLED_FRAME_IDS = np.asarray(
    (0, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18),
    dtype=np.int64,
)


def test_franka_alignment_uses_six_native_actions_per_wan_anchor() -> None:
    actions = np.repeat(np.arange(20, dtype=np.float32)[:, None], 10, axis=1)
    states = actions + np.float32(0.25)

    targets = build_franka_ee10_latent_targets(
        local_start_frame=0,
        local_end_frame=20,
        latent_frame_ids=SAMPLED_FRAME_IDS,
        converted_actions=actions,
        converted_states=states,
        action_q01=np.zeros(10, dtype=np.float32),
        action_q99=np.full(10, 20.0, dtype=np.float32),
    )

    assert targets.actions.shape == (20, 4, 6, 1)
    assert targets.valid_mask.shape == targets.actions.shape
    assert targets.anchor_frame_ids.tolist() == [0, 6, 12, 18]
    assert targets.slots_per_frame == 6
    assert targets.valid_mask[:10].all()
    assert not targets.valid_mask[10:].any()
    assert not targets.actions[10:].any()
    np.testing.assert_allclose(
        targets.actions[0, 0, :, 0],
        np.full(6, -0.975, dtype=np.float32),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        targets.actions[0, 1, :, 0],
        np.linspace(-1.0, -0.5, 6, dtype=np.float32),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        targets.actions[0, 2, :, 0],
        np.linspace(-0.4, 0.1, 6, dtype=np.float32),
        atol=2e-6,
    )


def test_franka_alignment_rejects_uniform_stride_assumption() -> None:
    actions = np.zeros((20, 10), dtype=np.float32)
    wrong = np.arange(13, dtype=np.int64)

    with pytest.raises(ValueError, match="advance six"):
        build_franka_ee10_latent_targets(
            local_start_frame=0,
            local_end_frame=20,
            latent_frame_ids=wrong,
            converted_actions=actions,
            converted_states=actions,
            action_q01=np.zeros(10, dtype=np.float32),
            action_q99=np.ones(10, dtype=np.float32),
        )

    with pytest.raises(ValueError, match="six action slots"):
        build_franka_ee10_latent_targets(
            local_start_frame=0,
            local_end_frame=20,
            latent_frame_ids=SAMPLED_FRAME_IDS,
            converted_actions=actions,
            converted_states=actions,
            action_q01=np.zeros(10, dtype=np.float32),
            action_q99=np.ones(10, dtype=np.float32),
            expected_slots_per_frame=4,
        )


def test_franka_alignment_rejects_missing_observation_state_rows() -> None:
    actions = np.zeros((20, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="states must match action shape"):
        build_franka_ee10_latent_targets(
            local_start_frame=0,
            local_end_frame=20,
            latent_frame_ids=SAMPLED_FRAME_IDS,
            converted_actions=actions,
            converted_states=actions[:-1],
            action_q01=np.zeros(10, dtype=np.float32),
            action_q99=np.ones(10, dtype=np.float32),
        )
