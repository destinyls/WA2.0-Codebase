# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import numpy as np
import pytest

from n0_twam.actions import build_action_codec
from n0_twam.data.qpos8_alignment import build_qpos8_latent_targets


def _identity_like_codec():
    return build_action_codec(
        "qpos8_next_step",
        q01=(0.0,) * 8,
        q99=(2.0,) * 8,
    )


def test_latent_alignment_starts_directly_at_next_step_target() -> None:
    actions = np.repeat(np.arange(9, dtype=np.float32)[:, None], 8, axis=1)

    targets = build_qpos8_latent_targets(
        local_start_frame=0,
        local_end_frame=9,
        latent_frame_ids=np.arange(9, dtype=np.int64),
        converted_actions=actions,
        codec=_identity_like_codec(),
        expected_slots_per_frame=4,
    )

    assert targets.actions.shape == (8, 3, 4, 1)
    assert targets.valid_mask.shape == targets.actions.shape
    np.testing.assert_allclose(
        targets.actions[0, 0, :, 0],
        np.asarray((-1.0, 0.0, 1.0, 2.0), dtype=np.float32),
        atol=2e-6,
    )
    assert targets.valid_mask[0, 0, :, 0].all()
    assert targets.valid_mask[0, 2, 0, 0]
    assert not targets.valid_mask[0, 2, 1:, 0].any()
    assert not targets.actions[0, 2, 1:, 0].any()


def test_latent_alignment_uses_segment_relative_action_offsets() -> None:
    actions = np.repeat(np.arange(8, dtype=np.float32)[:, None], 8, axis=1)

    targets = build_qpos8_latent_targets(
        local_start_frame=10,
        local_end_frame=18,
        latent_frame_ids=np.arange(10, 15, dtype=np.int64),
        converted_actions=actions,
        codec=_identity_like_codec(),
        expected_slots_per_frame=4,
    )

    assert targets.anchor_frame_ids.tolist() == [10, 14]
    assert targets.valid_mask.all()


def test_latent_alignment_rejects_schema_or_temporal_mismatch() -> None:
    actions = np.zeros((9, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="action_per_frame"):
        build_qpos8_latent_targets(
            local_start_frame=0,
            local_end_frame=9,
            latent_frame_ids=np.arange(9, dtype=np.int64),
            converted_actions=actions,
            codec=_identity_like_codec(),
            expected_slots_per_frame=3,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        build_qpos8_latent_targets(
            local_start_frame=0,
            local_end_frame=9,
            latent_frame_ids=np.asarray((0, 1, 1, 2, 3), dtype=np.int64),
            converted_actions=actions,
            codec=_identity_like_codec(),
            expected_slots_per_frame=4,
        )


def test_latent_alignment_rejects_zero_target_anchor_or_segment_drift() -> None:
    actions = np.zeros((5, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="action count"):
        build_qpos8_latent_targets(
            local_start_frame=0,
            local_end_frame=6,
            latent_frame_ids=np.arange(5, dtype=np.int64),
            converted_actions=actions,
            codec=_identity_like_codec(),
            expected_slots_per_frame=4,
        )
    with pytest.raises(ValueError, match="within segment bounds"):
        build_qpos8_latent_targets(
            local_start_frame=0,
            local_end_frame=5,
            latent_frame_ids=np.arange(9, dtype=np.int64),
            converted_actions=actions,
            codec=_identity_like_codec(),
            expected_slots_per_frame=4,
        )
