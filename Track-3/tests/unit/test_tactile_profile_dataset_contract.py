# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import pytest

from n0_twam.dataset.sample_shape_signature import (
    LatentTensorSpec,
    build_sample_shape_signature,
    episode_selection_sha256,
)
from n0_twam.tactile_profiles import validate_profile_segment_inventory


def test_episode_selection_identity_isolates_train_and_validation_caches() -> None:
    assert episode_selection_sha256((0, 1, 2)) != episode_selection_sha256((3, 4))
    assert episode_selection_sha256(None) is None


def test_profile_bound_repo_requires_at_least_one_valid_segment() -> None:
    with pytest.raises(ValueError, match="no valid segments"):
        validate_profile_segment_inventory(
            profile="mixed",
            repo_name="rgb",
            has_tactile=False,
            valid_segments=0,
            total_segments=2,
            rejection_counts={},
        )


@pytest.mark.parametrize(
    "reason",
    [
        "missing_tactile_latent",
        "bad_tactile_latent",
        "tactile_local_global_mismatch",
        "tactile_video_frame_mismatch",
    ],
)
def test_declared_tactile_repo_rejects_any_filtered_tactile_segment(
    reason: str,
) -> None:
    with pytest.raises(ValueError, match="invalid tactile segments"):
        validate_profile_segment_inventory(
            profile="mixed",
            repo_name="touch",
            has_tactile=True,
            valid_segments=1,
            total_segments=2,
            rejection_counts={reason: 1},
        )


def test_rgb_repo_can_retain_video_filtering_when_valid_segments_remain() -> None:
    validate_profile_segment_inventory(
        profile="mixed",
        repo_name="rgb",
        has_tactile=False,
        valid_segments=1,
        total_segments=2,
        rejection_counts={"missing_video_latent": 1},
    )


def _video_spec(frames: int) -> LatentTensorSpec:
    return LatentTensorSpec(
        channels=48,
        frames=frames,
        height=8,
        width=8,
        dtype="torch.bfloat16",
    )


def test_shape_signature_separates_equal_modality_counts_with_different_lengths() -> (
    None
):
    short = build_sample_shape_signature(
        video_specs=(_video_spec(3), _video_spec(3)),
        text_shape=(512, 4096),
        text_dtype="torch.bfloat16",
        frame_ids=tuple(range(0, 18, 2)),
        max_latent_frames=0,
        action_dim=20,
        action_slots_per_frame=12,
    )
    long = build_sample_shape_signature(
        video_specs=(_video_spec(5), _video_spec(5)),
        text_shape=(512, 4096),
        text_dtype="torch.bfloat16",
        frame_ids=tuple(range(0, 34, 2)),
        max_latent_frames=0,
        action_dim=20,
        action_slots_per_frame=12,
    )

    assert short != long
    assert dict((name, shape) for name, shape, _ in short)["latents"] == (
        48,
        3,
        8,
        16,
    )


def test_shape_signature_uses_fixed_crop_and_adapter_action_horizon() -> None:
    signature = build_sample_shape_signature(
        video_specs=(_video_spec(7),),
        text_shape=(512, 4096),
        text_dtype="torch.bfloat16",
        frame_ids=tuple(range(0, 50, 2)),
        max_latent_frames=5,
        action_dim=20,
        action_slots_per_frame=6,
    )
    shapes = {name: shape for name, shape, _ in signature}

    assert shapes["latents"] == (48, 5, 8, 8)
    assert shapes["actions"] == (20, 5, 6, 1)


def test_franka_shape_signature_accepts_fifteen_to_ten_frame_sampling() -> None:
    signature = build_sample_shape_signature(
        video_specs=(_video_spec(3),),
        text_shape=(512, 4096),
        text_dtype="torch.bfloat16",
        frame_ids=(0, 2, 3, 4, 6, 8, 9, 10, 12),
        max_latent_frames=0,
        action_dim=20,
        action_slots_per_frame=6,
    )

    assert {name: shape for name, shape, _ in signature}["actions"] == (
        20,
        3,
        6,
        1,
    )


def test_shape_signature_rejects_unstackable_tactile_sensors() -> None:
    with pytest.raises(ValueError, match="identical stackable"):
        build_sample_shape_signature(
            video_specs=(_video_spec(3),),
            text_shape=(512, 4096),
            text_dtype="torch.bfloat16",
            frame_ids=tuple(range(0, 18, 2)),
            max_latent_frames=0,
            action_dim=20,
            tactile_global_specs=(
                _video_spec(3),
                LatentTensorSpec(48, 3, 8, 4, "torch.bfloat16"),
            ),
            tactile_local_specs=(
                _video_spec(3),
                LatentTensorSpec(48, 3, 8, 4, "torch.bfloat16"),
            ),
        )
