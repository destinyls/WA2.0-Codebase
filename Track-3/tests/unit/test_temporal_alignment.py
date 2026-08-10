# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import numpy as np
import pytest

from n0_twam.data.timeline import (
    derive_temporal_alignment,
    derive_timestamp_alignment,
)


def test_alignment_derives_twelve_actions_from_n0_anchor_grid() -> None:
    alignment = derive_temporal_alignment(
        encoded_frame_ids=np.arange(0, 25, 3),
        latent_frame_count=3,
        source_frame_count=40,
        temporal_compression=4,
    )

    assert alignment.encoded_frame_stride == 3
    assert alignment.action_per_latent_frame == 12
    np.testing.assert_array_equal(alignment.latent_anchor_ids, (0, 12, 24))
    np.testing.assert_array_equal(alignment.target_frame_ids[0], np.arange(1, 13))
    assert not alignment.is_pad.any()


def test_alignment_derives_values_instead_of_assuming_twelve() -> None:
    alignment = derive_temporal_alignment(
        encoded_frame_ids=np.arange(0, 17, 2),
        latent_frame_count=3,
        source_frame_count=30,
        temporal_compression=4,
    )

    assert alignment.encoded_frame_stride == 2
    assert alignment.action_per_latent_frame == 8
    np.testing.assert_array_equal(alignment.latent_anchor_ids, (0, 8, 16))


def test_terminal_targets_are_clipped_and_masked() -> None:
    alignment = derive_temporal_alignment(
        encoded_frame_ids=np.arange(0, 25, 3),
        latent_frame_count=3,
        source_frame_count=30,
        temporal_compression=4,
    )

    np.testing.assert_array_equal(
        alignment.target_frame_ids[-1],
        (25, 26, 27, 28, 29, 29, 29, 29, 29, 29, 29, 29),
    )
    np.testing.assert_array_equal(
        alignment.is_pad[-1],
        (False, False, False, False, False, True, True, True, True, True, True, True),
    )


def test_alignment_rejects_nonuniform_or_mismatched_grids() -> None:
    with pytest.raises(ValueError, match="uniformly spaced"):
        derive_temporal_alignment(
            encoded_frame_ids=(0, 3, 7, 10, 13),
            latent_frame_count=2,
            source_frame_count=30,
            temporal_compression=4,
        )

    with pytest.raises(ValueError, match="expected 9 encoded frame IDs"):
        derive_temporal_alignment(
            encoded_frame_ids=np.arange(0, 22, 3),
            latent_frame_count=3,
            source_frame_count=30,
            temporal_compression=4,
        )


def test_alignment_rejects_anchor_without_a_future_target() -> None:
    with pytest.raises(ValueError, match="no future action target"):
        derive_temporal_alignment(
            encoded_frame_ids=np.arange(0, 25, 3),
            latent_frame_count=3,
            source_frame_count=25,
            temporal_compression=4,
        )


def test_timestamp_alignment_handles_independent_rates_and_jitter() -> None:
    observation_times = np.arange(9, dtype=np.float64) / 29.97
    observation_times[1:] += 1e-4 * np.sin(np.arange(1, 9))
    action_times = np.arange(32, dtype=np.float64) / 59.94
    action_times[1:] += 1e-4 * np.cos(np.arange(1, 32))

    alignment = derive_timestamp_alignment(
        encoded_observation_timestamps=observation_times,
        action_timestamps=action_times,
        latent_frame_count=3,
        temporal_compression=4,
        tolerance_seconds=0.001,
    )

    assert alignment.action_per_latent_frame == 8
    np.testing.assert_array_equal(alignment.target_action_indices[0], np.arange(1, 9))
    assert not alignment.is_pad.any()
    assert not alignment.target_action_indices.flags.writeable


def test_timestamp_alignment_only_pads_beyond_terminal_sample() -> None:
    alignment = derive_timestamp_alignment(
        encoded_observation_timestamps=np.arange(9, dtype=np.float64) * 0.02,
        action_timestamps=np.arange(21, dtype=np.float64) * 0.01,
        latent_frame_count=3,
        temporal_compression=4,
        tolerance_seconds=0.001,
    )

    np.testing.assert_array_equal(
        alignment.target_action_indices[-1],
        (17, 18, 19, 20, 20, 20, 20, 20),
    )
    np.testing.assert_array_equal(
        alignment.is_pad[-1],
        (False, False, False, False, True, True, True, True),
    )


def test_timestamp_alignment_rejects_dropped_interior_action() -> None:
    observation_times = np.arange(9, dtype=np.float64) / 30.0
    action_times = np.delete(np.arange(32, dtype=np.float64) / 60.0, 4)

    with pytest.raises(ValueError, match="no action sample within tolerance"):
        derive_timestamp_alignment(
            encoded_observation_timestamps=observation_times,
            action_timestamps=action_times,
            latent_frame_count=3,
            temporal_compression=4,
            tolerance_seconds=0.002,
        )


@pytest.mark.parametrize(
    ("observation_times", "action_times", "message"),
    [
        ((0.0, 0.1, 0.2, 0.2, 0.4), (0.0, 0.1), "strictly increasing"),
        ((0.0, 0.1, 0.2, 0.3, 0.4), (0.0, np.nan), "finite"),
    ],
)
def test_timestamp_alignment_rejects_invalid_timestamps(
    observation_times: tuple[float, ...],
    action_times: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_timestamp_alignment(
            encoded_observation_timestamps=observation_times,
            action_timestamps=action_times,
            latent_frame_count=2,
            temporal_compression=4,
        )
