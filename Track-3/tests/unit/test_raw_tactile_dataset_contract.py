# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import pytest
import torch

from n0_twam.evaluation.tactile_prediction_schema import strict_int64_vector
from n0_twam.evaluation.tactile_sampling import select_latent_crop_start
from n0_twam.evaluation.tactile_temporal_contract import (
    decoded_evaluation_frame_count,
)


def test_training_crop_default_remains_random(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    def _randint(low: int, high: int, shape: tuple[int, ...]) -> torch.Tensor:
        calls.append((low, high))
        assert shape == (1,)
        return torch.tensor([4])

    monkeypatch.setattr(torch, "randint", _randint)

    selected = select_latent_crop_start(
        num_latent_frames=12,
        max_latent_frames=5,
        deterministic_evaluation_crop_zero=False,
    )

    assert selected == 4
    assert calls == [(0, 8)]


def test_official_artifact_generation_can_gate_crop_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("crop-zero mode must not consume the training crop RNG")
        ),
    )

    selected = select_latent_crop_start(
        num_latent_frames=12,
        max_latent_frames=5,
        deterministic_evaluation_crop_zero=True,
    )

    assert selected == 0


def test_raw_tactile_metadata_rejects_fractional_row_ids() -> None:
    with pytest.raises(ValueError, match="row IDs must have an integer dtype"):
        strict_int64_vector(
            torch.arange(17, dtype=torch.float32).numpy() + 0.5,
            label="source row IDs",
            expected_length=17,
        )


def test_raw_tactile_metadata_rejects_fractional_step_ids() -> None:
    with pytest.raises(ValueError, match="steps must have an integer dtype"):
        strict_int64_vector(
            [float(index) + 0.5 for index in range(17)],
            label="converted LeRobot simulator steps",
            expected_length=17,
        )


@pytest.mark.parametrize(
    ("latent_frames", "decoded_frames"),
    [(5, 17), (11, 41)],
)
def test_raw_tactile_metadata_length_tracks_evaluation_horizon(
    latent_frames: int,
    decoded_frames: int,
) -> None:
    assert decoded_evaluation_frame_count(latent_frames) == decoded_frames


def test_raw_tactile_metadata_requires_positive_latent_horizon() -> None:
    with pytest.raises(ValueError, match="max_latent_frames"):
        decoded_evaluation_frame_count(0)
