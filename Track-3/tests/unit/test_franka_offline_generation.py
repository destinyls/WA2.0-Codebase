from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

from n0_twam.evaluation.franka_offline_generation import (
    _target_action_rows,
    _uint8_tiles,
)
from n0_twam.integrations.worldarena.franka_policy import DirectN0FrankaBackend
from n0_twam.models.utils import _install_wan_conv3d_fallback


def test_uint8_tiles_splits_canonical_camera_order() -> None:
    video = np.zeros((5, 8, 18, 3), dtype=np.uint8)
    video[:, :, :9] = 17
    video[:, :, 9:] = 29

    tiles = _uint8_tiles(video)

    assert tiles.shape == (2, 5, 8, 9, 3)
    assert np.all(tiles[0] == 17)
    assert np.all(tiles[1] == 29)


def test_target_action_rows_excludes_cold_condition_frame() -> None:
    normalized = torch.zeros((20, 2, 6, 1), dtype=torch.float32)
    normalized[:10, 0] = -0.75
    normalized[:10, 1] = 0.5
    mask = torch.zeros_like(normalized, dtype=torch.bool)
    mask[:10, 1] = True

    actions, valid = _target_action_rows(
        {"actions": normalized, "actions_mask": mask},
        q01=np.full(20, -1.0, dtype=np.float32),
        q99=np.full(20, 1.0, dtype=np.float32),
    )

    assert actions.shape == (6, 10)
    assert np.allclose(actions, 0.5, atol=2e-6)
    assert valid.tolist() == [True] * 6


class _FakeServer:
    def _infer(self, observation: object, frame_st_id: int) -> tuple[object, object]:
        assert frame_st_id == 0
        assert isinstance(observation, dict)
        return np.zeros((20, 2, 6), dtype=np.float32), torch.zeros(1, 48, 2, 1, 2)


def test_direct_backend_evaluation_uses_same_cold_infer_kernel() -> None:
    backend = object.__new__(DirectN0FrankaBackend)
    backend._started = True
    backend._server = _FakeServer()
    decoded = np.zeros((5, 8, 16, 3), dtype=np.uint8)
    backend.decode_video_latents = lambda value: decoded  # type: ignore[method-assign]

    actions, video = backend.infer_prediction_chunk(
        images={"observation.images.top": np.zeros((8, 8, 3), dtype=np.uint8)},
        current_ee20=np.zeros(20, dtype=np.float32),
    )

    assert actions.shape == (20, 2, 6)
    assert video is decoded


def test_direct_backend_evaluation_rejects_unstarted_backend() -> None:
    backend = object.__new__(DirectN0FrankaBackend)
    backend._started = False

    with pytest.raises(RuntimeError, match="reset"):
        backend.infer_prediction_chunk(
            images={}, current_ee20=np.zeros(20, dtype=np.float32)
        )


@pytest.mark.parametrize("with_cache", [False, True])
def test_wan_conv3d_fallback_matches_native_cpu(with_cache: bool) -> None:
    torch.manual_seed(7)
    native = WanCausalConv3d(2, 3, kernel_size=3, padding=1).double()
    fallback = copy.deepcopy(native)
    inputs = torch.randn(1, 2, 2, 5, 6, dtype=torch.float64)
    cache = torch.randn(1, 2, 1, 5, 6, dtype=torch.float64) if with_cache else None

    expected = native(inputs, cache)
    _install_wan_conv3d_fallback(fallback)
    actual = fallback(inputs, cache)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
