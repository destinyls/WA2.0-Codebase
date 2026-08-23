from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanAttentionBlock,
    WanCausalConv3d,
)

from n0_twam.evaluation.franka_offline_generation import (
    _current_state_row,
    _target_action_rows,
    _uint8_tiles,
)
from n0_twam.integrations.worldarena.franka_policy import DirectN0FrankaBackend
from n0_twam.models.utils import (
    _install_wan_attention_fallback,
    _install_wan_conv3d_fallback,
)
from n0_twam.n0_twam_server import TWAM_Server


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


def test_current_state_row_uses_cold_observation_not_first_future_target() -> None:
    normalized = torch.zeros((20, 2, 6, 1), dtype=torch.float32)
    normalized[:10, 0] = -0.5
    normalized[:10, 1] = 0.5
    mask = torch.zeros_like(normalized, dtype=torch.bool)
    mask[:10] = True

    current = _current_state_row(
        {"actions": normalized, "actions_mask": mask},
        q01=np.full(20, -1.0, dtype=np.float32),
        q99=np.full(20, 1.0, dtype=np.float32),
    )

    assert current.shape == (10,)
    assert np.allclose(current, -0.5, atol=2e-6)


class _FakeServer:
    observation: object | None = None

    def _infer(self, observation: object, frame_st_id: int) -> tuple[object, object]:
        assert frame_st_id == 0
        assert isinstance(observation, dict)
        self.observation = observation
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


def test_direct_backend_passes_training_latent_without_rgb_roundtrip() -> None:
    backend = object.__new__(DirectN0FrankaBackend)
    backend._started = True
    server = _FakeServer()
    backend._server = server
    latent = torch.zeros(1, 48, 1, 1, 2)

    backend.infer_prediction_latent_chunk(
        images={},
        current_ee20=np.zeros(20, dtype=np.float32),
        precomputed_video_latent=latent,
    )

    assert isinstance(server.observation, dict)
    assert server.observation["precomputed_video_latent"] is latent


def test_direct_backend_passes_complete_training_aligned_video_history() -> None:
    backend = object.__new__(DirectN0FrankaBackend)
    backend._started = True
    server = _FakeServer()
    backend._server = server
    history = tuple(
        {
            "observation.images.top": np.full((8, 8, 3), index, dtype=np.uint8),
            "observation.images.wrist_l": np.full(
                (8, 8, 3), index + 20, dtype=np.uint8
            ),
        }
        for index in range(5)
    )

    backend.infer_prediction_latent_chunk(
        images=history[-1],
        current_ee20=np.zeros(20, dtype=np.float32),
        training_aligned_video_history=history,
    )

    assert isinstance(server.observation, dict)
    passed = server.observation["training_aligned_video_history"]
    assert isinstance(passed, list)
    assert len(passed) == 5
    assert np.array_equal(
        passed[-1]["observation.images.top"],
        history[-1]["observation.images.top"],
    )


def test_direct_backend_rejects_history_that_does_not_end_at_current_frame() -> None:
    backend = object.__new__(DirectN0FrankaBackend)
    backend._started = True
    backend._server = _FakeServer()
    history = tuple(
        {"observation.images.top": np.full((8, 8, 3), index, dtype=np.uint8)}
        for index in range(5)
    )
    current = {"observation.images.top": np.full((8, 8, 3), 99, dtype=np.uint8)}

    with pytest.raises(ValueError, match="end at the current observation"):
        backend.infer_prediction_latent_chunk(
            images=current,
            current_ee20=np.zeros(20, dtype=np.float32),
            training_aligned_video_history=history,
        )


def test_server_accepts_exact_normalized_training_latent() -> None:
    server = object.__new__(TWAM_Server)
    server.latent_height = 2
    server.latent_width = 4
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    latent = torch.randn(1, 48, 1, 2, 4)

    encoded = server._encode_obs({"precomputed_video_latent": latent})

    torch.testing.assert_close(encoded, latent)


class _FakeStreamingVAE:
    def __init__(self) -> None:
        self.vae = torch.nn.Linear(1, 1, bias=False)
        self.calls: list[int] = []

    def clear_cache(self) -> None:
        return None

    def encode_chunk(self, video: torch.Tensor) -> torch.Tensor:
        self.calls.append(video.shape[2])
        frames = (video.shape[2] - 1) // 4 + 1
        values = torch.full(
            (1, 1, frames, 1, 1), len(self.calls), dtype=video.dtype
        )
        mu = values.expand(video.shape[0], 48, frames, 2, 2).clone()
        return torch.cat((mu, torch.zeros_like(mu)), dim=1)


def test_server_training_aligned_history_returns_only_current_latent() -> None:
    server = object.__new__(TWAM_Server)
    server.job_config = SimpleNamespace(
        obs_cam_keys=("observation.images.top", "observation.images.wrist_l")
    )
    server.height = 4
    server.width = 4
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.streaming_vae = _FakeStreamingVAE()
    server.vae = SimpleNamespace(
        config=SimpleNamespace(latents_mean=[0.0] * 48, latents_std=[1.0] * 48)
    )
    server.normalize_latents = lambda latent, mean, inverse_std: latent
    history = [
        {
            "observation.images.top": np.full((4, 4, 3), index, dtype=np.uint8),
            "observation.images.wrist_l": np.full(
                (4, 4, 3), index + 10, dtype=np.uint8
            ),
        }
        for index in range(5)
    ]

    encoded = server._encode_obs(
        {
            "obs": [history[-1]],
            "training_aligned_video_history": history,
        }
    )

    assert encoded.shape == (1, 48, 1, 2, 4)
    torch.testing.assert_close(encoded[:, :, :, :, :2], torch.full_like(encoded[:, :, :, :, :2], 2.0))
    torch.testing.assert_close(encoded[:, :, :, :, 2:], torch.full_like(encoded[:, :, :, :, 2:], 4.0))
    assert server.streaming_vae.calls == [1, 4, 1, 4]


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


def test_wan_attention_fallback_matches_sdpa_cpu() -> None:
    torch.manual_seed(11)
    native = WanAttentionBlock(8).float()
    fallback = copy.deepcopy(native)
    inputs = torch.randn(2, 8, 3, 4, 5)

    expected = native(inputs)
    _install_wan_attention_fallback(fallback)
    actual = fallback(inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
