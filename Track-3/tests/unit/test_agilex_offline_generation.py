# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from n0_twam.evaluation.agilex_evaluation_view import (
    load_agilex_evaluation_view,
)
from n0_twam.evaluation.agilex_offline_generation import (
    _set_environment,
)
from n0_twam.evaluation.agilex_generation_data import (
    current_qpos14,
    target_qpos14_rows,
    uint8_camera_tiles,
)
from n0_twam.integrations.worldarena.agilex_manifest import canonical_sha256
from n0_twam.integrations.worldarena.agilex_backend_prediction import (
    decode_video_latent_batch,
)


class _RecordingDecoder:
    def __init__(self) -> None:
        self.widths: list[int] = []

    def decode_one_video(self, latent: torch.Tensor, output_type: str) -> np.ndarray:
        assert output_type == "np"
        self.widths.append(int(latent.shape[-1]))
        batch, _, frames, height, width = latent.shape
        marker = latent[:, 0, 0, 0, 0].float().cpu().numpy()
        decoded = np.empty((batch, frames, height, width, 3), dtype=np.float32)
        decoded[...] = marker[:, None, None, None, None]
        return decoded


def _install_video_processor_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("diffusers.video_processor")
    module.VideoProcessor = lambda **kwargs: SimpleNamespace(**kwargs)
    package = ModuleType("diffusers")
    package.video_processor = module
    monkeypatch.setitem(sys.modules, "diffusers", package)
    monkeypatch.setitem(sys.modules, "diffusers.video_processor", module)


def test_rgb_latents_decode_each_camera_tile_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_video_processor_stub(monkeypatch)
    server = _RecordingDecoder()
    latents = torch.zeros((2, 48, 2, 8, 48), dtype=torch.float32)
    for sample in range(2):
        for camera in range(3):
            latents[sample, :, :, :, camera * 16 : (camera + 1) * 16] = (
                sample * 3 + camera
            ) / 10.0

    decoded = decode_video_latent_batch(
        server,
        latents,
        batch_size=1,
        spatial_tiles=3,
    )

    assert server.widths == [16] * 6
    assert decoded.shape == (2, 2, 8, 48, 3)
    np.testing.assert_array_equal(
        decoded[0, 0, 0, (0, 16, 32), 0],
        (0, 26, 51),
    )
    np.testing.assert_array_equal(
        decoded[1, 0, 0, (0, 16, 32), 0],
        (76, 102, 128),
    )


def test_latent_decoder_rejects_out_of_range_float_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_video_processor_stub(monkeypatch)

    class OutOfRangeDecoder:
        def decode_one_video(
            self, latent: torch.Tensor, output_type: str
        ) -> np.ndarray:
            return np.full((1, 2, 8, 16, 3), 1.25, dtype=np.float32)

    with pytest.raises(ValueError, match="outside"):
        decode_video_latent_batch(
            OutOfRangeDecoder(),
            torch.zeros((1, 48, 2, 8, 16), dtype=torch.float32),
            batch_size=1,
        )


def test_three_camera_tiles_exclude_conditioning_frame() -> None:
    video = np.zeros((5, 8, 27, 3), dtype=np.uint8)
    video[:, :, 9:18] = 1
    video[:, :, 18:] = 2

    tiles = uint8_camera_tiles(video)

    assert tiles.shape == (3, 5, 8, 9, 3)
    assert tuple(int(tiles[index, 0, 0, 0, 0]) for index in range(3)) == (0, 1, 2)


def test_qpos_targets_exclude_same_row_offset_zero() -> None:
    actions = torch.zeros((14, 2, 4, 1), dtype=torch.float32)
    for slot in range(4):
        actions[:, 0, slot, 0] = float(slot)
    valid = torch.ones((2, 4), dtype=torch.bool)
    sample = {"actions": actions, "action_valid_mask": valid}
    q01 = np.full(14, -1.0, dtype=np.float32)
    q99 = np.full(14, 1.0, dtype=np.float32)

    rows, mask, offsets = target_qpos14_rows(
        sample,
        action_offsets=(0, 1, 2, 3),
        q01=q01,
        q99=q99,
    )

    assert offsets == (1, 2, 3)
    assert rows.shape == (3, 14)
    np.testing.assert_allclose(rows[:, 0], (1.0, 2.0, 3.0), atol=2e-6)
    np.testing.assert_array_equal(mask, (True, True, True))


def test_current_qpos_accepts_scalar_tensor_global_index() -> None:
    class ExactIndexDataset:
        def __getitem__(self, index: object) -> dict[str, np.ndarray]:
            assert type(index) is int
            assert index == 3
            return {
                "observation.joint_qpos": np.arange(14, dtype=np.float32),
            }

    dataset = SimpleNamespace(
        _get_global_idx=lambda episode_id, row_id: torch.tensor(3),
        hf_dataset=ExactIndexDataset(),
    )

    result = current_qpos14(dataset, episode_id=2, row_id=5)

    np.testing.assert_array_equal(result, np.arange(14, dtype=np.float32))


def test_current_qpos_rejects_non_scalar_global_index() -> None:
    dataset = SimpleNamespace(
        _get_global_idx=lambda episode_id, row_id: torch.tensor([3]),
        hf_dataset={},
    )

    with pytest.raises(ValueError, match="scalar integer"):
        current_qpos14(dataset, episode_id=2, row_id=5)


def test_evaluation_view_is_ordered_and_self_hashed(tmp_path: Path) -> None:
    core = {
        "schema_version": 1,
        "view_id": "agilex-proxy10-v1",
        "entries": [
            {"repo_id": "agilex_rgb", "episode_id": 3, "task_id": "wipe_table"},
            {"repo_id": "agilex_touch", "episode_id": 8, "task_id": "insert"},
        ],
    }
    path = tmp_path / "view.json"
    path.write_text(
        json.dumps({**core, "view_sha256": canonical_sha256(core)}),
        encoding="utf-8",
    )

    view = load_agilex_evaluation_view(path)

    assert view.view_id == "agilex-proxy10-v1"
    assert [entry.task_id for entry in view.entries] == ["wipe_table", "insert"]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["episode_id"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self hash"):
        load_agilex_evaluation_view(path)


def test_hcu_evaluation_uses_supported_wan_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SimpleNamespace(
        runtime=SimpleNamespace(accelerator_profile="hcu_performance")
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.agilex_offline_generation."
        "build_agilex_request_environment",
        lambda request, environ: {"N0_EXISTING": "kept"},
    )
    monkeypatch.setenv("N0_STALE", "remove")

    _set_environment(request, device="0", seed=7)

    assert os.environ["N0_WAN_VAE_ATTENTION_FALLBACK"] == "1"
    assert os.environ["N0_EXISTING"] == "kept"
    assert "N0_STALE" not in os.environ
