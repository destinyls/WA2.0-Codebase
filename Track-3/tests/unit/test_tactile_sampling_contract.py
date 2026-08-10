# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch

from n0_twam.utils.scheduler import FlowMatchScheduler


class _CaptureModel:
    def __init__(self) -> None:
        self.video_contexts: list[dict[str, torch.Tensor]] = []

    def __call__(
        self, input_dict: dict[str, object], *, train_mode: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert train_mode is True
        latent = input_dict["latent_dict"]
        action = input_dict["action_dict"]
        self.video_contexts.append(
            {
                key: latent[key].clone()
                for key in ("latent", "noisy_latents", "timesteps", "cond_timesteps")
            }
        )
        tactile = action["tactile_global_clean_latent"]
        batch, sensors, channels, frames, height, width = tactile.shape
        tactile_tokens = sensors * frames * height * width
        return (
            torch.empty(batch, 0, channels),
            torch.empty(batch, 0, 1),
            torch.zeros(batch, tactile_tokens, channels),
        )


class _CaptureTrainer:
    def __init__(self, prepared: dict[str, object]) -> None:
        scheduler = FlowMatchScheduler(
            num_inference_steps=2,
            shift=1.0,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.transformer = _CaptureModel()
        self.train_scheduler_latent = scheduler
        self.train_scheduler_tactile = scheduler
        self.patch_size = [1, 1, 1]
        self._prepared = prepared
        self._tactile_cond_drop = False
        self.prepared_local: torch.Tensor | None = None
        self.prepared_global: torch.Tensor | None = None

    def _prepare_input_dict(self, batch: dict[str, object]) -> dict[str, object]:
        self.prepared_local = batch["tactile_local_latent"].clone()
        self.prepared_global = batch["tactile_global_latent"].clone()
        return self._prepared


def _load_sample_tactile_ar(
    monkeypatch: pytest.MonkeyPatch,
):
    configs = types.ModuleType("configs")
    configs.TWAM_CONFIGS = {}
    dataset = types.ModuleType("dataset")
    dataset.MultiLatentLeRobotDataset = object
    models = types.ModuleType("models")
    models.__path__ = []
    model_utils = types.ModuleType("models.utils")
    model_utils.load_mot_checkpoint = lambda *args, **kwargs: None
    train = types.ModuleType("train")
    train.Trainer = object
    utils = types.ModuleType("utils")

    def _data_seq_to_patch(
        patch_size,
        sequence: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        assert list(patch_size) == [1, 1, 1]
        return sequence.reshape(batch_size, frames, height, width, -1).permute(
            0, 4, 1, 2, 3
        )

    utils.data_seq_to_patch = _data_seq_to_patch
    for name, module in {
        "configs": configs,
        "dataset": dataset,
        "models": models,
        "models.utils": model_utils,
        "train": train,
        "utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("n0_twam.render_mot", None)
    return importlib.import_module("n0_twam.render_mot").sample_tactile_ar


def test_sample_tactile_ar_restores_raw_video_and_masks_future_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_video = torch.arange(96, dtype=torch.float32).reshape(1, 48, 2, 1, 1)
    raw_global = torch.zeros(1, 1, 48, 2, 1, 1)
    raw_local = torch.arange(96, dtype=torch.float32).reshape(1, 1, 48, 2, 1, 1)
    prepared_video = torch.full_like(raw_video, 99.0)
    prepared = {
        "latent_dict": {
            "latent": prepared_video.clone(),
            "noisy_latents": prepared_video.clone(),
            "timesteps": torch.ones(1, 2),
            "cond_timesteps": torch.ones(1, 2),
        },
        "action_dict": {
            "latent": torch.zeros(1, 1, 2, 1, 1),
            "noisy_latents": torch.ones(1, 1, 2, 1, 1),
            "timesteps": torch.ones(1, 2),
            "cond_timesteps": torch.ones(1, 2),
            "tactile_global_clean_latent": raw_global.clone(),
            "tactile_global_noisy_latent": raw_global.clone(),
            "tactile_global_timesteps": torch.ones(1, 2),
        },
    }
    trainer = _CaptureTrainer(prepared)
    batch = {
        "latents": raw_video,
        "tactile_global_latent": raw_global,
        "tactile_local_latent": raw_local,
        "tactile_sensor_ids": torch.tensor([[0]]),
    }

    sample_tactile_ar = _load_sample_tactile_ar(monkeypatch)
    sample_tactile_ar(trainer, batch, n_steps=1)

    assert trainer.prepared_local is not None
    assert trainer.prepared_global is not None
    expected_local = raw_local[..., 0:1, :, :].expand_as(raw_local)
    expected_global = raw_global[..., 0:1, :, :].expand_as(raw_global)
    torch.testing.assert_close(trainer.prepared_local, expected_local)
    torch.testing.assert_close(trainer.prepared_global, expected_global)
    assert trainer.transformer.video_contexts
    expected_t0 = trainer.train_scheduler_latent.timesteps[
        torch.argmin(trainer.train_scheduler_latent.sigmas.abs())
    ]
    for context in trainer.transformer.video_contexts:
        torch.testing.assert_close(context["latent"], raw_video)
        torch.testing.assert_close(context["noisy_latents"], raw_video)
        torch.testing.assert_close(
            context["timesteps"], torch.full_like(context["timesteps"], expected_t0)
        )
        torch.testing.assert_close(
            context["cond_timesteps"],
            torch.full_like(context["cond_timesteps"], expected_t0),
        )


def test_render_entrypoint_reuses_the_extracted_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_tactile_ar = _load_sample_tactile_ar(monkeypatch)
    from n0_twam.evaluation.tactile_sampling import sample_tactile_ar as extracted

    assert sample_tactile_ar is extracted


def test_global_and_local_future_changes_are_hidden_before_preparation() -> None:
    from n0_twam.evaluation.tactile_sampling import (
        isolate_tactile_conditioning_frame,
    )

    base_global = torch.arange(1 * 2 * 48 * 3).reshape(1, 2, 48, 3, 1, 1)
    base_local = base_global + 100
    changed_global = base_global.clone()
    changed_local = base_local.clone()
    changed_global[..., 1:, :, :] += 10_000
    changed_local[..., 1:, :, :] -= 10_000
    common = {"tactile_sensor_ids": torch.tensor([[0, 1]])}

    first = isolate_tactile_conditioning_frame(
        {
            **common,
            "tactile_global_latent": base_global,
            "tactile_local_latent": base_local,
        }
    )
    second = isolate_tactile_conditioning_frame(
        {
            **common,
            "tactile_global_latent": changed_global,
            "tactile_local_latent": changed_local,
        }
    )

    torch.testing.assert_close(
        first["tactile_global_latent"], second["tactile_global_latent"]
    )
    torch.testing.assert_close(
        first["tactile_local_latent"], second["tactile_local_latent"]
    )
