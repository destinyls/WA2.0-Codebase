# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch

from n0_twam.evaluation.fair_protocol import (
    CAUSAL_FUTURE_ONLY_PROTOCOL,
    causal_conditioning_sha256,
    sanitize_causal_future_only_batch,
    verify_future_gt_invariance,
)
from n0_twam.evaluation.tactile_sampling import (
    sample_tactile_causal_future_only,
)
from n0_twam.utils.scheduler import FlowMatchScheduler


def _batch() -> dict[str, torch.Tensor]:
    return {
        "latents": torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3, 1, 1),
        "actions": torch.arange(1 * 2 * 3, dtype=torch.float32).reshape(1, 2, 3, 1, 1),
        "actions_mask": torch.ones(1, 2, 3, 1, 1, dtype=torch.bool),
        "text_emb": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        "tactile_global_latent": torch.arange(
            1 * 2 * 4 * 3, dtype=torch.float32
        ).reshape(1, 2, 4, 3, 1, 1),
        "tactile_local_latent": torch.arange(
            1 * 2 * 4 * 3, dtype=torch.float32
        ).reshape(1, 2, 4, 3, 1, 1)
        + 100.0,
        "tactile_sensor_ids": torch.tensor([[0, 1]], dtype=torch.long),
    }


def _change_future_gt(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    changed = {key: value.clone() for key, value in batch.items()}
    changed["latents"][..., 1:, :, :] += 10_000.0
    changed["tactile_global_latent"][..., 1:, :, :] -= 20_000.0
    changed["tactile_local_latent"][..., 1:, :, :] += 30_000.0
    changed["actions"] += 40_000.0
    changed["actions_mask"].logical_not_()
    return changed


def test_causal_sanitizer_is_invariant_to_all_future_gt() -> None:
    base = _batch()
    changed = _change_future_gt(base)

    first = sanitize_causal_future_only_batch(base)
    second = sanitize_causal_future_only_batch(changed)

    assert set(first) == {
        "latents",
        "actions",
        "actions_mask",
        "text_emb",
        "tactile_global_latent",
        "tactile_local_latent",
        "tactile_sensor_ids",
    }
    assert torch.count_nonzero(first["actions"]) == 0
    assert torch.count_nonzero(first["actions_mask"]) == 0
    torch.testing.assert_close(first["text_emb"], base["text_emb"])
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    assert causal_conditioning_sha256(first) == causal_conditioning_sha256(second)
    contract = verify_future_gt_invariance(base, changed)
    assert contract["protocol_id"] == CAUSAL_FUTURE_ONLY_PROTOCOL
    assert contract["future_gt_invariant"] is True


def test_causal_sanitizer_hash_changes_when_frame_zero_changes() -> None:
    base = _batch()
    changed = {key: value.clone() for key, value in base.items()}
    changed["latents"][..., 0, :, :] += 1.0

    assert causal_conditioning_sha256(
        sanitize_causal_future_only_batch(base)
    ) != causal_conditioning_sha256(sanitize_causal_future_only_batch(changed))


@pytest.mark.parametrize("missing", ["actions", "actions_mask", "text_emb"])
def test_causal_sanitizer_fails_closed_on_missing_model_input(missing: str) -> None:
    batch = _batch()
    batch.pop(missing)

    with pytest.raises(ValueError, match=missing):
        sanitize_causal_future_only_batch(batch)


class _CaptureModel:
    def __init__(self) -> None:
        self.payload_hashes: list[str] = []

    def __call__(
        self, input_dict: dict[str, Any], *, train_mode: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert train_mode is True
        self.payload_hashes.append(causal_conditioning_sha256(input_dict))
        tactile = input_dict["action_dict"]["tactile_global_clean_latent"]
        batch, sensors, channels, frames, height, width = tactile.shape
        return (
            torch.empty(batch, 0, channels),
            torch.empty(batch, 0, 1),
            torch.zeros(batch, sensors * frames * height * width, channels),
        )


class _CaptureTrainer:
    def __init__(self) -> None:
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
        self._tactile_cond_drop = False

    def _prepare_input_dict(self, batch: dict[str, torch.Tensor]) -> dict[str, object]:
        frames = batch["latents"].shape[-3]
        clean_timestep = torch.zeros(1, frames)
        return {
            "latent_dict": {
                "latent": batch["latents"].clone(),
                "noisy_latents": batch["latents"].clone(),
                "timesteps": clean_timestep.clone(),
                "cond_timesteps": clean_timestep.clone(),
                "text_emb": batch["text_emb"].clone(),
            },
            "action_dict": {
                "latent": batch["actions"].clone(),
                "noisy_latents": batch["actions"].clone(),
                "timesteps": clean_timestep.clone(),
                "cond_timesteps": clean_timestep.clone(),
                "text_emb": batch["text_emb"].clone(),
                "actions_mask": batch["actions_mask"].clone(),
                "tactile_local_latent": batch["tactile_local_latent"].clone(),
                "tactile_sensor_ids": batch["tactile_sensor_ids"].clone(),
                "tactile_global_clean_latent": batch["tactile_global_latent"].clone(),
                "tactile_global_noisy_latent": batch["tactile_global_latent"].clone(),
                "tactile_global_timesteps": clean_timestep.clone(),
            },
        }


def _install_patch_inverse(monkeypatch: pytest.MonkeyPatch) -> None:
    utils = types.ModuleType("utils")

    def _data_seq_to_patch(
        patch_size: list[int],
        sequence: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        assert patch_size == [1, 1, 1]
        return sequence.reshape(batch_size, frames, height, width, -1).permute(
            0, 4, 1, 2, 3
        )

    utils.data_seq_to_patch = _data_seq_to_patch
    monkeypatch.setitem(sys.modules, "utils", utils)


def test_causal_sampler_model_payload_ignores_changed_future_gt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_patch_inverse(monkeypatch)
    base = _batch()
    changed = _change_future_gt(base)

    first_trainer = _CaptureTrainer()
    torch.manual_seed(123)
    first_generated, first_gt = sample_tactile_causal_future_only(
        first_trainer, base, n_steps=1
    )
    second_trainer = _CaptureTrainer()
    torch.manual_seed(123)
    second_generated, second_gt = sample_tactile_causal_future_only(
        second_trainer, changed, n_steps=1
    )

    assert first_trainer.transformer.payload_hashes
    assert (
        first_trainer.transformer.payload_hashes
        == second_trainer.transformer.payload_hashes
    )
    torch.testing.assert_close(first_generated, second_generated)
    assert not torch.equal(first_gt[..., 1:, :, :], second_gt[..., 1:, :, :])
