# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Leakage-safe autoregressive GlobalTactile sampling shared by evaluators."""

from __future__ import annotations

from typing import Any, Mapping, cast

import torch

from n0_twam.evaluation.fair_protocol import (
    CAUSAL_FUTURE_ONLY_PROTOCOL,
    sanitize_causal_future_only_batch,
)
from n0_twam.evaluation.render_helpers import timestep_for_sigma
from n0_twam.evaluation.tactile_provenance import restore_raw_video_context


def select_latent_crop_start(
    *,
    num_latent_frames: int,
    max_latent_frames: int,
    deterministic_evaluation_crop_zero: bool,
) -> int:
    """Select the existing random training crop or the gated Stage-A crop0."""

    if num_latent_frames <= max_latent_frames:
        raise ValueError("latent crop selection requires a truncated sequence")
    if max_latent_frames <= 0:
        raise ValueError("max_latent_frames must be positive")
    if deterministic_evaluation_crop_zero:
        return 0
    return int(torch.randint(0, num_latent_frames - max_latent_frames + 1, (1,)).item())


def isolate_tactile_conditioning_frame(
    batch: Mapping[str, object],
) -> dict[str, object]:
    """Hide every future GlobalTactile and LocalTactile value before preparation."""

    required = (
        "tactile_global_latent",
        "tactile_local_latent",
        "tactile_sensor_ids",
    )
    missing = [key for key in required if not isinstance(batch.get(key), torch.Tensor)]
    if missing:
        raise ValueError("offline tactile sample is missing: " + ", ".join(missing))
    global_tactile = cast(torch.Tensor, batch["tactile_global_latent"])
    local_tactile = cast(torch.Tensor, batch["tactile_local_latent"])
    sensor_ids = cast(torch.Tensor, batch["tactile_sensor_ids"])
    if global_tactile.ndim not in (5, 6) or local_tactile.shape != global_tactile.shape:
        raise ValueError("global/local tactile tensors must share a 5D or 6D layout")
    if global_tactile.shape[-3] <= 1:
        raise ValueError("offline tactile generation requires a future tactile frame")
    expected_sensors = (
        global_tactile.shape[0] * global_tactile.shape[1]
        if global_tactile.ndim == 6
        else global_tactile.shape[0]
    )
    if sensor_ids.numel() != expected_sensors:
        raise ValueError("tactile sensor IDs do not match tactile streams")
    conditioned = dict(batch)
    for key, value in (
        ("tactile_global_latent", global_tactile),
        ("tactile_local_latent", local_tactile),
    ):
        isolated = value.clone()
        isolated[..., 1:, :, :] = isolated[..., 0:1, :, :]
        conditioned[key] = isolated
    return conditioned


def _tactile_velocity_dense(
    trainer: Any,
    tactile_prediction: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Invert the training tactile patch sequence to ``[B,S,C,F,H,W]``."""

    from utils import data_seq_to_patch

    batch, sensors, channels, frames, height, width = reference.shape
    return (
        data_seq_to_patch(
            trainer.patch_size,
            tactile_prediction.reshape(
                batch * sensors, frames * height * width, channels
            ),
            frames,
            height,
            width,
            batch_size=batch * sensors,
        )
        .reshape(batch, sensors, channels, frames, height, width)
        .float()
    )


@torch.no_grad()
def sample_tactile_ar(
    trainer: Any,
    batch: Mapping[str, object],
    n_steps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate GlobalTactile while exposing only tactile frame 0 to the model."""

    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    raw_global = batch.get("tactile_global_latent")
    if not isinstance(raw_global, torch.Tensor):
        raise ValueError("offline tactile sample is missing tactile_global_latent")
    conditioned_batch = isolate_tactile_conditioning_frame(batch)

    model = trainer.transformer
    tactile_scheduler = trainer.train_scheduler_tactile
    trainer._tactile_cond_drop = False
    input_dict = trainer._prepare_input_dict(conditioned_batch)
    input_dict["chunk_size"] = 1
    input_dict["window_size"] = 256
    latent_dict = input_dict["latent_dict"]
    action_dict = input_dict["action_dict"]
    device = latent_dict["latent"].device
    timestep_dtype = latent_dict["timesteps"].dtype

    clean_timestep = timestep_for_sigma(trainer.train_scheduler_latent, 0.0)
    maximum_timestep = timestep_for_sigma(
        trainer.train_scheduler_latent,
        trainer.train_scheduler_latent.sigmas[0].item(),
    )
    restore_raw_video_context(
        conditioned_batch,
        input_dict,
        clean_timestep=clean_timestep,
    )
    action_dict["noisy_latents"] = action_dict["latent"].clone()
    action_dict["timesteps"] = torch.full_like(action_dict["timesteps"], clean_timestep)
    action_dict["cond_timesteps"] = torch.full_like(
        action_dict["cond_timesteps"], clean_timestep
    )

    if "tactile_global_clean_latent" not in action_dict:
        isolated_global = cast(
            torch.Tensor, conditioned_batch["tactile_global_latent"]
        ).to(device, latent_dict["latent"].dtype)
        if isolated_global.dim() == 5:
            isolated_global = isolated_global.unsqueeze(0)
        action_dict["tactile_global_clean_latent"] = isolated_global.clone()
        action_dict["tactile_global_noisy_latent"] = isolated_global.clone()
        action_dict["tactile_sensor_ids"] = conditioned_batch.get(
            "tactile_sensor_ids",
            torch.zeros(
                isolated_global.shape[0],
                isolated_global.shape[1],
                dtype=torch.long,
                device=device,
            ),
        )
        action_dict["tactile_cond_drop"] = torch.tensor(False, device=device)
        action_dict["tactile_global_timesteps"] = torch.full(
            (isolated_global.shape[0], isolated_global.shape[3]),
            clean_timestep,
            device=device,
            dtype=timestep_dtype,
        )

    ground_truth = raw_global.to(device, latent_dict["latent"].dtype).clone()
    if ground_truth.dim() == 5:
        ground_truth = ground_truth.unsqueeze(0)
    batch_size, sensors, channels, frames, height, width = ground_truth.shape
    sigma_indices = (
        torch.linspace(0, len(tactile_scheduler.sigmas) - 1, n_steps + 1).round().long()
    )
    sigmas = tactile_scheduler.sigmas[sigma_indices].tolist()

    generated = ground_truth[..., 0:1, :, :].expand_as(ground_truth).clone()
    base_timesteps = torch.full(
        (batch_size, frames),
        clean_timestep,
        device=device,
        dtype=action_dict["tactile_global_timesteps"].dtype,
    )
    action_dict["tactile_global_cond_timesteps"] = base_timesteps.clone()
    action_dict.pop("tactile_global_targets", None)

    for frame_index in range(1, frames):
        current = torch.randn(
            batch_size,
            sensors,
            channels,
            1,
            height,
            width,
            device=device,
            dtype=ground_truth.dtype,
        )
        for step_index in range(n_steps):
            sigma, next_sigma = sigmas[step_index], sigmas[step_index + 1]
            noisy = generated.clone()
            noisy[..., frame_index : frame_index + 1, :, :] = current
            if frame_index + 1 < frames:
                noisy[..., frame_index + 1 :, :, :] = torch.randn_like(
                    noisy[..., frame_index + 1 :, :, :]
                )
            timesteps = base_timesteps.clone()
            timesteps[:, frame_index] = timestep_for_sigma(tactile_scheduler, sigma)
            if frame_index + 1 < frames:
                timesteps[:, frame_index + 1 :] = maximum_timestep
            action_dict["tactile_global_noisy_latent"] = noisy
            action_dict["tactile_global_clean_latent"] = generated
            action_dict["tactile_global_timesteps"] = timesteps
            prediction = model(input_dict, train_mode=True)
            velocity = _tactile_velocity_dense(trainer, prediction[2], ground_truth)[
                ..., frame_index : frame_index + 1, :, :
            ]
            current = current + velocity * (next_sigma - sigma)
        generated[..., frame_index : frame_index + 1, :, :] = current
    return generated, ground_truth


@torch.no_grad()
def sample_tactile_causal_future_only(
    trainer: Any,
    batch: Mapping[str, object],
    n_steps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict future GlobalTactile under ``causal_future_only_v1``.

    This is intentionally separate from :func:`sample_tactile_ar`, whose
    historical/oracle behavior retains clean future video and action context.
    Here sanitization happens before ``_prepare_input_dict``: future video and
    tactile observations are replaced by frame 0, and every action value/mask
    is zeroed. The untouched future GlobalTactile is retained only as the
    returned metric target and never inserted into the model payload.
    """

    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    raw_global = batch.get("tactile_global_latent")
    if not isinstance(raw_global, torch.Tensor):
        raise ValueError("offline tactile sample is missing tactile_global_latent")
    causal_batch = sanitize_causal_future_only_batch(batch)

    model = trainer.transformer
    tactile_scheduler = trainer.train_scheduler_tactile
    trainer._tactile_cond_drop = False
    input_dict = trainer._prepare_input_dict(causal_batch)
    if not isinstance(input_dict, dict):
        raise ValueError("_prepare_input_dict must return a dictionary")
    input_dict["chunk_size"] = 1
    input_dict["window_size"] = 256
    latent_dict = input_dict.get("latent_dict")
    action_dict = input_dict.get("action_dict")
    if not isinstance(latent_dict, dict) or not isinstance(action_dict, dict):
        raise ValueError("prepared input is missing latent_dict/action_dict")
    prepared_video = latent_dict.get("latent")
    latent_timesteps = latent_dict.get("timesteps")
    if not isinstance(prepared_video, torch.Tensor) or not isinstance(
        latent_timesteps, torch.Tensor
    ):
        raise ValueError("prepared video latent/timesteps are required")
    device = prepared_video.device
    timestep_dtype = latent_timesteps.dtype

    clean_timestep = timestep_for_sigma(trainer.train_scheduler_latent, 0.0)
    maximum_timestep = timestep_for_sigma(
        trainer.train_scheduler_latent,
        trainer.train_scheduler_latent.sigmas[0].item(),
    )
    # This restores the *sanitized* frame0-repeated video, never raw future video.
    restore_raw_video_context(
        causal_batch,
        input_dict,
        clean_timestep=clean_timestep,
    )

    action_latent = action_dict.get("latent")
    if not isinstance(action_latent, torch.Tensor):
        raise ValueError("prepared action_dict is missing latent")
    zero_action = torch.zeros_like(action_latent)
    action_dict["latent"] = zero_action
    action_dict["noisy_latents"] = zero_action.clone()
    if isinstance(action_dict.get("targets"), torch.Tensor):
        action_dict["targets"] = torch.zeros_like(action_dict["targets"])
    for key in ("timesteps", "cond_timesteps"):
        value = action_dict.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"prepared action_dict is missing {key}")
        action_dict[key] = torch.full_like(value, clean_timestep)
    prepared_action_mask = action_dict.get("actions_mask")
    if isinstance(prepared_action_mask, torch.Tensor):
        action_dict["actions_mask"] = torch.zeros_like(prepared_action_mask)
    else:
        action_dict["actions_mask"] = causal_batch["actions_mask"].to(device)

    isolated_global = causal_batch["tactile_global_latent"].to(
        device=device, dtype=prepared_video.dtype
    )
    if isolated_global.dim() == 5:
        isolated_global = isolated_global.unsqueeze(0)
    action_dict["tactile_global_clean_latent"] = isolated_global.clone()
    action_dict["tactile_global_noisy_latent"] = isolated_global.clone()
    action_dict["tactile_sensor_ids"] = causal_batch["tactile_sensor_ids"].to(device)
    action_dict["tactile_cond_drop"] = torch.tensor(False, device=device)

    ground_truth = raw_global.to(device=device, dtype=prepared_video.dtype).clone()
    if ground_truth.dim() == 5:
        ground_truth = ground_truth.unsqueeze(0)
    if ground_truth.shape != isolated_global.shape:
        raise ValueError("raw and sanitized GlobalTactile shapes differ")
    batch_size, sensors, channels, frames, height, width = isolated_global.shape
    action_dict["tactile_global_timesteps"] = torch.full(
        (batch_size, frames),
        clean_timestep,
        device=device,
        dtype=timestep_dtype,
    )

    sigma_indices = (
        torch.linspace(0, len(tactile_scheduler.sigmas) - 1, n_steps + 1).round().long()
    )
    sigmas = tactile_scheduler.sigmas[sigma_indices].tolist()
    generated = isolated_global.clone()
    base_timesteps = torch.full(
        (batch_size, frames),
        clean_timestep,
        device=device,
        dtype=timestep_dtype,
    )
    action_dict["tactile_global_cond_timesteps"] = base_timesteps.clone()
    action_dict.pop("tactile_global_targets", None)

    for frame_index in range(1, frames):
        current = torch.randn(
            batch_size,
            sensors,
            channels,
            1,
            height,
            width,
            device=device,
            dtype=isolated_global.dtype,
        )
        for step_index in range(n_steps):
            sigma, next_sigma = sigmas[step_index], sigmas[step_index + 1]
            noisy = generated.clone()
            noisy[..., frame_index : frame_index + 1, :, :] = current
            if frame_index + 1 < frames:
                noisy[..., frame_index + 1 :, :, :] = torch.randn_like(
                    noisy[..., frame_index + 1 :, :, :]
                )
            timesteps = base_timesteps.clone()
            timesteps[:, frame_index] = timestep_for_sigma(tactile_scheduler, sigma)
            if frame_index + 1 < frames:
                timesteps[:, frame_index + 1 :] = maximum_timestep
            action_dict["tactile_global_noisy_latent"] = noisy
            action_dict["tactile_global_clean_latent"] = generated
            action_dict["tactile_global_timesteps"] = timesteps
            prediction = model(input_dict, train_mode=True)
            velocity = _tactile_velocity_dense(trainer, prediction[2], isolated_global)[
                ..., frame_index : frame_index + 1, :, :
            ]
            current = current + velocity * (next_sigma - sigma)
        generated[..., frame_index : frame_index + 1, :, :] = current
    return generated, ground_truth


__all__ = [
    "CAUSAL_FUTURE_ONLY_PROTOCOL",
    "isolate_tactile_conditioning_frame",
    "sample_tactile_ar",
    "sample_tactile_causal_future_only",
    "select_latent_crop_start",
]
