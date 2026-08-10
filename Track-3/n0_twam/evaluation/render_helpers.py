# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Shared decoding and comparison-artifact helpers for MoT rendering."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from n0_twam.evaluation.video_writer import write_rgb_mp4_pyav

LOGGER = logging.getLogger(__name__)


def sigma_for_timesteps(scheduler: Any, timesteps_flat: torch.Tensor) -> torch.Tensor:
    """Map per-frame timesteps to the nearest scheduler sigma."""

    timesteps = timesteps_flat.detach().float().cpu()
    indices = torch.argmin(
        (scheduler.timesteps[:, None] - timesteps[None, :]).abs(), dim=0
    )
    return scheduler.sigmas[indices]


def timestep_for_sigma(scheduler: Any, sigma: float) -> float:
    """Return the nearest training timestep for a target sigma."""

    index = torch.argmin((scheduler.sigmas - float(sigma)).abs())
    return float(scheduler.timesteps[index].item())


@torch.no_grad()
def _decode_one_signed(vae: Any, latent: torch.Tensor) -> torch.Tensor:
    latent_channels = vae.config.z_dim
    mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, latent_channels, 1, 1, 1)
        .to(latent.device, vae.dtype)
    )
    std = (
        torch.tensor(vae.config.latents_std)
        .view(1, latent_channels, 1, 1, 1)
        .to(latent.device, vae.dtype)
    )
    denormalized = latent.to(vae.dtype) * std + mean
    video = vae.decode(denormalized, return_dict=False)[0]
    return video.float().clamp(-1, 1).cpu()


@torch.no_grad()
def _decode_one(vae: Any, latent: torch.Tensor) -> torch.Tensor:
    signed = _decode_one_signed(vae, latent)
    return ((signed + 1) / 2 * 255).round().byte()


@torch.no_grad()
def decode_video_latent(
    vae: Any, latent: torch.Tensor, tile_width: int = 16
) -> torch.Tensor:
    """Decode normalized WAN latents, splitting camera tiles before VAE decode."""

    width = latent.shape[-1]
    if tile_width > 0 and width > tile_width and width % tile_width == 0:
        camera_count = width // tile_width
        cameras = [
            _decode_one(
                vae,
                latent[..., index * tile_width : (index + 1) * tile_width],
            )
            for index in range(camera_count)
        ]
        return torch.cat(cameras, dim=-1)
    return _decode_one(vae, latent)


@torch.no_grad()
def decode_signed_video_latent(
    vae: Any, latent: torch.Tensor, tile_width: int = 16
) -> torch.Tensor:
    """Decode normalized WAN latents without remapping signed residuals to uint8."""

    width = latent.shape[-1]
    if tile_width > 0 and width > tile_width and width % tile_width == 0:
        camera_count = width // tile_width
        cameras = [
            _decode_one_signed(
                vae,
                latent[..., index * tile_width : (index + 1) * tile_width],
            )
            for index in range(camera_count)
        ]
        return torch.cat(cameras, dim=-1)
    return _decode_one_signed(vae, latent)


def save_compare_video(
    left: torch.Tensor,
    right: torch.Tensor,
    output_directory: str | Path,
    tag: str,
    fps: int = 4,
    left_label: str = "generated",
    right_label: str = "ground truth",
    title: str | None = None,
) -> None:
    """Save a labelled side-by-side comparison as PNG frames and MP4."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame_count = min(left.shape[2], right.shape[2])
    separator_width = 4
    banner_height = 28
    frames: list[np.ndarray] = []
    for frame_index in range(frame_count):
        left_frame = left[0, :, frame_index].permute(1, 2, 0).numpy()
        right_frame = right[0, :, frame_index].permute(1, 2, 0).numpy()
        separator = np.full(
            (left_frame.shape[0], separator_width, 3), 255, dtype=np.uint8
        )
        body = np.concatenate([left_frame, separator, right_frame], axis=1)
        total_width = body.shape[1]
        banner = np.zeros((banner_height, total_width, 3), dtype=np.uint8)
        image = Image.fromarray(np.concatenate([banner, body], axis=0))
        draw = ImageDraw.Draw(image)
        caption = title or tag
        draw.text((4, 2), caption, fill=(255, 255, 0))
        frame_label = f"frame {frame_index:02d}"
        try:
            frame_label_width = draw.textlength(frame_label)
        except AttributeError:
            frame_label_width = 8 * len(frame_label)
        draw.text(
            (total_width - frame_label_width - 4, 2),
            frame_label,
            fill=(200, 200, 200),
        )
        draw.text((4, 15), left_label, fill=(0, 255, 0))
        draw.text(
            (left_frame.shape[1] + separator_width + 4, 15),
            right_label,
            fill=(0, 200, 255),
        )
        frames.append(np.asarray(image))
        image.save(directory / f"{tag}_f{frame_index:02d}.png")
    write_rgb_mp4_pyav(directory / f"{tag}.mp4", frames, fps=fps)
    LOGGER.info("saved %s comparison frames to %s", frame_count, directory)


def save_frames(video: torch.Tensor, output_directory: str | Path, tag: str) -> None:
    """Save every uint8 RGB video frame as a PNG."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame_count = video.shape[2]
    for frame_index in range(frame_count):
        frame = video[0, :, frame_index].permute(1, 2, 0).numpy()
        Image.fromarray(frame).save(directory / f"{tag}_f{frame_index:02d}.png")
    LOGGER.info("saved %s decoded frames to %s", frame_count, directory)
