# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Training-identical raw RGB encoder for Franka offline evaluation."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F


class TrainingAlignedFrankaVideoEncoder:
    """Encode raw camera prefixes before the memory-heavy Policy is loaded."""

    def __init__(
        self,
        *,
        base_model: Path,
        device: str,
        camera_keys: Sequence[str],
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> None:
        if not device.isdecimal():
            raise ValueError("Franka video encoder device must be numeric")
        if not camera_keys or height <= 0 or width <= 0:
            raise ValueError("Franka video encoder geometry is invalid")
        from n0_twam.models.utils import load_vae

        self._device = torch.device(f"cuda:{int(device)}")
        self._camera_keys = tuple(camera_keys)
        self._height = int(height)
        self._width = int(width)
        self._dtype = dtype
        self._vae = load_vae(
            Path(base_model).resolve(strict=True) / "vae",
            torch_dtype=dtype,
            torch_device=self._device,
        ).eval()

    @torch.no_grad()
    def encode(
        self,
        history: Sequence[Mapping[str, npt.NDArray[np.uint8]]],
    ) -> torch.Tensor:
        """Return the current normalized latent using the training batch encoder."""

        if not history or len(history) % 4 != 1:
            raise ValueError("training-aligned RGB history must contain 4*k+1 frames")
        camera_latents: list[torch.Tensor] = []
        for key in self._camera_keys:
            frames: list[npt.NDArray[np.uint8]] = []
            for index, row in enumerate(history):
                if key not in row:
                    raise ValueError(f"RGB history frame {index} is missing {key!r}")
                frame = np.asarray(row[key])
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError(f"RGB history camera {key!r} must be uint8 HWC")
                frames.append(np.ascontiguousarray(frame))
            video = torch.from_numpy(np.stack(frames)).float().permute(3, 0, 1, 2)
            video = F.interpolate(
                video,
                size=(self._height, self._width),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(0)
            video = (video / 255.0 * 2.0 - 1.0).to(
                device=self._device,
                dtype=self._dtype,
            )
            posterior = self._vae.encode(video).latent_dist
            mean = posterior.mean
            latents_mean = torch.tensor(
                self._vae.config.latents_mean,
                device=mean.device,
            ).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(
                self._vae.config.latents_std,
                device=mean.device,
            ).view(1, -1, 1, 1, 1)
            normalized = ((mean.float() - latents_mean) / latents_std).to(mean)
            camera_latents.append(normalized[:, :, -1:].detach().cpu())
            del normalized, mean, posterior, video
            torch.cuda.empty_cache()
        return torch.cat(camera_latents, dim=-1).contiguous()

    def close(self) -> None:
        """Release the VAE before the Policy transformer is materialized."""

        if hasattr(self, "_vae"):
            del self._vae
        gc.collect()
        torch.cuda.empty_cache()
