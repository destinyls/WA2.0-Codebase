# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Private joint-action/RGB prediction helpers for the AgileX backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from n0_twam.embodiments import AGILEX_RGB_KEYS

from .agilex_policy_contracts import RGB_KEYS, AgileXTaskRoute

FloatArray: TypeAlias = npt.NDArray[np.float32]
ImageArray: TypeAlias = npt.NDArray[np.uint8]


class VideoDecoder(Protocol):
    video_processor: object

    def decode_one_video(self, latents: object, output_type: str) -> object:
        pass


def prediction_observation(
    *,
    route: AgileXTaskRoute,
    images: Mapping[str, ImageArray],
    current_qpos14: FloatArray,
    tactile_images: Mapping[str, ImageArray] | None,
    wrench: Mapping[str, FloatArray] | None,
) -> dict[str, object]:
    """Validate route modalities and build the private server request."""

    from .agilex_sync_grounding import (
        rgb_images,
        tactile_images as validate_tactile_images,
        wrench_values,
    )

    qpos = np.asarray(current_qpos14)
    if qpos.dtype != np.float32 or qpos.shape != (14,) or not np.isfinite(qpos).all():
        raise ValueError("current_qpos14 must be finite float32[14]")
    tactile = validate_tactile_images(tactile_images, keys=route.tactile_keys)
    force = wrench_values(wrench, keys=route.wrench_keys)
    if route.tactile_required != (tactile is not None):
        raise ValueError("tactile presence differs from the task route")
    if route.wrench_required != (force is not None):
        raise ValueError("wrench presence differs from the task route")
    rgb = rgb_images(images)
    observation: dict[str, object] = {
        "obs": [
            {
                full: rgb[wire]
                for wire, full in zip(RGB_KEYS, AGILEX_RGB_KEYS, strict=True)
            }
        ],
        "current_state": np.ascontiguousarray(qpos),
        "action_anchor_state": np.ascontiguousarray(qpos),
        "state_action_format": "absolute",
        "compute_kv_cache": False,
        "full_replan": True,
        "tactile_keys": list(route.tactile_keys),
        "wrench_keys": list(route.wrench_keys),
    }
    if tactile is not None:
        observation["tactile"] = tactile
    if force is not None:
        observation["wrench"] = force
        observation["wrench_available_mask"] = dict.fromkeys(route.wrench_keys, True)
    return observation


def decode_video_latent_batch(
    server: VideoDecoder,
    latents: object,
    *,
    batch_size: int,
    spatial_tiles: int = 1,
) -> ImageArray:
    """Decode normalized Wan latents in bounded HCU batches and width tiles."""

    import torch
    from diffusers.video_processor import VideoProcessor

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("decode batch size must be a positive integer")
    if type(spatial_tiles) is not int or spatial_tiles <= 0:
        raise ValueError("spatial tile count must be a positive integer")
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise ValueError("video latents must be a five-dimensional tensor")
    if latents.shape[-1] % spatial_tiles != 0:
        raise ValueError("video latent width must be divisible by spatial tiles")
    server.video_processor = VideoProcessor(vae_scale_factor=1)
    tile_width = latents.shape[-1] // spatial_tiles
    with torch.no_grad():
        chunks = []
        for chunk in latents.split(batch_size):
            tiles = [
                np.asarray(
                    server.decode_one_video(
                        chunk[..., index * tile_width : (index + 1) * tile_width],
                        "np",
                    )
                )
                for index in range(spatial_tiles)
            ]
            if any(tile.ndim != 5 or tile.shape[-1] != 3 for tile in tiles):
                raise ValueError("decoded RGB tile has an invalid shape")
            chunks.append(np.concatenate(tiles, axis=3))
    decoded = np.concatenate(chunks, axis=0)
    if decoded.ndim != 5 or decoded.shape[-1] != 3:
        raise ValueError(f"decoded RGB batch has invalid shape: {decoded.shape}")
    if decoded.dtype != np.uint8:
        if not np.isfinite(decoded).all():
            raise ValueError("decoded RGB batch contains non-finite values")
        minimum = float(decoded.min(initial=0.0))
        maximum = float(decoded.max(initial=0.0))
        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise ValueError("decoded RGB batch is outside the [0,1] range")
        decoded = np.rint(np.clip(decoded, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cast(ImageArray, np.ascontiguousarray(decoded))


__all__ = ("decode_video_latent_batch", "prediction_observation")
