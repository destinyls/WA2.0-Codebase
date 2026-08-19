# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Runtime-only dataset and modality helpers for AgileX evaluation."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import torch

from .agilex_generation_data import route_wrench

FloatArray = npt.NDArray[np.float32]
ImageArray = npt.NDArray[np.uint8]


def dataset_index(dataset: object) -> dict[tuple[str, int], tuple[object, int]]:
    """Index the exact converted segment for each repo/episode pair."""

    mapping: dict[tuple[str, int], tuple[object, int]] = {}
    for child in dataset._datasets:  # noqa: SLF001
        for index, meta in enumerate(child.new_metas):
            key = (child.repo_name, int(meta["episode_index"]))
            if key in mapping:
                raise ValueError("AgileX evaluation view episode has multiple segments")
            mapping[key] = (child, index)
    return mapping


def target_latent(sample: dict[str, object]) -> torch.Tensor:
    """Return the two-frame target latent including the conditioning frame."""

    latent = sample.get("latents")
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise ValueError("AgileX evaluation sample has no four-dimensional RGB latent")
    if latent.shape[1] < 2:
        raise ValueError("AgileX evaluation sample requires two latent frames")
    return latent[:, :2].unsqueeze(0).detach().cpu()


def sample_modalities(
    *,
    backend: object,
    sample: dict[str, object],
    route: object,
    target_tiles: ImageArray,
    tactile_sensor_ids: dict[str, int],
    wrench_sensor_ids: dict[str, int],
    decode_batch_size: int,
) -> tuple[
    dict[str, ImageArray],
    dict[str, ImageArray] | None,
    dict[str, FloatArray] | None,
]:
    """Decode the first observation and route contact inputs by signed IDs."""

    images = {
        name: target_tiles[index, 0]
        for index, name in enumerate(("top", "wrist_l", "wrist_r"))
    }
    tactile = None
    if route.tactile_keys:
        local = sample.get("tactile_local_latent")
        ids = sample.get("tactile_sensor_ids")
        if not isinstance(local, torch.Tensor) or not isinstance(ids, torch.Tensor):
            raise ValueError("AgileX tactile route has no local tactile latent")
        decoded = backend.decode_video_latent_batch(
            local.detach().cpu(), batch_size=decode_batch_size
        )
        decoded_by_id = {
            int(sensor_id): decoded[index, 0]
            for index, sensor_id in enumerate(ids.detach().cpu().tolist())
        }
        tactile = {}
        for key in route.tactile_keys:
            sensor_id = tactile_sensor_ids.get(key)
            if sensor_id not in decoded_by_id:
                raise ValueError("AgileX tactile sensor ID is absent from the sample")
            tactile[key] = cast(ImageArray, decoded_by_id[sensor_id])
    wrench = route_wrench(
        sample,
        keys=route.wrench_keys,
        sensor_id_map=wrench_sensor_ids,
    )
    return images, tactile, wrench


__all__ = ("dataset_index", "sample_modalities", "target_latent")
