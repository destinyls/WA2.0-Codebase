#!/usr/bin/env python3
# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Small checkpoint byte-level helpers for the Track 3.1 preflight."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from safetensors import safe_open


def audit_tactile_sensor_embedding_capacity(
    weights_path: Path,
    *,
    tactile_contract: Mapping[str, object],
) -> dict[str, list[int]]:
    """Verify config capacity against the actual global/local sensor tables."""

    required_keys = ["sensor_id_embed.weight"]
    if bool(tactile_contract["use_local_tactile"]):
        required_keys.append("local_tactile_sensor_embed.weight")
    capacity_value = tactile_contract["max_tactile_streams"]
    if (
        isinstance(capacity_value, bool)
        or not isinstance(capacity_value, int)
        or capacity_value <= 0
    ):
        raise ValueError("tactile contract has an invalid sensor capacity")
    capacity = capacity_value
    try:
        with safe_open(
            str(weights_path.resolve(strict=True)), framework="pt", device="cpu"
        ) as checkpoint:
            available_keys = set(checkpoint.keys())
            missing = sorted(set(required_keys) - available_keys)
            if missing:
                raise ValueError(
                    "checkpoint is missing tactile sensor embeddings: "
                    + ", ".join(missing)
                )
            shapes = {
                key: [int(size) for size in checkpoint.get_slice(key).get_shape()]
                for key in required_keys
            }
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"invalid transformer safetensors: {weights_path}") from error
    for key, shape in shapes.items():
        if len(shape) != 2 or shape[0] != capacity:
            raise ValueError(
                f"tactile sensor embedding capacity mismatch for {key}: "
                f"shape={shape!r} expected_first_dim={capacity}"
            )
    return shapes


__all__ = ("audit_tactile_sensor_embedding_capacity",)
