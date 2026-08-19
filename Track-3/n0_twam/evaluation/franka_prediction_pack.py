# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Public materializer for the Franka future-prediction NPZ contract."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from n0_twam.evaluation.franka_prediction_artifact import (
    capture_prediction_input,
    publish_franka_predictions,
    require_prediction_unchanged,
)

_METADATA_KEYS = frozenset(
    (
        "checkpoint_identity_sha256",
        "dataset_view_id",
        "dataset_view_sha256",
        "decoder_sha256",
        "seed",
        "run_role",
        "prediction_mode",
        "wire_action_schema",
        "derived_action_schema",
        "quaternion_order",
    )
)
_ARRAY_KEYS = frozenset(
    (
        "view_names",
        "frame_offsets",
        "action_offsets",
        "sample_ids",
        "task_ids",
        "lerobot_episode_ids",
        "predicted_rgb",
        "target_rgb",
        "video_valid",
        "predicted_end_pose",
        "target_end_pose",
        "action_valid",
    )
)


def pack_franka_predictions(
    *, metadata: Path, arrays: Path, output: Path
) -> dict[str, object]:
    """Pack model outputs and targets through the strict production writer."""

    metadata_snapshot = capture_prediction_input(metadata)
    arrays_snapshot = capture_prediction_input(arrays)
    try:
        raw_metadata = json.loads(metadata_snapshot.raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid prediction metadata JSON") from exc
    if not isinstance(raw_metadata, dict) or set(raw_metadata) != _METADATA_KEYS:
        raise ValueError("prediction metadata fields differ from the public schema")
    try:
        with np.load(io.BytesIO(arrays_snapshot.raw), allow_pickle=False) as archive:
            if set(archive.files) != _ARRAY_KEYS:
                raise ValueError(
                    "prediction array fields differ from the public schema"
                )
            raw_arrays = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("prediction array"):
            raise
        raise ValueError("invalid prediction arrays NPZ") from exc
    require_prediction_unchanged(metadata_snapshot)
    require_prediction_unchanged(arrays_snapshot)
    result = publish_franka_predictions(
        output=output,
        metadata=raw_metadata,
        arrays=raw_arrays,
    )
    return {
        "status": "complete",
        "artifact_type": "n0_twam_track32_offline_predictions",
        "metadata_sha256": metadata_snapshot.sha256,
        "arrays_sha256": arrays_snapshot.sha256,
        **result,
    }


__all__ = ("pack_franka_predictions",)
