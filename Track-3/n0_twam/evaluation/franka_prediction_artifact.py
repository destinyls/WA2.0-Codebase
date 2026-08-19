# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict NPZ contract for Franka offline future predictions."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import numpy as np
import numpy.typing as npt

from n0_twam.evaluation.franka_atomic_io import publish_atomic_file
from n0_twam.evaluation.franka_prediction_io import (
    StablePredictionInput,
    capture_prediction_input,
    require_prediction_unchanged,
)
from n0_twam.evaluation.sealed_artifact_io import validate_sha256
from n0_twam.integrations.worldarena.franka_actions import (
    DERIVED_ACTION_SCHEMA,
    FRANKA_ACTION_SCHEMA,
    FRANKA_QUATERNION_ORDER,
)

PREDICTION_SCHEMA_VERSION = 2
PREDICTION_ARTIFACT_TYPE = "n0_twam_track32_offline_predictions"
OFFICIAL_TASKS = frozenset(("clear_up", "pour", "wipe"))
OFFICIAL_VIEWS = ("cam_high", "cam_left_wrist")
_EXPECTED_KEYS = frozenset(
    (
        "schema_version",
        "artifact_type",
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


@dataclass(frozen=True)
class FrankaPredictions:
    metadata: Mapping[str, object]
    sample_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    lerobot_episode_ids: tuple[int, ...]
    view_names: tuple[str, ...]
    predicted_rgb: npt.NDArray[np.uint8]
    target_rgb: npt.NDArray[np.uint8]
    video_valid: npt.NDArray[np.bool_]
    predicted_end_pose: npt.NDArray[np.floating[Any]]
    target_end_pose: npt.NDArray[np.floating[Any]]
    action_valid: npt.NDArray[np.bool_]


def _scalar(array: npt.NDArray[np.generic], *, label: str) -> object:
    if array.shape != ():
        raise ValueError(f"{label} must be a scalar")
    return array.item()


def _string(value: object, *, label: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label} must be UTF-8") from exc
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _string_vector(array: npt.NDArray[np.generic], *, label: str) -> tuple[str, ...]:
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{label} must be a one-dimensional string array")
    return tuple(_string(value, label=label) for value in array.tolist())


def _positive_offsets(array: npt.NDArray[np.generic], *, label: str) -> tuple[int, ...]:
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"} or len(array) == 0:
        raise ValueError(f"{label} must be a non-empty integer vector")
    values = tuple(int(value) for value in array.tolist())
    if values[0] <= 0 or any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{label} must be strictly increasing and exclude offset zero")
    return values


def _metadata(arrays: Mapping[str, npt.NDArray[np.generic]]) -> dict[str, object]:
    schema_version = _scalar(arrays["schema_version"], label="schema_version")
    seed = _scalar(arrays["seed"], label="seed")
    if type(schema_version) is not int or schema_version != PREDICTION_SCHEMA_VERSION:
        raise ValueError("unsupported prediction schema version")
    if type(seed) is not int or seed < 0:
        raise ValueError("prediction seed must be a non-negative integer")
    artifact_type = _string(
        _scalar(arrays["artifact_type"], label="artifact_type"),
        label="artifact_type",
    )
    if artifact_type != PREDICTION_ARTIFACT_TYPE:
        raise ValueError("unsupported prediction artifact type")
    run_role = _string(_scalar(arrays["run_role"], label="run_role"), label="run_role")
    if run_role not in {"development", "final_refit"}:
        raise ValueError("run_role must be development or final_refit")
    prediction_mode = _string(
        _scalar(arrays["prediction_mode"], label="prediction_mode"),
        label="prediction_mode",
    )
    if prediction_mode not in {"teacher_action", "policy_action"}:
        raise ValueError("prediction_mode must be teacher_action or policy_action")
    wire_action_schema = _string(
        _scalar(arrays["wire_action_schema"], label="wire_action_schema"),
        label="wire_action_schema",
    )
    derived_action_schema = _string(
        _scalar(arrays["derived_action_schema"], label="derived_action_schema"),
        label="derived_action_schema",
    )
    quaternion_order = _string(
        _scalar(arrays["quaternion_order"], label="quaternion_order"),
        label="quaternion_order",
    )
    if (
        wire_action_schema != FRANKA_ACTION_SCHEMA
        or derived_action_schema != DERIVED_ACTION_SCHEMA
        or quaternion_order != FRANKA_QUATERNION_ORDER
    ):
        raise ValueError("prediction action representation contract mismatch")
    return {
        "checkpoint_identity_sha256": validate_sha256(
            _string(
                _scalar(
                    arrays["checkpoint_identity_sha256"],
                    label="checkpoint_identity_sha256",
                ),
                label="checkpoint_identity_sha256",
            ),
            label="checkpoint identity SHA256",
        ),
        "dataset_view_id": _string(
            _scalar(arrays["dataset_view_id"], label="dataset_view_id"),
            label="dataset_view_id",
        ),
        "dataset_view_sha256": validate_sha256(
            _string(
                _scalar(arrays["dataset_view_sha256"], label="dataset_view_sha256"),
                label="dataset_view_sha256",
            ),
            label="dataset view SHA256",
        ),
        "decoder_sha256": validate_sha256(
            _string(
                _scalar(arrays["decoder_sha256"], label="decoder_sha256"),
                label="decoder_sha256",
            ),
            label="decoder SHA256",
        ),
        "seed": seed,
        "run_role": run_role,
        "prediction_mode": prediction_mode,
        "wire_action_schema": wire_action_schema,
        "derived_action_schema": derived_action_schema,
        "quaternion_order": quaternion_order,
    }


def load_franka_predictions(raw: bytes) -> FrankaPredictions:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != _EXPECTED_KEYS:
                raise ValueError(
                    "prediction artifact keys differ from schema version 2"
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("prediction artifact"):
            raise
        raise ValueError("invalid Track 3.2 prediction NPZ") from exc

    metadata = _metadata(arrays)
    sample_ids = _string_vector(arrays["sample_ids"], label="sample_ids")
    task_ids = _string_vector(arrays["task_ids"], label="task_ids")
    episode_ids_array = arrays["lerobot_episode_ids"]
    view_names = _string_vector(arrays["view_names"], label="view_names")
    frame_offsets = _positive_offsets(arrays["frame_offsets"], label="frame_offsets")
    action_offsets = _positive_offsets(arrays["action_offsets"], label="action_offsets")
    if view_names != OFFICIAL_VIEWS:
        raise ValueError("view_names must contain the canonical Franka RGB views")
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be non-empty and unique")
    if len(task_ids) != len(sample_ids) or not set(task_ids) <= OFFICIAL_TASKS:
        raise ValueError("task_ids must identify one official task per sample")
    if (
        episode_ids_array.ndim != 1
        or episode_ids_array.dtype.kind not in {"i", "u"}
        or len(episode_ids_array) != len(sample_ids)
    ):
        raise ValueError("lerobot_episode_ids must be integer [N]")
    lerobot_episode_ids = tuple(int(value) for value in episode_ids_array.tolist())
    if len(set(lerobot_episode_ids)) != len(lerobot_episode_ids) or any(
        value < 0 for value in lerobot_episode_ids
    ):
        raise ValueError("lerobot_episode_ids must be unique and non-negative")

    predicted_rgb = arrays["predicted_rgb"]
    target_rgb = arrays["target_rgb"]
    video_valid = arrays["video_valid"]
    expected_video_shape = (len(sample_ids), len(view_names), len(frame_offsets))
    if (
        predicted_rgb.dtype != np.uint8
        or target_rgb.dtype != np.uint8
        or predicted_rgb.shape != target_rgb.shape
        or predicted_rgb.ndim != 6
        or predicted_rgb.shape[:3] != expected_video_shape
        or predicted_rgb.shape[-1] != 3
        or min(predicted_rgb.shape[-3:-1]) < 7
    ):
        raise ValueError("RGB predictions must be matching uint8 [N,2,T,H,W,3]")
    if video_valid.dtype != np.bool_ or video_valid.shape != expected_video_shape:
        raise ValueError("video_valid must be bool [N,2,T]")
    if np.any(video_valid.sum(axis=2) == 0):
        raise ValueError("every sample/view must contain a valid future RGB frame")

    predicted_action = arrays["predicted_end_pose"]
    target_action = arrays["target_end_pose"]
    action_valid = arrays["action_valid"]
    expected_action_shape = (len(sample_ids), len(action_offsets), 8)
    if (
        predicted_action.dtype.kind != "f"
        or target_action.dtype.kind != "f"
        or predicted_action.shape != expected_action_shape
        or target_action.shape != expected_action_shape
        or not np.isfinite(predicted_action).all()
        or not np.isfinite(target_action).all()
    ):
        raise ValueError("end-pose predictions must be finite float [N,A,8]")
    if (
        action_valid.dtype != np.bool_
        or action_valid.shape != expected_action_shape[:2]
    ):
        raise ValueError("action_valid must be bool [N,A]")
    if np.any(action_valid.sum(axis=1) == 0):
        raise ValueError("every sample must contain at least one valid future action")
    for label, poses in (
        ("predicted", predicted_action),
        ("target", target_action),
    ):
        quaternion_norms = np.linalg.norm(poses[..., 3:7], axis=-1)
        if not np.allclose(quaternion_norms[action_valid], 1.0, atol=1e-4, rtol=1e-4):
            raise ValueError(f"{label} valid end-pose quaternions must be unit XYZW")

    metadata["frame_offsets"] = list(frame_offsets)
    metadata["action_offsets"] = list(action_offsets)
    return FrankaPredictions(
        metadata=metadata,
        sample_ids=sample_ids,
        task_ids=task_ids,
        lerobot_episode_ids=lerobot_episode_ids,
        view_names=view_names,
        predicted_rgb=predicted_rgb,
        target_rgb=target_rgb,
        video_valid=video_valid,
        predicted_end_pose=predicted_action,
        target_end_pose=target_action,
        action_valid=action_valid,
    )


def publish_franka_predictions(
    *,
    output: Path,
    metadata: Mapping[str, object],
    arrays: Mapping[str, npt.NDArray[np.generic]],
) -> dict[str, object]:
    """Atomically publish and round-trip validate one prediction NPZ."""

    required_metadata = {
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
    }
    required_arrays = _EXPECTED_KEYS - {
        "schema_version",
        "artifact_type",
        *required_metadata,
    }
    if set(metadata) != required_metadata or set(arrays) != required_arrays:
        raise ValueError("prediction publisher inputs differ from the public schema")
    payload = {
        "schema_version": np.asarray(PREDICTION_SCHEMA_VERSION, dtype=np.int64),
        "artifact_type": np.asarray(PREDICTION_ARTIFACT_TYPE),
        **{name: np.asarray(value) for name, value in metadata.items()},
        **{name: np.asarray(value) for name, value in arrays.items()},
    }

    def _write(handle: BinaryIO) -> None:
        np.savez(handle, **payload)

    def _validate(raw: bytes) -> None:
        load_franka_predictions(raw)

    published = publish_atomic_file(
        output=output,
        writer=_write,
        validator=_validate,
        label="prediction artifact",
    )
    return {
        "path": str(published.path),
        "sha256": published.sha256,
        "size_bytes": published.size,
    }


__all__ = (
    "FrankaPredictions",
    "OFFICIAL_TASKS",
    "PREDICTION_ARTIFACT_TYPE",
    "PREDICTION_SCHEMA_VERSION",
    "StablePredictionInput",
    "capture_prediction_input",
    "load_franka_predictions",
    "publish_franka_predictions",
    "require_prediction_unchanged",
)
