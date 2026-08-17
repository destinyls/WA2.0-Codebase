# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict NPZ contract for AgileX future RGB and qpos14 predictions."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping

import numpy as np
import numpy.typing as npt

from n0_twam.evaluation.franka_atomic_io import publish_atomic_file
from n0_twam.evaluation.franka_prediction_io import (
    StablePredictionInput,
    capture_prediction_input,
    require_prediction_unchanged,
)
from n0_twam.evaluation.sealed_artifact_io import validate_sha256

PREDICTION_SCHEMA_VERSION = 1
PREDICTION_ARTIFACT_TYPE = "n0_twam_track32_agilex_offline_predictions"
AGILEX_VIEWS = ("top", "wrist_l", "wrist_r")
_PROFILES = frozenset(("vision_tactile", "mixed", "vision_only"))
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
        "tactile_profile",
        "view_names",
        "frame_offsets",
        "action_offsets",
        "sample_ids",
        "task_ids",
        "repo_ids",
        "contact_condition_present",
        "predicted_rgb",
        "target_rgb",
        "video_valid",
        "predicted_qpos14",
        "target_qpos14",
        "action_valid",
    )
)


@dataclass(frozen=True)
class AgileXPredictions:
    metadata: Mapping[str, object]
    sample_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    repo_ids: tuple[str, ...]
    contact_condition_present: npt.NDArray[np.bool_]
    view_names: tuple[str, ...]
    predicted_rgb: npt.NDArray[np.uint8]
    target_rgb: npt.NDArray[np.uint8]
    video_valid: npt.NDArray[np.bool_]
    predicted_qpos14: npt.NDArray[np.float32]
    target_qpos14: npt.NDArray[np.float32]
    action_valid: npt.NDArray[np.bool_]


def _scalar(array: npt.NDArray[np.generic], *, label: str) -> object:
    if array.shape != ():
        raise ValueError(f"{label} must be a scalar")
    return array.item()


def _string(value: object, *, label: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} must be UTF-8") from error
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _strings(array: npt.NDArray[np.generic], *, label: str) -> tuple[str, ...]:
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{label} must be a one-dimensional string array")
    return tuple(_string(value, label=label) for value in array.tolist())


def _offsets(array: npt.NDArray[np.generic], *, label: str) -> tuple[int, ...]:
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"} or len(array) == 0:
        raise ValueError(f"{label} must be a non-empty integer vector")
    values = tuple(int(value) for value in array.tolist())
    if values[0] <= 0 or any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{label} must be strictly increasing and exclude offset zero")
    return values


def _metadata(arrays: Mapping[str, npt.NDArray[np.generic]]) -> dict[str, object]:
    schema = _scalar(arrays["schema_version"], label="schema_version")
    seed = _scalar(arrays["seed"], label="seed")
    if type(schema) is not int or schema != PREDICTION_SCHEMA_VERSION:
        raise ValueError("unsupported AgileX prediction schema version")
    if type(seed) is not int or seed < 0:
        raise ValueError("AgileX prediction seed must be non-negative")
    artifact_type = _string(
        _scalar(arrays["artifact_type"], label="artifact_type"),
        label="artifact_type",
    )
    if artifact_type != PREDICTION_ARTIFACT_TYPE:
        raise ValueError("unsupported AgileX prediction artifact type")
    run_role = _string(_scalar(arrays["run_role"], label="run_role"), label="run_role")
    mode = _string(
        _scalar(arrays["prediction_mode"], label="prediction_mode"),
        label="prediction_mode",
    )
    profile = _string(
        _scalar(arrays["tactile_profile"], label="tactile_profile"),
        label="tactile_profile",
    )
    if run_role not in {"development", "final_refit"}:
        raise ValueError("run_role must be development or final_refit")
    if mode not in {"teacher_action", "policy_action"}:
        raise ValueError("prediction_mode must be teacher_action or policy_action")
    if profile not in _PROFILES:
        raise ValueError("unknown AgileX tactile profile")
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
        "prediction_mode": mode,
        "tactile_profile": profile,
    }


def load_agilex_predictions(raw: bytes) -> AgileXPredictions:
    """Load one pickle-free artifact and enforce its full tensor contract."""

    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != _EXPECTED_KEYS:
                raise ValueError("AgileX prediction artifact keys differ from schema")
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("AgileX prediction"):
            raise
        raise ValueError("invalid AgileX prediction NPZ") from error

    metadata = _metadata(arrays)
    sample_ids = _strings(arrays["sample_ids"], label="sample_ids")
    task_ids = _strings(arrays["task_ids"], label="task_ids")
    repo_ids = _strings(arrays["repo_ids"], label="repo_ids")
    views = _strings(arrays["view_names"], label="view_names")
    frame_offsets = _offsets(arrays["frame_offsets"], label="frame_offsets")
    action_offsets = _offsets(arrays["action_offsets"], label="action_offsets")
    if views != AGILEX_VIEWS:
        raise ValueError("view_names must contain canonical AgileX RGB views")
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be non-empty and unique")
    if len(task_ids) != len(sample_ids) or len(repo_ids) != len(sample_ids):
        raise ValueError("task_ids and repo_ids must identify every sample")
    contact = arrays["contact_condition_present"]
    if contact.dtype != np.bool_ or contact.shape != (len(sample_ids),):
        raise ValueError("contact_condition_present must be bool [N]")

    prediction_rgb = arrays["predicted_rgb"]
    target_rgb = arrays["target_rgb"]
    video_valid = arrays["video_valid"]
    video_prefix = (len(sample_ids), len(views), len(frame_offsets))
    if (
        prediction_rgb.dtype != np.uint8
        or target_rgb.dtype != np.uint8
        or prediction_rgb.shape != target_rgb.shape
        or prediction_rgb.ndim != 6
        or prediction_rgb.shape[:3] != video_prefix
        or prediction_rgb.shape[-1] != 3
        or min(prediction_rgb.shape[-3:-1]) < 7
    ):
        raise ValueError("RGB predictions must be matching uint8 [N,3,T,H,W,3]")
    if video_valid.dtype != np.bool_ or video_valid.shape != video_prefix:
        raise ValueError("video_valid must be bool [N,3,T]")
    if np.any(video_valid.sum(axis=2) == 0):
        raise ValueError("every sample/view requires one valid future RGB frame")

    prediction_qpos = arrays["predicted_qpos14"]
    target_qpos = arrays["target_qpos14"]
    action_valid = arrays["action_valid"]
    action_shape = (len(sample_ids), len(action_offsets), 14)
    if (
        prediction_qpos.dtype != np.float32
        or target_qpos.dtype != np.float32
        or prediction_qpos.shape != action_shape
        or target_qpos.shape != action_shape
        or not np.isfinite(prediction_qpos).all()
        or not np.isfinite(target_qpos).all()
    ):
        raise ValueError("qpos14 predictions must be finite float32 [N,A,14]")
    if action_valid.dtype != np.bool_ or action_valid.shape != action_shape[:2]:
        raise ValueError("action_valid must be bool [N,A]")
    if np.any(action_valid.sum(axis=1) == 0):
        raise ValueError("every sample requires one valid future qpos14 target")

    metadata["frame_offsets"] = list(frame_offsets)
    metadata["action_offsets"] = list(action_offsets)
    return AgileXPredictions(
        metadata=metadata,
        sample_ids=sample_ids,
        task_ids=task_ids,
        repo_ids=repo_ids,
        contact_condition_present=contact,
        view_names=views,
        predicted_rgb=prediction_rgb,
        target_rgb=target_rgb,
        video_valid=video_valid,
        predicted_qpos14=prediction_qpos,
        target_qpos14=target_qpos,
        action_valid=action_valid,
    )


def publish_agilex_predictions(
    *,
    output: Path,
    metadata: Mapping[str, object],
    arrays: Mapping[str, npt.NDArray[np.generic]],
) -> dict[str, object]:
    """Atomically publish and round-trip one AgileX prediction artifact."""

    metadata_names = {
        "checkpoint_identity_sha256",
        "dataset_view_id",
        "dataset_view_sha256",
        "decoder_sha256",
        "seed",
        "run_role",
        "prediction_mode",
        "tactile_profile",
    }
    array_names = _EXPECTED_KEYS - {"schema_version", "artifact_type", *metadata_names}
    if set(metadata) != metadata_names or set(arrays) != array_names:
        raise ValueError("AgileX prediction publisher inputs differ from schema")
    payload = {
        "schema_version": np.asarray(PREDICTION_SCHEMA_VERSION, dtype=np.int64),
        "artifact_type": np.asarray(PREDICTION_ARTIFACT_TYPE),
        **{name: np.asarray(value) for name, value in metadata.items()},
        **{name: np.asarray(value) for name, value in arrays.items()},
    }

    def writer(handle: BinaryIO) -> None:
        np.savez(handle, **payload)

    published = publish_atomic_file(
        output=output,
        writer=writer,
        validator=load_agilex_predictions,
        label="AgileX prediction artifact",
    )
    return {
        "path": str(published.path),
        "sha256": published.sha256,
        "size_bytes": published.size,
    }


def verify_agilex_prediction_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> AgileXPredictions:
    """Load a stable prediction file and optionally bind its exact file hash."""

    snapshot = capture_prediction_input(path)
    if expected_sha256 is not None and snapshot.sha256 != validate_sha256(
        expected_sha256, label="prediction file SHA256"
    ):
        raise ValueError("AgileX prediction file SHA256 mismatch")
    artifact = load_agilex_predictions(snapshot.raw)
    require_prediction_unchanged(snapshot)
    return artifact


__all__ = (
    "AGILEX_VIEWS",
    "AgileXPredictions",
    "PREDICTION_ARTIFACT_TYPE",
    "PREDICTION_SCHEMA_VERSION",
    "StablePredictionInput",
    "capture_prediction_input",
    "load_agilex_predictions",
    "publish_agilex_predictions",
    "verify_agilex_prediction_file",
    "require_prediction_unchanged",
)
