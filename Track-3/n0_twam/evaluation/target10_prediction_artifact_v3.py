# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Sealed prediction-only artifact for the Target-10 reference9 protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from n0_twam.evaluation.fair_protocol import CAUSAL_FUTURE_ONLY_PROTOCOL
from n0_twam.evaluation.sealed_artifact_io import (
    nonnegative_integer,
    publish_sealed_npz_directory,
    read_json_object,
    sha256_file,
    validate_sha256,
    verify_sealed_file_inventory,
)
from n0_twam.evaluation.tactile_prediction_schema import json_ready
from n0_twam.evaluation.target10_reference_contract import (
    CONDITIONING_RAW_ROWS,
    REFERENCE_CONTRACT,
    TARGET_RAW_ROWS,
    TARGET_TASKS,
)

ARTIFACT_SCHEMA_VERSION = 5
ARTIFACT_TYPE = "n0_twam_target10_reference9_continuous41_prediction"
METADATA_NAME = "artifact.json"
PAYLOAD_NAME = "predictions.npz"
SEAL_NAME = "seal.json"
ACTIVE_SENSOR_IDS = (0, 1)
RESIDUAL_SHAPE = (2, 9, 128, 128, 3)
_NPZ_KEYS = frozenset(("signed_residual", "dataset_indices", "episode_indices"))

Float32Array: TypeAlias = npt.NDArray[np.float32]
Int64Array: TypeAlias = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Target10PredictionSample:
    """One prediction tied to the requested target grid, without future GT."""

    sample_id: str
    dataset_index: int
    episode_index: int
    task: str
    source_relative_path: str
    signed_residual: Float32Array


@dataclass(frozen=True)
class VerifiedTarget10PredictionArtifact:
    """Immutable arrays returned only after seal and schema validation."""

    root: Path
    metadata: dict[str, object]
    signed_residual: Float32Array
    dataset_indices: Int64Array
    episode_indices: Int64Array
    seal_sha256: str
    file_sha256: dict[str, str]


def _safe_component(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise ValueError(f"{label} must be one safe path component")
    return value


def _safe_source_path(value: object, *, task: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("source_relative_path must be a non-empty string")
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.suffix not in {".h5", ".hdf5"}
        or not path.parts
        or path.parts[0] != task
    ):
        raise ValueError("source_relative_path is outside its task roster")
    return path.as_posix()


def _validate_sample(sample: Target10PredictionSample) -> Target10PredictionSample:
    sample_id = _safe_component(sample.sample_id, label="sample_id")
    task = _safe_component(sample.task, label="task")
    if task not in TARGET_TASKS:
        raise ValueError("Target-10 prediction has an unsupported task")
    source_path = _safe_source_path(sample.source_relative_path, task=task)
    dataset_index = nonnegative_integer(sample.dataset_index, label="dataset_index")
    episode_index = nonnegative_integer(sample.episode_index, label="episode_index")
    residual = np.asarray(sample.signed_residual)
    if residual.dtype != np.float32 or residual.shape != RESIDUAL_SHAPE:
        raise ValueError(f"signed residual must be float32 {RESIDUAL_SHAPE}")
    if not np.isfinite(residual).all() or np.any(np.abs(residual) > 1.0):
        raise ValueError("signed residual must be finite and within [-1, 1]")
    return Target10PredictionSample(
        sample_id=sample_id,
        dataset_index=dataset_index,
        episode_index=episode_index,
        task=task,
        source_relative_path=source_path,
        signed_residual=np.ascontiguousarray(residual),
    )


def write_target10_prediction_artifact(
    output: Path,
    *,
    samples: Sequence[Target10PredictionSample],
    source_manifest_sha256: str,
    conversion_report_sha256: str,
    checkpoint_sha256: str,
    decoder_identity_sha256: str,
    evaluation_view_id: str,
    evaluation_view_sha256: str,
    generation_provenance: Mapping[str, object],
) -> Path:
    """Atomically publish the reference9 prediction artifact."""

    selected = tuple(_validate_sample(sample) for sample in samples)
    if len(selected) != REFERENCE_CONTRACT.expected_episode_count:
        raise ValueError("Target-10 artifact must contain exactly ten samples")
    sample_ids = [sample.sample_id for sample in selected]
    dataset_indices = [sample.dataset_index for sample in selected]
    identities = [
        (sample.source_relative_path, sample.task, sample.episode_index)
        for sample in selected
    ]
    if (
        len(set(sample_ids)) != len(selected)
        or len(set(dataset_indices)) != len(selected)
        or len(set(identities)) != len(selected)
    ):
        raise ValueError("Target-10 prediction sample identities must be unique")
    hashes = {
        "source_manifest_sha256": validate_sha256(
            source_manifest_sha256, label="source manifest SHA256"
        ),
        "conversion_report_sha256": validate_sha256(
            conversion_report_sha256, label="conversion report SHA256"
        ),
        "checkpoint_sha256": validate_sha256(
            checkpoint_sha256, label="checkpoint SHA256"
        ),
        "decoder_identity_sha256": validate_sha256(
            decoder_identity_sha256, label="decoder identity SHA256"
        ),
    }
    metadata: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "contract_id": REFERENCE_CONTRACT.contract_id,
        "contract_sha256": REFERENCE_CONTRACT.sha256,
        "pixel_domain": "signed_global_tactile_residual_float32",
        "per_sample_shape": list(RESIDUAL_SHAPE),
        "active_tactile_sensor_ids": list(ACTIVE_SENSOR_IDS),
        "conditioning_protocol_id": CAUSAL_FUTURE_ONLY_PROTOCOL,
        "conditioning_raw_rows": list(CONDITIONING_RAW_ROWS),
        "target_raw_rows": list(TARGET_RAW_ROWS),
        "temporal_binding": REFERENCE_CONTRACT.temporal_binding,
        "model_latent_frames": REFERENCE_CONTRACT.model_latent_frames,
        "model_decoded_frame_count": REFERENCE_CONTRACT.model_decoded_frame_count,
        "selected_output_indices": list(REFERENCE_CONTRACT.selected_output_indices),
        "decoded_frame_count": REFERENCE_CONTRACT.decoded_frame_count,
        "training_max_latent_frames": REFERENCE_CONTRACT.training_max_latent_frames,
        "training_decoded_frame_count": (
            REFERENCE_CONTRACT.training_decoded_frame_count
        ),
        "sequence_length_status": REFERENCE_CONTRACT.sequence_length_status,
        "future_gt_present_in_stage_a": False,
        "ground_truth_embedded": False,
        "published_score_comparable": False,
        "evaluation_view_id": evaluation_view_id,
        "evaluation_view_sha256": validate_sha256(
            evaluation_view_sha256, label="evaluation view SHA256"
        ),
        "sample_count": len(selected),
        **hashes,
        "generation_provenance": json_ready(generation_provenance),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "dataset_index": sample.dataset_index,
                "episode_index": sample.episode_index,
                "task": sample.task,
                "source_relative_path": sample.source_relative_path,
            }
            for sample in selected
        ],
    }
    return cast(
        Path,
        publish_sealed_npz_directory(
            Path(output),
            schema_version=ARTIFACT_SCHEMA_VERSION,
            artifact_type=ARTIFACT_TYPE,
            metadata_name=METADATA_NAME,
            payload_name=PAYLOAD_NAME,
            seal_name=SEAL_NAME,
            metadata=metadata,
            arrays={
                "signed_residual": np.stack(
                    [sample.signed_residual for sample in selected]
                ),
                "dataset_indices": np.asarray(dataset_indices, dtype=np.int64),
                "episode_indices": np.asarray(
                    [sample.episode_index for sample in selected], dtype=np.int64
                ),
            },
        ),
    )


def verify_target10_prediction_artifact(
    artifact: Path,
) -> VerifiedTarget10PredictionArtifact:
    """Fail closed on any prediction metadata, seal, or array drift."""

    root, seal_sha256, file_sha256 = verify_sealed_file_inventory(
        artifact,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        artifact_type=ARTIFACT_TYPE,
        metadata_name=METADATA_NAME,
        payload_name=PAYLOAD_NAME,
        seal_name=SEAL_NAME,
    )
    metadata = read_json_object(root / METADATA_NAME)
    if sha256_file(root / METADATA_NAME) != file_sha256[METADATA_NAME]:
        raise RuntimeError("Target-10 metadata changed while loading")
    required = {
        "schema_version",
        "artifact_type",
        "contract_id",
        "contract_sha256",
        "pixel_domain",
        "per_sample_shape",
        "active_tactile_sensor_ids",
        "conditioning_protocol_id",
        "conditioning_raw_rows",
        "target_raw_rows",
        "temporal_binding",
        "model_latent_frames",
        "model_decoded_frame_count",
        "selected_output_indices",
        "decoded_frame_count",
        "training_max_latent_frames",
        "training_decoded_frame_count",
        "sequence_length_status",
        "future_gt_present_in_stage_a",
        "ground_truth_embedded",
        "published_score_comparable",
        "evaluation_view_id",
        "evaluation_view_sha256",
        "sample_count",
        "source_manifest_sha256",
        "conversion_report_sha256",
        "checkpoint_sha256",
        "decoder_identity_sha256",
        "generation_provenance",
        "samples",
    }
    if set(metadata) != required:
        raise ValueError("Target-10 prediction metadata fields are invalid")
    expected_values = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "contract_id": REFERENCE_CONTRACT.contract_id,
        "contract_sha256": REFERENCE_CONTRACT.sha256,
        "pixel_domain": "signed_global_tactile_residual_float32",
        "per_sample_shape": list(RESIDUAL_SHAPE),
        "active_tactile_sensor_ids": list(ACTIVE_SENSOR_IDS),
        "conditioning_protocol_id": CAUSAL_FUTURE_ONLY_PROTOCOL,
        "conditioning_raw_rows": list(CONDITIONING_RAW_ROWS),
        "target_raw_rows": list(TARGET_RAW_ROWS),
        "temporal_binding": REFERENCE_CONTRACT.temporal_binding,
        "model_latent_frames": REFERENCE_CONTRACT.model_latent_frames,
        "model_decoded_frame_count": REFERENCE_CONTRACT.model_decoded_frame_count,
        "selected_output_indices": list(REFERENCE_CONTRACT.selected_output_indices),
        "decoded_frame_count": REFERENCE_CONTRACT.decoded_frame_count,
        "training_max_latent_frames": REFERENCE_CONTRACT.training_max_latent_frames,
        "training_decoded_frame_count": (
            REFERENCE_CONTRACT.training_decoded_frame_count
        ),
        "sequence_length_status": REFERENCE_CONTRACT.sequence_length_status,
        "future_gt_present_in_stage_a": False,
        "ground_truth_embedded": False,
        "published_score_comparable": False,
    }
    if any(metadata.get(key) != value for key, value in expected_values.items()):
        raise ValueError("Target-10 prediction contract is invalid")
    for key in (
        "evaluation_view_sha256",
        "source_manifest_sha256",
        "conversion_report_sha256",
        "checkpoint_sha256",
        "decoder_identity_sha256",
    ):
        validate_sha256(metadata.get(key), label=key)
    sample_count = nonnegative_integer(
        metadata.get("sample_count"), label="sample_count"
    )
    samples = metadata.get("samples")
    if sample_count != 10 or not isinstance(samples, list) or len(samples) != 10:
        raise ValueError("Target-10 prediction sample count is invalid")
    payload_path = root / PAYLOAD_NAME
    with np.load(payload_path, allow_pickle=False) as payload:
        if set(payload.files) != _NPZ_KEYS:
            raise ValueError("Target-10 prediction array names are invalid")
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    if sha256_file(payload_path) != file_sha256[PAYLOAD_NAME]:
        raise RuntimeError("Target-10 prediction arrays changed while loading")
    if arrays["signed_residual"].shape != (10, *RESIDUAL_SHAPE):
        raise ValueError("Target-10 residual batch shape is invalid")
    if arrays["signed_residual"].dtype != np.float32:
        raise ValueError("Target-10 residual batch must use float32")
    if not np.isfinite(arrays["signed_residual"]).all():
        raise ValueError("Target-10 residual batch contains non-finite values")
    for key in ("dataset_indices", "episode_indices"):
        if arrays[key].dtype != np.int64 or arrays[key].shape != (10,):
            raise ValueError(f"Target-10 {key} array is invalid")
    identities = []
    for index, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, Mapping) or set(raw_sample) != {
            "sample_id",
            "dataset_index",
            "episode_index",
            "task",
            "source_relative_path",
        }:
            raise ValueError("Target-10 sample metadata is invalid")
        sample = _validate_sample(
            Target10PredictionSample(
                sample_id=raw_sample["sample_id"],
                dataset_index=raw_sample["dataset_index"],
                episode_index=raw_sample["episode_index"],
                task=raw_sample["task"],
                source_relative_path=raw_sample["source_relative_path"],
                signed_residual=arrays["signed_residual"][index],
            )
        )
        if sample.dataset_index != int(arrays["dataset_indices"][index]):
            raise ValueError("Target-10 dataset indices disagree")
        if sample.episode_index != int(arrays["episode_indices"][index]):
            raise ValueError("Target-10 episode indices disagree")
        identities.append(
            (sample.sample_id, sample.dataset_index, sample.source_relative_path)
        )
    if len(set(identities)) != 10:
        raise ValueError("Target-10 sample identities are not unique")
    for array in arrays.values():
        array.setflags(write=False)
    return VerifiedTarget10PredictionArtifact(
        root=root,
        metadata=metadata,
        signed_residual=arrays["signed_residual"],
        dataset_indices=arrays["dataset_indices"],
        episode_indices=arrays["episode_indices"],
        seal_sha256=seal_sha256,
        file_sha256=file_sha256,
    )


__all__ = (
    "ARTIFACT_SCHEMA_VERSION",
    "ARTIFACT_TYPE",
    "RESIDUAL_SHAPE",
    "Target10PredictionSample",
    "VerifiedTarget10PredictionArtifact",
    "verify_target10_prediction_artifact",
    "write_target10_prediction_artifact",
)
