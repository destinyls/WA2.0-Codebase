# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Atomic, sealed Stage-A tactile prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from n0_twam.evaluation.fair_protocol import CAUSAL_FUTURE_ONLY_PROTOCOL
from n0_twam.evaluation.sealed_artifact_io import (
    nonnegative_integer,
    publish_sealed_npz_directory,
    read_json_object,
    sha256_file,
    validate_sha256,
    verify_sealed_file_inventory,
)
from n0_twam.evaluation.tactile_prediction_schema import (
    ACTIVE_TACTILE_SENSOR_IDS,
    MODEL_TACTILE_SENSOR_CAPACITY,
    PER_SAMPLE_RESIDUAL_SHAPE,
    TACTILE_FRAME_COUNT,
    TACTILE_HEIGHT,
    TACTILE_WIDTH,
    Float32Array,
    Int64Array,
    TactilePredictionSample,
)
from n0_twam.evaluation.tactile_prediction_schema import json_ready as _json_ready
from n0_twam.evaluation.tactile_prediction_schema import (
    validate_sample as _validate_sample,
)

__all__ = (
    "ACTIVE_TACTILE_SENSOR_IDS",
    "ARTIFACT_SCHEMA_VERSION",
    "ARTIFACT_TYPE",
    "METADATA_NAME",
    "MODEL_TACTILE_SENSOR_CAPACITY",
    "PREDICTION_PAYLOAD_NAME",
    "SEAL_NAME",
    "TACTILE_FRAME_COUNT",
    "TACTILE_HEIGHT",
    "TACTILE_WIDTH",
    "TactilePredictionSample",
    "VerifiedTactilePredictionArtifact",
    "verify_tactile_prediction_artifact",
    "write_tactile_prediction_artifact",
)

ARTIFACT_SCHEMA_VERSION = 2
ARTIFACT_TYPE = "n0_twam_raw_tactile_prediction"
PREDICTION_PAYLOAD_NAME = "predictions.npz"
METADATA_NAME = "artifact.json"
SEAL_NAME = "seal.json"
_NPZ_KEYS = frozenset(
    (
        "signed_residual",
        "source_row_ids",
        "source_step_ids",
        "dataset_indices",
        "lerobot_episode_indices",
        "active_sensor_ids",
    )
)


@dataclass(frozen=True)
class VerifiedTactilePredictionArtifact:
    """Read-only payload returned only after every seal check succeeds."""

    root: Path
    metadata: dict[str, object]
    signed_residual: Float32Array
    source_row_ids: Int64Array
    source_step_ids: Int64Array
    dataset_indices: Int64Array
    lerobot_episode_indices: Int64Array
    active_sensor_ids: Int64Array
    seal_sha256: str
    file_sha256: dict[str, str]


def write_tactile_prediction_artifact(
    output: Path,
    *,
    samples: Sequence[TactilePredictionSample],
    source_manifest_sha256: str,
    conversion_report_sha256: str,
    checkpoint_sha256: str,
    decoder_identity_sha256: str,
    conditioning_protocol_id: str,
    evaluation_view_id: str,
    evaluation_view_sha256: str,
    generation_provenance: Mapping[str, object],
) -> Path:
    """Publish a prediction-only directory with one atomic directory rename."""

    target = Path(output)
    if target.exists():
        raise FileExistsError(f"prediction artifact output already exists: {target}")
    selected = tuple(_validate_sample(sample) for sample in samples)
    if not selected:
        raise ValueError("prediction artifact requires at least one sample")
    sample_ids = [sample.sample_id for sample in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("prediction artifact sample IDs must be unique")
    dataset_indices = [sample.dataset_index for sample in selected]
    if len(dataset_indices) != len(set(dataset_indices)):
        raise ValueError("prediction artifact dataset indices must be unique")
    if conditioning_protocol_id != CAUSAL_FUTURE_ONLY_PROTOCOL:
        raise ValueError(
            "prediction artifact supports only the causal future-only protocol"
        )
    if not isinstance(evaluation_view_id, str) or not evaluation_view_id:
        raise ValueError("prediction artifact requires an evaluation view ID")

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
        "pixel_domain": "signed_global_tactile_residual_float32",
        "per_sensor_shape": [17, 128, 128, 3],
        "model_tactile_sensor_capacity": MODEL_TACTILE_SENSOR_CAPACITY,
        "active_tactile_sensor_ids": list(ACTIVE_TACTILE_SENSOR_IDS),
        "sample_count": len(selected),
        "source_row_id_semantics": "zero_based_raw_hdf5_array_index",
        "source_step_id_semantics": "raw_simulator_step_metadata_not_array_index",
        "leaderboard_compatible": False,
        "published_score_comparable": False,
        "ground_truth_embedded": False,
        "conditioning_protocol_id": conditioning_protocol_id,
        "evaluation_view_id": evaluation_view_id,
        "evaluation_view_sha256": validate_sha256(
            evaluation_view_sha256, label="evaluation view SHA256"
        ),
        **hashes,
        "generation_provenance": _json_ready(generation_provenance),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "dataset_index": sample.dataset_index,
                "lerobot_episode_index": sample.lerobot_episode_index,
                "task": sample.task,
                "source_relative_path": sample.source_relative_path,
            }
            for sample in selected
        ],
    }
    return publish_sealed_npz_directory(
        target,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        artifact_type=ARTIFACT_TYPE,
        metadata_name=METADATA_NAME,
        payload_name=PREDICTION_PAYLOAD_NAME,
        seal_name=SEAL_NAME,
        metadata=metadata,
        arrays={
            "signed_residual": np.stack(
                [sample.signed_residual for sample in selected]
            ),
            "source_row_ids": np.stack([sample.source_row_ids for sample in selected]),
            "source_step_ids": np.stack(
                [sample.source_step_ids for sample in selected]
            ),
            "dataset_indices": np.asarray(dataset_indices, dtype=np.int64),
            "lerobot_episode_indices": np.asarray(
                [sample.lerobot_episode_index for sample in selected], dtype=np.int64
            ),
            "active_sensor_ids": np.asarray(ACTIVE_TACTILE_SENSOR_IDS, dtype=np.int64),
        },
    )


def verify_tactile_prediction_artifact(
    artifact: Path,
) -> VerifiedTactilePredictionArtifact:
    """Verify seal, bytes, metadata, and typed arrays without partial acceptance."""

    root, seal_sha256, file_sha256 = verify_sealed_file_inventory(
        artifact,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        artifact_type=ARTIFACT_TYPE,
        metadata_name=METADATA_NAME,
        payload_name=PREDICTION_PAYLOAD_NAME,
        seal_name=SEAL_NAME,
    )
    metadata = read_json_object(root / METADATA_NAME)
    if sha256_file(root / METADATA_NAME) != file_sha256[METADATA_NAME]:
        raise RuntimeError("prediction artifact changed while metadata was loading")
    required_metadata = {
        "schema_version",
        "artifact_type",
        "pixel_domain",
        "per_sensor_shape",
        "model_tactile_sensor_capacity",
        "active_tactile_sensor_ids",
        "sample_count",
        "source_row_id_semantics",
        "source_step_id_semantics",
        "leaderboard_compatible",
        "published_score_comparable",
        "ground_truth_embedded",
        "conditioning_protocol_id",
        "evaluation_view_id",
        "evaluation_view_sha256",
        "source_manifest_sha256",
        "conversion_report_sha256",
        "checkpoint_sha256",
        "decoder_identity_sha256",
        "generation_provenance",
        "samples",
    }
    if set(metadata) != required_metadata:
        raise ValueError("prediction artifact metadata fields are invalid")
    if (
        metadata["schema_version"] != ARTIFACT_SCHEMA_VERSION
        or metadata["artifact_type"] != ARTIFACT_TYPE
        or metadata["pixel_domain"] != "signed_global_tactile_residual_float32"
        or metadata["per_sensor_shape"] != [17, 128, 128, 3]
        or metadata["model_tactile_sensor_capacity"] != 4
        or metadata["active_tactile_sensor_ids"] != [0, 1]
        or metadata["source_row_id_semantics"] != "zero_based_raw_hdf5_array_index"
        or metadata["source_step_id_semantics"]
        != "raw_simulator_step_metadata_not_array_index"
        or metadata["leaderboard_compatible"] is not False
        or metadata["published_score_comparable"] is not False
        or metadata["ground_truth_embedded"] is not False
        or metadata["conditioning_protocol_id"] != CAUSAL_FUTURE_ONLY_PROTOCOL
        or not isinstance(metadata["evaluation_view_id"], str)
        or not metadata["evaluation_view_id"]
        or not isinstance(metadata["generation_provenance"], dict)
    ):
        raise ValueError("prediction artifact metadata contract is invalid")
    for label in (
        "source_manifest_sha256",
        "conversion_report_sha256",
        "checkpoint_sha256",
        "decoder_identity_sha256",
        "evaluation_view_sha256",
    ):
        validate_sha256(metadata[label], label=label)
    sample_count = nonnegative_integer(metadata["sample_count"], label="sample_count")
    raw_samples = metadata["samples"]
    if (
        sample_count <= 0
        or not isinstance(raw_samples, list)
        or len(raw_samples) != sample_count
    ):
        raise ValueError("prediction artifact sample metadata count is invalid")

    payload_path = root / PREDICTION_PAYLOAD_NAME
    with np.load(payload_path, allow_pickle=False) as payload:
        if set(payload.files) != _NPZ_KEYS:
            raise ValueError("prediction artifact array names are invalid")
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    if sha256_file(payload_path) != file_sha256[PREDICTION_PAYLOAD_NAME]:
        raise RuntimeError("prediction artifact changed while arrays were loading")
    if arrays["signed_residual"].shape != (sample_count, *PER_SAMPLE_RESIDUAL_SHAPE):
        raise ValueError("prediction artifact signed residual batch shape is invalid")
    if arrays["source_row_ids"].shape != (sample_count, TACTILE_FRAME_COUNT):
        raise ValueError("prediction artifact source row shape is invalid")
    if arrays["source_step_ids"].shape != (sample_count, TACTILE_FRAME_COUNT):
        raise ValueError("prediction artifact source step shape is invalid")
    if arrays["dataset_indices"].shape != (sample_count,):
        raise ValueError("prediction artifact dataset index shape is invalid")
    if arrays["lerobot_episode_indices"].shape != (sample_count,):
        raise ValueError("prediction artifact episode index shape is invalid")
    for name in (
        "source_row_ids",
        "source_step_ids",
        "dataset_indices",
        "lerobot_episode_indices",
        "active_sensor_ids",
    ):
        if arrays[name].dtype != np.int64:
            raise ValueError(f"prediction artifact {name} must use int64")

    validated_samples = []
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, dict) or set(raw_sample) != {
            "sample_id",
            "dataset_index",
            "lerobot_episode_index",
            "task",
            "source_relative_path",
        }:
            raise ValueError("prediction artifact sample metadata is invalid")
        if raw_sample["dataset_index"] != int(arrays["dataset_indices"][index]):
            raise ValueError("prediction artifact dataset indices disagree")
        if raw_sample["lerobot_episode_index"] != int(
            arrays["lerobot_episode_indices"][index]
        ):
            raise ValueError("prediction artifact episode indices disagree")
        validated_samples.append(
            _validate_sample(
                TactilePredictionSample(
                    sample_id=raw_sample["sample_id"],
                    dataset_index=raw_sample["dataset_index"],
                    lerobot_episode_index=raw_sample["lerobot_episode_index"],
                    task=raw_sample["task"],
                    source_relative_path=raw_sample["source_relative_path"],
                    source_row_ids=arrays["source_row_ids"][index],
                    source_step_ids=arrays["source_step_ids"][index],
                    active_sensor_ids=arrays["active_sensor_ids"],
                    signed_residual=arrays["signed_residual"][index],
                )
            )
        )
    if len({sample.sample_id for sample in validated_samples}) != sample_count:
        raise ValueError("prediction artifact sample IDs are not unique")
    if len({sample.dataset_index for sample in validated_samples}) != sample_count:
        raise ValueError("prediction artifact dataset indices are not unique")
    for array in arrays.values():
        array.setflags(write=False)
    return VerifiedTactilePredictionArtifact(
        root=root,
        metadata=metadata,
        signed_residual=arrays["signed_residual"],
        source_row_ids=arrays["source_row_ids"],
        source_step_ids=arrays["source_step_ids"],
        dataset_indices=arrays["dataset_indices"],
        lerobot_episode_indices=arrays["lerobot_episode_indices"],
        active_sensor_ids=arrays["active_sensor_ids"],
        seal_sha256=seal_sha256,
        file_sha256=file_sha256,
    )
