# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from n0_twam.evaluation.fair_protocol import CAUSAL_FUTURE_ONLY_PROTOCOL
from n0_twam.evaluation.sealed_artifact_io import (
    canonical_json,
    sha256_bytes,
    sha256_file,
)
from n0_twam.evaluation.tactile_prediction_artifact import (
    ACTIVE_TACTILE_SENSOR_IDS,
    MODEL_TACTILE_SENSOR_CAPACITY,
    TactilePredictionSample,
    verify_tactile_prediction_artifact,
    write_tactile_prediction_artifact,
)


def _sample(*, dataset_index: int = 47) -> TactilePredictionSample:
    residual = np.zeros((2, 17, 128, 128, 3), dtype=np.float32)
    residual[0, 0] = 0.125
    return TactilePredictionSample(
        sample_id="sample_000000",
        dataset_index=dataset_index,
        lerobot_episode_index=3,
        task="lift_bottle",
        source_relative_path="lift_bottle/clean/900.hdf5",
        source_row_ids=np.arange(17, dtype=np.int64),
        source_step_ids=np.arange(100, 117, dtype=np.int64) * 7,
        active_sensor_ids=np.asarray((0, 1), dtype=np.int64),
        signed_residual=residual,
    )


def _write(output: Path, sample: TactilePredictionSample | None = None) -> None:
    write_tactile_prediction_artifact(
        output,
        samples=(_sample() if sample is None else sample,),
        source_manifest_sha256="a" * 64,
        conversion_report_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        decoder_identity_sha256="d" * 64,
        conditioning_protocol_id=CAUSAL_FUTURE_ONLY_PROTOCOL,
        evaluation_view_id="frozen_target10_v1",
        evaluation_view_sha256="e" * 64,
        generation_provenance={"seed": 20260801, "n_steps": 8},
    )


def test_prediction_artifact_is_atomic_sealed_and_shape_strict(tmp_path: Path) -> None:
    output = tmp_path / "prediction_artifact"

    _write(output)
    verified = verify_tactile_prediction_artifact(output)

    assert {path.name for path in output.iterdir()} == {
        "artifact.json",
        "predictions.npz",
        "seal.json",
    }
    assert verified.signed_residual.shape == (1, 2, 17, 128, 128, 3)
    assert verified.signed_residual.dtype == np.float32
    assert verified.signed_residual.flags.writeable is False
    assert verified.source_row_ids.tolist() == [list(range(17))]
    assert verified.source_step_ids[0, 0] == 700
    assert verified.metadata["model_tactile_sensor_capacity"] == 4
    assert verified.metadata["active_tactile_sensor_ids"] == [0, 1]
    assert verified.metadata["leaderboard_compatible"] is False
    assert verified.metadata["published_score_comparable"] is False
    assert verified.metadata["ground_truth_embedded"] is False
    assert verified.metadata["conditioning_protocol_id"] == (
        CAUSAL_FUTURE_ONLY_PROTOCOL
    )
    assert verified.metadata["evaluation_view_id"] == "frozen_target10_v1"
    with np.load(output / "predictions.npz", allow_pickle=False) as payload:
        assert "lerobot_h264_rgb" not in payload.files
    assert ACTIVE_TACTILE_SENSOR_IDS == (0, 1)
    assert MODEL_TACTILE_SENSOR_CAPACITY == 4


def test_prediction_artifact_rejects_byte_tampering(tmp_path: Path) -> None:
    output = tmp_path / "prediction_artifact"
    _write(output)
    payload = output / "predictions.npz"
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="sha256"):
        verify_tactile_prediction_artifact(output)


def test_prediction_artifact_rejects_resealed_semantic_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prediction_artifact"
    _write(output)
    metadata_path = output / "artifact.json"
    seal_path = output / "seal.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_row_id_semantics"] = "simulator_step"
    metadata_path.write_bytes(canonical_json(metadata) + b"\n")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["files"]["artifact.json"] = {
        "size_bytes": metadata_path.stat().st_size,
        "sha256": sha256_file(metadata_path),
    }
    unsigned = {key: value for key, value in seal.items() if key != "seal_sha256"}
    seal["seal_sha256"] = sha256_bytes(canonical_json(unsigned))
    seal_path.write_bytes(canonical_json(seal) + b"\n")

    with pytest.raises(ValueError, match="metadata contract"):
        verify_tactile_prediction_artifact(output)


def test_prediction_artifact_requires_float32_signed_residual(tmp_path: Path) -> None:
    sample = _sample()
    invalid = TactilePredictionSample(
        **{
            **sample.__dict__,
            "signed_residual": sample.signed_residual.astype(np.float64),
        }
    )

    with pytest.raises(ValueError, match="float32"):
        _write(tmp_path / "artifact", invalid)

    assert not (tmp_path / "artifact").exists()


def test_prediction_artifact_rejects_wrong_active_sensor_contract(
    tmp_path: Path,
) -> None:
    sample = _sample()
    invalid = TactilePredictionSample(
        **{
            **sample.__dict__,
            "active_sensor_ids": np.asarray((1, 2), dtype=np.int64),
        }
    )

    with pytest.raises(ValueError, match="active tactile sensor IDs"):
        _write(tmp_path / "artifact", invalid)


def test_prediction_artifact_dataset_index_is_not_used_as_manifest_index(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prediction_artifact"
    _write(output, _sample(dataset_index=9182))

    verified = verify_tactile_prediction_artifact(output)

    assert verified.metadata["samples"][0]["dataset_index"] == 9182
    assert verified.metadata["samples"][0]["lerobot_episode_index"] == 3
    assert (
        verified.metadata["samples"][0]["source_relative_path"]
        == "lift_bottle/clean/900.hdf5"
    )
