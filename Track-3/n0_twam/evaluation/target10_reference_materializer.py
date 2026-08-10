# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Stage-B raw-HDF5 materialization for the Target-10 reference9 contract."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, TypeAlias, cast

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from n0_twam.evaluation.raw_tactile_source_contract import (
    load_bound_source_contract,
    record_from_manifest,
)
from n0_twam.evaluation.sealed_artifact_io import sha256_file
from n0_twam.evaluation.target10_prediction_artifact_v3 import (
    VerifiedTarget10PredictionArtifact,
    verify_target10_prediction_artifact,
)
from n0_twam.evaluation.target10_reference_contract import (
    REFERENCE_CONTRACT,
    TARGET_EPISODE_IDS,
    TARGET_RAW_ROWS,
    TARGET_TASKS,
    load_target10_reference_roster,
    validate_view_against_reference_roster,
)
from n0_twam.evaluation.target10_reference_metric import (
    ReferenceVideoPair,
    evaluate_reference_video_pairs,
)
from n0_twam.evaluation.target10_view_contract import load_canonical_target10_view
from n0_twam.evaluation.video_writer import write_rgb_mp4_reference
from n0_twam.integrations.univtac.hdf5_reader import (
    open_verified_hdf5,
    read_image_at,
)
from n0_twam.integrations.univtac.dataset_view import DatasetView
from n0_twam.integrations.univtac.schema import UniVTACEpisodeRecord

RAW_TACTILE_PATHS = (
    "tactile/left_gsmini/rgb_marker",
    "tactile/right_gsmini/rgb_marker",
)
RawSensorGrid: TypeAlias = npt.NDArray[np.uint8]


def _resize_native(image: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("raw tactile image must be uint8 RGB [H,W,3]")
    resized = cv2.resize(array, (128, 128), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def resize_reference_rgb(image: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Apply the frozen torchvision-compatible PIL bilinear RGB resize."""

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("reference resize input must be uint8 RGB [H,W,3]")
    resized = Image.fromarray(array, mode="RGB").resize(
        (256, 192), resample=Image.Resampling.BILINEAR
    )
    output = np.asarray(resized, dtype=np.uint8)
    if output.shape != (192, 256, 3):
        raise RuntimeError("reference resize returned an invalid shape")
    return np.ascontiguousarray(output)


def read_reference_raw_tactile(record: UniVTACEpisodeRecord) -> RawSensorGrid:
    """Read the exact nine raw rows from both tactile sensors."""

    if record.length <= TARGET_RAW_ROWS[-1]:
        raise ValueError("raw Target-10 episode is shorter than requested row 40")
    with open_verified_hdf5(record) as handle:
        sensors = np.stack(
            [
                np.stack(
                    [
                        read_image_at(
                            handle,
                            hdf5_path=hdf5_path,
                            frame_index=row_id,
                        )
                        for row_id in TARGET_RAW_ROWS
                    ]
                )
                for hdf5_path in RAW_TACTILE_PATHS
            ]
        )
    if sensors.dtype != np.uint8 or sensors.shape[0:2] != (2, 9):
        raise RuntimeError("reference raw tactile reader returned an invalid grid")
    return cast(RawSensorGrid, np.ascontiguousarray(sensors))


def reconstruct_reference_prediction(
    raw_sensors: npt.ArrayLike,
    signed_residual: npt.ArrayLike,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Reconstruct predicted and GT combined videos in the reference domain."""

    raw = np.asarray(raw_sensors)
    residual = np.asarray(signed_residual)
    if raw.dtype != np.uint8 or raw.ndim != 5 or raw.shape[:2] != (2, 9):
        raise ValueError("raw sensors must be uint8 [2,9,H,W,3]")
    if residual.dtype != np.float32 or residual.shape != (2, 9, 128, 128, 3):
        raise ValueError("signed residual must be float32 [2,9,128,128,3]")
    if not np.isfinite(residual).all():
        raise ValueError("signed residual contains non-finite values")
    predicted_sensors = []
    ground_truth_sensors = []
    for sensor_index in range(2):
        frame0 = _resize_native(raw[sensor_index, 0])
        native = np.clip(
            np.rint(frame0.astype(np.float32)[None] + 255.0 * residual[sensor_index]),
            0.0,
            255.0,
        ).astype(np.uint8)
        predicted_sensors.append(
            np.stack([resize_reference_rgb(frame) for frame in native])
        )
        ground_truth_sensors.append(
            np.stack([resize_reference_rgb(frame) for frame in raw[sensor_index]])
        )
    prediction = np.concatenate(predicted_sensors, axis=2)
    ground_truth = np.concatenate(ground_truth_sensors, axis=2)
    expected_shape = (9, 192, 512, 3)
    if prediction.shape != expected_shape or ground_truth.shape != expected_shape:
        raise RuntimeError("reference tactile reconstruction shape is invalid")
    return np.ascontiguousarray(prediction), np.ascontiguousarray(ground_truth)


def _validate_artifact_view(
    artifact: VerifiedTarget10PredictionArtifact,
    *,
    evaluation_view_path: Path,
    manifest_path: Path,
    reference_metadata_path: Path,
) -> tuple[DatasetView, tuple[dict[str, object], ...]]:
    view = load_canonical_target10_view(
        view_path=evaluation_view_path,
        manifest_path=manifest_path,
    )
    roster = load_target10_reference_roster(reference_metadata_path)
    bound = validate_view_against_reference_roster(view, roster)
    if (
        artifact.metadata.get("evaluation_view_id") != view.view_id
        or artifact.metadata.get("evaluation_view_sha256") != view.view_sha256
    ):
        raise ValueError("reference prediction artifact is bound to another view")
    actual_samples = artifact.metadata.get("samples")
    if not isinstance(actual_samples, list):
        raise ValueError("reference prediction sample roster is missing")
    actual = tuple(
        (
            sample.get("source_relative_path"),
            sample.get("task"),
            sample.get("episode_index"),
        )
        for sample in actual_samples
        if isinstance(sample, Mapping)
    )
    expected = tuple(
        (row["relative_path"], row["task"], row["episode_id"]) for row in bound
    )
    if actual != expected:
        raise ValueError("reference prediction roster differs from Target-10")
    return view, bound


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def materialize_and_evaluate_target10_reference(
    *,
    prediction_artifact: Path,
    evaluation_view_path: Path,
    reference_metadata_path: Path,
    raw_root: Path,
    manifest_path: Path,
    conversion_report_path: Path,
    metric_script: Path,
    output: Path,
) -> dict[str, object]:
    """Materialize Stage B once and atomically publish strict metrics."""

    artifact = verify_target10_prediction_artifact(prediction_artifact)
    view, _ = _validate_artifact_view(
        artifact,
        evaluation_view_path=evaluation_view_path,
        manifest_path=manifest_path,
        reference_metadata_path=reference_metadata_path,
    )
    source_entries = load_bound_source_contract(
        artifact=artifact,
        manifest_path=manifest_path,
        conversion_report_path=conversion_report_path,
    )
    root = Path(raw_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"reference metric output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp.", dir=target.parent))
    try:
        artifacts_root = staging / "artifacts"
        persistence_root = staging / "persistence"
        pairs: list[ReferenceVideoPair] = []
        persistence_pairs: list[ReferenceVideoPair] = []
        sample_reports = []
        raw_samples = artifact.metadata.get("samples")
        if not isinstance(raw_samples, list):
            raise ValueError("reference prediction sample roster is missing")
        for index, sample in enumerate(raw_samples):
            if not isinstance(sample, Mapping):
                raise ValueError("reference prediction sample metadata is invalid")
            task = str(sample["task"])
            sample_index = index % len(TARGET_EPISODE_IDS)
            episode_id = int(sample["episode_index"])
            if (
                task not in TARGET_TASKS
                or episode_id != TARGET_EPISODE_IDS[sample_index]
            ):
                raise ValueError("reference prediction sample ordering is invalid")
            source_path = str(sample["source_relative_path"])
            expected_entry = source_entries.get(source_path)
            if expected_entry is None:
                raise ValueError("reference prediction source is absent from manifest")
            record = record_from_manifest(
                raw_root=root,
                relative_path=source_path,
                expected=expected_entry,
            )
            raw = read_reference_raw_tactile(record)
            prediction, ground_truth = reconstruct_reference_prediction(
                raw, artifact.signed_residual[index]
            )
            sample_id = f"sample_{sample_index:03d}"
            task_dir = artifacts_root / task
            persistence_dir = persistence_root / task
            pred_path = task_dir / f"{sample_id}_pred_tactile.mp4"
            gt_path = task_dir / f"{sample_id}_gt_tactile.mp4"
            baseline_path = persistence_dir / f"{sample_id}_pred_tactile.mp4"
            writer_evidence = {
                "prediction": write_rgb_mp4_reference(
                    pred_path, prediction, fps=REFERENCE_CONTRACT.fps
                ),
                "ground_truth": write_rgb_mp4_reference(
                    gt_path, ground_truth, fps=REFERENCE_CONTRACT.fps
                ),
                "persistence": write_rgb_mp4_reference(
                    baseline_path,
                    np.repeat(ground_truth[0:1], 9, axis=0),
                    fps=REFERENCE_CONTRACT.fps,
                ),
            }
            pair_fields = {
                "task": task,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "episode_id": episode_id,
            }
            pairs.append(
                ReferenceVideoPair(
                    task=task,
                    sample_id=sample_id,
                    sample_index=sample_index,
                    episode_id=episode_id,
                    prediction_path=pred_path,
                    ground_truth_path=gt_path,
                )
            )
            persistence_pairs.append(
                ReferenceVideoPair(
                    task=task,
                    sample_id=sample_id,
                    sample_index=sample_index,
                    episode_id=episode_id,
                    prediction_path=baseline_path,
                    ground_truth_path=gt_path,
                )
            )
            sample_reports.append(
                {
                    **pair_fields,
                    "source_relative_path": source_path,
                    "target_raw_rows": list(TARGET_RAW_ROWS),
                    "writer": writer_evidence,
                }
            )
        metric = evaluate_reference_video_pairs(
            pairs=pairs, metric_script=metric_script
        )
        persistence = evaluate_reference_video_pairs(
            pairs=persistence_pairs, metric_script=metric_script
        )
        verified_after = verify_target10_prediction_artifact(prediction_artifact)
        if (
            verified_after.seal_sha256 != artifact.seal_sha256
            or verified_after.file_sha256 != artifact.file_sha256
        ):
            raise RuntimeError("reference prediction artifact changed during Stage B")
        report: dict[str, object] = {
            "schema_version": 3,
            "protocol": REFERENCE_CONTRACT.contract_id,
            "contract_sha256": REFERENCE_CONTRACT.sha256,
            "published_score_comparable": False,
            "leaderboard_compatible": False,
            "organizer_contract_confirmed": False,
            "prediction_artifact": {
                "seal_sha256": artifact.seal_sha256,
                "file_sha256": artifact.file_sha256,
            },
            "evaluation_view": {
                "view_id": view.view_id,
                "view_sha256": view.view_sha256,
            },
            "reference_metadata_sha256": sha256_file(reference_metadata_path),
            "metric": metric,
            "persistence_baseline": persistence,
            "samples": sample_reports,
        }
        _write_report(staging / "tactile_prediction_quality.json", report)
        os.replace(staging, target)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = (
    "materialize_and_evaluate_target10_reference",
    "read_reference_raw_tactile",
    "reconstruct_reference_prediction",
    "resize_reference_rgb",
)
