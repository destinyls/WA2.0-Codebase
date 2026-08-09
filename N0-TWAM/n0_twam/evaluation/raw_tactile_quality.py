# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Stage-B raw-HDF5 tactile reconstruction and official-script evaluation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from n0_twam.evaluation.official_tactile_metric import (
    invoke_official_script,
)
from n0_twam.evaluation.official_tactile_metric import (  # noqa: F401
    load_strict_official_metrics as _load_strict_official_metrics,
)
from n0_twam.evaluation.raw_tactile_source_contract import (
    load_bound_source_contract as _load_bound_source_contract,
)
from n0_twam.evaluation.raw_tactile_source_contract import (
    record_from_manifest as _record_from_manifest,
)
from n0_twam.evaluation.tactile_prediction_artifact import (
    TACTILE_FRAME_COUNT,
    TACTILE_HEIGHT,
    TACTILE_WIDTH,
    verify_tactile_prediction_artifact,
)
from n0_twam.evaluation.video_writer import write_rgb_mp4_pyav
from n0_twam.integrations.univtac.dataset_view import load_dataset_view
from n0_twam.integrations.univtac.schema import UniVTACEpisodeRecord

RAW_TACTILE_PATHS = (
    "tactile/left_gsmini/rgb_marker",
    "tactile/right_gsmini/rgb_marker",
)
OFFICIAL_VIDEO_SHAPE = (17, 128, 256, 3)

UInt8Video = npt.NDArray[np.uint8]


def resize_legacy_rgb(image: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Apply the legacy RGB-preserving 128x128 ``INTER_AREA`` resize exactly."""

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("legacy tactile frame must be uint8 [H,W,3] RGB")
    resized = cv2.resize(
        array,
        (TACTILE_WIDTH, TACTILE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    if resized.dtype != np.uint8 or resized.shape != (128, 128, 3):
        raise RuntimeError("OpenCV returned an invalid legacy tactile resize")
    return np.ascontiguousarray(resized)


def read_raw_tactile_rows(
    record: UniVTACEpisodeRecord,
    *,
    source_row_ids: npt.ArrayLike,
    source_step_ids: npt.ArrayLike,
) -> UInt8Video:
    """Read by HDF5 array row; simulator step IDs are validation metadata only."""

    row_ids = np.asarray(source_row_ids)
    step_ids = np.asarray(source_step_ids)
    if row_ids.dtype != np.int64 or row_ids.shape != (TACTILE_FRAME_COUNT,):
        raise ValueError("source_row_ids must be int64 [17]")
    if step_ids.dtype != np.int64 or step_ids.shape != row_ids.shape:
        raise ValueError("source_step_ids must be int64 [17]")
    if np.any(np.diff(row_ids) <= 0) or np.any(np.diff(step_ids) <= 0):
        raise ValueError("source row/step IDs must be strictly increasing")
    if np.any(row_ids < 0) or np.any(row_ids >= record.length):
        raise ValueError("source row ID is outside the raw HDF5 episode")

    # Import only after the caller has verified the sealed Stage-A artifact.
    import h5py

    from n0_twam.integrations.univtac.hdf5_reader import (
        read_image_at,
        read_validated_steps,
        verify_source_record,
    )

    verify_source_record(record)
    with h5py.File(record.absolute_path, "r") as handle:
        raw_steps = read_validated_steps(
            handle,
            expected_length=record.length,
            source_label=record.relative_path,
        )
        if not np.array_equal(raw_steps[row_ids], step_ids):
            raise ValueError(
                "sealed simulator step IDs disagree with raw HDF5 row metadata"
            )
        videos = np.stack(
            [
                np.stack(
                    [
                        resize_legacy_rgb(
                            read_image_at(
                                handle,
                                hdf5_path=hdf5_path,
                                frame_index=int(row_id),
                            )
                        )
                        for row_id in row_ids
                    ]
                )
                for hdf5_path in RAW_TACTILE_PATHS
            ]
        )
    verify_source_record(record)
    if videos.shape != (2, 17, 128, 128, 3) or videos.dtype != np.uint8:
        raise RuntimeError("raw tactile reader returned an invalid sensor batch")
    return np.ascontiguousarray(videos)


def reconstruct_absolute_tactile(
    raw_frame0: npt.ArrayLike,
    signed_residual: npt.ArrayLike,
) -> UInt8Video:
    """Apply ``clip(round(raw_frame0 + 255 * residual[t]))`` for every frame."""

    frame0 = np.asarray(raw_frame0)
    residual = np.asarray(signed_residual)
    if frame0.dtype != np.uint8 or frame0.shape != (128, 128, 3):
        raise ValueError("raw_frame0 must be uint8 [128,128,3]")
    if residual.dtype != np.float32 or residual.shape != (17, 128, 128, 3):
        raise ValueError("signed_residual must be float32 [17,128,128,3]")
    if not np.isfinite(residual).all():
        raise ValueError("signed residual contains non-finite values")
    absolute = np.rint(frame0.astype(np.float32)[None] + 255.0 * residual)
    return np.clip(absolute, 0.0, 255.0).astype(np.uint8)


def concatenate_tactile_sensors(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
) -> UInt8Video:
    """Concatenate left/right tactile RGB by width."""

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    expected = (17, 128, 128, 3)
    if (
        left_array.dtype != np.uint8
        or right_array.dtype != np.uint8
        or left_array.shape != expected
        or right_array.shape != expected
    ):
        raise ValueError("left/right tactile videos must be uint8 [17,128,128,3]")
    combined = np.concatenate((left_array, right_array), axis=2)
    if combined.shape != OFFICIAL_VIDEO_SHAPE:
        raise RuntimeError("tactile sensor concatenation produced an invalid shape")
    return np.ascontiguousarray(combined)


def _invoke_official_script(
    *,
    staging_root: Path,
    expected_video_names: Sequence[str],
    script_path: Path,
    expected_script_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    return invoke_official_script(
        staging_root=staging_root,
        expected_video_names=expected_video_names,
        script_path=script_path,
        expected_script_sha256=expected_script_sha256,
    )


def _verify_evaluation_view_binding(
    *,
    artifact: object,
    evaluation_view_path: Path,
) -> dict[str, object]:
    """Require the sealed artifact to cover exactly one immutable frozen view."""

    view_path = Path(evaluation_view_path).resolve(strict=True)
    view = load_dataset_view(view_path)
    if view.role != "frozen_evaluation" or view.physical_split != "frozen40":
        raise ValueError("raw tactile metric requires a frozen40 evaluation view")
    metadata = getattr(artifact, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("prediction artifact metadata is invalid")
    if (
        metadata.get("evaluation_view_id") != view.view_id
        or metadata.get("evaluation_view_sha256") != view.view_sha256
        or metadata.get("source_manifest_sha256") != view.source_manifest_sha256
    ):
        raise ValueError("prediction artifact does not belong to evaluation view")
    raw_samples = metadata.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != len(view.entries):
        raise ValueError("prediction artifact does not cover the full evaluation view")
    sample_identities = {
        (
            sample.get("source_relative_path"),
            sample.get("task"),
            sample.get("lerobot_episode_index"),
        )
        for sample in raw_samples
        if isinstance(sample, dict)
    }
    expected_identities = {
        (entry.relative_path, entry.task, entry.lerobot_episode_id)
        for entry in view.entries
    }
    if sample_identities != expected_identities:
        raise ValueError("prediction samples differ from the evaluation view roster")
    return {
        "path": str(view_path),
        "view_id": view.view_id,
        "view_sha256": view.view_sha256,
        "episode_count": len(view.entries),
        "tasks": list(view.tasks),
    }


def evaluate_raw_tactile_quality(
    *,
    prediction_artifact: Path,
    evaluation_view_path: Path,
    raw_root: Path,
    manifest_path: Path,
    conversion_report_path: Path,
    output: Path,
    official_metric_script: Path,
    expected_official_metric_sha256: str,
    fps: int = 10,
) -> dict[str, object]:
    """Run Stage B and atomically publish only after post-verification succeeds."""

    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    sealed = verify_tactile_prediction_artifact(prediction_artifact)
    evaluation_view = _verify_evaluation_view_binding(
        artifact=sealed,
        evaluation_view_path=evaluation_view_path,
    )
    resolved_official_metric_script = Path(official_metric_script).resolve(strict=True)
    if not resolved_official_metric_script.is_file():
        raise FileNotFoundError(resolved_official_metric_script)
    source_entries = _load_bound_source_contract(
        artifact=sealed,
        manifest_path=manifest_path,
        conversion_report_path=conversion_report_path,
    )
    resolved_raw_root = Path(raw_root).resolve(strict=True)
    if not resolved_raw_root.is_dir():
        raise NotADirectoryError(resolved_raw_root)
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"raw tactile evaluation output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp.", dir=target.parent))
    try:
        prediction_dir = staging / "generate_videos"
        ground_truth_dir = staging / "gt_videos"
        prediction_dir.mkdir()
        ground_truth_dir.mkdir()
        sample_reports = []
        expected_names = []
        for index, sample in enumerate(sealed.metadata["samples"]):
            if not isinstance(sample, dict):
                raise ValueError("sealed sample metadata is invalid")
            source_path = str(sample["source_relative_path"])
            expected_entry = source_entries.get(source_path)
            if expected_entry is None:
                raise ValueError("sealed sample is not in the validation manifest")
            record = _record_from_manifest(
                raw_root=resolved_raw_root,
                relative_path=source_path,
                expected=expected_entry,
            )
            raw_sensors = read_raw_tactile_rows(
                record,
                source_row_ids=sealed.source_row_ids[index],
                source_step_ids=sealed.source_step_ids[index],
            )
            predicted_sensors = np.stack(
                [
                    reconstruct_absolute_tactile(
                        raw_sensors[sensor_index, 0],
                        sealed.signed_residual[index, sensor_index],
                    )
                    for sensor_index in range(2)
                ]
            )
            predicted = concatenate_tactile_sensors(
                predicted_sensors[0], predicted_sensors[1]
            )
            ground_truth = concatenate_tactile_sensors(raw_sensors[0], raw_sensors[1])
            if (
                predicted.shape != OFFICIAL_VIDEO_SHAPE
                or ground_truth.shape != OFFICIAL_VIDEO_SHAPE
            ):
                raise RuntimeError("official tactile arrays have an invalid shape")
            video_name = f"{sample['sample_id']}.mp4"
            if Path(video_name).name != video_name:
                raise ValueError("sealed sample ID is unsafe for official filenames")
            codecs = {
                "prediction": write_rgb_mp4_pyav(
                    prediction_dir / video_name, predicted, fps=fps
                ),
                "ground_truth": write_rgb_mp4_pyav(
                    ground_truth_dir / video_name, ground_truth, fps=fps
                ),
            }
            expected_names.append(video_name)
            sample_reports.append(
                {
                    **sample,
                    "video_name": video_name,
                    "frame_count": 17,
                    "shape": list(OFFICIAL_VIDEO_SHAPE),
                    "verified_h264_codecs": codecs,
                }
            )
        official_metrics, official_script = _invoke_official_script(
            staging_root=staging,
            expected_video_names=expected_names,
            script_path=resolved_official_metric_script,
            expected_script_sha256=expected_official_metric_sha256,
        )
        verified_after = verify_tactile_prediction_artifact(prediction_artifact)
        if (
            verified_after.seal_sha256 != sealed.seal_sha256
            or verified_after.file_sha256 != sealed.file_sha256
        ):
            raise RuntimeError("prediction artifact changed during raw evaluation")
        report: dict[str, object] = {
            "schema_version": 2,
            "protocol": "causal_raw17_legacy_metric_diagnostic_v1",
            "conditioning_protocol": sealed.metadata.get("conditioning_protocol_id"),
            "official_script_compatible": True,
            "leaderboard_oriented": False,
            "organizer_contract_confirmed": False,
            "leaderboard_compatible": False,
            "published_score_comparable": False,
            "published_score_blocker": (
                "Wan2.2 golden 21.26/0.746 frame/domain/backend contract "
                "has not been reproduced"
            ),
            "prediction_artifact_ground_truth_embedded": False,
            "internal_residual_evaluator_unchanged": True,
            "frame_policy": "all_17_frames_including_condition_frame",
            "pixel_domain": "raw_hdf5_rgb_marker_absolute_rgb_uint8",
            "prediction_artifact": {
                "path": str(sealed.root),
                "seal_sha256": sealed.seal_sha256,
                "file_sha256": sealed.file_sha256,
            },
            "evaluation_view": evaluation_view,
            "official_script": official_script,
            "official_metrics": official_metrics,
            "samples": sample_reports,
        }
        encoded = (
            json.dumps(
                report, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True
            )
            + "\n"
        )
        report_path = staging / "raw_tactile_quality.json"
        with report_path.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, target)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)
