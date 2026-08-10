# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from n0_twam.evaluation.official_tactile_metric import invoke_official_script
from n0_twam.evaluation.raw_tactile_quality import (
    _load_strict_official_metrics,
    _verify_evaluation_view_binding,
    concatenate_tactile_sensors,
    evaluate_raw_tactile_quality,
    read_raw_tactile_rows,
    reconstruct_absolute_tactile,
    resize_legacy_rgb,
)
from n0_twam.evaluation.raw_tactile_source_contract import (
    load_bound_source_contract,
    record_from_manifest,
)
from n0_twam.integrations.univtac.dataset_view import DatasetView, DatasetViewEntry
from n0_twam.integrations.univtac.hdf5_reader import audit_episode
from n0_twam.integrations.univtac.manifest import UniVTACDatasetManifest
from n0_twam.integrations.univtac.schema import UNIVTAC_ALL_TASKS


def _write_raw_episode(
    root: Path,
    *,
    task: str = "lift_bottle",
) -> tuple[Path, str]:
    relative_path = f"{task}/clean/7.hdf5"
    path = root / relative_path
    path.parent.mkdir(parents=True)
    frame_count = 20
    with h5py.File(path, "w") as handle:
        handle.create_dataset("embodiment/joint", data=np.zeros((frame_count, 8)))
        handle.create_dataset("step", data=np.arange(100, 100 + frame_count) * 11)
        for name, offset in (
            ("observation/head/rgb", 1),
            ("observation/wrist/rgb", 2),
            ("tactile/left_gsmini/rgb_marker", 3),
            ("tactile/right_gsmini/rgb_marker", 9),
        ):
            frames = np.stack(
                [
                    np.full((8, 6, 3), index + offset, dtype=np.uint8)
                    for index in range(frame_count)
                ]
            )
            handle.create_dataset(name, data=frames)
    return path, relative_path


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_raw_source_contract_accepts_standard_frozen40_conversion(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    _, relative_path = _write_raw_episode(raw_root)
    record = audit_episode(
        raw_root,
        relative_path,
        split="validation",
        hash_file=True,
    )
    manifest = UniVTACDatasetManifest(
        data_root=str(raw_root.resolve(strict=True)),
        entries=(record,),
        tasks=("lift_bottle",),
    )
    entry = record.to_json_dict()
    manifest_payload = manifest.to_json_dict()
    manifest_sha256 = manifest.manifest_sha256
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    conversion_core = {
        "schema_version": 2,
        "source_manifest_sha256": manifest_sha256,
        "source_image_encoding_contract": "opencv_imencode_rgb_input_v1",
        "output_color_space": "RGB",
        "conversions": {
            "frozen40": {
                "schema_version": 2,
                "action_schema": "qpos8_next_step",
                "source_relative_paths": [entry["relative_path"]],
                "source_sha256": [entry["sha256"]],
                "frame_count": record.converted_length,
                "temporal_contract": "content_addressed_source_range_v1",
                "source_temporal_selections": [
                    {
                        "relative_path": entry["relative_path"],
                        "raw_length": record.length,
                        "usable_source_range": [0, record.length],
                        "converted_length": record.converted_length,
                        "dropped_prefix_rows": 0,
                        "dropped_suffix_rows": 0,
                        "step_discontinuities_after_rows": [],
                        "temporal_policy": "strict_monotonic_v1",
                    }
                ],
            }
        },
    }
    conversion_sha256 = _canonical_sha256(conversion_core)
    conversion_path = tmp_path / "conversion.json"
    conversion_path.write_text(
        json.dumps(
            {
                **conversion_core,
                "conversion_report_sha256": conversion_sha256,
            }
        ),
        encoding="utf-8",
    )
    artifact = SimpleNamespace(
        metadata={
            "source_manifest_sha256": manifest_sha256,
            "conversion_report_sha256": conversion_sha256,
        }
    )

    result = load_bound_source_contract(
        artifact=artifact,
        manifest_path=manifest_path,
        conversion_report_path=conversion_path,
    )

    assert result == {entry["relative_path"]: entry}


def test_raw_source_contract_supports_frozen_other_tasks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _, relative_path = _write_raw_episode(raw_root, task="insert_hole")
    audited = audit_episode(
        raw_root,
        relative_path,
        split="validation",
        hash_file=True,
        allowed_tasks=UNIVTAC_ALL_TASKS,
    )

    loaded = record_from_manifest(
        raw_root=raw_root,
        relative_path=relative_path,
        expected=audited.to_json_dict(),
    )

    assert loaded.task == "insert_hole"
    assert loaded.temporal_policy == "strict_monotonic_v1"


def test_raw_reader_indexes_array_rows_not_simulator_steps(tmp_path: Path) -> None:
    _, relative_path = _write_raw_episode(tmp_path)
    record = audit_episode(
        tmp_path,
        relative_path,
        split="validation",
        hash_file=True,
    )
    row_ids = np.arange(17, dtype=np.int64)
    step_ids = np.arange(100, 117, dtype=np.int64) * 11

    frames = read_raw_tactile_rows(
        record,
        source_row_ids=row_ids,
        source_step_ids=step_ids,
    )

    assert frames.shape == (2, 17, 128, 128, 3)
    assert frames.dtype == np.uint8
    assert np.all(frames[0, 0] == 3)
    assert np.all(frames[0, 16] == 19)
    assert np.all(frames[1, 0] == 9)


def test_reconstruction_applies_uniform_formula_to_frame_zero() -> None:
    raw_frame0 = np.full((128, 128, 3), 100, dtype=np.uint8)
    signed_residual = np.zeros((17, 128, 128, 3), dtype=np.float32)
    signed_residual[0] = 0.1
    signed_residual[1] = -1.0
    signed_residual[2] = 1.0

    reconstructed = reconstruct_absolute_tactile(raw_frame0, signed_residual)

    assert reconstructed.shape == (17, 128, 128, 3)
    assert np.all(reconstructed[0] == 126)
    assert np.all(reconstructed[1] == 0)
    assert np.all(reconstructed[2] == 255)
    assert not np.array_equal(reconstructed[0], raw_frame0)


def test_sensor_concatenation_is_width_major() -> None:
    left = np.full((17, 128, 128, 3), 1, dtype=np.uint8)
    right = np.full_like(left, 2)

    combined = concatenate_tactile_sensors(left, right)

    assert combined.shape == (17, 128, 256, 3)
    assert np.all(combined[:, :, :128] == 1)
    assert np.all(combined[:, :, 128:] == 2)


def test_legacy_resize_always_uses_inter_area(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[int, int], int]] = []

    def _resize(
        image: np.ndarray, size: tuple[int, int], *, interpolation: int
    ) -> np.ndarray:
        calls.append((size, interpolation))
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    monkeypatch.setattr("cv2.resize", _resize)
    image = np.zeros((8, 6, 3), dtype=np.uint8)

    resized = resize_legacy_rgb(image)

    import cv2

    assert resized.shape == (128, 128, 3)
    assert calls == [((128, 128), cv2.INTER_AREA)]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_official_metrics_reject_nonfinite_json_constants(
    tmp_path: Path,
    constant: str,
) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        '{"num_videos":1,"average_psnr":%s,"average_ssim":0.5,"per_video":[]}'
        % constant,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite"):
        _load_strict_official_metrics(path, expected_video_names=("sample.mp4",))


def test_official_metrics_require_exact_names_counts_and_shapes(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "num_videos": 1,
                "average_psnr": 20.0,
                "average_ssim": 0.8,
                "per_video": [
                    {
                        "video_name": "wrong.mp4",
                        "num_frames": 17,
                        "psnr": 20.0,
                        "ssim": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="filenames"):
        _load_strict_official_metrics(path, expected_video_names=("sample.mp4",))


def test_official_metric_script_requires_preapproved_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "metric.py"
    script.write_text("# pinned metric\n", encoding="utf-8")
    monkeypatch.setattr(
        "n0_twam.evaluation.official_tactile_metric._verify_video_inputs",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="approved identity"):
        invoke_official_script(
            staging_root=tmp_path,
            expected_video_names=("sample.mp4",),
            script_path=script,
            expected_script_sha256="f" * 64,
        )


def test_prediction_artifact_must_cover_exact_frozen_view(tmp_path: Path) -> None:
    entry = DatasetViewEntry(
        relative_path="lift_bottle/clean/7.hdf5",
        realpath=str(tmp_path / "7.hdf5"),
        source_sha256="a" * 64,
        task="lift_bottle",
        source_split="validation",
        source_episode_id=7,
        lerobot_episode_id=0,
    )
    view = DatasetView(
        view_id="frozen_target10_v1",
        role="frozen_evaluation",
        physical_split="frozen40",
        source_manifest_sha256="b" * 64,
        tasks=("lift_bottle",),
        entries=(entry,),
        selection_method="unit_test",
    )
    view_path = tmp_path / "frozen.json"
    view_path.write_text(json.dumps(view.to_json_dict()), encoding="utf-8")
    artifact = SimpleNamespace(
        metadata={
            "evaluation_view_id": view.view_id,
            "evaluation_view_sha256": view.view_sha256,
            "source_manifest_sha256": view.source_manifest_sha256,
            "samples": [
                {
                    "source_relative_path": entry.relative_path,
                    "task": entry.task,
                    "lerobot_episode_index": entry.lerobot_episode_id,
                }
            ],
        }
    )

    bound = _verify_evaluation_view_binding(
        artifact=artifact,
        evaluation_view_path=view_path,
    )
    assert bound["view_sha256"] == view.view_sha256

    artifact.metadata["samples"] = []
    with pytest.raises(ValueError, match="full evaluation view"):
        _verify_evaluation_view_binding(
            artifact=artifact,
            evaluation_view_path=view_path,
        )


def test_stage_b_rejects_bad_seal_before_any_raw_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_calls: list[object] = []
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality.verify_tactile_prediction_artifact",
        lambda path: (_ for _ in ()).throw(ValueError("bad seal")),
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality._record_from_manifest",
        lambda **kwargs: raw_calls.append(kwargs),
    )
    official_script = tmp_path / "official.py"
    official_script.write_text("# test metric stub\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bad seal"):
        evaluate_raw_tactile_quality(
            prediction_artifact=tmp_path / "artifact",
            evaluation_view_path=tmp_path / "frozen_view.json",
            raw_root=tmp_path,
            manifest_path=tmp_path / "manifest.json",
            conversion_report_path=tmp_path / "conversion.json",
            official_metric_script=official_script,
            expected_official_metric_sha256="e" * 64,
            output=tmp_path / "result",
        )

    assert raw_calls == []
    assert not (tmp_path / "result").exists()


def test_stage_b_reverifies_after_metric_and_never_publishes_changed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = {
        "sample_id": "sample_000000",
        "dataset_index": 99,
        "lerobot_episode_index": 4,
        "task": "lift_bottle",
        "source_relative_path": "lift_bottle/clean/7.hdf5",
    }
    common = {
        "root": tmp_path / "artifact",
        "metadata": {"samples": [sample]},
        "source_row_ids": np.arange(17, dtype=np.int64)[None],
        "source_step_ids": np.arange(100, 117, dtype=np.int64)[None],
        "signed_residual": np.zeros((1, 2, 17, 128, 128, 3), dtype=np.float32),
        "seal_sha256": "a" * 64,
    }
    before = SimpleNamespace(**common, file_sha256={"predictions.npz": "b" * 64})
    after = SimpleNamespace(**common, file_sha256={"predictions.npz": "c" * 64})
    verifications = iter((before, after))
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality.verify_tactile_prediction_artifact",
        lambda path: next(verifications),
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality._verify_evaluation_view_binding",
        lambda **kwargs: {"view_id": "frozen_target10_v1"},
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality._load_bound_source_contract",
        lambda **kwargs: {sample["source_relative_path"]: {}},
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality._record_from_manifest",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality.read_raw_tactile_rows",
        lambda *args, **kwargs: np.zeros((2, 17, 128, 128, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality.write_rgb_mp4_pyav",
        lambda *args, **kwargs: "libx264",
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality._invoke_official_script",
        lambda **kwargs: (
            {
                "num_videos": 1,
                "average_psnr": 20.0,
                "average_ssim": 0.8,
                "per_video": [],
            },
            {"path": "official.py", "sha256": "d" * 64},
        ),
    )
    official_script = tmp_path / "official.py"
    official_script.write_text("# test metric stub\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed during"):
        evaluate_raw_tactile_quality(
            prediction_artifact=tmp_path / "artifact",
            evaluation_view_path=tmp_path / "frozen_view.json",
            raw_root=tmp_path,
            manifest_path=tmp_path / "manifest.json",
            conversion_report_path=tmp_path / "conversion.json",
            official_metric_script=official_script,
            expected_official_metric_sha256="e" * 64,
            output=tmp_path / "result",
        )

    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result.tmp.*"))


def test_stage_b_success_report_has_exact_compatibility_flags_without_gt_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = {
        "sample_id": "sample_000000",
        "dataset_index": 812,
        "lerobot_episode_index": 4,
        "task": "lift_bottle",
        "source_relative_path": "lift_bottle/clean/7.hdf5",
    }
    sealed = SimpleNamespace(
        root=tmp_path / "artifact",
        metadata={"samples": [sample]},
        source_row_ids=np.arange(17, dtype=np.int64)[None],
        source_step_ids=(np.arange(100, 117, dtype=np.int64) * 11)[None],
        signed_residual=np.zeros((1, 2, 17, 128, 128, 3), dtype=np.float32),
        seal_sha256="a" * 64,
        file_sha256={"predictions.npz": "b" * 64},
    )
    verifications: list[Path] = []

    def _verify(path: Path) -> SimpleNamespace:
        verifications.append(path)
        return sealed

    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality.verify_tactile_prediction_artifact",
        _verify,
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality._verify_evaluation_view_binding",
        lambda **kwargs: {"view_id": "frozen_target10_v1"},
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality._load_bound_source_contract",
        lambda **kwargs: {sample["source_relative_path"]: {}},
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality._record_from_manifest",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality.read_raw_tactile_rows",
        lambda *args, **kwargs: np.zeros((2, 17, 128, 128, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality.write_rgb_mp4_pyav",
        lambda *args, **kwargs: "libx264",
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.raw_tactile_quality._invoke_official_script",
        lambda **kwargs: (
            {
                "num_videos": 1,
                "average_psnr": 20.0,
                "average_ssim": 0.8,
                "per_video": [
                    {
                        "video_name": "sample_000000.mp4",
                        "num_frames": 17,
                        "psnr": 20.0,
                        "ssim": 0.8,
                    }
                ],
            },
            {"path": "official.py", "sha256": "d" * 64},
        ),
    )
    official_script = tmp_path / "official.py"
    official_script.write_text("# test metric stub\n", encoding="utf-8")

    report = evaluate_raw_tactile_quality(
        prediction_artifact=tmp_path / "artifact",
        evaluation_view_path=tmp_path / "frozen_view.json",
        raw_root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        conversion_report_path=tmp_path / "conversion.json",
        official_metric_script=official_script,
        expected_official_metric_sha256="e" * 64,
        output=tmp_path / "result",
    )

    assert verifications == [tmp_path / "artifact", tmp_path / "artifact"]
    assert report["official_script_compatible"] is True
    assert report["leaderboard_oriented"] is False
    assert report["organizer_contract_confirmed"] is False
    assert report["leaderboard_compatible"] is False
    assert report["published_score_comparable"] is False
    assert report["prediction_artifact_ground_truth_embedded"] is False
    assert "raw_hdf5_resized_vs_lerobot_h264_domain_gap" not in report
    assert (tmp_path / "result" / "raw_tactile_quality.json").is_file()
