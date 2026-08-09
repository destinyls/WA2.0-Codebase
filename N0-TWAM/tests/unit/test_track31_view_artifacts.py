# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from n0_twam.integrations.univtac.artifact_contracts import (
    verify_track31_artifact_pair,
    verify_track31_evaluation_bundle,
    verify_track31_physical_bundle,
    verify_track31_training_bundle,
)
from n0_twam.integrations.univtac.convert_lerobot import (
    build_lerobot_table_inventory,
)
from n0_twam.integrations.univtac.dataset_view import DatasetView, DatasetViewEntry
from n0_twam.integrations.univtac.manifest import UniVTACDatasetManifest
from n0_twam.integrations.univtac.normalizer import compute_qpos8_normalizer
from n0_twam.integrations.univtac.schema import (
    IMAGE_PATHS,
    OUTPUT_COLOR_SPACE,
    SOURCE_IMAGE_ENCODING_CONTRACT,
    UniVTACEpisodeRecord,
)
from tests.unit.latent_inventory_fixtures import ENCODER_SOURCE_IDENTITY


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_source(root: Path, *, episode_id: int, offset: float) -> Path:
    path = root / "insert_HDMI" / "clean" / f"{episode_id}.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "embodiment/joint",
            data=np.arange(36, dtype=np.float32).reshape(4, 9) + offset,
        )
    return path


def _record(path: Path, *, root: Path, split: str) -> UniVTACEpisodeRecord:
    relative_path = path.relative_to(root).as_posix()
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return UniVTACEpisodeRecord(
        relative_path=relative_path,
        absolute_path=path,
        task="insert_HDMI",
        split=split,
        length=4,
        length_source="embodiment/joint.shape[0]",
        joint_shape=(4, 9),
        image_shapes=tuple((name, (8, 10, 3)) for name, _ in IMAGE_PATHS),
        size_bytes=path.stat().st_size,
        sha256=source_sha256,
    )


def _view_entry(record: UniVTACEpisodeRecord, *, lerobot_id: int) -> DatasetViewEntry:
    assert record.sha256 is not None
    return DatasetViewEntry(
        relative_path=record.relative_path,
        realpath=record.realpath,
        source_sha256=record.sha256,
        task=record.task,
        source_split=record.split,
        source_episode_id=record.episode_id,
        lerobot_episode_id=lerobot_id,
    )


def _view(
    manifest: UniVTACDatasetManifest,
    *,
    view_id: str,
    role: str,
    records: tuple[UniVTACEpisodeRecord, ...],
) -> DatasetView:
    split = records[0].split
    physical_split = {"train": "train759", "validation": "frozen40"}[split]
    manifest_split = tuple(entry for entry in manifest.entries if entry.split == split)
    episode_ids = {
        entry.relative_path: index for index, entry in enumerate(manifest_split)
    }
    return DatasetView(
        view_id=view_id,
        role=role,
        physical_split=physical_split,
        source_manifest_sha256=manifest.manifest_sha256,
        tasks=("insert_HDMI",),
        entries=tuple(
            _view_entry(record, lerobot_id=episode_ids[record.relative_path])
            for record in records
        ),
        selection_method="unit_test",
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _temporal_payload(
    entries: tuple[UniVTACEpisodeRecord, ...],
) -> list[dict[str, object]]:
    return [
        {
            "relative_path": entry.relative_path,
            "raw_length": entry.length,
            "usable_source_range": [entry.usable_start, entry.usable_end],
            "converted_length": entry.converted_length,
            "dropped_prefix_rows": entry.usable_start,
            "dropped_suffix_rows": entry.length - int(entry.usable_end),
            "step_discontinuities_after_rows": list(
                entry.step_discontinuities_after_rows
            ),
            "temporal_policy": entry.temporal_policy,
        }
        for entry in entries
    ]


def _write_episode_tables(
    split_root: Path,
    entries: tuple[UniVTACEpisodeRecord, ...],
) -> None:
    episode_lines: list[str] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []
    source_paths: list[str] = []
    source_steps: list[int] = []
    for episode_index, entry in enumerate(entries):
        episode_lines.append(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "length": entry.converted_length,
                    "tasks": [entry.task],
                    "action_config": [
                        {
                            "start_frame": 0,
                            "end_frame": entry.converted_length,
                            "action_text": entry.task,
                        }
                    ],
                }
            )
        )
        for frame_index in range(entry.converted_length):
            episode_indices.append(episode_index)
            frame_indices.append(frame_index)
            source_paths.append(entry.relative_path)
            source_steps.append(frame_index)
    (split_root / "meta" / "episodes.jsonl").write_text(
        "\n".join(episode_lines) + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": episode_indices,
                "frame_index": frame_indices,
                "source.relative_path": source_paths,
                "source.row_index": [[[value]] for value in frame_indices],
                "source.frame_index": [[[value]] for value in source_steps],
            }
        ),
        split_root / "data" / "chunk.parquet",
    )


def _materialize_conversion(
    *,
    dataset_root: Path,
    manifest: UniVTACDatasetManifest,
) -> Path:
    split_root = dataset_root / "train"
    (split_root / "meta").mkdir(parents=True)
    (split_root / "data").mkdir()
    (split_root / "videos" / "chunk-000").mkdir(parents=True)
    (split_root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    train_entries = tuple(entry for entry in manifest.entries if entry.split == "train")
    _write_episode_tables(split_root, train_entries)
    for feature_name, _ in IMAGE_PATHS:
        feature_root = split_root / "videos" / "chunk-000" / feature_name
        feature_root.mkdir()
        for episode_id in range(len(train_entries)):
            (feature_root / f"episode_{episode_id:06d}.mp4").write_bytes(b"video")
    conversion = {
        "schema_version": 2,
        "action_schema": "qpos8_next_step",
        "output_root": str(split_root),
        "episode_count": len(train_entries),
        "frame_count": sum(entry.converted_length for entry in train_entries),
        "source_image_encoding_contract": SOURCE_IMAGE_ENCODING_CONTRACT,
        "output_color_space": OUTPUT_COLOR_SPACE,
        "source_relative_paths": [entry.relative_path for entry in train_entries],
        "source_sha256": [entry.sha256 for entry in train_entries],
        "temporal_contract": "content_addressed_source_range_v1",
        "source_temporal_selections": _temporal_payload(train_entries),
        "table_inventory": build_lerobot_table_inventory(split_root),
    }
    report = {
        "schema_version": 2,
        "source_manifest_sha256": manifest.manifest_sha256,
        "source_image_encoding_contract": SOURCE_IMAGE_ENCODING_CONTRACT,
        "output_color_space": OUTPUT_COLOR_SPACE,
        "conversions": {"train": conversion},
    }
    report["conversion_report_sha256"] = _canonical_sha256(report)
    path = dataset_root / "conversion_report.json"
    _write_json(path, report)
    return path


def _add_frozen_conversion(
    *,
    conversion_path: Path,
    manifest: UniVTACDatasetManifest,
) -> None:
    dataset_root = conversion_path.parent
    split_root = dataset_root / "frozen40"
    (split_root / "meta").mkdir(parents=True)
    (split_root / "data").mkdir()
    (split_root / "videos" / "chunk-000").mkdir(parents=True)
    (split_root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    entries = tuple(entry for entry in manifest.entries if entry.split == "validation")
    _write_episode_tables(split_root, entries)
    for feature_name, _ in IMAGE_PATHS:
        feature_root = split_root / "videos" / "chunk-000" / feature_name
        feature_root.mkdir()
        for episode_id in range(len(entries)):
            (feature_root / f"episode_{episode_id:06d}.mp4").write_bytes(
                b"frozen-video"
            )
    conversion = {
        "schema_version": 2,
        "action_schema": "qpos8_next_step",
        "output_root": str(split_root),
        "episode_count": len(entries),
        "frame_count": sum(entry.converted_length for entry in entries),
        "source_image_encoding_contract": SOURCE_IMAGE_ENCODING_CONTRACT,
        "output_color_space": OUTPUT_COLOR_SPACE,
        "source_relative_paths": [entry.relative_path for entry in entries],
        "source_sha256": [entry.sha256 for entry in entries],
        "temporal_contract": "content_addressed_source_range_v1",
        "source_temporal_selections": _temporal_payload(entries),
        "table_inventory": build_lerobot_table_inventory(split_root),
    }
    report = json.loads(conversion_path.read_text(encoding="utf-8"))
    report["conversions"]["frozen40"] = conversion
    report.pop("conversion_report_sha256")
    report["conversion_report_sha256"] = _canonical_sha256(report)
    _write_json(conversion_path, report)


def _artifact_fixture(tmp_path: Path) -> tuple[object, ...]:
    train_records = tuple(
        _record(
            _write_source(tmp_path / "source", episode_id=index, offset=float(index)),
            root=tmp_path / "source",
            split="train",
        )
        for index in (10, 11)
    )
    validation_record = _record(
        _write_source(tmp_path / "source", episode_id=0, offset=10_000.0),
        root=tmp_path / "source",
        split="validation",
    )
    manifest = UniVTACDatasetManifest(
        data_root=str(tmp_path / "source"),
        entries=train_records + (validation_record,),
        tasks=("insert_HDMI",),
    )
    train_view = _view(
        manifest,
        view_id="stage_a_dev719_v1",
        role="training",
        records=(train_records[0],),
    )
    internal_view = _view(
        manifest,
        view_id="internal_dev40_v1",
        role="internal_development",
        records=(train_records[1],),
    )
    normalizer = compute_qpos8_normalizer(manifest, view=train_view)
    artifact_root = tmp_path / "artifacts"
    manifest_path = artifact_root / "universe_manifest_v4.json"
    train_view_path = artifact_root / "views" / "train.json"
    internal_view_path = artifact_root / "views" / "internal.json"
    normalizer_path = artifact_root / "normalizer.json"
    _write_json(manifest_path, manifest.to_json_dict())
    _write_json(train_view_path, train_view.to_json_dict())
    _write_json(internal_view_path, internal_view.to_json_dict())
    _write_json(normalizer_path, normalizer.to_json_dict())
    dataset_root = tmp_path / "lerobot"
    conversion_path = _materialize_conversion(
        dataset_root=dataset_root,
        manifest=manifest,
    )
    return (
        manifest,
        train_view,
        internal_view,
        manifest_path,
        train_view_path,
        internal_view_path,
        normalizer_path,
        conversion_path,
    )


def test_normalizer_is_bound_to_exact_training_view(tmp_path: Path) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest, train_view = artifacts[0], artifacts[1]

    normalizer = compute_qpos8_normalizer(manifest, view=train_view)

    assert normalizer.source_view_id == train_view.view_id
    assert normalizer.source_view_sha256 == train_view.view_sha256
    assert normalizer.to_json_dict()["schema_version"] == 2


def test_normalizer_rejects_frozen_evaluation_view(tmp_path: Path) -> None:
    manifest, _, _, *_ = _artifact_fixture(tmp_path)
    validation_record = next(
        entry for entry in manifest.entries if entry.split == "validation"
    )
    frozen_view = _view(
        manifest,
        view_id="frozen_target10_v1",
        role="frozen_evaluation",
        records=(validation_record,),
    )

    with pytest.raises(ValueError, match="training view"):
        compute_qpos8_normalizer(manifest, view=frozen_view)


def test_formal_bundle_does_not_require_frozen_validation_repo(tmp_path: Path) -> None:
    artifacts = _artifact_fixture(tmp_path)
    train_view, internal_view = artifacts[1], artifacts[2]
    manifest_path, train_view_path, internal_view_path = artifacts[3:6]
    normalizer_path, conversion_path = artifacts[6:8]

    verified = verify_track31_training_bundle(
        manifest_path=manifest_path,
        normalizer_path=normalizer_path,
        conversion_report_path=conversion_path,
        dataset_root=conversion_path.parent,
        train_view_path=train_view_path,
        validation_view_path=internal_view_path,
    )

    assert verified.train_view_id == train_view.view_id
    assert verified.train_view_sha256 == train_view.view_sha256
    assert verified.validation_view_id == internal_view.view_id
    assert verified.validation_view_sha256 == internal_view.view_sha256
    assert verified.train_episode_count == 1
    assert verified.validation_episode_count == 1
    assert not (conversion_path.parent / "validation").exists()


def test_formal_training_bundle_binds_current_latent_inventory_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest_path, train_view_path, internal_view_path = artifacts[3:6]
    normalizer_path, conversion_path = artifacts[6:8]
    dataset_root = conversion_path.parent
    (dataset_root / "train759").mkdir()
    calls: list[tuple[Path, object]] = []

    def fake_validate_pair(
        root: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append((root, kwargs["expected_encoder_source_identity"]))
        return {
            "video_inventory_sha256": "1" * 64,
            "tactile_inventory_sha256": "2" * 64,
            "segment_count": 7,
            "video_artifact_count": 14,
            "tactile_artifact_count": 28,
        }

    monkeypatch.setattr(
        "n0_twam.integrations.univtac.artifact_contracts."
        "validate_latent_inventory_pair",
        fake_validate_pair,
    )

    verified = verify_track31_training_bundle(
        manifest_path=manifest_path,
        normalizer_path=normalizer_path,
        conversion_report_path=conversion_path,
        dataset_root=dataset_root,
        train_view_path=train_view_path,
        validation_view_path=internal_view_path,
        encoder_source_identity=ENCODER_SOURCE_IDENTITY,
    )

    assert calls == [(dataset_root / "train759", ENCODER_SOURCE_IDENTITY)]
    assert verified.video_inventory_sha256 == "1" * 64
    assert verified.tactile_inventory_sha256 == "2" * 64
    assert verified.to_json_dict()["video_inventory_sha256"] == "1" * 64
    assert verified.to_json_dict()["tactile_inventory_sha256"] == "2" * 64


def test_formal_training_bundle_rejects_frozen40_mount_before_latent_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest_path, train_view_path, internal_view_path = artifacts[3:6]
    normalizer_path, conversion_path = artifacts[6:8]
    dataset_root = conversion_path.parent
    (dataset_root / "train759").mkdir()
    (dataset_root / "frozen40").symlink_to(dataset_root / "sealed-frozen40")
    called = False

    def unexpected_latent_audit(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal called
        called = True
        raise AssertionError("latent audit must not inspect a non-isolated root")

    monkeypatch.setattr(
        "n0_twam.integrations.univtac.artifact_contracts."
        "validate_latent_inventory_pair",
        unexpected_latent_audit,
    )

    with pytest.raises(ValueError, match="frozen40"):
        verify_track31_training_bundle(
            manifest_path=manifest_path,
            normalizer_path=normalizer_path,
            conversion_report_path=conversion_path,
            dataset_root=dataset_root,
            train_view_path=train_view_path,
            validation_view_path=internal_view_path,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )
    assert called is False


def test_formal_bundle_rejects_self_rehashed_whole_episode_metadata(
    tmp_path: Path,
) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest_path, train_view_path = artifacts[3], artifacts[4]
    normalizer_path, conversion_path = artifacts[6], artifacts[7]
    split_root = conversion_path.parent / "train"
    episodes_path = split_root / "meta" / "episodes.jsonl"
    episode_records = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
    ]
    episode_records[0]["length"] += 1
    episode_records[0]["action_config"][0]["end_frame"] += 1
    episodes_path.write_text(
        "\n".join(json.dumps(record) for record in episode_records) + "\n",
        encoding="utf-8",
    )
    report = json.loads(conversion_path.read_text(encoding="utf-8"))
    report["conversions"]["train"]["table_inventory"] = build_lerobot_table_inventory(
        split_root
    )
    report.pop("conversion_report_sha256")
    report["conversion_report_sha256"] = _canonical_sha256(report)
    _write_json(conversion_path, report)

    with pytest.raises(ValueError, match="episode 0 length does not match"):
        verify_track31_training_bundle(
            manifest_path=manifest_path,
            normalizer_path=normalizer_path,
            conversion_report_path=conversion_path,
            dataset_root=conversion_path.parent,
            train_view_path=train_view_path,
        )


def test_formal_bundle_rejects_self_rehashed_shifted_source_rows(
    tmp_path: Path,
) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest_path, train_view_path = artifacts[3], artifacts[4]
    normalizer_path, conversion_path = artifacts[6], artifacts[7]
    split_root = conversion_path.parent / "train"
    parquet_path = split_root / "data" / "chunk.parquet"
    table = pq.read_table(parquet_path)
    shifted_rows = [
        [[int(value[0][0]) + 100]] for value in table["source.row_index"].to_pylist()
    ]
    table = table.set_column(
        table.schema.get_field_index("source.row_index"),
        "source.row_index",
        pa.array(shifted_rows),
    )
    pq.write_table(table, parquet_path)
    report = json.loads(conversion_path.read_text(encoding="utf-8"))
    report["conversions"]["train"]["table_inventory"] = build_lerobot_table_inventory(
        split_root
    )
    report.pop("conversion_report_sha256")
    report["conversion_report_sha256"] = _canonical_sha256(report)
    _write_json(conversion_path, report)

    with pytest.raises(ValueError, match="source row indices do not match"):
        verify_track31_training_bundle(
            manifest_path=manifest_path,
            normalizer_path=normalizer_path,
            conversion_report_path=conversion_path,
            dataset_root=conversion_path.parent,
            train_view_path=train_view_path,
        )


def test_formal_bundle_rejects_self_rehashed_wrong_episode_task(
    tmp_path: Path,
) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest_path, train_view_path = artifacts[3], artifacts[4]
    normalizer_path, conversion_path = artifacts[6], artifacts[7]
    split_root = conversion_path.parent / "train"
    episodes_path = split_root / "meta" / "episodes.jsonl"
    records = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["tasks"] = ["WRONG_TASK_PROMPT"]
    records[0]["action_config"][0]["action_text"] = "WRONG_TASK_PROMPT"
    episodes_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    report = json.loads(conversion_path.read_text(encoding="utf-8"))
    report["conversions"]["train"]["table_inventory"] = build_lerobot_table_inventory(
        split_root
    )
    report.pop("conversion_report_sha256")
    report["conversion_report_sha256"] = _canonical_sha256(report)
    _write_json(conversion_path, report)

    with pytest.raises(ValueError, match="task does not match manifest"):
        verify_track31_training_bundle(
            manifest_path=manifest_path,
            normalizer_path=normalizer_path,
            conversion_report_path=conversion_path,
            dataset_root=conversion_path.parent,
            train_view_path=train_view_path,
        )


def test_formal_bundle_rejects_normalizer_from_different_view(tmp_path: Path) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest, internal_view = artifacts[0], artifacts[2]
    manifest_path, train_view_path = artifacts[3], artifacts[4]
    normalizer_path, conversion_path = artifacts[6], artifacts[7]
    wrong = compute_qpos8_normalizer(
        manifest,
        view=replace(internal_view, role="training"),
    )
    _write_json(normalizer_path, wrong.to_json_dict())

    with pytest.raises(ValueError, match="normalizer.*declared training view"):
        verify_track31_training_bundle(
            manifest_path=manifest_path,
            normalizer_path=normalizer_path,
            conversion_report_path=conversion_path,
            dataset_root=conversion_path.parent,
            train_view_path=train_view_path,
        )


def test_formal_bundle_allows_stage_b_to_inherit_stage_a_normalizer(
    tmp_path: Path,
) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest = artifacts[0]
    manifest_path, train_view_path = artifacts[3], artifacts[4]
    normalizer_path, conversion_path = artifacts[6], artifacts[7]
    source_view = _view(
        manifest,
        view_id="stage_a_final759_v1",
        role="training",
        records=tuple(entry for entry in manifest.entries if entry.split == "train"),
    )
    source_path = train_view_path.parent / "source.json"
    _write_json(source_path, source_view.to_json_dict())
    _write_json(
        normalizer_path,
        compute_qpos8_normalizer(manifest, view=source_view).to_json_dict(),
    )

    verified = verify_track31_training_bundle(
        manifest_path=manifest_path,
        normalizer_path=normalizer_path,
        conversion_report_path=conversion_path,
        dataset_root=conversion_path.parent,
        train_view_path=train_view_path,
        normalizer_source_view_path=source_path,
    )

    assert verified.normalizer_source_view_id == source_view.view_id


def test_formal_bundle_rejects_frozen_view_as_training_input(tmp_path: Path) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest = artifacts[0]
    manifest_path, train_view_path = artifacts[3], artifacts[4]
    normalizer_path, conversion_path = artifacts[6], artifacts[7]
    validation_record = next(
        entry for entry in manifest.entries if entry.split == "validation"
    )
    frozen_view = _view(
        manifest,
        view_id="frozen_target10_v1",
        role="frozen_evaluation",
        records=(validation_record,),
    )
    _write_json(train_view_path, frozen_view.to_json_dict())

    with pytest.raises(ValueError, match="active train view.*training"):
        verify_track31_training_bundle(
            manifest_path=manifest_path,
            normalizer_path=normalizer_path,
            conversion_report_path=conversion_path,
            dataset_root=conversion_path.parent,
            train_view_path=train_view_path,
        )


def test_formal_bundle_rejects_view_episode_id_drift(tmp_path: Path) -> None:
    artifacts = _artifact_fixture(tmp_path)
    train_view = artifacts[1]
    manifest_path, train_view_path = artifacts[3], artifacts[4]
    normalizer_path, conversion_path = artifacts[6], artifacts[7]
    drifted_view = replace(
        train_view,
        entries=(replace(train_view.entries[0], lerobot_episode_id=1),),
    )
    _write_json(train_view_path, drifted_view.to_json_dict())

    with pytest.raises(ValueError, match="episode identity.*manifest"):
        verify_track31_training_bundle(
            manifest_path=manifest_path,
            normalizer_path=normalizer_path,
            conversion_report_path=conversion_path,
            dataset_root=conversion_path.parent,
            train_view_path=train_view_path,
        )


def test_evaluation_bundle_mounts_only_frozen_physical_repo(tmp_path: Path) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest, normalizer_source_view = artifacts[0], artifacts[1]
    manifest_path, normalizer_source_path = artifacts[3], artifacts[4]
    normalizer_path, conversion_path = artifacts[6], artifacts[7]
    frozen_record = next(
        entry for entry in manifest.entries if entry.split == "validation"
    )
    evaluation_view = _view(
        manifest,
        view_id="frozen_target10_v1",
        role="frozen_evaluation",
        records=(frozen_record,),
    )
    evaluation_view_path = normalizer_source_path.parent / "frozen.json"
    _write_json(evaluation_view_path, evaluation_view.to_json_dict())
    _add_frozen_conversion(
        conversion_path=conversion_path,
        manifest=manifest,
    )

    verified = verify_track31_evaluation_bundle(
        manifest_path=manifest_path,
        normalizer_path=normalizer_path,
        conversion_report_path=conversion_path,
        dataset_root=conversion_path.parent,
        evaluation_view_path=evaluation_view_path,
        normalizer_source_view_path=normalizer_source_path,
    )

    assert verified.evaluation_view_id == evaluation_view.view_id
    assert verified.evaluation_view_sha256 == evaluation_view.view_sha256
    assert verified.evaluation_episode_count == 1
    assert verified.normalizer_source_view_id == normalizer_source_view.view_id


def test_physical_bundle_verifies_frozen40_without_train759_mount(
    tmp_path: Path,
) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest, manifest_path = artifacts[0], artifacts[3]
    conversion_path = artifacts[7]
    _add_frozen_conversion(
        conversion_path=conversion_path,
        manifest=manifest,
    )

    verified = verify_track31_physical_bundle(
        manifest_path=manifest_path,
        conversion_report_path=conversion_path,
        dataset_root=conversion_path.parent / "frozen40",
        physical_split="frozen40",
    )

    assert verified.source_split == "validation"
    assert verified.physical_split == "frozen40"
    assert verified.episode_count == 1
    assert not (conversion_path.parent / "train759").exists()


def test_legacy_manifest_and_normalizer_remain_readable(tmp_path: Path) -> None:
    artifacts = _artifact_fixture(tmp_path)
    manifest_path, normalizer_path = artifacts[3], artifacts[6]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    for entry in manifest["entries"]:
        entry.pop("realpath")
        entry.pop("episode_id")
    manifest_payload = {
        key: manifest[key]
        for key in (
            "schema_version",
            "action_schema",
            "source_image_encoding_contract",
            "output_color_space",
            "tasks",
            "entries",
        )
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest_payload)
    _write_json(manifest_path, manifest)

    normalizer = json.loads(normalizer_path.read_text(encoding="utf-8"))
    normalizer["schema_version"] = 1
    normalizer["source_manifest_sha256"] = manifest["manifest_sha256"]
    normalizer.pop("source_view_id")
    normalizer.pop("source_view_sha256")
    normalizer_payload = {
        key: value for key, value in normalizer.items() if key != "normalizer_sha256"
    }
    normalizer["normalizer_sha256"] = _canonical_sha256(normalizer_payload)
    _write_json(normalizer_path, normalizer)

    verified = verify_track31_artifact_pair(
        manifest_path=manifest_path,
        normalizer_path=normalizer_path,
    )

    assert verified.train_episode_count == 2
    assert verified.validation_episode_count == 1
