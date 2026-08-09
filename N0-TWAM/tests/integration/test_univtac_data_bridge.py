# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import n0_twam.integrations.univtac.temporal_contract as temporal_contract_module
from n0_twam.integrations.univtac.artifact_contracts import (
    verify_track31_artifact_pair,
    verify_track31_training_bundle,
)
from n0_twam.integrations.univtac.convert_lerobot import (
    build_lerobot_features,
    build_lerobot_table_inventory,
    convert_episode_to_sink,
    freeze_episode_action_config,
    iter_episode_frames,
    write_lerobot_dataset,
)
from n0_twam.integrations.univtac.hdf5_reader import audit_episode, sha256_file
from n0_twam.integrations.univtac.manifest import (
    UniVTACDatasetManifest,
    build_dataset_manifest,
    derive_split_paths,
    load_dataset_manifest,
)
from n0_twam.integrations.univtac.normalizer import compute_qpos8_normalizer
from n0_twam.integrations.univtac.schema import (
    IMAGE_PATHS,
    OUTPUT_COLOR_SPACE,
    SOURCE_IMAGE_ENCODING_CONTRACT,
)
from n0_twam.integrations.univtac.temporal_contract import (
    PINNED_MAXIMAL_PREFIX_POLICY,
    PinnedTemporalContract,
    resolve_temporal_selection,
    validate_declared_temporal_selection,
)
from script.track3_1.prepare_univtac import prepare_artifacts


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_episode(
    root: Path,
    *,
    task: str,
    episode_id: int,
    offset: float,
    length: int = 6,
    height: int = 8,
    width: int = 10,
) -> str:
    relative_path = f"{task}/clean/{episode_id}.hdf5"
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    images = np.zeros((length, height, width, 3), dtype=np.uint8)
    for index in range(length):
        images[index, ..., 0] = episode_id + index
        images[index, ..., 1] = 2 * index
        images[index, ..., 2] = 100
    # Keep fixtures content-distinct across tasks so the schema-v4 global
    # content-identity guard tests real episodes instead of synthetic clones.
    task_offset = 0.0 if task == "insert_HDMI" else 0.125
    joint = (
        np.arange(length * 9, dtype=np.float32).reshape(length, 9)
        + offset
        + task_offset
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("step", data=np.arange(length, dtype=np.int64))
        handle.create_dataset("embodiment/joint", data=joint)
        handle.create_dataset(
            "observation/head/rgb",
            data=images,
        )
        handle.create_dataset(
            "observation/wrist/rgb",
            data=images + np.uint8(1),
        )
        handle.create_dataset(
            "tactile/left_gsmini/rgb_marker",
            data=images + np.uint8(2),
        )
        handle.create_dataset(
            "tactile/right_gsmini/rgb_marker",
            data=images + np.uint8(3),
        )
    return relative_path


def test_audit_and_converter_preserve_next_step_qpos8(tmp_path: Path) -> None:
    relative_path = _write_episode(
        tmp_path,
        task="insert_HDMI",
        episode_id=0,
        offset=10.0,
    )
    record = audit_episode(tmp_path, relative_path, split="train", hash_file=True)

    assert record.length == 6
    assert record.task == "insert_HDMI"
    assert record.joint_shape == (6, 9)
    assert len(record.sha256) == 64
    assert dict(record.image_shapes)["observation.images.top"] == (8, 10, 3)

    frames = list(iter_episode_frames(record, fps=10.0))
    assert len(frames) == 5
    np.testing.assert_array_equal(frames[0].state, np.arange(8) + 10.0)
    np.testing.assert_array_equal(frames[0].action, np.arange(9, 17) + 10.0)
    np.testing.assert_array_equal(frames[-1].state, np.arange(36, 44) + 10.0)
    np.testing.assert_array_equal(frames[-1].action, np.arange(45, 53) + 10.0)
    assert frames[0].source_frame_index == 0
    assert frames[0].source_row_index == 0
    assert frames[-1].source_timestamp == pytest.approx(0.4)
    assert not frames[0].action.flags.writeable


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        (
            np.asarray((0, 2, 2, 4, 6, 8), dtype=np.int64),
            "strictly increasing",
        ),
        (
            np.asarray((0, 2, 1, 4, 6, 8), dtype=np.int64),
            "strictly increasing",
        ),
        (
            np.asarray((-2, 0, 2, 4, 6, 8), dtype=np.int64),
            "non-negative",
        ),
    ],
    ids=("duplicate", "regression", "negative"),
)
def test_audit_rejects_invalid_source_steps(
    tmp_path: Path,
    steps: np.ndarray,
    message: str,
) -> None:
    relative_path = _write_episode(
        tmp_path,
        task="insert_HDMI",
        episode_id=0,
        offset=0.0,
    )
    with h5py.File(tmp_path / relative_path, "a") as handle:
        del handle["step"]
        handle.create_dataset("step", data=steps)

    with pytest.raises(ValueError, match=message):
        audit_episode(tmp_path, relative_path, split="train", hash_file=True)


def test_sampled_source_steps_keep_raw_provenance_and_row_timeline(
    tmp_path: Path,
) -> None:
    relative_path = _write_episode(
        tmp_path,
        task="insert_HDMI",
        episode_id=0,
        offset=0.0,
    )
    with h5py.File(tmp_path / relative_path, "a") as handle:
        handle["step"][:] = np.asarray((40, 42, 46, 48, 54, 56), dtype=np.int64)
    record = audit_episode(tmp_path, relative_path, split="train", hash_file=True)

    frames = list(iter_episode_frames(record, fps=10.0))

    assert frames[0].source_frame_index == 40
    assert frames[0].source_row_index == 0
    assert frames[0].source_timestamp == 0.0
    assert frames[-1].source_frame_index == 54
    assert frames[-1].source_timestamp == pytest.approx(0.4)
    assert frames[-1].timeline_timestamp == pytest.approx(0.4)
    np.testing.assert_array_equal(frames[0].action, np.arange(9, 17))


def test_pinned_temporal_prefix_excludes_reset_and_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_path = _write_episode(
        tmp_path,
        task="insert_HDMI",
        episode_id=7,
        offset=0.0,
    )
    record = audit_episode(tmp_path, relative_path, split="train", hash_file=True)
    source_path = tmp_path / relative_path
    with h5py.File(source_path, "a") as handle:
        handle["step"][:] = np.asarray((10, 11, 12, 13, 3, 4), dtype=np.int64)
    source_sha256 = sha256_file(source_path)
    test_contract = PinnedTemporalContract(
        relative_path=relative_path,
        sha256=source_sha256,
        raw_length=6,
        usable_start=0,
        usable_end=4,
        discontinuity_after_row=3,
        left_step=13,
        right_step=3,
    )
    monkeypatch.setattr(
        temporal_contract_module,
        "PINNED_TEMPORAL_CONTRACTS",
        (test_contract,),
    )
    record = replace(
        record,
        sha256=source_sha256,
        usable_start=0,
        usable_end=4,
        step_discontinuities_after_rows=(3,),
        temporal_policy=PINNED_MAXIMAL_PREFIX_POLICY,
    )

    frames = list(iter_episode_frames(record, fps=10.0))

    assert record.length == 6
    assert record.usable_length == 4
    assert record.converted_length == len(frames) == 3
    assert [frame.source_frame_index for frame in frames] == [10, 11, 12]
    np.testing.assert_array_equal(frames[-1].state, np.arange(18, 26))
    np.testing.assert_array_equal(frames[-1].action, np.arange(27, 35))

    validation_path = _write_episode(
        tmp_path,
        task="insert_HDMI",
        episode_id=0,
        offset=100.0,
    )
    validation_record = audit_episode(
        tmp_path,
        validation_path,
        split="validation",
        hash_file=True,
    )
    manifest = UniVTACDatasetManifest(
        data_root=str(tmp_path.resolve(strict=True)),
        entries=(record, validation_record),
        tasks=("insert_HDMI",),
    )
    normalizer = compute_qpos8_normalizer(manifest)
    assert normalizer.sample_count == 3

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_json_dict()),
        encoding="utf-8",
    )
    loaded = load_dataset_manifest(manifest_path, verify_sources=True)
    assert loaded.entries[0].usable_start == 0
    assert loaded.entries[0].usable_end == 4
    assert loaded.entries[0].step_discontinuities_after_rows == (3,)
    assert loaded.entries[0].temporal_policy == PINNED_MAXIMAL_PREFIX_POLICY


def test_temporal_exception_requires_exact_path_hash_and_boundary() -> None:
    steps = np.asarray((10, 11, 12, 13, 3, 4), dtype=np.int64)
    contract = PinnedTemporalContract(
        relative_path="insert_HDMI/clean/7.hdf5",
        sha256="a" * 64,
        raw_length=6,
        usable_start=0,
        usable_end=4,
        discontinuity_after_row=3,
        left_step=13,
        right_step=3,
    )

    selection = resolve_temporal_selection(
        steps,
        relative_path=contract.relative_path,
        source_sha256=contract.sha256,
        contracts=(contract,),
    )

    assert selection.usable_start == 0
    assert selection.usable_end == 4
    assert selection.converted_length == 3
    with pytest.raises(ValueError, match="restricted to train"):
        validate_declared_temporal_selection(
            relative_path=contract.relative_path,
            source_sha256=contract.sha256,
            raw_length=contract.raw_length,
            split="validation",
            selection=selection,
            contracts=(contract,),
        )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        resolve_temporal_selection(
            steps,
            relative_path=contract.relative_path,
            source_sha256="b" * 64,
            contracts=(contract,),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_temporal_selection(
            steps,
            relative_path="insert_HDMI/clean/8.hdf5",
            source_sha256=contract.sha256,
            contracts=(contract,),
        )


def test_manifest_and_normalizer_are_train_only(tmp_path: Path) -> None:
    train_paths = (
        _write_episode(
            tmp_path,
            task="insert_HDMI",
            episode_id=10,
            offset=0.0,
        ),
        _write_episode(
            tmp_path,
            task="lift_bottle",
            episode_id=10,
            offset=100.0,
        ),
    )
    validation_paths = (
        _write_episode(
            tmp_path,
            task="insert_HDMI",
            episode_id=0,
            offset=10_000.0,
        ),
        _write_episode(
            tmp_path,
            task="lift_bottle",
            episode_id=0,
            offset=20_000.0,
        ),
    )
    manifest = build_dataset_manifest(
        data_root=tmp_path,
        train_paths=train_paths,
        validation_paths=validation_paths,
        expected_task_counts={
            "train": {"insert_HDMI": 1, "lift_bottle": 1},
            "validation": {"insert_HDMI": 1, "lift_bottle": 1},
        },
    )
    normalizer = compute_qpos8_normalizer(manifest)

    assert manifest.manifest_sha256 == manifest.to_json_dict()["manifest_sha256"]
    assert normalizer.sample_count == 10
    assert max(normalizer.action_q99) < 1_000.0
    assert normalizer.source_manifest_sha256 == manifest.manifest_sha256
    assert len(normalizer.normalizer_sha256) == 64


def test_manifest_rejects_cross_split_content_overlap(tmp_path: Path) -> None:
    train_path = _write_episode(
        tmp_path,
        task="insert_HDMI",
        episode_id=10,
        offset=0.0,
    )
    source = tmp_path / train_path
    validation_path = "insert_HDMI/clean/0.hdf5"
    destination = tmp_path / validation_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="content overlap"):
        build_dataset_manifest(
            data_root=tmp_path,
            train_paths=(train_path,),
            validation_paths=(validation_path,),
        )


def test_manifest_consumers_reject_source_tampering(tmp_path: Path) -> None:
    train_path = _write_episode(
        tmp_path,
        task="insert_HDMI",
        episode_id=1,
        offset=0.0,
    )
    validation_path = _write_episode(
        tmp_path,
        task="insert_HDMI",
        episode_id=0,
        offset=100.0,
    )
    manifest = build_dataset_manifest(
        data_root=tmp_path,
        train_paths=(train_path,),
        validation_paths=(validation_path,),
    )
    train_record = next(entry for entry in manifest.entries if entry.split == "train")
    with h5py.File(tmp_path / train_path, "a") as handle:
        handle["embodiment/joint"][0, 0] = np.float32(1234.0)

    with pytest.raises(ValueError, match="source artifact .* changed after audit"):
        compute_qpos8_normalizer(manifest)
    with pytest.raises(ValueError, match="source artifact .* changed after audit"):
        list(iter_episode_frames(train_record, fps=10.0))


@pytest.mark.parametrize(
    ("storage_kind", "message"),
    [
        ("external_dataset", "external storage"),
        ("external_link", "ExternalLink"),
        ("virtual_dataset", "virtual dataset"),
    ],
)
def test_converter_rejects_indirect_hdf5_storage(
    tmp_path: Path,
    storage_kind: str,
    message: str,
) -> None:
    relative_path = _write_episode(
        tmp_path,
        task="insert_HDMI",
        episode_id=9,
        offset=0.0,
    )
    source_path = tmp_path / relative_path
    external_path = tmp_path / f"{storage_kind}.h5"
    with h5py.File(external_path, "w") as external:
        external.create_dataset("payload", data=np.asarray((7,), dtype=np.int64))
    with h5py.File(source_path, "a") as handle:
        if storage_kind == "external_dataset":
            joint = np.asarray(handle["embodiment/joint"][:], dtype=np.float32)
            del handle["embodiment/joint"]
            dataset = handle.create_dataset(
                "embodiment/joint",
                shape=joint.shape,
                dtype=joint.dtype,
                external=[
                    (
                        str(external_path.with_suffix(".raw")),
                        0,
                        joint.nbytes,
                    )
                ],
            )
            dataset[:] = joint
        elif storage_kind == "external_link":
            handle["indirect/external_link"] = h5py.ExternalLink(
                str(external_path),
                "/payload",
            )
        else:
            layout = h5py.VirtualLayout(shape=(1,), dtype=np.int64)
            layout[:] = h5py.VirtualSource(
                str(external_path),
                "/payload",
                shape=(1,),
            )
            handle.create_virtual_dataset("indirect/virtual_dataset", layout)
    record = audit_episode(tmp_path, relative_path, split="train", hash_file=True)

    with pytest.raises(ValueError, match=message):
        list(iter_episode_frames(record, fps=10.0))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_tactile", "Missing required HDF5 datasets"),
        ("unaligned", "Unaligned episode lengths"),
        ("nonfinite_joint", "non-finite qpos8"),
    ],
)
def test_audit_fails_closed_on_invalid_episode(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    relative_path = _write_episode(
        tmp_path,
        task="lift_bottle",
        episode_id=4,
        offset=0.0,
    )
    path = tmp_path / relative_path
    with h5py.File(path, "a") as handle:
        if mutation == "missing_tactile":
            del handle["tactile/right_gsmini/rgb_marker"]
        elif mutation == "unaligned":
            del handle["observation/wrist/rgb"]
            handle.create_dataset(
                "observation/wrist/rgb",
                data=np.zeros((5, 8, 10, 3), dtype=np.uint8),
            )
        else:
            handle["embodiment/joint"][2, 3] = np.nan

    with pytest.raises((KeyError, ValueError), match=message):
        audit_episode(tmp_path, relative_path, split="train", hash_file=False)


class _RecordingSink:
    def __init__(self) -> None:
        self.frames: list[tuple[dict[str, Any], str, float]] = []
        self.saved_episodes = 0

    def add_frame(self, frame: dict[str, Any], task: str, timestamp: float) -> None:
        self.frames.append((frame, task, timestamp))

    def save_episode(self) -> None:
        self.saved_episodes += 1


def test_lerobot_sink_contract_keeps_provenance(tmp_path: Path) -> None:
    relative_path = _write_episode(
        tmp_path,
        task="insert_HDMI",
        episode_id=3,
        offset=0.0,
    )
    record = audit_episode(tmp_path, relative_path, split="train", hash_file=True)
    features = build_lerobot_features(record)
    sink = _RecordingSink()

    converted = convert_episode_to_sink(record, sink=sink, fps=10.0)

    assert converted == 5
    assert sink.saved_episodes == 1
    assert len(sink.frames) == 5
    first_frame, task, timestamp = sink.frames[0]
    assert task == "insert_HDMI"
    assert timestamp == 0.0
    assert first_frame["source.relative_path"] == relative_path
    np.testing.assert_array_equal(first_frame["source.frame_index"], ((0,),))
    np.testing.assert_array_equal(first_frame["source.row_index"], ((0,),))
    np.testing.assert_array_equal(first_frame["source.timestamp"], ((0.0,),))
    np.testing.assert_array_equal(first_frame["action"], np.arange(9, 17))
    assert features["action"]["shape"] == (8,)
    assert features["source.frame_index"]["shape"] == (1, 1)
    assert features["source.row_index"]["shape"] == (1, 1)
    assert features["source.timestamp"]["shape"] == (1, 1)
    assert features["observation.images.tactile_a"]["dtype"] == "video"


def test_freeze_episode_action_config_is_atomic_and_idempotent(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "lerobot"
    meta_root = output_root / "meta"
    meta_root.mkdir(parents=True)
    episodes_path = meta_root / "episodes.jsonl"
    episodes_path.write_text(
        json.dumps(
            {
                "episode_index": 0,
                "tasks": ["insert_HDMI"],
                "length": 5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    freeze_episode_action_config(output_root)
    first_payload = episodes_path.read_bytes()
    freeze_episode_action_config(output_root)

    assert episodes_path.read_bytes() == first_payload
    record = json.loads(first_payload)
    assert record["action_config"] == [
        {
            "start_frame": 0,
            "end_frame": 5,
            "action_text": "insert_HDMI",
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_index", 0.0),
        ("episode_index", "0"),
        ("episode_index", True),
        ("length", 5.0),
        ("length", "5"),
        ("length", True),
    ],
)
def test_freeze_episode_action_config_rejects_coercive_integer_metadata(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    output_root = tmp_path / "lerobot"
    meta_root = output_root / "meta"
    meta_root.mkdir(parents=True)
    record: dict[str, object] = {
        "episode_index": 0,
        "tasks": ["insert_HDMI"],
        "length": 5,
    }
    record[field] = value
    episodes_path = meta_root / "episodes.jsonl"
    episodes_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid LeRobot episode metadata"):
        freeze_episode_action_config(output_root)


def test_split_derivation_filters_other_tasks_and_preserves_numeric_order(
    tmp_path: Path,
) -> None:
    paths = {
        _write_episode(
            tmp_path,
            task=task,
            episode_id=episode_id,
            offset=float(episode_id),
        )
        for task in ("insert_HDMI", "lift_bottle")
        for episode_id in (0, 2, 10)
    }
    validation_manifest = tmp_path / "validation.json"
    validation_manifest.write_text(
        json.dumps(
            [
                {"hdf5_path": "insert_HDMI/clean/0.hdf5", "task": "insert_HDMI"},
                {"hdf5_path": "lift_bottle/clean/0.hdf5", "task": "lift_bottle"},
                {"hdf5_path": "insert_hole/clean/0.hdf5", "task": "insert_hole"},
            ]
        ),
        encoding="utf-8",
    )

    split_paths = derive_split_paths(
        data_root=tmp_path,
        validation_manifest_path=validation_manifest,
    )

    assert set(split_paths["train"] + split_paths["validation"]) == paths
    assert split_paths["train"] == (
        "insert_HDMI/clean/2.hdf5",
        "insert_HDMI/clean/10.hdf5",
        "lift_bottle/clean/2.hdf5",
        "lift_bottle/clean/10.hdf5",
    )


def test_prepare_artifacts_writes_audited_contracts(tmp_path: Path) -> None:
    for task in ("insert_HDMI", "lift_bottle"):
        for episode_id in (0, 1, 2):
            _write_episode(
                tmp_path,
                task=task,
                episode_id=episode_id,
                offset=float(episode_id),
            )
    validation_manifest = tmp_path / "validation.json"
    validation_manifest.write_text(
        json.dumps(
            [
                {"hdf5_path": f"{task}/clean/0.hdf5", "task": task}
                for task in ("insert_HDMI", "lift_bottle")
            ]
        ),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    args = argparse.Namespace(
        data_root=tmp_path,
        validation_manifest=validation_manifest,
        quarantine_manifest=None,
        artifact_dir=artifact_dir,
        materialize_root=None,
        repo_id="unused",
        fps=10,
        expected_train_per_task=2,
        expected_validation_per_task=1,
        image_writer_threads=1,
    )

    report = prepare_artifacts(args)

    manifest_payload = json.loads(
        (artifact_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    normalizer_payload = json.loads(
        (artifact_dir / "qpos8_normalizer.json").read_text(encoding="utf-8")
    )
    assert report["split_counts"] == {"train": 4, "validation": 2}
    assert manifest_payload["manifest_sha256"] == report["manifest_sha256"]
    assert (
        manifest_payload["source_image_encoding_contract"]
        == SOURCE_IMAGE_ENCODING_CONTRACT
    )
    assert manifest_payload["output_color_space"] == OUTPUT_COLOR_SPACE
    assert normalizer_payload["normalizer_sha256"] == report["normalizer_sha256"]
    assert normalizer_payload["sample_count"] == 20

    verified = verify_track31_artifact_pair(
        manifest_path=artifact_dir / "dataset_manifest.json",
        normalizer_path=artifact_dir / "qpos8_normalizer.json",
    )
    assert verified.train_episode_count == 4
    assert verified.validation_episode_count == 2
    assert verified.normalizer_sample_count == 20

    dataset_root = tmp_path / "lerobot"
    conversions: dict[str, object] = {}
    for split in ("train", "validation"):
        split_root = dataset_root / split
        (split_root / "meta").mkdir(parents=True)
        (split_root / "data").mkdir()
        (split_root / "videos" / "chunk-000").mkdir(parents=True)
        (split_root / "meta" / "info.json").write_text(
            json.dumps({"split": split}), encoding="utf-8"
        )
        entries = [
            entry for entry in manifest_payload["entries"] if entry["split"] == split
        ]
        episode_lines: list[str] = []
        episode_indices: list[int] = []
        frame_indices: list[int] = []
        source_paths: list[str] = []
        source_steps: list[int] = []
        for episode_index, entry in enumerate(entries):
            usable_start, usable_end = entry["usable_source_range"]
            converted_length = int(usable_end) - int(usable_start) - 1
            episode_lines.append(
                json.dumps(
                    {
                        "episode_index": episode_index,
                        "length": converted_length,
                        "tasks": [entry["task"]],
                        "action_config": [
                            {
                                "start_frame": 0,
                                "end_frame": converted_length,
                                "action_text": entry["task"],
                            }
                        ],
                    }
                )
            )
            for frame_index in range(converted_length):
                episode_indices.append(episode_index)
                frame_indices.append(frame_index)
                source_paths.append(entry["relative_path"])
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
        for feature_name, _ in IMAGE_PATHS:
            feature_root = split_root / "videos" / "chunk-000" / feature_name
            feature_root.mkdir()
            for episode_index in range(len(entries)):
                (feature_root / f"episode_{episode_index:06d}.mp4").write_bytes(
                    f"fixed-{split}-{feature_name}-{episode_index}".encode("utf-8")
                )
        conversions[split] = {
            "schema_version": 2,
            "action_schema": "qpos8_next_step",
            "output_root": str(split_root),
            "episode_count": len(entries),
            "frame_count": sum(
                int(entry["usable_source_range"][1])
                - int(entry["usable_source_range"][0])
                - 1
                for entry in entries
            ),
            "source_image_encoding_contract": SOURCE_IMAGE_ENCODING_CONTRACT,
            "output_color_space": OUTPUT_COLOR_SPACE,
            "source_relative_paths": [entry["relative_path"] for entry in entries],
            "source_sha256": [entry["sha256"] for entry in entries],
            "temporal_contract": "content_addressed_source_range_v1",
            "source_temporal_selections": [
                {
                    "relative_path": entry["relative_path"],
                    "raw_length": entry["length"],
                    "usable_source_range": entry["usable_source_range"],
                    "converted_length": (
                        int(entry["usable_source_range"][1])
                        - int(entry["usable_source_range"][0])
                        - 1
                    ),
                    "dropped_prefix_rows": entry["usable_source_range"][0],
                    "dropped_suffix_rows": (
                        int(entry["length"]) - int(entry["usable_source_range"][1])
                    ),
                    "step_discontinuities_after_rows": entry[
                        "step_discontinuities_after_rows"
                    ],
                    "temporal_policy": entry["temporal_policy"],
                }
                for entry in entries
            ],
            "table_inventory": build_lerobot_table_inventory(split_root),
        }
    conversion_report = {
        "schema_version": 2,
        "source_manifest_sha256": manifest_payload["manifest_sha256"],
        "source_image_encoding_contract": SOURCE_IMAGE_ENCODING_CONTRACT,
        "output_color_space": OUTPUT_COLOR_SPACE,
        "conversions": conversions,
    }
    conversion_report["conversion_report_sha256"] = _canonical_sha256(conversion_report)
    conversion_path = artifact_dir / "conversion_report.json"
    conversion_path.write_text(json.dumps(conversion_report), encoding="utf-8")

    training_bundle = verify_track31_training_bundle(
        manifest_path=artifact_dir / "dataset_manifest.json",
        normalizer_path=artifact_dir / "qpos8_normalizer.json",
        conversion_report_path=conversion_path,
        dataset_root=dataset_root,
    )
    assert (
        training_bundle.conversion_report_sha256
        == conversion_report["conversion_report_sha256"]
    )

    (dataset_root / "train" / "data" / "chunk.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="table inventory changed"):
        verify_track31_training_bundle(
            manifest_path=artifact_dir / "dataset_manifest.json",
            normalizer_path=artifact_dir / "qpos8_normalizer.json",
            conversion_report_path=conversion_path,
            dataset_root=dataset_root,
        )


def test_artifact_pair_rejects_posthoc_normalizer_edit(tmp_path: Path) -> None:
    for task in ("insert_HDMI", "lift_bottle"):
        for episode_id in (0, 1):
            _write_episode(
                tmp_path,
                task=task,
                episode_id=episode_id,
                offset=float(episode_id),
            )
    validation_manifest = tmp_path / "validation.json"
    validation_manifest.write_text(
        json.dumps(
            [
                {"hdf5_path": f"{task}/clean/0.hdf5", "task": task}
                for task in ("insert_HDMI", "lift_bottle")
            ]
        ),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    prepare_artifacts(
        argparse.Namespace(
            data_root=tmp_path,
            validation_manifest=validation_manifest,
            quarantine_manifest=None,
            artifact_dir=artifact_dir,
            materialize_root=None,
            repo_id="unused",
            fps=10,
            expected_train_per_task=1,
            expected_validation_per_task=1,
            image_writer_threads=1,
        )
    )
    normalizer_path = artifact_dir / "qpos8_normalizer.json"
    payload = json.loads(normalizer_path.read_text(encoding="utf-8"))
    payload["action_q01"][0] += 1.0
    normalizer_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="normalizer sha256"):
        verify_track31_artifact_pair(
            manifest_path=artifact_dir / "dataset_manifest.json",
            normalizer_path=normalizer_path,
        )


def test_real_lerobot_033_materialization_and_reload(tmp_path: Path) -> None:
    pytest.importorskip(
        "lerobot", reason="LeRobot is an optional conversion dependency"
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    relative_path = _write_episode(
        tmp_path,
        task="insert_HDMI",
        episode_id=0,
        offset=0.0,
        height=128,
        width=128,
    )
    record = audit_episode(tmp_path, relative_path, split="train", hash_file=True)
    output_root = tmp_path / "lerobot"

    report = write_lerobot_dataset(
        (record,),
        output_root=str(output_root),
        repo_id="local/univtac_track31_test",
        fps=10,
        image_writer_threads=1,
    )
    reloaded = LeRobotDataset(
        repo_id="local/univtac_track31_test",
        root=output_root,
        revision=None,
        force_cache_sync=False,
    )

    assert report["episode_count"] == 1
    assert report["frame_count"] == 5
    assert report["table_inventory"]
    assert reloaded.num_episodes == 1
    assert reloaded.num_frames == 5
    assert tuple(reloaded.hf_dataset[0]["source.frame_index"].shape) == (1, 1)
    assert tuple(reloaded.hf_dataset[0]["source.row_index"].shape) == (1, 1)
    assert tuple(reloaded.hf_dataset[0]["source.timestamp"].shape) == (1, 1)
    episode_record = json.loads(
        (output_root / "meta" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert episode_record["action_config"] == [
        {
            "start_frame": 0,
            "end_frame": 5,
            "action_text": "insert_HDMI",
        }
    ]
