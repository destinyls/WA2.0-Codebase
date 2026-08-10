# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import pytest

from n0_twam.integrations.univtac.dataset_view import build_standard_dataset_views
from n0_twam.integrations.univtac.manifest import (
    MANIFEST_SCHEMA_VERSION,
    UniVTACDatasetManifest,
    build_dataset_manifest,
    load_dataset_manifest,
)
from n0_twam.integrations.univtac.schema import (
    CANONICAL_TASK_PROMPT_MAP,
    TRACK31_TARGET_TASKS,
    TRACK31_TASKS,
    UNIVTAC_ALL_TASKS,
    UniVTACEpisodeRecord,
)


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record(root: Path, relative_path: str, *, split: str) -> UniVTACEpisodeRecord:
    absolute_path = root / relative_path
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(absolute_path, "w") as handle:
        handle.create_dataset("step", data=list(range(45)))
        handle.attrs["fixture_identity"] = relative_path
    return UniVTACEpisodeRecord(
        relative_path=relative_path,
        absolute_path=absolute_path.resolve(strict=True),
        task=Path(relative_path).parts[0],
        split=split,
        length=45,
        length_source="embodiment/joint.shape[0]",
        joint_shape=(45, 9),
        image_shapes=(("observation.images.top", (8, 10, 3)),),
        size_bytes=absolute_path.stat().st_size,
        sha256=hashlib.sha256(absolute_path.read_bytes()).hexdigest(),
    )


def test_schema_exposes_eight_tasks_without_changing_legacy_alias() -> None:
    assert UNIVTAC_ALL_TASKS == (
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    )
    assert TRACK31_TARGET_TASKS == ("insert_HDMI", "lift_bottle")
    assert TRACK31_TASKS is TRACK31_TARGET_TASKS
    assert tuple(CANONICAL_TASK_PROMPT_MAP) == UNIVTAC_ALL_TASKS
    with pytest.raises(TypeError):
        CANONICAL_TASK_PROMPT_MAP["insert_HDMI"] = "changed"  # type: ignore[index]


def test_v3_manifest_has_explicit_task_scope_and_entry_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {
        ("insert_hole/clean/10.hdf5", "train"): _record(
            tmp_path, "insert_hole/clean/10.hdf5", split="train"
        ),
        ("insert_hole/clean/0.hdf5", "validation"): _record(
            tmp_path, "insert_hole/clean/0.hdf5", split="validation"
        ),
    }

    def fake_audit_episode(
        data_root: Path,
        relative_path: str,
        *,
        split: str,
        hash_file: bool,
        allowed_tasks: tuple[str, ...],
    ) -> UniVTACEpisodeRecord:
        assert data_root == tmp_path
        assert hash_file is True
        assert allowed_tasks == ("insert_hole",)
        return records[(relative_path, split)]

    monkeypatch.setattr(
        "n0_twam.integrations.univtac.hdf5_reader.audit_episode",
        fake_audit_episode,
    )
    manifest = build_dataset_manifest(
        data_root=tmp_path,
        train_paths=("insert_hole/clean/10.hdf5",),
        validation_paths=("insert_hole/clean/0.hdf5",),
        tasks=("insert_hole",),
        expected_task_counts={
            "train": {"insert_hole": 1},
            "validation": {"insert_hole": 1},
        },
    )

    payload = manifest.to_json_dict()
    assert MANIFEST_SCHEMA_VERSION == payload["schema_version"] == 4
    assert payload["tasks"] == ["insert_hole"]
    assert payload["task_counts"]["train"] == {"insert_hole": 1}
    assert payload["entries"][0]["episode_id"] == 10
    assert payload["entries"][0]["realpath"].endswith("insert_hole/clean/10.hdf5")


@pytest.mark.parametrize("schema_version", (1, 2, 3))
def test_legacy_manifest_is_validated_read_only(
    tmp_path: Path,
    schema_version: int,
) -> None:
    record = _record(
        tmp_path,
        "insert_HDMI/clean/10.hdf5",
        split="train",
    )
    validation = _record(
        tmp_path,
        "insert_HDMI/clean/0.hdf5",
        split="validation",
    )
    legacy_entries = []
    for entry in (record, validation):
        item = entry.to_json_dict()
        if schema_version < 3:
            item.pop("episode_id")
            item.pop("realpath")
        item.pop("usable_source_range")
        item.pop("step_discontinuities_after_rows")
        item.pop("temporal_policy")
        legacy_entries.append(item)
    canonical = {
        "schema_version": schema_version,
        "action_schema": "qpos8_next_step",
        "source_image_encoding_contract": "opencv_imencode_rgb_input_v1",
        "output_color_space": "RGB",
        "tasks": ["insert_HDMI"],
        "entries": legacy_entries,
    }
    payload = {
        **canonical,
        "data_root": str(tmp_path.resolve(strict=True)),
        "manifest_sha256": _sha256(canonical),
    }
    path = tmp_path / f"manifest_v{schema_version}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_dataset_manifest(path, verify_sources=True)

    assert loaded.schema_version == schema_version
    assert loaded.is_read_only
    assert loaded.manifest_sha256 == payload["manifest_sha256"]
    assert loaded.entries[0].episode_id == 10
    with pytest.raises(ValueError, match="read-only"):
        loaded.to_json_dict()
    with pytest.raises(ValueError, match="read-only"):
        replace(loaded, schema_version=MANIFEST_SCHEMA_VERSION)
    with pytest.raises(ValueError, match="read-only"):
        replace(loaded, entries=tuple(reversed(loaded.entries)))
    with pytest.raises((TypeError, ValueError)):
        replace(
            loaded,
            schema_version=MANIFEST_SCHEMA_VERSION,
            _origin_schema_version=None,
            _persisted_manifest_sha256=None,
            _read_only_identity_sha256=None,
        )
    with pytest.raises(ValueError, match="schema-v4"):
        build_standard_dataset_views(loaded, verify_sources=False)


def test_legacy_manifest_rejects_hash_edit(tmp_path: Path) -> None:
    record = _record(
        tmp_path,
        "insert_HDMI/clean/10.hdf5",
        split="train",
    )
    item = record.to_json_dict()
    item.pop("episode_id")
    item.pop("realpath")
    canonical = {
        "schema_version": 2,
        "action_schema": "qpos8_next_step",
        "source_image_encoding_contract": "opencv_imencode_rgb_input_v1",
        "output_color_space": "RGB",
        "tasks": ["insert_HDMI"],
        "entries": [item],
    }
    payload = {
        **canonical,
        "data_root": str(tmp_path),
        "manifest_sha256": "0" * 64,
    }
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256"):
        load_dataset_manifest(path, verify_sources=False)


def test_manifest_rejects_duplicate_content_within_one_split(tmp_path: Path) -> None:
    first = _record(tmp_path, "insert_HDMI/clean/10.hdf5", split="train")
    second = _record(tmp_path, "insert_HDMI/clean/11.hdf5", split="train")
    validation = _record(
        tmp_path,
        "insert_HDMI/clean/0.hdf5",
        split="validation",
    )

    with pytest.raises(ValueError, match="content SHA-256 overlap"):
        UniVTACDatasetManifest(
            data_root=str(tmp_path),
            entries=(
                first,
                replace(second, sha256=first.sha256),
                validation,
            ),
            tasks=("insert_HDMI",),
        )


def test_manifest_rejects_duplicate_content_across_split_and_task(
    tmp_path: Path,
) -> None:
    train = _record(tmp_path, "insert_HDMI/clean/10.hdf5", split="train")
    validation = _record(
        tmp_path,
        "lift_bottle/clean/0.hdf5",
        split="validation",
    )

    with pytest.raises(ValueError, match="content SHA-256 overlap"):
        UniVTACDatasetManifest(
            data_root=str(tmp_path),
            entries=(train, replace(validation, sha256=train.sha256)),
            tasks=("insert_HDMI", "lift_bottle"),
        )
