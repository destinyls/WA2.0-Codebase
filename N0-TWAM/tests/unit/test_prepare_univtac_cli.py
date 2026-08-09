# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from n0_twam.integrations.univtac.dataset_view import (
    DatasetView,
    DatasetViewEntry,
)
from n0_twam.integrations.univtac.manifest import UniVTACDatasetManifest
from n0_twam.integrations.univtac.normalizer import Qpos8Normalizer
from n0_twam.integrations.univtac.schema import (
    TRACK31_TARGET_TASKS,
    UNIVTAC_ALL_TASKS,
)
from script.track3_1.prepare_univtac import (
    _conversion_split_specs,
    _materialize_lerobot_splits,
    _parse_args,
    _prepare_contract,
    _write_standard_profile_normalizers,
)


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--data-root",
        str(tmp_path / "source"),
        "--validation-manifest",
        str(tmp_path / "validation.json"),
        "--artifact-dir",
        str(tmp_path / "artifacts"),
    ]


def test_cli_default_preserves_legacy_two_task_scope(tmp_path: Path) -> None:
    args = _parse_args(_required_args(tmp_path))

    tasks, counts, emit_standard_views = _prepare_contract(args)

    assert tasks == TRACK31_TARGET_TASKS
    assert counts == {
        "train": {task: 95 for task in TRACK31_TARGET_TASKS},
        "validation": {task: 5 for task in TRACK31_TARGET_TASKS},
        "quarantine": {task: 0 for task in TRACK31_TARGET_TASKS},
    }
    assert emit_standard_views is False
    assert _conversion_split_specs(False) == (
        ("train", "train"),
        ("validation", "validation"),
    )


def test_cli_explicit_legacy_mode_preserves_two_task_scope(tmp_path: Path) -> None:
    args = _parse_args([*_required_args(tmp_path), "--legacy-two-task"])

    tasks, _, emit_standard_views = _prepare_contract(args)

    assert tasks == TRACK31_TARGET_TASKS
    assert emit_standard_views is False


def test_cli_standard_mode_is_explicit_and_uses_physical_split_names(
    tmp_path: Path,
) -> None:
    args = _parse_args([*_required_args(tmp_path), "--emit-standard-views"])

    tasks, counts, emit_standard_views = _prepare_contract(args)

    assert tasks == UNIVTAC_ALL_TASKS
    assert counts["train"]["grasp_classify"] == 94
    assert counts["train"]["insert_HDMI"] == 95
    assert counts["validation"] == {task: 5 for task in UNIVTAC_ALL_TASKS}
    assert emit_standard_views is True
    assert _conversion_split_specs(True) == (
        ("train", "train759"),
        ("validation", "frozen40"),
    )


def test_cli_supports_explicit_custom_task_scope(tmp_path: Path) -> None:
    args = _parse_args(
        [
            *_required_args(tmp_path),
            "--tasks",
            "insert_hole",
            "--expected-train-counts",
            '{"insert_hole": 7}',
            "--expected-validation-counts",
            '{"insert_hole": 2}',
            "--expected-quarantine-counts",
            '{"insert_hole": 0}',
        ]
    )

    tasks, counts, emit_standard_views = _prepare_contract(args)

    assert tasks == ("insert_hole",)
    assert counts["train"] == {"insert_hole": 7}
    assert emit_standard_views is False


def test_cli_explicit_two_task_list_does_not_enable_standard_views(
    tmp_path: Path,
) -> None:
    args = _parse_args(
        [
            *_required_args(tmp_path),
            "--tasks",
            *TRACK31_TARGET_TASKS,
            "--expected-train-counts",
            '{"insert_HDMI": 95, "lift_bottle": 95}',
            "--expected-validation-counts",
            '{"insert_HDMI": 5, "lift_bottle": 5}',
            "--expected-quarantine-counts",
            '{"insert_HDMI": 0, "lift_bottle": 0}',
        ]
    )

    tasks, _, emit_standard_views = _prepare_contract(args)

    assert tasks == TRACK31_TARGET_TASKS
    assert emit_standard_views is False


def test_standard_mode_rejects_nonstandard_task_scope(tmp_path: Path) -> None:
    args = _parse_args(
        [
            *_required_args(tmp_path),
            "--tasks",
            "insert_HDMI",
            "lift_bottle",
            "--emit-standard-views",
        ]
    )

    with pytest.raises(ValueError, match="eight-task scope"):
        _prepare_contract(args)


def test_standard_mode_writes_stage_a_profile_normalizers_for_stage_b_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = DatasetViewEntry(
        relative_path="insert_HDMI/clean/10.hdf5",
        realpath=str(tmp_path / "insert_HDMI/clean/10.hdf5"),
        source_sha256="a" * 64,
        task="insert_HDMI",
        source_split="train",
        source_episode_id=10,
        lerobot_episode_id=0,
    )

    def view(view_id: str, selection_method: str) -> DatasetView:
        return DatasetView(
            view_id=view_id,
            role="training",
            physical_split="train759",
            source_manifest_sha256="b" * 64,
            tasks=("insert_HDMI",),
            entries=(entry,),
            selection_method=selection_method,
        )

    views = {
        "stage_a_dev719_v1": view("stage_a_dev719_v1", "dev"),
        "stage_a_final759_v1": view("stage_a_final759_v1", "final"),
    }
    values = (0.0,) * 8
    legacy = Qpos8Normalizer(
        action_q01=values,
        action_q99=values,
        state_q01=values,
        state_q99=values,
        observed_action_min=values,
        observed_action_max=values,
        sample_count=1,
        train_paths_sha256="c" * 64,
        source_manifest_sha256="b" * 64,
    )
    compute_calls: list[str] = []

    def fake_compute(
        manifest: UniVTACDatasetManifest,
        *,
        view: DatasetView | None = None,
    ) -> Qpos8Normalizer:
        del manifest
        assert view is not None
        compute_calls.append(view.view_id)
        return replace(
            legacy,
            source_view_id=view.view_id,
            source_view_sha256=view.view_sha256,
        )

    monkeypatch.setattr(
        "script.track3_1.prepare_univtac.compute_qpos8_normalizer",
        fake_compute,
    )

    hashes = _write_standard_profile_normalizers(
        artifact_dir=tmp_path / "artifacts",
        manifest=object(),  # type: ignore[arg-type]
        views=views,
        legacy_train759_normalizer=legacy,
    )

    assert compute_calls == ["stage_a_dev719_v1"]
    assert set(hashes) == {"qpos8_dev719_v1", "qpos8_final759_v1"}
    for normalizer_id, source_view_id in (
        ("qpos8_dev719_v1", "stage_a_dev719_v1"),
        ("qpos8_final759_v1", "stage_a_final759_v1"),
    ):
        path = tmp_path / "artifacts/normalizers" / f"{normalizer_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2
        assert payload["source_view_id"] == source_view_id
        assert payload["normalizer_sha256"] == hashes[normalizer_id]


def test_standard_materialization_uses_759_and_40_physical_repositories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = (
        tuple(SimpleNamespace(split="train", marker=index) for index in range(759))
        + tuple(
            SimpleNamespace(split="validation", marker=index) for index in range(40)
        )
        + (SimpleNamespace(split="quarantine", marker=0),)
    )
    calls: list[tuple[int, str, str]] = []

    def fake_write(
        records: tuple[SimpleNamespace, ...],
        *,
        output_root: str,
        repo_id: str,
        fps: int,
        image_writer_threads: int,
    ) -> dict[str, object]:
        assert fps == 10
        assert image_writer_threads == 3
        calls.append((len(records), output_root, repo_id))
        return {"episode_count": len(records), "output_root": output_root}

    monkeypatch.setattr(
        "script.track3_1.prepare_univtac.write_lerobot_dataset",
        fake_write,
    )

    conversions = _materialize_lerobot_splits(
        manifest=SimpleNamespace(entries=entries),  # type: ignore[arg-type]
        materialize_root=tmp_path / "lerobot",
        repo_id="unit",
        fps=10,
        image_writer_threads=3,
        emit_standard_views=True,
    )

    assert calls == [
        (759, str(tmp_path / "lerobot/train759"), "unit_train759"),
        (40, str(tmp_path / "lerobot/frozen40"), "unit_frozen40"),
    ]
    assert set(conversions) == {"train759", "frozen40"}
