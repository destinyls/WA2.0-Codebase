# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from script.track3_1.materialize_univtac import (
    INCOMPLETE_MARKER_NAME,
    MaterializeConfig,
    _canonical_sha256,
    _parse_args,
    materialize_from_artifacts,
)


def _config(tmp_path: Path) -> MaterializeConfig:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "universe_manifest_v4.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    return MaterializeConfig(
        artifact_dir=artifact_dir,
        target_root=tmp_path / "lerobot",
        repo_id="unit",
        fps=10,
        image_writer_threads=3,
    )


def _manifest(
    *,
    quarantine_path: str = "grasp_classify/clean/90.hdf5",
    quarantine_length: int = 43,
) -> SimpleNamespace:
    tasks = (
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    )
    task_counts = {
        "train": {task: 94 if task == "grasp_classify" else 95 for task in tasks},
        "validation": {task: 5 for task in tasks},
        "quarantine": {task: 1 if task == "grasp_classify" else 0 for task in tasks},
    }
    entries = tuple(
        SimpleNamespace(
            split=split,
            relative_path=(quarantine_path if split == "quarantine" else "unused"),
            length=(quarantine_length if split == "quarantine" else 45),
        )
        for split, count in (("train", 759), ("validation", 40), ("quarantine", 1))
        for _ in range(count)
    )
    return SimpleNamespace(
        schema_version=4,
        manifest_sha256="a" * 64,
        tasks=tasks,
        task_counts=task_counts,
        entries=entries,
    )


def _fake_materializer(
    *,
    manifest: object,
    materialize_root: Path,
    repo_id: str,
    fps: int,
    image_writer_threads: int,
    emit_standard_views: bool,
) -> dict[str, object]:
    del manifest
    assert repo_id == "unit"
    assert fps == 10
    assert image_writer_threads == 3
    assert emit_standard_views is True
    conversions: dict[str, object] = {}
    for split, count in (("train759", 759), ("frozen40", 40)):
        split_root = materialize_root / split
        split_root.mkdir(parents=True)
        conversions[split] = {
            "schema_version": 2,
            "output_root": str(split_root),
            "episode_count": count,
        }
    return conversions


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _staging_roots(config: MaterializeConfig) -> list[Path]:
    return sorted(
        config.target_root.parent.glob(f".{config.target_root.name}.incomplete-*")
    )


def test_parser_requires_artifact_and_target_roots(tmp_path: Path) -> None:
    args = _parse_args(
        [
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--target-root",
            str(tmp_path / "lerobot"),
        ]
    )

    assert args.repo_id == "univtac_track31"
    assert args.fps == 10
    assert args.image_writer_threads == 8


def test_success_uses_staging_then_publishes_final_root_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    old_report = b'{"old": true}\n'
    report_path = config.artifact_dir / "conversion_report.json"
    report_path.write_bytes(old_report)
    load_calls: list[bool] = []
    verify_calls: list[tuple[str, Path, str]] = []

    def fake_load(path: Path, *, verify_sources: bool) -> SimpleNamespace:
        assert path == config.artifact_dir / "universe_manifest_v4.json"
        load_calls.append(verify_sources)
        return _manifest()

    def fake_verify(
        *,
        manifest_path: Path,
        conversion_report_path: Path,
        dataset_root: Path,
        physical_split: str,
        allow_transaction_state: bool,
    ) -> SimpleNamespace:
        assert allow_transaction_state is True
        assert manifest_path == config.artifact_dir / "universe_manifest_v4.json"
        report = _read_json(conversion_report_path)
        reported_root = Path(report["conversions"][physical_split]["output_root"])
        assert dataset_root == reported_root
        assert dataset_root.is_dir()
        verify_calls.append((physical_split, dataset_root, conversion_report_path.name))
        return SimpleNamespace(
            episode_count=759 if physical_split == "train759" else 40
        )

    monkeypatch.setattr(
        "script.track3_1.materialize_univtac.load_dataset_manifest",
        fake_load,
    )
    monkeypatch.setattr(
        "script.track3_1.materialize_univtac._materialize_lerobot_splits",
        _fake_materializer,
    )
    monkeypatch.setattr(
        "script.track3_1.materialize_report.verify_track31_physical_bundle",
        fake_verify,
    )

    result = materialize_from_artifacts(config)

    assert load_calls == [False]
    assert result["status"] == "materialized"
    assert result["split_counts"] == {"train759": 759, "frozen40": 40}
    assert config.target_root.is_dir()
    assert not _staging_roots(config)
    marker = _read_json(config.target_root / INCOMPLETE_MARKER_NAME)
    assert marker["status"] == "complete"
    assert marker["phase"] == "complete"
    report = _read_json(report_path)
    digest = report.pop("conversion_report_sha256")
    assert digest == _canonical_sha256(report)
    assert report["source_manifest_sha256"] == "a" * 64
    for split in ("train759", "frozen40"):
        assert report["conversions"][split]["output_root"] == str(
            config.target_root / split
        )
    assert [item[0] for item in verify_calls] == [
        "train759",
        "frozen40",
        "train759",
        "frozen40",
    ]
    assert all(path.parent == config.target_root for _, path, _ in verify_calls[-2:])
    assert report_path.read_bytes() != old_report


def test_existing_target_is_rejected_without_touching_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.target_root.mkdir()
    report_path = config.artifact_dir / "conversion_report.json"
    old_report = b"immutable-old-report\n"
    report_path.write_bytes(old_report)
    monkeypatch.setattr(
        "script.track3_1.materialize_univtac.load_dataset_manifest",
        lambda *args, **kwargs: pytest.fail("target guard ran too late"),
    )

    with pytest.raises(FileExistsError, match="target root must not exist"):
        materialize_from_artifacts(config)

    assert report_path.read_bytes() == old_report
    assert not _staging_roots(config)


def test_dangling_target_symlink_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    old_report = b"immutable-old-report\n"
    report_path.write_bytes(old_report)
    config.target_root.symlink_to(tmp_path / "missing-target")

    with pytest.raises(FileExistsError, match="target root must not exist"):
        materialize_from_artifacts(config)

    assert config.target_root.is_symlink()
    assert report_path.read_bytes() == old_report
    assert not _staging_roots(config)


def test_conversion_failure_preserves_incomplete_staging_and_old_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    old_report = b"immutable-old-report\n"
    report_path.write_bytes(old_report)

    monkeypatch.setattr(
        "script.track3_1.materialize_univtac.load_dataset_manifest",
        lambda path, *, verify_sources: _manifest(),
    )

    def fail_materialization(**kwargs: object) -> dict[str, object]:
        staging_root = kwargs["materialize_root"]
        assert isinstance(staging_root, Path)
        (staging_root / "train759").mkdir()
        raise RuntimeError("conversion exploded")

    monkeypatch.setattr(
        "script.track3_1.materialize_univtac._materialize_lerobot_splits",
        fail_materialization,
    )

    with pytest.raises(RuntimeError, match="conversion exploded"):
        materialize_from_artifacts(config)

    assert not config.target_root.exists()
    assert report_path.read_bytes() == old_report
    staging_roots = _staging_roots(config)
    assert len(staging_roots) == 1
    marker = _read_json(staging_roots[0] / INCOMPLETE_MARKER_NAME)
    assert marker["status"] == "incomplete"
    assert marker["phase"] == "materializing"
    assert marker["error_type"] == "RuntimeError"
    assert marker["error_message"] == "conversion exploded"


def test_final_verification_failure_rolls_target_back_to_incomplete_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    old_report = b"immutable-old-report\n"
    report_path.write_bytes(old_report)
    verification_count = 0

    monkeypatch.setattr(
        "script.track3_1.materialize_univtac.load_dataset_manifest",
        lambda path, *, verify_sources: _manifest(),
    )
    monkeypatch.setattr(
        "script.track3_1.materialize_univtac._materialize_lerobot_splits",
        _fake_materializer,
    )

    def fail_final_verify(**kwargs: object) -> SimpleNamespace:
        nonlocal verification_count
        verification_count += 1
        if verification_count == 3:
            raise ValueError("final identity mismatch")
        episode_count = 759 if kwargs["physical_split"] == "train759" else 40
        return SimpleNamespace(episode_count=episode_count)

    monkeypatch.setattr(
        "script.track3_1.materialize_report.verify_track31_physical_bundle",
        fail_final_verify,
    )

    with pytest.raises(ValueError, match="final identity mismatch"):
        materialize_from_artifacts(config)

    assert verification_count == 3
    assert not config.target_root.exists()
    assert report_path.read_bytes() == old_report
    staging_roots = _staging_roots(config)
    assert len(staging_roots) == 1
    marker = _read_json(staging_roots[0] / INCOMPLETE_MARKER_NAME)
    assert marker["status"] == "incomplete"
    assert marker["phase"] == "recovered_to_staging"
    assert marker["error_message"] == "final identity mismatch"
    assert not list(config.artifact_dir.glob(".conversion_report.json.candidate-*"))


def test_report_publish_failure_rolls_back_and_preserves_old_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    old_report = b"immutable-old-report\n"
    report_path.write_bytes(old_report)
    from script.track3_1 import materialize_univtac as materialize_module

    monkeypatch.setattr(
        "script.track3_1.materialize_univtac.load_dataset_manifest",
        lambda path, *, verify_sources: _manifest(),
    )
    monkeypatch.setattr(
        "script.track3_1.materialize_univtac._materialize_lerobot_splits",
        _fake_materializer,
    )
    monkeypatch.setattr(
        "script.track3_1.materialize_report.verify_track31_physical_bundle",
        lambda **kwargs: SimpleNamespace(
            episode_count=(759 if kwargs["physical_split"] == "train759" else 40)
        ),
    )

    def fail_report_publish(**kwargs: object) -> object:
        assert kwargs["report_path"] == report_path
        raise OSError("report publish failed")

    monkeypatch.setattr(
        materialize_module,
        "publish_report_cas",
        fail_report_publish,
    )

    with pytest.raises(OSError, match="report publish failed"):
        materialize_from_artifacts(config)

    assert not config.target_root.exists()
    assert report_path.read_bytes() == old_report
    staging_roots = _staging_roots(config)
    assert len(staging_roots) == 1
    marker = _read_json(staging_roots[0] / INCOMPLETE_MARKER_NAME)
    assert marker["phase"] == "recovered_to_staging"
    assert marker["error_message"] == "report publish failed"
    assert not list(config.artifact_dir.glob(".conversion_report.json.candidate-*"))


def test_nonformal_manifest_is_rejected_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    legacy = _manifest()
    legacy.schema_version = 3
    monkeypatch.setattr(
        "script.track3_1.materialize_univtac.load_dataset_manifest",
        lambda path, *, verify_sources: legacy,
    )

    with pytest.raises(ValueError, match="schema-v4"):
        materialize_from_artifacts(config)

    assert not config.target_root.exists()
    assert not _staging_roots(config)
