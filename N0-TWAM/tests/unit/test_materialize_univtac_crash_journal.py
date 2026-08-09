# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import os
from pathlib import Path

import pytest

from script.track3_1 import (
    materialize_cleanup,
    materialize_publication_journal,
    materialize_transaction,
)
from script.track3_1 import materialize_univtac as materialize_module
from script.track3_1 import (
    materialize_writeahead,
)
from script.track3_1.materialize_univtac import (
    INCOMPLETE_MARKER_NAME,
    REPORT_CANDIDATE_PREFIX,
    materialize_from_artifacts,
)
from tests.unit.test_materialize_univtac_cli import (
    _config,
    _read_json,
    _staging_roots,
)
from tests.unit.test_materialize_univtac_recovery import (
    _patch_happy_path,
    _write_crashed_target,
)


def test_hard_crash_after_rollback_rename_is_reaped_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    staging_root, _ = _write_crashed_target(
        config,
        status="ready_to_publish",
        publishable_candidate=True,
    )
    (config.artifact_dir / "conversion_report.json").write_bytes(b"concurrent-report\n")
    _patch_happy_path(monkeypatch)
    real_rollback = materialize_module.rollback_directory_to_staging

    def rollback_then_die(**kwargs: object) -> Path:
        real_rollback(**kwargs)
        raise KeyboardInterrupt("hard crash after rollback rename")

    monkeypatch.setattr(
        materialize_module,
        "rollback_directory_to_staging",
        rollback_then_die,
    )
    with pytest.raises(KeyboardInterrupt, match="after rollback rename"):
        materialize_from_artifacts(config)

    assert not config.target_root.exists()
    marker = _read_json(staging_root / INCOMPLETE_MARKER_NAME)
    assert marker["status"] == "ready_to_publish"
    assert marker["phase"] == "ready_to_publish"

    monkeypatch.setattr(
        materialize_module,
        "rollback_directory_to_staging",
        real_rollback,
    )
    result = materialize_from_artifacts(config)
    assert result["status"] == "materialized"
    assert not staging_root.exists()
    assert not (
        config.artifact_dir / f"{REPORT_CANDIDATE_PREFIX}{staging_root.name}"
    ).exists()


def test_external_cleanup_journal_reaps_empty_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    target_root = config.target_root.resolve(strict=False)
    staging_root = target_root.parent / ".lerobot.incomplete-journal-crash"
    staging_root.mkdir()
    materialize_module._write_state_marker(
        staging_root,
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256="a" * 64,
        phase="materializing",
    )
    _patch_happy_path(monkeypatch)
    real_recover = materialize_cleanup._recover_cleanup_journal

    def die_after_journal(*args: object, **kwargs: object) -> Path:
        raise KeyboardInterrupt("hard crash after cleanup journal")

    monkeypatch.setattr(
        materialize_cleanup,
        "_recover_cleanup_journal",
        die_after_journal,
    )
    with pytest.raises(KeyboardInterrupt, match="after cleanup journal"):
        materialize_from_artifacts(config)

    # Persist the exact SIGKILL image after the internal marker was unlinked but
    # before the owned root's final rmdir. The external journal must retain the
    # ownership proof independently of that now-empty directory.
    (staging_root / INCOMPLETE_MARKER_NAME).unlink()
    assert staging_root.is_dir()
    assert list(staging_root.iterdir()) == []
    journals = list(
        config.artifact_dir.glob(f"{materialize_cleanup.CLEANUP_JOURNAL_PREFIX}*.json")
    )
    assert len(journals) == 1

    monkeypatch.setattr(
        materialize_cleanup,
        "_recover_cleanup_journal",
        real_recover,
    )
    result = materialize_from_artifacts(config)
    assert result["status"] == "materialized"
    assert not staging_root.exists()
    assert not journals[0].exists()
    assert _staging_roots(config) == []


def test_complete_reentry_preserves_unowned_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    assert materialize_from_artifacts(config)["status"] == "materialized"
    marker = _read_json(config.target_root / INCOMPLETE_MARKER_NAME)
    staging_root = Path(marker["staging_root"])
    candidate_path = config.artifact_dir / (
        f"{REPORT_CANDIDATE_PREFIX}{staging_root.name}"
    )
    foreign_bytes = b"foreign-concurrent-candidate\n"
    candidate_path.write_bytes(foreign_bytes)

    result = materialize_from_artifacts(config)

    assert result["status"] == "recovered_materialized"
    assert candidate_path.read_bytes() == foreign_bytes


def test_cleanup_journal_fifo_fails_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    journal_path = config.artifact_dir / (
        f"{materialize_cleanup.CLEANUP_JOURNAL_PREFIX}" ".lerobot.incomplete-fifo.json"
    )
    os.mkfifo(journal_path)
    _patch_happy_path(monkeypatch)

    with pytest.raises(RuntimeError, match="cleanup journal is not a regular file"):
        materialize_from_artifacts(config)


def test_staging_marker_fifo_is_preserved_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    staging_root = config.target_root.parent / ".lerobot.incomplete-fifo"
    staging_root.mkdir()
    marker_path = staging_root / INCOMPLETE_MARKER_NAME
    os.mkfifo(marker_path)
    _patch_happy_path(monkeypatch)

    result = materialize_from_artifacts(config)

    assert result["status"] == "materialized"
    assert staging_root.is_dir()
    assert marker_path.is_fifo()


def test_crash_after_staging_mkdir_before_initial_marker_is_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    real_reserve = materialize_module.reserve_staging_root
    crashed_staging: Path | None = None
    crashed_intent: Path | None = None

    def reserve_then_die(**kwargs: object) -> tuple[Path, Path]:
        nonlocal crashed_staging, crashed_intent
        crashed_staging, crashed_intent = real_reserve(**kwargs)
        raise KeyboardInterrupt("crash after staging mkdir before initial marker")

    monkeypatch.setattr(materialize_module, "reserve_staging_root", reserve_then_die)
    with pytest.raises(KeyboardInterrupt, match="before initial marker"):
        materialize_from_artifacts(config)

    assert crashed_staging is not None and crashed_staging.is_dir()
    assert crashed_intent is not None and crashed_intent.is_file()
    assert list(crashed_staging.iterdir()) == []

    monkeypatch.setattr(materialize_module, "reserve_staging_root", real_reserve)
    result = materialize_from_artifacts(config)

    assert result["status"] == "materialized"
    assert not crashed_staging.exists()
    assert not crashed_intent.exists()
    assert not list(
        config.artifact_dir.glob(
            f"{materialize_writeahead.CREATION_INTENT_PREFIX}*.json"
        )
    )


def test_cleanup_journal_create_never_replaces_injected_foreign_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    target_root = config.target_root.resolve(strict=False)
    staging_root = target_root.parent / ".lerobot.incomplete-journal-race"
    staging_root.mkdir()
    materialize_module._write_state_marker(
        staging_root,
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256="a" * 64,
        phase="materializing",
    )
    journal_path = materialize_cleanup._journal_path(
        config.artifact_dir,
        staging_root,
    )
    foreign_bytes = b"foreign-journal\n"
    real_publish = materialize_cleanup.rename_directory_noreplace

    def inject_before_publish(source: Path, target: Path) -> None:
        if target == journal_path:
            target.write_bytes(foreign_bytes)
        real_publish(source, target)

    monkeypatch.setattr(
        materialize_cleanup,
        "rename_directory_noreplace",
        inject_before_publish,
    )
    _patch_happy_path(monkeypatch)

    with pytest.raises(FileExistsError):
        materialize_from_artifacts(config)

    assert journal_path.read_bytes() == foreign_bytes
    assert staging_root.is_dir()


def test_cleanup_journal_recovery_preserves_load_to_unlink_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    target_root = config.target_root.resolve(strict=False)
    staging_root = target_root.parent / ".lerobot.incomplete-journal-replacement"
    staging_root.mkdir()
    materialize_module._write_state_marker(
        staging_root,
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256="a" * 64,
        phase="materializing",
    )
    _patch_happy_path(monkeypatch)
    real_recover = materialize_cleanup._recover_cleanup_journal
    monkeypatch.setattr(
        materialize_cleanup,
        "_recover_cleanup_journal",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("leave durable journal")
        ),
    )
    with pytest.raises(KeyboardInterrupt, match="leave durable journal"):
        materialize_from_artifacts(config)
    monkeypatch.setattr(
        materialize_cleanup,
        "_recover_cleanup_journal",
        real_recover,
    )
    journal_path = materialize_cleanup._journal_path(
        config.artifact_dir,
        staging_root,
    )
    assert journal_path.is_file()
    foreign_bytes = b"foreign-replacement-after-capture\n"
    real_load = materialize_cleanup._load_journal

    def load_then_replace(path: Path) -> dict[str, object]:
        payload = real_load(path)
        journal_path.write_bytes(foreign_bytes)
        return payload

    monkeypatch.setattr(materialize_cleanup, "_load_journal", load_then_replace)
    result = materialize_from_artifacts(config)

    assert result["status"] == "materialized"
    assert journal_path.read_bytes() == foreign_bytes
    assert not staging_root.exists()


def test_creation_intent_capture_crash_is_recovered_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    real_release = materialize_module.release_creation_intent
    captured_path: Path | None = None

    def capture_then_die(intent_path: Path) -> None:
        nonlocal captured_path
        captured_path = materialize_writeahead.capture_regular_file(intent_path).path
        raise KeyboardInterrupt("crash after creation intent capture")

    monkeypatch.setattr(
        materialize_module,
        "release_creation_intent",
        capture_then_die,
    )
    with pytest.raises(KeyboardInterrupt, match="intent capture"):
        materialize_from_artifacts(config)

    assert captured_path is not None and captured_path.is_file()
    monkeypatch.setattr(
        materialize_module,
        "release_creation_intent",
        real_release,
    )
    result = materialize_from_artifacts(config)

    assert result["status"] == "materialized"
    assert not captured_path.exists()


def test_unknown_legacy_marker_temp_is_preserved_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    target_root = config.target_root.resolve(strict=False)
    staging_root, intent_path = materialize_writeahead.reserve_staging_root(
        artifact_dir=config.artifact_dir,
        target_root=target_root,
        manifest_sha256="a" * 64,
    )
    marker_temp = staging_root / ".materialization_state.json.crashed.tmp"
    marker_temp.write_bytes(b"partial-marker\n")
    _patch_happy_path(monkeypatch)

    with pytest.raises(RuntimeError, match="unowned bootstrap bytes"):
        materialize_from_artifacts(config)

    assert staging_root.is_dir()
    assert marker_temp.read_bytes() == b"partial-marker\n"
    assert (
        intent_path.exists()
        or materialize_writeahead.capture_path_for(intent_path).exists()
    )


def test_private_publication_capture_crash_recovers_without_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    real_rename = materialize_transaction.rename_directory_noreplace
    crashed = False

    def capture_then_die(source: Path, target: Path) -> None:
        nonlocal crashed
        real_rename(source, target)
        if (
            source.name.startswith(".lerobot.incomplete-")
            and target.name.startswith(".materialize-publish-v1-")
            and not crashed
        ):
            crashed = True
            raise KeyboardInterrupt("crash after private publication capture")

    monkeypatch.setattr(
        materialize_transaction,
        "rename_directory_noreplace",
        capture_then_die,
    )
    with pytest.raises(KeyboardInterrupt, match="private publication capture"):
        materialize_from_artifacts(config)

    captures = list(tmp_path.glob(".materialize-publish-v1-*"))
    assert len(captures) == 1
    marker = _read_json(captures[0] / INCOMPLETE_MARKER_NAME)
    staging_root = Path(str(marker["staging_root"]))
    candidate = config.artifact_dir / (f"{REPORT_CANDIDATE_PREFIX}{staging_root.name}")
    assert not staging_root.exists()
    assert not config.target_root.exists()
    assert candidate.is_file()
    intents = list(
        config.artifact_dir.glob(
            f"{materialize_publication_journal.PUBLICATION_INTENT_PREFIX}*.json"
        )
    )
    assert len(intents) == 1

    monkeypatch.setattr(
        materialize_transaction,
        "rename_directory_noreplace",
        real_rename,
    )
    result = materialize_from_artifacts(config)
    assert result["status"] == "recovered_materialized"
    assert not staging_root.exists()
    assert config.target_root.is_dir()
    assert not candidate.exists()
    assert not list(tmp_path.glob(".materialize-publish-v1-*"))
    assert not list(
        config.artifact_dir.glob(
            f"{materialize_publication_journal.PUBLICATION_INTENT_PREFIX}*"
        )
    )

    retry = materialize_from_artifacts(config)
    assert retry["status"] == "recovered_materialized"
    assert not list(tmp_path.glob(".materialize-publish-v1-*"))
