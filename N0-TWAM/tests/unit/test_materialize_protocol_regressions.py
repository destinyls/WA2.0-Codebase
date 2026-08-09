# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import hashlib
import os
from pathlib import Path

import pytest

from script.track3_1 import (
    materialize_cleanup,
    materialize_state,
    materialize_transaction,
    materialize_tree,
    materialize_writeahead,
)


def _crash_after_matching_rename(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_name: str,
) -> None:
    real_rename = materialize_tree._rename_names_noreplace
    crashed = False

    def rename_then_crash(
        source_descriptor: int,
        source: str,
        target_descriptor: int,
        target: str,
    ) -> None:
        nonlocal crashed
        real_rename(
            source_descriptor,
            source,
            target_descriptor,
            target,
        )
        if source == source_name and not crashed:
            crashed = True
            raise KeyboardInterrupt("crash after deterministic tree capture")

    monkeypatch.setattr(materialize_tree, "_rename_names_noreplace", rename_then_crash)


def test_child_capture_crash_is_replayed_without_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / ".lerobot.incomplete-child-capture"
    staging_root.mkdir()
    (staging_root / "victim.bin").write_bytes(b"owned\n")
    real_rename = materialize_tree._rename_names_noreplace
    _crash_after_matching_rename(monkeypatch, source_name="victim.bin")

    with pytest.raises(KeyboardInterrupt, match="deterministic tree capture"):
        materialize_tree.remove_directory_tree_nofollow(staging_root)

    assert staging_root.is_dir()
    assert any(
        entry.name.startswith(materialize_tree._DELETE_CAPTURE_PREFIX)
        for entry in staging_root.iterdir()
    )
    monkeypatch.setattr(materialize_tree, "_rename_names_noreplace", real_rename)
    materialize_tree.remove_directory_tree_nofollow(staging_root)
    assert not staging_root.exists()


def test_root_capture_crash_is_replayed_without_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / ".lerobot.incomplete-root-capture"
    staging_root.mkdir()
    real_rename = materialize_tree._rename_names_noreplace
    _crash_after_matching_rename(monkeypatch, source_name=staging_root.name)

    with pytest.raises(KeyboardInterrupt, match="deterministic tree capture"):
        materialize_tree.remove_directory_tree_nofollow(staging_root)

    assert not staging_root.exists()
    assert any(
        entry.name.startswith(materialize_tree._DELETE_CAPTURE_PREFIX)
        for entry in tmp_path.iterdir()
    )
    monkeypatch.setattr(materialize_tree, "_rename_names_noreplace", real_rename)
    materialize_tree.remove_directory_tree_nofollow(staging_root)
    assert not any(
        entry.name.startswith(materialize_tree._DELETE_CAPTURE_PREFIX)
        for entry in tmp_path.iterdir()
    )


@pytest.mark.parametrize("crash_phase", ["capture", "final_delete"])
def test_candidate_capture_crash_has_one_recoverable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_phase: str,
) -> None:
    candidate = tmp_path / "candidate.json"
    owned = b"owned-candidate\n"
    candidate.write_bytes(owned)
    expected = hashlib.sha256(owned).hexdigest()
    capture = materialize_writeahead.capture_path_for(candidate)
    final_delete = materialize_writeahead.final_delete_path_for(candidate)
    real_rename = materialize_writeahead.rename_directory_noreplace
    crashed = False

    def rename_then_crash(source: Path, target: Path) -> None:
        nonlocal crashed
        real_rename(source, target)
        should_crash = (source, target) == (
            (candidate, capture)
            if crash_phase == "capture"
            else (capture, final_delete)
        )
        if should_crash and not crashed:
            crashed = True
            raise KeyboardInterrupt(f"crash after candidate {crash_phase}")

    monkeypatch.setattr(
        materialize_writeahead,
        "rename_directory_noreplace",
        rename_then_crash,
    )
    with pytest.raises(KeyboardInterrupt, match="crash after candidate"):
        materialize_cleanup.remove_owned_report_displacement(
            candidate,
            expected_sha256=expected,
        )

    monkeypatch.setattr(
        materialize_writeahead,
        "rename_directory_noreplace",
        real_rename,
    )
    materialize_cleanup.remove_owned_report_displacement(
        candidate,
        expected_sha256=expected,
    )
    assert not any(
        path.exists() or path.is_symlink()
        for path in (candidate, capture, final_delete)
    )
    assert not list(tmp_path.glob("*owned-delete*"))


@pytest.mark.parametrize(
    "record_name",
    [
        ".materialization_creation-.lerobot.incomplete-partial.json",
        materialize_cleanup.STAGING_CLEANUP_JOURNAL_NAME,
        materialize_state.COMPLETE_PREPARED_NAME,
    ],
)
def test_partial_prepare_write_is_preserved_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_name: str,
) -> None:
    owned_root = tmp_path / "owned"
    owned_root.mkdir()
    record = owned_root / record_name
    payload = {"schema_version": 1, "record": record_name}
    encoded = materialize_writeahead.canonical_json_bytes(payload)
    real_write_all = materialize_writeahead._write_all

    def half_write_then_crash(descriptor: int, content: bytes) -> None:
        os.write(descriptor, content[: max(1, len(content) // 2)])
        os.fsync(descriptor)
        raise KeyboardInterrupt("crash during prepare write")

    monkeypatch.setattr(materialize_writeahead, "_write_all", half_write_then_crash)
    with pytest.raises(KeyboardInterrupt, match="prepare write"):
        materialize_writeahead.prepare_json_file(
            record,
            payload,
            preserve_dir=tmp_path,
        )
    assert not record.exists()

    monkeypatch.setattr(materialize_writeahead, "_write_all", real_write_all)
    materialize_writeahead.prepare_json_file(
        record,
        payload,
        preserve_dir=tmp_path,
    )
    assert record.read_bytes() == encoded
    preserved = list(
        tmp_path.glob(f"{materialize_writeahead.FOREIGN_PRESERVED_PREFIX}*")
    )
    assert len(preserved) == 1
    assert encoded.startswith(preserved[0].read_bytes())


def test_partial_creation_intent_is_discovered_without_wedge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    target_root = tmp_path / "lerobot"
    real_write_all = materialize_writeahead._write_all

    def half_write_then_crash(descriptor: int, content: bytes) -> None:
        os.write(descriptor, content[: max(1, len(content) // 2)])
        raise KeyboardInterrupt("partial creation intent")

    monkeypatch.setattr(materialize_writeahead, "_write_all", half_write_then_crash)
    with pytest.raises(KeyboardInterrupt, match="partial creation intent"):
        materialize_writeahead.reserve_staging_root(
            artifact_dir=artifact_dir,
            target_root=target_root,
            manifest_sha256="a" * 64,
        )
    monkeypatch.setattr(materialize_writeahead, "_write_all", real_write_all)

    materialize_writeahead.recover_creation_intents(
        artifact_dir=artifact_dir,
        target_root=target_root,
        manifest_sha256="a" * 64,
        marker_name=materialize_state.DEFAULT_MARKER_NAME,
    )
    staging_root, intent_path = materialize_writeahead.reserve_staging_root(
        artifact_dir=artifact_dir,
        target_root=target_root,
        manifest_sha256="a" * 64,
    )
    assert staging_root.is_dir()
    assert intent_path.is_file()


def test_bootstrap_foreign_content_is_preserved_fail_closed(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    target_root = tmp_path / "lerobot"
    staging_root, intent_path = materialize_writeahead.reserve_staging_root(
        artifact_dir=artifact_dir,
        target_root=target_root,
        manifest_sha256="a" * 64,
    )
    foreign = staging_root / "foreign-user-bytes.bin"
    foreign.write_bytes(b"do-not-delete\n")

    with pytest.raises(RuntimeError, match="unowned bootstrap bytes"):
        materialize_writeahead.recover_creation_intents(
            artifact_dir=artifact_dir,
            target_root=target_root,
            manifest_sha256="a" * 64,
            marker_name=materialize_state.DEFAULT_MARKER_NAME,
        )

    assert staging_root.is_dir()
    assert foreign.read_bytes() == b"do-not-delete\n"
    assert (
        intent_path.exists()
        or materialize_writeahead.capture_path_for(intent_path).exists()
    )


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
@pytest.mark.parametrize("loader", ["state", "intent", "journal", "transaction"])
def test_formal_schema_version_rejects_non_integer_scalars(
    tmp_path: Path,
    schema_version: object,
    loader: str,
) -> None:
    payload: dict[str, object] = {"schema_version": schema_version}
    encoded = materialize_writeahead.canonical_json_bytes(payload)
    if loader == "state":
        with pytest.raises(RuntimeError, match="unsupported schema"):
            materialize_state._load_state_bytes(encoded)
    elif loader == "intent":
        captured = materialize_writeahead.CapturedFile(
            path=tmp_path / "intent",
            payload=encoded,
            sha256=hashlib.sha256(encoded).hexdigest(),
        )
        with pytest.raises(RuntimeError, match="unsupported schema"):
            materialize_writeahead._parse_intent(captured)
    elif loader == "journal":
        journal = tmp_path / "journal.json"
        journal.write_bytes(encoded)
        with pytest.raises(RuntimeError, match="unsupported schema"):
            materialize_cleanup._load_journal(journal)
    else:
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        target_root = tmp_path / "lerobot"
        target_root.mkdir()
        marker = target_root / materialize_state.DEFAULT_MARKER_NAME
        marker.write_bytes(encoded)
        with pytest.raises(FileExistsError, match="unsupported schema"):
            materialize_transaction.load_existing_transaction(
                artifact_dir=artifact_dir,
                target_root=target_root,
                manifest_sha256="a" * 64,
                marker_name=materialize_state.DEFAULT_MARKER_NAME,
                candidate_prefix=".candidate-",
            )


def test_terminal_capture_toctou_preserves_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_root = tmp_path / "lerobot"
    target_root.mkdir()
    staging_root = tmp_path / ".lerobot.incomplete-terminal"
    identity = materialize_state.directory_identity(target_root)
    marker_sha256 = materialize_state.write_state_marker(
        target_root,
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256="a" * 64,
        phase="report_committed",
        status="ready_to_publish",
        expected_root_identity=identity,
    )
    marker = target_root / materialize_state.DEFAULT_MARKER_NAME
    foreign = b"foreign-marker-replacement\n"
    real_capture = materialize_state.capture_regular_file
    replaced = False

    def replace_then_capture(path: Path) -> materialize_writeahead.CapturedFile:
        nonlocal replaced
        if path == marker and not replaced:
            replaced = True
            marker.write_bytes(foreign)
        return real_capture(path)

    monkeypatch.setattr(materialize_state, "capture_regular_file", replace_then_capture)
    with pytest.raises(RuntimeError, match="unreadable"):
        materialize_state.write_complete_marker(
            target_root=target_root,
            staging_root=staging_root,
            manifest_sha256="a" * 64,
            conversion_report_sha256="b" * 64,
            expected_target_identity=identity,
            expected_marker_sha256=marker_sha256,
        )
    assert marker.read_bytes() == foreign


def test_nonterminal_transition_rejects_valid_looking_foreign_marker(
    tmp_path: Path,
) -> None:
    target_root = tmp_path / "lerobot"
    target_root.mkdir()
    staging_root = tmp_path / ".lerobot.incomplete-transition"
    identity = materialize_state.directory_identity(target_root)
    marker_sha256 = materialize_state.write_state_marker(
        target_root,
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256="a" * 64,
        phase="publishing_report",
        status="ready_to_publish",
        expected_root_identity=identity,
    )
    marker = target_root / materialize_state.DEFAULT_MARKER_NAME
    foreign_payload = materialize_state.state_payload(
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256="a" * 64,
        phase="foreign-valid-looking-phase",
        status="ready_to_publish",
    )
    foreign = materialize_writeahead.canonical_json_bytes(foreign_payload)
    marker.write_bytes(foreign)

    with pytest.raises(RuntimeError, match="bytes changed"):
        materialize_state.write_state_marker(
            target_root,
            target_root=target_root,
            staging_root=staging_root,
            manifest_sha256="a" * 64,
            phase="report_committed",
            status="ready_to_publish",
            expected_root_identity=identity,
            expected_marker_sha256=marker_sha256,
        )
    assert marker.read_bytes() == foreign


def test_expected_marker_cas_never_claims_a_missing_marker(tmp_path: Path) -> None:
    target_root = tmp_path / "lerobot"
    target_root.mkdir()
    marker = target_root / materialize_state.DEFAULT_MARKER_NAME

    with pytest.raises(RuntimeError, match="expected state-marker disappeared"):
        materialize_state.write_state_marker(
            target_root,
            target_root=target_root,
            staging_root=tmp_path / ".lerobot.incomplete-missing-marker",
            manifest_sha256="a" * 64,
            phase="rollback_pending",
            expected_root_identity=materialize_state.directory_identity(target_root),
            expected_marker_sha256="b" * 64,
        )
    assert not marker.exists()
    assert not (target_root / materialize_state.STATE_PREPARED_NAME).exists()
