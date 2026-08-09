# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import hashlib
import os
from pathlib import Path

import pytest

from script.track3_1 import (
    materialize_cleanup,
    materialize_tree,
    materialize_writeahead,
)


def test_hash_to_capture_replacement_is_restored_not_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate.json"
    owned_bytes = b"owned-candidate\n"
    foreign_bytes = b"foreign-concurrent-replacement\n"
    candidate.write_bytes(owned_bytes)
    expected = hashlib.sha256(owned_bytes).hexdigest()
    real_sha256 = materialize_cleanup.file_sha256_or_none
    replaced = False

    def hash_then_replace(path: Path) -> str | None:
        nonlocal replaced
        digest = real_sha256(path)
        if path == candidate and not replaced:
            replaced = True
            candidate.write_bytes(foreign_bytes)
        return digest

    monkeypatch.setattr(materialize_cleanup, "file_sha256_or_none", hash_then_replace)

    with pytest.raises(RuntimeError, match="unowned bytes"):
        materialize_cleanup.remove_owned_report_displacement(
            candidate,
            expected_sha256=expected,
        )

    assert candidate.read_bytes() == foreign_bytes


def test_replacement_after_atomic_capture_survives_at_candidate_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate.json"
    owned_bytes = b"owned-candidate\n"
    foreign_bytes = b"foreign-after-capture\n"
    candidate.write_bytes(owned_bytes)
    expected = hashlib.sha256(owned_bytes).hexdigest()
    real_rename = materialize_writeahead.rename_directory_noreplace

    def capture_then_replace(source: Path, target: Path) -> None:
        real_rename(source, target)
        if source == candidate:
            candidate.write_bytes(foreign_bytes)

    monkeypatch.setattr(
        materialize_writeahead,
        "rename_directory_noreplace",
        capture_then_replace,
    )

    materialize_cleanup.remove_owned_report_displacement(
        candidate,
        expected_sha256=expected,
    )

    assert candidate.read_bytes() == foreign_bytes


def test_prepared_candidate_publish_is_idempotent_when_destination_matches(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "owned" / "candidate.prepared.json"
    prepared.parent.mkdir()
    candidate = tmp_path / "candidate.json"
    payload = {"conversion_report_sha256": "a" * 64}
    expected_sha256 = materialize_writeahead.prepare_json_file(prepared, payload)

    materialize_writeahead.publish_prepared_file_noreplace(
        prepared,
        candidate,
        expected_sha256=expected_sha256,
    )
    materialize_writeahead.prepare_json_file(prepared, payload)
    materialize_writeahead.publish_prepared_file_noreplace(
        prepared,
        candidate,
        expected_sha256=expected_sha256,
    )

    assert candidate.is_file()
    assert not prepared.exists()


def test_prepared_candidate_publish_preserves_foreign_destination(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "owned" / "candidate.prepared.json"
    prepared.parent.mkdir()
    candidate = tmp_path / "candidate.json"
    payload = {"conversion_report_sha256": "a" * 64}
    expected_sha256 = materialize_writeahead.prepare_json_file(prepared, payload)
    foreign_bytes = b"foreign-candidate\n"
    candidate.write_bytes(foreign_bytes)

    with pytest.raises(RuntimeError, match="foreign destination"):
        materialize_writeahead.publish_prepared_file_noreplace(
            prepared,
            candidate,
            expected_sha256=expected_sha256,
        )

    assert candidate.read_bytes() == foreign_bytes
    assert prepared.is_file()


def test_cleanup_capture_preserves_lstat_to_rename_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / ".lerobot.incomplete-race"
    staging_root.mkdir()
    victim = staging_root / "victim.bin"
    victim.write_bytes(b"owned\n")
    real_rename = materialize_tree._rename_names_noreplace
    injected = False

    def inject_replacement(
        source_descriptor: int,
        source_name: str,
        target_descriptor: int,
        target_name: str,
    ) -> None:
        nonlocal injected
        if source_name == "victim.bin" and not injected:
            injected = True
            os.rename(
                source_name,
                "owned-moved.bin",
                src_dir_fd=source_descriptor,
                dst_dir_fd=source_descriptor,
            )
            descriptor = os.open(
                source_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=source_descriptor,
            )
            try:
                os.write(descriptor, b"foreign\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        real_rename(
            source_descriptor,
            source_name,
            target_descriptor,
            target_name,
        )

    monkeypatch.setattr(
        materialize_tree,
        "_rename_names_noreplace",
        inject_replacement,
    )

    with pytest.raises(RuntimeError, match="changed during atomic capture"):
        materialize_tree.remove_directory_tree_nofollow(staging_root)

    preserved = list(staging_root.glob(".materialize-foreign-preserved-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"foreign\n"
    with pytest.raises(RuntimeError, match="unverified cleanup quarantine"):
        materialize_tree.remove_directory_tree_nofollow(staging_root)
