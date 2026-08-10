# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from pathlib import Path

import pytest

from script.track3_1 import materialize_transaction
from script.track3_1 import materialize_univtac as materialize_module
from script.track3_1.materialize_univtac import (
    INCOMPLETE_MARKER_NAME,
    REPORT_CANDIDATE_PREFIX,
    materialize_from_artifacts,
)
from tests.unit.test_materialize_univtac_cli import _config, _read_json
from tests.unit.test_materialize_univtac_recovery import (
    _patch_happy_path,
    _write_crashed_target,
)


def test_forward_commit_rejects_staging_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    real_bound = materialize_module.rename_directory_noreplace_bound
    foreign_root: Path | None = None
    owned_moved: Path | None = None

    def swap_then_commit(
        source: Path,
        target: Path,
        *,
        expected_source_identity: tuple[int, int],
    ) -> None:
        nonlocal foreign_root, owned_moved
        if target == config.target_root:
            owned_moved = source.parent / f"{source.name}.owned-moved"
            source.rename(owned_moved)
            source.mkdir()
            (source / "foreign.bin").write_bytes(b"preserve-forward\n")
            foreign_root = source
        real_bound(
            source,
            target,
            expected_source_identity=expected_source_identity,
        )

    monkeypatch.setattr(
        materialize_module,
        "rename_directory_noreplace_bound",
        swap_then_commit,
    )
    with pytest.raises(RuntimeError, match="source identity changed"):
        materialize_from_artifacts(config)
    assert foreign_root is not None
    assert (foreign_root / "foreign.bin").read_bytes() == b"preserve-forward\n"
    assert owned_moved is not None and owned_moved.is_dir()
    assert not config.target_root.exists()


def test_rollback_rejects_target_root_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _write_crashed_target(
        config,
        status="incomplete",
        publishable_candidate=False,
    )
    _patch_happy_path(monkeypatch)
    real_bound = materialize_transaction.rename_directory_noreplace_bound
    owned_moved = tmp_path / "owned-target-moved"
    injected = False

    def swap_then_rollback(
        source: Path,
        target: Path,
        *,
        expected_source_identity: tuple[int, int],
    ) -> None:
        nonlocal injected
        if source == config.target_root and not injected:
            injected = True
            source.rename(owned_moved)
            source.mkdir()
            (source / "foreign.bin").write_bytes(b"preserve-rollback\n")
        real_bound(
            source,
            target,
            expected_source_identity=expected_source_identity,
        )

    monkeypatch.setattr(
        materialize_transaction,
        "rename_directory_noreplace_bound",
        swap_then_rollback,
    )
    with pytest.raises(RuntimeError, match="source identity changed"):
        materialize_from_artifacts(config)
    assert (config.target_root / "foreign.bin").read_bytes() == (b"preserve-rollback\n")
    assert owned_moved.is_dir()


def test_post_commit_target_swap_is_never_claimed_or_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    real_bound = materialize_module.rename_directory_noreplace_bound
    owned_displaced = tmp_path / "trusted-owned-target-after-commit"
    foreign_bytes = b"foreign-after-successful-bound-rename\n"
    injected = False

    def commit_then_swap_target(
        source: Path,
        target: Path,
        *,
        expected_source_identity: tuple[int, int],
    ) -> None:
        nonlocal injected
        real_bound(
            source,
            target,
            expected_source_identity=expected_source_identity,
        )
        if target == config.target_root and not injected:
            injected = True
            target.rename(owned_displaced)
            target.mkdir()
            (target / "foreign.bin").write_bytes(foreign_bytes)

    monkeypatch.setattr(
        materialize_module,
        "rename_directory_noreplace_bound",
        commit_then_swap_target,
    )
    with pytest.raises(RuntimeError, match="identity changed after publication"):
        materialize_from_artifacts(config)

    foreign = config.target_root / "foreign.bin"
    assert foreign.read_bytes() == foreign_bytes
    assert not (config.target_root / INCOMPLETE_MARKER_NAME).exists()
    trusted_marker = _read_json(owned_displaced / INCOMPLETE_MARKER_NAME)
    staging_root = Path(str(trusted_marker["staging_root"]))
    candidate = config.artifact_dir / (f"{REPORT_CANDIDATE_PREFIX}{staging_root.name}")
    assert candidate.is_file()

    monkeypatch.setattr(
        materialize_module,
        "rename_directory_noreplace_bound",
        real_bound,
    )
    with pytest.raises(FileExistsError, match="target root must not exist"):
        materialize_from_artifacts(config)
    assert foreign.read_bytes() == foreign_bytes
    assert not (config.target_root / INCOMPLETE_MARKER_NAME).exists()
    assert candidate.is_file()


def test_bound_commit_mismatch_never_moves_foreign_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    real_noreplace = materialize_transaction.rename_directory_noreplace
    owned_displaced = tmp_path / "trusted-owned-target-during-bound-rename"
    foreign_bytes = b"foreign-during-bound-rename-verification\n"
    injected = False
    committed_staging: Path | None = None

    def rename_then_replace_target(source: Path, target: Path) -> None:
        nonlocal committed_staging, injected
        real_noreplace(source, target)
        if target == config.target_root and not injected:
            injected = True
            committed_staging = source
            target.rename(owned_displaced)
            target.mkdir()
            (target / "foreign.bin").write_bytes(foreign_bytes)

    monkeypatch.setattr(
        materialize_transaction,
        "rename_directory_noreplace",
        rename_then_replace_target,
    )
    with pytest.raises(RuntimeError, match="identity changed during atomic rename"):
        materialize_from_artifacts(config)

    foreign = config.target_root / "foreign.bin"
    assert foreign.read_bytes() == foreign_bytes
    assert not (config.target_root / INCOMPLETE_MARKER_NAME).exists()
    trusted_marker = _read_json(owned_displaced / INCOMPLETE_MARKER_NAME)
    staging_root = Path(str(trusted_marker["staging_root"]))
    assert committed_staging is not None
    assert not committed_staging.exists()
    assert not staging_root.exists()
    candidate = config.artifact_dir / (f"{REPORT_CANDIDATE_PREFIX}{staging_root.name}")
    assert candidate.is_file()

    monkeypatch.setattr(
        materialize_transaction,
        "rename_directory_noreplace",
        real_noreplace,
    )
    with pytest.raises(FileExistsError, match="target root must not exist"):
        materialize_from_artifacts(config)
    assert foreign.read_bytes() == foreign_bytes
    assert not (config.target_root / INCOMPLETE_MARKER_NAME).exists()
    assert not staging_root.exists()
    assert candidate.is_file()


def test_source_swap_before_first_rename_never_reaches_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    real_noreplace = materialize_transaction.rename_directory_noreplace
    owned_displaced = tmp_path / "trusted-owned-staging-before-first-rename"
    foreign_bytes = b"foreign-at-original-staging-path\n"
    injected = False
    staging_root: Path | None = None

    def swap_source_before_first_rename(source: Path, target: Path) -> None:
        nonlocal injected, staging_root
        if source.name.startswith(".lerobot.incomplete-") and not injected:
            injected = True
            staging_root = source
            source.rename(owned_displaced)
            source.mkdir()
            (source / "foreign.bin").write_bytes(foreign_bytes)
        real_noreplace(source, target)

    monkeypatch.setattr(
        materialize_transaction,
        "rename_directory_noreplace",
        swap_source_before_first_rename,
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        materialize_from_artifacts(config)

    assert staging_root is not None
    foreign = staging_root / "foreign.bin"
    assert foreign.read_bytes() == foreign_bytes
    assert not (staging_root / INCOMPLETE_MARKER_NAME).exists()
    assert not config.target_root.exists()
    assert not (config.target_root / INCOMPLETE_MARKER_NAME).exists()
    trusted_marker = _read_json(owned_displaced / INCOMPLETE_MARKER_NAME)
    assert Path(str(trusted_marker["staging_root"])) == staging_root
    candidate = config.artifact_dir / (f"{REPORT_CANDIDATE_PREFIX}{staging_root.name}")
    assert candidate.is_file()

    monkeypatch.setattr(
        materialize_transaction,
        "rename_directory_noreplace",
        real_noreplace,
    )
    with pytest.raises(RuntimeError, match="publication source evidence"):
        materialize_from_artifacts(config)
    assert foreign.read_bytes() == foreign_bytes
    assert not (staging_root / INCOMPLETE_MARKER_NAME).exists()
    assert not config.target_root.exists()
    assert candidate.is_file()


def test_private_capture_restore_conflict_preserves_both_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / ".lerobot.incomplete-conflict"
    target = tmp_path / "lerobot"
    owned_displaced = tmp_path / "owned-displaced"
    source.mkdir()
    (source / "owned.bin").write_bytes(b"owned\n")
    metadata = source.lstat()
    expected_identity = metadata.st_dev, metadata.st_ino
    real_noreplace = materialize_transaction.rename_directory_noreplace
    captured_path: Path | None = None
    rename_calls = 0

    def inject_source_and_restore_conflicts(first: Path, second: Path) -> None:
        nonlocal captured_path, rename_calls
        rename_calls += 1
        if rename_calls == 1:
            first.rename(owned_displaced)
            first.mkdir()
            (first / "captured-foreign.bin").write_bytes(b"captured-foreign\n")
            captured_path = second
        elif rename_calls == 2:
            second.mkdir()
            (second / "source-conflict.bin").write_bytes(b"source-conflict\n")
        real_noreplace(first, second)

    monkeypatch.setattr(
        materialize_transaction,
        "rename_directory_noreplace",
        inject_source_and_restore_conflicts,
    )
    with pytest.raises(RuntimeError, match="conflicting source and capture entries"):
        materialize_transaction.rename_directory_noreplace_bound(
            source,
            target,
            expected_source_identity=expected_identity,
        )

    assert captured_path is not None
    assert (captured_path / "captured-foreign.bin").read_bytes() == (
        b"captured-foreign\n"
    )
    assert (source / "source-conflict.bin").read_bytes() == b"source-conflict\n"
    assert (owned_displaced / "owned.bin").read_bytes() == b"owned\n"
    assert not target.exists()
