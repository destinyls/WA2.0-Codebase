# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import fcntl
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from script.track3_1 import materialize_transaction
from script.track3_1.materialize_transaction import (
    ARTIFACT_LOCK_NAME,
    _write_json_atomic,
    exclusive_artifact_lock,
    rename_directory_noreplace,
)
from script.track3_1.materialize_univtac import (
    INCOMPLETE_MARKER_NAME,
    materialize_from_artifacts,
)
from tests.unit.test_materialize_univtac_cli import (
    _config,
    _fake_materializer,
    _manifest,
    _read_json,
    _staging_roots,
)


def _verified_count(**kwargs: object) -> SimpleNamespace:
    count = 759 if kwargs["physical_split"] == "train759" else 40
    return SimpleNamespace(episode_count=count)


def test_atomic_directory_commit_never_replaces_existing_empty_target(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".lerobot.incomplete-unit"
    target = tmp_path / "lerobot"
    staging.mkdir()
    target.mkdir()

    with pytest.raises(FileExistsError):
        rename_directory_noreplace(staging, target)

    assert staging.is_dir()
    assert target.is_dir()


def test_transaction_probe_rejects_silently_downgraded_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def silently_rename_instead_of_exchange(
        source: Path,
        target: Path,
        *,
        darwin_flags: int,
        linux_flags: int,
    ) -> None:
        del darwin_flags
        assert linux_flags == materialize_transaction._RENAME_EXCHANGE
        source.replace(target)

    monkeypatch.setattr(
        materialize_transaction,
        "_rename_paths_with_flags",
        silently_rename_instead_of_exchange,
    )

    with pytest.raises(
        RuntimeError,
        match="atomic exchange capability probe returned wrong state",
    ):
        materialize_transaction.probe_transaction_filesystem(tmp_path)


def test_transaction_probe_can_require_only_noreplace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rename_flags: list[int] = []

    def emulate_noreplace(
        source: Path,
        target: Path,
        *,
        darwin_flags: int,
        linux_flags: int,
    ) -> None:
        del darwin_flags
        rename_flags.append(linux_flags)
        if linux_flags == materialize_transaction._RENAME_EXCHANGE:
            pytest.fail("a no-replace-only probe must not attempt exchange")
        assert linux_flags == materialize_transaction._RENAME_NOREPLACE
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        source.rename(target)

    monkeypatch.setattr(
        materialize_transaction,
        "_rename_paths_with_flags",
        emulate_noreplace,
    )

    materialize_transaction.probe_transaction_filesystem(
        tmp_path,
        require_exchange=False,
        require_noreplace=True,
    )

    assert rename_flags == [
        materialize_transaction._RENAME_NOREPLACE,
        materialize_transaction._RENAME_NOREPLACE,
    ]


def test_atomic_json_cleanup_never_masks_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from script.track3_1 import materialize_transaction as transaction_module

    real_unlink = Path.unlink

    def fail_replace(source: object, target: object) -> None:
        raise OSError("primary replace error")

    def interrupt_cleanup(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if path.name.endswith(".tmp"):
            raise KeyboardInterrupt("secondary cleanup interrupt")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(transaction_module.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", interrupt_cleanup)

    with pytest.raises(OSError, match="primary replace error"):
        _write_json_atomic(tmp_path / "report.json", {"status": "candidate"})


@pytest.mark.parametrize(
    ("quarantine_path", "quarantine_length"),
    [
        ("grasp_classify/clean/89.hdf5", 43),
        ("grasp_classify/clean/90.hdf5", 44),
    ],
)
def test_formal_manifest_rejects_wrong_quarantine_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quarantine_path: str,
    quarantine_length: int,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        "script.track3_1.materialize_univtac.load_dataset_manifest",
        lambda path, *, verify_sources: _manifest(
            quarantine_path=quarantine_path,
            quarantine_length=quarantine_length,
        ),
    )

    with pytest.raises(ValueError, match="quarantine identity"):
        materialize_from_artifacts(config)

    assert not config.target_root.exists()
    assert not _staging_roots(config)


def test_artifact_lock_rejects_concurrent_materialization(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    with exclusive_artifact_lock(config.artifact_dir):
        with pytest.raises(RuntimeError, match="already in progress"):
            materialize_from_artifacts(config)

    assert not config.target_root.exists()
    assert not _staging_roots(config)


def test_artifact_lock_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    external_lock = tmp_path / "external.lock"
    external_lock.write_bytes(b"do-not-touch\n")
    (artifact_dir / ARTIFACT_LOCK_NAME).symlink_to(external_lock)

    with pytest.raises(ValueError, match="symlink"):
        with exclusive_artifact_lock(artifact_dir):
            pytest.fail("a symlink lock must never be acquired")

    assert external_lock.read_bytes() == b"do-not-touch\n"


def test_artifact_lock_rejects_non_regular_file(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    lock_path = artifact_dir / ARTIFACT_LOCK_NAME
    lock_path.mkdir()

    with pytest.raises(ValueError, match="regular file"):
        with exclusive_artifact_lock(artifact_dir):
            pytest.fail("a directory lock must never be acquired")


def test_artifact_lock_fstat_rejects_fifo(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    lock_path = artifact_dir / ARTIFACT_LOCK_NAME
    os.mkfifo(lock_path)

    with pytest.raises(ValueError, match="regular file"):
        with exclusive_artifact_lock(artifact_dir):
            pytest.fail("a FIFO lock must never be acquired")


def test_artifact_lock_revalidates_path_inode_after_flock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    lock_path = artifact_dir / ARTIFACT_LOCK_NAME
    displaced_path = artifact_dir / "displaced.lock"
    real_flock = materialize_transaction.fcntl.flock
    replaced = False

    def replace_after_lock(descriptor: int, operation: int) -> None:
        nonlocal replaced
        real_flock(descriptor, operation)
        if operation & fcntl.LOCK_EX and not replaced:
            replaced = True
            lock_path.rename(displaced_path)
            lock_path.write_bytes(b"foreign-lock\n")

    monkeypatch.setattr(materialize_transaction.fcntl, "flock", replace_after_lock)

    with pytest.raises(ValueError, match="changed after locking"):
        with exclusive_artifact_lock(artifact_dir):
            pytest.fail("a replaced lock inode must never guard the transaction")

    assert lock_path.read_bytes() == b"foreign-lock\n"


def test_report_candidate_replacement_before_exchange_is_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidate.json"
    report_path = tmp_path / "report.json"
    candidate_path.write_bytes(b"owned-candidate\n")
    report_path.write_bytes(b"baseline-report\n")
    baseline_sha256 = materialize_transaction.file_sha256_or_none(report_path)
    real_exchange = materialize_transaction.rename_paths_exchange
    exchange_calls = 0

    def replace_then_exchange(source: Path, target: Path) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        if exchange_calls == 1:
            source.write_bytes(b"foreign-candidate\n")
        real_exchange(source, target)

    monkeypatch.setattr(
        materialize_transaction,
        "rename_paths_exchange",
        replace_then_exchange,
    )

    with pytest.raises(
        materialize_transaction.ConcurrentArtifactWriteError,
        match="candidate changed",
    ):
        materialize_transaction.publish_report_cas(
            candidate_path=candidate_path,
            report_path=report_path,
            expected_report_sha256=baseline_sha256,
        )

    assert report_path.read_bytes() == b"baseline-report\n"
    assert candidate_path.read_bytes() == b"foreign-candidate\n"


def test_concurrent_report_change_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    report_path.write_bytes(b"baseline-report\n")
    concurrent_report = b"new-concurrent-report\n"

    monkeypatch.setattr(
        "script.track3_1.materialize_univtac.load_dataset_manifest",
        lambda path, *, verify_sources: _manifest(),
    )

    def materialize_then_change_report(**kwargs: object) -> dict[str, object]:
        converted = _fake_materializer(**kwargs)
        report_path.write_bytes(concurrent_report)
        return converted

    monkeypatch.setattr(
        "script.track3_1.materialize_univtac._materialize_lerobot_splits",
        materialize_then_change_report,
    )
    monkeypatch.setattr(
        "script.track3_1.materialize_report.verify_track31_physical_bundle",
        _verified_count,
    )

    with pytest.raises(RuntimeError, match="changed during materialization"):
        materialize_from_artifacts(config)

    assert report_path.read_bytes() == concurrent_report
    assert not config.target_root.exists()
    staging_roots = _staging_roots(config)
    assert len(staging_roots) == 1
    marker = _read_json(staging_roots[0] / INCOMPLETE_MARKER_NAME)
    assert marker["phase"] == "recovered_to_staging"


def test_candidate_cleanup_failure_does_not_mask_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    old_report = b"immutable-old-report\n"
    report_path.write_bytes(old_report)
    verification_count = 0
    real_unlink = Path.unlink

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
            raise ValueError("primary verification error")
        return _verified_count(**kwargs)

    def fail_candidate_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(".conversion_report.json.candidate-"):
            raise OSError("secondary cleanup error")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        "script.track3_1.materialize_report.verify_track31_physical_bundle",
        fail_final_verify,
    )
    monkeypatch.setattr(Path, "unlink", fail_candidate_unlink)

    with pytest.raises(ValueError, match="primary verification error"):
        materialize_from_artifacts(config)

    assert not config.target_root.exists()
    assert report_path.read_bytes() == old_report
    staging_roots = _staging_roots(config)
    assert len(staging_roots) == 1
    marker = _read_json(staging_roots[0] / INCOMPLETE_MARKER_NAME)
    assert marker["error_message"] == "primary verification error"


def test_successful_report_publish_keeps_durable_complete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    real_unlink = Path.unlink
    marker_unlink_attempts = 0

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
        _verified_count,
    )

    def fail_final_marker_unlink(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal marker_unlink_attempts
        if path == config.target_root / INCOMPLETE_MARKER_NAME:
            marker_unlink_attempts += 1
            raise OSError("marker unlink failed")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_final_marker_unlink)

    result = materialize_from_artifacts(config)

    assert result["status"] == "materialized"
    marker = _read_json(config.target_root / INCOMPLETE_MARKER_NAME)
    assert marker["status"] == "complete"
    assert marker["phase"] == "complete"
    assert marker_unlink_attempts == 0
    assert (config.artifact_dir / "conversion_report.json").is_file()


def test_interrupt_after_report_commit_never_rolls_back_published_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    old_report = b"immutable-old-report\n"
    report_path.write_bytes(old_report)
    from script.track3_1 import materialize_univtac as materialize_module

    real_publish = materialize_module.publish_report_cas

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
        _verified_count,
    )

    def commit_then_interrupt(**kwargs: object) -> object:
        real_publish(**kwargs)
        raise KeyboardInterrupt("interrupt after report commit")

    monkeypatch.setattr(
        materialize_module,
        "publish_report_cas",
        commit_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="interrupt after report commit"):
        materialize_from_artifacts(config)

    assert config.target_root.is_dir()
    assert report_path.read_bytes() != old_report
    marker_path = config.target_root / INCOMPLETE_MARKER_NAME
    marker = _read_json(marker_path)
    assert marker["status"] == "ready_to_publish"

    result = materialize_from_artifacts(config)
    assert result["status"] == "recovered_materialized"
    completed_marker = _read_json(marker_path)
    assert completed_marker["status"] == "complete"
    assert completed_marker["phase"] == "complete"
