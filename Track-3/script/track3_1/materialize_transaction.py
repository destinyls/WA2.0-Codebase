# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed filesystem transaction primitives for materialization."""

import ctypes
import errno
import fcntl
import hashlib
import json
import logging
import os
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from script.track3_1.materialize_publication import (
    publish_directory_noreplace_bound,
)

ARTIFACT_LOCK_NAME = ".materialize_univtac.lock"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_DARWIN_RENAME_EXCL = 0x00000004
_DARWIN_RENAME_SWAP = 0x00000002
LOGGER = logging.getLogger(__name__)


class ConcurrentArtifactWriteError(RuntimeError):
    """Raised when a report changed after its transaction baseline was frozen."""


@dataclass(frozen=True)
class ReportPublishOutcome:
    """A committed report plus any interrupt observed after its atomic commit."""

    deferred_error: BaseException | None = None


@dataclass(frozen=True)
class ExistingTransaction:
    """Identity-bound state recovered from a committed target root."""

    status: str
    phase: str
    staging_root: Path
    candidate_path: Path
    conversion_report_sha256: str | None
    report_baseline_sha256: str | None
    baseline_declared: bool
    candidate_file_sha256: str | None
    marker_file_sha256: str
    target_identity: tuple[int, int]


def fsync_directory(path: Path) -> None:
    """Persist directory-entry updates on supported Darwin/Linux filesystems."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def unlink_best_effort(path: Path, *, label: str) -> None:
    """Remove one owned file without masking a primary transaction result."""

    try:
        path.unlink()
    except FileNotFoundError:
        return
    except BaseException:
        LOGGER.exception("Unable to remove %s: %s", label, path)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except BaseException:
                pass


@contextmanager
def exclusive_artifact_lock(artifact_dir: Path) -> Iterator[None]:
    """Serialize all materializers that can publish one conversion report."""

    lock_path = artifact_dir / ARTIFACT_LOCK_NAME
    artifact_descriptor = os.open(
        artifact_dir,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(
            ARTIFACT_LOCK_NAME,
            flags,
            0o600,
            dir_fd=artifact_descriptor,
        )
    except OSError as exc:
        os.close(artifact_descriptor)
        if exc.errno == errno.ELOOP:
            raise ValueError(
                f"artifact lock must not be a symlink: {lock_path}"
            ) from exc
        if exc.errno in {errno.EISDIR, errno.ENXIO}:
            raise ValueError(
                f"artifact lock must be a regular file: {lock_path}"
            ) from exc
        raise
    try:
        descriptor_metadata = os.fstat(descriptor)
        try:
            path_metadata = os.stat(
                ARTIFACT_LOCK_NAME,
                dir_fd=artifact_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError(f"artifact lock path changed: {lock_path}") from exc
        if stat.S_ISLNK(path_metadata.st_mode):
            raise ValueError(f"artifact lock must not be a symlink: {lock_path}")
        if not stat.S_ISREG(descriptor_metadata.st_mode) or not stat.S_ISREG(
            path_metadata.st_mode
        ):
            raise ValueError(f"artifact lock must be a regular file: {lock_path}")
        if (descriptor_metadata.st_dev, descriptor_metadata.st_ino) != (
            path_metadata.st_dev,
            path_metadata.st_ino,
        ):
            raise ValueError(f"artifact lock path changed while opening: {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"artifact materialization is already in progress: {lock_path}"
            ) from exc
        try:
            locked_path = os.stat(
                ARTIFACT_LOCK_NAME,
                dir_fd=artifact_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            raise ValueError(f"artifact lock path changed: {lock_path}") from exc
        if not stat.S_ISREG(locked_path.st_mode) or (
            locked_path.st_dev,
            locked_path.st_ino,
        ) != (descriptor_metadata.st_dev, descriptor_metadata.st_ino):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            raise ValueError(f"artifact lock path changed after locking: {lock_path}")
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
        os.close(artifact_descriptor)


def _read_regular_file_bytes(path: Path) -> bytes | None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno in {
            errno.ELOOP,
            errno.EISDIR,
            errno.ENOENT,
            errno.ENOTDIR,
        }:
            return None
        raise
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        final = os.fstat(descriptor)
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError:
            return None
        identities = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        )
        if identities != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        ) or identities != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ):
            return None
        return payload
    finally:
        os.close(descriptor)


def file_sha256_or_none(path: Path) -> str | None:
    """Return a regular-file identity while preserving an absent baseline."""

    payload = _read_regular_file_bytes(path)
    if payload is None:
        return None
    return hashlib.sha256(payload).hexdigest()


def report_logical_sha256(path: Path) -> str | None:
    """Return a report's declared logical identity without coercion."""
    encoded = _read_regular_file_bytes(path)
    if encoded is None:
        return None
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("conversion_report_sha256")
    return value if isinstance(value, str) and len(value) == 64 else None


def is_canonical_json_file(path: Path) -> bool:
    """Require the exact deterministic encoding written by this transaction."""
    encoded = _read_regular_file_bytes(path)
    if encoded is None:
        return False
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    canonical = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return encoded == canonical


def load_existing_transaction(
    *,
    artifact_dir: Path,
    target_root: Path,
    manifest_sha256: str,
    marker_name: str,
    candidate_prefix: str,
) -> ExistingTransaction:
    """Load one durable target marker and bind every recovery path."""

    target_identity = directory_identity_or_none(target_root)
    if target_identity is None:
        raise FileExistsError(f"existing target is not a real directory: {target_root}")
    marker_path = target_root / marker_name
    encoded_marker = _read_regular_file_bytes(marker_path)
    if encoded_marker is None:
        raise FileExistsError(
            "existing target has no recoverable materialization marker: "
            f"{target_root}"
        )
    try:
        payload = json.loads(encoded_marker.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FileExistsError(
            f"existing target has an unreadable materialization marker: {target_root}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise FileExistsError("existing target marker has an unsupported schema")
    if (
        payload.get("target_root") != str(target_root)
        or payload.get("manifest_sha256") != manifest_sha256
    ):
        raise FileExistsError("existing target marker identity does not match this run")
    raw_staging = payload.get("staging_root")
    if not isinstance(raw_staging, str) or not raw_staging:
        raise FileExistsError("existing target marker has no staging identity")
    staging_root = Path(raw_staging).resolve(strict=False)
    if (
        staging_root == target_root
        or staging_root.parent != target_root.parent
        or not staging_root.name.startswith(f".{target_root.name}.incomplete-")
    ):
        raise FileExistsError("existing target marker has an invalid staging identity")
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError("existing target and staging roots both exist")
    status = payload.get("status")
    if status not in {"incomplete", "ready_to_publish", "complete"}:
        raise FileExistsError("existing target marker has an unsupported status")
    phase = payload.get("phase")
    if not isinstance(phase, str) or not phase:
        raise FileExistsError("existing target marker has an invalid phase")

    def optional_sha256(field: str) -> str | None:
        value = payload.get(field)
        if value is None:
            return None
        if not isinstance(value, str) or len(value) != 64:
            raise FileExistsError(f"existing target marker has invalid {field}")
        return value

    if directory_identity_or_none(target_root) != target_identity:
        raise FileExistsError("existing target identity changed while loading marker")

    return ExistingTransaction(
        status=str(status),
        phase=phase,
        staging_root=staging_root,
        candidate_path=artifact_dir / f"{candidate_prefix}{staging_root.name}",
        conversion_report_sha256=optional_sha256("conversion_report_sha256"),
        report_baseline_sha256=optional_sha256("report_baseline_sha256"),
        baseline_declared="report_baseline_sha256" in payload,
        candidate_file_sha256=optional_sha256("candidate_file_sha256"),
        marker_file_sha256=hashlib.sha256(encoded_marker).hexdigest(),
        target_identity=target_identity,
    )


def directory_identity_or_none(path: Path) -> tuple[int, int] | None:
    """Return one no-follow directory identity, or ``None`` when untrusted."""

    try:
        metadata = path.lstat()
    except OSError:
        return None
    if not stat.S_ISDIR(metadata.st_mode):
        return None
    return metadata.st_dev, metadata.st_ino


def root_belongs_to_transaction(
    root: Path,
    *,
    target_root: Path,
    staging_root: Path,
    manifest_sha256: str,
    marker_name: str,
    expected_root_identity: tuple[int, int] | None = None,
    expected_marker_sha256: str | None = None,
) -> bool:
    """Return whether a regular state marker binds this exact root identity."""

    initial_identity = directory_identity_or_none(root)
    if initial_identity is None or (
        expected_root_identity is not None
        and initial_identity != expected_root_identity
    ):
        return False
    marker_path = root / marker_name
    encoded_marker = _read_regular_file_bytes(marker_path)
    if (
        encoded_marker is None
        or directory_identity_or_none(root) != initial_identity
        or (
            expected_marker_sha256 is not None
            and hashlib.sha256(encoded_marker).hexdigest() != expected_marker_sha256
        )
    ):
        return False
    try:
        payload = json.loads(encoded_marker.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and (
        payload.get("target_root"),
        payload.get("staging_root"),
        payload.get("manifest_sha256"),
    ) == (str(target_root), str(staging_root), manifest_sha256)


def rollback_directory_to_staging(
    *,
    target_root: Path,
    staging_root: Path,
    expected_target_identity: tuple[int, int],
) -> Path:
    """Best-effort rollback of an owned target directory to its sibling staging."""

    if not target_root.exists():
        return staging_root
    try:
        rename_directory_noreplace_bound(
            target_root,
            staging_root,
            expected_source_identity=expected_target_identity,
        )
    except RuntimeError:
        raise
    except BaseException:
        LOGGER.exception(
            "Unable to roll failed target back to staging; marker remains at %s",
            target_root,
        )
        return target_root
    return staging_root


def _raise_rename_error(result: int, *, source: Path, target: Path) -> None:
    if result == 0:
        return
    error_code = ctypes.get_errno()
    if error_code in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_code,
            "target root must not exist",
            str(target),
        )
    raise OSError(error_code, os.strerror(error_code), str(source), str(target))


def _rename_paths_with_flags(
    source: Path,
    target: Path,
    *,
    darwin_flags: int,
    linux_flags: int,
) -> None:
    source_parent = source.parent.resolve(strict=True)
    target_parent = target.parent.resolve(strict=True)
    if source_parent.stat().st_dev != target_parent.stat().st_dev:
        raise ValueError("atomic materialization rename requires one filesystem")
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, target_bytes, darwin_flags)
    elif sys.platform.startswith("linux"):
        try:
            rename = library.renameat2
        except AttributeError as exc:
            raise RuntimeError("renameat2 is required for no-clobber commit") from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            target_bytes,
            linux_flags,
        )
    else:
        raise RuntimeError(f"atomic no-replace rename is unsupported on {sys.platform}")
    _raise_rename_error(result, source=source, target=target)
    fsync_directory(source_parent)
    if target_parent != source_parent:
        fsync_directory(target_parent)


def rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically rename one sibling path without replacing any target."""

    _rename_paths_with_flags(
        source,
        target,
        darwin_flags=_DARWIN_RENAME_EXCL,
        linux_flags=_RENAME_NOREPLACE,
    )


def rename_directory_noreplace_bound(
    source: Path,
    target: Path,
    *,
    expected_source_identity: tuple[int, int],
) -> None:
    """Publish through an inode-verified private capture without replacement."""

    publish_directory_noreplace_bound(
        source,
        target,
        expected_source_identity=expected_source_identity,
        rename_noreplace=rename_directory_noreplace,
    )


def rename_paths_exchange(source: Path, target: Path) -> None:
    """Atomically exchange two existing sibling paths."""

    _rename_paths_with_flags(
        source,
        target,
        darwin_flags=_DARWIN_RENAME_SWAP,
        linux_flags=_RENAME_EXCHANGE,
    )


def probe_transaction_filesystem(
    parent: Path,
    *,
    require_exchange: bool = True,
    require_noreplace: bool = True,
) -> None:
    """Fail early unless this filesystem supports each required atomic rename."""

    resolved_parent = parent.resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix=".materialization-capability-",
        dir=resolved_parent,
    ) as temporary:
        probe_root = Path(temporary)
        source = probe_root / "source"
        target = probe_root / "target"
        source.write_bytes(b"source")
        target.write_bytes(b"target")
        if require_exchange:
            _rename_paths_with_flags(
                source,
                target,
                darwin_flags=_DARWIN_RENAME_SWAP,
                linux_flags=_RENAME_EXCHANGE,
            )
            if (
                _read_regular_file_bytes(source) != b"target"
                or _read_regular_file_bytes(target) != b"source"
            ):
                raise RuntimeError(
                    "atomic exchange capability probe returned wrong state"
                )
            _rename_paths_with_flags(
                source,
                target,
                darwin_flags=_DARWIN_RENAME_SWAP,
                linux_flags=_RENAME_EXCHANGE,
            )
            if (
                _read_regular_file_bytes(source) != b"source"
                or _read_regular_file_bytes(target) != b"target"
            ):
                raise RuntimeError(
                    "atomic exchange capability probe returned wrong state"
                )
        if require_noreplace:
            try:
                _rename_paths_with_flags(
                    source,
                    target,
                    darwin_flags=_DARWIN_RENAME_EXCL,
                    linux_flags=_RENAME_NOREPLACE,
                )
            except FileExistsError:
                pass
            else:
                raise RuntimeError(
                    "atomic no-replace capability probe clobbered a target"
                )
            if (
                _read_regular_file_bytes(source) != b"source"
                or _read_regular_file_bytes(target) != b"target"
            ):
                raise RuntimeError(
                    "atomic no-replace capability probe returned wrong state"
                )
            target.unlink()
            _rename_paths_with_flags(
                source,
                target,
                darwin_flags=_DARWIN_RENAME_EXCL,
                linux_flags=_RENAME_NOREPLACE,
            )
            if (
                _read_regular_file_bytes(source) is not None
                or _read_regular_file_bytes(target) != b"source"
            ):
                raise RuntimeError(
                    "atomic no-replace capability probe returned wrong state"
                )


def _rollback_report_exchange(
    *,
    candidate_path: Path,
    report_path: Path,
    candidate_sha256: str,
    previous_sha256: str,
) -> BaseException | None:
    rollback_error: BaseException | None = None
    try:
        rename_paths_exchange(candidate_path, report_path)
    except BaseException as exc:
        rollback_error = exc
    if (
        file_sha256_or_none(report_path) != previous_sha256
        or file_sha256_or_none(candidate_path) != candidate_sha256
    ):
        if rollback_error is not None:
            raise rollback_error
        raise RuntimeError("unable to restore concurrent conversion report")
    return rollback_error


def verify_interrupted_report_commit(
    *,
    candidate_path: Path,
    report_path: Path,
    expected_candidate_sha256: str,
    expected_report_sha256: str,
    expected_baseline_sha256: str | None,
) -> bool:
    """Verify a crashed CAS image or restore its displaced concurrent report."""

    if (
        file_sha256_or_none(report_path) != expected_candidate_sha256
        or report_logical_sha256(report_path) != expected_report_sha256
    ):
        return False
    displaced_sha256 = file_sha256_or_none(candidate_path)
    if expected_baseline_sha256 is None:
        if displaced_sha256 is None and not candidate_path.is_symlink():
            return True
        raise RuntimeError(
            "interrupted no-replace report commit left an unexpected candidate"
        )
    if displaced_sha256 == expected_baseline_sha256:
        return True
    if displaced_sha256 is None or candidate_path.is_symlink():
        raise RuntimeError("interrupted report exchange lost the displaced report")
    rollback_error = _rollback_report_exchange(
        candidate_path=candidate_path,
        report_path=report_path,
        candidate_sha256=expected_candidate_sha256,
        previous_sha256=displaced_sha256,
    )
    if rollback_error is not None:
        raise rollback_error
    raise ConcurrentArtifactWriteError(
        "conversion report changed during materialization"
    )


def publish_report_cas(
    *,
    candidate_path: Path,
    report_path: Path,
    expected_report_sha256: str | None,
) -> ReportPublishOutcome:
    """Publish a report with atomic no-replace or exchange-and-verify CAS."""

    candidate_sha256 = file_sha256_or_none(candidate_path)
    if candidate_sha256 is None:
        raise FileNotFoundError(candidate_path)
    if expected_report_sha256 is None:
        operation_error: BaseException | None = None
        try:
            rename_directory_noreplace(candidate_path, report_path)
        except BaseException as exc:
            operation_error = exc
        if (
            file_sha256_or_none(report_path) == candidate_sha256
            and not candidate_path.exists()
        ):
            return ReportPublishOutcome(deferred_error=operation_error)
        published_sha256 = file_sha256_or_none(report_path)
        if (
            operation_error is None
            and published_sha256 is not None
            and not (candidate_path.exists() or candidate_path.is_symlink())
        ):
            rename_directory_noreplace(report_path, candidate_path)
            if (
                file_sha256_or_none(candidate_path) != published_sha256
                or report_path.exists()
                or report_path.is_symlink()
            ):
                raise RuntimeError(
                    "unable to restore replaced no-replace report candidate"
                )
            raise ConcurrentArtifactWriteError(
                "conversion report candidate changed during publication"
            )
        if operation_error is not None:
            raise operation_error
        raise RuntimeError("report no-replace commit has an inconsistent state")

    operation_error = None
    previous_sha256: str | None = None
    try:
        rename_paths_exchange(candidate_path, report_path)
        previous_sha256 = file_sha256_or_none(candidate_path)
    except BaseException as exc:
        operation_error = exc
        if file_sha256_or_none(report_path) == candidate_sha256:
            previous_sha256 = file_sha256_or_none(candidate_path)
        else:
            raise
    published_sha256 = file_sha256_or_none(report_path)
    if (
        previous_sha256 == expected_report_sha256
        and published_sha256 == candidate_sha256
    ):
        return ReportPublishOutcome(deferred_error=operation_error)

    if previous_sha256 == expected_report_sha256 and published_sha256 is not None:
        rollback_error = _rollback_report_exchange(
            candidate_path=candidate_path,
            report_path=report_path,
            candidate_sha256=published_sha256,
            previous_sha256=previous_sha256,
        )
        if rollback_error is not None:
            raise rollback_error
        raise ConcurrentArtifactWriteError(
            "conversion report candidate changed during publication"
        )

    if previous_sha256 is None:
        raise RuntimeError("report exchange lost the previous report identity")
    rollback_error = _rollback_report_exchange(
        candidate_path=candidate_path,
        report_path=report_path,
        candidate_sha256=candidate_sha256,
        previous_sha256=previous_sha256,
    )
    if operation_error is not None:
        raise operation_error
    if rollback_error is not None:
        raise rollback_error
    raise ConcurrentArtifactWriteError(
        "conversion report changed during materialization"
    )
