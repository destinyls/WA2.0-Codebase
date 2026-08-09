#!/usr/bin/env python3
"""Race-aware file I/O for immutable Stage A collective smoke evidence."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

from script.track3_1.track31_collective_smoke_contract import canonical_report_bytes

MAX_REPORT_BYTES = 4 * 1024 * 1024


class CollectiveSmokeValidationError(ValueError):
    """Raised when evidence is unsafe, stale, incomplete, or identity-mismatched."""


def ensure_report_target_available(path: Path) -> Path:
    """Resolve the parent without following or accepting a preexisting target."""
    if not path.is_absolute():
        raise ValueError("report path must be absolute")
    parent = path.parent
    resolved_parent = parent.resolve(strict=True)
    if resolved_parent != parent or not stat.S_ISDIR(os.lstat(parent).st_mode):
        raise ValueError("report parent must be a canonical non-symlink directory")
    try:
        os.lstat(path)
    except FileNotFoundError:
        return resolved_parent
    raise FileExistsError(f"report target already exists: {path}")


def _write_all(file_descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(file_descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while publishing collective report")
        offset += written


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    file_descriptor = os.open(path, flags)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def publish_report_atomic(path: Path, report: Mapping[str, object]) -> str:
    """No-clobber publish canonical bytes after fsyncing data, mode, and directory."""
    parent = ensure_report_target_available(path)
    payload = canonical_report_bytes(report)
    digest = hashlib.sha256(payload).hexdigest()
    temporary_path = parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    file_descriptor: int | None = None
    temporary_identity: tuple[int, int, int] | None = None
    linked = False
    try:
        file_descriptor = os.open(temporary_path, flags, 0o600)
        _write_all(file_descriptor, payload)
        os.fsync(file_descriptor)
        os.fchmod(file_descriptor, 0o444)
        os.fsync(file_descriptor)
        temporary_metadata = os.fstat(file_descriptor)
        temporary_identity = (
            temporary_metadata.st_dev,
            temporary_metadata.st_ino,
            temporary_metadata.st_size,
        )
        os.close(file_descriptor)
        file_descriptor = None
        os.link(temporary_path, path, follow_symlinks=False)
        linked = True
        _fsync_directory(parent)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        if linked:
            _fsync_directory(parent)
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError("published report is not a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise RuntimeError("published report does not have mode 0444")
    final_identity = (metadata.st_dev, metadata.st_ino, metadata.st_size)
    if temporary_identity != final_identity or metadata.st_nlink != 1:
        raise RuntimeError("published report inode identity changed")
    return digest


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_nlink,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_nlink,
        right.st_size,
        right.st_mtime_ns,
    )


def _validation_error(
    message: str,
    error: Exception | None = None,
) -> NoReturn:
    if error is None:
        raise CollectiveSmokeValidationError(message)
    raise CollectiveSmokeValidationError(message) from error


def read_immutable_report(
    path: Path,
) -> tuple[dict[str, object], bytes, os.stat_result]:
    """Read a canonical 0444 regular file while detecting path/inode replacement."""
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        _validation_error(f"collective smoke report does not exist: {path}", error)
    if stat.S_ISLNK(metadata.st_mode):
        _validation_error("collective smoke report must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        _validation_error("collective smoke report must be a regular file")
    if metadata.st_nlink != 1:
        _validation_error("collective smoke report must have exactly one hard link")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        _validation_error("collective smoke report must have exact mode 0444")
    if metadata.st_size <= 0 or metadata.st_size > MAX_REPORT_BYTES:
        _validation_error("collective smoke report has an unsafe size")
    if not path.is_absolute() or path.parent.resolve(strict=True) != path.parent:
        _validation_error(
            "collective smoke report path must have a canonical non-symlink parent"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags)
    except OSError as error:
        _validation_error("collective smoke report could not be opened safely", error)
    try:
        if not _same_file(metadata, os.fstat(file_descriptor)):
            _validation_error("collective smoke report changed before it was opened")
        chunks: list[bytes] = []
        remaining = MAX_REPORT_BYTES + 1
        while remaining > 0:
            chunk = os.read(file_descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw_payload = b"".join(chunks)
        if len(raw_payload) > MAX_REPORT_BYTES:
            _validation_error("collective smoke report exceeds the size limit")
    finally:
        os.close(file_descriptor)
    if not _same_file(metadata, os.lstat(path)):
        _validation_error("collective smoke report changed while it was read")
    try:
        decoded = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _validation_error("collective smoke report is not valid UTF-8 JSON", error)
    if not isinstance(decoded, dict):
        _validation_error("collective smoke report root must be an object")
    report = {str(key): value for key, value in decoded.items()}
    if canonical_report_bytes(report) != raw_payload:
        _validation_error("collective smoke report is not canonical schema-v2 JSON")
    return report, raw_payload, metadata
