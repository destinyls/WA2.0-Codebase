# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Atomic no-clobber publication under one stable directory descriptor."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class PublishedFile:
    path: Path
    raw: bytes
    sha256: str
    size: int


def _fingerprint(result: os.stat_result) -> tuple[int, ...]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_mode,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _directory_identity(result: os.stat_result) -> tuple[int, int, int]:
    return result.st_dev, result.st_ino, result.st_mode


def _validate_private_directory(result: os.stat_result, *, label: str) -> None:
    if not stat.S_ISDIR(result.st_mode):
        raise RuntimeError(f"{label} staging entry is not a directory")
    if result.st_uid != os.geteuid():
        raise RuntimeError(f"{label} staging directory has an unexpected owner")
    if stat.S_IMODE(result.st_mode) != 0o700:
        raise RuntimeError(f"{label} staging directory must have mode 0700")


def _read_descriptor(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(8 * 1024 * 1024, size - offset), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    raw = b"".join(chunks)
    if len(raw) != size:
        raise RuntimeError("atomic artifact byte count changed before publication")
    return raw


def publish_atomic_file(
    *,
    output: Path,
    writer: Callable[[BinaryIO], None],
    validator: Callable[[bytes], None],
    label: str,
) -> PublishedFile:
    """Write, validate, and hard-link a file relative to a stable parent fd."""

    destination = Path(output).expanduser()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{label} already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_before = destination.parent.lstat()
    if stat.S_ISLNK(parent_before.st_mode) or not stat.S_ISDIR(parent_before.st_mode):
        raise ValueError(f"{label} parent must be a regular directory")
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open(destination.parent, parent_flags)
    staging_name = f".{destination.name}.{secrets.token_hex(16)}.staging"
    payload_name = "payload"
    staging_descriptor = -1
    temporary_descriptor = -1
    staging_identity: tuple[int, int, int] | None = None
    try:
        parent_opened = os.fstat(parent_descriptor)
        if _directory_identity(parent_opened) != _directory_identity(parent_before):
            raise RuntimeError(f"{label} parent changed while it was opened")
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
        staging_created = os.stat(
            staging_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        _validate_private_directory(staging_created, label=label)
        staging_identity = _directory_identity(staging_created)
        staging_descriptor = os.open(
            staging_name, parent_flags, dir_fd=parent_descriptor
        )
        staging_opened = os.fstat(staging_descriptor)
        _validate_private_directory(staging_opened, label=label)
        if _directory_identity(staging_opened) != staging_identity:
            raise RuntimeError(f"{label} staging directory changed while it was opened")
        temporary_descriptor = os.open(
            payload_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=staging_descriptor,
        )
        with os.fdopen(os.dup(temporary_descriptor), "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        opened = os.fstat(temporary_descriptor)
        raw = _read_descriptor(temporary_descriptor, opened.st_size)
        validator(raw)
        os.fchmod(temporary_descriptor, 0o444)
        opened = os.fstat(temporary_descriptor)
        named = os.stat(payload_name, dir_fd=staging_descriptor, follow_symlinks=False)
        if _fingerprint(named) != _fingerprint(opened):
            raise RuntimeError(f"{label} staging file changed before publication")
        try:
            os.link(
                payload_name,
                destination.name,
                src_dir_fd=staging_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise FileExistsError(f"{label} already exists: {destination}") from None
        published = os.stat(
            destination.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        linked_opened = os.fstat(temporary_descriptor)
        if _fingerprint(published) != _fingerprint(linked_opened):
            raise RuntimeError(f"{label} published inode differs from staging inode")
        os.fsync(parent_descriptor)
        parent_after = destination.parent.lstat()
        if _directory_identity(parent_after) != _directory_identity(parent_opened):
            raise RuntimeError(f"{label} parent changed during publication")
        return PublishedFile(
            path=destination,
            raw=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
        )
    finally:
        cleanup_error: RuntimeError | None = None
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if staging_descriptor >= 0:
            try:
                os.unlink(payload_name, dir_fd=staging_descriptor)
            except FileNotFoundError:
                pass
            os.close(staging_descriptor)
        if staging_identity is not None:
            try:
                staging_named = os.stat(
                    staging_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                cleanup_error = RuntimeError(
                    f"{label} staging directory disappeared before cleanup"
                )
            else:
                if _directory_identity(staging_named) != staging_identity:
                    cleanup_error = RuntimeError(
                        f"{label} staging directory changed before cleanup; "
                        "the replacement was left untouched"
                    )
                else:
                    try:
                        os.rmdir(staging_name, dir_fd=parent_descriptor)
                    except OSError as error:
                        cleanup_error = RuntimeError(
                            f"{label} verified staging directory could not be removed"
                        )
                        cleanup_error.__cause__ = error
        os.close(parent_descriptor)
        if cleanup_error is not None:
            raise cleanup_error


__all__ = ("PublishedFile", "publish_atomic_file")
