# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Identity-bound no-follow removal of owned materialization trees."""

import ctypes
import hashlib
import os
import re
import stat
import sys
import uuid
from pathlib import Path

_RENAME_NOREPLACE = 1
_DARWIN_RENAME_EXCL = 0x00000004
_DELETE_CAPTURE_PREFIX = ".materialize-delete-v1-"
_FOREIGN_CAPTURE_PREFIX = ".materialize-foreign-preserved-"
_CAPTURE_PATTERN = re.compile(
    rf"^{re.escape(_DELETE_CAPTURE_PREFIX)}"
    r"(?P<device>[0-9a-f]+)-(?P<inode>[0-9a-f]+)-(?P<name>[0-9a-f]{24})$"
)


def _name_digest(name: str) -> str:
    return hashlib.sha256(os.fsencode(name)).hexdigest()[:24]


def _delete_capture_name(name: str, *, device: int, inode: int) -> str:
    """Derive a retry-discoverable quarantine name from one opened identity."""

    return f"{_DELETE_CAPTURE_PREFIX}{device:x}-{inode:x}-{_name_digest(name)}"


def captured_tree_path(path: Path, *, device: int, inode: int) -> Path:
    """Return the deterministic sibling quarantine for an owned root."""

    return path.parent / _delete_capture_name(
        path.name,
        device=device,
        inode=inode,
    )


def _captured_identity(name: str) -> tuple[int, int] | None:
    matched = _CAPTURE_PATTERN.fullmatch(name)
    if matched is None:
        return None
    return int(matched.group("device"), 16), int(matched.group("inode"), 16)


def _rename_names_noreplace(
    source_descriptor: int,
    source_name: str,
    target_descriptor: int,
    target_name: str,
) -> None:
    """Rename dirfd-relative names atomically without replacing a target."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    target_bytes = os.fsencode(target_name)
    if sys.platform == "darwin":
        rename = library.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_descriptor,
            source_bytes,
            target_descriptor,
            target_bytes,
            _DARWIN_RENAME_EXCL,
        )
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_descriptor,
            source_bytes,
            target_descriptor,
            target_bytes,
            _RENAME_NOREPLACE,
        )
    else:
        raise RuntimeError(f"atomic no-replace rename is unsupported on {sys.platform}")
    if result != 0:
        error_code = ctypes.get_errno()
        raise OSError(error_code, os.strerror(error_code), source_name, target_name)
    os.fsync(source_descriptor)
    if target_descriptor != source_descriptor:
        os.fsync(target_descriptor)


def _capture_open_entry(
    descriptor: int,
    *,
    name: str,
    expected_device: int,
    expected_inode: int,
) -> str:
    """Move a child to a deterministic identity-bound quarantine."""

    captured_name = _delete_capture_name(
        name,
        device=expected_device,
        inode=expected_inode,
    )
    _rename_names_noreplace(
        descriptor,
        name,
        descriptor,
        captured_name,
    )
    captured = os.stat(
        captured_name,
        dir_fd=descriptor,
        follow_symlinks=False,
    )
    if (captured.st_dev, captured.st_ino) != (expected_device, expected_inode):
        preserved_name = f"{_FOREIGN_CAPTURE_PREFIX}{uuid.uuid4().hex}"
        try:
            _rename_names_noreplace(
                descriptor,
                captured_name,
                descriptor,
                preserved_name,
            )
        except BaseException:
            pass
        raise RuntimeError("owned staging entry changed during atomic capture")
    return captured_name


def _clear_open_directory(
    descriptor: int,
    *,
    expected_device: int,
    preserve_until_last: str | None = None,
) -> None:
    """Remove descendants through directory descriptors without following links."""

    with os.scandir(descriptor) as entries:
        names = sorted(entry.name for entry in entries)
    if any(name.startswith(_FOREIGN_CAPTURE_PREFIX) for name in names):
        raise RuntimeError("staging contains an unverified cleanup quarantine")
    if preserve_until_last is not None and preserve_until_last in names:
        names.remove(preserve_until_last)
        names.append(preserve_until_last)
    for name in names:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if metadata.st_dev != expected_device:
            raise RuntimeError("owned staging cleanup cannot cross filesystems")
        captured_identity = _captured_identity(name)
        if name.startswith(_DELETE_CAPTURE_PREFIX):
            if captured_identity is None or captured_identity != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise RuntimeError("staging contains an unverified cleanup quarantine")
            captured_name = name
        else:
            captured_name = ""
        if stat.S_ISDIR(metadata.st_mode):
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child)
                if not stat.S_ISDIR(opened.st_mode) or (
                    opened.st_dev,
                    opened.st_ino,
                ) != (metadata.st_dev, metadata.st_ino):
                    raise RuntimeError("owned staging entry changed during cleanup")
                _clear_open_directory(child, expected_device=expected_device)
            finally:
                os.close(child)
        if not captured_name:
            captured_name = _capture_open_entry(
                descriptor,
                name=name,
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
            )
        try:
            if stat.S_ISDIR(metadata.st_mode):
                os.rmdir(captured_name, dir_fd=descriptor)
            else:
                os.unlink(captured_name, dir_fd=descriptor)
        except BaseException:
            try:
                _rename_names_noreplace(
                    descriptor,
                    captured_name,
                    descriptor,
                    name,
                )
            except BaseException:
                pass
            raise
    os.fsync(descriptor)


def remove_directory_tree_nofollow(
    path: Path,
    *,
    preserve_until_last: str | None = None,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    """Delete an identity-bound sibling tree, including a prior crash capture."""

    parent_descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        if expected_identity is None:
            digest = _name_digest(path.name)
            matches: list[tuple[str, os.stat_result]] = []
            with os.scandir(parent_descriptor) as entries:
                for entry in entries:
                    identity = _captured_identity(entry.name)
                    if identity is None or not entry.name.endswith(f"-{digest}"):
                        continue
                    metadata = os.stat(
                        entry.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if identity == (metadata.st_dev, metadata.st_ino):
                        matches.append((entry.name, metadata))
            if len(matches) > 1:
                raise RuntimeError("multiple staging root captures match one path")
            if matches:
                captured_name, initial = matches[0]
                expected_identity = (initial.st_dev, initial.st_ino)
            else:
                initial = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                captured_name = _delete_capture_name(
                    path.name,
                    device=initial.st_dev,
                    inode=initial.st_ino,
                )
                expected_identity = (initial.st_dev, initial.st_ino)
        else:
            captured_name = _delete_capture_name(
                path.name,
                device=expected_identity[0],
                inode=expected_identity[1],
            )
            try:
                initial = os.stat(
                    captured_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                initial = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
        if (
            not stat.S_ISDIR(initial.st_mode)
            or (
                initial.st_dev,
                initial.st_ino,
            )
            != expected_identity
        ):
            raise ValueError(f"owned staging root identity changed: {path}")
        try:
            captured = os.stat(
                captured_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            captured_name = _capture_open_entry(
                parent_descriptor,
                name=path.name,
                expected_device=initial.st_dev,
                expected_inode=initial.st_ino,
            )
        else:
            if (captured.st_dev, captured.st_ino) != expected_identity:
                raise RuntimeError("owned staging root capture changed")
        try:
            root_descriptor = os.open(
                captured_name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            try:
                opened = os.fstat(root_descriptor)
                if (opened.st_dev, opened.st_ino) != expected_identity:
                    raise RuntimeError("owned staging root changed during cleanup")
                _clear_open_directory(
                    root_descriptor,
                    expected_device=initial.st_dev,
                    preserve_until_last=preserve_until_last,
                )
            finally:
                os.close(root_descriptor)
        except BaseException:
            try:
                _rename_names_noreplace(
                    parent_descriptor,
                    captured_name,
                    parent_descriptor,
                    path.name,
                )
            except BaseException:
                pass
            raise
        try:
            os.rmdir(captured_name, dir_fd=parent_descriptor)
        except BaseException:
            try:
                _rename_names_noreplace(
                    parent_descriptor,
                    captured_name,
                    parent_descriptor,
                    path.name,
                )
            except BaseException:
                pass
            raise
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
