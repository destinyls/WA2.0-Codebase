# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Identity-bound two-stage directory publication.

Cooperating writers are serialized by the artifact lock. Darwin and Linux do
not expose an inode-conditional directory rename, so a same-identity hostile
host writer can still swap the private name during the final syscall window.
Every mismatch observable here is fail-closed: unknown entries are preserved,
never overwritten or deleted, and the caller must not publish a state marker.
This module only renames entries and never mutates captured directory contents;
byte stability therefore follows for writers that honor the artifact lock.
"""

import hashlib
import os
import stat
from collections.abc import Callable
from pathlib import Path

RenameNoReplace = Callable[[Path, Path], None]
PUBLISH_CAPTURE_PREFIX = ".materialize-publish-v1-"


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _open_nofollow(path: Path) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        return descriptor, os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise


def publication_capture_path(
    source: Path,
    target: Path,
    *,
    source_identity: tuple[int, int],
) -> Path:
    """Derive a collision-resistant sibling name for one publication attempt."""

    device, inode = source_identity
    names = os.fsencode(f"{source.name}\0{target.name}")
    name_digest = hashlib.sha256(names).hexdigest()[:24]
    return source.parent / (
        f"{publication_capture_prefix(target)}{device:x}-{inode:x}-{name_digest}"
    )


def publication_capture_prefix(target: Path) -> str:
    """Return the target-scoped prefix used for startup discovery."""

    target_digest = hashlib.sha256(os.fsencode(target.name)).hexdigest()[:16]
    return f"{PUBLISH_CAPTURE_PREFIX}{target_digest}-"


def _restore_unverified_capture(
    *,
    capture: Path,
    source: Path,
    captured_metadata: os.stat_result,
    rename_noreplace: RenameNoReplace,
) -> None:
    """Restore an unknown captured entry only to its now-empty source name."""

    try:
        rename_noreplace(capture, source)
    except BaseException as exc:
        raise RuntimeError(
            "directory source changed during private capture; conflicting "
            "source and capture entries were preserved"
        ) from exc
    try:
        restored_descriptor, restored_metadata = _open_nofollow(source)
    except BaseException as exc:
        raise RuntimeError(
            "restored directory source could not be identity-verified"
        ) from exc
    try:
        if _identity(restored_metadata) != _identity(captured_metadata) or stat.S_IFMT(
            restored_metadata.st_mode
        ) != stat.S_IFMT(captured_metadata.st_mode):
            raise RuntimeError(
                "restored directory source identity changed after no-replace restore"
            )
    finally:
        os.close(restored_descriptor)


def publish_directory_noreplace_bound(
    source: Path,
    target: Path,
    *,
    expected_source_identity: tuple[int, int],
    rename_noreplace: RenameNoReplace,
) -> None:
    """Publish only a source captured and verified against one opened inode."""

    try:
        source_descriptor, source_metadata = _open_nofollow(source)
    except BaseException as exc:
        raise RuntimeError("directory source could not be opened safely") from exc
    try:
        if (
            not stat.S_ISDIR(source_metadata.st_mode)
            or _identity(source_metadata) != expected_source_identity
        ):
            raise RuntimeError(
                "directory source identity changed before private capture"
            )
        capture = publication_capture_path(
            source,
            target,
            source_identity=expected_source_identity,
        )
        rename_noreplace(source, capture)
        try:
            captured_descriptor, captured_metadata = _open_nofollow(capture)
        except BaseException as exc:
            raise RuntimeError(
                "private publication capture could not be identity-verified"
            ) from exc
        try:
            if not stat.S_ISDIR(captured_metadata.st_mode) or _identity(
                captured_metadata
            ) != _identity(source_metadata):
                _restore_unverified_capture(
                    capture=capture,
                    source=source,
                    captured_metadata=captured_metadata,
                    rename_noreplace=rename_noreplace,
                )
                raise RuntimeError(
                    "directory source identity changed during private capture"
                )
            rename_noreplace(capture, target)
        finally:
            os.close(captured_descriptor)
        try:
            target_metadata = target.lstat()
        except OSError as exc:
            raise RuntimeError("directory target vanished after atomic rename") from exc
        if (
            not stat.S_ISDIR(target_metadata.st_mode)
            or _identity(target_metadata) != expected_source_identity
        ):
            raise RuntimeError(
                "directory target identity changed during atomic rename; "
                "unknown target preserved"
            )
    finally:
        os.close(source_descriptor)
