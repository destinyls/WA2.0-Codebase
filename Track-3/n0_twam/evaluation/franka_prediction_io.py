# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Stable file snapshots shared by Franka offline evaluation commands."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StablePredictionInput:
    path: Path
    raw: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    mode: int


def _fingerprint(result: os.stat_result) -> tuple[int, ...]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_mode,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def capture_prediction_input(path: Path) -> StablePredictionInput:
    lexical = Path(path).expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    try:
        before = lexical.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(lexical) from None
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("prediction artifact must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical, flags)
    try:
        opened = os.fstat(descriptor)
        if _fingerprint(opened) != _fingerprint(before):
            raise ValueError("prediction artifact changed while it was opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = lexical.lstat()
    if _fingerprint(after) != _fingerprint(opened) or _fingerprint(
        current
    ) != _fingerprint(opened):
        raise ValueError("prediction artifact changed while it was read")
    raw = b"".join(chunks)
    if len(raw) != opened.st_size:
        raise ValueError("prediction artifact byte count changed while it was read")
    return StablePredictionInput(
        path=lexical,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        device=opened.st_dev,
        inode=opened.st_ino,
        size=opened.st_size,
        mtime_ns=opened.st_mtime_ns,
        ctime_ns=opened.st_ctime_ns,
        mode=opened.st_mode,
    )


def require_prediction_unchanged(snapshot: StablePredictionInput) -> None:
    current = capture_prediction_input(snapshot.path)
    if (
        current.sha256,
        current.device,
        current.inode,
        current.size,
        current.mtime_ns,
        current.ctime_ns,
        current.mode,
    ) != (
        snapshot.sha256,
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
        snapshot.mode,
    ):
        raise ValueError("prediction artifact changed during evaluation")


__all__ = (
    "StablePredictionInput",
    "capture_prediction_input",
    "require_prediction_unchanged",
)
