# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Frozen Hugging Face inventory contract for the official Franka dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

OFFICIAL_REPO_ID = "WorldArena/WorldArena2.0_Franka_FR3"
OFFICIAL_REVISION = "aed59b39c5a903be5e435c13c0ed1efdd54d5ad9"
OFFICIAL_RECORDS_SHA256 = (
    "67118a93230e13a5ecf8072df9cad4b30882367471017b4f1b49e43b6c8d4635"
)
OFFICIAL_FILE_COUNT = 3602
OFFICIAL_TOTAL_BYTES = 59_922_514_413
OFFICIAL_TASKS = ("clear_up", "pour", "wipe")
OFFICIAL_EPISODES_PER_TASK = 200


def canonical_sha256(payload: object) -> str:
    """Hash canonical compact JSON exactly as the frozen inventory generator."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path, *, size: int) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(raw_path: object) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("manifest record path must be a non-empty string")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"manifest record path is unsafe: {raw_path!r}")
    return path.as_posix()


@dataclass(frozen=True)
class FrankaFileRecord:
    relative_path: str
    size_bytes: int
    identity_kind: str
    identity_digest: str

    def verify(self, path: Path) -> None:
        metadata = path.lstat()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"dataset file must be regular and non-symlink: {path}")
        if metadata.st_size != self.size_bytes:
            raise ValueError(
                f"dataset file size mismatch for {self.relative_path}: "
                f"{metadata.st_size} vs {self.size_bytes}"
            )
        if self.identity_kind == "content_sha256":
            actual = sha256_file(path)
        elif self.identity_kind == "git_blob_sha1":
            actual = git_blob_sha1(path, size=self.size_bytes)
        else:  # pragma: no cover - guarded when loading
            raise ValueError(f"unsupported identity kind: {self.identity_kind}")
        if actual != self.identity_digest:
            raise ValueError(f"dataset file identity mismatch for {self.relative_path}")


@dataclass(frozen=True)
class FrankaDatasetInventory:
    source_path: Path
    repo_id: str
    revision: str
    records_sha256: str
    records: tuple[FrankaFileRecord, ...]
    total_bytes: int

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.source_path)

    def episode_paths(self) -> tuple[str, ...]:
        return tuple(
            record.relative_path
            for record in self.records
            if record.relative_path.endswith("/episode.hdf5")
        )


def _parse_identity(payload: object) -> tuple[str, str]:
    if not isinstance(payload, Mapping):
        raise ValueError("manifest record identity must be an object")
    kind = payload.get("kind")
    if kind == "content_sha256":
        digest = payload.get("sha256")
        expected_length = 64
    elif kind == "git_blob_sha1":
        digest = payload.get("git_blob_sha1")
        expected_length = 40
    else:
        raise ValueError(f"unsupported manifest identity kind: {kind!r}")
    if (
        not isinstance(digest, str)
        or len(digest) != expected_length
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"invalid {kind} digest")
    return kind, digest


def load_franka_inventory(path: Path) -> FrankaDatasetInventory:
    """Load and pin the exact official 600-episode inventory."""

    source = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Franka inventory: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("Franka inventory must contain a JSON object")
    if payload.get("status") != "frozen" or payload.get("schema_version") != 1:
        raise ValueError("Franka inventory must be frozen schema version 1")
    if payload.get("repo_id") != OFFICIAL_REPO_ID:
        raise ValueError("Franka inventory repository is not the official dataset")
    revisions = (payload.get("requested_revision"), payload.get("resolved_revision"))
    if revisions != (OFFICIAL_REVISION, OFFICIAL_REVISION):
        raise ValueError("Franka inventory revision differs from the pinned revision")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("Franka inventory records must be a list")
    if canonical_sha256(raw_records) != OFFICIAL_RECORDS_SHA256:
        raise ValueError("Franka inventory canonical record hash mismatch")
    if payload.get("canonical_records_sha256") != OFFICIAL_RECORDS_SHA256:
        raise ValueError("Franka inventory declares a different record hash")

    records: list[FrankaFileRecord] = []
    seen_paths: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValueError("Franka inventory record must be an object")
        relative_path = _safe_relative_path(raw_record.get("path"))
        if relative_path in seen_paths:
            raise ValueError(f"duplicate manifest path: {relative_path}")
        seen_paths.add(relative_path)
        size = raw_record.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid manifest size for {relative_path}")
        kind, digest = _parse_identity(raw_record.get("identity"))
        records.append(
            FrankaFileRecord(
                relative_path=relative_path,
                size_bytes=size,
                identity_kind=kind,
                identity_digest=digest,
            )
        )
    if len(records) != OFFICIAL_FILE_COUNT or payload.get("file_count") != len(records):
        raise ValueError("Franka inventory file count mismatch")
    total_bytes = sum(record.size_bytes for record in records)
    if total_bytes != OFFICIAL_TOTAL_BYTES or payload.get("total_bytes") != total_bytes:
        raise ValueError("Franka inventory total byte count mismatch")

    hdf5_paths = [
        PurePosixPath(record.relative_path)
        for record in records
        if record.relative_path.endswith("/episode.hdf5")
    ]
    task_counts = Counter(path.parts[0] for path in hdf5_paths)
    expected_counts = Counter(
        {task: OFFICIAL_EPISODES_PER_TASK for task in OFFICIAL_TASKS}
    )
    if task_counts != expected_counts:
        raise ValueError(f"Franka episode task counts mismatch: {dict(task_counts)}")
    for task in OFFICIAL_TASKS:
        expected = {f"episode_{index:03d}" for index in range(200)}
        actual = {
            path.parts[1]
            for path in hdf5_paths
            if path.parts[0] == task and len(path.parts) == 3
        }
        if actual != expected:
            raise ValueError(f"Franka episode IDs are incomplete for {task}")
    return FrankaDatasetInventory(
        source_path=source,
        repo_id=OFFICIAL_REPO_ID,
        revision=OFFICIAL_REVISION,
        records_sha256=OFFICIAL_RECORDS_SHA256,
        records=tuple(records),
        total_bytes=total_bytes,
    )


__all__ = (
    "FrankaDatasetInventory",
    "FrankaFileRecord",
    "OFFICIAL_EPISODES_PER_TASK",
    "OFFICIAL_FILE_COUNT",
    "OFFICIAL_RECORDS_SHA256",
    "OFFICIAL_REPO_ID",
    "OFFICIAL_REVISION",
    "OFFICIAL_TASKS",
    "OFFICIAL_TOTAL_BYTES",
    "canonical_sha256",
    "git_blob_sha1",
    "load_franka_inventory",
    "sha256_file",
)
