# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Hash-bound immutable snapshots for strict checkpoint sidecars."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

from safetensors.torch import load as load_safetensors_bytes

SIDECAR_SNAPSHOT_SCHEMA_VERSION = 1


def strict_integer(value: object, *, label: str, minimum: int = 0) -> int:
    """Reject bool/string coercion for checkpoint scalar fields."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _parse_json_bytes(data: bytes, *, label: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant in {label}: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to parse strict JSON snapshot: {label}") from error


def _fingerprint(result: os.stat_result) -> tuple[int, ...]:
    return (
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_mode),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def _validate_parent_chain(root: Path, relative: PurePosixPath) -> Path:
    current = root
    root_stat = current.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"checkpoint root must be a regular directory: {root}")
    for part in relative.parts[:-1]:
        current /= part
        try:
            result = current.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"missing checkpoint sidecar directory: {current}"
            ) from None
        if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
            raise ValueError(f"sidecar parent must be a regular directory: {current}")
    return root / relative.as_posix()


def _read_stable_regular_file(path: Path) -> tuple[bytes, os.stat_result, str]:
    try:
        path_before = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"missing checkpoint sidecar: {path}") from None
    if (
        stat.S_ISLNK(path_before.st_mode)
        or not stat.S_ISREG(path_before.st_mode)
        or path_before.st_size <= 0
    ):
        raise ValueError(f"sidecar must be a non-empty regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _fingerprint(opened) != _fingerprint(path_before):
            raise RuntimeError(f"sidecar changed while opening: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        final_opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except FileNotFoundError:
        raise RuntimeError(f"sidecar disappeared while reading: {path}") from None
    if _fingerprint(final_opened) != _fingerprint(opened) or _fingerprint(
        path_after
    ) != _fingerprint(opened):
        raise RuntimeError(f"sidecar changed while reading: {path}")
    data = b"".join(chunks)
    if len(data) != opened.st_size:
        raise RuntimeError(f"sidecar byte count changed while reading: {path}")
    return data, opened, digest.hexdigest()


def _relative_paths(paths: Sequence[str]) -> tuple[str, ...]:
    if isinstance(paths, (str, bytes)):
        raise ValueError("sidecar paths must be a sequence")
    normalized: list[str] = []
    for value in paths:
        if not isinstance(value, str):
            raise ValueError("sidecar path must be a string")
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or path.as_posix() != value
            or "." in path.parts
            or ".." in path.parts
        ):
            raise ValueError(f"invalid sidecar path: {value!r}")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError("sidecar paths contain duplicates")
    return tuple(sorted(normalized))


def _inventory_entries(
    payload: object,
    *,
    expected_paths: Sequence[str],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    expected = _relative_paths(expected_paths)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "files",
        "inventory_sha256",
    }:
        raise ValueError("sidecar inventory has an invalid field set")
    schema = strict_integer(
        payload["schema_version"],
        label="sidecar inventory schema_version",
        minimum=1,
    )
    raw_files = payload["files"]
    if schema != SIDECAR_SNAPSHOT_SCHEMA_VERSION or not isinstance(raw_files, list):
        raise ValueError("unsupported sidecar inventory schema")
    entries: dict[str, dict[str, object]] = {}
    normalized_files: list[dict[str, object]] = []
    for raw_entry in raw_files:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("sidecar inventory file entry is invalid")
        relative = raw_entry["path"]
        if not isinstance(relative, str):
            raise ValueError("sidecar inventory path must be a string")
        size = strict_integer(
            raw_entry["size_bytes"],
            label=f"sidecar size for {relative}",
            minimum=1,
        )
        sha256 = raw_entry["sha256"]
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"sidecar SHA256 is invalid for {relative}")
        entry = {"path": relative, "size_bytes": size, "sha256": sha256}
        entries[relative] = entry
        normalized_files.append(entry)
    recorded_paths = tuple(entry["path"] for entry in normalized_files)
    if recorded_paths != expected or len(entries) != len(normalized_files):
        raise ValueError("sidecar inventory path set/order differs from expected paths")
    core = {"schema_version": schema, "files": normalized_files}
    digest = payload["inventory_sha256"]
    if digest != hashlib.sha256(_canonical_json_bytes(core)).hexdigest():
        raise ValueError("sidecar inventory canonical digest is inconsistent")
    return {**core, "inventory_sha256": digest}, entries


@dataclass(frozen=True)
class StableFileSnapshot:
    relative_path: str
    size_bytes: int
    sha256: str
    data: bytes
    parsed_json: object | None

    def json_object(self, *, label: str) -> dict[str, object]:
        if not isinstance(self.parsed_json, dict):
            raise ValueError(f"{label} must contain a JSON object")
        return copy.deepcopy(self.parsed_json)


@dataclass(frozen=True)
class SidecarSnapshot:
    checkpoint_root: Path
    _inventory: dict[str, object]
    files: tuple[StableFileSnapshot, ...]

    @property
    def inventory(self) -> dict[str, object]:
        """Return an isolated copy of the hash-bound inventory."""

        return copy.deepcopy(self._inventory)

    def _file(self, relative_path: str) -> StableFileSnapshot:
        for file_snapshot in self.files:
            if file_snapshot.relative_path == relative_path:
                return file_snapshot
        raise KeyError(f"sidecar snapshot does not contain {relative_path}")

    def json_object(self, relative_path: str, *, label: str) -> dict[str, object]:
        return self._file(relative_path).json_object(label=label)

    def file_bytes(self, relative_path: str) -> bytes:
        return self._file(relative_path).data

    def load_rng_state(
        self,
        *,
        rank: int,
        expected_world_size: int,
    ) -> dict[str, object]:
        from ._strict_resume_state import load_rng_state_payloads

        tensor_name = f"rng_state_rank{rank}.safetensors"
        return load_rng_state_payloads(
            self.json_object(
                f"rng_state_rank{rank}.json",
                label=f"rank {rank} RNG metadata",
            ),
            load_safetensors_bytes(self.file_bytes(tensor_name)),
            rank=rank,
            expected_world_size=expected_world_size,
            tensor_file_name=tensor_name,
        )


def capture_stable_json_file(path: Path, *, label: str) -> StableFileSnapshot:
    resolved = Path(path)
    data, result, sha256 = _read_stable_regular_file(resolved)
    return StableFileSnapshot(
        relative_path=resolved.name,
        size_bytes=int(result.st_size),
        sha256=sha256,
        data=data,
        parsed_json=_parse_json_bytes(data, label=label),
    )


def capture_sidecar_snapshot(
    checkpoint_dir: Path,
    inventory_payload: object,
    expected_paths: Sequence[str],
) -> SidecarSnapshot:
    root = Path(checkpoint_dir).resolve(strict=True)
    inventory, entries = _inventory_entries(
        inventory_payload,
        expected_paths=expected_paths,
    )
    files: list[StableFileSnapshot] = []
    for relative_path in sorted(entries):
        relative = PurePosixPath(relative_path)
        path = _validate_parent_chain(root, relative)
        data, result, sha256 = _read_stable_regular_file(path)
        entry = entries[relative_path]
        if result.st_size != entry["size_bytes"] or sha256 != entry["sha256"]:
            raise ValueError(f"sidecar bytes differ from inventory: {relative_path}")
        files.append(
            StableFileSnapshot(
                relative_path=relative_path,
                size_bytes=int(result.st_size),
                sha256=sha256,
                data=data,
                parsed_json=(
                    _parse_json_bytes(data, label=relative_path)
                    if relative_path.endswith(".json")
                    else None
                ),
            )
        )
    return SidecarSnapshot(
        checkpoint_root=root,
        _inventory=copy.deepcopy(inventory),
        files=tuple(files),
    )


__all__ = (
    "SIDECAR_SNAPSHOT_SCHEMA_VERSION",
    "SidecarSnapshot",
    "StableFileSnapshot",
    "capture_sidecar_snapshot",
    "capture_stable_json_file",
    "strict_integer",
)
