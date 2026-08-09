# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Small fail-closed primitives for sealed NumPy directory artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import numpy.typing as npt


def canonical_json(payload: object) -> bytes:
    """Serialize JSON deterministically while rejecting non-finite numbers."""

    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"artifact JSON contains non-finite constant {value}")


def read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid artifact JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"artifact JSON must contain an object: {path}")
    return payload


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz(path: Path, arrays: Mapping[str, npt.NDArray[np.generic]]) -> None:
    with path.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def publish_sealed_npz_directory(
    target: Path,
    *,
    schema_version: int,
    artifact_type: str,
    metadata_name: str,
    payload_name: str,
    seal_name: str,
    metadata: Mapping[str, object],
    arrays: Mapping[str, npt.NDArray[np.generic]],
) -> Path:
    """Write, hash, seal, then expose a directory with one rename."""

    if target.exists():
        raise FileExistsError(f"sealed artifact output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp.", dir=target.parent))
    try:
        _write_npz(temporary / payload_name, arrays)
        _write_bytes(temporary / metadata_name, canonical_json(metadata) + b"\n")
        files = {
            name: {
                "size_bytes": (temporary / name).stat().st_size,
                "sha256": sha256_file(temporary / name),
            }
            for name in (metadata_name, payload_name)
        }
        seal: dict[str, object] = {
            "schema_version": schema_version,
            "artifact_type": artifact_type,
            "files": files,
        }
        seal["seal_sha256"] = sha256_bytes(canonical_json(seal))
        _write_bytes(temporary / seal_name, canonical_json(seal) + b"\n")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def verify_sealed_file_inventory(
    artifact: Path,
    *,
    schema_version: int,
    artifact_type: str,
    metadata_name: str,
    payload_name: str,
    seal_name: str,
) -> tuple[Path, str, dict[str, str]]:
    """Verify the exact inventory, self-hashed seal, sizes, and file hashes."""

    root = Path(artifact).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    expected_names = {metadata_name, payload_name, seal_name}
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise ValueError(f"sealed artifact filenames differ: {sorted(actual_names)}")
    for name in expected_names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"sealed artifact member must be a regular file: {name}")

    seal = read_json_object(root / seal_name)
    seal_sha256 = validate_sha256(seal.get("seal_sha256"), label="seal SHA256")
    unsigned = {key: value for key, value in seal.items() if key != "seal_sha256"}
    if sha256_bytes(canonical_json(unsigned)) != seal_sha256:
        raise ValueError("sealed artifact seal sha256 verification failed")
    if set(unsigned) != {"schema_version", "artifact_type", "files"}:
        raise ValueError("sealed artifact seal fields are invalid")
    if (
        unsigned["schema_version"] != schema_version
        or unsigned["artifact_type"] != artifact_type
    ):
        raise ValueError("sealed artifact contract is unsupported")
    files = unsigned["files"]
    if not isinstance(files, dict) or set(files) != {metadata_name, payload_name}:
        raise ValueError("sealed artifact file inventory is invalid")

    file_sha256: dict[str, str] = {}
    for name, record in files.items():
        if not isinstance(record, dict) or set(record) != {"size_bytes", "sha256"}:
            raise ValueError("sealed artifact file record is invalid")
        expected_size = nonnegative_integer(
            record["size_bytes"], label=f"{name} size_bytes"
        )
        expected_hash = validate_sha256(record["sha256"], label=f"{name} SHA256")
        path = root / name
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise ValueError(f"sealed artifact {name} sha256 verification failed")
        file_sha256[name] = expected_hash
    return root, seal_sha256, file_sha256
