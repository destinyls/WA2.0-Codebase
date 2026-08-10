"""Content identity for every base-model component used by latent encoders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ENCODER_SOURCE_SCHEMA_VERSION = 1
ENCODER_SOURCE_SUBDIRS = ("vae", "tokenizer", "text_encoder")
_REQUIRED_FILES = (
    "vae/config.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return digest


def build_encoder_source_identity(model_root: Path) -> dict[str, object]:
    """Hash the exact VAE/tokenizer/text-encoder files used for encoding."""

    root = Path(model_root).resolve(strict=True)
    for relative_path in _REQUIRED_FILES:
        if not (root / relative_path).is_file():
            raise FileNotFoundError(
                f"base model is missing encoder source file: {root / relative_path}"
            )
    paths = sorted(
        path
        for subdir in ENCODER_SOURCE_SUBDIRS
        for path in (root / subdir).rglob("*")
        if path.is_file()
    )
    if not paths:
        raise ValueError(f"base-model encoder source inventory is empty: {root}")
    files: list[dict[str, object]] = []
    for path in paths:
        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            raise ValueError(f"base-model encoder source file is empty: {path}")
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": size_bytes,
                "sha256": _sha256_file(path),
            }
        )
    core = {
        "schema_version": ENCODER_SOURCE_SCHEMA_VERSION,
        "files": files,
    }
    return {
        **core,
        "identity_sha256": hashlib.sha256(_canonical_bytes(core)).hexdigest(),
    }


def validate_encoder_source_identity(payload: object) -> dict[str, object]:
    """Validate a persisted encoder-source inventory and its canonical digest."""

    if not isinstance(payload, dict):
        raise ValueError("encoder source identity must be a JSON object")
    if int(payload.get("schema_version", 0)) != ENCODER_SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported encoder source identity schema")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("encoder source identity has no files")
    files: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError("encoder source file entry must be an object")
        relative_path = str(entry.get("relative_path", ""))
        parsed = Path(relative_path)
        if (
            not relative_path
            or parsed.is_absolute()
            or ".." in parsed.parts
            or relative_path in seen
        ):
            raise ValueError(f"invalid encoder source path: {relative_path!r}")
        seen.add(relative_path)
        size_bytes = int(entry.get("size_bytes", 0))
        if size_bytes <= 0:
            raise ValueError(f"invalid encoder source size: {relative_path}")
        files.append(
            {
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "sha256": _validate_sha256(
                    entry.get("sha256"), f"encoder source {relative_path} sha256"
                ),
            }
        )
    if files != sorted(files, key=lambda item: str(item["relative_path"])):
        raise ValueError("encoder source file inventory must be sorted")
    core: dict[str, Any] = {
        "schema_version": ENCODER_SOURCE_SCHEMA_VERSION,
        "files": files,
    }
    identity_sha256 = _validate_sha256(
        payload.get("identity_sha256"), "encoder source identity_sha256"
    )
    if identity_sha256 != hashlib.sha256(_canonical_bytes(core)).hexdigest():
        raise ValueError("encoder source identity digest is inconsistent")
    return {**core, "identity_sha256": identity_sha256}


def load_encoder_source_identity(path: Path) -> dict[str, object]:
    """Load and validate a run-local encoder-source identity cache."""

    source_path = Path(path)
    if source_path.is_symlink() or not source_path.is_file():
        raise FileNotFoundError(
            f"encoder source identity cache must be a regular file: {source_path}"
        )
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unable to read encoder source identity cache: {source_path}"
        ) from exc
    return validate_encoder_source_identity(payload)
