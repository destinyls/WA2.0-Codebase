# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Honest, secret-free provenance for public local-package training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from n0_twam import __version__
from n0_twam.checkpointing.runtime_provenance import (
    LOCAL_CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION,
    LOCAL_EXECUTION_TIER,
    LOCAL_RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION,
    validate_checkpoint_invocation_identity,
    validate_runtime_source_identity,
)

_DISTRIBUTIONS = (
    "accelerate",
    "av",
    "datasets",
    "diffusers",
    "draccus",
    "einops",
    "h5py",
    "imageio",
    "imageio-ffmpeg",
    "jsonlines",
    "lerobot",
    "n0-twam",
    "numpy",
    "opencv-python",
    "pandas",
    "Pillow",
    "pyarrow",
    "safetensors",
    "scikit-image",
    "scipy",
    "torch",
    "torchvision",
    "transformers",
    "wandb",
)


def canonical_json(payload: object) -> bytes:
    """Encode a JSON payload with deterministic bytes."""

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
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable_json(path: Path, payload: Mapping[str, object]) -> str:
    """Create one read-only JSON file and return its byte SHA256."""

    raw = canonical_json(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return sha256_bytes(raw)


def build_code_manifest(package_root: Path | None = None) -> dict[str, object]:
    """Hash packaged Python sources without including repository-local outputs."""

    root = (
        Path(__file__).resolve().parents[1]
        if package_root is None
        else Path(package_root).resolve(strict=True)
    )
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or not path.is_file() or path.is_symlink():
            continue
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    if not files:
        raise ValueError(f"no packaged Python sources found below {root}")
    core: dict[str, object] = {"schema_version": 1, "files": files}
    return {**core, "manifest_sha256": sha256_bytes(canonical_json(core))}


def package_import_root() -> Path:
    """Return the sole import root used by public training child processes."""

    package_root = Path(__file__).resolve(strict=True).parents[1]
    if package_root.name != "n0_twam" or not (package_root / "__init__.py").is_file():
        raise RuntimeError("unable to resolve the active n0_twam package root")
    return package_root.parent


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_environment_manifest() -> dict[str, object]:
    """Capture an allowlisted runtime description without copying process env."""

    core: dict[str, object] = {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "distributions": _distribution_versions(),
    }
    return {**core, "manifest_sha256": sha256_bytes(canonical_json(core))}


@dataclass(frozen=True)
class LocalProvenance:
    """Immutable identities and files attached to one public invocation."""

    invocation_id: str
    code_manifest_path: Path
    environment_manifest_path: Path
    launch_receipt_path: Path
    runtime_source_identity: dict[str, object]
    checkpoint_invocation_identity: dict[str, object]

    def environment(self) -> dict[str, str]:
        return {
            "N0_TRACK31_EXECUTION_TIER": LOCAL_EXECUTION_TIER,
            "N0_TRACK31_CODE_MANIFEST_SHA256": str(
                self.runtime_source_identity["code_manifest_sha256"]
            ),
            "N0_TRACK31_ENVIRONMENT_MANIFEST_SHA256": str(
                self.runtime_source_identity["environment_manifest_sha256"]
            ),
            "N0_TRACK31_PACKAGE_VERSION": str(
                self.runtime_source_identity["package_version"]
            ),
            "N0_TRACK31_INVOCATION_ID": self.invocation_id,
            "N0_TRACK31_LAUNCH_RECEIPT_SHA256": str(
                self.checkpoint_invocation_identity["launch_receipt_sha256"]
            ),
        }


def prepare_local_provenance(
    *,
    output_root: Path,
    run_id: str,
    request_sha256: str,
    launch_plan: Mapping[str, object],
) -> LocalProvenance:
    """Materialize immutable, non-formal source and invocation receipts."""

    invocation_id = f"{run_id}-{uuid.uuid4().hex[:12]}"
    provenance_root = Path(output_root) / "provenance" / invocation_id
    code_manifest = build_code_manifest()
    environment_manifest = build_environment_manifest()
    code_path = provenance_root / "code_manifest.json"
    environment_path = provenance_root / "environment_manifest.json"
    code_file_sha = write_immutable_json(code_path, code_manifest)
    environment_file_sha = write_immutable_json(environment_path, environment_manifest)
    if code_file_sha != sha256_file(code_path):
        raise RuntimeError("code manifest changed while publishing provenance")
    if environment_file_sha != sha256_file(environment_path):
        raise RuntimeError("environment manifest changed while publishing provenance")
    source_identity = validate_runtime_source_identity(
        {
            "schema_version": LOCAL_RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION,
            "execution_tier": LOCAL_EXECUTION_TIER,
            "code_manifest_sha256": code_manifest["manifest_sha256"],
            "environment_manifest_sha256": environment_manifest["manifest_sha256"],
            "empty_embedding_sha256": launch_plan["empty_embedding_sha256"],
            "package_version": __version__,
        }
    )
    receipt_core: dict[str, object] = {
        "schema_version": 1,
        "execution_tier": LOCAL_EXECUTION_TIER,
        "formal_track31": False,
        "leaderboard_eligible": False,
        "invocation_id": invocation_id,
        "request_sha256": request_sha256,
        "runtime_source_identity": source_identity,
        "code_manifest_file_sha256": code_file_sha,
        "environment_manifest_file_sha256": environment_file_sha,
        "launch_plan": dict(launch_plan),
    }
    receipt_path = provenance_root / "launch_receipt.json"
    receipt_sha = write_immutable_json(receipt_path, receipt_core)
    invocation_identity = validate_checkpoint_invocation_identity(
        {
            "schema_version": LOCAL_CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION,
            "execution_tier": LOCAL_EXECUTION_TIER,
            "invocation_id": invocation_id,
            "launch_receipt_sha256": receipt_sha,
        }
    )
    return LocalProvenance(
        invocation_id=invocation_id,
        code_manifest_path=code_path,
        environment_manifest_path=environment_path,
        launch_receipt_path=receipt_path,
        runtime_source_identity=source_identity,
        checkpoint_invocation_identity=invocation_identity,
    )


__all__ = (
    "LocalProvenance",
    "build_code_manifest",
    "build_environment_manifest",
    "canonical_json",
    "package_import_root",
    "prepare_local_provenance",
    "sha256_bytes",
    "sha256_file",
    "write_immutable_json",
)
