# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Private run-root checkpoint snapshots for race-free transformer loading."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
    validate_transformer_identity_match,
)
from .sidecar_snapshot import SidecarSnapshot

_SNAPSHOT_PARENT_NAME = ".verified_parent_snapshots"
_SNAPSHOT_PREFIX = "stage-a-parent-"


def validate_checkpoint_run_root_boundary(
    checkpoint_root: Path,
    run_root: Path,
) -> tuple[Path, Path]:
    """Require immutable input and writable output trees to be disjoint."""

    checkpoint = Path(checkpoint_root).resolve(strict=True)
    run = Path(run_root).resolve(strict=False)
    if checkpoint == run or checkpoint in run.parents or run in checkpoint.parents:
        raise ValueError(
            "formal checkpoint mount and writable run root must be disjoint"
        )
    return checkpoint, run


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class RuntimeTransformerSnapshot:
    checkpoint_root: Path
    transformer_dir: Path
    snapshot_parent: Path

    def cleanup(self) -> None:
        root = self.checkpoint_root
        try:
            result = root.lstat()
        except FileNotFoundError:
            return
        if (
            root.parent != self.snapshot_parent
            or not root.name.startswith(_SNAPSHOT_PREFIX)
            or stat.S_ISLNK(result.st_mode)
            or not stat.S_ISDIR(result.st_mode)
        ):
            raise RuntimeError("refusing to clean an unowned runtime snapshot")
        shutil.rmtree(root)


def create_runtime_transformer_snapshot(
    *,
    checkpoint_root: Path,
    run_root: Path,
    sidecars: SidecarSnapshot,
    transformer_identity: object,
) -> RuntimeTransformerSnapshot:
    """Create an exclusive config+hardlink checkpoint consumed by the loader."""

    source_root, resolved_run_root = validate_checkpoint_run_root_boundary(
        checkpoint_root,
        run_root,
    )
    resolved_run_root.mkdir(parents=True, exist_ok=True)
    snapshot_parent = resolved_run_root / _SNAPSHOT_PARENT_NAME
    snapshot_parent.mkdir(mode=0o700, exist_ok=True)
    if snapshot_parent.is_symlink() or not snapshot_parent.is_dir():
        raise ValueError("runtime snapshot parent must be a regular directory")
    snapshot_root = Path(tempfile.mkdtemp(prefix=_SNAPSHOT_PREFIX, dir=snapshot_parent))
    transformer_dir = snapshot_root / "transformer"
    transformer_dir.mkdir(mode=0o700)
    runtime_snapshot = RuntimeTransformerSnapshot(
        checkpoint_root=snapshot_root,
        transformer_dir=transformer_dir,
        snapshot_parent=snapshot_parent,
    )
    try:
        _write_exclusive(
            transformer_dir / "config.json",
            sidecars.file_bytes("transformer/config.json"),
        )
        os.link(
            source_root / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
            transformer_dir / TRANSFORMER_WEIGHTS_FILENAME,
            follow_symlinks=False,
        )
        linked_identity = audit_transformer_checkpoint(
            transformer_dir / TRANSFORMER_WEIGHTS_FILENAME,
            expected_action_dim=8,
        )
        validate_transformer_identity_match(
            transformer_identity,
            linked_identity,
            expected_action_dim=8,
            label="private runtime parent snapshot",
        )
        return runtime_snapshot
    except Exception:
        runtime_snapshot.cleanup()
        raise


def _config_value(config: object, field: str) -> object:
    if isinstance(config, Mapping):
        return config.get(field)
    return getattr(config, field, None)


def _canonical_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def validate_loaded_transformer_architecture(
    runtime_config: object,
    *,
    verified_config: Mapping[str, object],
    overrides: Mapping[str, object],
) -> None:
    """Compare every persisted non-private architecture field after loading."""

    for field, saved_value in verified_config.items():
        if field.startswith("_"):
            continue
        expected = overrides.get(field, saved_value)
        actual = _config_value(runtime_config, field)
        if _canonical_value(actual) != _canonical_value(expected):
            raise ValueError(
                "loaded transformer architecture mismatch for "
                f"{field}: runtime={actual!r} verified={expected!r}"
            )


__all__ = (
    "RuntimeTransformerSnapshot",
    "create_runtime_transformer_snapshot",
    "validate_loaded_transformer_architecture",
    "validate_checkpoint_run_root_boundary",
)
