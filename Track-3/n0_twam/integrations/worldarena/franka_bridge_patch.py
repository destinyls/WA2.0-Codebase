# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Exact participant-side patch for the pinned WorldArena Franka bridge."""

from __future__ import annotations

import subprocess
from importlib import resources
from pathlib import Path

from .franka_official_worker import (
    BRIDGE_RELATIVE_PATH,
    PINNED_ORIGINAL_BRIDGE_SHA256,
    PINNED_PATCHED_BRIDGE_SHA256,
    PINNED_WORLD_ARENA_REVISION,
    _changed_paths,
    _git,
    _regular_file,
    _root_path,
    _sha256_file,
    audit_worldarena_franka_bridge,
)

PATCH_RESOURCE = "patches/worldarena_6f5a981_franka_wxyz.patch"


def apply_pinned_worldarena_bridge_patch(
    worldarena_root: Path,
) -> dict[str, object]:
    """Apply the exact WXYZ bridge fix to a clean pinned official checkout."""
    root = _root_path(worldarena_root)
    revision = _git(root, "rev-parse", "HEAD")
    if revision != PINNED_WORLD_ARENA_REVISION:
        raise ValueError("bridge patch supports only the pinned WorldArena revision")
    bridge = _regular_file(root / BRIDGE_RELATIVE_PATH, label="WorldArena bridge")
    current_sha = _sha256_file(bridge)
    changed = _changed_paths(root)
    if current_sha == PINNED_PATCHED_BRIDGE_SHA256:
        return audit_worldarena_franka_bridge(
            root,
            expected_revision=PINNED_WORLD_ARENA_REVISION,
            expected_bridge_sha256=PINNED_PATCHED_BRIDGE_SHA256,
        )
    if current_sha != PINNED_ORIGINAL_BRIDGE_SHA256 or changed:
        raise ValueError("pinned bridge patch requires an otherwise clean checkout")
    patch_resource = resources.files(__package__).joinpath(PATCH_RESOURCE)
    with resources.as_file(patch_resource) as patch_path:
        _regular_file(patch_path, label="packaged WorldArena bridge patch")
        subprocess.run(
            ("git", "-C", str(root), "apply", "--check", str(patch_path)),
            check=True,
        )
        subprocess.run(
            ("git", "-C", str(root), "apply", str(patch_path)),
            check=True,
        )
    try:
        return audit_worldarena_franka_bridge(
            root,
            expected_revision=PINNED_WORLD_ARENA_REVISION,
            expected_bridge_sha256=PINNED_PATCHED_BRIDGE_SHA256,
        )
    except Exception:
        with resources.as_file(patch_resource) as patch_path:
            subprocess.run(
                ("git", "-C", str(root), "apply", "--reverse", str(patch_path)),
                check=True,
            )
        raise


__all__ = ("apply_pinned_worldarena_bridge_patch",)
