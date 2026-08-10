#!/usr/bin/env python3
# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Inventory an existing Wan2.2 Target-10 tactile golden artifact set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.evaluation.sealed_artifact_io import sha256_file  # noqa: E402
from n0_twam.evaluation.tactile_provenance import write_json_atomic  # noqa: E402
from n0_twam.evaluation.target10_golden_calibration import (  # noqa: E402
    GOLDEN_MANIFEST_SCHEMA_VERSION,
    GOLDEN_MANIFEST_TYPE,
)
from n0_twam.evaluation.target10_reference_contract import (  # noqa: E402
    REFERENCE_CONTRACT,
    TARGET_EPISODE_IDS,
    TARGET_TASKS,
)


def build_golden_manifest(root: Path, output: Path) -> dict[str, object]:
    """Write hashes for the exact 20 target tactile files, without approving them."""

    raw_root = root.expanduser()
    if raw_root.is_symlink():
        raise ValueError("golden root must be a real directory")
    resolved = raw_root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("golden root must be a real directory")
    records = []
    expected_members: set[str] = set()
    for task in TARGET_TASKS:
        for sample_index, episode_id in enumerate(TARGET_EPISODE_IDS):
            for kind in ("pred", "gt"):
                relative = f"{task}/sample_{sample_index:03d}_{kind}_tactile.mp4"
                expected_members.add(relative)
                path = resolved / relative
                if path.is_symlink() or not path.is_file():
                    raise FileNotFoundError(f"missing regular golden MP4: {path}")
                records.append(
                    {
                        "task": task,
                        "sample_index": sample_index,
                        "episode_id": episode_id,
                        "kind": kind,
                        "relative_path": relative,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    actual_members: set[str] = set()
    for member in resolved.rglob("*"):
        if member.is_symlink():
            raise ValueError("golden root must not contain symlinks")
        if member.is_dir():
            continue
        if not member.is_file():
            raise ValueError("golden root contains an unsupported member")
        actual_members.add(member.relative_to(resolved).as_posix())
    if actual_members != expected_members:
        raise ValueError("golden root must contain exactly the 20 expected MP4 files")
    payload: dict[str, object] = {
        "schema_version": GOLDEN_MANIFEST_SCHEMA_VERSION,
        "artifact_type": GOLDEN_MANIFEST_TYPE,
        "contract_id": REFERENCE_CONTRACT.contract_id,
        "published_psnr_display": REFERENCE_CONTRACT.published_psnr_display,
        "published_ssim_display": REFERENCE_CONTRACT.published_ssim_display,
        "files": records,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, payload)
    return {
        "manifest": str(output.resolve(strict=True)),
        "sha256": sha256_file(output),
        "files": len(records),
        "approval_status": "unapproved_until_calibration_passes",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        json.dumps(build_golden_manifest(args.golden_root, args.output), sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
