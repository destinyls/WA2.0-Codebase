#!/usr/bin/env python3
"""Re-open every manifest source and reject non-contained HDF5 storage."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.integrations.univtac.convert_lerobot import (  # noqa: E402
    _validate_hdf5_storage,
)
from n0_twam.integrations.univtac.hdf5_reader import (  # noqa: E402
    open_verified_hdf5,
)
from n0_twam.integrations.univtac.manifest import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    load_dataset_manifest,
)


def audit_manifest_sources(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_entries: int,
) -> dict[str, object]:
    """Hash stable file descriptors and audit HDF5 storage modes."""

    manifest = load_dataset_manifest(manifest_path, verify_sources=False)
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError("formal source audit requires a schema-v4 manifest")
    if manifest.manifest_sha256 != expected_manifest_sha256:
        message = "source manifest identity differs from the expected SHA256"
        raise ValueError(message)
    if len(manifest.entries) != expected_entries:
        raise ValueError(
            "source manifest episode count mismatch: "
            f"{len(manifest.entries)} vs {expected_entries}"
        )

    started_at = time.monotonic()
    split_counts: Counter[str] = Counter()
    bytes_audited = 0
    for record in manifest.entries:
        with open_verified_hdf5(record) as handle:
            _validate_hdf5_storage(handle, source_label=record.relative_path)
        split_counts[record.split] += 1
        bytes_audited += int(record.size_bytes)

    return {
        "schema_version": 1,
        "status": "source_storage_audit_passed",
        "manifest_sha256": manifest.manifest_sha256,
        "episode_count": len(manifest.entries),
        "split_counts": dict(sorted(split_counts.items())),
        "bytes_audited": bytes_audited,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "checks": [
            "stable_regular_file_descriptor",
            "source_sha256_before_and_after_hdf5_read",
            "temporal_selection",
            "no_external_link",
            "no_external_dataset_storage",
            "no_virtual_dataset",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-entries", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.expected_entries <= 0:
        raise ValueError("--expected-entries must be positive")
    if len(args.expected_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.expected_manifest_sha256
    ):
        raise ValueError("--expected-manifest-sha256 must be lowercase SHA256")
    report = audit_manifest_sources(
        args.manifest.resolve(strict=True),
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_entries=args.expected_entries,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
