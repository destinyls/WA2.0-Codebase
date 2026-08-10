#!/usr/bin/env python3
"""Download the frozen official Franka dataset through a verified mirror."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from n0_twam.integrations.worldarena.franka_download import (
    DEFAULT_ENDPOINT,
    download_franka_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = download_franka_dataset(
        inventory_path=args.inventory,
        output_root=args.output_root,
        receipt_path=args.receipt,
        endpoint=args.endpoint,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "downloaded_files": result.downloaded_files,
                "reused_files": result.reused_files,
                "total_files": result.total_files,
                "total_bytes": result.total_bytes,
                "receipt": str(result.receipt_path),
                "receipt_sha256": result.receipt_sha256,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
