#!/usr/bin/env python3
"""Freeze and download the pinned official WorldArena AgileX subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from n0_twam.integrations.worldarena.agilex_official_download import (
    DEFAULT_ENDPOINT,
    download_agilex_dataset,
    fetch_agilex_download_inventory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.inventory.exists():
        fetch_agilex_download_inventory(
            output=args.inventory,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout_seconds,
        )
    result = download_agilex_dataset(
        inventory_path=args.inventory,
        output_root=args.raw_root,
        receipt_path=args.receipt,
        endpoint=args.endpoint,
        workers=args.workers,
        retries=args.retries,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
