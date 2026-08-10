#!/usr/bin/env python3
"""Fetch and freeze the official Franka dataset inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from n0_twam.integrations.worldarena.franka_inventory_fetch import (
    DEFAULT_API_ENDPOINT,
    fetch_official_franka_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_API_ENDPOINT)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    result = fetch_official_franka_inventory(
        output=args.output,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
