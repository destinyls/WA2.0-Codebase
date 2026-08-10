#!/usr/bin/env python3
"""Create an immutable N0-TWAM Franka serve bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from n0_twam.integrations.worldarena.franka_serve_bundle import (
    build_franka_serve_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-identity-sha256", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--normalizer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_franka_serve_bundle(
        checkpoint=args.checkpoint,
        checkpoint_identity_sha256=args.checkpoint_identity_sha256,
        base_model=args.base_model,
        normalizer=args.normalizer,
        normalizer_sha256=args.normalizer_sha256,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
