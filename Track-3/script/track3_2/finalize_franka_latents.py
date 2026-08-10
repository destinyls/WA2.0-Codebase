#!/usr/bin/env python3
"""Audit all Franka video latents and publish the immutable inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from n0_twam.integrations.worldarena.franka_artifacts import (
    finalize_franka_video_latents,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_franka_video_latents(
        artifact_root=args.artifact_root,
        lerobot_root=args.lerobot_root,
        base_model=args.base_model,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
