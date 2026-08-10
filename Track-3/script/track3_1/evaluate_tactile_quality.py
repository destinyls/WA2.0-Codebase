#!/usr/bin/env python3
# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Compute strict PSNR/SSIM for decoded tactile prediction frames."""

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.evaluation.tactile_provenance import write_json_atomic  # noqa: E402
from n0_twam.evaluation.tactile_quality import (  # noqa: E402
    evaluate_tactile_prediction_quality,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-first-frames", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = evaluate_tactile_prediction_quality(
        prediction_root=args.prediction_root,
        ground_truth_root=args.ground_truth_root,
        skip_first_frames=args.skip_first_frames,
    )
    write_json_atomic(args.output, report)
    print(report["overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
