#!/usr/bin/env python3
# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Stage B: evaluate a sealed prediction artifact against raw UniVTAC HDF5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.evaluation.raw_tactile_quality import (  # noqa: E402
    evaluate_raw_tactile_quality,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-artifact", type=Path, required=True)
    parser.add_argument("--evaluation-view-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-report", type=Path, required=True)
    parser.add_argument(
        "--official-metric-script",
        type=Path,
        required=True,
        help="Exact val_psnr_ssim.py file to hash and invoke.",
    )
    parser.add_argument(
        "--official-metric-sha256",
        required=True,
        help="Pre-approved SHA256 of the exact metric implementation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_raw_tactile_quality(
        prediction_artifact=args.prediction_artifact,
        evaluation_view_path=args.evaluation_view_manifest,
        raw_root=args.raw_root,
        manifest_path=args.manifest,
        conversion_report_path=args.conversion_report,
        official_metric_script=args.official_metric_script,
        expected_official_metric_sha256=args.official_metric_sha256,
        output=args.output,
        fps=args.fps,
    )
    print(json.dumps(report["official_metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
