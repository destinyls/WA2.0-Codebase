#!/usr/bin/env python3
# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Run v18/1500 through the calibrated Target-10 reference9 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.evaluation.target10_evaluation_template import (  # noqa: E402
    target10_v18_step1500_reference_request_template,
)
from n0_twam.evaluation.target10_reference_pipeline import (  # noqa: E402
    load_target10_reference_request,
    run_target10_reference_evaluation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request", type=Path)
    group.add_argument(
        "--print-template", "--print-request-template", action="store_true"
    )
    parser.add_argument("--save-template", type=Path)
    return parser


def _write_template(payload: dict[str, object], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.print_template:
        payload = target10_v18_step1500_reference_request_template()
        _write_template(payload, args.save_template)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.save_template is not None:
        raise ValueError("--save-template is only valid with --print-template")
    if args.request is None:
        raise RuntimeError("argparse accepted neither request nor template mode")
    request = load_target10_reference_request(args.request)
    summary = run_target10_reference_evaluation(request)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
