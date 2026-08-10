#!/usr/bin/env python3
# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Run the complete calibrated Target-10 reference9 evaluation."""

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
    group.add_argument("--print-request-template", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.print_request_template:
        print(
            json.dumps(
                target10_v18_step1500_reference_request_template(),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    request = load_target10_reference_request(args.request)
    result = run_target10_reference_evaluation(request)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
