#!/usr/bin/env python3
"""CLI and stable import surface for strict Stage A collective validation."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.track3_1.track31_collective_smoke_contract import (  # noqa: E402
    HCU_PER_NODE,
    parse_assignments,
    parse_csv,
)
from script.track3_1.track31_collective_smoke_io import (  # noqa: E402
    CollectiveSmokeValidationError,
)
from script.track3_1.track31_collective_smoke_validation import (  # noqa: E402
    validate_collective_smoke_report,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--expected-report-sha256")
    parser.add_argument("--expected-smoke-id", "--smoke-id", dest="smoke_id")
    parser.add_argument("--expected-nodes")
    parser.add_argument("--expected-hcu-order")
    parser.add_argument(
        "--expected-nccl-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
    )
    parser.add_argument("--expected-image-id")
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument("--expected-overlay-manifest-sha256")
    parser.add_argument("--expected-run-role")
    parser.add_argument("--expected-batch-size", type=int)
    parser.add_argument("--expected-gradient-accumulation-steps", type=int)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--expected-node-count", type=int)
    parser.add_argument("--expected-hcu-per-node", type=int, default=HCU_PER_NODE)
    parser.add_argument(
        "--expected-container-id",
        action="append",
        default=[],
        metavar="NODE=ID",
        help="Bind any known live container identity; repeat for partial/full rosters.",
    )
    parser.add_argument(
        "--expected-hostname",
        action="append",
        default=[],
        metavar="NODE=HOSTNAME",
    )
    parser.add_argument("--max-age-seconds", type=int, default=300)
    return parser.parse_args(argv)


def _argument_or_environment(
    argument: object,
    environment_name: str,
    option_name: str,
) -> str:
    value = argument if argument is not None else os.environ.get(environment_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{option_name} or {environment_name} is required")
    return value


def _integer_argument_or_environment(
    argument: int | None,
    environment_name: str,
    option_name: str,
) -> int:
    if argument is not None:
        return argument
    raw_value = os.environ.get(environment_name)
    if raw_value is None:
        raise ValueError(f"{option_name} or {environment_name} is required")
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{environment_name} is not an integer") from error


def _expected_nccl_environment(values: Sequence[str]) -> dict[str, str]:
    expected = dict(parse_assignments(values, "--expected-nccl-env"))
    if expected:
        return expected
    return {
        name: value
        for name, value in os.environ.items()
        if name.startswith(("NCCL_", "TORCH_NCCL_"))
    }


def _run_cli(args: argparse.Namespace) -> None:
    expected_containers = parse_assignments(
        args.expected_container_id,
        "--expected-container-id",
    )
    expected_hostnames = parse_assignments(
        args.expected_hostname,
        "--expected-hostname",
    )
    validate_collective_smoke_report(
        args.report_path,
        expected_sha256=_argument_or_environment(
            args.expected_report_sha256,
            "N0_TRACK31_COLLECTIVE_SMOKE_REPORT_SHA256",
            "--expected-report-sha256",
        ),
        smoke_id=_argument_or_environment(
            args.smoke_id,
            "N0_TRACK31_INVOCATION_ID",
            "--expected-smoke-id",
        ),
        expected_nodes=parse_csv(
            _argument_or_environment(
                args.expected_nodes,
                "N0_TRACK31_NODES",
                "--expected-nodes",
            ),
            "--expected-nodes",
        ),
        expected_hcu_order=parse_csv(
            _argument_or_environment(
                args.expected_hcu_order,
                "HIP_VISIBLE_DEVICES",
                "--expected-hcu-order",
            ),
            "--expected-hcu-order",
        ),
        expected_nccl_environment=_expected_nccl_environment(args.expected_nccl_env),
        image_id=_argument_or_environment(
            args.expected_image_id,
            "N0_TRACK31_IMAGE_ID",
            "--expected-image-id",
        ),
        source_manifest_sha256=_argument_or_environment(
            args.expected_source_manifest_sha256,
            "N0_TRACK31_SOURCE_MANIFEST_SHA256",
            "--expected-source-manifest-sha256",
        ),
        overlay_manifest_sha256=_argument_or_environment(
            args.expected_overlay_manifest_sha256,
            "N0_TRACK31_OVERLAY_MANIFEST_SHA256",
            "--expected-overlay-manifest-sha256",
        ),
        run_role=_argument_or_environment(
            args.expected_run_role,
            "N0_TRACK31_RUN_ROLE",
            "--expected-run-role",
        ),
        batch_size=_integer_argument_or_environment(
            args.expected_batch_size,
            "N0_TRACK31_BATCH_SIZE",
            "--expected-batch-size",
        ),
        gradient_accumulation_steps=_integer_argument_or_environment(
            args.expected_gradient_accumulation_steps,
            "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS",
            "--expected-gradient-accumulation-steps",
        ),
        expected_container_ids=expected_containers or None,
        expected_hostnames=expected_hostnames or None,
        expected_world_size=args.expected_world_size,
        expected_node_count=args.expected_node_count,
        expected_hcu_per_node=args.expected_hcu_per_node,
        max_age_seconds=args.max_age_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _run_cli(args)
    except (CollectiveSmokeValidationError, ValueError, OSError) as error:
        print(f"collective smoke validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
