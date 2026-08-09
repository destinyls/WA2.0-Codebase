# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Public command-line interface for N0-TWAM reproducible workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from n0_twam import __version__


def _json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def _write_template(path: Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(_json(payload) + "\n")
    return destination


def _template_command(args: argparse.Namespace) -> dict[str, object]:
    if args.kind == "train":
        from n0_twam.track31.request import track31_train_request_template

        payload = track31_train_request_template()
    else:
        from n0_twam.evaluation.target10_evaluation_template import (
            target10_reference_request_template,
        )

        payload = target10_reference_request_template()
    result: dict[str, object] = {"kind": args.kind, "template": payload}
    if args.output is not None:
        result["output"] = str(_write_template(args.output, payload))
    return result


def _train_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.track31.request import load_track31_train_request
    from n0_twam.track31.runner import run_track31_training

    request = load_track31_train_request(args.config)
    return run_track31_training(request, dry_run=args.dry_run)


def _eval_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.evaluation.target10_reference_pipeline import (
        load_target10_reference_request,
        run_target10_reference_evaluation,
    )

    request = load_target10_reference_request(args.request)
    return run_target10_reference_evaluation(request)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="n0-twam", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    track31 = commands.add_parser(
        "track31", help="train or evaluate the UniVTAC Track 3.1 model"
    )
    track31_commands = track31.add_subparsers(dest="track31_command", required=True)

    template = track31_commands.add_parser(
        "template", help="print a portable strict JSON request template"
    )
    template.add_argument("kind", choices=("train", "eval"))
    template.add_argument("--output", type=Path)
    template.set_defaults(handler=_template_command)

    train = track31_commands.add_parser(
        "train", help="run preflight, training, and checkpoint verification"
    )
    train.add_argument("--config", type=Path, required=True)
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the request and print the resolved launch plan",
    )
    train.set_defaults(handler=_train_command)

    evaluate = track31_commands.add_parser(
        "eval", help="run the frozen Target-10 reference evaluation"
    )
    evaluate.add_argument("--request", type=Path, required=True)
    evaluate.set_defaults(handler=_eval_command)
    return parser


def run_cli(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Parse arguments and return a machine-readable result."""

    args = _build_parser().parse_args(argv)
    handler = args.handler
    result = handler(args)
    if not isinstance(result, dict):
        raise TypeError("CLI handler did not return a JSON object")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry with concise errors and JSON stdout."""

    try:
        result = run_cli(argv)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"n0-twam: {error}", file=sys.stderr)
        return 1
    print(_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
