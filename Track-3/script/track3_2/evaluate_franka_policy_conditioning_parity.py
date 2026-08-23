#!/usr/bin/env python3
"""Evaluate RGB/cached-latent parity through the deployable Franka Policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from n0_twam.evaluation.franka_policy_conditioning_parity import (
    evaluate_franka_policy_conditioning_parity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--dataset-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--action-selection",
        choices=("first_future_action", "last_future_action"),
        default="first_future_action",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_franka_policy_conditioning_parity(
        policy_config=args.policy_config,
        artifact_root=args.artifact_root,
        lerobot_root=args.lerobot_root,
        base_model=args.base_model,
        normalizer=args.normalizer,
        dataset_view=args.dataset_view,
        output=args.output,
        samples_output=args.samples_output,
        max_samples=args.max_samples,
        action_selection=args.action_selection,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
