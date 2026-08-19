# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Public AgileX offline-evaluation CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_view(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.evaluation.agilex_proxy_view import (
        build_agilex_proxy_evaluation_view,
    )

    return build_agilex_proxy_evaluation_view(
        config_path=args.config,
        dataset_root=args.dataset_root,
        output=args.output,
        view_id=args.view_id,
        samples_per_task=args.samples_per_task,
    )


def _generate(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.evaluation.agilex_offline_generation import (
        generate_agilex_offline_predictions,
    )

    return generate_agilex_offline_predictions(
        train_request=args.train_request,
        policy_config=args.config,
        dataset_view=args.dataset_view,
        output=args.output,
        seed=args.seed,
        device=args.device,
        decode_batch_size=args.decode_batch_size,
        max_samples=args.max_samples,
        replay_observation_output=args.replay_observation_output,
    )


def _score(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.evaluation.agilex_offline_metrics import (
        evaluate_agilex_offline_predictions,
    )

    return evaluate_agilex_offline_predictions(
        predictions=args.predictions,
        output=args.output,
        checkpoint_identity_sha256=args.checkpoint_identity_sha256,
        dataset_view_id=args.dataset_view_id,
        dataset_view_sha256=args.dataset_view_sha256,
        decoder_sha256=args.decoder_sha256,
    )


def _replay(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.integrations.worldarena.agilex_policy_replay import (
        run_agilex_policy_replay,
    )

    return run_agilex_policy_replay(
        config_path=args.config,
        observation_path=args.observation,
        steps=args.steps,
        output=args.output,
        control_hz=args.control_hz,
        require_realtime=args.require_realtime,
    )


def _pipeline(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.evaluation.agilex_offline_pipeline import (
        run_agilex_offline_evaluation,
    )

    return run_agilex_offline_evaluation(
        train_request=args.train_request,
        policy_config=args.config,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        seed=args.seed,
        device=args.device,
        decode_batch_size=args.decode_batch_size,
        samples_per_task=args.samples_per_task,
        replay_steps=args.replay_steps,
        control_hz=args.control_hz,
    )


def _closeout(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.evaluation.agilex_offline_pipeline import (
        recover_agilex_offline_closeout,
    )

    return recover_agilex_offline_closeout(
        train_request=args.train_request,
        policy_config=args.config,
        evaluation_view=args.evaluation_view,
        predictions=args.predictions,
        metrics=args.metrics,
        replay_observation=args.replay_observation,
        output_root=args.output_root,
        replay_steps=args.replay_steps,
        control_hz=args.control_hz,
        seed=args.seed,
        device=args.device,
    )


def add_agilex_offline_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the four-stage AgileX offline evaluation surface."""

    view = commands.add_parser(
        "agilex-build-eval-view",
        help="freeze a deterministic all-task training-distribution proxy roster",
    )
    view.add_argument("--config", type=Path, required=True)
    view.add_argument("--dataset-root", type=Path, required=True)
    view.add_argument("--output", type=Path, required=True)
    view.add_argument("--view-id", default="agilex-all-task-proxy-v1")
    view.add_argument("--samples-per-task", type=int, default=1)
    view.set_defaults(handler=_build_view)

    generate = commands.add_parser(
        "agilex-generate-predictions",
        help="generate future RGB and qpos14 predictions with the Direct backend",
    )
    generate.add_argument("--train-request", type=Path, required=True)
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--dataset-view", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=20260813)
    generate.add_argument("--device", default="0")
    generate.add_argument("--decode-batch-size", type=int, default=4)
    generate.add_argument("--max-samples", type=int)
    generate.add_argument("--replay-observation-output", type=Path)
    generate.set_defaults(handler=_generate)

    score = commands.add_parser(
        "agilex-score-predictions",
        help="score AgileX future RGB PSNR/SSIM and qpos14 MAE/RMSE",
    )
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--checkpoint-identity-sha256", required=True)
    score.add_argument("--dataset-view-id", required=True)
    score.add_argument("--dataset-view-sha256", required=True)
    score.add_argument("--decoder-sha256", required=True)
    score.set_defaults(handler=_score)

    replay = commands.add_parser(
        "agilex-policy-replay",
        help="run offline AgileX Policy protocol, safety, and latency replay",
    )
    replay.add_argument("--config", type=Path, required=True)
    replay.add_argument("--observation", type=Path, required=True)
    replay.add_argument("--steps", type=int, default=7)
    replay.add_argument("--control-hz", type=float, default=10.0)
    replay.add_argument("--require-realtime", action="store_true")
    replay.add_argument("--output", type=Path, required=True)
    replay.set_defaults(handler=_replay)

    pipeline = commands.add_parser(
        "agilex-offline-eval",
        help="run the complete AgileX proxy metrics and Policy replay closeout",
    )
    pipeline.add_argument("--train-request", type=Path, required=True)
    pipeline.add_argument("--config", type=Path, required=True)
    pipeline.add_argument("--dataset-root", type=Path, required=True)
    pipeline.add_argument("--output-root", type=Path, required=True)
    pipeline.add_argument("--seed", type=int, default=20260813)
    pipeline.add_argument("--device", default="0")
    pipeline.add_argument("--decode-batch-size", type=int, default=4)
    pipeline.add_argument("--samples-per-task", type=int, default=1)
    pipeline.add_argument("--replay-steps", type=int, default=7)
    pipeline.add_argument("--control-hz", type=float, default=10.0)
    pipeline.set_defaults(handler=_pipeline)

    closeout = commands.add_parser(
        "agilex-offline-closeout",
        help="replay and close out verified existing AgileX offline artifacts",
    )
    closeout.add_argument("--train-request", type=Path, required=True)
    closeout.add_argument("--config", type=Path, required=True)
    closeout.add_argument("--evaluation-view", type=Path, required=True)
    closeout.add_argument("--predictions", type=Path, required=True)
    closeout.add_argument("--metrics", type=Path, required=True)
    closeout.add_argument("--replay-observation", type=Path, required=True)
    closeout.add_argument("--output-root", type=Path, required=True)
    closeout.add_argument("--replay-steps", type=int, default=7)
    closeout.add_argument("--control-hz", type=float, default=10.0)
    closeout.add_argument("--seed", type=int, default=20260813)
    closeout.add_argument("--device", default="0")
    closeout.set_defaults(handler=_closeout)


__all__ = ("add_agilex_offline_commands",)
