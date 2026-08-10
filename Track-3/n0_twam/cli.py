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


def _track32_template_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.track32.request import track32_train_request_template

    payload = track32_train_request_template()
    result: dict[str, object] = {"kind": "train", "template": payload}
    if args.output is not None:
        result["output"] = str(_write_template(args.output, payload))
    return result


def _track32_train_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.track32.request import load_track32_train_request
    from n0_twam.track32.runner import run_track32_training

    request = load_track32_train_request(args.config)
    return run_track32_training(request, dry_run=args.dry_run)


def _track32_policy_template_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.integrations.worldarena.franka_policy import (
        franka_policy_config_template,
    )

    payload = franka_policy_config_template()
    result: dict[str, object] = {"kind": "policy", "template": payload}
    if args.output is not None:
        result["output"] = str(_write_template(args.output, payload))
    return result


def _track32_serve_bundle_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.integrations.worldarena.franka_serve_bundle import (
        build_franka_serve_bundle,
    )

    return build_franka_serve_bundle(
        checkpoint=args.checkpoint,
        checkpoint_identity_sha256=args.checkpoint_identity_sha256,
        base_model=args.base_model,
        normalizer=args.normalizer,
        normalizer_sha256=args.normalizer_sha256,
        output=args.output,
    )


def _track32_policy_replay_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.integrations.worldarena.franka_policy_replay import (
        run_franka_policy_replay,
    )

    return run_franka_policy_replay(
        config_path=args.config,
        observation_path=args.observation,
        prompt=args.prompt,
        steps=args.steps,
        output=args.output,
    )


def _track32_score_predictions_command(
    args: argparse.Namespace,
) -> dict[str, object]:
    from n0_twam.evaluation.franka_offline_metrics import (
        evaluate_franka_offline_predictions,
    )

    return evaluate_franka_offline_predictions(
        predictions=args.predictions,
        output=args.output,
        checkpoint_identity_sha256=args.checkpoint_identity_sha256,
        dataset_view_id=args.dataset_view_id,
        dataset_view_sha256=args.dataset_view_sha256,
        decoder_sha256=args.decoder_sha256,
    )


def _track32_generate_predictions_command(
    args: argparse.Namespace,
) -> dict[str, object]:
    from n0_twam.evaluation.franka_offline_generation import (
        generate_franka_offline_predictions,
    )

    return generate_franka_offline_predictions(
        checkpoint=args.checkpoint,
        checkpoint_identity_sha256=args.checkpoint_identity_sha256,
        serve_bundle=args.serve_bundle,
        serve_bundle_receipt_sha256=args.serve_bundle_receipt_sha256,
        serve_output=args.serve_output,
        artifact_root=args.artifact_root,
        lerobot_root=args.lerobot_root,
        base_model=args.base_model,
        normalizer=args.normalizer,
        dataset_view=args.dataset_view,
        output=args.output,
        seed=args.seed,
        device=args.device,
        distributed_port=args.distributed_port,
        video_inference_steps=args.video_inference_steps,
        action_inference_steps=args.action_inference_steps,
        decode_batch_size=args.decode_batch_size,
        max_samples=args.max_samples,
    )


def _track32_pack_predictions_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.evaluation.franka_prediction_pack import pack_franka_predictions

    return pack_franka_predictions(
        metadata=args.metadata,
        arrays=args.arrays,
        output=args.output,
    )


def _track32_bridge_patch_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.integrations.worldarena.franka_bridge_patch import (
        apply_pinned_worldarena_bridge_patch,
    )

    return apply_pinned_worldarena_bridge_patch(args.worldarena_root)


def _track32_bridge_audit_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.integrations.worldarena.franka_official_worker import (
        PINNED_PATCHED_BRIDGE_SHA256,
        PINNED_WORLD_ARENA_REVISION,
        audit_worldarena_franka_bridge,
    )

    return audit_worldarena_franka_bridge(
        args.worldarena_root,
        expected_revision=args.expected_revision or PINNED_WORLD_ARENA_REVISION,
        expected_bridge_sha256=(
            args.expected_bridge_sha256 or PINNED_PATCHED_BRIDGE_SHA256
        ),
    )


def _track32_worker_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.integrations.worldarena.franka_official_worker import (
        PINNED_PATCHED_BRIDGE_SHA256,
        PINNED_WORLD_ARENA_REVISION,
        run_official_franka_worker,
    )

    return run_official_franka_worker(
        worldarena_root=args.worldarena_root,
        expected_revision=args.expected_revision or PINNED_WORLD_ARENA_REVISION,
        expected_bridge_sha256=(
            args.expected_bridge_sha256 or PINNED_PATCHED_BRIDGE_SHA256
        ),
        config_path=args.config,
        hub_url=args.hub_url,
        worker_key=args.worker_key,
        allow_local_http=args.allow_local_http,
        dry_run=args.dry_run,
    )


def _track32_build_request_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.track32.request_builder import build_track32_train_request

    devices = tuple(int(value) for value in args.devices.split(",") if value)
    return build_track32_train_request(
        destination=args.output,
        run_id=args.run_id,
        devices=devices,
        master_port=args.master_port,
        accelerator_profile=args.accelerator_profile,
        collective_network_interface=args.collective_network_interface,
        artifact_root=args.artifact_root,
        lerobot_root=args.lerobot_root,
        base_model=args.base_model,
        empty_embedding=args.empty_embedding,
        init_from=args.init_from,
        resume_from=args.resume_from,
        output_root=args.output_root,
        run_role=args.run_role,
        num_steps=args.num_steps,
        stop_after_step=args.stop_after_step,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_latent_frames=args.max_latent_frames,
        seed=args.seed,
    )


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

    track32 = commands.add_parser(
        "track32", help="post-train the vision-only Franka Track 3.2 model"
    )
    track32_commands = track32.add_subparsers(dest="track32_command", required=True)
    track32_template = track32_commands.add_parser(
        "template", help="write a strict portable Franka training request"
    )
    track32_template.add_argument("--output", type=Path)
    track32_template.set_defaults(handler=_track32_template_command)
    build_request = track32_commands.add_parser(
        "build-request", help="audit inputs and write a hash-complete request"
    )
    build_request.add_argument("--output", type=Path, required=True)
    build_request.add_argument("--run-id", required=True)
    build_request.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    build_request.add_argument("--master-port", type=int, default=29632)
    build_request.add_argument(
        "--accelerator-profile",
        choices=("portable", "hcu_performance"),
        default="portable",
    )
    build_request.add_argument(
        "--collective-network-interface",
        help=(
            "single NCCL/Gloo interface required by hcu_performance; "
            "omit for portable"
        ),
    )
    build_request.add_argument("--artifact-root", type=Path, required=True)
    build_request.add_argument("--lerobot-root", type=Path, required=True)
    build_request.add_argument("--base-model", type=Path, required=True)
    build_request.add_argument("--empty-embedding", type=Path, required=True)
    route = build_request.add_mutually_exclusive_group(required=True)
    route.add_argument("--init-from", type=Path)
    route.add_argument("--resume-from", type=Path)
    build_request.add_argument("--output-root", type=Path, required=True)
    build_request.add_argument(
        "--run-role", choices=("development", "final_refit"), default="development"
    )
    build_request.add_argument("--num-steps", type=int, default=1500)
    build_request.add_argument("--stop-after-step", type=int, default=1500)
    build_request.add_argument("--save-interval", type=int, default=300)
    build_request.add_argument("--val-interval", type=int, default=100)
    build_request.add_argument("--batch-size", type=int, default=1)
    build_request.add_argument("--gradient-accumulation-steps", type=int, default=1)
    build_request.add_argument("--max-latent-frames", type=int, default=5)
    build_request.add_argument("--seed", type=int, default=20260810)
    build_request.set_defaults(handler=_track32_build_request_command)
    track32_train = track32_commands.add_parser(
        "train", help="run preflight, training, and checkpoint verification"
    )
    track32_train.add_argument("--config", type=Path, required=True)
    track32_train.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the request and print the resolved launch plan",
    )
    track32_train.set_defaults(handler=_track32_train_command)
    policy_template = track32_commands.add_parser(
        "policy-template", help="write the official Franka Policy config template"
    )
    policy_template.add_argument("--output", type=Path)
    policy_template.set_defaults(handler=_track32_policy_template_command)
    serve_bundle = track32_commands.add_parser(
        "serve-bundle", help="seal a trained checkpoint for the Franka Policy"
    )
    serve_bundle.add_argument("--checkpoint", type=Path, required=True)
    serve_bundle.add_argument("--checkpoint-identity-sha256", required=True)
    serve_bundle.add_argument("--base-model", type=Path, required=True)
    serve_bundle.add_argument("--normalizer", type=Path, required=True)
    serve_bundle.add_argument("--normalizer-sha256", required=True)
    serve_bundle.add_argument("--output", type=Path, required=True)
    serve_bundle.set_defaults(handler=_track32_serve_bundle_command)
    policy_replay = track32_commands.add_parser(
        "policy-replay",
        help="run an offline Franka Policy protocol/safety replay",
    )
    policy_replay.add_argument("--config", type=Path, required=True)
    policy_replay.add_argument("--observation", type=Path, required=True)
    policy_replay.add_argument("--prompt", required=True)
    policy_replay.add_argument("--steps", type=int, default=7)
    policy_replay.add_argument("--output", type=Path, required=True)
    policy_replay.set_defaults(handler=_track32_policy_replay_command)

    score_predictions = track32_commands.add_parser(
        "score-predictions",
        help="score future RGB and Franka end-pose prediction artifacts",
    )
    score_predictions.add_argument("--predictions", type=Path, required=True)
    score_predictions.add_argument("--output", type=Path, required=True)
    score_predictions.add_argument("--checkpoint-identity-sha256", required=True)
    score_predictions.add_argument("--dataset-view-id", required=True)
    score_predictions.add_argument("--dataset-view-sha256", required=True)
    score_predictions.add_argument("--decoder-sha256", required=True)
    score_predictions.set_defaults(handler=_track32_score_predictions_command)

    generate_predictions = track32_commands.add_parser(
        "generate-predictions",
        help="run the Direct Franka backend on the frozen validation view",
    )
    generate_predictions.add_argument("--checkpoint", type=Path, required=True)
    generate_predictions.add_argument("--checkpoint-identity-sha256", required=True)
    generate_predictions.add_argument("--serve-bundle", type=Path, required=True)
    generate_predictions.add_argument("--serve-bundle-receipt-sha256", required=True)
    generate_predictions.add_argument("--serve-output", type=Path, required=True)
    generate_predictions.add_argument("--artifact-root", type=Path, required=True)
    generate_predictions.add_argument("--lerobot-root", type=Path, required=True)
    generate_predictions.add_argument("--base-model", type=Path, required=True)
    generate_predictions.add_argument("--normalizer", type=Path, required=True)
    generate_predictions.add_argument("--dataset-view", type=Path, required=True)
    generate_predictions.add_argument("--output", type=Path, required=True)
    generate_predictions.add_argument("--seed", type=int, default=20260810)
    generate_predictions.add_argument("--device", default="0")
    generate_predictions.add_argument("--distributed-port", type=int, default=29651)
    generate_predictions.add_argument("--video-inference-steps", type=int, default=3)
    generate_predictions.add_argument("--action-inference-steps", type=int, default=4)
    generate_predictions.add_argument("--decode-batch-size", type=int, default=4)
    generate_predictions.add_argument("--max-samples", type=int)
    generate_predictions.set_defaults(handler=_track32_generate_predictions_command)

    pack_predictions = track32_commands.add_parser(
        "pack-predictions",
        help="materialize the strict Franka future-prediction artifact",
    )
    pack_predictions.add_argument("--metadata", type=Path, required=True)
    pack_predictions.add_argument("--arrays", type=Path, required=True)
    pack_predictions.add_argument("--output", type=Path, required=True)
    pack_predictions.set_defaults(handler=_track32_pack_predictions_command)

    bridge_patch = track32_commands.add_parser(
        "bridge-patch",
        help="apply the pinned Franka WXYZ fix to a clean WorldArena checkout",
    )
    bridge_patch.add_argument("--worldarena-root", type=Path, required=True)
    bridge_patch.set_defaults(handler=_track32_bridge_patch_command)

    bridge_audit = track32_commands.add_parser(
        "bridge-audit",
        help="prove the WorldArena Franka WXYZ/XYZW bridge in both directions",
    )
    bridge_audit.add_argument("--worldarena-root", type=Path, required=True)
    bridge_audit.add_argument("--expected-revision")
    bridge_audit.add_argument("--expected-bridge-sha256")
    bridge_audit.set_defaults(handler=_track32_bridge_audit_command)

    worker = track32_commands.add_parser(
        "worker", help="run the official outbound Franka Hub worker"
    )
    worker.add_argument("--worldarena-root", type=Path, required=True)
    worker.add_argument("--expected-revision")
    worker.add_argument("--expected-bridge-sha256")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--hub-url", required=True)
    worker.add_argument("--worker-key", required=True)
    worker.add_argument(
        "--allow-local-http",
        action="store_true",
        help="allow HTTP only for localhost dummy-Hub testing",
    )
    worker.add_argument(
        "--dry-run", action="store_true", help="audit without loading the model"
    )
    worker.set_defaults(handler=_track32_worker_command)
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
