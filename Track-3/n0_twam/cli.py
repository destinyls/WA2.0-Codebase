# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Public command-line interface for N0-TWAM reproducible workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
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


def _verified_file(path: Path, digest: str, *, label: str) -> Path:
    from n0_twam.integrations.worldarena.agilex_manifest import sha256_file

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} must be a non-symlink regular file")
    source = candidate.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file")
    if sha256_file(source) != digest:
        raise ValueError(f"{label} SHA256 mismatch")
    return source


def _publish_artifact_pair(
    artifacts: tuple[tuple[Path, bytes], tuple[Path, bytes]],
) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, content in artifacts:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.tmp"
            )
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            staged.append((temporary, destination))
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, destination in staged:
            os.link(temporary, destination, follow_symlinks=False)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError(f"{label} must be a non-symlink regular file")
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


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


def _track32_agilex_template_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.track32_agilex.request import agilex_train_request_template

    payload = agilex_train_request_template()
    result: dict[str, object] = {"kind": "agilex-train", "template": payload}
    if args.output is not None:
        result["output"] = str(_write_template(args.output, payload))
    return result


def _track32_agilex_train_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.track32_agilex.request import load_agilex_train_request
    from n0_twam.track32_agilex.runner import run_agilex_training

    request = load_agilex_train_request(args.config)
    return run_agilex_training(request, dry_run=args.dry_run)


def _track32_agilex_build_artifacts_command(
    args: argparse.Namespace,
) -> dict[str, object]:
    from n0_twam.configs import twam_track3_agilex_contracts as contracts
    from n0_twam.integrations.worldarena import agilex_artifacts as artifacts
    from n0_twam.integrations.worldarena import agilex_manifest as manifest_io

    source = _verified_file(
        args.source_manifest, args.source_manifest_sha256, label="source manifest"
    )
    route_file = _verified_file(
        args.repo_route_manifest,
        args.repo_route_manifest_file_sha256,
        label="repo-route manifest",
    )
    temporal_file = _verified_file(
        args.temporal_alignment,
        args.temporal_alignment_file_sha256,
        label="temporal alignment",
    )
    raw_dataset = Path(args.dataset_root).expanduser()
    dataset_root = raw_dataset.resolve(strict=True)
    binding = contracts.load_repo_route_binding(route_file)
    repo_ids = tuple(sorted(binding.routes))
    temporal = contracts.load_temporal_binding(
        temporal_file,
        repo_names=repo_ids,
        repo_route_manifest_sha256=binding.manifest.contract_sha256,
    )
    manifest = manifest_io.load_agilex_manifest(
        source,
        expected_file_sha256=args.source_manifest_sha256,
        selected_repo_ids=repo_ids,
    )
    route_ids = dict(temporal.route_identities)
    temporal_ids = dict(temporal.temporal_identities)
    for route in manifest.routes:
        configured = binding.routes[route.repo_id]
        expected = {
            "embodiment": route.embodiment,
            "action_schema": route.action_schema,
            "rgb_keys": list(route.rgb_keys),
            "tactile_keys": list(route.tactile_keys),
            "wrench_keys": list(route.wrench_keys),
        }
        if configured != expected or route_ids[route.repo_id] != route.route_identity:
            raise ValueError("AgileX source, route, and temporal identities differ")
        if temporal_ids[route.repo_id] != route.temporal_alignment_identity:
            raise ValueError("AgileX source temporal alignment identity mismatch")
    outputs = tuple(
        Path(value).expanduser().resolve(strict=False)
        for value in (args.conversion_output, args.latent_output)
    )
    inputs = (dataset_root, source, route_file, temporal_file)
    if any(
        output == item or output in item.parents or item in output.parents
        for index, output in enumerate(outputs)
        for item in (*outputs[index + 1 :], *inputs)
    ):
        raise ValueError("AgileX artifact outputs must be disjoint from all inputs")
    if any(
        Path(value).expanduser().is_symlink() or output.exists()
        for value, output in zip(
            (args.conversion_output, args.latent_output), outputs, strict=True
        )
    ):
        raise FileExistsError("AgileX artifact outputs must be new non-symlink files")
    conversion = artifacts.build_agilex_conversion_receipt(
        dataset_root=dataset_root,
        routes=manifest.routes,
        source_manifest_sha256=args.source_manifest_sha256,
        repo_route_manifest_sha256=binding.manifest.contract_sha256,
        temporal_alignment_contract_sha256=temporal.contract_sha256,
    )
    conversion_bytes = (_json(conversion) + "\n").encode("utf-8")
    conversion_sha = hashlib.sha256(conversion_bytes).hexdigest()
    latent = artifacts.build_agilex_latent_inventory(
        dataset_root=dataset_root,
        routes=manifest.routes,
        conversion_receipt=conversion,
        conversion_receipt_sha256=conversion_sha,
    )
    latent_bytes = (_json(latent) + "\n").encode("utf-8")
    _publish_artifact_pair(((outputs[0], conversion_bytes), (outputs[1], latent_bytes)))
    return {
        "kind": "agilex-artifacts",
        "status": "complete",
        "conversion_output": str(outputs[0]),
        "conversion_file_sha256": conversion_sha,
        "conversion_identity_sha256": conversion["conversion_identity_sha256"],
        "latent_output": str(outputs[1]),
        "latent_file_sha256": hashlib.sha256(latent_bytes).hexdigest(),
        "latent_inventory_sha256": latent["inventory_sha256"],
        "record_count": latent["record_count"],
    }


def _track32_agilex_policy_template_command(
    args: argparse.Namespace,
) -> dict[str, object]:
    from n0_twam.integrations.worldarena import agilex_policy_io

    payload = agilex_policy_io.agilex_policy_config_template()
    result: dict[str, object] = {"kind": "agilex-policy", "template": payload}
    if args.output is not None:
        result["output"] = str(_write_template(args.output, payload))
    return result


def _track32_agilex_policy_check_command(
    args: argparse.Namespace,
) -> dict[str, object]:
    from n0_twam.integrations.worldarena import agilex_policy_io

    config = agilex_policy_io.load_agilex_policy_config(args.config)
    return {
        "kind": "agilex-policy-check",
        "status": "verified",
        "config": str(config.source_path),
        "policy_id": config.policy.policy_id,
        "tactile_profile": config.policy.tactile_profile,
        "task_ids": sorted(config.policy.task_routes),
        "policy_config_file_sha256": config.policy_config_file_sha256,
        "config_contract_sha256": config.config_contract_sha256,
        "serve_bundle": str(config.serve_bundle),
        "serve_bundle_identity_sha256": config.serve_bundle_identity_sha256,
        "checkpoint_identity_sha256": config.checkpoint_identity_sha256,
        "normalizer_contract_sha256": config.normalizer_contract_sha256,
        "repo_route_manifest_sha256": config.repo_route_manifest_sha256,
    }


def _track32_agilex_serve_bundle_command(
    args: argparse.Namespace,
) -> dict[str, object]:
    from n0_twam.integrations.worldarena.agilex_serve_bundle import (
        build_agilex_serve_bundle,
    )

    return build_agilex_serve_bundle(
        checkpoint=args.checkpoint,
        checkpoint_identity_sha256=args.checkpoint_identity_sha256,
        base_model=args.base_model,
        normalizer=args.normalizer,
        normalizer_file_sha256=args.normalizer_file_sha256,
        normalizer_contract_sha256=args.normalizer_contract_sha256,
        source_manifest_sha256=args.source_manifest_sha256,
        repo_route_manifest=args.repo_route_manifest,
        repo_route_manifest_file_sha256=args.repo_route_manifest_file_sha256,
        repo_route_manifest_sha256=args.repo_route_manifest_sha256,
        tactile_profile=args.tactile_profile,
        contact_profile_contract_sha256=args.contact_profile_contract_sha256,
        task_routes=_read_json_object(args.task_routes, label="AgileX task routes"),
        task_routes_sha256=args.task_routes_sha256,
        safety_contract=_read_json_object(
            args.safety_contract, label="AgileX safety contract"
        ),
        safety_contract_sha256=args.safety_contract_sha256,
        output=args.output,
    )


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
        control_hz=args.control_hz,
        require_realtime=args.require_realtime,
        minimum_refill_samples=args.minimum_refill_samples,
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


def _track32_bridge_audit_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.integrations.worldarena.franka_official_worker import (
        PINNED_ORIGINAL_BRIDGE_SHA256,
        PINNED_WORLD_ARENA_REVISION,
        audit_worldarena_franka_bridge,
    )

    return audit_worldarena_franka_bridge(
        args.worldarena_root,
        expected_revision=args.expected_revision or PINNED_WORLD_ARENA_REVISION,
        expected_bridge_sha256=(
            args.expected_bridge_sha256 or PINNED_ORIGINAL_BRIDGE_SHA256
        ),
    )


def _track32_worker_command(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.integrations.worldarena.franka_official_worker import (
        PINNED_ORIGINAL_BRIDGE_SHA256,
        PINNED_WORLD_ARENA_REVISION,
        run_official_franka_worker,
    )

    return run_official_franka_worker(
        worldarena_root=args.worldarena_root,
        expected_revision=args.expected_revision or PINNED_WORLD_ARENA_REVISION,
        expected_bridge_sha256=(
            args.expected_bridge_sha256 or PINNED_ORIGINAL_BRIDGE_SHA256
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
    agilex_template = track32_commands.add_parser(
        "agilex-template",
        help="write a strict hash-complete AgileX training request",
    )
    agilex_template.add_argument("--output", type=Path)
    agilex_template.set_defaults(handler=_track32_agilex_template_command)
    agilex_train = track32_commands.add_parser(
        "agilex-train",
        help="run AgileX preflight, qpos14 training, and checkpoint verification",
    )
    agilex_train.add_argument("--config", type=Path, required=True)
    agilex_train.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the request schema and print the resolved AgileX launch plan",
    )
    agilex_train.set_defaults(handler=_track32_agilex_train_command)
    agilex_artifacts = track32_commands.add_parser(
        "agilex-build-artifacts",
        help="build immutable AgileX conversion and latent receipts",
    )
    agilex_artifacts.add_argument("--dataset-root", type=Path, required=True)
    agilex_artifacts.add_argument("--source-manifest", type=Path, required=True)
    agilex_artifacts.add_argument("--source-manifest-sha256", required=True)
    for name in ("repo-route-manifest", "temporal-alignment"):
        agilex_artifacts.add_argument(f"--{name}", type=Path, required=True)
        agilex_artifacts.add_argument(f"--{name}-file-sha256", required=True)
    agilex_artifacts.add_argument("--conversion-output", type=Path, required=True)
    agilex_artifacts.add_argument("--latent-output", type=Path, required=True)
    agilex_artifacts.set_defaults(handler=_track32_agilex_build_artifacts_command)
    agilex_policy_template = track32_commands.add_parser(
        "agilex-policy-template",
        help="write a strict local AgileX Policy config template",
    )
    agilex_policy_template.add_argument("--output", type=Path)
    agilex_policy_template.set_defaults(handler=_track32_agilex_policy_template_command)
    agilex_policy_check = track32_commands.add_parser(
        "agilex-policy-check",
        help="verify an AgileX Policy config and sealed artifacts without loading",
    )
    agilex_policy_check.add_argument("--config", type=Path, required=True)
    agilex_policy_check.set_defaults(handler=_track32_agilex_policy_check_command)
    agilex_bundle = track32_commands.add_parser(
        "agilex-serve-bundle",
        help="seal a strict qpos14 checkpoint for the AgileX Policy",
    )
    agilex_bundle.add_argument("--checkpoint", type=Path, required=True)
    agilex_bundle.add_argument("--checkpoint-identity-sha256", required=True)
    agilex_bundle.add_argument("--base-model", type=Path, required=True)
    agilex_bundle.add_argument("--normalizer", type=Path, required=True)
    agilex_bundle.add_argument("--normalizer-file-sha256", required=True)
    agilex_bundle.add_argument("--normalizer-contract-sha256", required=True)
    agilex_bundle.add_argument("--source-manifest-sha256", required=True)
    agilex_bundle.add_argument("--repo-route-manifest", type=Path, required=True)
    agilex_bundle.add_argument("--repo-route-manifest-file-sha256", required=True)
    agilex_bundle.add_argument("--repo-route-manifest-sha256", required=True)
    agilex_bundle.add_argument(
        "--tactile-profile",
        choices=("vision_tactile", "mixed", "vision_only"),
        required=True,
    )
    agilex_bundle.add_argument("--contact-profile-contract-sha256", required=True)
    agilex_bundle.add_argument("--task-routes", type=Path, required=True)
    agilex_bundle.add_argument("--task-routes-sha256", required=True)
    agilex_bundle.add_argument("--safety-contract", type=Path, required=True)
    agilex_bundle.add_argument("--safety-contract-sha256", required=True)
    agilex_bundle.add_argument("--output", type=Path, required=True)
    agilex_bundle.set_defaults(handler=_track32_agilex_serve_bundle_command)
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
    policy_replay.add_argument(
        "--control-hz",
        type=float,
        default=15.0,
        help="control frequency used by the realtime latency gate",
    )
    policy_replay.add_argument(
        "--minimum-refill-samples",
        type=int,
        default=2,
        help="minimum synchronous refill samples required for a realtime pass",
    )
    policy_replay.add_argument(
        "--require-realtime",
        action="store_true",
        help="fail without publishing a receipt unless queue/refill p99 meets deadline",
    )
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

    bridge_audit = track32_commands.add_parser(
        "bridge-audit",
        help="prove the unmodified WorldArena Franka XYZW bridge in both directions",
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
    worker.add_argument("--dry-run", action="store_true", help="audit without loading")
    worker.set_defaults(handler=_track32_worker_command)

    from n0_twam.track32_agilex.offline_cli import add_agilex_offline_commands

    add_agilex_offline_commands(track32_commands)
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
