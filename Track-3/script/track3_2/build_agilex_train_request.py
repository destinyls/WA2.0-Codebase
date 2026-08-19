#!/usr/bin/env python3
"""Build a hash-complete official AgileX mixed training request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
)
from n0_twam.embodiments import AGILEX_ACTION_SCHEMA
from n0_twam.integrations.worldarena.agilex_manifest import sha256_file
from n0_twam.integrations.worldarena.agilex_official_contracts import write_new_json
from n0_twam.track32_agilex.preflight import (
    agilex_profile_id,
    validate_qpos14_stage_b_checkpoint,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _recipe(args: argparse.Namespace) -> dict[str, int | str]:
    defaults = {
        "num_steps": 1500,
        "stop_after_step": 1500,
        "save_interval": 300,
        "val_interval": 100,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_latent_frames": 5,
    }
    values: dict[str, int] = {}
    for field, fallback in defaults.items():
        value = getattr(args, field, fallback)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
        values[field] = value
    if values["stop_after_step"] > values["num_steps"]:
        raise ValueError("stop_after_step cannot exceed num_steps")
    if values["batch_size"] != 1:
        raise ValueError("mixed AgileX training requires batch_size=1")
    return {"run_role": "final_refit", **values, "seed": int(args.seed)}


def build(args: argparse.Namespace) -> dict[str, object]:
    recipe = _recipe(args)
    raw_root = args.raw_root.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    artifact_root = args.artifact_root.expanduser().resolve(strict=True)
    base_model = args.base_model.expanduser().resolve(strict=True)
    empty_embedding = args.empty_embedding.expanduser().resolve(strict=True)
    init_from = args.init_from.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    request_path = args.output.expanduser().resolve(strict=False)
    transformer = init_from / "transformer" / TRANSFORMER_WEIGHTS_FILENAME
    transformer_config = json.loads(
        (init_from / "transformer" / "config.json").read_text(encoding="utf-8")
    )
    if not isinstance(transformer_config, dict):
        raise ValueError("initial transformer config must contain a JSON object")
    action_dim = transformer_config.get("action_dim")
    action_schema = transformer_config.get("action_schema")
    if action_dim == 20 and action_schema in (None, "ee20_pi05"):
        init_identity = audit_transformer_checkpoint(
            transformer, expected_action_dim=20
        )
    elif action_dim == 14 and action_schema == AGILEX_ACTION_SCHEMA:
        init_identity = audit_transformer_checkpoint(
            transformer, expected_action_dim=14
        )
    else:
        raise ValueError("initial checkpoint is not a compatible 20D or qpos14 model")
    files = {
        "source_manifest": artifact_root / "source_manifest.json",
        "conversion_receipt": artifact_root / "conversion_receipt.json",
        "latent_inventory": artifact_root / "latent_inventory.json",
        "repo_route_manifest": artifact_root / "repo_routes.json",
        "temporal_alignment": artifact_root / "temporal_alignment.json",
        "normalizer": artifact_root / "qpos14_normalizer.json",
    }
    artifact_identity = {
        "schema_version": 1,
        "embodiment_profile_id": "agilex_dual_qpos14_v1",
        "action_schema": AGILEX_ACTION_SCHEMA,
        "tactile_profile": "mixed",
        "source_manifest_file_sha256": sha256_file(files["source_manifest"]),
        "conversion_receipt_file_sha256": sha256_file(files["conversion_receipt"]),
        "latent_inventory_file_sha256": sha256_file(files["latent_inventory"]),
        "repo_route_manifest_file_sha256": sha256_file(files["repo_route_manifest"]),
        "temporal_alignment_file_sha256": sha256_file(files["temporal_alignment"]),
        "normalizer_file_sha256": sha256_file(files["normalizer"]),
    }
    if action_dim == 14:
        validate_qpos14_stage_b_checkpoint(
            init_from,
            expected_transformer_sha256=str(init_identity["sha256"]),
            expected_profile_id=agilex_profile_id("mixed"),
            expected_run_role=str(recipe["run_role"]),
            expected_artifact_identity=artifact_identity,
            audited_transformer_identity=init_identity,
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "profile": "mixed",
        "runtime": {
            "devices": list(range(args.devices)),
            "master_port": args.master_port,
            "accelerator_profile": "hcu_performance",
            "collective_network_interface": args.network_interface,
        },
        "paths": {
            "source_root": str(raw_root),
            "source_manifest": str(files["source_manifest"]),
            "source_manifest_sha256": artifact_identity["source_manifest_file_sha256"],
            "dataset_root": str(dataset_root),
            "artifact_root": str(artifact_root),
            "conversion_receipt": str(files["conversion_receipt"]),
            "conversion_receipt_sha256": artifact_identity[
                "conversion_receipt_file_sha256"
            ],
            "latent_inventory": str(files["latent_inventory"]),
            "latent_inventory_sha256": artifact_identity[
                "latent_inventory_file_sha256"
            ],
            "repo_route_manifest": str(files["repo_route_manifest"]),
            "repo_route_manifest_sha256": artifact_identity[
                "repo_route_manifest_file_sha256"
            ],
            "temporal_alignment": str(files["temporal_alignment"]),
            "temporal_alignment_sha256": artifact_identity[
                "temporal_alignment_file_sha256"
            ],
            "normalizer": str(files["normalizer"]),
            "normalizer_sha256": artifact_identity["normalizer_file_sha256"],
            "base_model": str(base_model),
            "empty_embedding": str(empty_embedding),
            "empty_embedding_sha256": sha256_file(empty_embedding),
            "init_from": str(init_from),
            "init_transformer_sha256": init_identity["sha256"],
            "resume_from": None,
            "resume_checkpoint_identity_sha256": None,
            "output_root": str(output_root),
        },
        "train": recipe,
    }
    request_sha = write_new_json(request_path, payload)
    return {
        "schema_version": 1,
        "status": "complete",
        "request": str(request_path),
        "request_sha256": request_sha,
        "run_id": args.run_id,
        "output_root": str(output_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--empty-embedding", type=Path, required=True)
    parser.add_argument("--init-from", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=29653)
    parser.add_argument("--network-interface", default="bond1")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--num-steps", type=_positive_int, default=1500)
    parser.add_argument("--stop-after-step", type=_positive_int, default=1500)
    parser.add_argument("--save-interval", type=_positive_int, default=300)
    parser.add_argument("--val-interval", type=_positive_int, default=100)
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=1)
    parser.add_argument("--max-latent-frames", type=_positive_int, default=5)
    return parser


def main() -> int:
    result = build(_parser().parse_args())
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
