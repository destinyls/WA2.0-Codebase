# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Construction-only template data for the portable AgileX request schema."""

from __future__ import annotations


def build_agilex_train_request_template(
    schema_version: int,
) -> dict[str, object]:
    """Return a new editable AgileX training-request template."""

    return {
        "schema_version": schema_version,
        "run_id": "agilex-development-v1",
        "profile": "mixed",
        "runtime": {
            "devices": list(range(8)),
            "master_port": 29643,
            "accelerator_profile": "portable",
            "collective_network_interface": None,
        },
        "paths": {
            "source_root": "/path/to/frozen/raw",
            "source_manifest": "/path/to/agilex-source-manifest.json",
            "source_manifest_sha256": "0" * 64,
            "dataset_root": "/path/to/converted/lerobot",
            "artifact_root": "/path/to/agilex/artifacts",
            "conversion_receipt": "/path/to/conversion-receipt.json",
            "conversion_receipt_sha256": "0" * 64,
            "latent_inventory": "/path/to/latent-inventory.json",
            "latent_inventory_sha256": "0" * 64,
            "repo_route_manifest": "/path/to/repo-route-manifest.json",
            "repo_route_manifest_sha256": "0" * 64,
            "temporal_alignment": "/path/to/temporal-alignment.json",
            "temporal_alignment_sha256": "0" * 64,
            "normalizer": "/path/to/qpos14-normalizer.json",
            "normalizer_sha256": "0" * 64,
            "base_model": "/path/to/n0-twam-base",
            "empty_embedding": "/path/to/n0-twam-base/empty_emb.pt",
            "empty_embedding_sha256": "0" * 64,
            "init_from": "/path/to/released-checkpoint",
            "init_transformer_sha256": "0" * 64,
            "resume_from": None,
            "resume_checkpoint_identity_sha256": None,
            "output_root": "/path/to/new/agilex-run",
        },
        "train": {
            "run_role": "development",
            "num_steps": 1500,
            "stop_after_step": 1500,
            "save_interval": 300,
            "val_interval": 100,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_latent_frames": 5,
            "seed": 20260811,
        },
    }


__all__ = ("build_agilex_train_request_template",)
