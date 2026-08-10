# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Derive every hash field for a strict Franka training request."""

from __future__ import annotations

import json
import os
from pathlib import Path

from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
)
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.integrations.worldarena.franka_artifacts import (
    verify_franka_training_artifacts,
)
from n0_twam.integrations.worldarena.franka_manifest import sha256_file

from .request import REQUEST_SCHEMA_VERSION, load_track32_train_request


def build_track32_train_request(
    *,
    destination: Path,
    run_id: str,
    devices: tuple[int, ...],
    master_port: int,
    accelerator_profile: str,
    collective_network_interface: str | None,
    artifact_root: Path,
    lerobot_root: Path,
    base_model: Path,
    empty_embedding: Path,
    init_from: Path | None,
    resume_from: Path | None,
    output_root: Path,
    run_role: str,
    num_steps: int,
    stop_after_step: int,
    save_interval: int,
    val_interval: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_latent_frames: int,
    seed: int,
) -> dict[str, object]:
    """Audit existing inputs, create one immutable request, and parse it back."""

    if bool(init_from) == bool(resume_from):
        raise ValueError("exactly one init_from/resume_from is required")
    artifacts = verify_franka_training_artifacts(
        artifact_root=artifact_root,
        lerobot_root=lerobot_root,
        base_model=base_model,
        run_role=run_role,
    )
    empty = Path(empty_embedding).expanduser().resolve(strict=True)
    if empty.is_symlink() or not empty.is_file():
        raise ValueError("empty embedding must be a regular non-symlink file")
    init_path = None
    init_sha = None
    resume_path = None
    resume_sha = None
    if init_from is not None:
        init_path = Path(init_from).expanduser().resolve(strict=True)
        init_identity = audit_transformer_checkpoint(
            init_path / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
            expected_action_dim=20,
        )
        init_sha = str(init_identity["sha256"])
    else:
        assert resume_from is not None
        resume_path = Path(resume_from).expanduser().resolve(strict=True)
        resume_identity = build_strict_checkpoint_identity(
            capture_strict_checkpoint_snapshot(resume_path)
        )
        resume_sha = str(resume_identity["identity_sha256"])
    payload: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "run_id": run_id,
        "runtime": {
            "devices": list(devices),
            "master_port": master_port,
            "accelerator_profile": accelerator_profile,
            "collective_network_interface": collective_network_interface,
        },
        "paths": {
            "artifact_root": str(Path(artifact_root).expanduser().resolve(strict=True)),
            "lerobot_root": str(Path(lerobot_root).expanduser().resolve(strict=True)),
            "base_model": str(Path(base_model).expanduser().resolve(strict=True)),
            "empty_embedding": str(empty),
            "empty_embedding_sha256": sha256_file(empty),
            "init_from": None if init_path is None else str(init_path),
            "init_transformer_sha256": init_sha,
            "resume_from": None if resume_path is None else str(resume_path),
            "resume_checkpoint_identity_sha256": resume_sha,
            "output_root": str(Path(output_root).expanduser().resolve(strict=False)),
        },
        "artifacts": {
            "prepare_receipt_sha256": artifacts.prepare_receipt_sha256,
            "conversion_report_sha256": artifacts.conversion_report_sha256,
            "latent_inventory_file_sha256": artifacts.latent_inventory_sha256,
            "train_view_sha256": artifacts.train_view.view_sha256,
            "validation_view_sha256": (
                None
                if artifacts.validation_view is None
                else artifacts.validation_view.view_sha256
            ),
            "normalizer_sha256": artifacts.normalizer["normalizer_sha256"],
        },
        "train": {
            "run_role": run_role,
            "num_steps": num_steps,
            "stop_after_step": stop_after_step,
            "save_interval": save_interval,
            "val_interval": val_interval,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_latent_frames": max_latent_frames,
            "seed": seed,
        },
    }
    path = Path(destination).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    load_track32_train_request(path)
    return {
        "schema_version": 1,
        "status": "complete",
        "request": str(path),
        "request_sha256": sha256_file(path),
        "payload": payload,
    }


__all__ = ("build_track32_train_request",)
