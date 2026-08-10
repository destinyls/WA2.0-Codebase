# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Immutable self-contained serve-bundle identity for Franka Track 3.2."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.data.encoder_source_identity import (
    build_encoder_source_identity,
    validate_encoder_source_identity,
)

from .franka_manifest import canonical_sha256, sha256_file

SERVE_BUNDLE_SCHEMA_VERSION = 1


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _write_new_json(path: Path, payload: object) -> str:
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
    return sha256_file(path)


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def build_franka_serve_bundle(
    *,
    checkpoint: Path,
    checkpoint_identity_sha256: str,
    base_model: Path,
    normalizer: Path,
    normalizer_sha256: str,
    output: Path,
) -> dict[str, object]:
    """Publish a new bundle without copying the large model components."""

    checkpoint_root = Path(checkpoint).expanduser().resolve(strict=True)
    base_root = Path(base_model).expanduser().resolve(strict=True)
    normalizer_path = Path(normalizer).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"serve bundle already exists: {destination}")
    if normalizer_path.is_symlink() or not normalizer_path.is_file():
        raise ValueError("Franka normalizer must be a regular non-symlink file")
    if any(
        _overlaps(destination, source)
        for source in (checkpoint_root, base_root, normalizer_path)
    ):
        raise ValueError("serve bundle output must be disjoint from all inputs")
    snapshot = capture_strict_checkpoint_snapshot(checkpoint_root)
    checkpoint_identity = build_strict_checkpoint_identity(snapshot)
    if checkpoint_identity.get("identity_sha256") != checkpoint_identity_sha256:
        raise ValueError("checkpoint identity differs from requested serve source")
    meta = snapshot.train_meta
    if (
        meta.get("track32_profile_id") != "franka_track32_vision_only_v1"
        or meta.get("action_schema") != "ee20_absee"
        or meta.get("tactile_profile", "vision_only") != "vision_only"
        or meta.get("tactile_mode") != "disabled"
        or meta.get("used_action_channel_ids") != list(range(10))
    ):
        raise ValueError("checkpoint is not a compatible Franka Track 3.2 model")
    normalizer_payload = _load_json(normalizer_path, label="Franka normalizer")
    if normalizer_payload.get("normalizer_sha256") != normalizer_sha256:
        raise ValueError("normalizer semantic SHA256 differs from request")
    encoder_identity = validate_encoder_source_identity(
        build_encoder_source_identity(base_root)
    )

    components = {
        "transformer": checkpoint_root / "transformer",
        "vae": base_root / "vae",
        "tokenizer": base_root / "tokenizer",
        "text_encoder": base_root / "text_encoder",
    }
    for source in components.values():
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f"serve source must be a real directory: {source}")
    optional_assets = base_root / "assets"
    if optional_assets.exists() and (
        not optional_assets.is_dir() or optional_assets.is_symlink()
    ):
        raise ValueError("serve assets source must be a real directory")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{os.getpid()}.incomplete"
    if staging.exists():
        raise FileExistsError(f"serve bundle staging path already exists: {staging}")
    staging.mkdir(mode=0o755)
    try:
        for name, source in components.items():
            (staging / name).symlink_to(source)
        component_names = list(components)
        if optional_assets.is_dir():
            (staging / "assets").symlink_to(optional_assets)
            component_names.append("assets")
        shutil.copyfile(normalizer_path, staging / "normalizer.json")
        os.chmod(staging / "normalizer.json", 0o444)
        normalizer_file_sha256 = sha256_file(staging / "normalizer.json")
        core: dict[str, object] = {
            "schema_version": SERVE_BUNDLE_SCHEMA_VERSION,
            "status": "complete",
            "profile": "franka_track32_vision_only_v1",
            "run_role": meta.get("run_role"),
            "checkpoint_root": str(checkpoint_root),
            "checkpoint_identity": checkpoint_identity,
            "base_model_root": str(base_root),
            "encoder_source_identity": encoder_identity,
            "component_names": component_names,
            "normalizer_sha256": normalizer_sha256,
            "normalizer_file_sha256": normalizer_file_sha256,
            "action_schema": "ee20_absee",
            "active_action_channel_ids": list(range(10)),
            "tactile_profile": "vision_only",
            "tactile_mode": "disabled",
        }
        receipt = {**core, "bundle_identity_sha256": canonical_sha256(core)}
        staging_receipt = staging / "serve_bundle_receipt.json"
        receipt_file_sha256 = _write_new_json(staging_receipt, receipt)
        os.replace(staging, destination)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    receipt_path = destination / "serve_bundle_receipt.json"
    return {
        **receipt,
        "bundle_root": str(destination),
        "receipt": str(receipt_path),
        "receipt_file_sha256": receipt_file_sha256,
    }


def verify_franka_serve_bundle(
    bundle: Path,
    *,
    expected_receipt_file_sha256: str,
) -> dict[str, object]:
    """Re-audit every checkpoint, encoder, and normalizer binding."""

    root = Path(bundle).expanduser().resolve(strict=True)
    receipt_path = root / "serve_bundle_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("serve bundle receipt must be a regular file")
    if sha256_file(receipt_path) != expected_receipt_file_sha256:
        raise ValueError("serve bundle receipt file SHA256 mismatch")
    receipt = _load_json(receipt_path, label="serve bundle receipt")
    core = {
        key: value for key, value in receipt.items() if key != "bundle_identity_sha256"
    }
    if (
        receipt.get("schema_version") != SERVE_BUNDLE_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("bundle_identity_sha256") != canonical_sha256(core)
        or receipt.get("profile") != "franka_track32_vision_only_v1"
        or receipt.get("action_schema") != "ee20_absee"
        or receipt.get("active_action_channel_ids") != list(range(10))
        or receipt.get("tactile_mode") != "disabled"
        or receipt.get("component_names")
        not in (
            ["transformer", "vae", "tokenizer", "text_encoder"],
            ["transformer", "vae", "tokenizer", "text_encoder", "assets"],
        )
    ):
        raise ValueError("serve bundle receipt contract mismatch")
    checkpoint = Path(str(receipt["checkpoint_root"])).resolve(strict=True)
    current_checkpoint = build_strict_checkpoint_identity(
        capture_strict_checkpoint_snapshot(checkpoint)
    )
    if current_checkpoint != receipt.get("checkpoint_identity"):
        raise ValueError("serve checkpoint changed after bundle publication")
    base_model = Path(str(receipt["base_model_root"])).resolve(strict=True)
    current_encoder = validate_encoder_source_identity(
        build_encoder_source_identity(base_model)
    )
    if current_encoder != receipt.get("encoder_source_identity"):
        raise ValueError("serve encoder sources changed after bundle publication")
    for name, expected in {
        "transformer": checkpoint / "transformer",
        "vae": base_model / "vae",
        "tokenizer": base_model / "tokenizer",
        "text_encoder": base_model / "text_encoder",
    }.items():
        link = root / name
        if not link.is_symlink() or link.resolve(strict=True) != expected:
            raise ValueError(f"serve bundle link changed: {name}")
    assets_link = root / "assets"
    expects_assets = "assets" in receipt["component_names"]
    if expects_assets:
        if (
            not assets_link.is_symlink()
            or assets_link.resolve(strict=True) != base_model / "assets"
        ):
            raise ValueError("serve bundle assets link changed")
    elif assets_link.exists() or assets_link.is_symlink():
        raise ValueError("serve bundle contains an undeclared assets component")
    normalizer_path = root / "normalizer.json"
    if (
        normalizer_path.is_symlink()
        or not normalizer_path.is_file()
        or sha256_file(normalizer_path) != receipt.get("normalizer_file_sha256")
    ):
        raise ValueError("serve bundle normalizer bytes changed")
    normalizer_payload = _load_json(normalizer_path, label="serve normalizer")
    if normalizer_payload.get("normalizer_sha256") != receipt.get("normalizer_sha256"):
        raise ValueError("serve normalizer semantic identity changed")
    return receipt


__all__ = (
    "build_franka_serve_bundle",
    "verify_franka_serve_bundle",
)
