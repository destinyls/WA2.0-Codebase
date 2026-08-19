# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Atomic immutable serve-bundles for all AgileX tactile profiles."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path

from n0_twam.embodiments import AGILEX_ACTION_SCHEMA
from n0_twam.tactile_profiles import TACTILE_PROFILES

from ._agilex_serve_bundle_validation import (
    COMPONENTS,
    DIGEST_FIELDS,
    RECEIPT_FIELDS,
    RECEIPT_TO_DIGEST,
    audit_sources,
    canonical_sha256,
    digest,
    directory,
    file,
    json_object,
    validate_safety_contract,
    validate_task_routes,
)
from .agilex_manifest import sha256_file

SERVE_BUNDLE_SCHEMA_VERSION = 1


def _write_new(path: Path, data: bytes) -> str:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    result: str = sha256_file(path)
    return result


def _write_json(path: Path, payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    return _write_new(path, (raw + "\n").encode("utf-8"))


def _requested_digests(
    *,
    checkpoint: str,
    normalizer_file: str,
    normalizer_contract: str,
    source: str,
    route_file: str,
    route: str,
    contact: str,
    task: str,
    safety: str,
) -> dict[str, str]:
    values = locals()
    return {key: digest(value, label=key) for key, value in values.items()}


def _disjoint(destination: Path, sources: tuple[Path, ...]) -> None:
    if any(
        destination == source
        or destination in source.parents
        or source in destination.parents
        for source in sources
    ):
        raise ValueError("AgileX serve bundle output must be disjoint from inputs")


def _receipt_core(
    *,
    profile: str,
    checkpoint: Path,
    checkpoint_identity: Mapping[str, object],
    base_model: Path,
    encoder_identity: Mapping[str, object],
    component_names: list[str],
    digests: Mapping[str, str],
) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": SERVE_BUNDLE_SCHEMA_VERSION,
        "status": "complete",
        "action_schema": AGILEX_ACTION_SCHEMA,
        "tactile_profile": profile,
        "checkpoint_root": str(checkpoint),
        "checkpoint_identity": dict(checkpoint_identity),
        "base_model_root": str(base_model),
        "encoder_source_identity": dict(encoder_identity),
        "component_names": component_names,
    }
    core.update({field: digests[key] for field, key in RECEIPT_TO_DIGEST.items()})
    return core


def build_agilex_serve_bundle(
    *,
    checkpoint: Path,
    checkpoint_identity_sha256: str,
    base_model: Path,
    normalizer: Path,
    normalizer_file_sha256: str,
    normalizer_contract_sha256: str,
    source_manifest_sha256: str,
    repo_route_manifest: Path,
    repo_route_manifest_file_sha256: str,
    repo_route_manifest_sha256: str,
    tactile_profile: str,
    contact_profile_contract_sha256: str,
    task_routes: Mapping[str, object],
    task_routes_sha256: str,
    safety_contract: Mapping[str, object],
    safety_contract_sha256: str,
    output: Path,
) -> dict[str, object]:
    """Validate exact training/deployment contracts and publish a new bundle."""

    if tactile_profile not in TACTILE_PROFILES:
        raise ValueError("unknown AgileX tactile profile")
    digests = _requested_digests(
        checkpoint=checkpoint_identity_sha256,
        normalizer_file=normalizer_file_sha256,
        normalizer_contract=normalizer_contract_sha256,
        source=source_manifest_sha256,
        route_file=repo_route_manifest_file_sha256,
        route=repo_route_manifest_sha256,
        contact=contact_profile_contract_sha256,
        task=task_routes_sha256,
        safety=safety_contract_sha256,
    )
    audited = audit_sources(
        checkpoint=checkpoint,
        base_model=base_model,
        normalizer=normalizer,
        route_path=repo_route_manifest,
        profile=tactile_profile,
        digests=digests,
    )
    checkpoint_root, base_root, identity, encoder, components, training_route = audited
    validate_task_routes(
        task_routes,
        expected_sha256=digests["task"],
        training_route=training_route,
        profile=tactile_profile,
    )
    validate_safety_contract(safety_contract, expected_sha256=digests["safety"])
    destination = Path(output).expanduser().resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"AgileX serve bundle already exists: {destination}")
    normalizer_path = file(normalizer, label="AgileX normalizer")
    route_path = file(repo_route_manifest, label="AgileX route manifest")
    _disjoint(
        destination,
        (checkpoint_root, base_root, normalizer_path, route_path),
    )
    assets = base_root / "assets"
    has_assets = assets.exists()
    if has_assets:
        directory(assets, label="AgileX assets source")
    names = [*components, *(["assets"] if has_assets else [])]
    core = _receipt_core(
        profile=tactile_profile,
        checkpoint=checkpoint_root,
        checkpoint_identity=identity,
        base_model=base_root,
        encoder_identity=encoder,
        component_names=names,
        digests=digests,
    )
    receipt = {**core, "bundle_identity_sha256": canonical_sha256(core)}

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.incomplete"
    staging.mkdir(mode=0o755)
    try:
        for name, source in components.items():
            (staging / name).symlink_to(source)
        if has_assets:
            (staging / "assets").symlink_to(assets)
        copied = (
            _write_new(staging / "normalizer.json", normalizer_path.read_bytes()),
            _write_new(staging / "repo_route_manifest.json", route_path.read_bytes()),
        )
        if copied != (digests["normalizer_file"], digests["route_file"]):
            raise RuntimeError("AgileX copied artifact identity changed")
        receipt_file_sha = _write_json(staging / "serve_bundle_receipt.json", receipt)
        os.replace(staging, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        **receipt,
        "bundle_root": str(destination),
        "receipt": str(destination / "serve_bundle_receipt.json"),
        "receipt_file_sha256": receipt_file_sha,
    }


def verify_agilex_serve_bundle(
    bundle: Path,
    *,
    expected_receipt_file_sha256: str,
) -> dict[str, object]:
    """Recompute every checkpoint, encoder, link, and copied-file identity."""

    root = directory(bundle, label="AgileX serve bundle")
    receipt_path = file(
        root / "serve_bundle_receipt.json", label="AgileX serve receipt"
    )
    if sha256_file(receipt_path) != digest(
        expected_receipt_file_sha256, label="receipt file"
    ):
        raise ValueError("AgileX serve receipt file SHA-256 mismatch")
    receipt = json_object(receipt_path, label="AgileX serve receipt")
    core = {
        key: value for key, value in receipt.items() if key != "bundle_identity_sha256"
    }
    if (
        set(receipt) != RECEIPT_FIELDS
        or receipt.get("schema_version") != SERVE_BUNDLE_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("action_schema") != AGILEX_ACTION_SCHEMA
        or receipt.get("tactile_profile") not in TACTILE_PROFILES
        or receipt.get("bundle_identity_sha256") != canonical_sha256(core)
    ):
        raise ValueError("AgileX serve receipt contract mismatch")
    for field in DIGEST_FIELDS:
        digest(receipt.get(field), label=field)
    digests = {key: str(receipt[field]) for field, key in RECEIPT_TO_DIGEST.items()}
    checkpoint = directory(
        Path(str(receipt["checkpoint_root"])), label="AgileX checkpoint source"
    )
    base = directory(
        Path(str(receipt["base_model_root"])), label="AgileX base model source"
    )
    if (
        str(checkpoint) != receipt["checkpoint_root"]
        or str(base) != receipt["base_model_root"]
    ):
        raise ValueError("AgileX serve receipt source paths are not canonical")
    audited = audit_sources(
        checkpoint=checkpoint,
        base_model=base,
        normalizer=root / "normalizer.json",
        route_path=root / "repo_route_manifest.json",
        profile=str(receipt["tactile_profile"]),
        digests=digests,
    )
    _, _, identity, encoder, components, _ = audited
    if (
        identity != receipt["checkpoint_identity"]
        or encoder != receipt["encoder_source_identity"]
    ):
        raise ValueError("AgileX serve source identity changed after publication")
    names = receipt.get("component_names")
    if names not in (list(COMPONENTS), [*COMPONENTS, "assets"]):
        raise ValueError("AgileX serve component contract mismatch")
    if "assets" in names:
        components["assets"] = directory(base / "assets", label="AgileX assets source")
    expected = {
        *names,
        "normalizer.json",
        "repo_route_manifest.json",
        "serve_bundle_receipt.json",
    }
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("AgileX serve bundle contains undeclared entries")
    for name, target in components.items():
        link = root / name
        if not link.is_symlink() or link.resolve(strict=True) != target:
            raise ValueError(f"AgileX serve bundle link changed: {name}")
    return receipt


__all__ = ("build_agilex_serve_bundle", "verify_agilex_serve_bundle")
