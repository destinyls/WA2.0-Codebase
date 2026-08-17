# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Content-addressed aggregate conversion and latent artifacts for AgileX."""

from __future__ import annotations

import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from n0_twam.integrations.univtac.convert_lerobot import (
    build_lerobot_table_inventory,
)

from ._agilex_artifact_validation import (
    CONVERSION_SCHEMA_VERSION,
    LATENT_INVENTORY_SCHEMA_VERSION,
    require_digest,
    require_integer,
    resolve_dataset_layout,
    validate_conversion_shape,
    validate_latent_shape,
)
from .agilex_manifest import (
    AgileXRepoRoute,
    canonical_sha256,
    sha256_file,
)


@dataclass(frozen=True)
class VerifiedAgileXConversion:
    dataset_root: Path
    repo_ids: tuple[str, ...]
    conversion_identity_sha256: str


@dataclass(frozen=True)
class VerifiedAgileXLatents:
    dataset_root: Path
    repo_ids: tuple[str, ...]
    conversion_receipt_sha256: str
    inventory_sha256: str
    record_count: int


def _conversion_core(
    *,
    dataset_root: Path,
    routes: Sequence[AgileXRepoRoute],
    source_manifest_sha256: str,
    repo_route_manifest_sha256: str,
    temporal_alignment_contract_sha256: str,
) -> dict[str, object]:
    root, ordered, repo_paths = resolve_dataset_layout(dataset_root, routes)
    repositories: list[dict[str, object]] = []
    for route in ordered:
        repo = repo_paths[route.repo_id]
        repositories.append(
            {
                "repo_id": route.repo_id,
                "dataset_relative_path": route.repo_id,
                "formal": True,
                "action_schema": route.action_schema,
                "action_label_source": route.action_label_source,
                "action_label_offset": route.action_contract.label_offset,
                "repo_route_identity": route.route_identity,
                "temporal_alignment_identity": route.temporal_alignment_identity,
                "table_inventory": build_lerobot_table_inventory(repo),
            }
        )
    return {
        "schema_version": CONVERSION_SCHEMA_VERSION,
        "status": "complete",
        "kind": "agilex_conversion",
        "source_manifest_sha256": require_digest(
            source_manifest_sha256, label="source manifest SHA-256"
        ),
        "repo_route_manifest_sha256": require_digest(
            repo_route_manifest_sha256, label="repo-route manifest SHA-256"
        ),
        "temporal_alignment_contract_sha256": require_digest(
            temporal_alignment_contract_sha256,
            label="temporal alignment contract SHA-256",
        ),
        "dataset_root": str(root),
        "repo_ids": [route.repo_id for route in ordered],
        "repositories": repositories,
    }


def build_agilex_conversion_receipt(
    *,
    dataset_root: Path,
    routes: Sequence[AgileXRepoRoute],
    source_manifest_sha256: str,
    repo_route_manifest_sha256: str,
    temporal_alignment_contract_sha256: str,
) -> dict[str, object]:
    """Recompute all selected LeRobot bytes and build a canonical receipt."""

    core = _conversion_core(
        dataset_root=dataset_root,
        routes=routes,
        source_manifest_sha256=source_manifest_sha256,
        repo_route_manifest_sha256=repo_route_manifest_sha256,
        temporal_alignment_contract_sha256=temporal_alignment_contract_sha256,
    )
    receipt = {**core, "conversion_identity_sha256": canonical_sha256(core)}
    validate_conversion_shape(receipt)
    return receipt


def verify_agilex_conversion_receipt(
    payload: Mapping[str, object],
    *,
    dataset_root: Path,
    routes: Sequence[AgileXRepoRoute],
    source_manifest_sha256: str,
    repo_route_manifest_sha256: str,
    temporal_alignment_contract_sha256: str,
) -> VerifiedAgileXConversion:
    """Verify receipt schema, canonical identity, routes, and current table bytes."""

    validate_conversion_shape(payload)
    core = {
        key: value
        for key, value in payload.items()
        if key != "conversion_identity_sha256"
    }
    identity = require_digest(
        payload["conversion_identity_sha256"], label="conversion identity"
    )
    if canonical_sha256(core) != identity:
        raise ValueError("AgileX conversion canonical identity mismatch")
    expected = _conversion_core(
        dataset_root=dataset_root,
        routes=routes,
        source_manifest_sha256=source_manifest_sha256,
        repo_route_manifest_sha256=repo_route_manifest_sha256,
        temporal_alignment_contract_sha256=temporal_alignment_contract_sha256,
    )
    if core != expected:
        raise ValueError("AgileX conversion receipt differs from live data/routes")
    return VerifiedAgileXConversion(
        dataset_root=Path(str(expected["dataset_root"])),
        repo_ids=tuple(cast(list[str], expected["repo_ids"])),
        conversion_identity_sha256=identity,
    )


def _verify_conversion_for_latents(
    *,
    dataset_root: Path,
    routes: Sequence[AgileXRepoRoute],
    conversion_receipt: Mapping[str, object],
) -> VerifiedAgileXConversion:
    validate_conversion_shape(conversion_receipt)
    return verify_agilex_conversion_receipt(
        conversion_receipt,
        dataset_root=dataset_root,
        routes=routes,
        source_manifest_sha256=require_digest(
            conversion_receipt["source_manifest_sha256"], label="source manifest"
        ),
        repo_route_manifest_sha256=require_digest(
            conversion_receipt["repo_route_manifest_sha256"], label="route manifest"
        ),
        temporal_alignment_contract_sha256=require_digest(
            conversion_receipt["temporal_alignment_contract_sha256"],
            label="temporal alignment contract",
        ),
    )


def _hash_payload(path: Path, root: Path) -> dict[str, object]:
    initial = path.lstat()
    if (
        stat.S_ISLNK(initial.st_mode)
        or not stat.S_ISREG(initial.st_mode)
        or initial.st_size <= 0
    ):
        raise ValueError(f"AgileX latent payload must be a regular file: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"AgileX latent payload escaped dataset root: {path}"
        ) from error
    digest = sha256_file(resolved)
    final = path.lstat()
    before = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
    after = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
    if before != after or stat.S_ISLNK(final.st_mode):
        raise ValueError(f"AgileX latent payload changed while hashing: {path}")
    return {"path": relative, "size_bytes": final.st_size, "sha256": digest}


def _latent_records(
    root: Path,
    routes: Sequence[AgileXRepoRoute],
    repo_paths: Mapping[str, Path],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for route in routes:
        repo = repo_paths[route.repo_id]
        names = ["latents"]
        if route.tactile_keys:
            names.append("latents_tactile")
        for name in names:
            payload_root = repo / name
            try:
                metadata = payload_root.lstat()
            except OSError as error:
                raise ValueError(f"AgileX {name} root is unavailable") from error
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"AgileX {name} root must be a regular directory")
            payload_root = payload_root.resolve(strict=True)
            try:
                payload_root.relative_to(repo)
            except ValueError as error:
                raise ValueError(f"AgileX {name} root escaped repo") from error
            before = len(records)
            for candidate in payload_root.rglob("*"):
                metadata = candidate.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValueError(
                        f"AgileX latent tree contains a symlink: {candidate}"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(
                        f"AgileX latent tree contains a special file: {candidate}"
                    )
                if candidate.suffix == ".pth":
                    records.append(_hash_payload(candidate, root))
            if len(records) == before:
                raise ValueError(f"AgileX {name} root contains no .pth payloads")
    return sorted(records, key=lambda record: str(record["path"]))


def _latent_core(
    *,
    dataset_root: Path,
    routes: Sequence[AgileXRepoRoute],
    conversion_receipt: Mapping[str, object],
    conversion_receipt_sha256: str,
) -> dict[str, object]:
    root, ordered, repo_paths = resolve_dataset_layout(dataset_root, routes)
    digest = require_digest(
        conversion_receipt_sha256, label="conversion receipt file SHA-256"
    )
    records = _latent_records(root, ordered, repo_paths)
    return {
        "schema_version": LATENT_INVENTORY_SCHEMA_VERSION,
        "status": "complete",
        "kind": "agilex_latent_inventory",
        "source_manifest_sha256": conversion_receipt["source_manifest_sha256"],
        "repo_route_manifest_sha256": conversion_receipt["repo_route_manifest_sha256"],
        "temporal_alignment_contract_sha256": conversion_receipt[
            "temporal_alignment_contract_sha256"
        ],
        "dataset_root": str(root),
        "repo_ids": list(cast(list[str], conversion_receipt["repo_ids"])),
        "conversion_receipt_sha256": digest,
        "conversion_identity_sha256": conversion_receipt["conversion_identity_sha256"],
        "record_count": len(records),
        "records": records,
    }


def build_agilex_latent_inventory(
    *,
    dataset_root: Path,
    routes: Sequence[AgileXRepoRoute],
    conversion_receipt: Mapping[str, object],
    conversion_receipt_sha256: str,
) -> dict[str, object]:
    """Enumerate exactly the profile-relevant immutable ``.pth`` payloads."""

    _verify_conversion_for_latents(
        dataset_root=dataset_root,
        routes=routes,
        conversion_receipt=conversion_receipt,
    )
    core = _latent_core(
        dataset_root=dataset_root,
        routes=routes,
        conversion_receipt=conversion_receipt,
        conversion_receipt_sha256=conversion_receipt_sha256,
    )
    inventory = {**core, "inventory_sha256": canonical_sha256(core)}
    validate_latent_shape(inventory)
    return inventory


def verify_agilex_latent_inventory(
    payload: Mapping[str, object],
    *,
    dataset_root: Path,
    routes: Sequence[AgileXRepoRoute],
    conversion_receipt: Mapping[str, object],
    conversion_receipt_sha256: str,
) -> VerifiedAgileXLatents:
    """Verify exact latent roster, bytes, provenance, and conversion binding."""

    _verify_conversion_for_latents(
        dataset_root=dataset_root,
        routes=routes,
        conversion_receipt=conversion_receipt,
    )
    validate_latent_shape(payload)
    core = {key: value for key, value in payload.items() if key != "inventory_sha256"}
    identity = require_digest(
        payload["inventory_sha256"], label="latent inventory identity"
    )
    if canonical_sha256(core) != identity:
        raise ValueError("AgileX latent inventory canonical identity mismatch")
    expected = _latent_core(
        dataset_root=dataset_root,
        routes=routes,
        conversion_receipt=conversion_receipt,
        conversion_receipt_sha256=conversion_receipt_sha256,
    )
    if core != expected:
        raise ValueError("AgileX latent inventory differs from live payload bytes")
    return VerifiedAgileXLatents(
        dataset_root=Path(str(expected["dataset_root"])),
        repo_ids=tuple(cast(list[str], expected["repo_ids"])),
        conversion_receipt_sha256=str(expected["conversion_receipt_sha256"]),
        inventory_sha256=identity,
        record_count=require_integer(expected["record_count"], label="record_count"),
    )


__all__ = (
    "CONVERSION_SCHEMA_VERSION",
    "LATENT_INVENTORY_SCHEMA_VERSION",
    "VerifiedAgileXConversion",
    "VerifiedAgileXLatents",
    "build_agilex_conversion_receipt",
    "build_agilex_latent_inventory",
    "verify_agilex_conversion_receipt",
    "verify_agilex_latent_inventory",
)
