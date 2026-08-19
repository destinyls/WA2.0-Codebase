# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Private fail-closed validation for AgileX serve-bundle publication."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
    validate_transformer_identity_match,
)
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    StrictCheckpointSnapshot,
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.checkpointing.training_lineage import (
    validate_action_migration_report_for_contract,
)
from n0_twam.data.encoder_source_identity import (
    build_encoder_source_identity,
    validate_encoder_source_identity,
)
from n0_twam.embodiments import (
    AGILEX_ACTION_SCHEMA,
    AGILEX_RGB_KEYS,
    validate_repo_route_manifest_contract,
)
from n0_twam.tactile_profiles import VISION_ONLY, validate_tactile_profile_contract

from .agilex_manifest import canonical_sha256, sha256_file
from .agilex_normalizer import load_agilex_normalizer
from .agilex_policy_schema import parse_routes, parse_safety

COMPONENTS = ("transformer", "vae", "tokenizer", "text_encoder")
ARTIFACT_FIELDS = frozenset(
    "schema_version embodiment_profile_id action_schema tactile_profile "
    "source_manifest_file_sha256 conversion_receipt_file_sha256 "
    "latent_inventory_file_sha256 repo_route_manifest_file_sha256 "
    "temporal_alignment_file_sha256 normalizer_file_sha256".split()
)
RECEIPT_FIELDS = frozenset(
    "schema_version status action_schema tactile_profile checkpoint_root "
    "checkpoint_identity checkpoint_identity_sha256 base_model_root "
    "encoder_source_identity normalizer_file_sha256 normalizer_contract_sha256 "
    "source_manifest_sha256 repo_route_manifest_file_sha256 "
    "repo_route_manifest_sha256 contact_profile_contract_sha256 "
    "task_routes_sha256 safety_contract_sha256 component_names "
    "bundle_identity_sha256".split()
)
NON_DIGEST_FIELDS = frozenset(
    "schema_version status action_schema tactile_profile checkpoint_root "
    "checkpoint_identity base_model_root encoder_source_identity "
    "component_names".split()
)
DIGEST_FIELDS = RECEIPT_FIELDS - NON_DIGEST_FIELDS
RECEIPT_TO_DIGEST = {
    "checkpoint_identity_sha256": "checkpoint",
    "normalizer_file_sha256": "normalizer_file",
    "normalizer_contract_sha256": "normalizer_contract",
    "source_manifest_sha256": "source",
    "repo_route_manifest_file_sha256": "route_file",
    "repo_route_manifest_sha256": "route",
    "contact_profile_contract_sha256": "contact",
    "task_routes_sha256": "task",
    "safety_contract_sha256": "safety",
}


def digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not value.strip("0")
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a non-placeholder lowercase SHA-256")
    return value


def regular(path: Path, *, label: str, directory: bool) -> Path:
    raw = Path(path).expanduser()
    valid = raw.is_dir() if directory else raw.is_file()
    if raw.is_symlink() or not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} must be a regular non-symlink {kind}")
    return raw.resolve(strict=True)


def directory(path: Path, *, label: str) -> Path:
    return regular(path, label=label, directory=True)


def file(path: Path, *, label: str) -> Path:
    return regular(path, label=label, directory=False)


def json_object(path: Path, *, label: str) -> dict[str, object]:
    source = file(path, label=label)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _route_modalities(
    route: Mapping[str, object], profile: str
) -> tuple[frozenset[str], frozenset[str]]:
    routes = route.get("routes")
    if not isinstance(routes, Mapping) or not routes:
        raise ValueError("AgileX route manifest contains no repositories")
    values = tuple(routes.values())
    if any(not isinstance(value, Mapping) for value in values):
        raise ValueError("AgileX route manifest contains an invalid route")
    expected_rgb = list(AGILEX_RGB_KEYS)
    if any(value.get("rgb_keys") != expected_rgb for value in values):
        raise ValueError("every AgileX training route must use canonical RGB keys")
    tactile = tuple(bool(value.get("tactile_keys")) for value in values)
    wrench = tuple(bool(value.get("wrench_keys")) for value in values)
    if profile == "vision_tactile" and not (all(tactile) and all(wrench)):
        raise ValueError("vision_tactile route profile is incomplete")
    mixed = (
        any(tactile)
        and any(not value for value in tactile)
        and all(
            touch or not force for touch, force in zip(tactile, wrench, strict=True)
        )
    )
    if profile == "mixed" and not mixed:
        raise ValueError("mixed route profile is incompatible")
    if profile == VISION_ONLY and (any(tactile) or any(wrench)):
        raise ValueError("vision_only route profile exposes contact inputs")
    tactile_keys = frozenset(
        key for value in values for key in value.get("tactile_keys", ())
    )
    wrench_keys = frozenset(
        key for value in values for key in value.get("wrench_keys", ())
    )
    return tactile_keys, wrench_keys


def validate_task_routes(
    task_routes: Mapping[str, object],
    *,
    expected_sha256: str,
    training_route: Mapping[str, object],
    profile: str,
) -> None:
    routes, routes_sha256 = parse_routes(task_routes)
    if routes_sha256 != expected_sha256:
        raise ValueError("AgileX task routes semantic SHA-256 differs")
    allowed_tactile, allowed_wrench = _route_modalities(training_route, profile)
    for route in routes.values():
        if route.tactile_required != bool(route.tactile_keys):
            raise ValueError("tactile_required must match non-empty tactile_keys")
        if route.wrench_required != bool(route.wrench_keys):
            raise ValueError("wrench_required must match non-empty wrench_keys")
        if route.wrench_required and not route.tactile_required:
            raise ValueError("wrench-required routes must also require tactile")
        if not set(route.tactile_keys) <= allowed_tactile:
            raise ValueError("task tactile keys exceed the signed training routes")
        if not set(route.wrench_keys) <= allowed_wrench:
            raise ValueError("task wrench keys exceed the signed training routes")
        if profile == "vision_tactile" and (
            not route.tactile_required or not route.wrench_required
        ):
            raise ValueError(
                "vision_tactile task routes must require tactile and wrench"
            )


def validate_safety_contract(
    safety_contract: Mapping[str, object], *, expected_sha256: str
) -> None:
    if parse_safety(safety_contract).contract_sha256 != expected_sha256:
        raise ValueError("AgileX safety contract semantic SHA-256 differs")


def _validate_checkpoint(
    snapshot: StrictCheckpointSnapshot,
    *,
    profile: str,
    digests: Mapping[str, str],
) -> None:
    config, meta = snapshot.transformer_config, snapshot.train_meta
    if config.get("action_schema") != AGILEX_ACTION_SCHEMA:
        raise ValueError("AgileX checkpoint action schema is incompatible")
    if config.get("action_dim") != 14:
        raise ValueError("AgileX checkpoint action dimension must be 14")
    expected_mode = "disabled" if profile == VISION_ONLY else "enabled"
    if (
        meta.get("track32_profile_id") != f"agilex_track3_{profile}_v1"
        or meta.get("tactile_profile") != profile
        or meta.get("tactile_mode") != expected_mode
    ):
        raise ValueError("AgileX checkpoint profile differs from the request")
    tactile = validate_tactile_profile_contract(meta.get("tactile_profile_contract"))
    if tactile["profile"] != profile:
        raise ValueError("AgileX checkpoint tactile profile contract differs")
    actual = audit_transformer_checkpoint(
        snapshot.checkpoint_root / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
        expected_action_dim=14,
    )
    payloads = (meta, snapshot.training_state, snapshot.completion)
    for label, payload in zip(
        ("metadata", "state", "completion"), payloads, strict=True
    ):
        validate_transformer_identity_match(
            payload.get("transformer_identity"),
            actual,
            expected_action_dim=14,
            label=f"AgileX {label}",
        )
        if payload.get("tactile_profile_contract") != tactile:
            raise ValueError("AgileX checkpoint tactile profile sidecars differ")
    artifacts = meta.get("track32_artifact_identity")
    if not isinstance(artifacts, Mapping) or set(artifacts) != ARTIFACT_FIELDS:
        raise ValueError("AgileX checkpoint lacks a formal artifact identity")
    if any(
        payload.get("track32_artifact_identity") != artifacts for payload in payloads
    ):
        raise ValueError("AgileX checkpoint artifact identity sidecars differ")
    expected_artifacts = {
        "schema_version": 1,
        "embodiment_profile_id": "agilex_dual_qpos14_v1",
        "action_schema": AGILEX_ACTION_SCHEMA,
        "tactile_profile": profile,
        "source_manifest_file_sha256": digests["source"],
        "repo_route_manifest_file_sha256": digests["route_file"],
        "normalizer_file_sha256": digests["normalizer_file"],
    }
    if any(artifacts.get(key) != value for key, value in expected_artifacts.items()):
        raise ValueError("AgileX checkpoint formal artifact identity differs")
    for key in ARTIFACT_FIELDS - set(expected_artifacts):
        digest(artifacts.get(key), label=f"artifact {key}")
    lineage = meta.get("training_lineage")
    if not isinstance(lineage, Mapping) or any(
        payload.get("training_lineage") != lineage for payload in payloads
    ):
        raise ValueError("AgileX checkpoint training lineage is incomplete")
    expected_lineage = {
        "action_schema": AGILEX_ACTION_SCHEMA,
        "tactile_profile": profile,
        "contact_profile_contract_sha256": digests["contact"],
        "repo_route_manifest_sha256": digests["route"],
        "normalizer_sha256": digests["normalizer_file"],
    }
    if any(lineage.get(key) != value for key, value in expected_lineage.items()):
        raise ValueError("AgileX checkpoint training lineage differs")
    migration = snapshot.action_migration_report
    if migration is None:
        raise ValueError("AgileX checkpoint lost its action migration lineage")
    validate_action_migration_report_for_contract(
        migration,
        source_action_dim=20,
        source_action_schema="ee20_pi05",
        target_action_dim=14,
        target_action_schema=AGILEX_ACTION_SCHEMA,
        initialized_target_only_prefixes=("local_tactile_", "agilex_wrench_"),
    )


def component_sources(checkpoint: Path, base: Path) -> dict[str, Path]:
    result = {
        "transformer": checkpoint / "transformer",
        "vae": base / "vae",
        "tokenizer": base / "tokenizer",
        "text_encoder": base / "text_encoder",
    }
    for name, source in result.items():
        directory(source, label=f"AgileX {name} source")
    return result


def audit_sources(
    *,
    checkpoint: Path,
    base_model: Path,
    normalizer: Path,
    route_path: Path,
    profile: str,
    digests: Mapping[str, str],
) -> tuple[
    Path,
    Path,
    dict[str, object],
    dict[str, object],
    dict[str, Path],
    dict[str, object],
]:
    checkpoint_root = directory(checkpoint, label="AgileX checkpoint")
    base_root = directory(base_model, label="AgileX base model")
    normalizer_path = file(normalizer, label="AgileX normalizer")
    route_file = file(route_path, label="AgileX route manifest")
    if sha256_file(normalizer_path) != digests["normalizer_file"]:
        raise ValueError("AgileX normalizer file SHA-256 differs")
    if sha256_file(route_file) != digests["route_file"]:
        raise ValueError("AgileX route manifest file SHA-256 differs")
    route = validate_repo_route_manifest_contract(
        json_object(route_file, label="AgileX route manifest")
    )
    if route["contract_sha256"] != digests["route"]:
        raise ValueError("AgileX route manifest semantic SHA-256 differs")
    _route_modalities(route, profile)
    normalizer_contract = load_agilex_normalizer(
        normalizer_path,
        expected_file_sha256=digests["normalizer_file"],
        expected_source_manifest_sha256=digests["source"],
        expected_repo_route_manifest_sha256=digests["route"],
    )
    if normalizer_contract.contract_sha256 != digests["normalizer_contract"]:
        raise ValueError("AgileX normalizer semantic SHA-256 differs")
    snapshot = capture_strict_checkpoint_snapshot(checkpoint_root)
    identity = build_strict_checkpoint_identity(snapshot)
    if identity["identity_sha256"] != digests["checkpoint"]:
        raise ValueError("AgileX checkpoint identity differs from the request")
    _validate_checkpoint(snapshot, profile=profile, digests=digests)
    encoder = validate_encoder_source_identity(build_encoder_source_identity(base_root))
    return (
        checkpoint_root,
        base_root,
        identity,
        encoder,
        component_sources(checkpoint_root, base_root),
        route,
    )


__all__ = (
    "COMPONENTS",
    "DIGEST_FIELDS",
    "RECEIPT_FIELDS",
    "RECEIPT_TO_DIGEST",
    "audit_sources",
    "canonical_sha256",
    "digest",
    "directory",
    "file",
    "json_object",
    "validate_safety_contract",
    "validate_task_routes",
)
