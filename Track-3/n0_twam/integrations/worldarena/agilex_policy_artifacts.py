# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed artifact verification for local AgileX policy serving."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.data.encoder_source_identity import (
    build_encoder_source_identity,
    validate_encoder_source_identity,
)
from n0_twam.embodiments import (
    AGILEX_ACTION_SCHEMA,
    validate_repo_route_manifest_contract,
)

from .agilex_manifest import canonical_sha256, sha256_file
from .agilex_normalizer import load_agilex_normalizer
from .agilex_policy_io import AgileXDirectPolicyConfig

IdentityBuilder = Callable[[Path], Mapping[str, object]]
_MODEL_COMPONENTS = ("transformer", "vae", "tokenizer", "text_encoder")


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _strict_checkpoint_identity(root: Path) -> Mapping[str, object]:
    return cast(
        Mapping[str, object],
        build_strict_checkpoint_identity(capture_strict_checkpoint_snapshot(root)),
    )


def _encoder_source_identity(root: Path) -> Mapping[str, object]:
    return cast(
        Mapping[str, object],
        validate_encoder_source_identity(build_encoder_source_identity(root)),
    )


def _canonical_source(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an absolute canonical path")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{label} must be an absolute canonical path")
    try:
        resolved = raw.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} cannot be resolved") from error
    if str(resolved) != value or not resolved.is_dir():
        raise ValueError(f"{label} must be an existing canonical directory")
    return resolved


def _expected_receipt(
    config: AgileXDirectPolicyConfig,
    *,
    checkpoint_root: Path,
    checkpoint_identity: Mapping[str, object],
    base_model_root: Path,
    encoder_source_identity: Mapping[str, object],
    component_names: list[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete",
        "action_schema": AGILEX_ACTION_SCHEMA,
        "tactile_profile": config.policy.tactile_profile,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_identity": dict(checkpoint_identity),
        "checkpoint_identity_sha256": config.checkpoint_identity_sha256,
        "base_model_root": str(base_model_root),
        "encoder_source_identity": dict(encoder_source_identity),
        "normalizer_file_sha256": config.normalizer_file_sha256,
        "normalizer_contract_sha256": config.normalizer_contract_sha256,
        "source_manifest_sha256": config.source_manifest_sha256,
        "repo_route_manifest_file_sha256": config.repo_route_manifest_file_sha256,
        "repo_route_manifest_sha256": config.repo_route_manifest_sha256,
        "contact_profile_contract_sha256": (config.contact_profile_contract_sha256),
        "task_routes_sha256": config.task_routes_sha256,
        "safety_contract_sha256": config.policy.safety.contract_sha256,
        "component_names": component_names,
    }


def _verify_link(link: Path, expected: Path, *, label: str) -> None:
    if not link.is_symlink():
        raise ValueError(f"AgileX serve component must be a symlink: {label}")
    try:
        actual = link.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"AgileX serve component link is invalid: {label}") from error
    if actual != expected or not actual.is_dir():
        raise ValueError(f"AgileX serve component link changed: {label}")


def _validate_task_routes_against_training_route(
    config: AgileXDirectPolicyConfig,
    route: Mapping[str, object],
) -> None:
    tactile_keys = frozenset(cast(list[str], route["global_tactile_keys"]))
    wrench_keys = frozenset(cast(list[str], route["global_wrench_keys"]))
    tactile_map = cast(Mapping[str, object], route["tactile_sensor_id_map"])
    wrench_map = cast(Mapping[str, object], route["wrench_sensor_id_map"])
    for task_id, task in config.policy.task_routes.items():
        task_tactile = frozenset(task.tactile_keys)
        task_wrench = frozenset(task.wrench_keys)
        if not task_tactile <= tactile_keys or not task_tactile <= frozenset(
            tactile_map
        ):
            raise ValueError(f"task {task_id!r} tactile route was not trained")
        if not task_wrench <= wrench_keys or not task_wrench <= frozenset(wrench_map):
            raise ValueError(f"task {task_id!r} wrench route was not trained")


def verify_agilex_policy_artifacts(
    config: AgileXDirectPolicyConfig,
    *,
    checkpoint_identity_builder: IdentityBuilder = _strict_checkpoint_identity,
    encoder_identity_builder: IdentityBuilder = _encoder_source_identity,
) -> Mapping[str, object]:
    """Recompute the complete checkpoint, encoder, route, and policy bindings."""

    if (
        config.source_path.is_symlink()
        or not config.source_path.is_file()
        or sha256_file(config.source_path) != config.policy_config_file_sha256
    ):
        raise ValueError("AgileX policy config file identity mismatch")
    receipt_path = config.serve_bundle / "serve_bundle_receipt.json"
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or sha256_file(receipt_path) != config.serve_bundle_receipt_sha256
    ):
        raise ValueError("AgileX serve bundle receipt identity mismatch")
    receipt = _json_object(receipt_path, label="AgileX serve bundle receipt")

    # The production policy loader never injects identity builders.  Reuse the
    # bundle verifier in that path so a hand-written receipt cannot bypass the
    # qpos14/profile/migration and actual-transformer audits performed at seal
    # time.  Injectable builders remain available only for small unit fixtures.
    if (
        checkpoint_identity_builder is _strict_checkpoint_identity
        and encoder_identity_builder is _encoder_source_identity
    ):
        from .agilex_serve_bundle import verify_agilex_serve_bundle

        sealed = verify_agilex_serve_bundle(
            config.serve_bundle,
            expected_receipt_file_sha256=config.serve_bundle_receipt_sha256,
        )
        if sealed != receipt:
            raise ValueError("AgileX serve bundle audit differs from its receipt")

    checkpoint_root = _canonical_source(
        receipt.get("checkpoint_root"), label="checkpoint_root"
    )
    checkpoint_identity = dict(checkpoint_identity_builder(checkpoint_root))
    if (
        checkpoint_identity != receipt.get("checkpoint_identity")
        or checkpoint_identity.get("identity_sha256")
        != config.checkpoint_identity_sha256
    ):
        raise ValueError("AgileX strict checkpoint identity mismatch")
    base_model_root = _canonical_source(
        receipt.get("base_model_root"), label="base_model_root"
    )
    encoder_identity = dict(encoder_identity_builder(base_model_root))
    encoder_identity = validate_encoder_source_identity(encoder_identity)
    if encoder_identity != receipt.get("encoder_source_identity"):
        raise ValueError("AgileX encoder source identity mismatch")

    component_names = receipt.get("component_names")
    allowed = (list(_MODEL_COMPONENTS), [*_MODEL_COMPONENTS, "assets"])
    if component_names not in allowed:
        raise ValueError("AgileX serve bundle component contract mismatch")
    expected = _expected_receipt(
        config,
        checkpoint_root=checkpoint_root,
        checkpoint_identity=checkpoint_identity,
        base_model_root=base_model_root,
        encoder_source_identity=encoder_identity,
        component_names=list(component_names),
    )
    if set(receipt) != {*expected, "bundle_identity_sha256"}:
        raise ValueError("AgileX serve bundle receipt field contract mismatch")
    core = {
        key: value for key, value in receipt.items() if key != "bundle_identity_sha256"
    }
    if core != expected or receipt["bundle_identity_sha256"] != canonical_sha256(core):
        raise ValueError("AgileX serve bundle receipt contract mismatch")
    if receipt["bundle_identity_sha256"] != config.serve_bundle_identity_sha256:
        raise ValueError("AgileX serve bundle semantic identity mismatch")

    targets = {
        "transformer": checkpoint_root / "transformer",
        "vae": base_model_root / "vae",
        "tokenizer": base_model_root / "tokenizer",
        "text_encoder": base_model_root / "text_encoder",
    }
    for name, target in targets.items():
        _verify_link(config.serve_bundle / name, target, label=name)
    assets_link = config.serve_bundle / "assets"
    if "assets" in component_names:
        _verify_link(assets_link, base_model_root / "assets", label="assets")
    elif assets_link.exists() or assets_link.is_symlink():
        raise ValueError("AgileX serve bundle contains undeclared assets")

    route_path = config.serve_bundle / "repo_route_manifest.json"
    if (
        route_path.is_symlink()
        or not route_path.is_file()
        or sha256_file(route_path) != config.repo_route_manifest_file_sha256
    ):
        raise ValueError("AgileX route manifest file identity mismatch")
    route = validate_repo_route_manifest_contract(
        _json_object(route_path, label="AgileX route manifest")
    )
    if route["contract_sha256"] != config.repo_route_manifest_sha256:
        raise ValueError("AgileX route manifest contract identity mismatch")
    _validate_task_routes_against_training_route(config, route)
    normalizer = load_agilex_normalizer(
        config.serve_bundle / "normalizer.json",
        expected_file_sha256=config.normalizer_file_sha256,
        expected_source_manifest_sha256=config.source_manifest_sha256,
        expected_repo_route_manifest_sha256=config.repo_route_manifest_sha256,
    )
    if normalizer != config.normalizer:
        raise ValueError("AgileX normalizer changed after policy config load")
    return receipt


__all__ = ("verify_agilex_policy_artifacts",)
