# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed input audit for one AgileX post-training request."""

from __future__ import annotations

import json
import os
import socket
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
)
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.checkpointing.training_lineage import (
    validate_action_migration_report_for_contract,
)
from n0_twam.configs.twam_track3_agilex_contracts import (
    AgileXRepoRouteBinding,
    AgileXTemporalBinding,
    load_repo_route_binding,
    load_temporal_binding,
)
from n0_twam.configs.twam_track3_agilex_recipe import (
    build_agilex_resume_recipe_contract,
    validate_agilex_resume_recipe_contract,
)
from n0_twam.embodiments import AGILEX_ACTION_SCHEMA
from n0_twam.integrations.worldarena.agilex_artifacts import (
    verify_agilex_conversion_receipt,
    verify_agilex_latent_inventory,
)
from n0_twam.integrations.worldarena.agilex_manifest import (
    AgileXDatasetManifest,
    load_agilex_manifest,
    sha256_file,
)
from n0_twam.integrations.worldarena.agilex_normalizer import (
    load_agilex_normalizer,
)

from .request import (
    AgileXTrainRequest,
    load_agilex_train_request,
    require_agilex_request_unchanged,
)

AGILEX_PROFILE_PREFIX = "agilex_track3"


def agilex_profile_id(profile: str) -> str:
    return f"{AGILEX_PROFILE_PREFIX}_{profile}_v1"


def build_artifact_identity(request: AgileXTrainRequest) -> dict[str, object]:
    paths = request.paths
    return {
        "schema_version": 1,
        "embodiment_profile_id": "agilex_dual_qpos14_v1",
        "action_schema": AGILEX_ACTION_SCHEMA,
        "tactile_profile": request.profile,
        "source_manifest_file_sha256": paths.source_manifest_sha256,
        "conversion_receipt_file_sha256": paths.conversion_receipt_sha256,
        "latent_inventory_file_sha256": paths.latent_inventory_sha256,
        "repo_route_manifest_file_sha256": paths.repo_route_manifest_sha256,
        "temporal_alignment_file_sha256": paths.temporal_alignment_sha256,
        "normalizer_file_sha256": paths.normalizer_sha256,
    }


def _regular(path: Path, *, label: str, directory: bool = False) -> Path:
    candidate = Path(path).expanduser()
    metadata = candidate.lstat()
    expected = (
        stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    )
    if stat.S_ISLNK(metadata.st_mode) or not expected:
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} must be a regular non-symlink {kind}")
    return candidate.resolve(strict=True)


def _hash(path: Path, expected: str, *, label: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 differs from the request")


def _json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _validate_profile(routes: Mapping[str, Mapping[str, object]], profile: str) -> None:
    tactile = [bool(route.get("tactile_keys")) for route in routes.values()]
    wrench = [bool(route.get("wrench_keys")) for route in routes.values()]
    if profile == "vision_tactile" and (
        not tactile or not all(tactile) or not all(wrench)
    ):
        raise ValueError("vision_tactile requires tactile and wrench for every repo")
    if profile == "mixed":
        if not any(tactile) or not any(not present for present in tactile):
            raise ValueError("mixed requires tactile and vision-only repositories")
        if any(
            not touch and force for touch, force in zip(tactile, wrench, strict=True)
        ):
            raise ValueError("mixed vision-only repositories may not expose wrench")
    if profile == "vision_only" and (any(tactile) or any(wrench)):
        raise ValueError("vision_only repositories may not expose contact inputs")


def _validate_route_stack(request: AgileXTrainRequest) -> dict[str, object]:
    paths = request.paths
    route_file = _regular(paths.repo_route_manifest, label="repo route manifest")
    temporal_file = _regular(paths.temporal_alignment, label="temporal alignment")
    normalizer_file = _regular(paths.normalizer, label="qpos14 normalizer")
    _hash(route_file, paths.repo_route_manifest_sha256, label="repo route manifest")
    _hash(temporal_file, paths.temporal_alignment_sha256, label="temporal alignment")
    _hash(normalizer_file, paths.normalizer_sha256, label="qpos14 normalizer")
    route_binding = load_repo_route_binding(route_file)
    if route_binding.is_placeholder:
        raise ValueError("formal AgileX training cannot use placeholder routes")
    routes = route_binding.routes
    repo_ids = tuple(sorted(routes))
    _validate_profile(routes, request.profile)
    temporal = load_temporal_binding(
        temporal_file,
        repo_names=repo_ids,
        repo_route_manifest_sha256=route_binding.manifest.contract_sha256,
    )
    if temporal.is_placeholder or temporal.action_per_frame is None:
        raise ValueError("formal AgileX training requires a complete temporal binding")
    normalizer = load_agilex_normalizer(
        normalizer_file,
        expected_file_sha256=paths.normalizer_sha256,
        expected_source_manifest_sha256=paths.source_manifest_sha256,
        expected_repo_route_manifest_sha256=route_binding.manifest.contract_sha256,
    )
    return {
        "binding": route_binding,
        "routes": routes,
        "repo_ids": repo_ids,
        "temporal": temporal,
        "normalizer": normalizer,
    }


def _validate_source(
    request: AgileXTrainRequest, route_stack: Mapping[str, object]
) -> AgileXDatasetManifest:
    paths = request.paths
    source_root = _regular(paths.source_root, label="source root", directory=True)
    source_file = _regular(paths.source_manifest, label="source manifest")
    manifest = load_agilex_manifest(
        source_file,
        expected_file_sha256=paths.source_manifest_sha256,
        selected_repo_ids=cast(tuple[str, ...], route_stack["repo_ids"]),
    )
    for record in manifest.records:
        record.verify(source_root)
    routes = route_stack["routes"]
    temporal = cast(AgileXTemporalBinding, route_stack["temporal"])
    assert isinstance(routes, Mapping)
    temporal_route_ids = dict(temporal.route_identities)
    temporal_ids = dict(temporal.temporal_identities)
    for source_route in manifest.routes:
        configured = routes[source_route.repo_id]
        expected = {
            "embodiment": source_route.embodiment,
            "action_schema": source_route.action_schema,
            "rgb_keys": list(source_route.rgb_keys),
            "tactile_keys": list(source_route.tactile_keys),
            "wrench_keys": list(source_route.wrench_keys),
        }
        if configured != expected:
            raise ValueError(f"source/config route mismatch: {source_route.repo_id}")
        if temporal_route_ids[source_route.repo_id] != source_route.route_identity:
            raise ValueError("temporal binding repo-route identity mismatch")
        if (
            temporal_ids[source_route.repo_id]
            != source_route.temporal_alignment_identity
        ):
            raise ValueError("temporal alignment identity mismatch")
    return manifest


def _validate_initialization(request: AgileXTrainRequest) -> dict[str, object]:
    paths = request.paths
    if paths.init_from is not None:
        root = _regular(paths.init_from, label="initial checkpoint", directory=True)
        config = _json(root / "transformer" / "config.json", label="init config")
        action_dim = config.get("action_dim")
        action_schema = config.get("action_schema")
        if action_dim == 14 and action_schema == AGILEX_ACTION_SCHEMA:
            return validate_qpos14_stage_b_checkpoint(
                root,
                expected_transformer_sha256=paths.init_transformer_sha256,
                expected_profile_id=agilex_profile_id(request.profile),
                expected_run_role=request.train.run_role,
                expected_artifact_identity=build_artifact_identity(request),
            )
        if action_dim != 20 or action_schema not in (None, "ee20_pi05"):
            raise ValueError(
                "initial checkpoint is not a compatible 20D or qpos14 model"
            )
        identity = audit_transformer_checkpoint(
            root / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
            expected_action_dim=20,
        )
        if identity["sha256"] != paths.init_transformer_sha256:
            raise ValueError("initial 20D transformer differs from the request")
        return {"mode": "init20_migrate_qpos14", "transformer_identity": identity}
    assert paths.resume_from is not None
    snapshot = capture_strict_checkpoint_snapshot(paths.resume_from)
    identity = build_strict_checkpoint_identity(snapshot)
    if identity["identity_sha256"] != paths.resume_checkpoint_identity_sha256:
        raise ValueError("resume qpos14 checkpoint identity differs from request")
    if snapshot.world_size != len(request.runtime.devices):
        raise ValueError("resume checkpoint world_size differs from request")
    if snapshot.transformer_config.get("action_schema") != AGILEX_ACTION_SCHEMA:
        raise ValueError("resume checkpoint is not qpos14")
    expected_profile = agilex_profile_id(request.profile)
    if snapshot.train_meta.get("track32_profile_id") != expected_profile:
        raise ValueError("resume checkpoint AgileX profile differs from request")
    if snapshot.train_meta.get("run_role") != request.train.run_role:
        raise ValueError("resume checkpoint run_role differs from request")
    expected_artifacts = build_artifact_identity(request)
    sidecars = (
        ("train metadata", snapshot.train_meta),
        ("training state", snapshot.training_state),
        ("completion marker", snapshot.completion),
    )
    for label, payload in sidecars:
        if payload.get("track32_artifact_identity") != expected_artifacts:
            raise ValueError("resume checkpoint artifact identity differs from request")
        lineage = payload.get("training_lineage")
        if not isinstance(lineage, Mapping):
            raise ValueError(f"resume checkpoint {label} has no training lineage")
        try:
            validate_agilex_resume_recipe_contract(
                lineage.get("resume_recipe_contract"),
                current=request.train,
            )
        except ValueError as error:
            raise ValueError(
                f"resume checkpoint {label} recipe differs from request"
            ) from error
    return {"mode": "strict_resume_qpos14", "checkpoint_identity": identity}


def validate_qpos14_stage_b_checkpoint(
    checkpoint_root: Path,
    *,
    expected_transformer_sha256: str | None,
    expected_profile_id: str,
    expected_run_role: str,
    expected_artifact_identity: Mapping[str, object],
    audited_transformer_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate one complete qpos14 parent for weights-only Stage-B init."""

    if expected_transformer_sha256 is None:
        raise ValueError("qpos14 Stage-B transformer SHA-256 is missing")
    snapshot = capture_strict_checkpoint_snapshot(checkpoint_root)
    checkpoint_identity = build_strict_checkpoint_identity(snapshot)
    if (
        snapshot.transformer_config.get("action_dim") != 14
        or snapshot.transformer_config.get("action_schema") != AGILEX_ACTION_SCHEMA
    ):
        raise ValueError("Stage-B parent transformer is not qpos14")
    transformer_identity = dict(
        audited_transformer_identity
        if audited_transformer_identity is not None
        else audit_transformer_checkpoint(
            checkpoint_root / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
            expected_action_dim=14,
        )
    )
    if transformer_identity["sha256"] != expected_transformer_sha256:
        raise ValueError("initial qpos14 transformer differs from the request")
    if snapshot.train_meta.get("track32_profile_id") != expected_profile_id:
        raise ValueError("Stage-B parent AgileX profile differs from request")
    if snapshot.train_meta.get("run_role") != expected_run_role:
        raise ValueError("Stage-B parent run_role differs from request")
    if snapshot.action_migration_report is None:
        raise ValueError("Stage-B parent has no action migration report")
    validate_action_migration_report_for_contract(
        snapshot.action_migration_report,
        source_action_dim=20,
        source_action_schema="ee20_pi05",
        target_action_dim=14,
        target_action_schema=AGILEX_ACTION_SCHEMA,
        initialized_target_only_prefixes=("local_tactile_", "agilex_wrench_"),
    )
    for label, payload in (
        ("train metadata", snapshot.train_meta),
        ("training state", snapshot.training_state),
        ("completion marker", snapshot.completion),
    ):
        if payload.get("track32_artifact_identity") != expected_artifact_identity:
            raise ValueError(
                f"Stage-B parent {label} artifact identity differs from request"
            )
    return {
        "mode": "init14_weights_only_stage_b",
        "transformer_identity": transformer_identity,
        "checkpoint_identity": checkpoint_identity,
    }


def _validate_accelerator(request: AgileXTrainRequest) -> None:
    interface = request.runtime.collective_network_interface
    if request.runtime.accelerator_profile == "portable":
        if interface is not None:
            raise ValueError("portable runtime cannot bind a collective interface")
        return
    bindings = {
        "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
        "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME"),
    }
    if interface is None or any(value != interface for value in bindings.values()):
        raise ValueError(f"collective socket interface mismatch: {bindings}")
    if interface not in {name for _, name in socket.if_nameindex()}:
        raise ValueError(f"collective interface is unavailable: {interface}")


def audit_agilex_inputs(request: AgileXTrainRequest) -> dict[str, object]:
    """Audit immutable inputs and return identities consumed by training."""

    if request.profile == "mixed" and request.train.batch_size != 1:
        raise ValueError("mixed AgileX training requires batch_size=1")
    paths = request.paths
    if paths.output_root.exists():
        raise FileExistsError(f"AgileX output_root must be new: {paths.output_root}")
    artifact_root = _regular(paths.artifact_root, label="artifact root", directory=True)
    dataset_root = _regular(paths.dataset_root, label="dataset root", directory=True)
    _regular(paths.base_model, label="base model", directory=True)
    empty = _regular(paths.empty_embedding, label="empty embedding")
    _hash(empty, paths.empty_embedding_sha256, label="empty embedding")
    route_stack = _validate_route_stack(request)
    manifest = _validate_source(request, route_stack)
    repo_ids = route_stack["repo_ids"]
    assert isinstance(repo_ids, tuple)
    binding = cast(AgileXRepoRouteBinding, route_stack["binding"])
    temporal = cast(AgileXTemporalBinding, route_stack["temporal"])
    conversion = _regular(paths.conversion_receipt, label="conversion receipt")
    latent = _regular(paths.latent_inventory, label="latent inventory")
    _hash(conversion, paths.conversion_receipt_sha256, label="conversion receipt")
    _hash(latent, paths.latent_inventory_sha256, label="latent inventory")
    conversion_payload = _json(conversion, label="conversion receipt")
    verified_conversion = verify_agilex_conversion_receipt(
        conversion_payload,
        dataset_root=dataset_root,
        routes=manifest.routes,
        source_manifest_sha256=paths.source_manifest_sha256,
        repo_route_manifest_sha256=binding.manifest.contract_sha256,
        temporal_alignment_contract_sha256=temporal.contract_sha256,
    )
    latent_payload = _json(latent, label="latent inventory")
    verified_latents = verify_agilex_latent_inventory(
        latent_payload,
        dataset_root=dataset_root,
        routes=manifest.routes,
        conversion_receipt=conversion_payload,
        conversion_receipt_sha256=paths.conversion_receipt_sha256,
    )
    if any(
        artifact_root not in path.parents and path != artifact_root
        for path in (
            paths.source_manifest,
            conversion,
            latent,
            paths.repo_route_manifest,
            paths.temporal_alignment,
            paths.normalizer,
        )
    ):
        raise ValueError("AgileX provenance files must be inside artifact_root")
    _validate_accelerator(request)
    return {
        "artifact_identity": build_artifact_identity(request),
        "data_artifacts": {
            "conversion_identity_sha256": (
                verified_conversion.conversion_identity_sha256
            ),
            "latent_inventory_sha256": verified_latents.inventory_sha256,
            "latent_record_count": verified_latents.record_count,
        },
        "training_lineage": {
            "repo_route_manifest_sha256": binding.manifest.contract_sha256,
            "repo_route_manifest_source_file_sha256": paths.repo_route_manifest_sha256,
            "temporal_alignment_contract_sha256": temporal.contract_sha256,
            "temporal_alignment_source_file_sha256": paths.temporal_alignment_sha256,
            "normalizer_sha256": paths.normalizer_sha256,
            "action_schema": AGILEX_ACTION_SCHEMA,
            "tactile_profile": request.profile,
            "resume_recipe_contract": build_agilex_resume_recipe_contract(
                request.train
            ),
        },
        "checkpoint": _validate_initialization(request),
    }


def run_preflight() -> dict[str, object]:
    request_path = os.environ.get("N0_TRACK3_AGILEX_REQUEST")
    if not request_path:
        raise ValueError(
            "required environment variable is unset: N0_TRACK3_AGILEX_REQUEST"
        )
    request = load_agilex_train_request(Path(request_path))
    expected_request_sha256 = os.environ.get("N0_TRACK3_AGILEX_REQUEST_SHA256")
    if request.source_sha256 != expected_request_sha256:
        raise ValueError("AgileX request SHA-256 environment mismatch")
    audit = audit_agilex_inputs(request)
    require_agilex_request_unchanged(request)
    expected_identity = json.dumps(
        audit["artifact_identity"], sort_keys=True, separators=(",", ":")
    )
    if os.environ.get("N0_TRACK3_AGILEX_ARTIFACT_IDENTITY_JSON") != expected_identity:
        raise ValueError("AgileX artifact identity environment mismatch")
    return {
        "schema_version": 1,
        "status": "pass",
        "request_sha256": request.source_sha256,
        "profile": agilex_profile_id(request.profile),
        "world_size": len(request.runtime.devices),
        **audit,
    }


def main() -> int:
    print(json.dumps(run_preflight(), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
