# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Canonical stable-byte snapshot for one complete strict checkpoint."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from n0_twam.tactile_profiles import validate_tactile_profile_contract
from n0_twam.distributed.optimizer_checkpoint import (
    OPTIMIZER_STATE_FORMAT,
    STRICT_CHECKPOINT_SCHEMA_VERSION,
)

from .runtime_provenance import validate_checkpoint_runtime_provenance
from .sidecar_snapshot import (
    SidecarSnapshot,
    StableFileSnapshot,
    capture_sidecar_snapshot,
    capture_stable_json_file,
    strict_integer,
)
from .strict_resume import expected_sidecar_paths

_FORMAL_TRACK31_PROFILE_IDS = frozenset(("multitask_pretrain_v1", "target_finetune_v1"))


def _is_formal_track31_payload(payload: Mapping[str, object]) -> bool:
    profile_identity = payload.get("training_profile_identity")
    return (
        payload.get("training_profile_id") in _FORMAL_TRACK31_PROFILE_IDS
        or (
            isinstance(profile_identity, Mapping)
            and profile_identity.get("training_profile_id")
            in _FORMAL_TRACK31_PROFILE_IDS
        )
        or payload.get("track31_artifacts") is not None
    )


@dataclass(frozen=True)
class StrictCheckpointSnapshot:
    checkpoint_root: Path
    completion_file: StableFileSnapshot
    sidecars: SidecarSnapshot
    completion: dict[str, object]
    training_state: dict[str, object]
    train_meta: dict[str, object]
    transformer_config: dict[str, object]
    action_migration_report: dict[str, object] | None
    runtime_source_identity: dict[str, object] | None
    checkpoint_invocation_identity: dict[str, object] | None
    step: int
    world_size: int
    data_batches_consumed: int
    gradient_accumulation_steps: int


def build_strict_checkpoint_identity(
    snapshot: StrictCheckpointSnapshot,
) -> dict[str, object]:
    """Build a portable, hash-bound identity for a captured strict checkpoint."""

    core: dict[str, object] = {
        "schema_version": 1,
        "completion_file": {
            "relative_path": snapshot.completion_file.relative_path,
            "size_bytes": snapshot.completion_file.size_bytes,
            "sha256": snapshot.completion_file.sha256,
        },
        "sidecar_inventory": snapshot.sidecars.inventory,
        "runtime_source_identity": snapshot.runtime_source_identity,
        "checkpoint_invocation_identity": snapshot.checkpoint_invocation_identity,
        "step": snapshot.step,
        "world_size": snapshot.world_size,
        "data_batches_consumed": snapshot.data_batches_consumed,
        "gradient_accumulation_steps": snapshot.gradient_accumulation_steps,
    }
    canonical = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        **core,
        "identity_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def capture_strict_checkpoint_snapshot(
    checkpoint_dir: Path,
) -> StrictCheckpointSnapshot:
    """Capture and parse completion plus every hash-bound sidecar exactly once."""

    root = Path(checkpoint_dir).resolve(strict=True)
    completion_file = capture_stable_json_file(
        root / "checkpoint_complete.json",
        label="checkpoint completion marker",
    )
    completion = completion_file.json_object(label="checkpoint completion marker")
    completion_schema = strict_integer(
        completion.get("schema_version"),
        label="completion schema_version",
        minimum=1,
    )
    completion_step = strict_integer(
        completion.get("step"),
        label="completion step",
    )
    completion_world = strict_integer(
        completion.get("world_size"),
        label="completion world_size",
        minimum=1,
    )
    action_schema = completion.get("action_schema")
    if not isinstance(action_schema, str) or not action_schema:
        raise ValueError("strict checkpoint action schema is invalid")
    if (
        completion_schema != STRICT_CHECKPOINT_SCHEMA_VERSION
        or completion.get("status") != "complete"
        or completion.get("optimizer_state_format") != OPTIMIZER_STATE_FORMAT
    ):
        raise ValueError("strict checkpoint completion marker is incompatible")
    raw_inventory = completion.get("sidecar_inventory")
    raw_files = (
        raw_inventory.get("files") if isinstance(raw_inventory, Mapping) else None
    )
    if not isinstance(raw_files, list):
        raise ValueError("strict checkpoint sidecar inventory is invalid")
    inventory_paths = {
        entry.get("path") for entry in raw_files if isinstance(entry, Mapping)
    }
    include_action_migration = "action_migration_report.json" in inventory_paths
    sidecars = capture_sidecar_snapshot(
        root,
        completion.get("sidecar_inventory"),
        expected_sidecar_paths(
            completion_world,
            include_transformer_config=True,
            include_action_migration=include_action_migration,
        ),
    )
    training_state = sidecars.json_object(
        "training_state.json",
        label="checkpoint training state",
    )
    state_schema = strict_integer(
        training_state.get("schema_version"),
        label="training state schema_version",
        minimum=1,
    )
    state_step = strict_integer(
        training_state.get("step"),
        label="training state step",
    )
    state_world = strict_integer(
        training_state.get("world_size"),
        label="training state world_size",
        minimum=1,
    )
    data_batches = strict_integer(
        training_state.get("data_batches_consumed"),
        label="training state data_batches_consumed",
    )
    accumulation = strict_integer(
        training_state.get("gradient_accumulation_steps"),
        label="training state gradient_accumulation_steps",
        minimum=1,
    )
    if (
        state_schema != STRICT_CHECKPOINT_SCHEMA_VERSION
        or state_step != completion_step
        or state_world != completion_world
        or training_state.get("action_schema") != action_schema
        or training_state.get("optimizer_state_format") != OPTIMIZER_STATE_FORMAT
    ):
        raise ValueError("strict checkpoint training state is incompatible")
    if data_batches != state_step * accumulation:
        raise ValueError("strict checkpoint progress counters are inconsistent")
    train_meta = sidecars.json_object(
        "train_meta.json",
        label="checkpoint train metadata",
    )
    if train_meta.get("action_schema") != action_schema:
        raise ValueError("strict checkpoint train metadata action schema differs")
    transformer_config = sidecars.json_object(
        "transformer/config.json",
        label="checkpoint transformer config",
    )
    if transformer_config.get("action_schema") != action_schema:
        raise ValueError("strict checkpoint transformer action schema differs")
    tactile_profiles = (
        train_meta.get("tactile_profile_contract"),
        training_state.get("tactile_profile_contract"),
        completion.get("tactile_profile_contract"),
    )
    if any(value is not None for value in tactile_profiles):
        if any(value is None for value in tactile_profiles):
            raise ValueError("strict checkpoint tactile profile is incomplete")
        canonical_profiles = tuple(
            validate_tactile_profile_contract(value) for value in tactile_profiles
        )
        serialized = {json.dumps(value, sort_keys=True) for value in canonical_profiles}
        if len(serialized) != 1:
            raise ValueError("strict checkpoint tactile profile sidecars differ")
        profile = canonical_profiles[0]
        if (
            transformer_config.get("tactile_profile") != profile["profile"]
            or transformer_config.get("tactile_profile_contract_sha256")
            != profile["contract_sha256"]
        ):
            raise ValueError("strict checkpoint transformer tactile profile differs")
    provenance_payloads = (
        ("checkpoint train metadata", train_meta),
        ("checkpoint training state", training_state),
        ("checkpoint completion marker", completion),
    )
    is_formal_track31 = any(
        _is_formal_track31_payload(payload) for _, payload in provenance_payloads
    )
    has_runtime_provenance = any(
        "runtime_source_identity" in payload
        or "checkpoint_invocation_identity" in payload
        for _, payload in provenance_payloads
    )
    runtime_source_identity = None
    checkpoint_invocation_identity = None
    if is_formal_track31 or has_runtime_provenance:
        runtime_source_identity, checkpoint_invocation_identity = (
            validate_checkpoint_runtime_provenance(provenance_payloads)
        )
    return StrictCheckpointSnapshot(
        checkpoint_root=root,
        completion_file=completion_file,
        sidecars=sidecars,
        completion=completion,
        training_state=training_state,
        train_meta=train_meta,
        transformer_config=transformer_config,
        action_migration_report=(
            sidecars.json_object(
                "action_migration_report.json",
                label="checkpoint action migration report",
            )
            if include_action_migration
            else None
        ),
        runtime_source_identity=runtime_source_identity,
        checkpoint_invocation_identity=checkpoint_invocation_identity,
        step=state_step,
        world_size=state_world,
        data_batches_consumed=data_batches,
        gradient_accumulation_steps=accumulation,
    )


__all__ = (
    "StrictCheckpointSnapshot",
    "build_strict_checkpoint_identity",
    "capture_strict_checkpoint_snapshot",
)
