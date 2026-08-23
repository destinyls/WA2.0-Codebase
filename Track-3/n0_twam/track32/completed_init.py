# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fast receipt-bound validation for a completed weights-only checkpoint init."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    validate_recorded_transformer_identity,
    validate_sha256,
)
from n0_twam.integrations.worldarena.franka_actions import (
    DERIVED_ACTION_SCHEMA,
    FRANKA_ACTION_SCHEMA,
)
from n0_twam.integrations.worldarena.franka_artifacts import MODEL_ACTION_SCHEMA
from n0_twam.integrations.worldarena.franka_manifest import sha256_file

_CHECKPOINT_NAME = re.compile(r"^checkpoint_step_(\d+)$")


@dataclass(frozen=True)
class CompletedWeightsInit:
    """Validated identity needed for weights-only initialization."""

    checkpoint_root: Path
    completion_sha256: str
    transformer_identity: dict[str, object]
    step: int
    world_size: int


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"missing {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if metadata.st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")
    return metadata


def validate_completed_weights_init(
    checkpoint_root: Path,
    *,
    expected_completion_sha256: str | None = None,
) -> CompletedWeightsInit:
    """Validate a completed checkpoint without rehashing its 14 GB weights.

    The small immutable completion marker already records the full transformer
    byte identity.  This path binds that marker, its config sidecar, the on-disk
    weight size, and the Franka XYZW/vision-only contract.  Safetensors parsing
    still happens naturally when the model is loaded.
    """

    source = Path(checkpoint_root).expanduser()
    if source.is_symlink():
        raise ValueError("completed init checkpoint may not be a symlink")
    root = source.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("completed init checkpoint must be a directory")

    name_match = _CHECKPOINT_NAME.fullmatch(root.name)
    if name_match is None:
        raise ValueError("completed init checkpoint name must encode its step")

    completion_path = root / "checkpoint_complete.json"
    _regular_file(completion_path, label="checkpoint completion marker")
    completion_sha256 = sha256_file(completion_path)
    if expected_completion_sha256 is not None and completion_sha256 != validate_sha256(
        expected_completion_sha256,
        label="checkpoint completion SHA256",
    ):
        raise ValueError("checkpoint completion marker differs from the request")
    completion = _json_object(
        completion_path,
        label="checkpoint completion marker",
    )

    step = _positive_integer(completion.get("step"), label="checkpoint step")
    world_size = _positive_integer(
        completion.get("world_size"),
        label="checkpoint world_size",
    )
    if (
        completion.get("status") != "complete"
        or step != int(name_match.group(1))
        or completion.get("action_schema") != MODEL_ACTION_SCHEMA
    ):
        raise ValueError("checkpoint completion contract is invalid")

    lineage = completion.get("training_lineage")
    expected_lineage = {
        "source_action_schema": FRANKA_ACTION_SCHEMA,
        "derived_action_schema": DERIVED_ACTION_SCHEMA,
        "target_action_schema": MODEL_ACTION_SCHEMA,
        "tactile_mode": "disabled",
        "tactile_profile": "vision_only",
    }
    if not isinstance(lineage, dict) or any(
        lineage.get(field) != expected
        for field, expected in expected_lineage.items()
    ):
        raise ValueError("checkpoint Franka XYZW lineage is invalid")

    tactile = completion.get("tactile_profile_contract")
    if (
        not isinstance(tactile, dict)
        or tactile.get("profile") != "vision_only"
        or tactile.get("tactile_mode") != "disabled"
        or tactile.get("freeze_tactile_parameters") is not True
    ):
        raise ValueError("checkpoint vision-only tactile contract is invalid")

    transformer_identity = validate_recorded_transformer_identity(
        completion.get("transformer_identity"),
        expected_action_dim=20,
    )
    transformer_root = root / "transformer"
    weights_path = transformer_root / TRANSFORMER_WEIGHTS_FILENAME
    weights_stat = _regular_file(weights_path, label="transformer weights")
    if weights_stat.st_size != transformer_identity["size_bytes"]:
        raise ValueError("transformer weight size differs from completion receipt")

    config_path = transformer_root / "config.json"
    config_stat = _regular_file(config_path, label="transformer config")
    config = _json_object(config_path, label="transformer config")
    if config.get("action_dim") != 20 or config.get("action_schema") != MODEL_ACTION_SCHEMA:
        raise ValueError("completed init transformer config is incompatible")

    sidecars = completion.get("sidecar_inventory")
    raw_files = sidecars.get("files") if isinstance(sidecars, dict) else None
    if not isinstance(raw_files, list):
        raise ValueError("checkpoint completion marker lacks sidecar inventory")
    config_record = next(
        (
            record
            for record in raw_files
            if isinstance(record, dict)
            and record.get("path") == "transformer/config.json"
        ),
        None,
    )
    if (
        not isinstance(config_record, dict)
        or config_record.get("size_bytes") != config_stat.st_size
        or config_record.get("sha256") != sha256_file(config_path)
    ):
        raise ValueError("transformer config differs from completion sidecar")

    return CompletedWeightsInit(
        checkpoint_root=root,
        completion_sha256=completion_sha256,
        transformer_identity=transformer_identity,
        step=step,
        world_size=world_size,
    )


__all__ = ("CompletedWeightsInit", "validate_completed_weights_init")
