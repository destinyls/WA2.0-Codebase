# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from n0_twam.checkpointing.identity import (
    ACTION_TENSOR_KEYS,
    TRANSFORMER_ACTION_INNER_DIM,
    TRANSFORMER_SENTINEL_KEYS,
    TRANSFORMER_WEIGHTS_FILENAME,
)
from n0_twam.integrations.worldarena.franka_actions import (
    DERIVED_ACTION_SCHEMA,
    FRANKA_ACTION_SCHEMA,
)
from n0_twam.integrations.worldarena.franka_manifest import sha256_file
from n0_twam.track32.completed_init import validate_completed_weights_init


def _write_completed_checkpoint(tmp_path: Path) -> Path:
    root = tmp_path / "checkpoint_step_5000"
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    weights = transformer / TRANSFORMER_WEIGHTS_FILENAME
    weights.write_bytes(b"receipt-bound-test-weights")
    config = transformer / "config.json"
    config.write_text(
        json.dumps({"action_dim": 20, "action_schema": "ee20_absee"}),
        encoding="utf-8",
    )
    identity = {
        "schema_version": 1,
        "file_name": TRANSFORMER_WEIGHTS_FILENAME,
        "size_bytes": weights.stat().st_size,
        "sha256": "a" * 64,
        "tensor_count": len(ACTION_TENSOR_KEYS) + len(TRANSFORMER_SENTINEL_KEYS),
        "action_dim": 20,
        "action_shapes": {
            "action_embedder.weight": [TRANSFORMER_ACTION_INNER_DIM, 20],
            "action_embedder.bias": [TRANSFORMER_ACTION_INNER_DIM],
            "action_proj_out.weight": [20, TRANSFORMER_ACTION_INNER_DIM],
            "action_proj_out.bias": [20],
        },
        "required_sentinel_keys": list(TRANSFORMER_SENTINEL_KEYS),
    }
    completion = {
        "status": "complete",
        "step": 5000,
        "world_size": 48,
        "action_schema": "ee20_absee",
        "training_lineage": {
            "source_action_schema": FRANKA_ACTION_SCHEMA,
            "derived_action_schema": DERIVED_ACTION_SCHEMA,
            "target_action_schema": "ee20_absee",
            "tactile_mode": "disabled",
            "tactile_profile": "vision_only",
        },
        "tactile_profile_contract": {
            "profile": "vision_only",
            "tactile_mode": "disabled",
            "freeze_tactile_parameters": True,
        },
        "transformer_identity": identity,
        "sidecar_inventory": {
            "files": [
                {
                    "path": "transformer/config.json",
                    "size_bytes": config.stat().st_size,
                    "sha256": sha256_file(config),
                }
            ]
        },
    }
    marker = root / "checkpoint_complete.json"
    marker.write_text(json.dumps(completion), encoding="utf-8")
    return root


def test_completed_init_uses_receipt_and_small_sidecars(tmp_path: Path) -> None:
    root = _write_completed_checkpoint(tmp_path)
    marker_sha256 = sha256_file(root / "checkpoint_complete.json")

    validated = validate_completed_weights_init(
        root,
        expected_completion_sha256=marker_sha256,
    )

    assert validated.step == 5000
    assert validated.world_size == 48
    assert validated.transformer_identity["sha256"] == "a" * 64


def test_completed_init_rejects_config_drift(tmp_path: Path) -> None:
    root = _write_completed_checkpoint(tmp_path)
    config = root / "transformer" / "config.json"
    config.write_text(
        json.dumps({"action_dim": 20, "action_schema": "wrong"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config is incompatible"):
        validate_completed_weights_init(root)
