"""Stable-FD and immutable-byte regressions for checkpoint sidecars."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from n0_twam.checkpointing.runtime_checkpoint_snapshot import (
    create_runtime_transformer_snapshot,
    validate_checkpoint_run_root_boundary,
    validate_loaded_transformer_architecture,
)
from n0_twam.checkpointing.sidecar_snapshot import (
    capture_sidecar_snapshot,
    capture_stable_json_file,
)
from n0_twam.checkpointing.strict_resume import expected_sidecar_paths
from n0_twam.checkpointing.training_lineage import (
    validate_stage_a_parent_checkpoint,
)
from tests.unit.test_track31_checkpoint_review_regressions import _write_resume
from tests.unit.test_track31_preflight import _TRACK31_ARTIFACT_IDENTITY


def test_sidecar_snapshot_does_not_reread_replaced_path(tmp_path: Path) -> None:
    _write_resume(tmp_path)
    completion = json.loads(
        (tmp_path / "checkpoint_complete.json").read_text(encoding="utf-8")
    )
    snapshot = capture_sidecar_snapshot(
        tmp_path,
        completion["sidecar_inventory"],
        expected_sidecar_paths(1),
    )
    original = snapshot.json_object("train_meta.json", label="train metadata")
    exposed_inventory = snapshot.inventory
    exposed_inventory["inventory_sha256"] = "0" * 64
    (tmp_path / "train_meta.json").write_text(
        json.dumps({"replaced": True}),
        encoding="utf-8",
    )

    assert snapshot.json_object("train_meta.json", label="train metadata") == original
    assert snapshot.inventory["inventory_sha256"] != "0" * 64


def test_stable_json_capture_rejects_path_swap_during_fd_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.json"
    replacement = tmp_path / "replacement.json"
    target.write_text(json.dumps({"version": 1}), encoding="utf-8")
    replacement.write_text(json.dumps({"version": 2}), encoding="utf-8")
    original_read = os.read
    swapped = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        data = original_read(descriptor, count)
        if data and not swapped:
            swapped = True
            os.replace(replacement, target)
        return data

    monkeypatch.setattr(
        "n0_twam.checkpointing.sidecar_snapshot.os.read",
        racing_read,
    )

    with pytest.raises(RuntimeError, match="changed while reading"):
        capture_stable_json_file(target, label="racing state")


def test_runtime_parent_loader_consumes_verified_private_config_snapshot(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_resume(checkpoint)
    contract = validate_stage_a_parent_checkpoint(
        checkpoint,
        expected_run_role="final_refit",
        expected_track31_artifacts=_TRACK31_ARTIFACT_IDENTITY,
    )
    verified_config = contract["transformer_config"]
    assert isinstance(verified_config, dict)
    config_path = checkpoint / "transformer" / "config.json"
    config_path.write_text(json.dumps({"replaced": True}), encoding="utf-8")

    runtime_snapshot = create_runtime_transformer_snapshot(
        checkpoint_root=checkpoint,
        run_root=tmp_path / "run",
        sidecars=contract["sidecar_snapshot"],
        transformer_identity=contract["transformer_identity"],
    )
    try:
        runtime_config = json.loads(
            (runtime_snapshot.transformer_dir / "config.json").read_text(
                encoding="utf-8"
            )
        )
        assert runtime_config == verified_config
        assert runtime_config != {"replaced": True}
    finally:
        runtime_snapshot.cleanup()

    assert not runtime_snapshot.checkpoint_root.exists()


def test_runtime_snapshot_requires_disjoint_checkpoint_and_run_roots(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="disjoint"):
        validate_checkpoint_run_root_boundary(
            checkpoint,
            checkpoint / "nested-run",
        )


def test_loaded_transformer_architecture_checks_all_persisted_fields() -> None:
    verified = {
        "action_dim": 8,
        "hidden_size": 3072,
        "nested": {"layers": [1, 2]},
        "_name_or_path": "/untrusted/source",
    }

    validate_loaded_transformer_architecture(
        {
            "action_dim": 8,
            "hidden_size": 4096,
            "nested": {"layers": (1, 2)},
        },
        verified_config=verified,
        overrides={"hidden_size": 4096},
    )
    with pytest.raises(ValueError, match="nested"):
        validate_loaded_transformer_architecture(
            {
                "action_dim": 8,
                "hidden_size": 4096,
                "nested": {"layers": [1, 3]},
            },
            verified_config=verified,
            overrides={"hidden_size": 4096},
        )
