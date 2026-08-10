"""Reviewer counterexamples for strict checkpoint identity and scalar typing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from script.track3_1.preflight_train import _audit_checkpoint
from tests.unit.test_track31_invocation_contract import train_module  # noqa: F401
from tests.unit.test_track31_preflight import (
    _RUNTIME_SOURCE_IDENTITY,
    _set_current_runtime_provenance_env,
    _write_completion_marker,
    _write_resume_sidecars,
    _write_training_state,
    _write_transformer_checkpoint,
)


def _write_resume(root: Path) -> dict[str, object]:
    _write_transformer_checkpoint(
        root,
        action_dim=8,
        action_schema="qpos8_next_step",
    )
    strict_fields = _write_resume_sidecars(root, world_size=1)
    _write_training_state(root, strict_fields, world_size=1)
    return strict_fields


def _set_resume_env(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("N0_TRACK31_TRAIN_PROFILE", "multitask_pretrain_v1")
    monkeypatch.setenv("N0_TRACK31_RUN_ROLE", "final_refit")
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(root))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")
    monkeypatch.setenv("N0_TRACK31_MAX_LATENT_FRAMES", "5")
    monkeypatch.setenv("N0_TRACK31_GRADIENT_ACCUMULATION_STEPS", "4")
    monkeypatch.delenv("N0_TRACK31_INIT_FROM", raising=False)
    _set_current_runtime_provenance_env(monkeypatch)


def test_preflight_rejects_train_meta_transformer_identity_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict_fields = _write_resume(tmp_path)
    metadata_path = tmp_path / "train_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["transformer_identity"]["sha256"] = "9" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_completion_marker(tmp_path, strict_fields, world_size=1, step=7)
    _set_resume_env(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="train metadata transformer identity"):
        _audit_checkpoint()


@pytest.mark.parametrize(
    ("field", "value"),
    (("world_size", True), ("world_size", "1"), ("step", "7")),
)
def test_preflight_rejects_non_integer_completion_scalars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    _write_resume(tmp_path)
    completion_path = tmp_path / "checkpoint_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion[field] = value
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    _set_resume_env(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match=f"completion {field}.*integer"):
        _audit_checkpoint()


def test_trainer_rejects_train_meta_transformer_identity_tamper(  # noqa: F811
    train_module: Any,  # noqa: F811
    tmp_path: Path,
) -> None:
    strict_fields = _write_resume(tmp_path)
    metadata_path = tmp_path / "train_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["transformer_identity"]["sha256"] = "9" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_completion_marker(tmp_path, strict_fields, world_size=1, step=7)

    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.config = SimpleNamespace(
        world_size=1,
        rank=0,
        action_dim=8,
        learning_rate=1e-4,
        sampler_coverage_mode=None,
    )
    trainer.gradient_accumulation_steps = 4
    trainer.action_codec = SimpleNamespace(spec=SimpleNamespace(name="qpos8_next_step"))
    trainer.training_execution_contract = strict_fields["training_execution_contract"]
    trainer.track31_artifacts = None
    trainer.track31_tactile_contract = None
    trainer.runtime_source_identity = _RUNTIME_SOURCE_IDENTITY

    with pytest.raises(ValueError, match="train metadata transformer identity"):
        trainer._load_and_validate_resume_sidecars(
            tmp_path,
            actual_transformer_identity=strict_fields["transformer_identity"],
        )


def test_preflight_rejects_missing_formal_checkpoint_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict_fields = _write_resume(tmp_path)
    metadata_path = tmp_path / "train_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["runtime_source_identity"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_completion_marker(tmp_path, strict_fields, world_size=1, step=7)
    _set_resume_env(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="train metadata.*provenance is invalid"):
        _audit_checkpoint()


def test_preflight_rejects_runtime_source_drift_from_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_resume(tmp_path)
    _set_resume_env(monkeypatch, tmp_path)
    monkeypatch.setenv("N0_TRACK31_OVERLAY_MANIFEST_SHA256", "f" * 64)

    with pytest.raises(ValueError, match="current launch"):
        _audit_checkpoint()


def test_trainer_rejects_provenance_disagreement_across_sidecars(  # noqa: F811
    train_module: Any,  # noqa: F811
    tmp_path: Path,
) -> None:
    strict_fields = _write_resume(tmp_path)
    metadata_path = tmp_path / "train_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_invocation_identity"]["launch_manifest_sha256"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_completion_marker(tmp_path, strict_fields, world_size=1, step=7)

    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.config = SimpleNamespace(
        world_size=1,
        rank=0,
        action_dim=8,
        learning_rate=1e-4,
        sampler_coverage_mode=None,
    )
    trainer.gradient_accumulation_steps = 4
    trainer.action_codec = SimpleNamespace(spec=SimpleNamespace(name="qpos8_next_step"))
    trainer.training_execution_contract = strict_fields["training_execution_contract"]
    trainer.track31_artifacts = None
    trainer.track31_tactile_contract = None
    trainer.runtime_source_identity = _RUNTIME_SOURCE_IDENTITY

    with pytest.raises(
        ValueError,
        match="checkpoint_invocation_identity differs across sidecars",
    ):
        trainer._load_and_validate_resume_sidecars(
            tmp_path,
            actual_transformer_identity=strict_fields["transformer_identity"],
        )
