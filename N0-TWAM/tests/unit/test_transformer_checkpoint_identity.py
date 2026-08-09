# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from n0_twam.checkpointing.identity import audit_transformer_checkpoint
from script.track3_1.preflight_train import _audit_checkpoint
from tests.unit.test_track31_preflight import (
    _set_current_runtime_provenance_env,
)


def _tensors(
    action_dim: int,
    *,
    tensor_action_dim: int | None = None,
    missing_key: str | None = None,
    fill_value: float = 0.0,
) -> dict[str, torch.Tensor]:
    stored_action_dim = tensor_action_dim or action_dim
    tensors = {
        "action_embedder.weight": torch.full((3072, stored_action_dim), fill_value),
        "action_embedder.bias": torch.zeros(3072),
        "action_proj_out.weight": torch.zeros(stored_action_dim, 3072),
        "action_proj_out.bias": torch.zeros(stored_action_dim),
        "condition_embedder.text_embedder.linear_1.weight": torch.zeros(1),
        "mot.experts.action.in_proj.weight": torch.zeros(1),
        "mot.experts.tactile.in_proj.weight": torch.zeros(1),
    }
    if missing_key is not None:
        del tensors[missing_key]
    return tensors


def _write_release(
    root: Path,
    *,
    tensor_action_dim: int | None = None,
    missing_key: str | None = None,
    fill_value: float = 0.0,
) -> Path:
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(
        json.dumps({"is_mot": True, "action_dim": 20, "action_schema": "ee20_pi05"}),
        encoding="utf-8",
    )
    weights_path = transformer / "diffusion_pytorch_model.safetensors"
    save_file(
        _tensors(
            20,
            tensor_action_dim=tensor_action_dim,
            missing_key=missing_key,
            fill_value=fill_value,
        ),
        weights_path,
    )
    return weights_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_initial_preflight(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    expected_sha256: str | None,
) -> None:
    _set_current_runtime_provenance_env(monkeypatch)
    monkeypatch.delenv("N0_TRACK31_RESUME_FROM", raising=False)
    monkeypatch.setenv("N0_RELEASED_CHECKPOINT", str(root))
    if expected_sha256 is None:
        monkeypatch.delenv("N0_RELEASED_TRANSFORMER_SHA256", raising=False)
    else:
        monkeypatch.setenv("N0_RELEASED_TRANSFORMER_SHA256", expected_sha256)


def test_audit_reads_real_safetensors_header_and_hash(tmp_path: Path) -> None:
    weights_path = _write_release(tmp_path)

    identity = audit_transformer_checkpoint(
        weights_path,
        expected_action_dim=20,
    )

    assert identity["sha256"] == _sha256(weights_path)
    assert identity["tensor_count"] == 7
    assert identity["action_shapes"]["action_embedder.weight"] == [3072, 20]
    assert identity["action_shapes"]["action_proj_out.weight"] == [20, 3072]


def test_audit_rejects_symlink(tmp_path: Path) -> None:
    weights_path = _write_release(tmp_path / "real")
    symlink_path = tmp_path / "linked.safetensors"
    symlink_path.symlink_to(weights_path)

    with pytest.raises(ValueError, match="regular non-symlink"):
        audit_transformer_checkpoint(symlink_path, expected_action_dim=20)


def test_audit_rejects_malformed_safetensors(tmp_path: Path) -> None:
    weights_path = tmp_path / "malformed.safetensors"
    weights_path.write_bytes(b"not-a-safetensors-file")

    with pytest.raises(ValueError, match="invalid transformer safetensors"):
        audit_transformer_checkpoint(weights_path, expected_action_dim=20)


def test_audit_rejects_wrong_action_shape(tmp_path: Path) -> None:
    weights_path = _write_release(tmp_path, tensor_action_dim=19)

    with pytest.raises(ValueError, match="action_dim=20"):
        audit_transformer_checkpoint(weights_path, expected_action_dim=20)


def test_audit_rejects_missing_sentinel(tmp_path: Path) -> None:
    missing_key = "mot.experts.tactile.in_proj.weight"
    weights_path = _write_release(tmp_path, missing_key=missing_key)

    with pytest.raises(ValueError, match=missing_key):
        audit_transformer_checkpoint(weights_path, expected_action_dim=20)


def test_initial_preflight_requires_released_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_release(tmp_path)
    _configure_initial_preflight(monkeypatch, tmp_path, None)

    with pytest.raises(ValueError, match="N0_RELEASED_TRANSFORMER_SHA256"):
        _audit_checkpoint()


@pytest.mark.parametrize("invalid_hash", ("0" * 63, "A" * 64))
def test_initial_preflight_rejects_noncanonical_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_hash: str,
) -> None:
    _write_release(tmp_path)
    _configure_initial_preflight(monkeypatch, tmp_path, invalid_hash)

    with pytest.raises(ValueError, match="64 lowercase hex"):
        _audit_checkpoint()


def test_initial_preflight_rejects_wrong_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_release(tmp_path)
    _configure_initial_preflight(monkeypatch, tmp_path, "0" * 64)

    with pytest.raises(ValueError, match="does not match"):
        _audit_checkpoint()


def test_initial_preflight_rejects_tampered_valid_safetensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights_path = _write_release(tmp_path)
    original_sha256 = _sha256(weights_path)
    save_file(_tensors(20, fill_value=1.0), weights_path)
    assert _sha256(weights_path) != original_sha256
    _configure_initial_preflight(monkeypatch, tmp_path, original_sha256)

    with pytest.raises(ValueError, match="does not match"):
        _audit_checkpoint()
