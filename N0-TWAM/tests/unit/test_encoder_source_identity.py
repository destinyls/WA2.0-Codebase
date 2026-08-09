# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import json
from pathlib import Path

import pytest

from n0_twam.data.encoder_source_identity import (
    build_encoder_source_identity,
    load_encoder_source_identity,
    validate_encoder_source_identity,
)


def _write_model(root: Path) -> None:
    files = {
        "vae/config.json": b"{}",
        "vae/diffusion_pytorch_model.safetensors": b"vae-weights",
        "tokenizer/tokenizer_config.json": b"{}",
        "tokenizer/spiece.model": b"tokenizer-weights",
        "text_encoder/config.json": b"{}",
        "text_encoder/model.safetensors": b"text-weights",
    }
    for relative_path, payload in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def test_encoder_source_identity_binds_every_used_model_file(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    _write_model(model_root)

    identity = build_encoder_source_identity(model_root)

    assert validate_encoder_source_identity(identity) == identity
    assert {entry["relative_path"] for entry in identity["files"]} == {
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "tokenizer/tokenizer_config.json",
        "tokenizer/spiece.model",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
    }


def test_encoder_source_weight_tampering_changes_identity(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    _write_model(model_root)
    before = build_encoder_source_identity(model_root)
    (model_root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(
        b"tampered-weights"
    )

    after = build_encoder_source_identity(model_root)

    assert after["identity_sha256"] != before["identity_sha256"]


def test_encoder_source_identity_rejects_digest_tampering(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    _write_model(model_root)
    identity = build_encoder_source_identity(model_root)
    identity["files"][0]["size_bytes"] += 1

    with pytest.raises(ValueError, match="digest is inconsistent"):
        validate_encoder_source_identity(identity)


def test_run_local_encoder_source_identity_cache_is_validated(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    _write_model(model_root)
    identity = build_encoder_source_identity(model_root)
    cache_path = tmp_path / "identity.json"
    cache_path.write_text(json.dumps(identity), encoding="utf-8")

    assert load_encoder_source_identity(cache_path) == identity


def test_encoder_source_identity_cache_rejects_symlinks(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    _write_model(model_root)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(
        json.dumps(build_encoder_source_identity(model_root)),
        encoding="utf-8",
    )
    link = tmp_path / "identity-link.json"
    link.symlink_to(identity_path)

    with pytest.raises(FileNotFoundError, match="regular file"):
        load_encoder_source_identity(link)
