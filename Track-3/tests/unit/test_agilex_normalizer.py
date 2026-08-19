# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""AgileX qpos14 normalizer identity and masking tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from n0_twam.integrations.worldarena.agilex_normalizer import (
    fit_agilex_normalizer,
    load_agilex_normalizer,
)


def _actions() -> np.ndarray:
    base = np.linspace(-1.0, 1.0, 6, dtype=np.float32)[:, None]
    offsets = np.linspace(0.0, 0.13, 14, dtype=np.float32)[None, :]
    actions = base + offsets
    actions[-1] = 1_000.0
    return actions


def test_fit_excludes_invalid_terminal_action_and_round_trips() -> None:
    actions = _actions()
    normalizer = fit_agilex_normalizer(
        action_batches=(actions,),
        valid_mask_batches=(np.asarray((True, True, True, True, True, False)),),
        source_manifest_sha256="a" * 64,
        repo_route_manifest_sha256="b" * 64,
    )

    decoded = normalizer.denormalize(normalizer.normalize(actions[:5]))

    np.testing.assert_allclose(decoded, actions[:5], atol=2e-6)
    assert max(normalizer.q99) < 10.0
    assert normalizer.sample_count == 5
    assert "lower_bounds" not in normalizer.to_json_dict()
    assert "upper_bounds" not in normalizer.to_json_dict()


def test_load_binds_file_hash_and_canonical_contract(tmp_path) -> None:
    normalizer = fit_agilex_normalizer(
        action_batches=(_actions(),),
        valid_mask_batches=(np.asarray((True, True, True, True, True, False)),),
        source_manifest_sha256="a" * 64,
        repo_route_manifest_sha256="b" * 64,
    )
    path = tmp_path / "normalizer.json"
    path.write_text(
        json.dumps(normalizer.to_json_dict(), sort_keys=True),
        encoding="utf-8",
    )

    loaded = load_agilex_normalizer(
        path,
        expected_file_sha256=normalizer.file_sha256(path),
        expected_source_manifest_sha256="a" * 64,
        expected_repo_route_manifest_sha256="b" * 64,
    )

    assert loaded.contract_sha256 == normalizer.contract_sha256
    with pytest.raises(ValueError, match="source manifest"):
        load_agilex_normalizer(
            path,
            expected_file_sha256=normalizer.file_sha256(path),
            expected_source_manifest_sha256="c" * 64,
            expected_repo_route_manifest_sha256="b" * 64,
        )


def test_load_rejects_noncanonical_numeric_metadata(tmp_path) -> None:
    normalizer = fit_agilex_normalizer(
        action_batches=(_actions(),),
        valid_mask_batches=(np.asarray((True, True, True, True, True, False)),),
        source_manifest_sha256="a" * 64,
        repo_route_manifest_sha256="b" * 64,
    )
    payload = normalizer.to_json_dict()
    payload["sample_count"] = str(payload["sample_count"])
    path = tmp_path / "noncanonical-normalizer.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_count.*integer"):
        load_agilex_normalizer(
            path,
            expected_file_sha256=normalizer.file_sha256(path),
            expected_source_manifest_sha256="a" * 64,
            expected_repo_route_manifest_sha256="b" * 64,
        )


def test_fit_rejects_empty_or_non_finite_valid_population() -> None:
    actions = _actions()
    with pytest.raises(ValueError, match="no valid"):
        fit_agilex_normalizer(
            action_batches=(actions,),
            valid_mask_batches=(np.zeros(6, dtype=np.bool_),),
            source_manifest_sha256="a" * 64,
            repo_route_manifest_sha256="b" * 64,
        )
    actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_agilex_normalizer(
            action_batches=(actions,),
            valid_mask_batches=(np.ones(6, dtype=np.bool_),),
            source_manifest_sha256="a" * 64,
            repo_route_manifest_sha256="b" * 64,
        )
