# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Route-sealing regressions for AgileX serve-bundle publication."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from n0_twam.integrations.worldarena.agilex_manifest import canonical_sha256
from n0_twam.integrations.worldarena.agilex_serve_bundle import (
    build_agilex_serve_bundle,
    verify_agilex_serve_bundle,
)
from tests.unit.test_agilex_serve_bundle import (
    _fixture,
    _reseal_checkpoint,
    _routes,
    _write_json,
)


def test_builder_rejects_noncanonical_training_rgb(tmp_path: Path) -> None:
    bad_routes = _routes("vision_only")
    bad_routes["official_rgb"]["rgb_keys"] = ["observation.images.top"]

    with pytest.raises(ValueError, match="canonical AgileX"):
        _fixture(tmp_path, "vision_only", bad_routes)


def test_builder_rejects_wrench_route_without_tactile(tmp_path: Path) -> None:
    request = _fixture(tmp_path, "mixed")
    task_routes = request["task_routes"]
    assert isinstance(task_routes, dict)
    route = dict(task_routes["pick"])
    route.update({"tactile_required": False, "tactile_keys": []})
    core = {key: value for key, value in route.items() if key != "contract_sha256"}
    request["task_routes"] = {
        "pick": {**core, "contract_sha256": canonical_sha256(core)}
    }
    request["task_routes_sha256"] = canonical_sha256(request["task_routes"])

    with pytest.raises(ValueError, match="requires an active tactile"):
        build_agilex_serve_bundle(**request)


def test_builder_rejects_vision_tactile_task_without_wrench(tmp_path: Path) -> None:
    request = _fixture(tmp_path, "vision_tactile")
    task_routes = request["task_routes"]
    assert isinstance(task_routes, dict)
    route = dict(task_routes["pick"])
    route.update({"wrench_required": False, "wrench_keys": []})
    core = {key: value for key, value in route.items() if key != "contract_sha256"}
    request["task_routes"] = {
        "pick": {**core, "contract_sha256": canonical_sha256(core)}
    }
    request["task_routes_sha256"] = canonical_sha256(request["task_routes"])

    with pytest.raises(ValueError, match="require tactile and wrench"):
        build_agilex_serve_bundle(**request)


def test_builder_rejects_tampered_action_migration(tmp_path: Path) -> None:
    request = _fixture(tmp_path, "vision_only")
    checkpoint = request["checkpoint"]
    assert isinstance(checkpoint, Path)
    report_path = checkpoint / "action_migration_report.json"
    report = json.loads(report_path.read_text())
    report["plan"]["unexpected_source_keys"] = ["obsolete.weight"]
    _write_json(report_path, report)
    _reseal_checkpoint(request)

    with pytest.raises(ValueError, match="unexpected source"):
        build_agilex_serve_bundle(**request)


def test_verifier_reaudits_checkpoint_weight_bytes(tmp_path: Path) -> None:
    request = _fixture(tmp_path, "vision_only")
    receipt = build_agilex_serve_bundle(**request)
    checkpoint = request["checkpoint"]
    assert isinstance(checkpoint, Path)
    weights = checkpoint / "transformer" / "diffusion_pytorch_model.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)

    with pytest.raises(ValueError, match="transformer identity"):
        verify_agilex_serve_bundle(
            request["output"],
            expected_receipt_file_sha256=receipt["receipt_file_sha256"],
        )
