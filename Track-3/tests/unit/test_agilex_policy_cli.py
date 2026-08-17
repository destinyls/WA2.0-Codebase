# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Public CLI contracts for local AgileX Policy preparation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from n0_twam.cli import run_cli
from n0_twam.integrations.worldarena import agilex_policy_io


def test_agilex_policy_template_prints_and_writes_strict_json(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "policy" / "agilex.policy.json"

    result = run_cli(
        [
            "track32",
            "agilex-policy-template",
            "--output",
            str(destination),
        ]
    )

    assert result["kind"] == "agilex-policy"
    template = result["template"]
    assert isinstance(template, dict)
    assert template["policy_id"] == "n0-twam-agilex"
    assert template["tactile_profile"] == "vision_only"
    assert template["task_routes"] == {}
    assert str(template["checkpoint_identity_sha256"]).startswith("REPLACE_")
    assert json.loads(destination.read_text(encoding="utf-8")) == template
    assert Path(str(result["output"])) == destination.resolve()

    with pytest.raises(FileExistsError):
        run_cli(
            [
                "track32",
                "agilex-policy-template",
                "--output",
                str(destination),
            ]
        )


def test_agilex_policy_check_uses_fail_closed_loader_without_model_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "agilex.policy.json"
    config_path.write_text("{}\n", encoding="utf-8")
    bundle = tmp_path / "serve-bundle"
    calls: list[Path] = []

    def load(path: Path) -> SimpleNamespace:
        calls.append(path)
        return SimpleNamespace(
            source_path=config_path.resolve(),
            policy=SimpleNamespace(
                policy_id="n0-twam-agilex",
                tactile_profile="mixed",
                task_routes={"insert": object(), "wipe": object()},
            ),
            policy_config_file_sha256="1" * 64,
            config_contract_sha256="2" * 64,
            serve_bundle=bundle.resolve(),
            serve_bundle_identity_sha256="3" * 64,
            checkpoint_identity_sha256="4" * 64,
            normalizer_contract_sha256="5" * 64,
            repo_route_manifest_sha256="6" * 64,
        )

    monkeypatch.setattr(agilex_policy_io, "load_agilex_policy_config", load)

    result = run_cli(["track32", "agilex-policy-check", "--config", str(config_path)])

    assert calls == [config_path]
    assert result == {
        "kind": "agilex-policy-check",
        "status": "verified",
        "config": str(config_path.resolve()),
        "policy_id": "n0-twam-agilex",
        "tactile_profile": "mixed",
        "task_ids": ["insert", "wipe"],
        "policy_config_file_sha256": "1" * 64,
        "config_contract_sha256": "2" * 64,
        "serve_bundle": str(bundle.resolve()),
        "serve_bundle_identity_sha256": "3" * 64,
        "checkpoint_identity_sha256": "4" * 64,
        "normalizer_contract_sha256": "5" * 64,
        "repo_route_manifest_sha256": "6" * 64,
    }
    assert not (tmp_path / "serve-output").exists()


def test_agilex_policy_check_propagates_identity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "agilex.policy.json"
    config_path.write_text("{}\n", encoding="utf-8")

    def reject(_: Path) -> SimpleNamespace:
        raise ValueError("AgileX serve bundle identity mismatch")

    monkeypatch.setattr(agilex_policy_io, "load_agilex_policy_config", reject)

    with pytest.raises(ValueError, match="serve bundle identity mismatch"):
        run_cli(["track32", "agilex-policy-check", "--config", str(config_path)])
