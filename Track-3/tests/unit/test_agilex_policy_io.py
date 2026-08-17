# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict local policy configuration and artifact bindings for AgileX."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from n0_twam.data.encoder_source_identity import build_encoder_source_identity
from n0_twam.embodiments import build_agilex_repo_route_manifest
from n0_twam.integrations.worldarena.agilex_normalizer import (
    AgileXQpos14Normalizer,
)
from n0_twam.integrations.worldarena.agilex_policy_artifacts import (
    verify_agilex_policy_artifacts as verify_artifacts,
)
from n0_twam.integrations.worldarena.agilex_policy_io import (
    AgileXDirectPolicyConfig,
    agilex_policy_config_template,
    load_agilex_policy_config,
)
from n0_twam.integrations.worldarena.agilex_policy_contracts import AgileXTaskRoute


def _canonical(payload: object) -> str:
    raw = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hashed(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "contract_sha256": _canonical(payload)}


def _policy_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "transformer").mkdir(parents=True)
    base = tmp_path / "base"
    for component, filename in (
        ("vae", "config.json"),
        ("tokenizer", "tokenizer_config.json"),
        ("text_encoder", "config.json"),
    ):
        source = base / component
        source.mkdir(parents=True)
        _write_json(source / filename, {"component": component})
    (bundle / "transformer").symlink_to(checkpoint / "transformer")
    for component in ("vae", "tokenizer", "text_encoder"):
        (bundle / component).symlink_to(base / component)

    route_contract = build_agilex_repo_route_manifest(
        {
            "agilex_touch": {
                "embodiment": "agilex_dual_qpos14_v1",
                "action_schema": "qpos14_joint_absolute_v1",
                "rgb_keys": [
                    "observation.images.top",
                    "observation.images.wrist_l",
                    "observation.images.wrist_r",
                ],
                "tactile_keys": [
                    "observation.images.tactile_l",
                    "observation.images.tactile_r",
                ],
                "wrench_keys": [
                    "observation.wrench.left",
                    "observation.wrench.right",
                ],
            }
        }
    ).to_json_dict()
    route_file_sha = _write_json(bundle / "repo_route_manifest.json", route_contract)
    source_sha = "1" * 64
    normalizer = AgileXQpos14Normalizer(
        q01=(-1.0,) * 14,
        q99=(1.0,) * 14,
        sample_count=128,
        source_manifest_sha256=source_sha,
        repo_route_manifest_sha256=str(route_contract["contract_sha256"]),
    )
    normalizer_file_sha = _write_json(
        bundle / "normalizer.json", normalizer.to_json_dict()
    )
    safety = _self_hashed(
        {
            "lower_bounds": [-2.0] * 14,
            "upper_bounds": [2.0] * 14,
            "max_step_per_second": [0.5] * 14,
            "min_execution_dt_s": 0.01,
            "max_execution_dt_s": 0.2,
            "max_state_age_s": 0.25,
            "max_inference_latency_s": 1.0,
        }
    )
    task_route = _self_hashed(
        {
            "task_id": "wipe",
            "prompt": "wipe the table",
            "tactile_required": True,
            "wrench_required": True,
            "tactile_keys": [
                "observation.images.tactile_l",
                "observation.images.tactile_r",
            ],
            "wrench_keys": [
                "observation.wrench.left",
                "observation.wrench.right",
            ],
        }
    )
    task_routes = {"wipe": task_route}
    checkpoint_sha = "2" * 64
    checkpoint_identity: dict[str, object] = {
        "schema_version": 1,
        "identity_sha256": checkpoint_sha,
    }
    encoder_identity = build_encoder_source_identity(base)
    contact_sha = "3" * 64
    receipt_core: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "action_schema": "qpos14_joint_absolute_v1",
        "tactile_profile": "vision_tactile",
        "checkpoint_root": str(checkpoint.resolve()),
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_identity_sha256": checkpoint_sha,
        "base_model_root": str(base.resolve()),
        "encoder_source_identity": encoder_identity,
        "normalizer_file_sha256": normalizer_file_sha,
        "normalizer_contract_sha256": normalizer.contract_sha256,
        "source_manifest_sha256": source_sha,
        "repo_route_manifest_file_sha256": route_file_sha,
        "repo_route_manifest_sha256": route_contract["contract_sha256"],
        "contact_profile_contract_sha256": contact_sha,
        "task_routes_sha256": _canonical(task_routes),
        "safety_contract_sha256": safety["contract_sha256"],
        "component_names": [
            "transformer",
            "vae",
            "tokenizer",
            "text_encoder",
        ],
    }
    receipt = {
        **receipt_core,
        "bundle_identity_sha256": _canonical(receipt_core),
    }
    receipt_file_sha = _write_json(bundle / "serve_bundle_receipt.json", receipt)
    config_core: dict[str, object] = {
        "schema_version": 1,
        "policy_id": "n0-twam-agilex",
        "tactile_profile": "vision_tactile",
        "serve_bundle": "./bundle",
        "serve_output": "./output",
        "serve_bundle_receipt_sha256": receipt_file_sha,
        "serve_bundle_identity_sha256": receipt["bundle_identity_sha256"],
        "checkpoint_identity_sha256": checkpoint_sha,
        "normalizer_file_sha256": normalizer_file_sha,
        "normalizer_contract_sha256": normalizer.contract_sha256,
        "source_manifest_sha256": source_sha,
        "repo_route_manifest_file_sha256": route_file_sha,
        "repo_route_manifest_sha256": route_contract["contract_sha256"],
        "contact_profile_contract_sha256": contact_sha,
        "cuda_visible_device": "0",
        "distributed_port": 29643,
        "episode_seed": 20260811,
        "max_chunk_actions": 12,
        "video_inference_steps": 3,
        "action_inference_steps": 4,
        "task_routes": task_routes,
        "safety": safety,
    }
    payload = {
        **config_core,
        "config_contract_sha256": _canonical(config_core),
    }
    config_path = tmp_path / "policy.json"
    _write_json(config_path, payload)
    identities = {
        "checkpoint": checkpoint_identity,
        "encoder": encoder_identity,
    }
    return config_path, payload, identities


def _load_without_deep_artifact_scan(
    monkeypatch: pytest.MonkeyPatch, path: Path
) -> AgileXDirectPolicyConfig:
    monkeypatch.setattr(
        "n0_twam.integrations.worldarena.agilex_policy_io."
        "verify_agilex_policy_artifacts",
        lambda _: {},
    )
    return load_agilex_policy_config(path)


def _verify(
    loaded: AgileXDirectPolicyConfig, identities: dict[str, object]
) -> dict[str, object]:
    receipt = verify_artifacts(
        loaded,
        checkpoint_identity_builder=lambda _: cast(
            Mapping[str, object], identities["checkpoint"]
        ),
        encoder_identity_builder=lambda _: cast(
            Mapping[str, object], identities["encoder"]
        ),
    )
    return dict(receipt)


def test_loader_binds_policy_bundle_checkpoint_routes_and_normalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, identities = _policy_fixture(tmp_path)
    loaded = _load_without_deep_artifact_scan(monkeypatch, path)

    assert loaded.policy.policy_id == "n0-twam-agilex"
    assert loaded.policy.tactile_profile == "vision_tactile"
    assert tuple(loaded.policy.task_routes) == ("wipe",)
    assert loaded.serve_bundle == tmp_path / "bundle"
    assert loaded.checkpoint_identity_sha256 == "2" * 64
    assert loaded.normalizer.contract_sha256 == loaded.normalizer_contract_sha256
    assert _verify(loaded, identities)["status"] == "complete"


def test_loader_rejects_unknown_fields_and_tampered_self_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, payload, _ = _policy_fixture(tmp_path)
    payload["unexpected"] = True
    _write_json(path, payload)
    with pytest.raises(ValueError, match="unexpected"):
        load_agilex_policy_config(path)

    path, payload, _ = _policy_fixture(tmp_path / "second")
    assert isinstance(payload["safety"], dict)
    safety = dict(payload["safety"])
    safety["max_state_age_s"] = 9.0
    payload["safety"] = safety
    core = {
        key: value for key, value in payload.items() if key != "config_contract_sha256"
    }
    payload["config_contract_sha256"] = _canonical(core)
    _write_json(path, payload)
    monkeypatch.setattr(
        "n0_twam.integrations.worldarena.agilex_policy_io."
        "verify_agilex_policy_artifacts",
        lambda _: {},
    )
    with pytest.raises(ValueError, match="safety.*hash"):
        load_agilex_policy_config(path)


def test_artifact_recheck_rejects_normalizer_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, identities = _policy_fixture(tmp_path)
    loaded = _load_without_deep_artifact_scan(monkeypatch, path)
    (loaded.serve_bundle / "normalizer.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="normalizer.*identity"):
        _verify(loaded, identities)


def test_artifact_recheck_rejects_component_retarget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, identities = _policy_fixture(tmp_path)
    loaded = _load_without_deep_artifact_scan(monkeypatch, path)
    link = loaded.serve_bundle / "transformer"
    link.unlink()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    link.symlink_to(replacement)

    with pytest.raises(ValueError, match="component link changed"):
        _verify(loaded, identities)


def test_artifact_recheck_rejects_untrained_task_contact_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, identities = _policy_fixture(tmp_path)
    loaded = _load_without_deep_artifact_scan(monkeypatch, path)
    untrained = AgileXTaskRoute(
        task_id="wipe",
        prompt="wipe the table",
        tactile_required=True,
        wrench_required=True,
        tactile_keys=("observation.images.unknown_tactile",),
        wrench_keys=("observation.wrench.left",),
        contract_sha256="a" * 64,
    )
    changed_policy = replace(loaded.policy, task_routes={"wipe": untrained})
    changed = replace(loaded, policy=changed_policy)

    with pytest.raises(ValueError, match="tactile route was not trained"):
        _verify(changed, identities)


def test_production_artifact_recheck_runs_full_serve_bundle_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, _ = _policy_fixture(tmp_path)
    loaded = _load_without_deep_artifact_scan(monkeypatch, path)

    def reject_unsealed_bundle(*_: object, **__: object) -> dict[str, object]:
        raise ValueError("full qpos14 bundle audit rejected")

    monkeypatch.setattr(
        "n0_twam.integrations.worldarena.agilex_serve_bundle."
        "verify_agilex_serve_bundle",
        reject_unsealed_bundle,
    )
    with pytest.raises(ValueError, match="full qpos14 bundle audit rejected"):
        verify_artifacts(loaded)


def test_template_is_explicitly_non_runnable() -> None:
    template = agilex_policy_config_template()
    assert template["tactile_profile"] == "vision_only"
    checkpoint_sha = template["checkpoint_identity_sha256"]
    assert isinstance(checkpoint_sha, str)
    assert checkpoint_sha.startswith("REPLACE_")
    assert template["task_routes"] == {}
