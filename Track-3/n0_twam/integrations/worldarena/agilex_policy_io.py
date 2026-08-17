# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict, content-addressed configuration for local AgileX inference.

This module only defines a local model boundary.  It deliberately contains no
organizer transport, robot driver, or official-evaluation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from n0_twam.tactile_profiles import VISION_ONLY

from .agilex_manifest import canonical_sha256, sha256_file
from .agilex_normalizer import AgileXQpos14Normalizer, load_agilex_normalizer
from .agilex_policy_contracts import AgileXPolicyConfig
from .agilex_policy_schema import (
    DEVICE,
    POLICY_CONFIG_SCHEMA_VERSION,
    POLICY_ID,
    PROFILES,
    exact_fields,
    load_json_file,
    parse_routes,
    parse_safety,
    positive_int,
    resolve_input,
    resolve_output,
    sha256,
)


@dataclass(frozen=True)
class AgileXDirectPolicyConfig:
    """Policy-core config plus immutable local serving identities."""

    policy: AgileXPolicyConfig
    source_path: Path
    policy_config_file_sha256: str
    config_contract_sha256: str
    serve_bundle: Path
    serve_output: Path
    serve_bundle_receipt_sha256: str
    serve_bundle_identity_sha256: str
    checkpoint_identity_sha256: str
    normalizer_file_sha256: str
    normalizer_contract_sha256: str
    source_manifest_sha256: str
    repo_route_manifest_file_sha256: str
    repo_route_manifest_sha256: str
    contact_profile_contract_sha256: str
    task_routes_sha256: str
    cuda_visible_device: str
    distributed_port: int
    video_inference_steps: int
    action_inference_steps: int
    normalizer: AgileXQpos14Normalizer


def load_agilex_policy_config(path: Path) -> AgileXDirectPolicyConfig:
    """Load and immediately verify one immutable local policy config."""

    raw_source = Path(path).expanduser()
    payload = load_json_file(raw_source, label="AgileX policy config")
    source = raw_source.resolve(strict=True)
    fields = {
        "schema_version",
        "policy_id",
        "tactile_profile",
        "serve_bundle",
        "serve_output",
        "serve_bundle_receipt_sha256",
        "serve_bundle_identity_sha256",
        "checkpoint_identity_sha256",
        "normalizer_file_sha256",
        "normalizer_contract_sha256",
        "source_manifest_sha256",
        "repo_route_manifest_file_sha256",
        "repo_route_manifest_sha256",
        "contact_profile_contract_sha256",
        "cuda_visible_device",
        "distributed_port",
        "episode_seed",
        "max_chunk_actions",
        "video_inference_steps",
        "action_inference_steps",
        "task_routes",
        "safety",
        "config_contract_sha256",
    }
    exact_fields(payload, fields, label="AgileX policy config")
    config_contract_sha = sha256(
        payload["config_contract_sha256"], label="policy config contract SHA256"
    )
    config_core = {
        key: value for key, value in payload.items() if key != "config_contract_sha256"
    }
    if canonical_sha256(config_core) != config_contract_sha:
        raise ValueError("AgileX policy config self hash mismatch")
    if payload["schema_version"] != POLICY_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported AgileX policy config schema")
    policy_id = payload["policy_id"]
    if not isinstance(policy_id, str) or not POLICY_ID.fullmatch(policy_id):
        raise ValueError("policy_id contains unsupported characters")
    profile = payload["tactile_profile"]
    if profile not in PROFILES:
        raise ValueError("unknown AgileX tactile profile")
    device = payload["cuda_visible_device"]
    if not isinstance(device, str) or not DEVICE.fullmatch(device):
        raise ValueError("cuda_visible_device must name one numeric device")
    seed = payload["episode_seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("episode_seed must be a non-negative integer")
    routes, routes_sha = parse_routes(payload["task_routes"])
    safety = parse_safety(payload["safety"])
    bundle = resolve_input(source, payload["serve_bundle"], label="serve_bundle")
    if not bundle.is_dir():
        raise ValueError("serve_bundle must be a directory")
    output = resolve_output(source, payload["serve_output"])
    if output == bundle or output in bundle.parents or bundle in output.parents:
        raise ValueError("serve_output and immutable serve_bundle must be disjoint")
    policy = AgileXPolicyConfig(
        policy_id=policy_id,
        tactile_profile=profile,
        task_routes=routes,
        episode_seed=seed,
        max_chunk_actions=positive_int(
            payload["max_chunk_actions"], label="max_chunk_actions", maximum=128
        ),
        safety=safety,
    )
    normalizer = load_agilex_normalizer(
        bundle / "normalizer.json",
        expected_file_sha256=sha256(
            payload["normalizer_file_sha256"], label="normalizer file SHA256"
        ),
        expected_source_manifest_sha256=sha256(
            payload["source_manifest_sha256"], label="source manifest SHA256"
        ),
        expected_repo_route_manifest_sha256=sha256(
            payload["repo_route_manifest_sha256"],
            label="route manifest contract SHA256",
        ),
    )
    normalizer_contract_sha = sha256(
        payload["normalizer_contract_sha256"], label="normalizer contract SHA256"
    )
    if normalizer.contract_sha256 != normalizer_contract_sha:
        raise ValueError("AgileX normalizer contract identity mismatch")
    loaded = AgileXDirectPolicyConfig(
        policy=policy,
        source_path=source,
        policy_config_file_sha256=sha256_file(source),
        config_contract_sha256=config_contract_sha,
        serve_bundle=bundle,
        serve_output=output,
        serve_bundle_receipt_sha256=sha256(
            payload["serve_bundle_receipt_sha256"], label="serve bundle receipt SHA256"
        ),
        serve_bundle_identity_sha256=sha256(
            payload["serve_bundle_identity_sha256"],
            label="serve bundle identity SHA256",
        ),
        checkpoint_identity_sha256=sha256(
            payload["checkpoint_identity_sha256"], label="checkpoint identity SHA256"
        ),
        normalizer_file_sha256=sha256(
            payload["normalizer_file_sha256"], label="normalizer file SHA256"
        ),
        normalizer_contract_sha256=normalizer_contract_sha,
        source_manifest_sha256=sha256(
            payload["source_manifest_sha256"], label="source manifest SHA256"
        ),
        repo_route_manifest_file_sha256=sha256(
            payload["repo_route_manifest_file_sha256"],
            label="route manifest file SHA256",
        ),
        repo_route_manifest_sha256=sha256(
            payload["repo_route_manifest_sha256"],
            label="route manifest contract SHA256",
        ),
        contact_profile_contract_sha256=sha256(
            payload["contact_profile_contract_sha256"],
            label="contact profile contract SHA256",
        ),
        task_routes_sha256=routes_sha,
        cuda_visible_device=device,
        distributed_port=positive_int(
            payload["distributed_port"], label="distributed_port", maximum=65535
        ),
        video_inference_steps=positive_int(
            payload["video_inference_steps"], label="video_inference_steps", maximum=100
        ),
        action_inference_steps=positive_int(
            payload["action_inference_steps"],
            label="action_inference_steps",
            maximum=100,
        ),
        normalizer=normalizer,
    )
    verify_agilex_policy_artifacts(loaded)
    return loaded


def verify_agilex_policy_artifacts(
    config: AgileXDirectPolicyConfig,
) -> dict[str, object]:
    """Recheck all small identity-bearing artifacts before model allocation."""

    from .agilex_policy_artifacts import verify_agilex_policy_artifacts as verify

    return dict(verify(config))


def agilex_policy_config_template() -> dict[str, object]:
    """Return a deliberately non-runnable, calibration-explicit template."""

    replace = "REPLACE_WITH_64_LOWERCASE_HEX"
    core: dict[str, object] = {
        "schema_version": POLICY_CONFIG_SCHEMA_VERSION,
        "policy_id": "n0-twam-agilex",
        "tactile_profile": VISION_ONLY,
        "serve_bundle": "./serve-bundle",
        "serve_output": "./serve-output",
        "serve_bundle_receipt_sha256": replace,
        "serve_bundle_identity_sha256": replace,
        "checkpoint_identity_sha256": "REPLACE_CHECKPOINT_IDENTITY_SHA256",
        "normalizer_file_sha256": replace,
        "normalizer_contract_sha256": replace,
        "source_manifest_sha256": replace,
        "repo_route_manifest_file_sha256": replace,
        "repo_route_manifest_sha256": replace,
        "contact_profile_contract_sha256": replace,
        "cuda_visible_device": "0",
        "distributed_port": 29643,
        "episode_seed": 20260811,
        "max_chunk_actions": 12,
        "video_inference_steps": 3,
        "action_inference_steps": 4,
        "task_routes": {},
        "safety": {
            "lower_bounds": ["CALIBRATE"] * 14,
            "upper_bounds": ["CALIBRATE"] * 14,
            "max_step_per_second": ["CALIBRATE"] * 14,
            "min_execution_dt_s": "CALIBRATE",
            "max_execution_dt_s": "CALIBRATE",
            "max_state_age_s": "CALIBRATE",
            "max_inference_latency_s": "CALIBRATE",
            "contract_sha256": replace,
        },
    }
    return {**core, "config_contract_sha256": canonical_sha256(core)}


__all__ = (
    "AgileXDirectPolicyConfig",
    "agilex_policy_config_template",
    "load_agilex_policy_config",
    "verify_agilex_policy_artifacts",
)
