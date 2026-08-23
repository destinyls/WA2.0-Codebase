# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Immutable configuration contract for the WorldArena Franka Policy."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .franka_live_contract import FrankaLiveContractConfig, parse_live_contract_config

POLICY_CONFIG_SCHEMA_VERSION = 4
TRAINING_ALIGNED_POLICY_CONFIG_SCHEMA_VERSION = 3
LEGACY_POLICY_CONFIG_SCHEMA_VERSION = 2

_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DEVICE_PATTERN = re.compile(r"^[0-9]+$")


@dataclass(frozen=True)
class FrankaSafetyConfig:
    """Signed workspace, pose-step, and gripper limits."""

    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    max_translation_step_m: float
    max_rotation_step_rad: float
    gripper_min: float
    gripper_max: float
    max_gripper_step: float


@dataclass(frozen=True)
class FrankaPolicyConfig:
    """Complete immutable identity and execution configuration."""

    policy_id: str
    serve_bundle: Path
    serve_bundle_receipt_sha256: str
    serve_output: Path
    cuda_visible_device: str
    distributed_port: int
    episode_seed: int
    max_chunk_actions: int
    external_chunk_actions: int
    video_inference_steps: int
    action_inference_steps: int
    safety: FrankaSafetyConfig
    live_contract: FrankaLiveContractConfig | None = None


def mapping(value: object, *, label: str) -> Mapping[str, object]:
    """Require a JSON-style mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(payload: Mapping[str, object], fields: set[str], *, label: str) -> None:
    if set(payload) != fields:
        raise ValueError(
            f"invalid {label}: missing={sorted(fields - set(payload))}, "
            f"unexpected={sorted(set(payload) - fields)}"
        )


def _positive_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _vector3(value: object, *, label: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain three finite values")
    return tuple(float(item) for item in array)


def load_franka_policy_config(path: Path) -> FrankaPolicyConfig:
    """Load and validate a fail-closed Policy configuration."""

    source = Path(path).expanduser().resolve(strict=True)
    payload = mapping(json.loads(source.read_text(encoding="utf-8")), label="policy")
    fields = {
        "schema_version",
        "policy_id",
        "serve_bundle",
        "serve_bundle_receipt_sha256",
        "serve_output",
        "cuda_visible_device",
        "distributed_port",
        "episode_seed",
        "max_chunk_actions",
        "external_chunk_actions",
        "video_inference_steps",
        "action_inference_steps",
        "safety",
    }
    schema_version = payload.get("schema_version")
    if schema_version in {
        POLICY_CONFIG_SCHEMA_VERSION,
        TRAINING_ALIGNED_POLICY_CONFIG_SCHEMA_VERSION,
    }:
        fields.add("live_contract")
    elif schema_version != LEGACY_POLICY_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported Franka policy config schema")
    _exact(payload, fields, label="Franka policy config")

    policy_id = payload.get("policy_id")
    if not isinstance(policy_id, str) or not re.fullmatch(
        r"[A-Za-z0-9._-]{1,64}", policy_id
    ):
        raise ValueError("policy_id contains unsupported characters")
    device = payload.get("cuda_visible_device")
    if not isinstance(device, str) or not _DEVICE_PATTERN.fullmatch(device):
        raise ValueError("cuda_visible_device must name exactly one numeric device")
    receipt_sha = payload.get("serve_bundle_receipt_sha256")
    if not isinstance(receipt_sha, str) or not _SHA_PATTERN.fullmatch(receipt_sha):
        raise ValueError("serve_bundle_receipt_sha256 must be lowercase SHA256")

    port = payload.get("distributed_port")
    seed = payload.get("episode_seed")
    chunk = payload.get("max_chunk_actions")
    external_chunk = payload.get("external_chunk_actions")
    video_steps = payload.get("video_inference_steps")
    action_steps = payload.get("action_inference_steps")
    for value, label, maximum in (
        (port, "distributed_port", 65535),
        (chunk, "max_chunk_actions", 128),
        (external_chunk, "external_chunk_actions", 128),
        (video_steps, "video_inference_steps", 100),
        (action_steps, "action_inference_steps", 100),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise ValueError(f"{label} must be an integer in [1,{maximum}]")
    if chunk != 12:
        raise ValueError(
            "max_chunk_actions must be 12 for frame_chunk_size=2 and "
            "action_per_frame=6"
        )
    if external_chunk not in {1, 6}:
        raise ValueError(
            "external_chunk_actions must be 1 for per-observation replanning or "
            "6 for one future-action batch followed by fresh replanning"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("episode_seed must be a non-negative integer")

    safety_raw = mapping(payload.get("safety"), label="safety")
    _exact(
        safety_raw,
        {
            "workspace_min",
            "workspace_max",
            "max_translation_step_m",
            "max_rotation_step_rad",
            "gripper_min",
            "gripper_max",
            "max_gripper_step",
        },
        label="safety",
    )
    workspace_min = _vector3(safety_raw.get("workspace_min"), label="workspace_min")
    workspace_max = _vector3(safety_raw.get("workspace_max"), label="workspace_max")
    if any(low >= high for low, high in zip(workspace_min, workspace_max, strict=True)):
        raise ValueError("each workspace_min coordinate must be below workspace_max")
    gripper_min = float(safety_raw.get("gripper_min"))
    gripper_max = float(safety_raw.get("gripper_max"))
    if not np.isfinite((gripper_min, gripper_max)).all() or gripper_min >= gripper_max:
        raise ValueError("gripper limits are invalid")

    bundle_raw = Path(str(payload["serve_bundle"])).expanduser()
    serve_bundle = (
        bundle_raw.resolve(strict=True)
        if bundle_raw.is_absolute()
        else (source.parent / bundle_raw).resolve(strict=True)
    )
    output_raw = Path(str(payload["serve_output"])).expanduser()
    serve_output = (
        output_raw if output_raw.is_absolute() else source.parent / output_raw
    ).resolve(strict=False)
    if (
        serve_output == serve_bundle
        or serve_bundle in serve_output.parents
        or serve_output in serve_bundle.parents
    ):
        raise ValueError("serve_output and immutable serve bundle must be disjoint")

    live_contract = (
        None
        if schema_version == LEGACY_POLICY_CONFIG_SCHEMA_VERSION
        else parse_live_contract_config(
            payload.get("live_contract"),
            require_execution_fields=(schema_version == POLICY_CONFIG_SCHEMA_VERSION),
        )
    )
    if schema_version == POLICY_CONFIG_SCHEMA_VERSION:
        if live_contract is None:
            raise ValueError("schema-v4 requires a strict live contract")
        if external_chunk != live_contract.actions_per_replan:
            raise ValueError(
                "schema-v4 external_chunk_actions must equal actions_per_replan"
            )

    return FrankaPolicyConfig(
        policy_id=policy_id,
        serve_bundle=serve_bundle,
        serve_bundle_receipt_sha256=receipt_sha,
        serve_output=serve_output,
        cuda_visible_device=device,
        distributed_port=int(port),
        episode_seed=int(seed),
        max_chunk_actions=int(chunk),
        external_chunk_actions=int(external_chunk),
        video_inference_steps=int(video_steps),
        action_inference_steps=int(action_steps),
        safety=FrankaSafetyConfig(
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            max_translation_step_m=_positive_number(
                safety_raw.get("max_translation_step_m"),
                label="max_translation_step_m",
            ),
            max_rotation_step_rad=_positive_number(
                safety_raw.get("max_rotation_step_rad"),
                label="max_rotation_step_rad",
            ),
            gripper_min=gripper_min,
            gripper_max=gripper_max,
            max_gripper_step=_positive_number(
                safety_raw.get("max_gripper_step"), label="max_gripper_step"
            ),
        ),
        live_contract=live_contract,
    )


def franka_policy_config_template() -> dict[str, object]:
    """Return the non-runnable schema-v4 configuration template."""

    return {
        "schema_version": POLICY_CONFIG_SCHEMA_VERSION,
        "policy_id": "n0-twam-franka",
        "serve_bundle": "./serve-bundle",
        "serve_bundle_receipt_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        "serve_output": "./serve-output",
        "cuda_visible_device": "0",
        "distributed_port": 29642,
        "episode_seed": 20260810,
        "max_chunk_actions": 12,
        "external_chunk_actions": 6,
        "video_inference_steps": 3,
        "action_inference_steps": 4,
        "live_contract": {
            "conditioning_mode": "training_aligned_rgb_replan_v1",
            "target_fps": 10,
            "frame_interval_tolerance_ms": 15.0,
            "max_state_image_skew_ms": 50.0,
            "max_history_frames": 5,
            "action_hz": 15,
            "actions_per_replan": 6,
            "future_start_index": 6,
            "require_dual_camera_history": True,
            "require_fresh_observation_after_chunk": True,
        },
        "safety": {
            "workspace_min": ["CALIBRATE_X_MIN", "CALIBRATE_Y_MIN", "CALIBRATE_Z_MIN"],
            "workspace_max": ["CALIBRATE_X_MAX", "CALIBRATE_Y_MAX", "CALIBRATE_Z_MAX"],
            "max_translation_step_m": "CALIBRATE_TRANSLATION_STEP",
            "max_rotation_step_rad": "CALIBRATE_ROTATION_STEP",
            "gripper_min": "CALIBRATE_GRIPPER_MIN",
            "gripper_max": "CALIBRATE_GRIPPER_MAX",
            "max_gripper_step": "CALIBRATE_GRIPPER_STEP",
        },
    }


__all__ = (
    "LEGACY_POLICY_CONFIG_SCHEMA_VERSION",
    "POLICY_CONFIG_SCHEMA_VERSION",
    "TRAINING_ALIGNED_POLICY_CONFIG_SCHEMA_VERSION",
    "FrankaPolicyConfig",
    "FrankaSafetyConfig",
    "franka_policy_config_template",
    "load_franka_policy_config",
    "mapping",
)
