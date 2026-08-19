# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed preflight for official Franka Track 3.2 post-training."""

from __future__ import annotations

import json
import os
import socket
import stat
from collections.abc import Mapping
from pathlib import Path

from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
    validate_sha256,
)
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.configs.twam_track32_franka_cfg import twam_track32_franka_cfg
from n0_twam.integrations.worldarena.franka_actions import (
    DERIVED_ACTION_SCHEMA,
    TRACK32_PROFILE_ID,
)
from n0_twam.integrations.worldarena.franka_artifacts import (
    MODEL_ACTION_SCHEMA,
    SOURCE_ACTION_SCHEMA,
    verify_franka_training_artifacts,
)
from n0_twam.integrations.worldarena.franka_manifest import sha256_file


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required environment variable is unset: {name}")
    return value


def _required_path(name: str) -> Path:
    path = Path(_required_env(name)).expanduser().resolve(strict=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{name} may not be a symlink")
    return path


def _required_sha(name: str) -> str:
    return validate_sha256(_required_env(name), label=name)


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _require_hash(path: Path, expected: str, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA256 differs from the request")


def _require_config_contract(config: object) -> None:
    expected: dict[str, object] = {
        "dataset_adapter": "worldarena_franka_ee10",
        "track32_profile_id": TRACK32_PROFILE_ID,
        "action_schema": MODEL_ACTION_SCHEMA,
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "derived_action_schema": DERIVED_ACTION_SCHEMA,
        "accelerator_profile": _required_env("N0_TRACK32_ACCELERATOR_PROFILE"),
        "action_dim": 20,
        "action_per_frame": 6,
        "tactile_profile": "vision_only",
        "tactile_mode": "disabled",
        "use_local_tactile": False,
        "use_contact_gate": False,
        "tactile_diffusion_loss_weight": 0.0,
        "freeze_tactile_parameters": True,
        "strict_training_resume": True,
    }
    mismatches = {
        field: (getattr(config, field, None), wanted)
        for field, wanted in expected.items()
        if getattr(config, field, None) != wanted
    }
    if mismatches:
        raise ValueError(f"Franka config contract mismatch: {mismatches}")
    if list(getattr(config, "used_action_channel_ids", [])) != list(range(10)):
        raise ValueError("Franka active action channels must be exactly 0..9")
    if list(getattr(config, "tactile_keys", [])):
        raise ValueError("Franka vision-only config must declare tactile_keys=[]")
    if tuple(getattr(config, "mot_cross_attn_experts", ())) != (
        "video",
        "action",
    ):
        raise ValueError("Franka cross-attention experts must exclude tactile")


def _require_recipe(config: object) -> None:
    expected = {
        "run_role": _required_env("N0_TRACK32_RUN_ROLE"),
        "num_steps": int(_required_env("N0_TRACK32_NUM_STEPS")),
        "stop_after_step": int(_required_env("N0_TRACK32_STOP_AFTER_STEP")),
        "save_interval": int(_required_env("N0_TRACK32_SAVE_INTERVAL")),
        "val_interval": int(_required_env("N0_TRACK32_VAL_INTERVAL")),
        "batch_size": int(_required_env("N0_TRACK32_BATCH_SIZE")),
        "gradient_accumulation_steps": int(
            _required_env("N0_TRACK32_GRADIENT_ACCUMULATION_STEPS")
        ),
        "max_latent_frames": int(_required_env("N0_TRACK32_MAX_LATENT_FRAMES")),
        "seed": int(_required_env("N0_TRACK32_SEED")),
    }
    mismatches = {
        field: (getattr(config, field, None), wanted)
        for field, wanted in expected.items()
        if getattr(config, field, None) != wanted
    }
    if mismatches:
        raise ValueError(f"Franka recipe differs from request: {mismatches}")
    expected_world = int(_required_env("N0_TRACK32_EXPECTED_WORLD_SIZE"))
    visible = tuple(
        value for value in _required_env("CUDA_VISIBLE_DEVICES").split(",") if value
    )
    if expected_world <= 0 or len(visible) != expected_world:
        raise ValueError("visible device count differs from expected world size")


def _require_accelerator_profile() -> None:
    profile = _required_env("N0_TRACK32_ACCELERATOR_PROFILE")
    expected = {
        "hcu_performance": {
            "N0_FLEX_ATTENTION_BACKEND": "grouped_flash_attn",
            "N0_MOT_CROSS_ATTENTION_BACKEND": "flash_attn",
            "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "1",
            "N0_FSDP_REDUCE_DTYPE": "bfloat16",
            "N0_FSDP_EXPERT_RESHARD_POLICY": "after_layer",
            "N0_MOT_ACTIVATION_CHECKPOINTING": "0",
        },
        "portable": {
            "N0_FLEX_ATTENTION_BACKEND": "grouped_sdpa",
            "N0_MOT_CROSS_ATTENTION_BACKEND": "sdpa",
            "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "0",
            "N0_FSDP_REDUCE_DTYPE": "float32",
            "N0_FSDP_EXPERT_RESHARD_POLICY": "after_call",
            "N0_MOT_ACTIVATION_CHECKPOINTING": "1",
        },
    }
    if profile not in expected:
        raise ValueError("unsupported Track 3.2 accelerator profile")
    mismatches = {
        name: (os.environ.get(name), value)
        for name, value in expected[profile].items()
        if os.environ.get(name) != value
    }
    if mismatches:
        raise ValueError(f"accelerator profile environment mismatch: {mismatches}")
    network_names = (
        "N0_TRACK32_COLLECTIVE_NETWORK_INTERFACE",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
    )
    if profile == "portable":
        leaked = {
            name: os.environ.get(name) for name in network_names if name in os.environ
        }
        if leaked:
            raise ValueError(
                f"portable profile inherited HCU network bindings: {leaked}"
            )
        return
    interface = _required_env("N0_TRACK32_COLLECTIVE_NETWORK_INTERFACE")
    socket_bindings = {
        "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
        "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME"),
    }
    if any(value != interface for value in socket_bindings.values()):
        raise ValueError(f"collective socket interface mismatch: {socket_bindings}")
    try:
        available = frozenset(name for _, name in socket.if_nameindex())
    except OSError as error:
        raise RuntimeError("unable to enumerate collective interfaces") from error
    if interface not in available:
        raise ValueError(
            f"collective interface is unavailable in this container: {interface}"
        )


def _require_initial_checkpoint(path: Path, expected_sha256: str) -> dict[str, object]:
    transformer = path / "transformer"
    weights = transformer / TRANSFORMER_WEIGHTS_FILENAME
    identity = audit_transformer_checkpoint(weights, expected_action_dim=20)
    if identity["sha256"] != expected_sha256:
        raise ValueError("initial transformer SHA256 differs from the request")
    config = _json_object(
        transformer / "config.json", label="initial transformer config"
    )
    if config.get("action_dim") != 20:
        raise ValueError("initial transformer does not retain the released 20D head")
    action_schema = config.get("action_schema")
    if action_schema not in (None, MODEL_ACTION_SCHEMA):
        raise ValueError("initial transformer declares an incompatible action schema")
    return {"mode": "init", "transformer_identity": identity}


def _require_resume_checkpoint(path: Path, expected_identity: str) -> dict[str, object]:
    snapshot = capture_strict_checkpoint_snapshot(path)
    identity = build_strict_checkpoint_identity(snapshot)
    if identity.get("identity_sha256") != expected_identity:
        raise ValueError("resume checkpoint identity differs from the request")
    if snapshot.transformer_config.get("action_schema") != MODEL_ACTION_SCHEMA:
        raise ValueError("resume checkpoint action schema is not ee20_absee")
    meta = snapshot.train_meta
    expected_meta: dict[str, object] = {
        "track32_profile_id": TRACK32_PROFILE_ID,
        "action_schema": MODEL_ACTION_SCHEMA,
        "action_dim": 20,
        "action_per_frame": 6,
        "used_action_channel_ids": list(range(10)),
        "tactile_mode": "disabled",
        "tactile_keys": [],
        "use_local_tactile": False,
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "derived_action_schema": DERIVED_ACTION_SCHEMA,
        "accelerator_profile": _required_env("N0_TRACK32_ACCELERATOR_PROFILE"),
    }
    mismatches = {
        field: (meta.get(field), wanted)
        for field, wanted in expected_meta.items()
        if meta.get(field) != wanted
    }
    trainability = meta.get("trainability_contract")
    if not isinstance(trainability, Mapping):
        raise ValueError("resume checkpoint lacks a trainability contract")
    if (
        trainability.get("policy") != "freeze_tactile_only_v1"
        or trainability.get("tactile_mode") != "disabled"
        or trainability.get("tactile_profile", "vision_only") != "vision_only"
        or int(trainability.get("frozen_parameter_count", 0)) <= 0
    ):
        raise ValueError("resume checkpoint did not freeze tactile-only parameters")
    if mismatches:
        raise ValueError(f"resume Franka metadata mismatch: {mismatches}")
    return {
        "mode": "resume",
        "checkpoint_identity": identity,
        "start_step": snapshot.step,
    }


def run_preflight() -> dict[str, object]:
    """Audit all request-bound inputs before distributed model loading."""

    artifact_root = _required_path("N0_TRACK32_ARTIFACT_ROOT")
    lerobot_root = _required_path("N0_TRACK32_LEROBOT_ROOT")
    base_model = _required_path("N0_BASE_MODEL")
    empty_embedding = _required_path("N0_EMPTY_EMBEDDING")
    if not empty_embedding.is_file():
        raise ValueError("N0_EMPTY_EMBEDDING must name a regular file")
    _require_hash(
        empty_embedding,
        _required_sha("N0_EMPTY_EMBEDDING_SHA256"),
        label="empty embedding",
    )
    _require_config_contract(twam_track32_franka_cfg)
    _require_recipe(twam_track32_franka_cfg)
    _require_accelerator_profile()

    artifacts = verify_franka_training_artifacts(
        artifact_root=artifact_root,
        lerobot_root=lerobot_root,
        base_model=base_model,
        run_role=_required_env("N0_TRACK32_RUN_ROLE"),
    )
    expected_artifacts = {
        "prepare receipt file": (
            artifacts.prepare_receipt_sha256,
            _required_sha("N0_TRACK32_PREPARE_RECEIPT_SHA256"),
        ),
        "conversion report file": (
            artifacts.conversion_report_sha256,
            _required_sha("N0_TRACK32_CONVERSION_REPORT_SHA256"),
        ),
        "latent inventory file": (
            artifacts.latent_inventory_sha256,
            _required_sha("N0_TRACK32_LATENT_INVENTORY_FILE_SHA256"),
        ),
        "training view": (
            artifacts.train_view.view_sha256,
            _required_sha("N0_TRACK32_TRAIN_VIEW_SHA256"),
        ),
        "normalizer": (
            artifacts.normalizer.get("normalizer_sha256"),
            _required_sha("N0_TRACK32_NORMALIZER_SHA256"),
        ),
    }
    validation_expected = os.environ.get("N0_TRACK32_VALIDATION_VIEW_SHA256")
    if artifacts.validation_view is None:
        if validation_expected:
            raise ValueError("final_refit may not declare a validation view")
    else:
        if artifacts.validation_view.view_sha256 != validate_sha256(
            validation_expected,
            label="N0_TRACK32_VALIDATION_VIEW_SHA256",
        ):
            raise ValueError("validation view differs from the request")
    mismatches = [
        label
        for label, (actual, expected) in expected_artifacts.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("Franka artifact identity mismatch: " + ", ".join(mismatches))

    init_raw = os.environ.get("N0_TRACK32_INIT_FROM")
    resume_raw = os.environ.get("N0_TRACK32_RESUME_FROM")
    if bool(init_raw) == bool(resume_raw):
        raise ValueError("exactly one Franka init/resume checkpoint is required")
    if init_raw:
        checkpoint = _require_initial_checkpoint(
            Path(init_raw).expanduser().resolve(strict=True),
            _required_sha("N0_TRACK32_INIT_TRANSFORMER_SHA256"),
        )
    else:
        checkpoint = _require_resume_checkpoint(
            Path(str(resume_raw)).expanduser().resolve(strict=True),
            _required_sha("N0_TRACK32_RESUME_CHECKPOINT_IDENTITY_SHA256"),
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "profile": TRACK32_PROFILE_ID,
        "run_role": _required_env("N0_TRACK32_RUN_ROLE"),
        "world_size": int(_required_env("N0_TRACK32_EXPECTED_WORLD_SIZE")),
        "accelerator_profile": _required_env("N0_TRACK32_ACCELERATOR_PROFILE"),
        "tactile_profile": "vision_only",
        "tactile_mode": "disabled",
        "action_route": "end_pose_base8_xyzw_to_ee10_to_ee20_mask_0_9",
        "train_view_sha256": artifacts.train_view.view_sha256,
        "validation_view_sha256": (
            None
            if artifacts.validation_view is None
            else artifacts.validation_view.view_sha256
        ),
        "normalizer_sha256": artifacts.normalizer["normalizer_sha256"],
        "latent_inventory_sha256": artifacts.latent_inventory["inventory_sha256"],
        "checkpoint": checkpoint,
    }


def main() -> int:
    print(json.dumps(run_preflight(), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
