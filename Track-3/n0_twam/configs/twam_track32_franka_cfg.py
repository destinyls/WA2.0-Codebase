# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Official WorldArena Track 3.2 Franka vision-only post-training config.

The verified wire action is an 8D base-frame end pose
``[x, y, z, qx, qy, qz, qw, gripper]``.  Conversion maps it to one EE10 arm
(``xyz + rot6d + gripper``), embeds that arm in the released 20D action head,
and masks channels 10..19.  Tactile data and tactile loss are disabled
explicitly; the released tactile weights remain frozen and byte-preserved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from easydict import EasyDict

from n0_twam.integrations.worldarena.franka_actions import (
    DERIVED_ACTION_SCHEMA,
    FRANKA_ACTION_SCHEMA,
    TRACK32_PROFILE_ID,
)
from n0_twam.integrations.worldarena.franka_views import (
    DEVELOPMENT_TRAIN_VIEW,
    DEVELOPMENT_VALIDATION_VIEW,
    FINAL_REFIT_VIEW,
)
from n0_twam.tactile_profiles import VISION_ONLY, validate_tactile_profile_config

from .twam_base_cfg import twam_base_cfg


def _path_env(name: str, fallback: Path | str) -> Path:
    return Path(os.environ.get(name, str(fallback))).expanduser()


def _positive_int_env(name: str, fallback: int) -> int:
    raw = os.environ.get(name, str(fallback))
    if not raw.isdecimal() or int(raw) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return int(raw)


def _nonnegative_int_env(name: str, fallback: int) -> int:
    raw = os.environ.get(name, str(fallback))
    if not raw.isdecimal():
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return int(raw)


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _resolve_role() -> str:
    role = os.environ.get("N0_TRACK32_RUN_ROLE", "development")
    if role not in {"development", "final_refit"}:
        raise ValueError("N0_TRACK32_RUN_ROLE must be development or final_refit")
    return role


def _resolve_stop_after(num_steps: int) -> object:
    raw = os.environ.get("N0_TRACK32_STOP_AFTER_STEP")
    if raw is None:
        return num_steps
    return int(raw) if raw.isdecimal() else raw


_ROLE = _resolve_role()
_ARTIFACT_ROOT = _path_env("N0_TRACK32_ARTIFACT_ROOT", "/path/to/franka/artifacts")
_LEROBOT_ROOT = _path_env("N0_TRACK32_LEROBOT_ROOT", "/path/to/franka/lerobot")
_BASE_MODEL = _path_env("N0_BASE_MODEL", "/path/to/n0-twam-base")
_EMPTY_EMBEDDING = _path_env("N0_EMPTY_EMBEDDING", _BASE_MODEL / "empty_emb.pt")
_TRAIN_VIEW_ID = DEVELOPMENT_TRAIN_VIEW if _ROLE == "development" else FINAL_REFIT_VIEW
_VALIDATION_VIEW_ID = DEVELOPMENT_VALIDATION_VIEW if _ROLE == "development" else None
_NORMALIZER_VIEW_ID = _TRAIN_VIEW_ID
_NORMALIZER_PATH = _path_env(
    "N0_TRACK32_NORMALIZER_PATH",
    _ARTIFACT_ROOT / "normalizers" / f"{_NORMALIZER_VIEW_ID}.json",
)
_RESUME_FROM_RAW = os.environ.get("N0_TRACK32_RESUME_FROM")
_INIT_FROM_RAW = os.environ.get("N0_TRACK32_INIT_FROM")


def _artifact_identity_from_environment() -> dict[str, object] | None:
    names = {
        "prepare_receipt_file_sha256": "N0_TRACK32_PREPARE_RECEIPT_SHA256",
        "conversion_report_file_sha256": "N0_TRACK32_CONVERSION_REPORT_SHA256",
        "latent_inventory_file_sha256": ("N0_TRACK32_LATENT_INVENTORY_FILE_SHA256"),
        "train_view_sha256": "N0_TRACK32_TRAIN_VIEW_SHA256",
        "normalizer_sha256": "N0_TRACK32_NORMALIZER_SHA256",
    }
    present = {field: os.environ.get(name) for field, name in names.items()}
    validation = os.environ.get("N0_TRACK32_VALIDATION_VIEW_SHA256")
    if not any(value for value in (*present.values(), validation)):
        return None
    if any(not value for value in present.values()):
        raise ValueError("Track 3.2 artifact identity environment is incomplete")
    return {
        "schema_version": 1,
        "profile": TRACK32_PROFILE_ID,
        "run_role": _ROLE,
        "source_records_sha256": (
            "67118a93230e13a5ecf8072df9cad4b30882367471017b4f1b49e43b6c8d4635"
        ),
        "train_view_id": _TRAIN_VIEW_ID,
        "validation_view_id": _VALIDATION_VIEW_ID,
        "validation_view_sha256": validation,
        **{field: str(value) for field, value in present.items()},
    }


cfg = EasyDict(twam_base_cfg.copy())
cfg.__name__ = "Config: N0-TWAM official Franka Track 3.2 vision-only"

# Immutable converted data and views.
cfg.dataset_adapter = "worldarena_franka_ee10"
cfg.dataset_path = str(_LEROBOT_ROOT / "all600")
cfg.val_dataset_path = (
    str(_LEROBOT_ROOT / "all600") if _VALIDATION_VIEW_ID is not None else None
)
cfg.lerobot_root = str(_LEROBOT_ROOT)
cfg.artifact_root = str(_ARTIFACT_ROOT)
cfg.conversion_report_path = str(_ARTIFACT_ROOT / "conversion_report.json")
cfg.prepare_receipt_path = str(_ARTIFACT_ROOT / "prepare_receipt.json")
cfg.train_view_id = _TRAIN_VIEW_ID
cfg.dataset_view_path = str(_ARTIFACT_ROOT / "views" / f"{_TRAIN_VIEW_ID}.json")
cfg.validation_view_id = _VALIDATION_VIEW_ID
cfg.val_dataset_view_path = (
    None
    if _VALIDATION_VIEW_ID is None
    else str(_ARTIFACT_ROOT / "views" / f"{_VALIDATION_VIEW_ID}.json")
)
cfg.normalizer_source_view_id = _NORMALIZER_VIEW_ID
cfg.normalizer_source_view_path = str(
    _ARTIFACT_ROOT / "views" / f"{_NORMALIZER_VIEW_ID}.json"
)
cfg.run_role = _ROLE
cfg.accelerator_profile = os.environ.get("N0_TRACK32_ACCELERATOR_PROFILE", "portable")
if cfg.accelerator_profile not in {"portable", "hcu_performance"}:
    raise ValueError("invalid N0_TRACK32_ACCELERATOR_PROFILE")
cfg.training_profile_id = None
cfg.track32_profile_id = TRACK32_PROFILE_ID
cfg.track32_artifact_identity = _artifact_identity_from_environment()
cfg.capture_runtime_provenance = True
cfg.sampler_coverage_mode = "pad_global"
cfg.sampler_rank_alignment = "contiguous"
cfg.crop_window_policy_id = "epoch_content_addressed_crop_v1"

# Two official RGB streams. No synthetic or zero-image tactile surrogate is used.
cfg.obs_cam_keys = ["observation.images.top", "observation.images.wrist_l"]
cfg.per_repo_obs_cam_keys = {}
cfg.tactile_profile = VISION_ONLY
cfg.tactile_mode = "disabled"
cfg.tactile_keys = []
cfg.per_repo_tactile_keys = {}
cfg.tactile_optional = False
cfg.synthetic_tactile_data = False
cfg.tactile_sensor_id_map = {}
cfg.active_tactile_sensor_count = 0
cfg.use_local_tactile = False
cfg.use_contact_gate = False
cfg.tactile_global_zero = False
cfg.tactile_cfg_prob = 0.0
cfg.noisy_cond_prob_tactile = 0.0
cfg.tactile_diffusion_loss_weight = 0.0
cfg.freeze_tactile_parameters = True

# Keep the released dual-arm 20D head intact. Only the left/first EE10 is active.
cfg.action_schema = "ee20_absee"
cfg.source_action_schema = FRANKA_ACTION_SCHEMA
cfg.derived_action_schema = DERIVED_ACTION_SCHEMA
cfg.action_dim = 20
cfg.action_delta_mode = "none"
cfg.action_per_frame = 6
cfg.used_action_channel_ids = list(range(10))
cfg.per_repo_used_action_channel_ids = {}
cfg.inverse_used_action_channel_ids = list(range(10)) + [10] * 10
cfg.action_norm_method = "q01q99"

if _NORMALIZER_PATH.is_file():
    _normalizer = _load_json(_NORMALIZER_PATH, label="Franka EE20 normalizer")
    cfg.norm_stat = {
        "q01": list(_normalizer["action_q01"]),
        "q99": list(_normalizer["action_q99"]),
    }
    cfg.normalizer_sha256 = str(_normalizer["normalizer_sha256"])
    cfg.norm_stat_path = str(_NORMALIZER_PATH)
else:
    cfg.norm_stat = {"q01": [-1.0] * 20, "q99": [1.0] * 20}
    cfg.normalizer_sha256 = None
    cfg.norm_stat_path = f"{_NORMALIZER_PATH} (MISSING)"

# The released N0-TWAM checkpoint already has the correct 20D EE head.  Its
# historical config lacks action_schema, so adoption is allowed only after the
# request-bound transformer SHA is verified by load_mot_checkpoint.
cfg.wan22_pretrained_model_name_or_path = str(_BASE_MODEL)
cfg.empty_emb_path = str(_EMPTY_EMBEDDING)
cfg.resume_from = (
    None if not _RESUME_FROM_RAW else str(Path(_RESUME_FROM_RAW).expanduser())
)
cfg.init_from = (
    None
    if cfg.resume_from
    else str(Path(_INIT_FROM_RAW).expanduser() if _INIT_FROM_RAW else _BASE_MODEL)
)
cfg.checkpoint_compatibility = "strict"
cfg.checkpoint_source_action_dim = 20
cfg.checkpoint_source_action_schema = "ee20_absee"
cfg.adopt_missing_action_schema = cfg.resume_from is None
cfg.expected_init_transformer_sha256 = os.environ.get(
    "N0_TRACK32_INIT_TRANSFORMER_SHA256"
)
cfg.strict_training_resume = True
cfg.inherit_action_migration_report = False
cfg.training_lineage = {
    "schema_version": 1,
    "profile": cfg.track32_profile_id,
    "source_action_schema": cfg.source_action_schema,
    "derived_action_schema": cfg.derived_action_schema,
    "target_action_schema": cfg.action_schema,
    "tactile_mode": cfg.tactile_mode,
    "tactile_profile": cfg.tactile_profile,
}

# Retain the native N0 MoT architecture; tactile modules exist but are frozen.
cfg.use_mot = True
cfg.mot_cross_attn_experts = ("video", "action")
cfg.mot_warmstart_experts = ("video", "action", "tactile")
cfg.mot_expert_hidden_dim = {"action": 1024, "tactile": 1024}
cfg.mot_expert_ffn_dim = {"action": 4096, "tactile": 4096}

# Reproducible step-based post-training. Public requests override via env only.
cfg.seed = _nonnegative_int_env("N0_TRACK32_SEED", 20260810)
cfg.batch_size = _positive_int_env("N0_TRACK32_BATCH_SIZE", 1)
cfg.gradient_accumulation_steps = _positive_int_env(
    "N0_TRACK32_GRADIENT_ACCUMULATION_STEPS", 1
)
cfg.num_steps = _positive_int_env("N0_TRACK32_NUM_STEPS", 1500)
cfg.stop_after_step = _resolve_stop_after(cfg.num_steps)
cfg.save_interval = _positive_int_env("N0_TRACK32_SAVE_INTERVAL", 300)
cfg.val_interval = _positive_int_env("N0_TRACK32_VAL_INTERVAL", 100)
cfg.max_latent_frames = _positive_int_env("N0_TRACK32_MAX_LATENT_FRAMES", 5)
cfg.load_worker = _nonnegative_int_env("N0_TRACK32_LOAD_WORKER", 0)
cfg.num_init_worker = 1
cfg.learning_rate = float(os.environ.get("N0_TRACK32_LEARNING_RATE", "1e-4"))
cfg.lr_schedule = "cosine"
cfg.lr_min_ratio = 0.1
cfg.warmup_steps = _nonnegative_int_env("N0_TRACK32_WARMUP_STEPS", 20)
cfg.cfg_prob = 0.0
cfg.enable_wandb = False
cfg.save_root = os.environ.get(
    "N0_TRACK32_SAVE_ROOT", str(_ARTIFACT_ROOT / "runs" / _ROLE)
)
cfg.eval_prompt = "perform the instructed Franka manipulation task"
cfg.raw_tactile_evaluation = False
cfg.deterministic_evaluation_crop_zero = False

if cfg.action_dim != len(cfg.norm_stat["q01"]) or cfg.action_dim != len(
    cfg.norm_stat["q99"]
):
    raise ValueError("Franka normalizer must match the 20D model action head")
if cfg.used_action_channel_ids != list(range(10)):
    raise ValueError("Franka must activate exactly model action channels 0..9")
if bool(cfg.resume_from) == bool(cfg.init_from):
    raise ValueError("Franka config requires exactly one init/resume checkpoint")
if _ROLE == "final_refit" and cfg.val_dataset_path is not None:
    raise ValueError("Franka final_refit must not use a validation view")
validate_tactile_profile_config(cfg)

twam_track32_franka_cfg = cfg
