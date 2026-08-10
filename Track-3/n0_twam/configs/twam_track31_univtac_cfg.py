# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""N0-native UniVTAC Track 3.1 training configuration.

Paths are supplied through environment variables so credentials, hostnames, and
machine-specific storage layouts never enter the repository. NVIDIA simulator
evaluation is intentionally outside this training configuration and remains the
last integration phase.
"""

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

from easydict import EasyDict

from .twam_base_cfg import twam_base_cfg
from n0_twam.tactile_profiles import VISION_TACTILE, validate_tactile_profile_config
from .twam_track31_training_profiles import (
    build_track31_training_profile_contract,
    resolve_track31_run_role,
    resolve_track31_train_profile,
)

TRACK31_TACTILE_LATENT_CHANNELS = 48
TRACK31_TRANSFORMER_TACTILE_FIELDS = (
    "patch_size",
    "use_local_tactile",
    "max_tactile_streams",
    "tactile_in_channels",
    "tactile_num_tokens",
    "tactile_encoder_dim",
)
TRACK31_TACTILE_METADATA_FIELDS = (
    "patch_size",
    "snr_shift",
    "use_local_tactile",
    "max_tactile_streams",
    "active_tactile_sensor_count",
    "active_tactile_sensor_ids",
    "tactile_sensor_id_map",
    "tactile_in_channels",
    "tactile_num_tokens",
    "tactile_encoder_dim",
    "tactile_latent_channels",
)


def _env_path(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback)).expanduser()


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _load_parent_train_meta(checkpoint: str | None) -> dict[str, object] | None:
    if checkpoint is None:
        return None
    metadata_path = Path(checkpoint).expanduser() / "train_meta.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "profile-bound checkpoint initialization requires train_meta.json: "
            f"{metadata_path}"
        )
    return _load_json_object(metadata_path, label="parent train metadata")


def _positive_env_int(name: str, fallback: str) -> int:
    raw_value = os.environ.get(name, fallback)
    if not raw_value or any(character not in "0123456789" for character in raw_value):
        raise ValueError(
            f"{name} must be a positive base-10 integer, got {raw_value!r}"
        )
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _nonnegative_env_int(name: str, fallback: str) -> int:
    raw_value = os.environ.get(name, fallback)
    if not raw_value or any(character not in "0123456789" for character in raw_value):
        raise ValueError(
            f"{name} must be a non-negative base-10 integer, got {raw_value!r}"
        )
    return int(raw_value)


def resolve_track31_max_latent_frames() -> int:
    """Resolve the bounded Track 3.1 sequence length for this process."""
    return _positive_env_int("N0_TRACK31_MAX_LATENT_FRAMES", "5")


def resolve_track31_load_worker() -> int:
    """Keep random crop continuation on the checkpointed rank RNG stream."""
    load_worker = _nonnegative_env_int("N0_TRACK31_LOAD_WORKER", "0")
    if load_worker != 0:
        raise ValueError(
            "N0_TRACK31_LOAD_WORKER must be 0 while strict_training_resume=True"
        )
    return load_worker


def resolve_track31_sampler_rank_alignment() -> str:
    """Resolve deterministic cross-rank shape scheduling for Stage A."""
    value = os.environ.get(
        "N0_TRACK31_SAMPLER_RANK_ALIGNMENT",
        "contiguous",
    )
    if value not in ("contiguous", "shape_balanced"):
        raise ValueError(
            "N0_TRACK31_SAMPLER_RANK_ALIGNMENT must be contiguous or "
            "shape_balanced"
        )
    return value


def resolve_track31_stop_after_step(*, num_steps: int) -> int:
    """Resolve the absolute optimizer step that ends this invocation."""
    if num_steps <= 0:
        raise ValueError("num_steps must be greater than zero")
    if "N0_TRACK31_STOP_AFTER_STEP" not in os.environ:
        return num_steps
    stop_after_step = _positive_env_int("N0_TRACK31_STOP_AFTER_STEP", str(num_steps))
    if stop_after_step > num_steps:
        raise ValueError(
            "N0_TRACK31_STOP_AFTER_STEP must be less than or equal to "
            f"N0_TRACK31_NUM_STEPS ({num_steps}), got {stop_after_step}"
        )
    return stop_after_step


def _resolve_track31_stop_after_step_for_trainer(*, num_steps: int) -> object:
    """Parse valid integers but defer invalid-value errors to rank collectives."""
    raw_value = os.environ.get("N0_TRACK31_STOP_AFTER_STEP")
    if raw_value is None:
        return num_steps
    if raw_value and all(character in "0123456789" for character in raw_value):
        return int(raw_value)
    return raw_value


def build_track31_invocation_contract(
    *,
    start_step: int,
    stop_after_step: int,
    num_steps: int,
) -> dict[str, int | bool]:
    """Validate a bounded run without changing the saved execution contract."""
    values = (start_step, stop_after_step, num_steps)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(
            "Track 3.1 invocation requires integer optimizer-step boundaries"
        )
    if not 0 <= start_step < stop_after_step <= num_steps:
        raise ValueError(
            "Track 3.1 invocation requires "
            "0 <= start_step < stop_after_step <= num_steps; "
            f"got start_step={start_step}, stop_after_step={stop_after_step}, "
            f"num_steps={num_steps}"
        )
    return {
        "start_step": start_step,
        "stop_after_step": stop_after_step,
        "num_steps": num_steps,
        "optimizer_steps_this_invocation": stop_after_step - start_step,
        "is_final_invocation": stop_after_step == num_steps,
    }


def build_track31_tactile_training_contract(
    config: object,
) -> dict[str, object]:
    """Canonicalize model capacity and active UniVTAC tactile streams."""
    raw_patch_size = getattr(config, "patch_size", None)
    if (
        not isinstance(raw_patch_size, (list, tuple))
        or len(raw_patch_size) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in raw_patch_size
        )
    ):
        raise ValueError("Track 3.1 patch_size must contain three positive integers")

    positive_integer_fields = (
        "max_tactile_streams",
        "tactile_in_channels",
        "tactile_num_tokens",
        "tactile_encoder_dim",
    )
    positive_values: dict[str, int] = {}
    for field in positive_integer_fields:
        value = getattr(config, field, None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Track 3.1 {field} must be a positive integer")
        positive_values[field] = value

    snr_shift = getattr(config, "snr_shift", None)
    if (
        isinstance(snr_shift, bool)
        or not isinstance(snr_shift, (int, float))
        or not math.isfinite(float(snr_shift))
        or float(snr_shift) <= 0.0
    ):
        raise ValueError("Track 3.1 snr_shift must be positive")
    use_local_tactile = getattr(config, "use_local_tactile", None)
    if not isinstance(use_local_tactile, bool):
        raise ValueError("Track 3.1 use_local_tactile must be boolean")

    tactile_keys = getattr(config, "tactile_keys", None)
    sensor_id_map = getattr(config, "tactile_sensor_id_map", None)
    if not isinstance(tactile_keys, (list, tuple)) or not tactile_keys:
        raise ValueError("Track 3.1 tactile_keys must be a non-empty sequence")
    if not isinstance(sensor_id_map, dict):
        raise ValueError("Track 3.1 tactile_sensor_id_map must be a dictionary")
    if set(sensor_id_map) != set(tactile_keys):
        raise ValueError(
            "Track 3.1 tactile_sensor_id_map must exactly cover tactile_keys"
        )
    active_sensor_ids = []
    for key in tactile_keys:
        sensor_id = sensor_id_map[key]
        if (
            isinstance(sensor_id, bool)
            or not isinstance(sensor_id, int)
            or sensor_id < 0
        ):
            raise ValueError(
                f"Track 3.1 tactile sensor ID for {key!r} must be non-negative"
            )
        active_sensor_ids.append(sensor_id)
    if len(set(active_sensor_ids)) != len(active_sensor_ids):
        raise ValueError("Track 3.1 active tactile sensor IDs must be unique")
    recorded_active_count = getattr(
        config, "active_tactile_sensor_count", len(active_sensor_ids)
    )
    if (
        isinstance(recorded_active_count, bool)
        or not isinstance(recorded_active_count, int)
        or recorded_active_count != len(active_sensor_ids)
    ):
        raise ValueError(
            "Track 3.1 active_tactile_sensor_count must match the sensor map"
        )
    model_capacity = positive_values["max_tactile_streams"]
    if any(sensor_id >= model_capacity for sensor_id in active_sensor_ids):
        raise ValueError(
            "Track 3.1 active tactile sensor IDs must be below model capacity"
        )

    return {
        "patch_size": [int(value) for value in raw_patch_size],
        "snr_shift": float(snr_shift),
        "use_local_tactile": use_local_tactile,
        **positive_values,
        "tactile_latent_channels": TRACK31_TACTILE_LATENT_CHANNELS,
        "active_tactile_sensor_count": len(active_sensor_ids),
        "active_tactile_sensor_ids": sorted(active_sensor_ids),
        "tactile_sensor_id_map": {
            str(key): int(sensor_id_map[key]) for key in tactile_keys
        },
    }


def _canonical_tactile_checkpoint_value(
    field: str,
    value: object,
) -> object:
    invalid = ("invalid", field, repr(value))
    if field == "patch_size":
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in value
            )
        ):
            return invalid
        return list(value)
    if field == "use_local_tactile":
        return value if isinstance(value, bool) else invalid
    if field in {
        "max_tactile_streams",
        "active_tactile_sensor_count",
        "tactile_in_channels",
        "tactile_num_tokens",
        "tactile_encoder_dim",
        "tactile_latent_channels",
    }:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return invalid
        return value
    if field == "snr_shift":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            return invalid
        return float(value)
    if field == "active_tactile_sensor_ids":
        if not isinstance(value, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        ):
            return invalid
        return list(value)
    if field == "tactile_sensor_id_map":
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str)
            or isinstance(sensor_id, bool)
            or not isinstance(sensor_id, int)
            or sensor_id < 0
            for key, sensor_id in value.items()
        ):
            return invalid
        return dict(value)
    return value


def validate_track31_checkpoint_tactile_contract(
    *,
    transformer_config: Mapping[str, object],
    train_meta: Mapping[str, object] | None,
    expected_contract: Mapping[str, object],
    released_prior: bool,
) -> dict[str, object]:
    """Validate saved model shape and tactile semantics against Track 3.1."""
    expected_transformer = dict(expected_contract)
    if released_prior:
        expected_transformer["use_local_tactile"] = False
    for field in TRACK31_TRANSFORMER_TACTILE_FIELDS:
        actual_value = _canonical_tactile_checkpoint_value(
            field, transformer_config.get(field)
        )
        expected_value = _canonical_tactile_checkpoint_value(
            field, expected_transformer[field]
        )
        if actual_value != expected_value:
            raise ValueError(
                f"checkpoint tactile model mismatch for {field}: "
                f"checkpoint={actual_value!r} expected={expected_value!r}"
            )
    if released_prior:
        return {
            field: _canonical_tactile_checkpoint_value(
                field, expected_transformer[field]
            )
            for field in TRACK31_TRANSFORMER_TACTILE_FIELDS
        }
    if train_meta is None:
        raise ValueError("resume checkpoint has no tactile training metadata")

    for field in TRACK31_TACTILE_METADATA_FIELDS:
        actual_value = _canonical_tactile_checkpoint_value(field, train_meta.get(field))
        expected_value = _canonical_tactile_checkpoint_value(
            field, expected_contract[field]
        )
        if actual_value != expected_value:
            raise ValueError(
                f"resume tactile metadata mismatch for {field}: "
                f"metadata={actual_value!r} expected={expected_value!r}"
            )
    if transformer_config.get("snr_shift") != expected_contract["snr_shift"]:
        raise ValueError(
            "resume transformer config snr_shift differs from training contract"
        )
    if (
        transformer_config.get("tactile_latent_channels")
        != expected_contract["tactile_latent_channels"]
    ):
        raise ValueError(
            "resume transformer config tactile_latent_channels differs from "
            "training contract"
        )
    return {
        field: expected_contract[field] for field in TRACK31_TACTILE_METADATA_FIELDS
    }


def resolve_track31_training_overrides() -> dict[str, int]:
    """Resolve strict positive launcher overrides without editing source."""
    return {
        "num_steps": _positive_env_int("N0_TRACK31_NUM_STEPS", "2000"),
        "save_interval": _positive_env_int("N0_TRACK31_SAVE_INTERVAL", "500"),
        "val_interval": _positive_env_int("N0_TRACK31_VAL_INTERVAL", "100"),
        "gradient_accumulation_steps": _positive_env_int(
            "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS", "1"
        ),
        "batch_size": _positive_env_int("N0_TRACK31_BATCH_SIZE", "1"),
    }


_ARTIFACT_ROOT = _env_path(
    "N0_TRACK31_ARTIFACT_ROOT",
    "/path/to/univtac_track31/artifacts",
)
_DATASET_ROOT = _env_path(
    "N0_TRACK31_LEROBOT_ROOT",
    "/path/to/univtac_track31/lerobot",
)
_BASE_MODEL = _env_path("N0_BASE_MODEL", "/path/to/base-model")
_RELEASED_CHECKPOINT = _env_path(
    "N0_RELEASED_CHECKPOINT",
    "/path/to/n0-twam-checkpoint",
)
_EMPTY_EMBEDDING = _env_path(
    "N0_EMPTY_EMBEDDING",
    "/path/to/base-model/empty_emb.pt",
)
_RESUME_CHECKPOINT = os.environ.get("N0_TRACK31_RESUME_FROM")
_INIT_CHECKPOINT = os.environ.get("N0_TRACK31_INIT_FROM")
_TRAINING_PROFILE_ID = resolve_track31_train_profile()
_RUN_ROLE = resolve_track31_run_role()
_PARENT_CHECKPOINT = _RESUME_CHECKPOINT or _INIT_CHECKPOINT
_PARENT_TRAIN_META = _load_parent_train_meta(_PARENT_CHECKPOINT)
_PROFILE_CONTRACT = build_track31_training_profile_contract(
    training_profile_id=_TRAINING_PROFILE_ID,
    run_role=_RUN_ROLE,
    released_checkpoint=_RELEASED_CHECKPOINT,
    init_from=_INIT_CHECKPOINT,
    resume_from=_RESUME_CHECKPOINT,
    parent_train_meta=_PARENT_TRAIN_META,
)
_TRAIN_VIEW_ID = str(_PROFILE_CONTRACT["train_view_id"])
_VALIDATION_VIEW_VALUE = _PROFILE_CONTRACT["validation_view_id"]
_VALIDATION_VIEW_ID = (
    None if _VALIDATION_VIEW_VALUE is None else str(_VALIDATION_VIEW_VALUE)
)
_NORMALIZER_ID = str(_PROFILE_CONTRACT["normalizer_id"])
_NORMALIZER_SOURCE_VIEW_ID = str(_PROFILE_CONTRACT["normalizer_source_view_id"])
_NORMALIZER_PATH = _env_path(
    "N0_TRACK31_NORMALIZER_PATH",
    str(_ARTIFACT_ROOT / "normalizers" / f"{_NORMALIZER_ID}.json"),
)
_MANIFEST_PATH = _env_path(
    "N0_TRACK31_MANIFEST_PATH",
    str(_ARTIFACT_ROOT / "universe_manifest_v4.json"),
)
_TRAIN_VIEW_PATH = _env_path(
    "N0_TRACK31_TRAIN_VIEW_PATH",
    str(_ARTIFACT_ROOT / "views" / f"{_TRAIN_VIEW_ID}.json"),
)
_NORMALIZER_SOURCE_VIEW_PATH = _env_path(
    "N0_TRACK31_NORMALIZER_SOURCE_VIEW_PATH",
    str(_ARTIFACT_ROOT / "views" / f"{_NORMALIZER_SOURCE_VIEW_ID}.json"),
)
_VALIDATION_VIEW_PATH = (
    None
    if _VALIDATION_VIEW_ID is None
    else _env_path(
        "N0_TRACK31_VALIDATION_VIEW_PATH",
        str(_ARTIFACT_ROOT / "views" / f"{_VALIDATION_VIEW_ID}.json"),
    )
)

cfg = EasyDict(twam_base_cfg.copy())
cfg.__name__ = "Config: N0-TWAM UniVTAC Track 3.1 qpos8"

# All training profiles read immutable views over the physical train759 repo.
# Frozen40 is intentionally never mounted by this training configuration.
cfg.dataset_path = str(_DATASET_ROOT / "train759")
cfg.val_dataset_path = (
    str(_DATASET_ROOT / "train759") if _VALIDATION_VIEW_PATH is not None else None
)
cfg.lerobot_root = str(_DATASET_ROOT)
cfg.dataset_manifest_path = str(_MANIFEST_PATH)
cfg.conversion_report_path = str(_ARTIFACT_ROOT / "conversion_report.json")
cfg.training_profile_id = _TRAINING_PROFILE_ID
cfg.run_role = _RUN_ROLE
cfg.training_profile_contract = dict(_PROFILE_CONTRACT)
cfg.train_view_id = _TRAIN_VIEW_ID
cfg.dataset_view_path = str(_TRAIN_VIEW_PATH)
cfg.normalizer_source_view_id = _NORMALIZER_SOURCE_VIEW_ID
cfg.normalizer_source_view_path = str(_NORMALIZER_SOURCE_VIEW_PATH)
cfg.validation_view_id = _VALIDATION_VIEW_ID
cfg.val_dataset_view_path = (
    None if _VALIDATION_VIEW_PATH is None else str(_VALIDATION_VIEW_PATH)
)
cfg.sampler_coverage_mode = "pad_global"
cfg.sampler_rank_alignment = resolve_track31_sampler_rank_alignment()
cfg.crop_window_policy_id = "epoch_content_addressed_crop_v1"

# UniVTAC observation contract.
cfg.obs_cam_keys = [
    "observation.images.top",
    "observation.images.wrist_l",
]
cfg.tactile_profile = VISION_TACTILE
cfg.tactile_mode = "enabled"
cfg.tactile_keys = [
    "observation.images.tactile_a",
    "observation.images.tactile_b",
]
cfg.per_repo_obs_cam_keys = {}
cfg.per_repo_tactile_keys = {}
cfg.tactile_optional = False
cfg.synthetic_tactile_data = False
cfg.tactile_sensor_id_map = {key: index for index, key in enumerate(cfg.tactile_keys)}
# Preserve the released model's four-row sensor embedding. UniVTAC activates
# only IDs 0/1; narrowing this capacity would be a non-action weight migration.
cfg.max_tactile_streams = 4
cfg.active_tactile_sensor_count = len(cfg.tactile_sensor_id_map)
cfg.use_local_tactile = True
cfg.local_tactile_mode = "current"
cfg.tactile_global_zero = False
cfg.tactile_cfg_prob = 0.0
cfg.tactile_diffusion_loss_weight = 1.0
cfg.freeze_tactile_parameters = False

# Explicit native action schema. A dimension of 8 alone never selects qpos8.
cfg.action_schema = "qpos8_next_step"
cfg.action_dim = 8
cfg.action_delta_mode = "none"
cfg.action_per_frame = 4
cfg.used_action_channel_ids = list(range(8))
cfg.per_repo_used_action_channel_ids = {}
cfg.inverse_used_action_channel_ids = list(range(8))
cfg.action_norm_method = "q01q99"

if _NORMALIZER_PATH.is_file():
    _normalizer = _load_json_object(_NORMALIZER_PATH, label="qpos8 normalizer")
    cfg.norm_stat = {
        "q01": list(_normalizer["action_q01"]),
        "q99": list(_normalizer["action_q99"]),
    }
    cfg.normalizer_sha256 = str(_normalizer["normalizer_sha256"])
    cfg.source_manifest_sha256 = str(_normalizer["source_manifest_sha256"])
    cfg.norm_stat_path = str(_NORMALIZER_PATH)
else:
    cfg.norm_stat = {"q01": [-1.0] * 8, "q99": [1.0] * 8}
    cfg.normalizer_sha256 = None
    cfg.source_manifest_sha256 = None
    cfg.norm_stat_path = f"{_NORMALIZER_PATH} (MISSING)"

# The released 20D EE checkpoint is a prior, not an ACT policy. The entire
# semantic action projection is reset under a recorded seed; no tensor slicing.
cfg.wan22_pretrained_model_name_or_path = str(_BASE_MODEL)
cfg.empty_emb_path = str(_EMPTY_EMBEDDING)
cfg.resume_from = _RESUME_CHECKPOINT
cfg.init_from = (
    None if _RESUME_CHECKPOINT else str(_PROFILE_CONTRACT["checkpoint_path"])
)
cfg.checkpoint_compatibility = str(_PROFILE_CONTRACT["checkpoint_compatibility"])
cfg.strict_training_resume = True
cfg.checkpoint_source_action_dim = int(
    _PROFILE_CONTRACT["checkpoint_source_action_dim"]
)
cfg.checkpoint_source_action_schema = str(
    _PROFILE_CONTRACT["checkpoint_source_action_schema"]
)
cfg.initialization_mode = str(_PROFILE_CONTRACT["initialization_mode"])
cfg.training_lineage = dict(_PROFILE_CONTRACT["training_lineage"])
cfg.inherit_action_migration_report = bool(
    _PROFILE_CONTRACT["inherit_action_migration_report"]
)
cfg.action_init_seed = int(os.environ.get("N0_ACTION_INIT_SEED", "0"))
cfg.seed = int(os.environ.get("N0_TRAIN_SEED", "20260801"))
cfg.save_root = os.environ.get(
    "N0_TRACK31_SAVE_ROOT",
    str(_ARTIFACT_ROOT / "runs" / "n0_qpos8"),
)
cfg.eval_prompt = "manipulate the object using visual and tactile feedback"

# Retain the released N0 MoT prior and train its native visual/tactile/action path.
cfg.use_mot = True
cfg.mot_cross_attn_experts = ("video", "action")
cfg.mot_warmstart_experts = ("video", "action", "tactile")
cfg.mot_expert_hidden_dim = {"action": 1024, "tactile": 1024}
cfg.mot_expert_ffn_dim = {"action": 4096, "tactile": 4096}
cfg.use_contact_gate = False

# Reproducible smoke-ready defaults; distributed launch may override them.
cfg.tactile_cfg_prob = 0.0
cfg.cfg_prob = 0.0
cfg.noisy_cond_prob_tactile = 0.0
cfg.enable_wandb = False
cfg.num_init_worker = 1
cfg.load_worker = resolve_track31_load_worker()
cfg.update(resolve_track31_training_overrides())
cfg.stop_after_step = _resolve_track31_stop_after_step_for_trainer(
    num_steps=cfg.num_steps
)
cfg.learning_rate = 1e-4
cfg.lr_schedule = "cosine"
cfg.lr_min_ratio = 0.1
cfg.warmup_steps = 20
cfg.max_latent_frames = resolve_track31_max_latent_frames()
# Stage-A overrides both flags on a private deepcopy. Training remains random-crop.
cfg.raw_tactile_evaluation = False
cfg.deterministic_evaluation_crop_zero = False

assert cfg.action_schema == "qpos8_next_step"
assert cfg.action_dim == len(cfg.norm_stat["q01"]) == len(cfg.norm_stat["q99"])
assert cfg.action_per_frame == 4
assert cfg.max_latent_frames > 0
_track31_tactile_contract = build_track31_tactile_training_contract(cfg)
assert (
    cfg.active_tactile_sensor_count
    == _track31_tactile_contract["active_tactile_sensor_count"]
)
assert cfg.checkpoint_compatibility == _PROFILE_CONTRACT["checkpoint_compatibility"]
assert bool(cfg.resume_from) != bool(cfg.init_from)
assert (cfg.val_dataset_path is None) == (_RUN_ROLE == "final_refit")
validate_tactile_profile_config(cfg)

twam_track31_univtac_cfg = cfg
