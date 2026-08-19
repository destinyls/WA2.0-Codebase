# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Shared AgileX dual-arm qpos14 training contract.

The embodiment/action route is fixed here. Tactile availability is selected by
one of the three profile modules and remains orthogonal to the robot schema.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import cast

from easydict import EasyDict

from n0_twam.embodiments import (
    AGILEX_ACTION_ROUTE_SPEC,
    AGILEX_ACTION_SCHEMA,
    AGILEX_DATASET_ADAPTER,
    AGILEX_EMBODIMENT_PROFILE_ID,
    AGILEX_EMBODIMENT_SPEC,
    AGILEX_POLICY_ADAPTER,
    AGILEX_RGB_KEYS,
    AGILEX_TACTILE_KEYS,
    AGILEX_WRENCH_KEYS,
)
from n0_twam.tactile_profiles import (
    MIXED,
    VISION_ONLY,
    VISION_TACTILE,
)

from .twam_base_cfg import twam_base_cfg
from .twam_track3_agilex_contracts import (
    AgileXRepoRouteBinding,
    build_repo_route_binding,
    load_repo_route_binding,
    load_temporal_binding,
)
from .twam_track3_agilex_profiles import (  # noqa: F401
    apply_agilex_runtime_contract,
    apply_agilex_tactile_profile,
)

PROFILE_NAMES = (VISION_TACTILE, MIXED, VISION_ONLY)


def _path_env(name: str, fallback: str | Path) -> Path:
    return Path(os.environ.get(name, str(fallback))).expanduser()


def _positive_int_env(name: str, fallback: int) -> int:
    raw = os.environ.get(name, str(fallback))
    if not raw.isdecimal() or int(raw) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return int(raw)


def _profile_positive_int_env(name: str, fallback: int, profile: str | None) -> int:
    selected = os.environ.get("N0_TRACK3_AGILEX_TACTILE_PROFILE")
    if profile is not None and selected is not None and selected != profile:
        return int(fallback)
    return _positive_int_env(name, fallback)


def _nonnegative_int_env(name: str, fallback: int) -> int:
    raw = os.environ.get(name, str(fallback))
    if not raw.isdecimal():
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return int(raw)


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _weights_only_init_contract(
    checkpoint_root: Path,
) -> tuple[str, int, str, str]:
    config = _json_object(
        checkpoint_root / "transformer" / "config.json",
        label="AgileX initialization transformer config",
    )
    action_dim = config.get("action_dim")
    action_schema = config.get("action_schema")
    if action_dim == 20 and action_schema in (None, "ee20_pi05"):
        return "migrate_action", 20, "ee20_pi05", "init20_migrate_qpos14"
    if action_dim == 14 and action_schema == AGILEX_ACTION_SCHEMA:
        return "strict", 14, AGILEX_ACTION_SCHEMA, "init14_weights_only_stage_b"
    raise ValueError(
        "initial checkpoint is not a compatible 20D or qpos14 model: "
        f"action_dim={action_dim!r}, action_schema={action_schema!r}"
    )


def _route(
    *,
    tactile: bool,
) -> dict[str, object]:
    return {
        "embodiment": AGILEX_EMBODIMENT_PROFILE_ID,
        "action_schema": AGILEX_ACTION_SCHEMA,
        "rgb_keys": list(AGILEX_RGB_KEYS),
        "tactile_keys": list(AGILEX_TACTILE_KEYS) if tactile else [],
        "wrench_keys": list(AGILEX_WRENCH_KEYS) if tactile else [],
    }


def default_agilex_repo_routes(profile: str) -> dict[str, dict[str, object]]:
    """Return explicit import-safe template routes; formal runs replace names."""

    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown AgileX tactile profile {profile!r}")
    touch_repo = os.environ.get("N0_TRACK3_AGILEX_TACTILE_REPO", "agilex_touch_repo")
    rgb_repo = os.environ.get("N0_TRACK3_AGILEX_RGB_REPO", "agilex_rgb_repo")
    if not touch_repo or not rgb_repo or touch_repo == rgb_repo:
        raise ValueError("AgileX template repo names must be distinct and non-empty")
    if profile == VISION_TACTILE:
        return {touch_repo: _route(tactile=True)}
    if profile == MIXED:
        return {
            touch_repo: _route(tactile=True),
            rgb_repo: _route(tactile=False),
        }
    return {rgb_repo: _route(tactile=False)}


def load_agilex_repo_route_binding(profile: str) -> AgileXRepoRouteBinding:
    """Load one signed route contract or an explicit import-safe placeholder."""

    profile_env = {
        VISION_TACTILE: "N0_TRACK3_AGILEX_VISION_TACTILE_REPO_ROUTE_MANIFEST",
        MIXED: "N0_TRACK3_AGILEX_MIXED_REPO_ROUTE_MANIFEST",
        VISION_ONLY: "N0_TRACK3_AGILEX_VISION_ONLY_REPO_ROUTE_MANIFEST",
    }[profile]
    raw_path = os.environ.get(profile_env)
    selected_profile = os.environ.get("N0_TRACK3_AGILEX_TACTILE_PROFILE")
    if raw_path is None and selected_profile == profile:
        raw_path = os.environ.get("N0_TRACK3_AGILEX_REPO_ROUTE_MANIFEST")
    if raw_path is None:
        return build_repo_route_binding(
            default_agilex_repo_routes(profile),
            is_placeholder=True,
        )
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"AgileX repo route manifest not found: {path}")
    return load_repo_route_binding(path)


def load_agilex_repo_routes(profile: str) -> dict[str, dict[str, object]]:
    """Compatibility wrapper returning routes from the preserved binding."""

    return cast(
        dict[str, dict[str, object]],
        load_agilex_repo_route_binding(profile).routes,
    )


def _temporal_alignment_path(profile: str | None) -> Path | None:
    if profile is None:
        return None
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown AgileX tactile profile {profile!r}")
    profile_env = {
        VISION_TACTILE: ("N0_TRACK3_AGILEX_VISION_TACTILE_TEMPORAL_ALIGNMENT"),
        MIXED: "N0_TRACK3_AGILEX_MIXED_TEMPORAL_ALIGNMENT",
        VISION_ONLY: "N0_TRACK3_AGILEX_VISION_ONLY_TEMPORAL_ALIGNMENT",
    }[profile]
    raw_path = os.environ.get(profile_env)
    if (
        raw_path is None
        and os.environ.get("N0_TRACK3_AGILEX_TACTILE_PROFILE") == profile
    ):
        raw_path = os.environ.get("N0_TRACK3_AGILEX_TEMPORAL_ALIGNMENT")
    return None if raw_path is None else Path(raw_path).expanduser()


def _load_normalizer(path: Path) -> tuple[dict[str, list[float]], str | None, bool]:
    if not path.is_file():
        return {"q01": [-1.0] * 14, "q99": [1.0] * 14}, None, True
    payload = _json_object(path, label="AgileX qpos14 normalizer")
    q01 = payload.get("q01", payload.get("action_q01"))
    q99 = payload.get("q99", payload.get("action_q99"))
    if not isinstance(q01, list) or not isinstance(q99, list):
        raise ValueError("AgileX normalizer must contain q01/q99 arrays")
    if len(q01) != 14 or len(q99) != 14:
        raise ValueError("AgileX normalizer must contain exactly 14 channels")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"q01": list(map(float, q01)), "q99": list(map(float, q99))}, digest, False


def _manifest_config_fields(
    cfg: EasyDict,
    binding: AgileXRepoRouteBinding,
) -> None:
    manifest = binding.manifest
    payload = manifest.to_json_dict()
    routes = cast(dict[str, dict[str, object]], payload["routes"])
    cfg.repo_route_manifest = payload
    cfg.repo_route_manifest_sha256 = manifest.contract_sha256
    cfg.repo_route_manifest_original_sha256 = manifest.contract_sha256
    cfg.repo_route_manifest_source_path = binding.source_path
    cfg.repo_route_manifest_source_file_sha256 = binding.source_file_sha256
    cfg.repo_route_manifest_is_placeholder = binding.is_placeholder
    cfg.selected_repo_names = list(routes)
    cfg.obs_cam_keys = list(manifest.global_rgb_keys)
    cfg.per_repo_obs_cam_keys = {
        repo: list(cast(list[str], route["rgb_keys"])) for repo, route in routes.items()
    }
    cfg.tactile_keys = list(manifest.global_tactile_keys)
    cfg.per_repo_tactile_keys = {
        repo: list(cast(list[str], route["tactile_keys"]))
        for repo, route in routes.items()
    }
    cfg.tactile_sensor_id_map = dict(manifest.tactile_sensor_id_map)
    cfg.active_tactile_sensor_count = len(manifest.tactile_sensor_id_map)
    tactile_ids = tuple(int(value) for value in cfg.tactile_sensor_id_map.values())
    cfg.max_tactile_streams = max(4, 1 + max(tactile_ids, default=-1))
    cfg.wrench_keys = list(manifest.global_wrench_keys)
    cfg.per_repo_wrench_keys = {
        repo: list(cast(list[str], route["wrench_keys"]))
        for repo, route in routes.items()
    }
    cfg.wrench_sensor_id_map = dict(manifest.wrench_sensor_id_map)
    cfg.active_wrench_sensor_count = len(manifest.wrench_sensor_id_map)
    wrench_ids = tuple(int(value) for value in cfg.wrench_sensor_id_map.values())
    cfg.max_wrench_streams = max(2, 1 + max(wrench_ids, default=-1))
    cfg.wrench_arm_count = cfg.max_wrench_streams


def build_track3_agilex_base_config(
    *,
    repo_routes: Mapping[str, Mapping[str, object]],
    repo_route_binding: AgileXRepoRouteBinding | None = None,
    tactile_profile: str | None = None,
) -> EasyDict:
    """Build the common qpos14 contract without choosing a tactile profile."""

    if repo_route_binding is None:
        binding = build_repo_route_binding(repo_routes, is_placeholder=True)
    else:
        binding = repo_route_binding
        normalized_routes = build_repo_route_binding(
            repo_routes,
            is_placeholder=binding.is_placeholder,
        ).routes
        if binding.routes != normalized_routes:
            raise ValueError("AgileX repo routes differ from the preserved binding")
    cfg = EasyDict(deepcopy(dict(twam_base_cfg)))
    cfg.__name__ = "Config: N0-TWAM AgileX Track 3 qpos14 base"

    cfg.embodiment_profile = AGILEX_EMBODIMENT_PROFILE_ID
    cfg.embodiment_contract = AGILEX_EMBODIMENT_SPEC.to_json_dict()
    cfg.embodiment_contract_sha256 = AGILEX_EMBODIMENT_SPEC.contract_sha256
    cfg.action_route = AGILEX_ACTION_ROUTE_SPEC.name
    cfg.action_route_contract = AGILEX_ACTION_ROUTE_SPEC.to_json_dict()
    cfg.action_route_contract_sha256 = AGILEX_ACTION_ROUTE_SPEC.contract_sha256
    cfg.dataset_adapter = AGILEX_DATASET_ADAPTER
    cfg.policy_adapter = AGILEX_POLICY_ADAPTER
    _manifest_config_fields(cfg, binding)
    apply_agilex_runtime_contract(cfg, tactile_profile)

    cfg.action_schema = AGILEX_ACTION_SCHEMA
    cfg.source_action_schema = AGILEX_ACTION_SCHEMA
    cfg.derived_action_schema = AGILEX_ACTION_SCHEMA
    cfg.action_dim = 14
    cfg.state_dim = 14
    cfg.action_delta_mode = "none"
    cfg.used_action_channel_ids = list(range(14))
    cfg.per_repo_used_action_channel_ids = {}
    cfg.inverse_used_action_channel_ids = list(range(14))
    cfg.action_norm_method = "q01q99"
    cfg.server_action_output_format = "absolute"
    cfg.server_return_action_channel_ids = list(range(14))

    temporal = load_temporal_binding(
        _temporal_alignment_path(tactile_profile),
        repo_names=tuple(cfg.selected_repo_names),
        repo_route_manifest_sha256=cfg.repo_route_manifest_sha256,
    )
    cfg.temporal_alignment_contract = temporal.contract
    cfg.temporal_alignment_contract_sha256 = temporal.contract_sha256
    cfg.temporal_alignment_source_file_sha256 = temporal.source_file_sha256
    cfg.temporal_alignment_is_placeholder = temporal.is_placeholder
    cfg.action_per_frame = temporal.action_per_frame
    cfg.per_repo_action_offsets_per_anchor = {
        repo: list(offsets) for repo, offsets in temporal.action_offsets
    }
    cfg.per_repo_route_identity = dict(temporal.route_identities)
    cfg.per_repo_temporal_alignment_identity = dict(temporal.temporal_identities)
    cfg.require_temporal_alignment_contract = True

    artifact_root = _path_env(
        "N0_TRACK3_AGILEX_ARTIFACT_ROOT", "/path/to/agilex/artifacts"
    )
    dataset_root = _path_env("N0_TRACK3_AGILEX_DATASET_ROOT", "/path/to/agilex/lerobot")
    normalizer_path = _path_env(
        "N0_TRACK3_AGILEX_NORMALIZER", artifact_root / "qpos14_normalizer.json"
    )
    norm_stat, normalizer_sha, placeholder = _load_normalizer(normalizer_path)
    cfg.dataset_path = str(dataset_root)
    cfg.val_dataset_path = str(dataset_root)
    cfg.artifact_root = str(artifact_root)
    cfg.norm_stat = norm_stat
    cfg.norm_stat_path = str(normalizer_path)
    cfg.normalizer_sha256 = normalizer_sha
    cfg.normalizer_is_placeholder = placeholder

    base_model = _path_env("N0_BASE_MODEL", "/path/to/n0-twam-base")
    resume_raw = os.environ.get("N0_TRACK3_AGILEX_RESUME_FROM")
    init_raw = os.environ.get("N0_TRACK3_AGILEX_INIT_FROM")
    cfg.wan22_pretrained_model_name_or_path = str(base_model)
    cfg.empty_emb_path = str(
        _path_env("N0_EMPTY_EMBEDDING", base_model / "empty_emb.pt")
    )
    cfg.resume_from = None if resume_raw is None else str(Path(resume_raw).expanduser())
    cfg.init_from = (
        None
        if cfg.resume_from is not None
        else str(Path(init_raw).expanduser() if init_raw else base_model)
    )
    if cfg.resume_from is not None:
        initialization = ("strict", 14, AGILEX_ACTION_SCHEMA, "strict_resume_qpos14")
    elif init_raw is not None:
        initialization = _weights_only_init_contract(Path(cfg.init_from))
    else:
        initialization = (
            "migrate_action",
            20,
            "ee20_pi05",
            "init20_migrate_qpos14",
        )
    (
        cfg.checkpoint_compatibility,
        cfg.checkpoint_source_action_dim,
        cfg.checkpoint_source_action_schema,
        cfg.initialization_mode,
    ) = initialization
    cfg.strict_training_resume = cfg.resume_from is not None
    cfg.expected_init_transformer_sha256 = (
        None
        if cfg.resume_from is not None
        else os.environ.get("N0_RELEASED_TRANSFORMER_SHA256")
    )
    cfg.adopt_missing_action_schema = False
    cfg.action_init_seed = _nonnegative_int_env("N0_TRACK3_AGILEX_SEED", 20260811)
    cfg.initialized_target_only_prefixes = (
        "local_tactile_",
        "agilex_wrench_",
    )

    cfg.seed = cfg.action_init_seed
    cfg.batch_size = _profile_positive_int_env(
        "N0_TRACK3_AGILEX_BATCH_SIZE", 1, tactile_profile
    )
    cfg.gradient_accumulation_steps = _positive_int_env(
        "N0_TRACK3_AGILEX_GRADIENT_ACCUMULATION_STEPS", 1
    )
    cfg.num_steps = _positive_int_env("N0_TRACK3_AGILEX_NUM_STEPS", 1500)
    cfg.stop_after_step = _positive_int_env(
        "N0_TRACK3_AGILEX_STOP_AFTER_STEP", cfg.num_steps
    )
    if cfg.stop_after_step > cfg.num_steps:
        raise ValueError("AgileX stop_after_step cannot exceed num_steps")
    cfg.save_interval = _positive_int_env("N0_TRACK3_AGILEX_SAVE_INTERVAL", 300)
    cfg.val_interval = _positive_int_env("N0_TRACK3_AGILEX_VAL_INTERVAL", 100)
    cfg.max_latent_frames = _positive_int_env("N0_TRACK3_AGILEX_MAX_LATENT_FRAMES", 5)
    cfg.wrench_max_frames = _positive_int_env(
        "N0_TRACK3_AGILEX_WRENCH_MAX_FRAMES",
        max(64, int(cfg.max_latent_frames)),
    )
    if cfg.wrench_max_frames < cfg.max_latent_frames:
        raise ValueError("AgileX wrench_max_frames must cover max_latent_frames")
    cfg.load_worker = _nonnegative_int_env("N0_TRACK3_AGILEX_LOAD_WORKER", 0)
    cfg.num_init_worker = 1
    cfg.learning_rate = float(os.environ.get("N0_TRACK3_AGILEX_LR", "1e-4"))
    cfg.lr_schedule = "cosine"
    cfg.lr_min_ratio = 0.1
    cfg.warmup_steps = _nonnegative_int_env("N0_TRACK3_AGILEX_WARMUP_STEPS", 20)
    cfg.enable_wandb = False
    cfg.save_root = os.environ.get(
        "N0_TRACK3_AGILEX_SAVE_ROOT", str(artifact_root / "runs")
    )
    cfg.eval_prompt = "perform the instructed AgileX dual-arm manipulation task"
    cfg.require_tactile_profile_receipt = True
    cfg.require_repo_route_manifest_receipt = True
    return cfg


twam_track3_agilex_base_cfg = build_track3_agilex_base_config(
    repo_routes=default_agilex_repo_routes(VISION_ONLY),
    tactile_profile=VISION_ONLY,
)
