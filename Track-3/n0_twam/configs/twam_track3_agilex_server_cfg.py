# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Profile-bound AgileX qpos14 inference-server configuration."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

from easydict import EasyDict  # type: ignore[import-untyped]

from n0_twam.embodiments import AGILEX_ACTION_SCHEMA
from n0_twam.tactile_profiles import MIXED, VISION_ONLY, VISION_TACTILE

from .twam_track3_agilex_contracts import canonical_sha256
from .twam_track3_agilex_mixed_cfg import twam_track3_agilex_mixed_cfg
from .twam_track3_agilex_vision_only_cfg import twam_track3_agilex_vision_only_cfg
from .twam_track3_agilex_vision_tactile_cfg import (
    twam_track3_agilex_vision_tactile_cfg,
)


def _action_kv_reuse_from_env() -> bool:
    raw_value = os.environ.get("N0_ACTION_DENOISE_KV_REUSE", "1")
    if raw_value not in {"0", "1"}:
        raise ValueError(
            "N0_ACTION_DENOISE_KV_REUSE must be exactly '0' or '1', "
            f"got {raw_value!r}"
        )
    return raw_value == "1"


def build_track3_agilex_server_config(*, training_config: EasyDict) -> EasyDict:
    """Bind serving to the exact training embodiment/profile/route hashes."""

    server = EasyDict(deepcopy(dict(training_config)))
    server.__name__ = "Config: N0-TWAM AgileX Track 3 SERVER"
    server.infer_mode = "server"
    server.host = "127.0.0.1"
    server.port = int(os.environ.get("N0_TRACK3_AGILEX_SERVER_PORT", "29643"))
    server.wan22_pretrained_model_name_or_path = os.environ.get(
        "N0_TRACK3_AGILEX_SERVE_BUNDLE", "/path/to/agilex/serve-bundle"
    )
    server.resume_from = None
    server.init_from = None
    server.save_root = os.environ.get(
        "N0_TRACK3_AGILEX_SERVE_OUTPUT", "/path/to/agilex/serve-output"
    )
    server.prompt = "perform the instructed AgileX dual-arm manipulation task"
    server.num_inference_steps = int(
        os.environ.get("N0_TRACK3_AGILEX_VIDEO_STEPS", "3")
    )
    server.action_num_inference_steps = int(
        os.environ.get("N0_TRACK3_AGILEX_ACTION_STEPS", "4")
    )
    server.guidance_scale = 1
    server.action_guidance_scale = 1
    server.action_denoise_kv_reuse = _action_kv_reuse_from_env()
    server.action_denoise_kv_contract = "fixed_context_causal_v1"
    server.action_denoise_kv_activation_policy = "requires_tactile_global_tokens_v1"
    server.action_denoise_kv_fallback_policy = "rerun_full_v1"
    server.enable_offload = False
    server.save_inference_artifacts = False
    server.empty_cache_each_request = False
    server.show_inference_progress = False
    server.server_action_output_format = "absolute"
    server.server_return_action_channel_ids = list(range(14))
    server.require_signed_task_route = True
    server.signed_task_route_manifest_path = os.environ.get(
        "N0_TRACK3_AGILEX_SIGNED_TASK_ROUTE",
        "/path/to/agilex/signed-task-route.json",
    )
    server.signed_task_route_manifest_sha256 = os.environ.get(
        "N0_TRACK3_AGILEX_SIGNED_TASK_ROUTE_SHA256"
    )
    server.require_tactile_profile_receipt = True
    server.require_repo_route_manifest_receipt = True
    profile = str(server.tactile_profile)
    if profile not in {VISION_TACTILE, MIXED, VISION_ONLY}:
        raise ValueError(f"unknown AgileX tactile profile {profile!r}")
    server.server_tactile_denoise = profile in {VISION_TACTILE, MIXED}
    runtime_core: dict[str, object] = {
        "schema_version": 1,
        "action_schema": server.action_schema,
        "embodiment_contract_sha256": server.embodiment_contract_sha256,
        "action_route_contract_sha256": server.action_route_contract_sha256,
        "repo_route_manifest_sha256": server.repo_route_manifest_sha256,
        "tactile_profile_contract_sha256": server.tactile_profile_contract_sha256,
        "contact_profile_contract_sha256": server.contact_profile_contract_sha256,
        "tactile_profile": profile,
        "server_tactile_denoise": server.server_tactile_denoise,
        "action_denoise_kv_reuse": server.action_denoise_kv_reuse,
        "action_denoise_kv_contract": server.action_denoise_kv_contract,
        "action_denoise_kv_activation_policy": (
            server.action_denoise_kv_activation_policy
        ),
        "action_denoise_kv_fallback_policy": (server.action_denoise_kv_fallback_policy),
        "full_replan_cache_policy": "clear_predicted_preserve_committed_v1",
    }
    runtime_sha256 = canonical_sha256(runtime_core)
    server.server_runtime_contract = {
        **runtime_core,
        "contract_sha256": runtime_sha256,
    }
    server.server_runtime_contract_sha256 = runtime_sha256
    if server.action_schema != AGILEX_ACTION_SCHEMA:
        raise ValueError("AgileX server requires qpos14_joint_absolute_v1")
    bundle = Path(server.wan22_pretrained_model_name_or_path)
    if bundle.exists() and not bundle.is_dir():
        raise ValueError("N0_TRACK3_AGILEX_SERVE_BUNDLE must name a directory")
    return server


_PROFILE = os.environ.get("N0_TRACK3_AGILEX_TACTILE_PROFILE", VISION_ONLY)
_TRAINING_CONFIGS = {
    VISION_TACTILE: twam_track3_agilex_vision_tactile_cfg,
    MIXED: twam_track3_agilex_mixed_cfg,
    VISION_ONLY: twam_track3_agilex_vision_only_cfg,
}
if _PROFILE not in _TRAINING_CONFIGS:
    raise ValueError(f"unknown N0_TRACK3_AGILEX_TACTILE_PROFILE {_PROFILE!r}")

twam_track3_agilex_server_cfg = build_track3_agilex_server_config(
    training_config=_TRAINING_CONFIGS[_PROFILE]
)
