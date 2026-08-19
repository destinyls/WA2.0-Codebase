# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Inference config for an immutable Franka Track 3.2 serve bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from easydict import EasyDict  # type: ignore[import-untyped]

from .twam_track32_franka_cfg import twam_track32_franka_cfg


def _action_kv_reuse_from_env() -> bool:
    raw_value = os.environ.get("N0_ACTION_DENOISE_KV_REUSE", "1")
    if raw_value not in {"0", "1"}:
        raise ValueError(
            "N0_ACTION_DENOISE_KV_REUSE must be exactly '0' or '1', "
            f"got {raw_value!r}"
        )
    return raw_value == "1"


def _runtime_contract_sha256(contract: dict[str, object]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_track32_franka_server_config() -> EasyDict:
    """Build a deployment config with the Action KV switch in its identity."""

    server = EasyDict()
    server.update(twam_track32_franka_cfg)
    server.__name__ = "Config: N0-TWAM Franka Track 3.2 SERVER"
    server.infer_mode = "server"
    server.host = "127.0.0.1"
    server.port = int(os.environ.get("N0_TRACK32_SERVER_PORT", "29642"))
    server.wan22_pretrained_model_name_or_path = os.environ.get(
        "N0_TRACK32_SERVE_BUNDLE", "/path/to/franka/serve-bundle"
    )
    server.resume_from = None
    server.init_from = None
    server.save_root = os.environ.get(
        "N0_TRACK32_SERVE_OUTPUT", "/path/to/franka/serve-output"
    )
    server.prompt = "perform the instructed Franka manipulation task"
    server.num_inference_steps = int(os.environ.get("N0_TRACK32_VIDEO_STEPS", "3"))
    server.action_num_inference_steps = int(
        os.environ.get("N0_TRACK32_ACTION_STEPS", "4")
    )
    server.video_exec_step = -1
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
    server.deterministic_episode_seed = True
    server.cold_seed_mode = "free"
    server.delta_smooth = False
    server.server_action_output_format = "absolute"
    server.server_return_action_channel_ids = list(range(20))
    server.server_tactile_denoise = False
    server.tactile_profile = "vision_only"
    server.require_tactile_profile_receipt = True
    server.tactile_keys = []
    server.tactile_mode = "disabled"
    server.synthetic_tactile_data = False
    server.use_local_tactile = False
    server.use_contact_gate = False
    server.tactile_diffusion_loss_weight = 0.0

    runtime_core: dict[str, object] = {
        "schema_version": 1,
        "action_schema": server.action_schema,
        "tactile_profile": server.tactile_profile,
        "server_tactile_denoise": server.server_tactile_denoise,
        "action_denoise_kv_reuse": server.action_denoise_kv_reuse,
        "action_denoise_kv_contract": server.action_denoise_kv_contract,
        "action_denoise_kv_activation_policy": (
            server.action_denoise_kv_activation_policy
        ),
        "action_denoise_kv_fallback_policy": (server.action_denoise_kv_fallback_policy),
        "full_replan_cache_policy": "clear_predicted_preserve_committed_v1",
    }
    runtime_sha256 = _runtime_contract_sha256(runtime_core)
    server.server_runtime_contract = {
        **runtime_core,
        "contract_sha256": runtime_sha256,
    }
    server.server_runtime_contract_sha256 = runtime_sha256

    bundle = Path(server.wan22_pretrained_model_name_or_path)
    if bundle.exists() and not bundle.is_dir():
        raise ValueError("N0_TRACK32_SERVE_BUNDLE must name a directory")
    if server.action_schema != "ee20_absee" or server.used_action_channel_ids != list(
        range(10)
    ):
        raise ValueError("Franka server must retain EE20 with active channels 0..9")
    return server


twam_track32_franka_server_cfg = build_track32_franka_server_config()
