# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Inference config for an immutable Franka Track 3.2 serve bundle."""

from __future__ import annotations

import os
from pathlib import Path

from easydict import EasyDict

from .twam_track32_franka_cfg import twam_track32_franka_cfg

s = EasyDict()
s.update(twam_track32_franka_cfg)
s.__name__ = "Config: N0-TWAM Franka Track 3.2 SERVER"
s.infer_mode = "server"
s.host = "127.0.0.1"
s.port = int(os.environ.get("N0_TRACK32_SERVER_PORT", "29642"))
s.wan22_pretrained_model_name_or_path = os.environ.get(
    "N0_TRACK32_SERVE_BUNDLE", "/path/to/franka/serve-bundle"
)
s.resume_from = None
s.init_from = None
s.save_root = os.environ.get("N0_TRACK32_SERVE_OUTPUT", "/path/to/franka/serve-output")
s.prompt = "perform the instructed Franka manipulation task"
s.num_inference_steps = int(os.environ.get("N0_TRACK32_VIDEO_STEPS", "3"))
s.action_num_inference_steps = int(os.environ.get("N0_TRACK32_ACTION_STEPS", "4"))
s.video_exec_step = -1
s.guidance_scale = 1
s.action_guidance_scale = 1
s.enable_offload = False
s.deterministic_episode_seed = True
s.cold_seed_mode = "free"
s.delta_smooth = False
s.server_action_output_format = "absolute"
s.server_return_action_channel_ids = list(range(20))
s.server_tactile_denoise = False
s.tactile_profile = "vision_only"
s.require_tactile_profile_receipt = True
s.tactile_keys = []
s.tactile_mode = "disabled"
s.synthetic_tactile_data = False
s.use_local_tactile = False
s.use_contact_gate = False
s.tactile_diffusion_loss_weight = 0.0

if (
    Path(s.wan22_pretrained_model_name_or_path).exists()
    and not Path(s.wan22_pretrained_model_name_or_path).is_dir()
):
    raise ValueError("N0_TRACK32_SERVE_BUNDLE must name a directory")
if s.action_schema != "ee20_absee" or s.used_action_channel_ids != list(range(10)):
    raise ValueError("Franka server must retain EE20 with active channels 0..9")

twam_track32_franka_server_cfg = s
