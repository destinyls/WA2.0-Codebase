# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Mixed tactile/RGB-only post-training template.

Edit the two repository names and tactile-key lists so they exactly match the
repositories under ``twam_posttrain_cfg.dataset_path``. Additional repos must be
listed explicitly; an empty list is the only valid no-tactile declaration.
"""

from easydict import EasyDict

from n0_twam.tactile_profiles import MIXED, validate_tactile_profile_config
from .twam_posttrain_cfg import twam_posttrain_cfg

cfg = EasyDict(twam_posttrain_cfg.copy())
cfg.__name__ = "Config: N0-TWAM mixed tactile/RGB-only post-train"
cfg.tactile_profile = MIXED
cfg.tactile_mode = "enabled"
cfg.tactile_optional = False
cfg.synthetic_tactile_data = False

# EDIT ME: keys are LeRobot repo directory names, not filesystem paths.
cfg.per_repo_tactile_keys = {
    "repo_with_touch": list(cfg.tactile_keys),
    "repo_rgb_only": [],
}

# Real tactile batches supervise/condition the tactile path. RGB-only batches
# set tactile_cond_drop=True and keep the same trainable tactile parameter graph
# through the model's zero-anchor path.
cfg.tactile_cfg_prob = 0.1
cfg.tactile_diffusion_loss_weight = 1.0
cfg.freeze_tactile_parameters = False

validate_tactile_profile_config(cfg)

twam_mixed_cfg = cfg
