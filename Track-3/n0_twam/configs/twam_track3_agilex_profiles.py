# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""AgileX tactile/wrench topology and training-lineage profiles."""

from __future__ import annotations

import os

from easydict import EasyDict

from n0_twam.tactile_profiles import (
    MIXED,
    VISION_ONLY,
    VISION_TACTILE,
    validate_tactile_profile_config,
)

from .twam_track3_agilex_contracts import (
    canonical_sha256,
    load_runner_artifact_identity,
)
from .twam_track3_agilex_recipe import (
    build_agilex_resume_recipe_contract,
)

PROFILE_NAMES = (VISION_TACTILE, MIXED, VISION_ONLY)


def apply_agilex_runtime_contract(
    cfg: EasyDict,
    profile: str | None,
) -> EasyDict:
    """Bind runner metadata without leaking one selected profile into others."""

    role = os.environ.get("N0_TRACK3_AGILEX_RUN_ROLE", "development")
    if role not in {"development", "final_refit"}:
        raise ValueError("N0_TRACK3_AGILEX_RUN_ROLE is invalid")
    accelerator = os.environ.get("N0_TRACK3_AGILEX_ACCELERATOR_PROFILE", "portable")
    if accelerator not in {"portable", "hcu_performance"}:
        raise ValueError("N0_TRACK3_AGILEX_ACCELERATOR_PROFILE is invalid")
    world_raw = os.environ.get("N0_TRACK3_AGILEX_EXPECTED_WORLD_SIZE")
    if world_raw is not None and (not world_raw.isdecimal() or int(world_raw) <= 0):
        raise ValueError("N0_TRACK3_AGILEX_EXPECTED_WORLD_SIZE must be positive")
    cfg.run_role = role
    cfg.accelerator_profile = accelerator
    cfg.expected_world_size = None if world_raw is None else int(world_raw)
    cfg.capture_runtime_provenance = True
    cfg.training_profile_id = None
    if profile is None:
        cfg.track32_profile_id = None
        cfg.track32_artifact_identity = None
        return cfg
    expected_profile_id = f"agilex_track3_{profile}_v1"
    selected = os.environ.get("N0_TRACK3_AGILEX_TACTILE_PROFILE")
    supplied_profile_id = os.environ.get("N0_TRACK3_AGILEX_PROFILE_ID")
    applies = selected is None or selected == profile
    if applies and supplied_profile_id not in {None, expected_profile_id}:
        raise ValueError("N0_TRACK3_AGILEX_PROFILE_ID differs from config profile")
    cfg.track32_profile_id = expected_profile_id
    cfg.track32_artifact_identity = load_runner_artifact_identity(profile)
    return cfg


def _validate_repo_contact_routes(cfg: EasyDict, profile: str) -> None:
    tactile_routes = dict(cfg.per_repo_tactile_keys)
    wrench_routes = dict(cfg.per_repo_wrench_keys)
    has_touch = [bool(keys) for keys in tactile_routes.values()]
    if profile == VISION_TACTILE:
        if not has_touch or not all(has_touch) or not all(wrench_routes.values()):
            raise ValueError(
                "vision_tactile requires tactile and wrench for every repository"
            )
        return
    if profile == MIXED:
        if not any(has_touch) or not any(not value for value in has_touch):
            raise ValueError("mixed requires both tactile and no-tactile repositories")
        for repo, keys in tactile_routes.items():
            if not keys and wrench_routes[repo]:
                raise ValueError("no-tactile mixed routes cannot expose wrench")
        return
    if any(has_touch) or any(wrench_routes.values()):
        raise ValueError("vision_only cannot route tactile or wrench observations")


def _build_contact_profile_contract(
    cfg: EasyDict,
    *,
    profile: str,
) -> tuple[dict[str, object], str]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "profile": profile,
        "repo_route_manifest_sha256": cfg.repo_route_manifest_sha256,
        "tactile_profile_contract_sha256": cfg.tactile_profile_contract_sha256,
        "instantiate_local_tactile": cfg.instantiate_local_tactile,
        "use_local_tactile": cfg.use_local_tactile,
        "bypass_local_tactile": cfg.bypass_local_tactile,
        "freeze_local_tactile_parameters": cfg.freeze_local_tactile_parameters,
        "instantiate_wrench_conditioner": cfg.instantiate_wrench_conditioner,
        "use_wrench_conditioner": cfg.use_wrench_conditioner,
        "bypass_wrench_conditioner": cfg.bypass_wrench_conditioner,
        "freeze_wrench_parameters": cfg.freeze_wrench_parameters,
        "max_tactile_streams": cfg.max_tactile_streams,
        "max_wrench_streams": cfg.max_wrench_streams,
        "wrench_arm_count": cfg.wrench_arm_count,
        "wrench_max_frames": cfg.wrench_max_frames,
        "per_repo_tactile_keys": dict(cfg.per_repo_tactile_keys),
        "per_repo_wrench_keys": dict(cfg.per_repo_wrench_keys),
        "contact_conditioning_bypass": cfg.contact_conditioning_bypass,
        "contact_cond_drop_fixed_value": cfg.contact_cond_drop_fixed_value,
        "contact_cond_drop_strategy": cfg.contact_cond_drop_strategy,
        "contact_cond_drop_batch_contract": cfg.contact_cond_drop_batch_contract,
        "contact_cond_drop_modalities": list(cfg.contact_cond_drop_modalities),
    }
    digest = canonical_sha256(payload)
    return {**payload, "contract_sha256": digest}, digest


def apply_agilex_tactile_profile(cfg: EasyDict, profile: str) -> EasyDict:
    """Apply and validate one complete tactile/wrench trainability profile."""

    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown AgileX tactile profile {profile!r}")
    _validate_repo_contact_routes(cfg, profile)
    enabled = profile != VISION_ONLY
    cfg.tactile_profile = profile
    cfg.tactile_mode = "enabled" if enabled else "disabled"
    cfg.tactile_optional = False
    cfg.synthetic_tactile_data = False
    cfg.tactile_cfg_prob = 0.1 if profile == MIXED else 0.0
    cfg.noisy_cond_prob_tactile = 0.0
    cfg.tactile_diffusion_loss_weight = 1.0 if enabled else 0.0
    cfg.freeze_tactile_parameters = not enabled
    cfg.instantiate_local_tactile = True
    cfg.use_local_tactile = enabled
    cfg.bypass_local_tactile = not enabled
    cfg.freeze_local_tactile_parameters = not enabled
    cfg.local_tactile_mode = "current" if enabled else "disabled"
    cfg.use_contact_gate = False
    cfg.tactile_global_zero = False
    cfg.cfg_prob = 0.0

    cfg.instantiate_wrench_conditioner = True
    cfg.use_wrench_conditioner = enabled and bool(cfg.wrench_keys)
    cfg.freeze_wrench_parameters = not enabled
    cfg.bypass_wrench_conditioner = not enabled
    cfg.wrench_conditioner_schema = "agilex_wrench_conditioner_v1"
    cfg.contact_conditioning_bypass = not enabled
    cfg.contact_cond_drop_fixed_value = True if not enabled else None
    cfg.contact_cond_drop_strategy = "content_addressed_v1"
    cfg.contact_cond_drop_batch_contract = "uniform_scalar_v1"
    if profile == MIXED and int(cfg.batch_size) != 1:
        raise ValueError(
            "AgileX mixed profile requires batch_size=1 until contact_cond_drop "
            "supports per-sample conditioning"
        )
    cfg.contact_cond_drop_modalities = [
        "tactile",
        "local_tactile",
        "global_tactile",
        "wrench",
        "contact_gate",
    ]
    cfg.mask_contract = {
        "polarity": "true_is_valid_or_enabled",
        "action_valid_mask": "bool[B,F,H]",
        "temporal_valid_mask": "bool[B,F]",
        "tactile_available_mask": "bool[B,F,S]",
        "wrench_available_mask": "bool[B,F,A]",
        "contact_cond_drop": "bool[B]",
    }
    cfg.vision_only_ignore_valid_extra_tactile = profile == VISION_ONLY

    tactile_contract = validate_tactile_profile_config(
        cfg, repo_names=cfg.selected_repo_names
    )
    cfg.tactile_profile_contract = tactile_contract.to_json_dict()
    cfg.tactile_profile_contract_sha256 = tactile_contract.contract_sha256
    contact_contract, contact_sha256 = _build_contact_profile_contract(
        cfg,
        profile=profile,
    )
    cfg.contact_profile_contract = contact_contract
    cfg.contact_profile_contract_sha256 = contact_sha256
    cfg.agilex_contact_profile_contract = contact_contract
    cfg.agilex_contact_profile_contract_sha256 = contact_sha256
    cfg.training_lineage = {
        "schema_version": 1,
        "embodiment_contract_sha256": cfg.embodiment_contract_sha256,
        "action_route_contract_sha256": cfg.action_route_contract_sha256,
        "tactile_profile_contract_sha256": cfg.tactile_profile_contract_sha256,
        "contact_profile_contract_sha256": contact_sha256,
        "repo_route_manifest_sha256": cfg.repo_route_manifest_sha256,
        "repo_route_manifest_source_file_sha256": (
            cfg.repo_route_manifest_source_file_sha256
        ),
        "temporal_alignment_contract_sha256": (cfg.temporal_alignment_contract_sha256),
        "temporal_alignment_source_file_sha256": (
            cfg.temporal_alignment_source_file_sha256
        ),
        "normalizer_sha256": cfg.normalizer_sha256,
        "action_schema": cfg.action_schema,
        "tactile_profile": profile,
        "resume_recipe_contract": build_agilex_resume_recipe_contract(cfg),
    }
    return cfg


__all__ = (
    "PROFILE_NAMES",
    "apply_agilex_runtime_contract",
    "apply_agilex_tactile_profile",
)
