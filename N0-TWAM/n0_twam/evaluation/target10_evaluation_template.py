# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Request template helpers for Target-10 one-command evaluation."""

from __future__ import annotations

REQUEST_SCHEMA_VERSION = 1
REFERENCE_REQUEST_SCHEMA_VERSION = 6

DEFAULT_UNIFIED_EVALUATION_VIEW_ID = "frozen_target10_v1"
_DEFAULT_EVALUATION_VIEW_FILE = f"{DEFAULT_UNIFIED_EVALUATION_VIEW_ID}.json"


def target10_request_template() -> dict[str, object]:
    """Return a schema-valid template that requires real input identities."""

    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "checkpoint": "/absolute/path/to/checkpoint_step_1500",
        "checkpoint_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        "vae": "/absolute/path/to/wan_vae",
        "evaluation_view_manifest": (
            f"/absolute/path/to/{_DEFAULT_EVALUATION_VIEW_FILE}"
        ),
        "raw_root": "/absolute/path/to/UniVTAC",
        "manifest": "/absolute/path/to/universe_manifest_v4.json",
        "conversion_report": "/absolute/path/to/conversion_report.json",
        "official_metric_script": "/absolute/path/to/val_psnr_ssim.py",
        "official_metric_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        "output_root": "/absolute/path/to/target10_step1500_eval",
        "config_name": "track31_univtac",
        "device": "cuda:0",
        "fps": 10,
        "hip_visible_devices": "0",
        "n_steps": 50,
        "seed": 2026,
    }


def target10_v18_step1500_request_template() -> dict[str, object]:
    """Return a portable compatibility template for the v18/1500 evaluator."""

    base_model = "/absolute/path/to/n0-twam-base"
    artifact_root = "/absolute/path/to/track31-artifacts"
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "checkpoint": "/absolute/path/to/checkpoint_step_1500",
        "checkpoint_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        "vae": f"{base_model}/vae",
        "base_model": base_model,
        "lerobot_root": "/absolute/path/to/lerobot",
        "normalizer": f"{artifact_root}/normalizers/qpos8_final759_v1.json",
        "normalizer_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        "normalizer_source_view": f"{artifact_root}/views/stage_a_final759_v1.json",
        "evaluation_view_manifest": (
            f"{artifact_root}/views/{_DEFAULT_EVALUATION_VIEW_FILE}"
        ),
        "raw_root": "/absolute/path/to/UniVTAC",
        "manifest": f"{artifact_root}/universe_manifest_v4.json",
        "conversion_report": f"{artifact_root}/conversion_report.json",
        "official_metric_script": "/absolute/path/to/val_psnr_ssim.py",
        "official_metric_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        "output_root": "/absolute/path/to/outputs/target10_v18_step1500",
        "config_name": "track31_univtac",
        "device": "cuda:0",
        "fps": 10,
        "hip_visible_devices": "0",
        "n_steps": 50,
        "seed": 2026,
    }


def target10_v18_step1500_reference_request_template() -> dict[str, object]:
    """Return the portable strict reference request used by the public CLI."""

    metric_root = "/absolute/path/to/evaluation-assets/reference-contract"
    golden_root = "/absolute/path/to/evaluation-assets/published-golden"
    base_model = "/absolute/path/to/n0-twam-base"
    artifact_root = "/absolute/path/to/track31-artifacts"
    return {
        "schema_version": REFERENCE_REQUEST_SCHEMA_VERSION,
        "checkpoint": "/absolute/path/to/checkpoint_step_1500",
        "checkpoint_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        "vae": f"{base_model}/vae",
        "base_model": base_model,
        "empty_embedding": f"{base_model}/empty_emb.pt",
        "empty_embedding_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        "lerobot_root": "/absolute/path/to/lerobot",
        "normalizer": f"{artifact_root}/normalizers/qpos8_final759_v1.json",
        "normalizer_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        "normalizer_source_view": (f"{artifact_root}/views/stage_a_final759_v1.json"),
        "evaluation_view_manifest": (
            f"{artifact_root}/views/{_DEFAULT_EVALUATION_VIEW_FILE}"
        ),
        "reference_metadata": f"{metric_root}/metadata_val.json",
        "raw_root": "/absolute/path/to/UniVTAC",
        "manifest": f"{artifact_root}/universe_manifest_v4.json",
        "conversion_report": f"{artifact_root}/conversion_report.json",
        "official_metric_script": f"{metric_root}/stage1_holdout_metrics.py",
        "golden_root": golden_root,
        "golden_manifest": f"{golden_root}/golden_manifest.json",
        "golden_manifest_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        "calibration_policy": "require_published_golden",
        "output_root": "/absolute/path/to/outputs/target10-reference",
        "config_name": "track31_univtac",
        "device": "cuda:0",
        "hip_visible_devices": "0",
        "n_steps": 50,
        "seed": 2026,
    }


def target10_reference_request_template() -> dict[str, object]:
    """Return the public portable request for unified Target-10 evaluation."""

    return target10_v18_step1500_reference_request_template()


__all__ = (
    "target10_reference_request_template",
    "target10_request_template",
    "target10_v18_step1500_reference_request_template",
    "target10_v18_step1500_request_template",
)
