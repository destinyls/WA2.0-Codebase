# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Formal UniVTAC artifact identity validation for a Stage A parent."""

from __future__ import annotations

from collections.abc import Mapping

from n0_twam.configs.twam_track31_training_profiles import (
    DEVELOPMENT_RUN_ROLE,
    MULTITASK_PRETRAIN_PROFILE,
)
from n0_twam.data.track31_training_identity import build_track31_profile_identity

from .identity import validate_sha256

_ARTIFACT_SHA256_FIELDS = (
    "manifest_sha256",
    "normalizer_sha256",
    "conversion_report_sha256",
    "train_view_sha256",
    "normalizer_source_view_sha256",
    "video_inventory_sha256",
    "tactile_inventory_sha256",
)
_CURRENT_ARTIFACT_IDENTITY_FIELDS = (
    "manifest_sha256",
    "normalizer_sha256",
    "conversion_report_sha256",
    "normalizer_source_view_id",
    "normalizer_source_view_sha256",
    "video_inventory_sha256",
    "tactile_inventory_sha256",
)


def _required_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _stage_a_view_ids(run_role: str) -> tuple[str, str | None]:
    if run_role == DEVELOPMENT_RUN_ROLE:
        return "stage_a_dev719_v1", "internal_dev40_v1"
    return "stage_a_final759_v1", None


def validate_stage_a_artifact_identity(
    train_meta: Mapping[str, object],
    *,
    run_role: str,
    expected_track31_artifacts: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Bind the parent profile to its exact train/validation views."""

    if train_meta.get("training_profile_id") != MULTITASK_PRETRAIN_PROFILE:
        raise ValueError("Stage A parent training profile is incompatible")
    if train_meta.get("run_role") != run_role:
        raise ValueError("Stage A parent run role is incompatible")
    artifacts = dict(
        _required_mapping(
            train_meta.get("track31_artifacts"),
            label="Stage A parent Track 3.1 artifacts",
        )
    )
    train_view_id, validation_view_id = _stage_a_view_ids(run_role)
    if (
        train_meta.get("train_view_id") != train_view_id
        or train_meta.get("validation_view_id") != validation_view_id
        or artifacts.get("train_view_id") != train_view_id
        or artifacts.get("validation_view_id") != validation_view_id
        or artifacts.get("normalizer_source_view_id") != train_view_id
    ):
        raise ValueError("Stage A parent artifact view identity mismatch")
    for field in _ARTIFACT_SHA256_FIELDS:
        validate_sha256(
            artifacts.get(field),
            label=f"Stage A parent artifact {field}",
        )
    validation_view_sha256 = artifacts.get("validation_view_sha256")
    if validation_view_id is None:
        if validation_view_sha256 is not None:
            raise ValueError("Stage A final parent has a validation view SHA256")
    else:
        validate_sha256(
            validation_view_sha256,
            label="Stage A parent validation_view_sha256",
        )
    if artifacts["normalizer_source_view_sha256"] != artifacts["train_view_sha256"]:
        raise ValueError("Stage A parent normalizer source differs from its train view")

    if expected_track31_artifacts is not None:
        for field in _CURRENT_ARTIFACT_IDENTITY_FIELDS:
            expected_value = expected_track31_artifacts.get(field)
            actual_value = (
                artifacts["train_view_id"]
                if field == "normalizer_source_view_id"
                else artifacts.get(field)
            )
            if actual_value != expected_value:
                raise ValueError(
                    f"Stage A parent artifact identity mismatch for {field}"
                )
        if artifacts.get("validation_view_id") != expected_track31_artifacts.get(
            "parent_validation_view_id"
        ) or artifacts.get("validation_view_sha256") != expected_track31_artifacts.get(
            "parent_validation_view_sha256"
        ):
            raise ValueError("Stage A parent validation view identity mismatch")

    profile_identity = build_track31_profile_identity(
        {
            "training_profile_id": MULTITASK_PRETRAIN_PROFILE,
            "run_role": run_role,
            "train_view_id": train_view_id,
            "sampler_coverage_mode": "pad_global",
            "normalizer_source_view_id": train_view_id,
            "validation_view_id": validation_view_id,
            "source_manifest_sha256": artifacts["manifest_sha256"],
            "normalizer_sha256": artifacts["normalizer_sha256"],
            "train_view_sha256": artifacts["train_view_sha256"],
            "validation_view_sha256": validation_view_sha256,
            "normalizer_source_view_sha256": artifacts["normalizer_source_view_sha256"],
            "video_inventory_sha256": artifacts["video_inventory_sha256"],
            "tactile_inventory_sha256": artifacts["tactile_inventory_sha256"],
        }
    )
    if profile_identity is None:
        raise RuntimeError("Stage A profile identity was not constructed")
    if train_meta.get("training_profile_identity") != profile_identity:
        raise ValueError("Stage A parent training profile identity is incompatible")
    return artifacts, profile_identity


__all__ = ("validate_stage_a_artifact_identity",)
