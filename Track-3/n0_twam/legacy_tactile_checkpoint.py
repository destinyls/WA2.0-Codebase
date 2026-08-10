# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed validation for pre-profile tactile checkpoint metadata."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from n0_twam.data.track31_training_identity import build_track31_profile_identity

_OLD_TRAINABILITY_FIELDS = {
    "schema_version",
    "policy",
    "tactile_mode",
    "frozen_parameter_names",
    "frozen_parameter_count",
    "frozen_numel",
    "trainable_parameter_count",
    "trainable_numel",
    "contract_sha256",
}
_TRACK31_ARTIFACT_FIELDS = {
    "manifest_sha256",
    "normalizer_sha256",
    "train_episode_count",
    "validation_episode_count",
    "normalizer_sample_count",
    "action_q01",
    "action_q99",
    "conversion_report_sha256",
    "train_view_id",
    "train_view_sha256",
    "validation_view_id",
    "validation_view_sha256",
    "normalizer_source_view_id",
    "normalizer_source_view_sha256",
    "parent_validation_view_id",
    "parent_validation_view_sha256",
    "video_inventory_sha256",
    "tactile_inventory_sha256",
    "latent_segment_count",
    "video_latent_artifact_count",
    "tactile_latent_artifact_count",
}
_PROFILE_IDS = {"multitask_pretrain_v1", "target_finetune_v1"}
_TACTILE_PARAMETER_PREFIXES = (
    "tactile_",
    "sensor_id_embed.",
    "local_tactile_",
    "contact_gate.",
    "mot.experts.tactile.",
)
_TRACK31_VIEW_CONTRACTS = {
    ("multitask_pretrain_v1", "development"): (
        "stage_a_dev719_v1",
        "internal_dev40_v1",
        "stage_a_dev719_v1",
        None,
        719,
        40,
    ),
    ("multitask_pretrain_v1", "final_refit"): (
        "stage_a_final759_v1",
        None,
        "stage_a_final759_v1",
        None,
        759,
        0,
    ),
    ("target_finetune_v1", "development"): (
        "stage_b_dev180_v1",
        "internal_target_dev10_v1",
        "stage_a_dev719_v1",
        "internal_dev40_v1",
        180,
        10,
    ),
    ("target_finetune_v1", "final_refit"): (
        "stage_b_final190_v1",
        None,
        "stage_a_final759_v1",
        None,
        190,
        0,
    ),
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_int(value: object, *, allow_zero: bool = False) -> bool:
    lower_bound = 0 if allow_zero else 1
    return (
        isinstance(value, int) and not isinstance(value, bool) and value >= lower_bound
    )


def _finite_vector(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    )


def validate_legacy_trainability_contract(
    payload: object,
    *,
    expected_mode: str,
    expected_policy: str,
) -> bool:
    """Validate the complete, self-hashed pre-profile trainability contract."""

    if not isinstance(payload, Mapping) or set(payload) != _OLD_TRAINABILITY_FIELDS:
        return False
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("policy") != expected_policy
        or payload.get("tactile_mode") != expected_mode
    ):
        return False
    names = payload.get("frozen_parameter_names")
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name for name in names
    ):
        return False
    frozen_count = payload.get("frozen_parameter_count")
    if (
        len(names) != len(set(names))
        or not _positive_int(frozen_count, allow_zero=True)
        or frozen_count != len(names)
    ):
        return False
    frozen_numel = payload.get("frozen_numel")
    if not _positive_int(frozen_numel, allow_zero=True):
        return False
    if not _positive_int(payload.get("trainable_parameter_count")) or not _positive_int(
        payload.get("trainable_numel")
    ):
        return False
    if expected_mode == "enabled" and (names or frozen_numel != 0):
        return False
    if expected_mode == "disabled" and (not names or frozen_numel == 0):
        return False
    if expected_mode == "disabled" and any(
        not name.removeprefix("module.").startswith(_TACTILE_PARAMETER_PREFIXES)
        for name in names
    ):
        return False
    digest = payload.get("contract_sha256")
    if not _is_sha256(digest):
        return False
    canonical = {key: payload[key] for key in payload if key != "contract_sha256"}
    expected_digest = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return digest == expected_digest


def validate_legacy_track31_checkpoint_identity(
    train_meta: Mapping[str, object],
    *,
    expected_tactile_keys: Sequence[str],
) -> bool:
    """Validate the complete formal Track 3.1 identity stored before profiles."""

    profile = train_meta.get("training_profile_identity")
    artifacts = train_meta.get("track31_artifacts")
    if not isinstance(profile, Mapping) or not isinstance(artifacts, Mapping):
        return False
    if set(artifacts) != _TRACK31_ARTIFACT_FIELDS:
        return False
    if type(profile.get("schema_version")) is not int:
        return False
    profile_id = train_meta.get("training_profile_id")
    if (
        profile_id not in _PROFILE_IDS
        or profile.get("training_profile_id") != profile_id
    ):
        return False
    run_role = train_meta.get("run_role")
    if not isinstance(run_role, str):
        return False
    view_contract = _TRACK31_VIEW_CONTRACTS.get((profile_id, run_role))
    if view_contract is None:
        return False
    (
        expected_train_view,
        expected_validation_view,
        expected_normalizer_view,
        expected_parent_view,
        expected_train_count,
        expected_validation_count,
    ) = view_contract
    try:
        rebuilt_profile = build_track31_profile_identity(profile)
    except (TypeError, ValueError):
        return False
    if rebuilt_profile is None or dict(profile) != rebuilt_profile:
        return False
    for field in (
        "manifest_sha256",
        "normalizer_sha256",
        "conversion_report_sha256",
        "train_view_sha256",
        "normalizer_source_view_sha256",
        "video_inventory_sha256",
        "tactile_inventory_sha256",
    ):
        if not _is_sha256(artifacts.get(field)):
            return False
    validation_view_id = artifacts.get("validation_view_id")
    validation_view_sha256 = artifacts.get("validation_view_sha256")
    if validation_view_id is None:
        if validation_view_sha256 is not None:
            return False
    elif not isinstance(validation_view_id, str) or not _is_sha256(
        validation_view_sha256
    ):
        return False
    for field in (
        "normalizer_sample_count",
        "latent_segment_count",
        "video_latent_artifact_count",
        "tactile_latent_artifact_count",
    ):
        if not _positive_int(artifacts.get(field)):
            return False
    train_count = artifacts.get("train_episode_count")
    validation_count = artifacts.get("validation_episode_count")
    if (
        not _positive_int(train_count)
        or not _positive_int(validation_count, allow_zero=True)
        or train_count != expected_train_count
        or validation_count != expected_validation_count
    ):
        return False
    if not _finite_vector(artifacts.get("action_q01")) or not _finite_vector(
        artifacts.get("action_q99")
    ):
        return False
    if len(artifacts["action_q01"]) != len(artifacts["action_q99"]):
        return False
    if len(artifacts["action_q01"]) != 8 or any(
        lower > upper
        for lower, upper in zip(artifacts["action_q01"], artifacts["action_q99"])
    ):
        return False
    parent_view_id = artifacts.get("parent_validation_view_id")
    parent_view_sha256 = artifacts.get("parent_validation_view_sha256")
    if parent_view_id != expected_parent_view:
        return False
    if expected_parent_view is None:
        if parent_view_sha256 is not None:
            return False
    elif not _is_sha256(parent_view_sha256):
        return False
    if (
        profile.get("sampler_coverage_mode") != "pad_global"
        or profile.get("train_view_id") != expected_train_view
        or profile.get("validation_view_id") != expected_validation_view
        or profile.get("normalizer_source_view_id") != expected_normalizer_view
    ):
        return False
    links = {
        "manifest_sha256": "source_manifest_sha256",
        "normalizer_sha256": "normalizer_sha256",
        "train_view_id": "train_view_id",
        "train_view_sha256": "train_view_sha256",
        "validation_view_id": "validation_view_id",
        "validation_view_sha256": "validation_view_sha256",
        "normalizer_source_view_id": "normalizer_source_view_id",
        "normalizer_source_view_sha256": "normalizer_source_view_sha256",
        "video_inventory_sha256": "video_inventory_sha256",
        "tactile_inventory_sha256": "tactile_inventory_sha256",
    }
    if any(artifacts.get(left) != profile.get(right) for left, right in links.items()):
        return False
    for field in ("video_inventory_sha256", "tactile_inventory_sha256"):
        if train_meta.get(field) != profile.get(field):
            return False
    if any(
        train_meta.get(field) != profile.get(field)
        for field in (
            "training_profile_id",
            "run_role",
            "train_view_id",
            "validation_view_id",
        )
    ):
        return False
    keys = train_meta.get("tactile_keys")
    return isinstance(keys, list) and keys == list(expected_tactile_keys)


__all__ = (
    "validate_legacy_track31_checkpoint_identity",
    "validate_legacy_trainability_contract",
)
