# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Canonical Track 3.1 training and latent checkpoint identities."""

from __future__ import annotations

from collections.abc import Mapping

from n0_twam.checkpointing.identity import validate_sha256

LATENT_INVENTORY_SHA256_FIELDS = (
    "video_inventory_sha256",
    "tactile_inventory_sha256",
)
TRACK31_PROFILE_IDENTITY_SCHEMA_VERSION = 2


def _read_field(payload: object, field: str) -> object:
    if isinstance(payload, Mapping):
        return payload.get(field)
    return getattr(payload, field, None)


def validate_latent_inventory_identity(
    payload: object,
    *,
    label: str,
) -> dict[str, str]:
    """Require both current latent inventory digests from one payload."""

    return {
        field: validate_sha256(
            _read_field(payload, field),
            label=f"{label} {field}",
        )
        for field in LATENT_INVENTORY_SHA256_FIELDS
    }


def build_track31_profile_identity(config: object) -> dict[str, object] | None:
    """Return the schema-v2 strict-resume identity for one formal run."""

    profile_id = _read_field(config, "training_profile_id")
    if profile_id is None:
        return None
    required_strings = {
        "training_profile_id": profile_id,
        "run_role": _read_field(config, "run_role"),
        "train_view_id": _read_field(config, "train_view_id"),
        "sampler_coverage_mode": _read_field(config, "sampler_coverage_mode"),
        "normalizer_source_view_id": _read_field(config, "normalizer_source_view_id"),
    }
    for field, value in required_strings.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"Track 3.1 profile identity requires {field}")
    validation_view_id = _read_field(config, "validation_view_id")
    if validation_view_id is not None and (
        not isinstance(validation_view_id, str) or not validation_view_id
    ):
        raise ValueError("Track 3.1 validation_view_id must be a string or None")
    if required_strings["run_role"] == "final_refit" and validation_view_id is not None:
        raise ValueError("Track 3.1 final_refit must not have a validation view")
    validation_view_sha256 = _read_field(config, "validation_view_sha256")
    if validation_view_id is None:
        if validation_view_sha256 is not None:
            raise ValueError("final profile has an unexpected validation view SHA256")
    else:
        validation_view_sha256 = validate_sha256(
            validation_view_sha256,
            label="Track 3.1 validation view SHA256",
        )
    latent_identity = validate_latent_inventory_identity(
        config,
        label="Track 3.1 profile",
    )
    return {
        "schema_version": TRACK31_PROFILE_IDENTITY_SCHEMA_VERSION,
        **required_strings,
        "validation_view_id": validation_view_id,
        "source_manifest_sha256": validate_sha256(
            _read_field(config, "source_manifest_sha256"),
            label="Track 3.1 source manifest SHA256",
        ),
        "normalizer_sha256": validate_sha256(
            _read_field(config, "normalizer_sha256"),
            label="Track 3.1 normalizer SHA256",
        ),
        "train_view_sha256": validate_sha256(
            _read_field(config, "train_view_sha256"),
            label="Track 3.1 train view SHA256",
        ),
        "validation_view_sha256": validation_view_sha256,
        "normalizer_source_view_sha256": validate_sha256(
            _read_field(config, "normalizer_source_view_sha256"),
            label="Track 3.1 normalizer source view SHA256",
        ),
        **latent_identity,
    }


def validate_checkpoint_latent_inventory_binding(
    *,
    expected: object,
    payloads: tuple[tuple[str, Mapping[str, object]], ...],
) -> dict[str, str]:
    """Require every formal checkpoint payload to bind the same inventories."""

    expected_identity = validate_latent_inventory_identity(
        expected,
        label="current Track 3.1 artifacts",
    )
    for label, payload in payloads:
        if (
            validate_latent_inventory_identity(payload, label=label)
            != expected_identity
        ):
            raise ValueError(f"{label} latent inventory identity mismatch")
        profile_identity = payload.get("training_profile_identity")
        if (
            not isinstance(profile_identity, Mapping)
            or profile_identity.get("schema_version")
            != TRACK31_PROFILE_IDENTITY_SCHEMA_VERSION
            or validate_latent_inventory_identity(
                profile_identity,
                label=f"{label} training profile identity",
            )
            != expected_identity
        ):
            raise ValueError(f"{label} training profile identity is not schema v2")
    return expected_identity


__all__ = (
    "LATENT_INVENTORY_SHA256_FIELDS",
    "TRACK31_PROFILE_IDENTITY_SCHEMA_VERSION",
    "build_track31_profile_identity",
    "validate_checkpoint_latent_inventory_binding",
    "validate_latent_inventory_identity",
)
