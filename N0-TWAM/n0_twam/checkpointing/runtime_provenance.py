# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed runtime provenance for formal and packaged Track 3.1 runs."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping, Sequence

from .identity import validate_sha256

RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION = 1
CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION = 1
LOCAL_RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION = 2
LOCAL_CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION = 2
FORMAL_EXECUTION_TIER = "formal_hcu"
LOCAL_EXECUTION_TIER = "local_package"

_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "code_manifest_sha256",
        "image_id",
        "overlay_manifest_sha256",
        "empty_embedding_sha256",
    }
)
_INVOCATION_FIELDS = frozenset(
    {
        "schema_version",
        "invocation_id",
        "launch_manifest_sha256",
    }
)
_LOCAL_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "execution_tier",
        "code_manifest_sha256",
        "environment_manifest_sha256",
        "empty_embedding_sha256",
        "package_version",
    }
)
_LOCAL_INVOCATION_FIELDS = frozenset(
    {
        "schema_version",
        "execution_tier",
        "invocation_id",
        "launch_receipt_sha256",
    }
)
_INVOCATION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_PACKAGE_VERSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,63}$")


def _required_environment_value(
    environ: Mapping[str, str],
    name: str,
) -> str:
    value = environ.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required environment variable is unset: {name}")
    return value


def _verify_local_empty_embedding(
    environ: Mapping[str, str], expected_sha256: str
) -> None:
    path = os.path.expanduser(
        _required_environment_value(environ, "N0_EMPTY_EMBEDDING")
    )
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("local empty embedding must be a regular non-symlink file")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = os.lstat(path)
    fingerprints = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    final_fingerprints = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if fingerprints != final_fingerprints:
        raise ValueError("local empty embedding changed while hashing")
    if digest.hexdigest() != expected_sha256:
        raise ValueError("local empty embedding SHA256 does not match the request")


def validate_runtime_source_identity(payload: object) -> dict[str, object]:
    """Validate the stable execution-source identity stored by a checkpoint."""

    if not isinstance(payload, Mapping):
        raise ValueError("runtime_source_identity has an invalid field set")
    schema_version = payload.get("schema_version")
    if schema_version == LOCAL_RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION:
        if set(payload) != _LOCAL_SOURCE_FIELDS:
            raise ValueError("runtime_source_identity has an invalid field set")
        if payload.get("execution_tier") != LOCAL_EXECUTION_TIER:
            raise ValueError("local runtime source has an invalid execution tier")
        package_version = payload.get("package_version")
        if not isinstance(
            package_version, str
        ) or not _PACKAGE_VERSION_PATTERN.fullmatch(package_version):
            raise ValueError("local runtime source package_version is invalid")
        return {
            "schema_version": LOCAL_RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION,
            "execution_tier": LOCAL_EXECUTION_TIER,
            "code_manifest_sha256": validate_sha256(
                payload.get("code_manifest_sha256"),
                label="local runtime code manifest SHA256",
            ),
            "environment_manifest_sha256": validate_sha256(
                payload.get("environment_manifest_sha256"),
                label="local runtime environment manifest SHA256",
            ),
            "empty_embedding_sha256": validate_sha256(
                payload.get("empty_embedding_sha256"),
                label="local runtime empty embedding SHA256",
            ),
            "package_version": package_version,
        }
    if set(payload) != _SOURCE_FIELDS:
        raise ValueError("runtime_source_identity has an invalid field set")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION
    ):
        raise ValueError("runtime_source_identity has an invalid schema version")
    image_id = payload.get("image_id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ValueError("runtime source image_id must be a full SHA256 image ID")
    validate_sha256(image_id.removeprefix("sha256:"), label="runtime source image ID")
    return {
        "schema_version": RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION,
        "code_manifest_sha256": validate_sha256(
            payload.get("code_manifest_sha256"),
            label="runtime source code manifest SHA256",
        ),
        "image_id": image_id,
        "overlay_manifest_sha256": validate_sha256(
            payload.get("overlay_manifest_sha256"),
            label="runtime source overlay manifest SHA256",
        ),
        "empty_embedding_sha256": validate_sha256(
            payload.get("empty_embedding_sha256"),
            label="runtime source empty embedding SHA256",
        ),
    }


def validate_checkpoint_invocation_identity(payload: object) -> dict[str, object]:
    """Validate the identity of the invocation that produced a checkpoint."""

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint_invocation_identity has an invalid field set")
    schema_version = payload.get("schema_version")
    if schema_version == LOCAL_CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION:
        if set(payload) != _LOCAL_INVOCATION_FIELDS:
            raise ValueError("checkpoint_invocation_identity has an invalid field set")
        if payload.get("execution_tier") != LOCAL_EXECUTION_TIER:
            raise ValueError(
                "local checkpoint invocation has an invalid execution tier"
            )
        invocation_id = payload.get("invocation_id")
        if not isinstance(invocation_id, str) or not _INVOCATION_PATTERN.fullmatch(
            invocation_id
        ):
            raise ValueError("checkpoint invocation_id is invalid")
        return {
            "schema_version": LOCAL_CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION,
            "execution_tier": LOCAL_EXECUTION_TIER,
            "invocation_id": invocation_id,
            "launch_receipt_sha256": validate_sha256(
                payload.get("launch_receipt_sha256"),
                label="checkpoint launch receipt SHA256",
            ),
        }
    if set(payload) != _INVOCATION_FIELDS:
        raise ValueError("checkpoint_invocation_identity has an invalid field set")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint_invocation_identity has an invalid schema version")
    invocation_id = payload.get("invocation_id")
    if not isinstance(invocation_id, str) or not _INVOCATION_PATTERN.fullmatch(
        invocation_id
    ):
        raise ValueError("checkpoint invocation_id is invalid")
    return {
        "schema_version": CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "launch_manifest_sha256": validate_sha256(
            payload.get("launch_manifest_sha256"),
            label="checkpoint launch manifest SHA256",
        ),
    }


def capture_runtime_source_identity(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Capture the selected formal or local execution source identity."""

    values = os.environ if environ is None else environ
    execution_tier = values.get("N0_TRACK31_EXECUTION_TIER", FORMAL_EXECUTION_TIER)
    if execution_tier == LOCAL_EXECUTION_TIER:
        empty_embedding_sha256 = validate_sha256(
            _required_environment_value(values, "N0_EMPTY_EMBEDDING_SHA256"),
            label="local runtime empty embedding SHA256",
        )
        _verify_local_empty_embedding(values, empty_embedding_sha256)
        return validate_runtime_source_identity(
            {
                "schema_version": LOCAL_RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION,
                "execution_tier": LOCAL_EXECUTION_TIER,
                "code_manifest_sha256": _required_environment_value(
                    values, "N0_TRACK31_CODE_MANIFEST_SHA256"
                ),
                "environment_manifest_sha256": _required_environment_value(
                    values, "N0_TRACK31_ENVIRONMENT_MANIFEST_SHA256"
                ),
                "empty_embedding_sha256": empty_embedding_sha256,
                "package_version": _required_environment_value(
                    values, "N0_TRACK31_PACKAGE_VERSION"
                ),
            }
        )
    if execution_tier != FORMAL_EXECUTION_TIER:
        raise ValueError(f"unsupported Track 3.1 execution tier: {execution_tier}")
    return validate_runtime_source_identity(
        {
            "schema_version": RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION,
            "code_manifest_sha256": _required_environment_value(
                values, "N0_TRACK31_SOURCE_MANIFEST_SHA256"
            ),
            "image_id": _required_environment_value(values, "N0_TRACK31_IMAGE_ID"),
            "overlay_manifest_sha256": _required_environment_value(
                values, "N0_TRACK31_OVERLAY_MANIFEST_SHA256"
            ),
            "empty_embedding_sha256": _required_environment_value(
                values, "N0_EMPTY_EMBEDDING_SHA256"
            ),
        }
    )


def capture_checkpoint_invocation_identity(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Capture the selected formal or local invocation identity."""

    values = os.environ if environ is None else environ
    execution_tier = values.get("N0_TRACK31_EXECUTION_TIER", FORMAL_EXECUTION_TIER)
    if execution_tier == LOCAL_EXECUTION_TIER:
        return validate_checkpoint_invocation_identity(
            {
                "schema_version": LOCAL_CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION,
                "execution_tier": LOCAL_EXECUTION_TIER,
                "invocation_id": _required_environment_value(
                    values, "N0_TRACK31_INVOCATION_ID"
                ),
                "launch_receipt_sha256": _required_environment_value(
                    values, "N0_TRACK31_LAUNCH_RECEIPT_SHA256"
                ),
            }
        )
    if execution_tier != FORMAL_EXECUTION_TIER:
        raise ValueError(f"unsupported Track 3.1 execution tier: {execution_tier}")
    return validate_checkpoint_invocation_identity(
        {
            "schema_version": CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION,
            "invocation_id": _required_environment_value(
                values, "N0_TRACK31_INVOCATION_ID"
            ),
            "launch_manifest_sha256": _required_environment_value(
                values, "N0_TRACK31_LAUNCH_MANIFEST_SHA256"
            ),
        }
    )


def capture_formal_checkpoint_provenance(
    *,
    formal_track31: bool,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Capture provenance for Track 3.1 training, preserving the legacy API."""

    if not formal_track31:
        return None, None
    return (
        capture_runtime_source_identity(environ),
        capture_checkpoint_invocation_identity(environ),
    )


def validate_checkpoint_runtime_provenance(
    payloads: Sequence[tuple[str, Mapping[str, object]]],
    *,
    current_runtime_source_identity: object | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Require identical valid provenance in train/state/completion sidecars."""

    if len(payloads) != 3:
        raise ValueError("checkpoint provenance requires exactly three sidecars")
    source_identities: list[dict[str, object]] = []
    invocation_identities: list[dict[str, object]] = []
    for label, payload in payloads:
        try:
            source_identities.append(
                validate_runtime_source_identity(payload.get("runtime_source_identity"))
            )
            invocation_identities.append(
                validate_checkpoint_invocation_identity(
                    payload.get("checkpoint_invocation_identity")
                )
            )
        except ValueError as error:
            raise ValueError(
                f"{label} checkpoint provenance is invalid: {error}"
            ) from error
    if any(value != source_identities[0] for value in source_identities[1:]):
        raise ValueError("checkpoint runtime_source_identity differs across sidecars")
    if any(value != invocation_identities[0] for value in invocation_identities[1:]):
        raise ValueError(
            "checkpoint checkpoint_invocation_identity differs across sidecars"
        )
    if current_runtime_source_identity is not None:
        current = validate_runtime_source_identity(current_runtime_source_identity)
        if source_identities[0] != current:
            raise ValueError(
                "resume runtime_source_identity differs from the current launch"
            )
    return source_identities[0], invocation_identities[0]


__all__ = (
    "CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION",
    "FORMAL_EXECUTION_TIER",
    "LOCAL_CHECKPOINT_INVOCATION_IDENTITY_SCHEMA_VERSION",
    "LOCAL_EXECUTION_TIER",
    "LOCAL_RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION",
    "RUNTIME_SOURCE_IDENTITY_SCHEMA_VERSION",
    "capture_checkpoint_invocation_identity",
    "capture_formal_checkpoint_provenance",
    "capture_runtime_source_identity",
    "validate_checkpoint_invocation_identity",
    "validate_checkpoint_runtime_provenance",
    "validate_runtime_source_identity",
)
