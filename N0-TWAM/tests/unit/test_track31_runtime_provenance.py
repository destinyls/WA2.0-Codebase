"""Regression tests for formal checkpoint runtime provenance."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from n0_twam.checkpointing.runtime_provenance import (
    LOCAL_EXECUTION_TIER,
    capture_checkpoint_invocation_identity,
    capture_formal_checkpoint_provenance,
    capture_runtime_source_identity,
    validate_checkpoint_runtime_provenance,
)
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.checkpointing.strict_resume import (
    build_sidecar_inventory,
    expected_sidecar_paths,
)
from tests.unit.test_track31_checkpoint_review_regressions import _write_resume

RUNTIME_SOURCE_IDENTITY = {
    "schema_version": 1,
    "code_manifest_sha256": "a" * 64,
    "image_id": "sha256:" + "b" * 64,
    "overlay_manifest_sha256": "c" * 64,
    "empty_embedding_sha256": "d" * 64,
}
CHECKPOINT_INVOCATION_IDENTITY = {
    "schema_version": 1,
    "invocation_id": "stage-a-20",
    "launch_manifest_sha256": "e" * 64,
}
LOCAL_RUNTIME_SOURCE_IDENTITY = {
    "schema_version": 2,
    "execution_tier": LOCAL_EXECUTION_TIER,
    "code_manifest_sha256": "1" * 64,
    "environment_manifest_sha256": "2" * 64,
    "empty_embedding_sha256": "3" * 64,
    "package_version": "0.1.0",
}
LOCAL_CHECKPOINT_INVOCATION_IDENTITY = {
    "schema_version": 2,
    "execution_tier": LOCAL_EXECUTION_TIER,
    "invocation_id": "public-stage-a",
    "launch_receipt_sha256": "4" * 64,
}


def _environment() -> dict[str, str]:
    return {
        "N0_TRACK31_SOURCE_MANIFEST_SHA256": "a" * 64,
        "N0_TRACK31_IMAGE_ID": "sha256:" + "b" * 64,
        "N0_TRACK31_OVERLAY_MANIFEST_SHA256": "c" * 64,
        "N0_EMPTY_EMBEDDING_SHA256": "d" * 64,
        "N0_TRACK31_INVOCATION_ID": "stage-a-20",
        "N0_TRACK31_LAUNCH_MANIFEST_SHA256": "e" * 64,
    }


def _sidecars() -> list[tuple[str, dict[str, object]]]:
    payload = {
        "runtime_source_identity": copy.deepcopy(RUNTIME_SOURCE_IDENTITY),
        "checkpoint_invocation_identity": copy.deepcopy(CHECKPOINT_INVOCATION_IDENTITY),
    }
    return [
        ("train metadata", copy.deepcopy(payload)),
        ("training state", copy.deepcopy(payload)),
        ("completion marker", copy.deepcopy(payload)),
    ]


def test_capture_formal_runtime_provenance_uses_distinct_code_manifest_name() -> None:
    source = capture_runtime_source_identity(_environment())
    invocation = capture_checkpoint_invocation_identity(_environment())

    assert source == RUNTIME_SOURCE_IDENTITY
    assert "source_manifest_sha256" not in source
    assert invocation == CHECKPOINT_INVOCATION_IDENTITY


def test_capture_local_package_provenance_never_fabricates_image_identity(
    tmp_path: Path,
) -> None:
    empty_embedding = tmp_path / "empty_emb.pt"
    empty_embedding.write_bytes(b"public-empty-embedding")
    empty_sha256 = hashlib.sha256(empty_embedding.read_bytes()).hexdigest()
    environment = {
        "N0_TRACK31_EXECUTION_TIER": LOCAL_EXECUTION_TIER,
        "N0_TRACK31_CODE_MANIFEST_SHA256": "1" * 64,
        "N0_TRACK31_ENVIRONMENT_MANIFEST_SHA256": "2" * 64,
        "N0_EMPTY_EMBEDDING": str(empty_embedding),
        "N0_EMPTY_EMBEDDING_SHA256": empty_sha256,
        "N0_TRACK31_PACKAGE_VERSION": "0.1.0",
        "N0_TRACK31_INVOCATION_ID": "public-stage-a",
        "N0_TRACK31_LAUNCH_RECEIPT_SHA256": "4" * 64,
    }

    source = capture_runtime_source_identity(environment)
    invocation = capture_checkpoint_invocation_identity(environment)

    assert source == {
        **LOCAL_RUNTIME_SOURCE_IDENTITY,
        "empty_embedding_sha256": empty_sha256,
    }
    assert invocation == LOCAL_CHECKPOINT_INVOCATION_IDENTITY
    assert "image_id" not in source
    assert "launch_manifest_sha256" not in invocation


def test_local_package_capture_rejects_claimed_empty_embedding_hash(
    tmp_path: Path,
) -> None:
    empty_embedding = tmp_path / "empty_emb.pt"
    empty_embedding.write_bytes(b"public-empty-embedding")

    with pytest.raises(ValueError, match="does not match"):
        capture_runtime_source_identity(
            {
                "N0_TRACK31_EXECUTION_TIER": LOCAL_EXECUTION_TIER,
                "N0_TRACK31_CODE_MANIFEST_SHA256": "1" * 64,
                "N0_TRACK31_ENVIRONMENT_MANIFEST_SHA256": "2" * 64,
                "N0_EMPTY_EMBEDDING": str(empty_embedding),
                "N0_EMPTY_EMBEDDING_SHA256": "3" * 64,
                "N0_TRACK31_PACKAGE_VERSION": "0.1.0",
            }
        )


def test_cross_tier_strict_resume_is_rejected() -> None:
    with pytest.raises(ValueError, match="current launch"):
        validate_checkpoint_runtime_provenance(
            _sidecars(),
            current_runtime_source_identity=LOCAL_RUNTIME_SOURCE_IDENTITY,
        )


@pytest.mark.parametrize(
    "missing_name",
    (
        "N0_TRACK31_SOURCE_MANIFEST_SHA256",
        "N0_TRACK31_IMAGE_ID",
        "N0_TRACK31_OVERLAY_MANIFEST_SHA256",
        "N0_EMPTY_EMBEDDING_SHA256",
        "N0_TRACK31_INVOCATION_ID",
        "N0_TRACK31_LAUNCH_MANIFEST_SHA256",
    ),
)
def test_formal_runtime_provenance_fails_closed_when_env_is_missing(
    missing_name: str,
) -> None:
    environment = _environment()
    del environment[missing_name]

    with pytest.raises(ValueError, match=missing_name):
        if missing_name in {
            "N0_TRACK31_INVOCATION_ID",
            "N0_TRACK31_LAUNCH_MANIFEST_SHA256",
        }:
            capture_checkpoint_invocation_identity(environment)
        else:
            capture_runtime_source_identity(environment)


def test_resume_requires_current_source_identity_to_equal_parent() -> None:
    current = copy.deepcopy(RUNTIME_SOURCE_IDENTITY)
    current["overlay_manifest_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="current launch"):
        validate_checkpoint_runtime_provenance(
            _sidecars(),
            current_runtime_source_identity=current,
        )


@pytest.mark.parametrize(
    "field",
    ("runtime_source_identity", "checkpoint_invocation_identity"),
)
def test_checkpoint_provenance_must_match_across_all_three_sidecars(
    field: str,
) -> None:
    sidecars = _sidecars()
    identity = sidecars[1][1][field]
    assert isinstance(identity, dict)
    digest_field = (
        "code_manifest_sha256"
        if field == "runtime_source_identity"
        else "launch_manifest_sha256"
    )
    identity[digest_field] = "f" * 64

    with pytest.raises(ValueError, match=f"{field} differs across sidecars"):
        validate_checkpoint_runtime_provenance(sidecars)


def test_new_invocation_may_differ_from_parent_invocation() -> None:
    parent_source, parent_invocation = validate_checkpoint_runtime_provenance(
        _sidecars(),
        current_runtime_source_identity=RUNTIME_SOURCE_IDENTITY,
    )
    current_invocation = capture_checkpoint_invocation_identity(
        {
            **_environment(),
            "N0_TRACK31_INVOCATION_ID": "stage-a-25",
            "N0_TRACK31_LAUNCH_MANIFEST_SHA256": "f" * 64,
        }
    )

    assert parent_source == RUNTIME_SOURCE_IDENTITY
    assert parent_invocation != current_invocation


def test_legacy_callers_can_skip_formal_capture() -> None:
    runtime_source_identity, checkpoint_invocation_identity = (
        capture_formal_checkpoint_provenance(
            formal_track31=False,
            environ={},
        )
    )

    assert runtime_source_identity is None
    assert checkpoint_invocation_identity is None


def test_legacy_non_track31_strict_snapshot_does_not_require_provenance(
    tmp_path: Path,
) -> None:
    _write_resume(tmp_path)
    metadata_path = tmp_path / "train_meta.json"
    state_path = tmp_path / "training_state.json"
    completion_path = tmp_path / "checkpoint_complete.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    for payload in (metadata, state, completion):
        payload.pop("runtime_source_identity", None)
        payload.pop("checkpoint_invocation_identity", None)
        payload.pop("training_profile_identity", None)
    for field in (
        "training_profile_id",
        "run_role",
        "track31_artifacts",
    ):
        metadata.pop(field, None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    completion["sidecar_inventory"] = build_sidecar_inventory(
        tmp_path,
        expected_sidecar_paths(1),
    )
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    snapshot = capture_strict_checkpoint_snapshot(tmp_path)

    assert snapshot.runtime_source_identity is None
    assert snapshot.checkpoint_invocation_identity is None


def test_strict_checkpoint_identity_binds_all_sidecars_and_completion(
    tmp_path: Path,
) -> None:
    _write_resume(tmp_path)

    snapshot = capture_strict_checkpoint_snapshot(tmp_path)
    identity = build_strict_checkpoint_identity(snapshot)

    assert identity["completion_file"] == {
        "relative_path": "checkpoint_complete.json",
        "size_bytes": snapshot.completion_file.size_bytes,
        "sha256": snapshot.completion_file.sha256,
    }
    assert identity["sidecar_inventory"] == snapshot.sidecars.inventory
    assert identity["runtime_source_identity"] == snapshot.runtime_source_identity
    assert identity["checkpoint_invocation_identity"] == (
        snapshot.checkpoint_invocation_identity
    )
    assert isinstance(identity["identity_sha256"], str)
    assert len(identity["identity_sha256"]) == 64
