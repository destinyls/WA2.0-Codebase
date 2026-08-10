# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Manifest-bound raw HDF5 source validation for Stage B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from n0_twam.evaluation.tactile_prediction_artifact import (
    VerifiedTactilePredictionArtifact,
)
from n0_twam.integrations.univtac.artifact_validation import (
    verify_conversion_report,
    verify_manifest_payload,
)
from n0_twam.integrations.univtac.manifest import (
    MANIFEST_SCHEMA_VERSION,
    load_dataset_manifest,
)
from n0_twam.integrations.univtac.schema import (
    UNIVTAC_ALL_TASKS,
    UniVTACEpisodeRecord,
)
from n0_twam.integrations.univtac.temporal_contract import STRICT_MONOTONIC_POLICY


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"JSON contains non-finite constant {value}")


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def load_bound_source_contract(
    *,
    artifact: VerifiedTactilePredictionArtifact,
    manifest_path: Path,
    conversion_report_path: Path,
) -> dict[str, dict[str, object]]:
    """Verify logical manifest/conversion identities and validation path mapping."""

    resolved_manifest_path = manifest_path.resolve(strict=True)
    manifest = _read_json_object(resolved_manifest_path)
    conversion = _read_json_object(conversion_report_path.resolve(strict=True))
    loaded_manifest = load_dataset_manifest(
        resolved_manifest_path,
        verify_sources=False,
    )
    if loaded_manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError("raw evaluation requires the current manifest schema")
    manifest_sha, schema_version = verify_manifest_payload(manifest)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError("raw evaluation requires the current manifest schema")
    conversions, conversion_sha = verify_conversion_report(
        conversion,
        manifest_sha256=manifest_sha,
    )
    if artifact.metadata["source_manifest_sha256"] != manifest_sha:
        raise ValueError("raw evaluation manifest identity is invalid")
    if artifact.metadata["conversion_report_sha256"] != conversion_sha:
        raise ValueError("raw evaluation conversion identity is invalid")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not isinstance(conversions, dict):
        raise ValueError("raw evaluation source contracts are incomplete")
    validation_records = tuple(
        entry for entry in loaded_manifest.entries if entry.split == "validation"
    )
    if not validation_records:
        raise ValueError("raw evaluation manifest has no validation episodes")
    if any(
        record.temporal_policy != STRICT_MONOTONIC_POLICY
        or record.usable_start != 0
        or record.usable_end != record.length
        or record.step_discontinuities_after_rows
        for record in validation_records
    ):
        raise ValueError("raw evaluation requires full monotonic validation episodes")
    validation_entries = [record.to_json_dict() for record in validation_records]
    expected_paths = [str(entry.get("relative_path")) for entry in validation_entries]
    expected_hashes = [str(entry.get("sha256")) for entry in validation_entries]
    candidates = [
        payload
        for payload in conversions.values()
        if isinstance(payload, dict)
        and payload.get("source_relative_paths") == expected_paths
        and payload.get("source_sha256") == expected_hashes
    ]
    if len(candidates) != 1:
        raise ValueError(
            "manifest validation episodes must map to exactly one conversion repo"
        )
    candidate = candidates[0]
    if (
        int(candidate.get("schema_version", -1)) != 2
        or candidate.get("action_schema") != "qpos8_next_step"
    ):
        raise ValueError("raw evaluation conversion schema is invalid")
    expected_temporal = [
        {
            "relative_path": entry["relative_path"],
            "raw_length": entry["length"],
            "usable_source_range": entry["usable_source_range"],
            "converted_length": int(entry["length"]) - 1,
            "dropped_prefix_rows": 0,
            "dropped_suffix_rows": 0,
            "step_discontinuities_after_rows": [],
            "temporal_policy": STRICT_MONOTONIC_POLICY,
        }
        for entry in validation_entries
    ]
    if (
        candidate.get("temporal_contract") != "content_addressed_source_range_v1"
        or candidate.get("source_temporal_selections") != expected_temporal
        or int(candidate.get("frame_count", -1))
        != sum(int(entry["length"]) - 1 for entry in validation_entries)
    ):
        raise ValueError("raw evaluation conversion temporal contract is invalid")
    by_path = {str(entry["relative_path"]): entry for entry in validation_entries}
    return by_path


def record_from_manifest(
    *,
    raw_root: Path,
    relative_path: str,
    expected: Mapping[str, object],
) -> UniVTACEpisodeRecord:
    """Re-audit one raw episode and compare every manifest-bound source field."""

    from n0_twam.integrations.univtac.hdf5_reader import audit_episode

    record = audit_episode(
        raw_root,
        relative_path,
        split="validation",
        hash_file=True,
        allowed_tasks=UNIVTAC_ALL_TASKS,
    )
    actual = record.to_json_dict()
    mismatched = [
        key
        for key in (
            "relative_path",
            "task",
            "split",
            "length",
            "length_source",
            "joint_shape",
            "image_shapes",
            "size_bytes",
            "sha256",
            "usable_source_range",
            "step_discontinuities_after_rows",
            "temporal_policy",
        )
        if actual[key] != expected.get(key)
    ]
    if mismatched:
        raise ValueError("raw HDF5 disagrees with manifest: " + ", ".join(mismatched))
    if (
        record.temporal_policy != STRICT_MONOTONIC_POLICY
        or record.usable_start != 0
        or record.usable_end != record.length
    ):
        raise ValueError("raw evaluation episode does not have a full timeline")
    return record
