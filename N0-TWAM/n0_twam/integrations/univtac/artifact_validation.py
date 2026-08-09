# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Internal validators for view-bound UniVTAC conversion artifacts."""

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from .dataset_view import DatasetView
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    READABLE_MANIFEST_SCHEMA_VERSIONS,
    UniVTACDatasetManifest,
)
from .materialized_validation import verify_materialized_episode_tables
from .normalizer import (
    LEGACY_NORMALIZER_SCHEMA_VERSION,
    NORMALIZER_SCHEMA_VERSION,
)
from .schema import IMAGE_PATHS, OUTPUT_COLOR_SPACE, SOURCE_IMAGE_ENCODING_CONTRACT
from .temporal_contract import STRICT_MONOTONIC_POLICY

PHYSICAL_SOURCE_SPLITS = {
    "train759": "train",
    "frozen40": "validation",
    "quarantine1": "quarantine",
}
MANIFEST_CANONICAL_KEYS = (
    "schema_version",
    "action_schema",
    "source_image_encoding_contract",
    "output_color_space",
    "tasks",
    "entries",
)
NORMALIZER_V1_KEYS = (
    "schema_version",
    "action_schema",
    "action_q01",
    "action_q99",
    "state_q01",
    "state_q99",
    "observed_action_min",
    "observed_action_max",
    "sample_count",
    "train_paths_sha256",
    "source_manifest_sha256",
)
NORMALIZER_V2_KEYS = NORMALIZER_V1_KEYS + (
    "source_view_id",
    "source_view_sha256",
)
MATERIALIZATION_MARKER_NAME = ".materialization_state.json"


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest_payload(manifest: dict[str, Any]) -> tuple[str, int]:
    """Verify any supported manifest generation without reading source HDF5."""

    try:
        schema_version = int(manifest["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("dataset manifest has invalid schema version") from exc
    if schema_version not in READABLE_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError("unsupported dataset manifest schema version")
    if schema_version == MANIFEST_SCHEMA_VERSION and any(
        key not in manifest for key in MANIFEST_CANONICAL_KEYS
    ):
        raise ValueError("current dataset manifest canonical payload is incomplete")
    canonical = {
        key: manifest[key] for key in MANIFEST_CANONICAL_KEYS if key in manifest
    }
    actual_sha256 = _sha256(canonical)
    if manifest.get("manifest_sha256") != actual_sha256:
        raise ValueError("dataset manifest sha256 verification failed")
    if manifest.get("action_schema") != "qpos8_next_step":
        raise ValueError("dataset manifest is not qpos8_next_step")
    if (
        "source_image_encoding_contract" in manifest
        and manifest["source_image_encoding_contract"] != SOURCE_IMAGE_ENCODING_CONTRACT
    ):
        raise ValueError("dataset manifest has an unsupported image contract")
    if (
        "output_color_space" in manifest
        and manifest["output_color_space"] != OUTPUT_COLOR_SPACE
    ):
        raise ValueError("dataset manifest output color space must be RGB")
    return actual_sha256, schema_version


def verify_normalizer_payload(
    normalizer: dict[str, Any],
    *,
    manifest_sha256: str,
) -> tuple[str, int]:
    """Verify legacy or view-bound qpos8 normalizer fields and digest."""

    try:
        schema_version = int(normalizer["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("qpos8 normalizer has invalid schema version") from exc
    keys = {
        LEGACY_NORMALIZER_SCHEMA_VERSION: NORMALIZER_V1_KEYS,
        NORMALIZER_SCHEMA_VERSION: NORMALIZER_V2_KEYS,
    }.get(schema_version)
    if keys is None or any(key not in normalizer for key in keys):
        raise ValueError("unsupported or incomplete qpos8 normalizer schema")
    if set(normalizer) != set(keys) | {"normalizer_sha256"}:
        raise ValueError("qpos8 normalizer fields do not match its schema")
    actual_sha256 = _sha256({key: normalizer[key] for key in keys})
    if normalizer.get("normalizer_sha256") != actual_sha256:
        raise ValueError("qpos8 normalizer sha256 verification failed")
    if normalizer.get("source_manifest_sha256") != manifest_sha256:
        raise ValueError("normalizer does not belong to the dataset manifest")
    if normalizer.get("action_schema") != "qpos8_next_step":
        raise ValueError("normalizer is not qpos8_next_step")
    channel_fields = (
        "action_q01",
        "action_q99",
        "state_q01",
        "state_q99",
        "observed_action_min",
        "observed_action_max",
    )
    try:
        channels = {
            field: tuple(float(value) for value in normalizer[field])
            for field in channel_fields
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("qpos8 normalizer channels must be numeric") from exc
    if any(len(values) != 8 for values in channels.values()):
        raise ValueError("qpos8 normalizer must contain exactly 8 channels")
    if any(
        not math.isfinite(value) for values in channels.values() for value in values
    ):
        raise ValueError("qpos8 normalizer channels must be finite")
    if any(
        lower > upper
        for lower, upper in zip(channels["action_q01"], channels["action_q99"])
    ):
        raise ValueError("qpos8 action normalizer quantiles are reversed")
    sample_count = int(normalizer.get("sample_count", 0))
    if sample_count <= 0:
        raise ValueError("qpos8 normalizer sample_count must be positive")
    return actual_sha256, sample_count


def _first_symlink_component(path: Path) -> Path | None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except (FileNotFoundError, NotADirectoryError):
            return None
        if stat.S_ISLNK(metadata.st_mode):
            return current
    return None


def verify_materialization_marker(
    materialization_root: Path,
    *,
    manifest_sha256: str,
    conversion_report_sha256: str,
    allow_transaction_state: bool = False,
) -> None:
    """Reject incomplete materializations while accepting marker-free legacy roots."""

    symlink = _first_symlink_component(materialization_root)
    if symlink is not None:
        raise ValueError(f"materialization root contains a symlink: {symlink}")
    resolved_root = materialization_root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(resolved_root)
    marker_path = resolved_root / MATERIALIZATION_MARKER_NAME
    if marker_path.is_symlink():
        raise ValueError("materialization marker must not be a symlink")
    if not marker_path.exists():
        return
    try:
        marker_metadata = marker_path.lstat()
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read materialization marker") from exc
    if not stat.S_ISREG(marker_metadata.st_mode) or not isinstance(payload, dict):
        raise ValueError("materialization marker must be a regular JSON object")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported materialization marker schema")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("materialization marker manifest identity mismatch")

    raw_target = payload.get("target_root")
    raw_staging = payload.get("staging_root")
    if not isinstance(raw_target, str) or not isinstance(raw_staging, str):
        raise ValueError("materialization marker root identities are invalid")
    target_root = Path(raw_target).resolve(strict=False)
    staging_root = Path(raw_staging).resolve(strict=False)
    if target_root == staging_root or target_root.parent != staging_root.parent:
        raise ValueError("materialization marker roots are not sibling identities")

    status = payload.get("status")
    phase = payload.get("phase")
    if allow_transaction_state and status in {"incomplete", "ready_to_publish"}:
        if resolved_root not in {target_root, staging_root}:
            raise ValueError("materialization marker does not belong to this root")
        reported_sha256 = payload.get("conversion_report_sha256")
        if status == "ready_to_publish" and reported_sha256 != conversion_report_sha256:
            raise ValueError("materialization marker conversion report mismatch")
        return

    if status != "complete" or phase != "complete":
        raise ValueError("materialization marker is not complete")
    if target_root != resolved_root:
        raise ValueError("materialization marker target root mismatch")
    if payload.get("conversion_report_sha256") != conversion_report_sha256:
        raise ValueError("materialization marker conversion report identity mismatch")


def _table_inventory(root: Path) -> list[dict[str, object]]:
    symlink = _first_symlink_component(root)
    if symlink is not None:
        raise ValueError(f"LeRobot inventory contains a symlink: {symlink}")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    files: list[Path] = []
    for directory in (resolved / "meta", resolved / "data", resolved / "videos"):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"LeRobot inventory directory is invalid: {directory}")
        for path in directory.rglob("*"):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"LeRobot inventory contains a symlink: {path}")
            try:
                path.resolve(strict=True).relative_to(resolved)
            except ValueError as exc:
                raise ValueError(f"LeRobot inventory escapes its root: {path}") from exc
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"LeRobot inventory entry is not regular: {path}")
            files.append(path)
    files.sort()
    return [
        {
            "relative_path": path.relative_to(resolved).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in files
    ]


def verify_conversion_report(
    report: dict[str, Any],
    *,
    manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify the report digest and return its logical conversions."""

    keys = (
        "schema_version",
        "source_manifest_sha256",
        "source_image_encoding_contract",
        "output_color_space",
        "conversions",
    )
    if any(key not in report for key in keys):
        raise ValueError("conversion report is incomplete")
    if set(report) != set(keys) | {"conversion_report_sha256"}:
        raise ValueError("conversion report fields do not match its schema")
    if int(report.get("schema_version", -1)) != 2:
        raise ValueError("unsupported conversion report schema version")
    report_sha256 = _sha256({key: report[key] for key in keys})
    if report.get("conversion_report_sha256") != report_sha256:
        raise ValueError("conversion report sha256 verification failed")
    if report.get("source_manifest_sha256") != manifest_sha256:
        raise ValueError("conversion report does not belong to dataset manifest")
    if report.get("source_image_encoding_contract") != SOURCE_IMAGE_ENCODING_CONTRACT:
        raise ValueError("conversion report has an unsupported image contract")
    if report.get("output_color_space") != OUTPUT_COLOR_SPACE:
        raise ValueError("conversion report output color space must be RGB")
    conversions = report.get("conversions")
    if not isinstance(conversions, dict) or not conversions:
        raise ValueError("conversion report must contain conversions")
    return conversions, report_sha256


def verify_conversion(
    *,
    logical_split: str,
    conversion: object,
    expected_entries: list[dict[str, Any]],
    dataset_root: Path,
) -> dict[str, Any]:
    """Verify one materialized LeRobot repo and its source episode order."""

    if not isinstance(conversion, dict):
        raise ValueError(f"invalid conversion report split: {logical_split}")
    if int(conversion.get("schema_version", -1)) != 2:
        raise ValueError(f"{logical_split} conversion has an invalid schema")
    if conversion.get("action_schema") != "qpos8_next_step":
        raise ValueError(f"{logical_split} conversion has an invalid action schema")
    split_root = (dataset_root / logical_split).resolve(strict=True)
    reported_root = Path(str(conversion.get("output_root", ""))).resolve(strict=True)
    if reported_root != split_root or split_root.name != logical_split:
        raise ValueError(
            f"conversion output_root does not match {logical_split} dataset"
        )
    expected_paths = [entry["relative_path"] for entry in expected_entries]
    expected_hashes = [entry["sha256"] for entry in expected_entries]
    if conversion.get("source_relative_paths") != expected_paths:
        raise ValueError(
            f"{logical_split} conversion source path set does not match manifest"
        )
    if conversion.get("source_sha256") != expected_hashes:
        raise ValueError(
            f"{logical_split} conversion source hashes do not match manifest"
        )
    if int(conversion.get("episode_count", -1)) != len(expected_entries):
        raise ValueError(
            f"{logical_split} conversion episode count does not match manifest"
        )
    temporal_flags = ["usable_source_range" in entry for entry in expected_entries]
    if any(temporal_flags) and not all(temporal_flags):
        raise ValueError("manifest mixes legacy and temporal source contracts")
    if all(temporal_flags):
        expected_temporal_selections = []
        expected_frame_count = 0
        for entry in expected_entries:
            raw_range = entry.get("usable_source_range")
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raise ValueError("manifest entry lacks a usable source range")
            usable_start, usable_end = (int(value) for value in raw_range)
            converted_length = usable_end - usable_start - 1
            expected_frame_count += converted_length
            expected_temporal_selections.append(
                {
                    "relative_path": entry["relative_path"],
                    "raw_length": int(entry["length"]),
                    "usable_source_range": [usable_start, usable_end],
                    "converted_length": converted_length,
                    "dropped_prefix_rows": usable_start,
                    "dropped_suffix_rows": int(entry["length"]) - usable_end,
                    "step_discontinuities_after_rows": entry.get(
                        "step_discontinuities_after_rows"
                    ),
                    "temporal_policy": entry.get("temporal_policy"),
                }
            )
        if conversion.get("temporal_contract") != "content_addressed_source_range_v1":
            raise ValueError(f"{logical_split} conversion lacks a temporal contract")
        if conversion.get("source_temporal_selections") != expected_temporal_selections:
            raise ValueError(
                f"{logical_split} conversion temporal selections do not match manifest"
            )
        if int(conversion.get("frame_count", -1)) != expected_frame_count:
            raise ValueError(f"{logical_split} conversion frame count does not match")
    if (
        conversion.get("source_image_encoding_contract")
        != SOURCE_IMAGE_ENCODING_CONTRACT
    ):
        raise ValueError(f"{logical_split} conversion image contract does not match")
    if conversion.get("output_color_space") != OUTPUT_COLOR_SPACE:
        raise ValueError(f"{logical_split} conversion output color space must be RGB")
    current_inventory = _table_inventory(split_root)
    if conversion.get("table_inventory") != current_inventory:
        raise ValueError(
            f"{logical_split} LeRobot table inventory changed after conversion"
        )
    if all(temporal_flags):
        verify_materialized_episode_tables(
            split_root,
            expected_entries=expected_entries,
        )
    inventory_paths = [str(item["relative_path"]) for item in current_inventory]
    video_paths = [
        item
        for item in inventory_paths
        if item.startswith("videos/") and item.endswith(".mp4")
    ]
    expected_video_count = len(expected_entries) * len(IMAGE_PATHS)
    if len(video_paths) != expected_video_count:
        raise ValueError(
            f"{logical_split} video inventory count mismatch: "
            f"{len(video_paths)} vs {expected_video_count}"
        )
    for feature_name, _ in IMAGE_PATHS:
        count = sum(f"/{feature_name}/" in f"/{item}" for item in video_paths)
        if count != len(expected_entries):
            raise ValueError(
                f"{logical_split} video inventory for {feature_name} is incomplete"
            )
    return conversion


def verify_view(
    view: DatasetView,
    *,
    manifest: UniVTACDatasetManifest,
    required_role: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Bind every view identity and LeRobot ID to a manifest physical split."""

    if view.role != required_role:
        label = (
            "active train view" if required_role == "training" else "validation view"
        )
        raise ValueError(f"{label} must have role {required_role}")
    if view.source_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("dataset view does not belong to dataset manifest")
    source_split = PHYSICAL_SOURCE_SPLITS.get(view.physical_split)
    if source_split is None:
        raise ValueError("dataset view has unsupported physical split")
    manifest_entries = [
        entry for entry in manifest.entries if entry.split == source_split
    ]
    by_path = {
        entry.relative_path: (episode_id, entry)
        for episode_id, entry in enumerate(manifest_entries)
    }
    raw_entries: list[dict[str, Any]] = []
    for item in view.entries:
        indexed = by_path.get(item.relative_path)
        if indexed is None:
            raise ValueError("dataset view episode is outside its physical split")
        episode_id, record = indexed
        if (
            item.realpath,
            item.source_sha256,
            item.task,
            item.source_split,
            item.source_episode_id,
            item.lerobot_episode_id,
        ) != (
            record.realpath,
            record.sha256,
            record.task,
            source_split,
            record.episode_id,
            episode_id,
        ):
            raise ValueError("dataset view episode identity does not match manifest")
        raw_entries.append(
            {
                "relative_path": record.relative_path,
                "sha256": record.sha256,
                "lerobot_episode_id": episode_id,
                "length": record.length,
                "converted_length": record.converted_length,
            }
        )
        if required_role == "frozen_evaluation" and (
            record.temporal_policy != STRICT_MONOTONIC_POLICY
            or record.usable_start != 0
            or record.usable_end != record.length
        ):
            raise ValueError("frozen evaluation episodes must retain full timelines")
    return source_split, raw_entries


def reject_view_overlap(train_view: DatasetView, validation_view: DatasetView) -> None:
    """Reject leakage through any stable episode identity."""

    dimensions = (
        "relative_path",
        "realpath",
        "source_sha256",
        "lerobot_episode_id",
    )
    for field in dimensions:
        train_values = {getattr(entry, field) for entry in train_view.entries}
        validation_values = {getattr(entry, field) for entry in validation_view.entries}
        if train_values & validation_values:
            raise ValueError(
                f"training and internal validation views overlap by {field}"
            )
