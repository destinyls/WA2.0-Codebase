# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Content-addressed completion inventories for Track 3.1 latent encoding."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from ._latent_inventory_payload import (
    LATENT_INVENTORY_SCHEMA_VERSION,
    TRACK31_FORMAL_SPLITS,
    TRACK31_TACTILE_PAYLOAD_PROVENANCE_SCHEMA_VERSION,
    TRACK31_TACTILE_ENCODING_CONTRACT,
    TRACK31_TACTILE_KEYS,
    TRACK31_TACTILE_MODES,
    TRACK31_VIDEO_ENCODING_CONTRACT,
    TRACK31_VIDEO_PAYLOAD_PROVENANCE_SCHEMA_VERSION,
    TRACK31_VIDEO_KEYS,
    InventoryKind,
    build_inventory_core,
    build_track31_payload_provenance,
    build_track31_source_video_identity,
    canonical_bytes,
    invalidate_latent_inventory,
    inventory_path,
    load_expected_segments,
    validate_existing_track31_payload,
)
from ._latent_payload_provenance import TRACK31_PAYLOAD_PROVENANCE_SCHEMA_VERSION
from .encoder_source_identity import (
    build_encoder_source_identity,
    load_encoder_source_identity,
)

__all__ = (
    "TRACK31_TACTILE_KEYS",
    "TRACK31_TACTILE_ENCODING_CONTRACT",
    "TRACK31_TACTILE_MODES",
    "TRACK31_VIDEO_KEYS",
    "TRACK31_VIDEO_ENCODING_CONTRACT",
    "TRACK31_PAYLOAD_PROVENANCE_SCHEMA_VERSION",
    "TRACK31_TACTILE_PAYLOAD_PROVENANCE_SCHEMA_VERSION",
    "TRACK31_VIDEO_PAYLOAD_PROVENANCE_SCHEMA_VERSION",
    "build_encoder_source_identity",
    "load_encoder_source_identity",
    "build_track31_payload_provenance",
    "build_track31_source_video_identity",
    "canonical_bytes",
    "finalize_latent_inventory",
    "invalidate_latent_inventory",
    "inventory_path",
    "load_expected_segments",
    "validate_existing_track31_payload",
    "validate_latent_inventory",
    "validate_latent_inventory_pair",
    "validate_track31_training_dataset_isolation",
)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def validate_track31_training_dataset_isolation(dataset_root: Path) -> Path:
    """Return the physical train repo only when frozen40 is not mounted.

    A broken symlink is still a mounted-name capability and must therefore be
    rejected with ``lexists`` semantics.  The active train repo itself must be
    a real directory so a container cannot redirect it to the sealed cohort.
    """

    root = Path(dataset_root).resolve(strict=True)
    frozen_root = root / "frozen40"
    if os.path.lexists(frozen_root):
        raise ValueError(
            "formal Track 3.1 training root must not contain or symlink frozen40"
        )
    train_root = root / "train759"
    if train_root.is_symlink() or not train_root.is_dir():
        raise ValueError(
            "formal Track 3.1 training root requires a regular train759 directory"
        )
    resolved_train_root = train_root.resolve(strict=True)
    try:
        resolved_train_root.relative_to(root)
    except ValueError as exc:
        raise ValueError("train759 resolved outside the training dataset root") from exc
    return resolved_train_root


def finalize_latent_inventory(
    dataset_root: Path,
    *,
    kind: InventoryKind,
    split: str,
    manifest_sha256: str,
    conversion_report_sha256: str,
    encoder_source_identity: object,
) -> dict[str, object]:
    """Write ``status=ready`` atomically only after every artifact verifies."""

    # A failed rebuild must never leave a marker from an older payload set.
    invalidate_latent_inventory(dataset_root, kind)
    core = build_inventory_core(
        dataset_root,
        kind=kind,
        split=split,
        manifest_sha256=manifest_sha256,
        conversion_report_sha256=conversion_report_sha256,
        encoder_source_identity=encoder_source_identity,
    )
    payload = dict(core)
    payload["inventory_sha256"] = hashlib.sha256(canonical_bytes(core)).hexdigest()
    _write_json_atomic(inventory_path(dataset_root, kind), payload)
    return payload


def validate_latent_inventory(
    dataset_root: Path,
    *,
    kind: InventoryKind,
    expected_split: str,
    expected_manifest_sha256: str,
    expected_conversion_report_sha256: str,
    expected_encoder_source_identity: object,
) -> dict[str, object]:
    """Rebuild the inventory and require current byte/contract equality."""

    path = inventory_path(dataset_root, kind)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing latent completion inventory: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read latent inventory: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("latent inventory must be a JSON object")
    recorded_digest = payload.pop("inventory_sha256", None)
    if recorded_digest != hashlib.sha256(canonical_bytes(payload)).hexdigest():
        raise ValueError("latent inventory digest is inconsistent")
    if (
        expected_split in TRACK31_FORMAL_SPLITS
        and payload.get("schema_version") != LATENT_INVENTORY_SCHEMA_VERSION
    ):
        raise ValueError(
            "formal Track 3.1 requires latent inventory schema 3; re-encode "
            "all payloads instead of migrating an old ready marker"
        )
    if (
        payload.get("kind") != kind
        or payload.get("split") != expected_split
        or payload.get("manifest_sha256") != expected_manifest_sha256
        or payload.get("conversion_report_sha256") != expected_conversion_report_sha256
        or payload.get("encoder_source_identity") != expected_encoder_source_identity
    ):
        raise ValueError("latent inventory source/split contract mismatch")
    expected = build_inventory_core(
        dataset_root,
        kind=kind,
        split=expected_split,
        manifest_sha256=expected_manifest_sha256,
        conversion_report_sha256=expected_conversion_report_sha256,
        encoder_source_identity=expected_encoder_source_identity,
    )
    if payload != expected:
        raise ValueError("latent inventory does not match current split/contract/files")
    payload["inventory_sha256"] = recorded_digest
    return payload


def _temporal_projection(inventory: dict[str, object]) -> list[dict[str, object]]:
    keys = (
        "episode_index",
        "start_frame",
        "end_frame",
        "latent_num_frames",
        "source_frame_count",
        "frame_ids_sha256",
    )
    segments = inventory.get("segments")
    if not isinstance(segments, list):
        raise ValueError("latent inventory segments must be a list")
    projection: list[dict[str, object]] = []
    for segment in segments:
        if not isinstance(segment, dict) or any(key not in segment for key in keys):
            raise ValueError("latent inventory segment has an invalid schema")
        projection.append({key: segment[key] for key in keys})
    return projection


def _artifact_count(inventory: dict[str, object]) -> int:
    segments = inventory.get("segments")
    if not isinstance(segments, list):
        raise ValueError("latent inventory segments must be a list")
    count = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("latent inventory segment has an invalid schema")
        artifacts = segment.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("latent inventory artifacts must be a list")
        count += len(artifacts)
    return count


def validate_latent_inventory_pair(
    dataset_root: Path,
    *,
    expected_split: str,
    expected_manifest_sha256: str,
    expected_conversion_report_sha256: str,
    expected_encoder_source_identity: object,
) -> dict[str, object]:
    """Validate both modality inventories and their shared temporal grid."""

    video = validate_latent_inventory(
        dataset_root,
        kind="video",
        expected_split=expected_split,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_conversion_report_sha256=expected_conversion_report_sha256,
        expected_encoder_source_identity=expected_encoder_source_identity,
    )
    tactile = validate_latent_inventory(
        dataset_root,
        kind="tactile",
        expected_split=expected_split,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_conversion_report_sha256=expected_conversion_report_sha256,
        expected_encoder_source_identity=expected_encoder_source_identity,
    )
    video_temporal = _temporal_projection(video)
    tactile_temporal = _temporal_projection(tactile)
    if video_temporal != tactile_temporal:
        raise ValueError(
            "video and tactile latent inventories are temporally misaligned"
        )
    return {
        "video_inventory_sha256": video["inventory_sha256"],
        "tactile_inventory_sha256": tactile["inventory_sha256"],
        "segment_count": len(video_temporal),
        "video_artifact_count": _artifact_count(video),
        "tactile_artifact_count": _artifact_count(tactile),
    }
