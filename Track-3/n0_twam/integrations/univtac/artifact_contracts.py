# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Verify persisted UniVTAC manifest/normalizer pairs before training."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from n0_twam.data.latent_inventory import (
    validate_latent_inventory_pair,
    validate_track31_training_dataset_isolation,
)

from .artifact_validation import (
    PHYSICAL_SOURCE_SPLITS,
    reject_view_overlap,
    verify_conversion,
    verify_conversion_report,
    verify_manifest_payload,
    verify_materialization_marker,
    verify_normalizer_payload,
    verify_view,
)
from .dataset_view import load_dataset_view
from .manifest import MANIFEST_SCHEMA_VERSION, load_dataset_manifest
from .normalizer import NORMALIZER_SCHEMA_VERSION


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read artifact JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


@dataclass(frozen=True)
class VerifiedTrack31Artifacts:
    """Content identities required to start one Track 3.1 training run."""

    manifest_sha256: str
    normalizer_sha256: str
    train_episode_count: int
    validation_episode_count: int
    normalizer_sample_count: int
    action_q01: tuple[float, ...]
    action_q99: tuple[float, ...]
    conversion_report_sha256: str | None = None
    train_view_id: str | None = None
    train_view_sha256: str | None = None
    validation_view_id: str | None = None
    validation_view_sha256: str | None = None
    normalizer_source_view_id: str | None = None
    normalizer_source_view_sha256: str | None = None
    parent_validation_view_id: str | None = None
    parent_validation_view_sha256: str | None = None
    video_inventory_sha256: str | None = None
    tactile_inventory_sha256: str | None = None
    latent_segment_count: int | None = None
    video_latent_artifact_count: int | None = None
    tactile_latent_artifact_count: int | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "normalizer_sha256": self.normalizer_sha256,
            "train_episode_count": self.train_episode_count,
            "validation_episode_count": self.validation_episode_count,
            "normalizer_sample_count": self.normalizer_sample_count,
            "action_q01": list(self.action_q01),
            "action_q99": list(self.action_q99),
            "conversion_report_sha256": self.conversion_report_sha256,
            "train_view_id": self.train_view_id,
            "train_view_sha256": self.train_view_sha256,
            "validation_view_id": self.validation_view_id,
            "validation_view_sha256": self.validation_view_sha256,
            "normalizer_source_view_id": self.normalizer_source_view_id,
            "normalizer_source_view_sha256": self.normalizer_source_view_sha256,
            "parent_validation_view_id": self.parent_validation_view_id,
            "parent_validation_view_sha256": self.parent_validation_view_sha256,
            "video_inventory_sha256": self.video_inventory_sha256,
            "tactile_inventory_sha256": self.tactile_inventory_sha256,
            "latent_segment_count": self.latent_segment_count,
            "video_latent_artifact_count": self.video_latent_artifact_count,
            "tactile_latent_artifact_count": self.tactile_latent_artifact_count,
        }

    def latent_inventory_report(self) -> dict[str, object]:
        """Return the already rehashed pair report without touching storage."""

        required: dict[str, object] = {
            "video_inventory_sha256": self.video_inventory_sha256,
            "tactile_inventory_sha256": self.tactile_inventory_sha256,
            "segment_count": self.latent_segment_count,
            "video_artifact_count": self.video_latent_artifact_count,
            "tactile_artifact_count": self.tactile_latent_artifact_count,
        }
        if any(value is None for value in required.values()):
            raise ValueError("Track 3.1 latent inventories were not verified")
        return required


def _latent_report_count(
    report: dict[str, object],
    key: str,
) -> int:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"latent inventory report has invalid {key}")
    return value


@dataclass(frozen=True)
class VerifiedTrack31EvaluationArtifacts:
    """Content identities required by a frozen-cohort prediction process."""

    manifest_sha256: str
    normalizer_sha256: str
    conversion_report_sha256: str
    evaluation_view_id: str
    evaluation_view_sha256: str
    evaluation_episode_count: int
    normalizer_source_view_id: str
    normalizer_source_view_sha256: str
    action_q01: tuple[float, ...]
    action_q99: tuple[float, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "normalizer_sha256": self.normalizer_sha256,
            "conversion_report_sha256": self.conversion_report_sha256,
            "evaluation_view_id": self.evaluation_view_id,
            "evaluation_view_sha256": self.evaluation_view_sha256,
            "evaluation_episode_count": self.evaluation_episode_count,
            "normalizer_source_view_id": self.normalizer_source_view_id,
            "normalizer_source_view_sha256": (self.normalizer_source_view_sha256),
            "action_q01": list(self.action_q01),
            "action_q99": list(self.action_q99),
        }


@dataclass(frozen=True)
class VerifiedTrack31PhysicalArtifacts:
    """Manifest and conversion identity for one materialized physical repo."""

    manifest_sha256: str
    conversion_report_sha256: str
    physical_split: str
    source_split: str
    episode_count: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "conversion_report_sha256": self.conversion_report_sha256,
            "physical_split": self.physical_split,
            "source_split": self.source_split,
            "episode_count": self.episode_count,
        }


def verify_track31_artifact_pair(
    *,
    manifest_path: Path,
    normalizer_path: Path,
) -> VerifiedTrack31Artifacts:
    """Fail if either persisted artifact was edited or paired incorrectly."""

    manifest = _read_json_object(manifest_path)
    normalizer = _read_json_object(normalizer_path)
    actual_manifest_sha256, schema_version = verify_manifest_payload(manifest)
    if schema_version == MANIFEST_SCHEMA_VERSION:
        loaded_manifest = load_dataset_manifest(
            manifest_path,
            verify_sources=False,
        )
        if loaded_manifest.manifest_sha256 != actual_manifest_sha256:
            raise ValueError("current manifest identity changed while loading")
    actual_normalizer_sha256, sample_count = verify_normalizer_payload(
        normalizer,
        manifest_sha256=actual_manifest_sha256,
    )

    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("dataset manifest entries must be a list")
    split_counts = {
        split: sum(
            isinstance(entry, dict) and entry.get("split") == split for entry in entries
        )
        for split in ("train", "validation")
    }
    if min(split_counts.values()) <= 0:
        raise ValueError("dataset manifest requires train and validation episodes")
    return VerifiedTrack31Artifacts(
        manifest_sha256=actual_manifest_sha256,
        normalizer_sha256=actual_normalizer_sha256,
        train_episode_count=split_counts["train"],
        validation_episode_count=split_counts["validation"],
        normalizer_sample_count=sample_count,
        action_q01=tuple(float(value) for value in normalizer["action_q01"]),
        action_q99=tuple(float(value) for value in normalizer["action_q99"]),
    )


def verify_track31_training_bundle(
    *,
    manifest_path: Path,
    normalizer_path: Path,
    conversion_report_path: Path,
    dataset_root: Path,
    train_view_path: Path | None = None,
    validation_view_path: Path | None = None,
    normalizer_source_view_path: Path | None = None,
    parent_validation_view_path: Path | None = None,
    encoder_source_identity: object | None = None,
) -> VerifiedTrack31Artifacts:
    """Bind artifacts to legacy splits or explicit formal training views."""

    verified = verify_track31_artifact_pair(
        manifest_path=manifest_path,
        normalizer_path=normalizer_path,
    )
    manifest_payload = _read_json_object(manifest_path)
    conversions, report_sha256 = verify_conversion_report(
        _read_json_object(conversion_report_path),
        manifest_sha256=verified.manifest_sha256,
    )
    resolved_dataset_root = dataset_root.resolve(strict=True)
    verify_materialization_marker(
        resolved_dataset_root,
        manifest_sha256=verified.manifest_sha256,
        conversion_report_sha256=report_sha256,
    )
    raw_entries = manifest_payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("dataset manifest entries must be a list")

    if train_view_path is None:
        if encoder_source_identity is not None:
            raise ValueError(
                "latent inventory verification requires formal training views"
            )
        if (
            validation_view_path is not None
            or normalizer_source_view_path is not None
            or parent_validation_view_path is not None
        ):
            raise ValueError("view paths require train_view_path")
        if set(conversions) != {"train", "validation"}:
            raise ValueError("conversion report must contain train and validation")
        for split in ("train", "validation"):
            verify_conversion(
                logical_split=split,
                conversion=conversions[split],
                expected_entries=[
                    entry for entry in raw_entries if entry.get("split") == split
                ],
                dataset_root=resolved_dataset_root,
            )
        return replace(verified, conversion_report_sha256=report_sha256)

    _, schema_version = verify_manifest_payload(manifest_payload)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError("formal dataset views require a schema-v4 manifest")
    manifest = load_dataset_manifest(manifest_path, verify_sources=False)
    train_view = load_dataset_view(train_view_path)
    source_split, selected_train = verify_view(
        train_view,
        manifest=manifest,
        required_role="training",
    )
    if source_split != "train" or train_view.physical_split != "train759":
        raise ValueError("active train view must be backed by physical train759")
    validation_view = (
        None
        if validation_view_path is None
        else load_dataset_view(validation_view_path)
    )
    if validation_view is not None:
        validation_split, _ = verify_view(
            validation_view,
            manifest=manifest,
            required_role="internal_development",
        )
        if (
            validation_split != source_split
            or validation_view.physical_split != "train759"
        ):
            raise ValueError("internal validation view must be backed by train759")
        reject_view_overlap(train_view, validation_view)

    normalizer_source_view = (
        train_view
        if normalizer_source_view_path is None
        else load_dataset_view(normalizer_source_view_path)
    )
    normalizer_split, selected_normalizer = verify_view(
        normalizer_source_view,
        manifest=manifest,
        required_role="training",
    )
    if (
        normalizer_split != source_split
        or normalizer_source_view.physical_split != "train759"
    ):
        raise ValueError("normalizer source view must be backed by train759")
    active_identities = {
        (entry["relative_path"], entry["sha256"]) for entry in selected_train
    }
    normalizer_identities = {
        (entry["relative_path"], entry["sha256"]) for entry in selected_normalizer
    }
    if not active_identities.issubset(normalizer_identities):
        raise ValueError("active training view is outside normalizer source view")

    parent_validation_view = (
        None
        if parent_validation_view_path is None
        else load_dataset_view(parent_validation_view_path)
    )
    if parent_validation_view is not None:
        parent_validation_split, _ = verify_view(
            parent_validation_view,
            manifest=manifest,
            required_role="internal_development",
        )
        if (
            parent_validation_split != source_split
            or parent_validation_view.physical_split != "train759"
        ):
            raise ValueError("parent validation view must be backed by train759")
        reject_view_overlap(normalizer_source_view, parent_validation_view)

    normalizer = _read_json_object(normalizer_path)
    if int(normalizer.get("schema_version", 0)) != NORMALIZER_SCHEMA_VERSION:
        raise ValueError("formal training requires a view-bound normalizer")
    if (
        normalizer.get("source_view_id") != normalizer_source_view.view_id
        or normalizer.get("source_view_sha256") != normalizer_source_view.view_sha256
    ):
        raise ValueError("normalizer does not belong to its declared training view")
    expected_paths_payload = "".join(
        f"{entry['relative_path']}\t{entry['sha256']}\n"
        for entry in selected_normalizer
    ).encode("utf-8")
    if (
        normalizer.get("train_paths_sha256")
        != hashlib.sha256(expected_paths_payload).hexdigest()
    ):
        raise ValueError("normalizer training path set does not match active view")
    expected_sample_count = sum(
        int(entry["converted_length"]) for entry in selected_normalizer
    )
    if int(normalizer.get("sample_count", -1)) != expected_sample_count:
        raise ValueError("normalizer sample count does not match active view")

    physical_entries = [
        entry for entry in raw_entries if entry.get("split") == source_split
    ]
    expected_paths = [entry["relative_path"] for entry in physical_entries]
    expected_hashes = [entry["sha256"] for entry in physical_entries]
    candidates = [
        (name, conversion)
        for name, conversion in conversions.items()
        if isinstance(conversion, dict)
        and conversion.get("source_relative_paths") == expected_paths
        and conversion.get("source_sha256") == expected_hashes
    ]
    if len(candidates) != 1:
        raise ValueError("physical train759 must map to exactly one conversion repo")
    logical_split, conversion = candidates[0]
    verified_conversion = verify_conversion(
        logical_split=logical_split,
        conversion=conversion,
        expected_entries=physical_entries,
        dataset_root=resolved_dataset_root,
    )
    conversion_paths = verified_conversion["source_relative_paths"]
    conversion_hashes = verified_conversion["source_sha256"]
    for entry in selected_train:
        episode_id = int(entry["lerobot_episode_id"])
        if (
            conversion_paths[episode_id] != entry["relative_path"]
            or conversion_hashes[episode_id] != entry["sha256"]
        ):
            raise ValueError("training view LeRobot episode ID mapping changed")

    latent_report: dict[str, object] | None = None
    if encoder_source_identity is not None:
        train_repo = validate_track31_training_dataset_isolation(resolved_dataset_root)
        latent_report = validate_latent_inventory_pair(
            train_repo,
            expected_split="train759",
            expected_manifest_sha256=verified.manifest_sha256,
            expected_conversion_report_sha256=report_sha256,
            expected_encoder_source_identity=encoder_source_identity,
        )

    return VerifiedTrack31Artifacts(
        manifest_sha256=verified.manifest_sha256,
        normalizer_sha256=verified.normalizer_sha256,
        train_episode_count=len(train_view.entries),
        validation_episode_count=(
            0 if validation_view is None else len(validation_view.entries)
        ),
        normalizer_sample_count=verified.normalizer_sample_count,
        action_q01=verified.action_q01,
        action_q99=verified.action_q99,
        conversion_report_sha256=report_sha256,
        train_view_id=train_view.view_id,
        train_view_sha256=train_view.view_sha256,
        validation_view_id=(
            None if validation_view is None else validation_view.view_id
        ),
        validation_view_sha256=(
            None if validation_view is None else validation_view.view_sha256
        ),
        normalizer_source_view_id=normalizer_source_view.view_id,
        normalizer_source_view_sha256=normalizer_source_view.view_sha256,
        parent_validation_view_id=(
            None if parent_validation_view is None else parent_validation_view.view_id
        ),
        parent_validation_view_sha256=(
            None
            if parent_validation_view is None
            else parent_validation_view.view_sha256
        ),
        video_inventory_sha256=(
            None
            if latent_report is None
            else str(latent_report["video_inventory_sha256"])
        ),
        tactile_inventory_sha256=(
            None
            if latent_report is None
            else str(latent_report["tactile_inventory_sha256"])
        ),
        latent_segment_count=(
            None
            if latent_report is None
            else _latent_report_count(latent_report, "segment_count")
        ),
        video_latent_artifact_count=(
            None
            if latent_report is None
            else _latent_report_count(latent_report, "video_artifact_count")
        ),
        tactile_latent_artifact_count=(
            None
            if latent_report is None
            else _latent_report_count(latent_report, "tactile_artifact_count")
        ),
    )


def verify_track31_evaluation_bundle(
    *,
    manifest_path: Path,
    normalizer_path: Path,
    conversion_report_path: Path,
    dataset_root: Path,
    evaluation_view_path: Path,
    normalizer_source_view_path: Path,
) -> VerifiedTrack31EvaluationArtifacts:
    """Verify one frozen view without opening the physical training repository."""

    verified = verify_track31_artifact_pair(
        manifest_path=manifest_path,
        normalizer_path=normalizer_path,
    )
    manifest_payload = _read_json_object(manifest_path)
    _, schema_version = verify_manifest_payload(manifest_payload)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError("formal evaluation requires a schema-v4 manifest")
    manifest = load_dataset_manifest(manifest_path, verify_sources=False)

    evaluation_view = load_dataset_view(evaluation_view_path)
    source_split, selected_evaluation = verify_view(
        evaluation_view,
        manifest=manifest,
        required_role="frozen_evaluation",
    )
    if source_split != "validation" or evaluation_view.physical_split != "frozen40":
        raise ValueError("evaluation view must be backed by physical frozen40")

    normalizer_source_view = load_dataset_view(normalizer_source_view_path)
    normalizer_split, selected_normalizer = verify_view(
        normalizer_source_view,
        manifest=manifest,
        required_role="training",
    )
    if (
        normalizer_split != "train"
        or normalizer_source_view.physical_split != "train759"
    ):
        raise ValueError("normalizer source view must be backed by train759")

    normalizer = _read_json_object(normalizer_path)
    if int(normalizer.get("schema_version", 0)) != NORMALIZER_SCHEMA_VERSION:
        raise ValueError("formal evaluation requires a view-bound normalizer")
    if (
        normalizer.get("source_view_id") != normalizer_source_view.view_id
        or normalizer.get("source_view_sha256") != normalizer_source_view.view_sha256
    ):
        raise ValueError("normalizer does not belong to its declared training view")
    expected_paths_payload = "".join(
        f"{entry['relative_path']}\t{entry['sha256']}\n"
        for entry in selected_normalizer
    ).encode("utf-8")
    if (
        normalizer.get("train_paths_sha256")
        != hashlib.sha256(expected_paths_payload).hexdigest()
    ):
        raise ValueError("normalizer training path set does not match source view")
    expected_sample_count = sum(
        int(entry["converted_length"]) for entry in selected_normalizer
    )
    if int(normalizer.get("sample_count", -1)) != expected_sample_count:
        raise ValueError("normalizer sample count does not match source view")

    conversions, report_sha256 = verify_conversion_report(
        _read_json_object(conversion_report_path),
        manifest_sha256=verified.manifest_sha256,
    )
    raw_entries = manifest_payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("dataset manifest entries must be a list")
    physical_entries = [
        entry for entry in raw_entries if entry.get("split") == source_split
    ]
    expected_paths = [entry["relative_path"] for entry in physical_entries]
    expected_hashes = [entry["sha256"] for entry in physical_entries]
    candidates = [
        (name, conversion)
        for name, conversion in conversions.items()
        if isinstance(conversion, dict)
        and conversion.get("source_relative_paths") == expected_paths
        and conversion.get("source_sha256") == expected_hashes
    ]
    if len(candidates) != 1:
        raise ValueError("physical frozen40 must map to exactly one conversion repo")
    logical_split, conversion = candidates[0]
    resolved_dataset_root = dataset_root.resolve(strict=True)
    verify_materialization_marker(
        resolved_dataset_root,
        manifest_sha256=verified.manifest_sha256,
        conversion_report_sha256=report_sha256,
    )
    verified_conversion = verify_conversion(
        logical_split=logical_split,
        conversion=conversion,
        expected_entries=physical_entries,
        dataset_root=resolved_dataset_root,
    )
    conversion_paths = verified_conversion["source_relative_paths"]
    conversion_hashes = verified_conversion["source_sha256"]
    for entry in selected_evaluation:
        episode_id = int(entry["lerobot_episode_id"])
        if (
            conversion_paths[episode_id] != entry["relative_path"]
            or conversion_hashes[episode_id] != entry["sha256"]
        ):
            raise ValueError("evaluation view LeRobot episode ID mapping changed")

    return VerifiedTrack31EvaluationArtifacts(
        manifest_sha256=verified.manifest_sha256,
        normalizer_sha256=verified.normalizer_sha256,
        conversion_report_sha256=report_sha256,
        evaluation_view_id=evaluation_view.view_id,
        evaluation_view_sha256=evaluation_view.view_sha256,
        evaluation_episode_count=len(evaluation_view.entries),
        normalizer_source_view_id=normalizer_source_view.view_id,
        normalizer_source_view_sha256=normalizer_source_view.view_sha256,
        action_q01=verified.action_q01,
        action_q99=verified.action_q99,
    )


def verify_track31_physical_bundle(
    *,
    manifest_path: Path,
    conversion_report_path: Path,
    dataset_root: Path,
    physical_split: str,
    allow_transaction_state: bool = False,
) -> VerifiedTrack31PhysicalArtifacts:
    """Verify exactly one formal physical repo without opening its sibling."""

    source_split = PHYSICAL_SOURCE_SPLITS.get(physical_split)
    if physical_split not in {"train759", "frozen40"} or source_split is None:
        raise ValueError("formal latent encoding requires train759 or frozen40")
    manifest_payload = _read_json_object(manifest_path)
    manifest_sha256, schema_version = verify_manifest_payload(manifest_payload)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError("formal latent encoding requires a schema-v4 manifest")
    loaded_manifest = load_dataset_manifest(manifest_path, verify_sources=False)
    if loaded_manifest.manifest_sha256 != manifest_sha256:
        raise ValueError("formal manifest identity changed while loading")
    raw_entries = manifest_payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("dataset manifest entries must be a list")
    expected_entries = [
        entry for entry in raw_entries if entry.get("split") == source_split
    ]
    if not expected_entries:
        raise ValueError("physical split has no manifest episodes")

    conversions, report_sha256 = verify_conversion_report(
        _read_json_object(conversion_report_path),
        manifest_sha256=manifest_sha256,
    )
    if physical_split not in conversions:
        raise ValueError(f"conversion report has no {physical_split} repo")
    resolved_repo = Path(dataset_root).resolve(strict=True)
    if resolved_repo.name != physical_split:
        raise ValueError("dataset root does not match physical split")
    verify_materialization_marker(
        resolved_repo.parent,
        manifest_sha256=manifest_sha256,
        conversion_report_sha256=report_sha256,
        allow_transaction_state=allow_transaction_state,
    )
    verify_conversion(
        logical_split=physical_split,
        conversion=conversions[physical_split],
        expected_entries=expected_entries,
        dataset_root=resolved_repo.parent,
    )
    return VerifiedTrack31PhysicalArtifacts(
        manifest_sha256=manifest_sha256,
        conversion_report_sha256=report_sha256,
        physical_split=physical_split,
        source_split=source_split,
        episode_count=len(expected_entries),
    )
