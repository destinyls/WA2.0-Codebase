# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Durable journal and startup recovery for directory publication."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from script.track3_1.materialize_publication import (
    publication_capture_path,
    publication_capture_prefix,
)
from script.track3_1.materialize_transaction import (
    file_sha256_or_none,
    rename_directory_noreplace,
    rename_directory_noreplace_bound,
    root_belongs_to_transaction,
)
from script.track3_1.materialize_writeahead import (
    CAPTURE_SUFFIX,
    FINAL_DELETE_SUFFIX,
    CapturedFile,
    _logical_prepared_temp_from_any,
    capture_path_for,
    capture_regular_file,
    final_delete_path_for,
    prepare_json_file,
    preserve_path_noreplace,
    remove_captured_file,
    remove_exact_regular_file,
)

PUBLICATION_INTENT_PREFIX = ".materialization_publication-"


@dataclass(frozen=True)
class PublicationIntent:
    """One durable source-to-target publication declaration."""

    path: Path
    sha256: str


def publication_intent_path(artifact_dir: Path, source_root: Path) -> Path:
    """Return the deterministic journal path for one staging publication."""

    return artifact_dir / f"{PUBLICATION_INTENT_PREFIX}{source_root.name}.json"


def publication_intent_present(intent_path: Path) -> bool:
    """Return whether any crash-recoverable state of an intent remains."""

    return any(
        path.exists() or path.is_symlink()
        for path in (
            intent_path,
            capture_path_for(intent_path),
            final_delete_path_for(intent_path),
        )
    )


def prepare_publication_intent(
    *,
    artifact_dir: Path,
    source_root: Path,
    target_root: Path,
    candidate_path: Path,
    manifest_sha256: str,
    marker_name: str,
    marker_file_sha256: str,
    candidate_file_sha256: str,
    source_identity: tuple[int, int],
) -> PublicationIntent:
    """Durably bind all evidence before the first publication rename."""

    capture_root = publication_capture_path(
        source_root,
        target_root,
        source_identity=source_identity,
    )
    intent_path = publication_intent_path(artifact_dir, source_root)
    intent_sha256 = prepare_json_file(
        intent_path,
        {
            "schema_version": 1,
            "status": "publication_planned",
            "source_root": str(source_root),
            "target_root": str(target_root),
            "capture_root": str(capture_root),
            "candidate_path": str(candidate_path),
            "manifest_sha256": manifest_sha256,
            "marker_name": marker_name,
            "marker_file_sha256": marker_file_sha256,
            "candidate_file_sha256": candidate_file_sha256,
            "source_device": source_identity[0],
            "source_inode": source_identity[1],
        },
    )
    return PublicationIntent(path=intent_path, sha256=intent_sha256)


def release_publication_intent(intent: PublicationIntent) -> None:
    """Remove exactly one verified publication intent after target commit."""

    remove_exact_regular_file(intent.path, expected_sha256=intent.sha256)


def publish_directory_with_intent(
    artifact_dir: Path,
    source_root: Path,
    target_root: Path,
    candidate_path: Path,
    manifest_sha256: str,
    marker_file_sha256: str,
    candidate_file_sha256: str,
    source_identity: tuple[int, int],
    rename_bound: Callable[..., None] = rename_directory_noreplace_bound,
    *,
    marker_name: str = ".materialization_state.json",
) -> None:
    """Journal, publish, verify, and retire one directory commit."""

    intent = prepare_publication_intent(
        artifact_dir=artifact_dir,
        source_root=source_root,
        target_root=target_root,
        candidate_path=candidate_path,
        manifest_sha256=manifest_sha256,
        marker_name=marker_name,
        marker_file_sha256=marker_file_sha256,
        candidate_file_sha256=candidate_file_sha256,
        source_identity=source_identity,
    )
    try:
        rename_bound(
            source_root,
            target_root,
            expected_source_identity=source_identity,
        )
    except BaseException:
        if root_belongs_to_transaction(
            target_root,
            target_root=target_root,
            staging_root=source_root,
            manifest_sha256=manifest_sha256,
            marker_name=marker_name,
            expected_root_identity=source_identity,
            expected_marker_sha256=marker_file_sha256,
        ):
            release_publication_intent(intent)
        raise
    release_publication_intent(intent)


def _parse_intent(captured: CapturedFile) -> dict[str, object]:
    try:
        payload = json.loads(captured.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("publication intent is unreadable") from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise RuntimeError("publication intent has an unsupported schema")
    return payload


def _discover_intents(artifact_dir: Path, *, expected_prefix: str) -> tuple[Path, ...]:
    logical_intents: set[Path] = set()
    captured_prefix = f".{expected_prefix}"
    final_prefix = f"..{expected_prefix}"
    for entry in sorted(artifact_dir.iterdir()):
        if entry.name.startswith(expected_prefix) and entry.name.endswith(".json"):
            logical_intents.add(entry)
        elif entry.name.startswith(captured_prefix) and entry.name.endswith(
            f".json{CAPTURE_SUFFIX}"
        ):
            logical_intents.add(artifact_dir / entry.name[1 : -len(CAPTURE_SUFFIX)])
        elif entry.name.startswith(final_prefix) and entry.name.endswith(
            f".json{CAPTURE_SUFFIX}{FINAL_DELETE_SUFFIX}"
        ):
            logical_intents.add(
                artifact_dir
                / entry.name[2 : -len(f"{CAPTURE_SUFFIX}{FINAL_DELETE_SUFFIX}")]
            )
        else:
            logical = _logical_prepared_temp_from_any(entry)
            if logical is not None and logical.name.startswith(expected_prefix):
                logical_intents.add(logical)
    return tuple(sorted(logical_intents))


def recover_publication_intents(
    artifact_dir: Path,
    target_root: Path,
    manifest_sha256: str,
    *,
    marker_name: str = ".materialization_state.json",
    candidate_prefix: str = ".conversion_report.json.candidate-",
) -> tuple[Path, ...]:
    """Recover a crash between private capture and verified target commit."""

    expected_prefix = f"{PUBLICATION_INTENT_PREFIX}.{target_root.name}.incomplete-"
    intent_paths = _discover_intents(artifact_dir, expected_prefix=expected_prefix)
    capture_prefix = publication_capture_prefix(target_root)
    unclaimed_captures = {
        entry
        for entry in target_root.parent.iterdir()
        if entry.name.startswith(capture_prefix)
    }
    recovered: list[Path] = []
    for intent_path in intent_paths:
        if not publication_intent_present(intent_path):
            for residue in tuple(artifact_dir.iterdir()):
                if _logical_prepared_temp_from_any(residue) == intent_path:
                    preserve_path_noreplace(residue)
            continue
        captured_intent = capture_regular_file(intent_path)
        payload = _parse_intent(captured_intent)
        staging_name = intent_path.name[len(PUBLICATION_INTENT_PREFIX) : -len(".json")]
        source_root = target_root.parent / staging_name
        source_device = payload.get("source_device")
        source_inode = payload.get("source_inode")
        marker_sha256 = payload.get("marker_file_sha256")
        candidate_sha256 = payload.get("candidate_file_sha256")
        if (
            type(source_device) is not int
            or type(source_inode) is not int
            or not isinstance(marker_sha256, str)
            or len(marker_sha256) != 64
            or not isinstance(candidate_sha256, str)
            or len(candidate_sha256) != 64
        ):
            raise RuntimeError("publication intent has invalid ownership hashes")
        source_identity = source_device, source_inode
        capture_root = publication_capture_path(
            source_root,
            target_root,
            source_identity=source_identity,
        )
        candidate_path = artifact_dir / f"{candidate_prefix}{source_root.name}"
        if (
            payload.get("status") != "publication_planned"
            or payload.get("source_root") != str(source_root)
            or payload.get("target_root") != str(target_root)
            or payload.get("capture_root") != str(capture_root)
            or payload.get("candidate_path") != str(candidate_path)
            or payload.get("manifest_sha256") != manifest_sha256
            or payload.get("marker_name") != marker_name
            or publication_intent_path(artifact_dir, source_root) != intent_path
            or source_root.parent != target_root.parent
            or not source_root.name.startswith(f".{target_root.name}.incomplete-")
        ):
            raise RuntimeError("publication intent identity does not match this run")
        unclaimed_captures.discard(capture_root)
        if file_sha256_or_none(candidate_path) != candidate_sha256:
            raise RuntimeError("publication intent candidate identity changed")

        def root_is_owned(root: Path) -> bool:
            return bool(
                root_belongs_to_transaction(
                    root,
                    target_root=target_root,
                    staging_root=source_root,
                    manifest_sha256=manifest_sha256,
                    marker_name=marker_name,
                    expected_root_identity=source_identity,
                    expected_marker_sha256=marker_sha256,
                )
            )

        source_present = source_root.exists() or source_root.is_symlink()
        capture_present = capture_root.exists() or capture_root.is_symlink()
        target_present = target_root.exists() or target_root.is_symlink()
        if target_present:
            if source_present or capture_present or not root_is_owned(target_root):
                raise RuntimeError(
                    "publication target conflicts with durable source evidence"
                )
            remove_captured_file(captured_intent)
            recovered.append(target_root)
            continue
        if source_present and capture_present:
            raise RuntimeError(
                "publication source and capture both exist; preserving both"
            )
        if capture_present:
            if not root_is_owned(capture_root):
                raise RuntimeError("publication capture identity changed")
            rename_directory_noreplace(capture_root, source_root)
            if not root_is_owned(source_root):
                raise RuntimeError("restored publication source identity changed")
        elif not source_present or not root_is_owned(source_root):
            raise RuntimeError("publication source evidence is missing or foreign")
        rename_directory_noreplace_bound(
            source_root,
            target_root,
            expected_source_identity=source_identity,
        )
        if not root_is_owned(target_root):
            raise RuntimeError("recovered publication target identity changed")
        remove_captured_file(captured_intent)
        recovered.append(target_root)
    if unclaimed_captures:
        raise RuntimeError("private publication capture has no durable intent")
    return tuple(recovered)
