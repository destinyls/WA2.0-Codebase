# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Durable state and terminal-claim records for formal materialization."""

import hashlib
import json
import stat
from pathlib import Path
from typing import Mapping, cast

from script.track3_1.materialize_transaction import (
    _read_regular_file_bytes,
    rename_directory_noreplace,
)
from script.track3_1.materialize_writeahead import (
    CapturedFile,
    capture_path_for,
    capture_regular_file,
    final_delete_path_for,
    prepare_json_file,
    publish_prepared_file_noreplace,
    remove_captured_file,
)

COMPLETE_PREPARED_NAME = ".materialization_complete.prepared.json"
STATE_PREPARED_NAME = ".materialization_state.prepared.json"
DEFAULT_MARKER_NAME = ".materialization_state.json"


def directory_identity(path: Path) -> tuple[int, int]:
    """Return the identity of one real directory without following symlinks."""

    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"materialization target is not a directory: {path}")
    return metadata.st_dev, metadata.st_ino


def state_payload(
    *,
    target_root: Path,
    staging_root: Path,
    manifest_sha256: str,
    phase: str,
    status: str = "incomplete",
    conversion_report_sha256: str | None = None,
    report_baseline_sha256: str | None = None,
    candidate_file_sha256: str | None = None,
    error: BaseException | None = None,
) -> dict[str, object]:
    """Build the canonical transaction-state payload."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "phase": phase,
        "target_root": str(target_root),
        "staging_root": str(staging_root),
        "manifest_sha256": manifest_sha256,
        "conversion_report_sha256": conversion_report_sha256,
        "report_baseline_sha256": report_baseline_sha256,
        "candidate_file_sha256": candidate_file_sha256,
    }
    if error is not None:
        payload.update(
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
    return payload


def write_state_marker(
    root: Path,
    *,
    marker_name: str = DEFAULT_MARKER_NAME,
    target_root: Path,
    staging_root: Path,
    manifest_sha256: str,
    phase: str,
    status: str = "incomplete",
    conversion_report_sha256: str | None = None,
    report_baseline_sha256: str | None = None,
    candidate_file_sha256: str | None = None,
    error: BaseException | None = None,
    expected_root_identity: tuple[int, int] | None = None,
    expected_marker_sha256: str | None = None,
) -> str:
    """Persist a marker by capture-before-load and absent-only publication."""

    if expected_root_identity is not None and (
        directory_identity(root) != expected_root_identity
    ):
        raise RuntimeError("state-marker root identity changed before transition")
    payload = state_payload(
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256=manifest_sha256,
        phase=phase,
        status=status,
        conversion_report_sha256=conversion_report_sha256,
        report_baseline_sha256=report_baseline_sha256,
        candidate_file_sha256=candidate_file_sha256,
        error=error,
    )
    marker_path = root / marker_name
    captured = _capture_owned_marker_for_transition(
        marker_path,
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256=manifest_sha256,
        expected_marker_sha256=expected_marker_sha256,
    )
    if expected_root_identity is not None and (
        directory_identity(root) != expected_root_identity
    ):
        if captured is not None:
            _restore_captured_marker(captured, marker_path)
        raise RuntimeError("state-marker root identity changed during capture")
    expected_sha256 = prepare_json_file(
        root / STATE_PREPARED_NAME,
        payload,
        preserve_dir=root.parent,
        preserve_foreign=True,
    )
    publish_prepared_file_noreplace(
        root / STATE_PREPARED_NAME,
        marker_path,
        expected_sha256=expected_sha256,
    )
    if expected_root_identity is not None and (
        directory_identity(root) != expected_root_identity
    ):
        raise RuntimeError("state-marker root identity changed during publication")
    if captured is not None:
        remove_captured_file(captured)
    return cast(str, expected_sha256)


def _load_state_bytes(encoded: bytes) -> dict[str, object]:
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("materialization marker is unreadable") from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise RuntimeError("materialization marker has an unsupported schema")
    return payload


def _validate_owned_state(
    encoded: bytes,
    *,
    target_root: Path,
    staging_root: Path,
    manifest_sha256: str,
) -> None:
    payload = _load_state_bytes(encoded)
    if (
        payload.get("target_root"),
        payload.get("staging_root"),
        payload.get("manifest_sha256"),
        payload.get("status"),
    ) not in {
        (str(target_root), str(staging_root), manifest_sha256, "incomplete"),
        (str(target_root), str(staging_root), manifest_sha256, "ready_to_publish"),
        (str(target_root), str(staging_root), manifest_sha256, "complete"),
    }:
        raise RuntimeError("terminal claim would replace an unowned marker")


def _restore_captured_marker(captured: CapturedFile, marker_path: Path) -> None:
    """Best-effort restoration used only after a captured marker is rejected."""

    if marker_path.exists() or marker_path.is_symlink():
        return
    try:
        rename_directory_noreplace(captured.path, marker_path)
    except BaseException:
        pass


def _capture_owned_marker_for_transition(
    marker_path: Path,
    *,
    target_root: Path,
    staging_root: Path,
    manifest_sha256: str,
    expected_marker_sha256: str | None = None,
) -> CapturedFile | None:
    """Capture and validate the current logical marker, including crash states."""

    capture_exists = any(
        path.exists() or path.is_symlink()
        for path in (capture_path_for(marker_path), final_delete_path_for(marker_path))
    )
    if capture_exists:
        prior = capture_regular_file(marker_path)
        try:
            _validate_owned_state(
                prior.payload,
                target_root=target_root,
                staging_root=staging_root,
                manifest_sha256=manifest_sha256,
            )
        except BaseException:
            _restore_captured_marker(prior, marker_path)
            raise
        if not (marker_path.exists() or marker_path.is_symlink()):
            if (
                expected_marker_sha256 is not None
                and prior.sha256 != expected_marker_sha256
            ):
                _restore_captured_marker(prior, marker_path)
                raise RuntimeError("state-marker bytes changed before transition")
            return prior
        current = _read_regular_file_bytes(marker_path)
        if current is None:
            raise RuntimeError("state-marker path is not a regular file")
        try:
            _validate_owned_state(
                current,
                target_root=target_root,
                staging_root=staging_root,
                manifest_sha256=manifest_sha256,
            )
        except BaseException:
            raise RuntimeError("state-marker path has a foreign replacement") from None
        if expected_marker_sha256 is not None and (
            hashlib.sha256(current).hexdigest() != expected_marker_sha256
        ):
            raise RuntimeError("state-marker bytes changed before transition")
        remove_captured_file(prior)
    if marker_path.exists() or marker_path.is_symlink():
        captured = capture_regular_file(marker_path)
        try:
            _validate_owned_state(
                captured.payload,
                target_root=target_root,
                staging_root=staging_root,
                manifest_sha256=manifest_sha256,
            )
        except BaseException:
            _restore_captured_marker(captured, marker_path)
            raise
        if (
            expected_marker_sha256 is not None
            and captured.sha256 != expected_marker_sha256
        ):
            _restore_captured_marker(captured, marker_path)
            raise RuntimeError("state-marker bytes changed before transition")
        return captured
    if expected_marker_sha256 is not None:
        raise RuntimeError("expected state-marker disappeared before transition")
    return None


def resolve_terminal_staging_root(
    *,
    target_root: Path,
    manifest_sha256: str,
    marker_name: str = DEFAULT_MARKER_NAME,
    fallback: Path,
) -> Path:
    """Recover the original staging identity from a partial terminal transition."""

    marker_path = target_root / marker_name
    recovery_paths = (
        capture_path_for(marker_path),
        final_delete_path_for(marker_path),
        target_root / COMPLETE_PREPARED_NAME,
    )
    for recovery_path in recovery_paths:
        if not (recovery_path.exists() or recovery_path.is_symlink()):
            continue
        encoded = _read_regular_file_bytes(recovery_path)
        if encoded is None:
            raise RuntimeError("terminal recovery record is not a regular file")
        payload = _load_state_bytes(encoded)
        raw_staging = payload.get("staging_root")
        if (
            payload.get("target_root") != str(target_root)
            or payload.get("manifest_sha256") != manifest_sha256
            or payload.get("status")
            not in {"incomplete", "ready_to_publish", "complete"}
            or not isinstance(raw_staging, str)
        ):
            raise RuntimeError("terminal recovery record is not owned by this run")
        staging_root = Path(raw_staging).resolve(strict=False)
        if (
            staging_root == target_root
            or staging_root.parent != target_root.parent
            or not staging_root.name.startswith(f".{target_root.name}.incomplete-")
        ):
            raise RuntimeError("terminal recovery staging identity is invalid")
        return staging_root
    return fallback


def write_complete_marker(
    *,
    target_root: Path,
    staging_root: Path,
    manifest_sha256: str,
    conversion_report_sha256: str,
    marker_name: str = DEFAULT_MARKER_NAME,
    expected_target_identity: tuple[int, int],
    expected_marker_sha256: str | None = None,
) -> None:
    """Publish a terminal claim with capture plus absent-only no-replace."""

    if directory_identity(target_root) != expected_target_identity:
        raise RuntimeError("verified materialization target identity changed")
    marker_path = target_root / marker_name
    prepared_path = target_root / COMPLETE_PREPARED_NAME
    complete_payload = state_payload(
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256=manifest_sha256,
        phase="complete",
        status="complete",
        conversion_report_sha256=conversion_report_sha256,
    )
    captured = _capture_owned_marker_for_transition(
        marker_path,
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256=manifest_sha256,
        expected_marker_sha256=expected_marker_sha256,
    )

    expected_sha256 = prepare_json_file(
        prepared_path,
        complete_payload,
        preserve_dir=target_root.parent,
        preserve_foreign=True,
    )
    publish_prepared_file_noreplace(
        prepared_path,
        marker_path,
        expected_sha256=expected_sha256,
    )
    if directory_identity(target_root) != expected_target_identity:
        raise RuntimeError("materialization target changed while claiming completion")
    if captured is not None:
        remove_captured_file(captured)


def committed_result(
    *,
    manifest_sha256: str,
    conversion_report_sha256: str,
    target_root: Path,
    split_counts: Mapping[str, int],
    status: str,
) -> dict[str, object]:
    """Build the stable public return payload for a committed transaction."""

    return {
        "status": status,
        "manifest_sha256": manifest_sha256,
        "conversion_report_sha256": conversion_report_sha256,
        "target_root": str(target_root),
        "split_counts": dict(split_counts),
    }
