# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Crash-recoverable cleanup for owned materialization staging trees."""

import json
import stat
from pathlib import Path

from script.track3_1.materialize_transaction import (
    _read_regular_file_bytes,
    file_sha256_or_none,
    rename_directory_noreplace,
    report_logical_sha256,
)
from script.track3_1.materialize_tree import (
    captured_tree_path,
    remove_directory_tree_nofollow,
)
from script.track3_1.materialize_writeahead import (
    CAPTURE_SUFFIX,
    FINAL_DELETE_SUFFIX,
    CapturedFile,
    capture_path_for,
    capture_regular_file,
    final_delete_path_for,
    prepare_json_file,
    remove_captured_file,
    remove_exact_regular_file,
)

CLEANUP_JOURNAL_PREFIX = ".materialization_cleanup-"
STAGING_CLEANUP_JOURNAL_NAME = ".materialization_cleanup.prepared.json"


class UnownedReportCandidateError(RuntimeError):
    """Raised when an external candidate name now contains foreign bytes."""


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _validate_owned_candidate(
    path: Path,
    *,
    expected_file_sha256: object,
    expected_report_sha256: object,
) -> Path | None:
    locations = tuple(
        item
        for item in (final_delete_path_for(path), capture_path_for(path), path)
        if _path_present(item)
    )
    if not locations:
        return None
    if not isinstance(expected_file_sha256, str) or len(expected_file_sha256) != 64:
        raise RuntimeError(
            "owned staging has an unverified conversion report candidate"
        )
    for location in locations:
        if (
            not location.is_symlink()
            and location.is_file()
            and file_sha256_or_none(location) == expected_file_sha256
            and (
                expected_report_sha256 is None
                or report_logical_sha256(location) == expected_report_sha256
            )
        ):
            return location
    raise RuntimeError("owned staging has an unverified conversion report candidate")


def _capture_owned_candidate(
    path: Path,
    *,
    expected_file_sha256: object,
    expected_report_sha256: object,
) -> CapturedFile | None:
    location = _validate_owned_candidate(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_report_sha256=expected_report_sha256,
    )
    if location is None:
        return None
    captured = capture_regular_file(path)
    try:
        valid = (
            isinstance(expected_file_sha256, str)
            and captured.sha256 == expected_file_sha256
            and (
                expected_report_sha256 is None
                or report_logical_sha256(captured.path) == expected_report_sha256
            )
        )
        if not valid:
            raise RuntimeError(
                "owned staging has an unverified conversion report candidate"
            )
    except BaseException:
        if not _path_present(path):
            try:
                rename_directory_noreplace(captured.path, path)
            except BaseException:
                pass
        raise
    return captured


def remove_owned_report_displacement(
    path: Path,
    *,
    expected_sha256: str | None,
) -> None:
    """Remove only the exact baseline report displaced by a committed CAS."""

    try:
        captured = _capture_owned_candidate(
            path,
            expected_file_sha256=expected_sha256,
            expected_report_sha256=None,
        )
    except RuntimeError as exc:
        raise UnownedReportCandidateError(
            "report candidate path contains unowned bytes"
        ) from exc
    if captured is None:
        return
    assert expected_sha256 is not None
    remove_captured_file(captured)


def _journal_path(artifact_dir: Path, staging_root: Path) -> Path:
    return artifact_dir / f"{CLEANUP_JOURNAL_PREFIX}{staging_root.name}.json"


def _journal_capture_path(journal_path: Path) -> Path:
    return journal_path.parent / f".{journal_path.name}{CAPTURE_SUFFIX}"


def _journal_final_delete_path(journal_path: Path) -> Path:
    captured_path = _journal_capture_path(journal_path)
    return journal_path.parent / f".{captured_path.name}{FINAL_DELETE_SUFFIX}"


def _publish_cleanup_journal(
    *,
    journal_path: Path,
    staging_root: Path,
    payload: dict[str, object],
) -> None:
    """Publish a staged cleanup journal without replacing any existing entry."""

    prepared_path = staging_root / STAGING_CLEANUP_JOURNAL_NAME
    expected_sha256 = prepare_json_file(
        prepared_path,
        payload,
        preserve_dir=journal_path.parent,
        preserve_foreign=True,
    )
    try:
        rename_directory_noreplace(prepared_path, journal_path)
    except FileExistsError:
        if file_sha256_or_none(journal_path) != expected_sha256:
            raise FileExistsError(
                f"cleanup journal path already exists: {journal_path}"
            ) from None
        remove_exact_regular_file(
            prepared_path,
            expected_sha256=expected_sha256,
        )
    if file_sha256_or_none(journal_path) != expected_sha256:
        raise RuntimeError("cleanup journal no-replace publish was inconsistent")


def _load_journal(path: Path) -> dict[str, object]:
    encoded = _read_regular_file_bytes(path)
    if encoded is None:
        raise RuntimeError(f"cleanup journal is not a regular file: {path}")
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cleanup journal is unreadable: {path}") from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise RuntimeError(f"cleanup journal has an unsupported schema: {path}")
    return payload


def _recover_cleanup_journal(
    journal_path: Path,
    *,
    artifact_dir: Path,
    target_root: Path,
    manifest_sha256: str,
    marker_name: str,
    candidate_prefix: str,
) -> Path:
    try:
        captured_journal = capture_regular_file(journal_path)
    except RuntimeError as exc:
        raise RuntimeError(
            f"cleanup journal is not a regular file: {journal_path}"
        ) from exc
    payload = _load_journal(captured_journal.path)
    if file_sha256_or_none(captured_journal.path) != captured_journal.sha256:
        raise RuntimeError("captured cleanup journal changed while loading")
    raw_staging = payload.get("staging_root")
    if not isinstance(raw_staging, str):
        raise RuntimeError("cleanup journal has no staging identity")
    staging_root = Path(raw_staging).resolve(strict=False)
    expected_journal = _journal_path(artifact_dir, staging_root)
    marker_payload = payload.get("marker_payload")
    if (
        journal_path != expected_journal
        or staging_root.parent != target_root.parent
        or not staging_root.name.startswith(f".{target_root.name}.incomplete-")
        or payload.get("target_root") != str(target_root)
        or payload.get("manifest_sha256") != manifest_sha256
        or not isinstance(marker_payload, dict)
        or marker_payload.get("status") not in {"incomplete", "ready_to_publish"}
        or (
            marker_payload.get("target_root"),
            marker_payload.get("staging_root"),
            marker_payload.get("manifest_sha256"),
            marker_payload.get("candidate_file_sha256"),
            marker_payload.get("conversion_report_sha256"),
        )
        != (
            str(target_root),
            str(staging_root),
            manifest_sha256,
            payload.get("candidate_file_sha256"),
            payload.get("conversion_report_sha256"),
        )
    ):
        raise RuntimeError("cleanup journal identity does not match this transaction")
    expected_device = payload.get("staging_device")
    expected_inode = payload.get("staging_inode")
    if type(expected_device) is not int or type(expected_inode) is not int:
        raise RuntimeError("cleanup journal has no strict staging identity")
    expected_identity = expected_device, expected_inode
    captured_root = captured_tree_path(
        staging_root,
        device=expected_device,
        inode=expected_inode,
    )
    owned_root: Path | None = None
    for location in (captured_root, staging_root):
        try:
            metadata = location.lstat()
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (
                metadata.st_dev,
                metadata.st_ino,
            )
            != expected_identity
        ):
            if location == staging_root and captured_root.exists():
                continue
            raise RuntimeError("cleanup staging identity changed before recovery")
        owned_root = location
        break
    candidate_path = artifact_dir / f"{candidate_prefix}{staging_root.name}"
    captured_candidate = _capture_owned_candidate(
        candidate_path,
        expected_file_sha256=payload.get("candidate_file_sha256"),
        expected_report_sha256=payload.get("conversion_report_sha256"),
    )
    if captured_candidate is not None:
        remove_captured_file(captured_candidate)
    if owned_root is not None:
        try:
            remove_directory_tree_nofollow(
                staging_root,
                preserve_until_last=marker_name,
                expected_identity=expected_identity,
            )
        except BaseException:
            try:
                recovery_root = (
                    captured_root if captured_root.exists() else staging_root
                )
                current = recovery_root.lstat()
            except OSError:
                current = None
            if (
                current is not None
                and (
                    current.st_dev,
                    current.st_ino,
                )
                == expected_identity
            ):
                prepare_json_file(
                    recovery_root / marker_name,
                    marker_payload,
                    preserve_dir=recovery_root.parent,
                )
            raise
    remove_captured_file(captured_journal)
    return staging_root


def cleanup_owned_staging_transactions(
    *,
    artifact_dir: Path,
    target_root: Path,
    manifest_sha256: str,
    marker_name: str,
    candidate_prefix: str,
) -> tuple[Path, ...]:
    """Remove exact owned staging trees using an external durable journal."""

    removed: list[Path] = []
    journal_prefix = f"{CLEANUP_JOURNAL_PREFIX}.{target_root.name}.incomplete-"
    logical_journals: set[Path] = set()
    captured_prefix = f".{journal_prefix}"
    final_prefix = f"..{journal_prefix}"
    for entry in sorted(artifact_dir.iterdir()):
        if entry.name.startswith(journal_prefix):
            logical_journals.add(entry)
        elif entry.name.startswith(captured_prefix) and entry.name.endswith(
            CAPTURE_SUFFIX
        ):
            logical_name = entry.name[1 : -len(CAPTURE_SUFFIX)]
            logical_journals.add(artifact_dir / logical_name)
        elif entry.name.startswith(final_prefix) and entry.name.endswith(
            f"{CAPTURE_SUFFIX}{FINAL_DELETE_SUFFIX}"
        ):
            logical_name = entry.name[
                2 : -len(f"{CAPTURE_SUFFIX}{FINAL_DELETE_SUFFIX}")
            ]
            logical_journals.add(artifact_dir / logical_name)
    for journal_path in sorted(logical_journals):
        removed.append(
            _recover_cleanup_journal(
                journal_path,
                artifact_dir=artifact_dir,
                target_root=target_root,
                manifest_sha256=manifest_sha256,
                marker_name=marker_name,
                candidate_prefix=candidate_prefix,
            )
        )

    prefix = f".{target_root.name}.incomplete-"
    for staging_root in sorted(target_root.parent.iterdir()):
        try:
            metadata = staging_root.lstat()
        except OSError:
            continue
        if not staging_root.name.startswith(prefix) or not stat.S_ISDIR(
            metadata.st_mode
        ):
            continue
        candidate_path = artifact_dir / f"{candidate_prefix}{staging_root.name}"
        candidate_locations = (
            candidate_path,
            capture_path_for(candidate_path),
            final_delete_path_for(candidate_path),
        )
        candidate_present = any(_path_present(path) for path in candidate_locations)
        encoded_marker = _read_regular_file_bytes(staging_root / marker_name)
        if encoded_marker is None:
            if candidate_present:
                raise RuntimeError(
                    "unowned staging root has a preserved report candidate"
                )
            continue
        try:
            marker_payload = json.loads(encoded_marker.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if candidate_present:
                raise RuntimeError(
                    "unowned staging root has a preserved report candidate"
                ) from exc
            continue
        if not isinstance(marker_payload, dict):
            if candidate_present:
                raise RuntimeError(
                    "unowned staging root has a preserved report candidate"
                )
            continue
        if (
            marker_payload.get("target_root"),
            marker_payload.get("staging_root"),
            marker_payload.get("manifest_sha256"),
        ) != (str(target_root), str(staging_root), manifest_sha256) or (
            marker_payload.get("status") not in {"incomplete", "ready_to_publish"}
        ):
            if candidate_present:
                raise RuntimeError(
                    "unowned staging root has a preserved report candidate"
                )
            continue
        _validate_owned_candidate(
            candidate_path,
            expected_file_sha256=marker_payload.get("candidate_file_sha256"),
            expected_report_sha256=marker_payload.get("conversion_report_sha256"),
        )
        journal_path = _journal_path(artifact_dir, staging_root)
        journal_payload: dict[str, object] = {
            "schema_version": 1,
            "target_root": str(target_root),
            "staging_root": str(staging_root),
            "manifest_sha256": manifest_sha256,
            "staging_device": metadata.st_dev,
            "staging_inode": metadata.st_ino,
            "candidate_file_sha256": marker_payload.get("candidate_file_sha256"),
            "conversion_report_sha256": marker_payload.get("conversion_report_sha256"),
            "marker_payload": marker_payload,
        }
        if not (
            _path_present(journal_path)
            or _path_present(_journal_capture_path(journal_path))
            or _path_present(_journal_final_delete_path(journal_path))
        ):
            _publish_cleanup_journal(
                journal_path=journal_path,
                staging_root=staging_root,
                payload=journal_payload,
            )
        removed.append(
            _recover_cleanup_journal(
                journal_path,
                artifact_dir=artifact_dir,
                target_root=target_root,
                manifest_sha256=manifest_sha256,
                marker_name=marker_name,
                candidate_prefix=candidate_prefix,
            )
        )
    return tuple(removed)
