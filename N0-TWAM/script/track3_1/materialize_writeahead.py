# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Durable no-clobber write-ahead records for UniVTAC materialization."""

import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path

from script.track3_1.materialize_transaction import (
    _read_regular_file_bytes,
    file_sha256_or_none,
    fsync_directory,
    rename_directory_noreplace,
)
from script.track3_1.materialize_tree import remove_directory_tree_nofollow

CREATION_INTENT_PREFIX = ".materialization_creation-"
CAPTURE_SUFFIX = ".owned-delete"
FINAL_DELETE_SUFFIX = ".final-delete"
PREPARE_TEMP_SEPARATOR = ".prepare-"
PREPARE_TEMP_SUFFIX = ".tmp"
FOREIGN_PRESERVED_PREFIX = ".materialization-foreign-preserved-"


@dataclass(frozen=True)
class CapturedFile:
    """One file atomically removed from its public name and identity-checked."""

    path: Path
    payload: bytes
    sha256: str


def canonical_json_bytes(payload: object) -> bytes:
    """Encode JSON exactly as formal transaction artifacts are encoded."""

    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("durable write made no progress")
        view = view[written:]


def _create_regular_file_exclusive(path: Path, payload: bytes) -> None:
    """Create one temporary record durably, never replacing an entry."""

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"write-ahead record is not regular: {path}")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def prepared_temp_path_for(path: Path, *, expected_sha256: str) -> Path:
    """Return the deterministic, content-bound temporary path for a record."""

    return path.parent / (
        f".{path.name}{PREPARE_TEMP_SEPARATOR}{expected_sha256}{PREPARE_TEMP_SUFFIX}"
    )


def logical_path_from_prepared_temp(path: Path) -> Path | None:
    """Decode a deterministic prepare-temp name without trusting its bytes."""

    if not path.name.startswith(".") or not path.name.endswith(PREPARE_TEMP_SUFFIX):
        return None
    body = path.name[1 : -len(PREPARE_TEMP_SUFFIX)]
    logical_name, separator, digest = body.rpartition(PREPARE_TEMP_SEPARATOR)
    if (
        not separator
        or not logical_name
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        return None
    return path.parent / logical_name


def preserve_path_noreplace(path: Path, *, destination_dir: Path | None = None) -> Path:
    """Move possibly foreign bytes aside without replacing or deleting them."""

    destination_parent = destination_dir or path.parent
    while True:
        preserved = destination_parent / (
            f"{FOREIGN_PRESERVED_PREFIX}{uuid.uuid4().hex}-{path.name}"
        )
        try:
            rename_directory_noreplace(path, preserved)
        except FileExistsError:
            continue
        return preserved


def _remove_or_preserve_prepare_residue(
    path: Path,
    *,
    expected_payload: bytes,
    expected_sha256: str,
    preserve_dir: Path,
) -> None:
    """Clear a prior capture state, preserving every unverified byte string."""

    if not any(
        candidate.exists() or candidate.is_symlink()
        for candidate in (capture_path_for(path), final_delete_path_for(path))
    ):
        return
    captured = capture_regular_file(path)
    if captured.sha256 == expected_sha256 and captured.payload == expected_payload:
        remove_captured_file(captured)
        return
    preserve_path_noreplace(captured.path, destination_dir=preserve_dir)


def prepare_json_file(
    path: Path,
    payload: object,
    *,
    preserve_dir: Path | None = None,
    preserve_foreign: bool = False,
) -> str:
    """Prepare deterministic JSON through a crash-safe absent-only publish."""

    encoded = canonical_json_bytes(payload)
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    preserved_parent = preserve_dir or path.parent
    existing = _read_regular_file_bytes(path)
    if existing is not None:
        if existing != encoded:
            if not preserve_foreign:
                raise RuntimeError(f"prepared JSON path contains foreign bytes: {path}")
            preserve_path_noreplace(path, destination_dir=preserved_parent)
        else:
            return expected_sha256
    if path.exists() or path.is_symlink():
        if not preserve_foreign:
            raise RuntimeError(f"prepared JSON path is not a regular file: {path}")
        preserve_path_noreplace(path, destination_dir=preserved_parent)
    temporary_path = prepared_temp_path_for(
        path,
        expected_sha256=expected_sha256,
    )
    _remove_or_preserve_prepare_residue(
        temporary_path,
        expected_payload=encoded,
        expected_sha256=expected_sha256,
        preserve_dir=preserved_parent,
    )
    temporary = _read_regular_file_bytes(temporary_path)
    if temporary is not None and temporary != encoded:
        preserve_path_noreplace(
            temporary_path,
            destination_dir=preserved_parent,
        )
        temporary = None
    elif temporary is None and (temporary_path.exists() or temporary_path.is_symlink()):
        preserve_path_noreplace(
            temporary_path,
            destination_dir=preserved_parent,
        )
    if temporary is None:
        try:
            _create_regular_file_exclusive(temporary_path, encoded)
        except FileExistsError:
            if _read_regular_file_bytes(temporary_path) != encoded:
                raise RuntimeError(
                    f"prepared JSON temporary path was concurrently replaced: "
                    f"{temporary_path}"
                ) from None
    try:
        rename_directory_noreplace(temporary_path, path)
    except FileExistsError:
        if _read_regular_file_bytes(path) != encoded:
            raise RuntimeError(
                f"prepared JSON path was concurrently replaced: {path}"
            ) from None
        remove_exact_regular_file(
            temporary_path,
            expected_sha256=expected_sha256,
        )
    if _read_regular_file_bytes(path) != encoded:
        raise RuntimeError(f"prepared JSON bytes changed after write: {path}")
    return expected_sha256


def publish_prepared_file_noreplace(
    prepared_path: Path,
    destination_path: Path,
    *,
    expected_sha256: str,
) -> None:
    """Publish an owned prepared file without clobbering any destination."""

    if file_sha256_or_none(prepared_path) != expected_sha256:
        raise RuntimeError("prepared file identity does not match its durable marker")
    try:
        rename_directory_noreplace(prepared_path, destination_path)
    except FileExistsError:
        if file_sha256_or_none(destination_path) != expected_sha256:
            raise RuntimeError(
                f"foreign destination blocks no-replace publish: {destination_path}"
            ) from None
        remove_exact_regular_file(prepared_path, expected_sha256=expected_sha256)
    if file_sha256_or_none(destination_path) != expected_sha256:
        raise RuntimeError("no-replace publish returned an inconsistent destination")


def capture_path_for(path: Path) -> Path:
    """Return the deterministic first capture path for one public record."""

    return path.parent / f".{path.name}{CAPTURE_SUFFIX}"


def final_delete_path_for(path: Path) -> Path:
    """Return the deterministic final-delete path for one public record."""

    captured_path = capture_path_for(path)
    return path.parent / f".{captured_path.name}{FINAL_DELETE_SUFFIX}"


def capture_regular_file(path: Path) -> CapturedFile:
    """Capture a public record before loading it, preserving later replacements."""

    capture_path = capture_path_for(path)
    final_delete_path = final_delete_path_for(path)
    if final_delete_path.exists() or final_delete_path.is_symlink():
        payload = _read_regular_file_bytes(final_delete_path)
        if payload is None:
            raise RuntimeError(
                f"final-delete record is not regular: {final_delete_path}"
            )
        return CapturedFile(
            path=final_delete_path,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    if capture_path.exists() or capture_path.is_symlink():
        payload = _read_regular_file_bytes(capture_path)
        if payload is None:
            raise RuntimeError(f"captured record is not regular: {capture_path}")
        return CapturedFile(
            path=capture_path,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    payload = _read_regular_file_bytes(path)
    if payload is None:
        raise RuntimeError(f"write-ahead record is not regular: {path}")
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    rename_directory_noreplace(path, capture_path)
    captured = _read_regular_file_bytes(capture_path)
    if captured is None or hashlib.sha256(captured).hexdigest() != expected_sha256:
        if not (path.exists() or path.is_symlink()):
            try:
                rename_directory_noreplace(capture_path, path)
            except BaseException:
                pass
        raise RuntimeError("captured write-ahead bytes changed during quarantine")
    return CapturedFile(
        path=capture_path,
        payload=captured,
        sha256=expected_sha256,
    )


def remove_captured_file(captured: CapturedFile) -> None:
    """Delete only the still-identical captured path, never its public replacement."""

    if file_sha256_or_none(captured.path) != captured.sha256:
        raise RuntimeError("captured write-ahead record changed before removal")
    if captured.path.name.endswith(FINAL_DELETE_SUFFIX):
        final_path = captured.path
    else:
        final_path = captured.path.parent / (
            f".{captured.path.name}{FINAL_DELETE_SUFFIX}"
        )
        rename_directory_noreplace(captured.path, final_path)
        if file_sha256_or_none(final_path) != captured.sha256:
            raise RuntimeError("final-delete capture changed after atomic rename")
    final_path.unlink()
    fsync_directory(final_path.parent)


def remove_exact_regular_file(path: Path, *, expected_sha256: str) -> None:
    """Capture then remove only exact bytes, preserving public replacements."""

    captured = capture_regular_file(path)
    if captured.sha256 != expected_sha256:
        raise RuntimeError(f"captured file does not match owned bytes: {path}")
    remove_captured_file(captured)


def _creation_intent_path(artifact_dir: Path, staging_root: Path) -> Path:
    return artifact_dir / f"{CREATION_INTENT_PREFIX}{staging_root.name}.json"


def reserve_staging_root(
    *,
    artifact_dir: Path,
    target_root: Path,
    manifest_sha256: str,
) -> tuple[Path, Path]:
    """Durably declare a unique staging path before creating its directory."""

    staging_root = target_root.parent / (
        f".{target_root.name}.incomplete-{uuid.uuid4().hex}"
    )
    intent_path = _creation_intent_path(artifact_dir, staging_root)
    prepare_json_file(
        intent_path,
        {
            "schema_version": 1,
            "status": "creation_planned",
            "target_root": str(target_root),
            "staging_root": str(staging_root),
            "manifest_sha256": manifest_sha256,
        },
    )
    os.mkdir(staging_root, 0o700)
    fsync_directory(staging_root.parent)
    return staging_root.resolve(strict=True), intent_path


def release_creation_intent(intent_path: Path) -> None:
    """Remove one intent only after the internal ownership marker is durable."""

    captured = capture_regular_file(intent_path)
    remove_captured_file(captured)


def _parse_intent(captured: CapturedFile) -> dict[str, object]:
    try:
        payload = json.loads(captured.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("creation intent is unreadable") from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise RuntimeError("creation intent has an unsupported schema")
    return payload


def _marker_binds_staging(
    encoded: bytes,
    *,
    target_root: Path,
    staging_root: Path,
    manifest_sha256: str,
) -> bool:
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and (
        payload.get("target_root"),
        payload.get("staging_root"),
        payload.get("manifest_sha256"),
    ) == (str(target_root), str(staging_root), manifest_sha256)


def _logical_prepared_temp_from_any(path: Path) -> Path | None:
    """Decode a public or captured prepare-temp state to its logical record."""

    logical_temp = path
    if path.name.endswith(f"{CAPTURE_SUFFIX}{FINAL_DELETE_SUFFIX}"):
        logical_temp = (
            path.parent / path.name[2 : -len(f"{CAPTURE_SUFFIX}{FINAL_DELETE_SUFFIX}")]
        )
    elif path.name.endswith(CAPTURE_SUFFIX):
        logical_temp = path.parent / path.name[1 : -len(CAPTURE_SUFFIX)]
    return logical_path_from_prepared_temp(logical_temp)


def _state_prepare_residues(
    staging_root: Path, *, marker_name: str
) -> tuple[Path, ...]:
    """Return only protocol-private initial-state prepare residues."""

    if not marker_name.endswith(".json"):
        return ()
    prepared_name = f"{marker_name[:-len('.json')]}.prepared.json"
    prepared_path = staging_root / prepared_name
    allowed = {
        prepared_path,
        capture_path_for(prepared_path),
        final_delete_path_for(prepared_path),
    }
    residues: list[Path] = []
    for entry in staging_root.iterdir():
        if entry in allowed or _logical_prepared_temp_from_any(entry) == prepared_path:
            residues.append(entry)
    return tuple(residues)


def _intent_logical_path_from_prepare_residue(path: Path) -> Path | None:
    logical = _logical_prepared_temp_from_any(path)
    if logical is None or not logical.name.startswith(CREATION_INTENT_PREFIX):
        return None
    return logical


def recover_creation_intents(
    *,
    artifact_dir: Path,
    target_root: Path,
    manifest_sha256: str,
    marker_name: str,
) -> tuple[Path, ...]:
    """Recover bootstrap crashes that occurred before an internal marker existed."""

    recovered: list[Path] = []
    expected_prefix = f"{CREATION_INTENT_PREFIX}.{target_root.name}.incomplete-"
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
            logical = _intent_logical_path_from_prepare_residue(entry)
            if logical is not None and logical.name.startswith(expected_prefix):
                logical_intents.add(logical)
    for intent_path in sorted(logical_intents):
        intent_present = any(
            candidate.exists() or candidate.is_symlink()
            for candidate in (
                intent_path,
                capture_path_for(intent_path),
                final_delete_path_for(intent_path),
            )
        )
        if not intent_present:
            prepare_residues = tuple(
                entry
                for entry in artifact_dir.iterdir()
                if _intent_logical_path_from_prepare_residue(entry) == intent_path
            )
            staging_name = intent_path.name[len(CREATION_INTENT_PREFIX) : -len(".json")]
            staging_root = (target_root.parent / staging_name).resolve(strict=False)
            if staging_root.exists() or staging_root.is_symlink():
                raise RuntimeError(
                    "unpublished creation intent cannot claim an existing staging path"
                )
            for residue in prepare_residues:
                preserve_path_noreplace(residue)
            continue
        captured = capture_regular_file(intent_path)
        payload = _parse_intent(captured)
        raw_staging = payload.get("staging_root")
        if not isinstance(raw_staging, str):
            raise RuntimeError("creation intent has no staging identity")
        staging_root = Path(raw_staging).resolve(strict=False)
        if (
            payload.get("status") != "creation_planned"
            or payload.get("target_root") != str(target_root)
            or payload.get("manifest_sha256") != manifest_sha256
            or staging_root.parent != target_root.parent
            or not staging_root.name.startswith(f".{target_root.name}.incomplete-")
            or _creation_intent_path(artifact_dir, staging_root) != intent_path
        ):
            raise RuntimeError("creation intent identity does not match this run")
        try:
            metadata = staging_root.lstat()
        except FileNotFoundError:
            metadata = None
        if metadata is not None:
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError("creation intent staging path is not a directory")
            marker = _read_regular_file_bytes(staging_root / marker_name)
            if marker is None:
                residues = _state_prepare_residues(
                    staging_root,
                    marker_name=marker_name,
                )
                entries = tuple(staging_root.iterdir())
                if set(entries) != set(residues):
                    raise RuntimeError(
                        "creation intent staging contains unowned bootstrap bytes"
                    )
                for residue in residues:
                    preserve_path_noreplace(
                        residue,
                        destination_dir=artifact_dir,
                    )
                remove_directory_tree_nofollow(staging_root)
                recovered.append(staging_root)
            elif not _marker_binds_staging(
                marker,
                target_root=target_root,
                staging_root=staging_root,
                manifest_sha256=manifest_sha256,
            ):
                raise RuntimeError("creation intent staging marker is foreign")
        else:
            try:
                remove_directory_tree_nofollow(staging_root)
            except FileNotFoundError:
                pass
        remove_captured_file(captured)
    return tuple(recovered)
