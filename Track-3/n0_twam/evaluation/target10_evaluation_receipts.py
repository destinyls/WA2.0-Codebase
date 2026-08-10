# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Locking and receipt sealing helpers for Target-10 evaluations."""

from __future__ import annotations

import json
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import fcntl

from n0_twam.evaluation.sealed_artifact_io import read_json_object, sha256_file
from n0_twam.evaluation.tactile_prediction_artifact import (
    VerifiedTactilePredictionArtifact,
)
from n0_twam.evaluation.tactile_provenance import write_json_atomic


def _now_utc() -> str:
    """Return a compact UTC timestamp for an operator-visible receipt."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class _EvaluationLock:
    """An advisory lock that remains safe when its owner process exits."""

    path: Path
    descriptor: int
    owner_token: str
    schema_version: int

    def release(self) -> None:
        """Publish release state before freeing the held file descriptor lock."""

        try:
            payload = {
                "schema_version": self.schema_version,
                "status": "released",
                "owner_token": self.owner_token,
                "released_at_utc": _now_utc(),
            }
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
            os.lseek(self.descriptor, 0, os.SEEK_SET)
            os.ftruncate(self.descriptor, 0)
            os.write(self.descriptor, encoded)
            os.fsync(self.descriptor)
        finally:
            try:
                fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self.descriptor)


def _acquire_lock(root: Path, *, schema_version: int) -> _EvaluationLock:
    """Acquire one persistent OS-level lock for a single output root."""

    lock_path = root / ".evaluation.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError(f"evaluation is already active: {lock_path}") from exc
    owner_token = uuid.uuid4().hex
    payload = {
        "schema_version": schema_version,
        "status": "active",
        "owner_token": owner_token,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "acquired_at_utc": _now_utc(),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
    os.ftruncate(descriptor, 0)
    os.write(descriptor, encoded)
    os.fsync(descriptor)
    return _EvaluationLock(lock_path, descriptor, owner_token, schema_version)


def _metric_directory_inventory(root: Path) -> list[dict[str, object]]:
    """Hash the complete Stage-B directory before it can be reused."""

    resolved_root = Path(root).resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(resolved_root)
    records = []
    for path in sorted(resolved_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"metric output must not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"metric output has an unsupported member: {path}")
        records.append(
            {
                "relative_path": path.relative_to(resolved_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ValueError("metric output directory is empty")
    return records


def _write_metric_receipt(
    path: Path,
    *,
    schema_version: int,
    metric_root: Path,
    report_path: Path,
    artifact: VerifiedTactilePredictionArtifact,
    request_sha256: str,
    input_identity: Mapping[str, object],
    evaluation_view_id: str,
    evaluation_view_sha256: str,
    official_metric_script_sha256: str,
) -> None:
    """Seal the metric directory inventory for future safe reuse."""

    write_json_atomic(
        path,
        {
            "schema_version": schema_version,
            "request_sha256": request_sha256,
            "input_identity": dict(input_identity),
            "prediction_artifact": {
                "file_sha256": artifact.file_sha256,
                "seal_sha256": artifact.seal_sha256,
            },
            "evaluation_view": {
                "view_id": evaluation_view_id,
                "view_sha256": evaluation_view_sha256,
            },
            "official_metric_script_sha256": official_metric_script_sha256,
            "metric_report": {
                "relative_path": report_path.name,
                "sha256": sha256_file(report_path),
            },
            "metric_directory_inventory": _metric_directory_inventory(metric_root),
        },
    )


def _metric_receipt_matches(
    receipt: Mapping[str, object],
    *,
    schema_version: int,
    metric_root: Path,
    report_path: Path,
    artifact: VerifiedTactilePredictionArtifact,
    request_sha256: str,
    input_identity: Mapping[str, object],
    evaluation_view_id: str,
    evaluation_view_sha256: str,
    official_metric_script_sha256: str,
) -> bool:
    """Reject a reused report if any Stage-B output or identity has drifted."""

    prediction = receipt.get("prediction_artifact")
    evaluation_view = receipt.get("evaluation_view")
    report_record = receipt.get("metric_report")
    inventory = receipt.get("metric_directory_inventory")
    return (
        receipt.get("schema_version") == schema_version
        and receipt.get("request_sha256") == request_sha256
        and receipt.get("input_identity") == dict(input_identity)
        and isinstance(prediction, Mapping)
        and prediction.get("file_sha256") == artifact.file_sha256
        and prediction.get("seal_sha256") == artifact.seal_sha256
        and isinstance(evaluation_view, Mapping)
        and evaluation_view.get("view_id") == evaluation_view_id
        and evaluation_view.get("view_sha256") == evaluation_view_sha256
        and receipt.get("official_metric_script_sha256")
        == official_metric_script_sha256
        and isinstance(report_record, Mapping)
        and report_record.get("relative_path") == report_path.name
        and report_record.get("sha256") == sha256_file(report_path)
        and isinstance(inventory, list)
        and inventory == _metric_directory_inventory(metric_root)
    )
