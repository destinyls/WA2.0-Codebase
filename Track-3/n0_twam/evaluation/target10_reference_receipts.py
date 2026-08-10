# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Tamper-evident receipts for reusable Target-10 reference metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, cast

from n0_twam.evaluation.sealed_artifact_io import (
    read_json_object,
    sha256_file,
)
from n0_twam.evaluation.tactile_provenance import write_json_atomic
from n0_twam.evaluation.target10_prediction_artifact_v3 import (
    VerifiedTarget10PredictionArtifact,
)
from n0_twam.evaluation.target10_reference_contract import (
    REFERENCE_CONTRACT,
    REFERENCE_METRIC_SHA256,
)

RECEIPT_SCHEMA_VERSION = 1


def metric_directory_inventory(root: Path) -> list[dict[str, object]]:
    """Hash every regular member of a materialized metric directory."""

    resolved = Path(root).resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    records: list[dict[str, object]] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ValueError("reference metric directory must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("reference metric directory has unsupported members")
        records.append(
            {
                "relative_path": path.relative_to(resolved).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ValueError("reference metric directory is empty")
    return records


def write_reference_metric_receipt(
    path: Path,
    *,
    metric_root: Path,
    report_path: Path,
    artifact: VerifiedTarget10PredictionArtifact,
    request_sha256: str,
    input_identity: Mapping[str, object],
    evaluation_view_id: str,
    evaluation_view_sha256: str,
    golden_calibration_path: Path,
) -> None:
    """Bind the complete Stage-B output to every immutable request identity."""

    write_json_atomic(
        path,
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "contract_sha256": REFERENCE_CONTRACT.sha256,
            "input_identity": dict(input_identity),
            "prediction_artifact": {
                "seal_sha256": artifact.seal_sha256,
                "file_sha256": artifact.file_sha256,
            },
            "evaluation_view": {
                "view_id": evaluation_view_id,
                "view_sha256": evaluation_view_sha256,
            },
            "reference_metric_sha256": REFERENCE_METRIC_SHA256,
            "golden_calibration_sha256": sha256_file(golden_calibration_path),
            "metric_report": {
                "relative_path": report_path.name,
                "sha256": sha256_file(report_path),
            },
            "metric_directory_inventory": metric_directory_inventory(metric_root),
        },
    )


def verify_reference_metric_receipt(
    path: Path,
    *,
    metric_root: Path,
    report_path: Path,
    artifact: VerifiedTarget10PredictionArtifact,
    request_sha256: str,
    input_identity: Mapping[str, object],
    evaluation_view_id: str,
    evaluation_view_sha256: str,
    golden_calibration_path: Path,
) -> dict[str, object]:
    """Reject reuse unless the receipt and full directory still match exactly."""

    receipt = read_json_object(Path(path).resolve(strict=True))
    prediction = receipt.get("prediction_artifact")
    view = receipt.get("evaluation_view")
    report = receipt.get("metric_report")
    expected = (
        receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION
        and receipt.get("request_sha256") == request_sha256
        and receipt.get("contract_sha256") == REFERENCE_CONTRACT.sha256
        and receipt.get("input_identity") == dict(input_identity)
        and isinstance(prediction, Mapping)
        and prediction.get("seal_sha256") == artifact.seal_sha256
        and prediction.get("file_sha256") == artifact.file_sha256
        and isinstance(view, Mapping)
        and view.get("view_id") == evaluation_view_id
        and view.get("view_sha256") == evaluation_view_sha256
        and receipt.get("reference_metric_sha256") == REFERENCE_METRIC_SHA256
        and receipt.get("golden_calibration_sha256")
        == sha256_file(golden_calibration_path)
        and isinstance(report, Mapping)
        and report.get("relative_path") == report_path.name
        and report.get("sha256") == sha256_file(report_path)
        and receipt.get("metric_directory_inventory")
        == metric_directory_inventory(metric_root)
    )
    if not expected:
        raise ValueError("reference metric receipt or directory identity has drifted")
    return cast(dict[str, object], receipt)


__all__ = (
    "metric_directory_inventory",
    "verify_reference_metric_receipt",
    "write_reference_metric_receipt",
)
