# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Golden-score calibration gate for the Target-10 reference9 protocol."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Mapping

from n0_twam.evaluation.sealed_artifact_io import (
    read_json_object,
    sha256_file,
    validate_sha256,
)
from n0_twam.evaluation.target10_reference_contract import (
    REFERENCE_CONTRACT,
    TARGET_EPISODE_IDS,
    TARGET_TASKS,
    validate_reference_metric,
)
from n0_twam.evaluation.target10_reference_metric import (
    ReferenceVideoPair,
    evaluate_reference_video_pairs,
)

GOLDEN_MANIFEST_SCHEMA_VERSION = 2
GOLDEN_MANIFEST_TYPE = "wan22_target10_reference9_continuous41_v3_golden"


class PublishedGoldenScoreMismatch(ValueError):
    """Raised only after an intact golden set fails the published-score gate."""

    def __init__(self, evidence: dict[str, object]) -> None:
        published = evidence.get("published_display")
        if not isinstance(published, Mapping):
            raise TypeError("golden mismatch evidence lacks published_display")
        self.evidence = evidence
        super().__init__(
            "golden calibration does not reproduce published 21.26 / 0.746: "
            f"got {published.get('psnr')} / {published.get('ssim')}"
        )


def format_published_score(value: float, *, places: int) -> str:
    """Format a finite score using explicit decimal half-up rounding."""

    quantum = Decimal(1).scaleb(-places)
    return format(
        Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP),
        f".{places}f",
    )


def _safe_golden_file(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".mp4":
        raise ValueError("golden relative path is unsafe")
    candidate = root
    for component in path.parts:
        candidate /= component
        if candidate.is_symlink():
            raise ValueError("golden paths must not contain symlinks")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError("golden MP4 is outside its root")
    return resolved


def calibrate_target10_golden(
    *,
    golden_root: Path,
    golden_manifest_path: Path,
    expected_manifest_sha256: str,
    metric_script: Path,
) -> dict[str, object]:
    """Require a content-addressed golden set to reproduce 21.26 / 0.746."""

    raw_root = Path(golden_root).expanduser()
    if raw_root.is_symlink():
        raise ValueError("golden root must be a real directory")
    root = raw_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("golden root must be a real directory")
    raw_manifest_path = Path(golden_manifest_path).expanduser()
    if raw_manifest_path.is_symlink():
        raise ValueError("golden manifest must not be a symlink")
    manifest_path = raw_manifest_path.resolve(strict=True)
    approved_manifest_sha = validate_sha256(
        expected_manifest_sha256, label="golden manifest SHA256"
    )
    if sha256_file(manifest_path) != approved_manifest_sha:
        raise ValueError("golden manifest SHA256 differs from the request")
    manifest = read_json_object(manifest_path)
    if set(manifest) != {
        "schema_version",
        "artifact_type",
        "contract_id",
        "published_psnr_display",
        "published_ssim_display",
        "files",
    }:
        raise ValueError("golden manifest fields are invalid")
    if (
        manifest.get("schema_version") != GOLDEN_MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_type") != GOLDEN_MANIFEST_TYPE
        or manifest.get("contract_id") != REFERENCE_CONTRACT.contract_id
        or manifest.get("published_psnr_display")
        != REFERENCE_CONTRACT.published_psnr_display
        or manifest.get("published_ssim_display")
        != REFERENCE_CONTRACT.published_ssim_display
    ):
        raise ValueError("golden manifest contract is invalid")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != 20:
        raise ValueError("golden manifest must list exactly 20 tactile MP4 files")
    by_identity: dict[tuple[str, int, str], tuple[Path, dict[str, object]]] = {}
    inventory = []
    expected_members: set[str] = set()
    for record in raw_files:
        if not isinstance(record, Mapping) or set(record) != {
            "task",
            "sample_index",
            "episode_id",
            "kind",
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("golden file record fields are invalid")
        task = record.get("task")
        sample_index = record.get("sample_index")
        episode_id = record.get("episode_id")
        kind = record.get("kind")
        relative_path = record.get("relative_path")
        size_bytes = record.get("size_bytes")
        if (
            task not in TARGET_TASKS
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index not in range(5)
            or isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
            or episode_id != TARGET_EPISODE_IDS[sample_index]
            or kind not in {"pred", "gt"}
            or not isinstance(relative_path, str)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError("golden file record values are invalid")
        expected_relative = f"{task}/sample_{sample_index:03d}_{kind}_tactile.mp4"
        if relative_path != expected_relative:
            raise ValueError("golden file naming differs from the reference layout")
        expected_members.add(relative_path)
        path = _safe_golden_file(root, relative_path)
        expected_sha = validate_sha256(record.get("sha256"), label="golden MP4 SHA256")
        if path.stat().st_size != size_bytes or sha256_file(path) != expected_sha:
            raise ValueError("golden MP4 identity differs from its manifest")
        identity = (str(task), sample_index, str(kind))
        if identity in by_identity:
            raise ValueError("golden manifest contains duplicate file identities")
        normalized = dict(record)
        by_identity[identity] = (path, normalized)
        inventory.append(normalized)
    if manifest_path.is_relative_to(root):
        expected_members.add(manifest_path.relative_to(root).as_posix())
    actual_members: set[str] = set()
    for member in root.rglob("*"):
        if member.is_symlink():
            raise ValueError("golden root must not contain symlinks")
        if member.is_dir():
            continue
        if not member.is_file():
            raise ValueError("golden root contains an unsupported member")
        actual_members.add(member.relative_to(root).as_posix())
    if actual_members != expected_members:
        raise ValueError("golden root inventory is not exactly the approved set")
    pairs = []
    for task in TARGET_TASKS:
        for sample_index, episode_id in enumerate(TARGET_EPISODE_IDS):
            try:
                prediction = by_identity[(task, sample_index, "pred")][0]
                ground_truth = by_identity[(task, sample_index, "gt")][0]
            except KeyError as exc:
                raise ValueError(
                    "golden manifest Target-10 roster is incomplete"
                ) from exc
            pairs.append(
                ReferenceVideoPair(
                    task=task,
                    sample_id=f"sample_{sample_index:03d}",
                    sample_index=sample_index,
                    episode_id=episode_id,
                    prediction_path=prediction,
                    ground_truth_path=ground_truth,
                )
            )
    metric = evaluate_reference_video_pairs(pairs=pairs, metric_script=metric_script)
    overall = metric["overall"]
    if not isinstance(overall, Mapping):
        raise RuntimeError("golden metric overall row is missing")
    psnr_display = format_published_score(float(overall["average_psnr"]), places=2)
    ssim_display = format_published_score(float(overall["average_ssim"]), places=3)
    calibration = {
        "schema_version": 1,
        "contract_id": REFERENCE_CONTRACT.contract_id,
        "contract_sha256": REFERENCE_CONTRACT.sha256,
        "golden_root": str(root),
        "golden_manifest": {
            "path": str(manifest_path),
            "sha256": approved_manifest_sha,
        },
        "metric_script_sha256": sha256_file(validate_reference_metric(metric_script)),
        "file_inventory": inventory,
        "metric": metric,
        "published_display": {
            "psnr": psnr_display,
            "ssim": ssim_display,
        },
        "expected_published_display": {
            "psnr": REFERENCE_CONTRACT.published_psnr_display,
            "ssim": REFERENCE_CONTRACT.published_ssim_display,
        },
        "leaderboard_compatible": False,
        "organizer_contract_confirmed": False,
    }
    if (
        psnr_display != REFERENCE_CONTRACT.published_psnr_display
        or ssim_display != REFERENCE_CONTRACT.published_ssim_display
    ):
        raise PublishedGoldenScoreMismatch(
            {
                **calibration,
                "status": "failed_score_mismatch",
                "published_score_comparable": False,
                "failure_reason": "published_display_mismatch",
            }
        )
    return {
        **calibration,
        "status": "pass",
        "published_score_comparable": True,
    }


__all__ = (
    "GOLDEN_MANIFEST_SCHEMA_VERSION",
    "GOLDEN_MANIFEST_TYPE",
    "PublishedGoldenScoreMismatch",
    "calibrate_target10_golden",
    "format_published_score",
)
