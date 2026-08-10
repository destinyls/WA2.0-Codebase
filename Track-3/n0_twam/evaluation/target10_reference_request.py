# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Immutable operator request for calibrated Target-10 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from n0_twam.evaluation.sealed_artifact_io import read_json_object, validate_sha256
from n0_twam.evaluation.target10_reference_contract import REFERENCE_CONTRACT

REQUEST_SCHEMA_VERSION = 6
CALIBRATION_POLICY_REQUIRE_PUBLISHED_GOLDEN = "require_published_golden"
CALIBRATION_POLICY_ALLOW_PROTOCOL_ALIGNED_REPORT = "allow_protocol_aligned_report"
CALIBRATION_POLICIES = frozenset(
    {
        CALIBRATION_POLICY_REQUIRE_PUBLISHED_GOLDEN,
        CALIBRATION_POLICY_ALLOW_PROTOCOL_ALIGNED_REPORT,
    }
)
_REQUIRED_FIELDS = {
    "schema_version",
    "checkpoint",
    "checkpoint_sha256",
    "vae",
    "base_model",
    "empty_embedding",
    "empty_embedding_sha256",
    "lerobot_root",
    "normalizer",
    "normalizer_sha256",
    "normalizer_source_view",
    "evaluation_view_manifest",
    "reference_metadata",
    "raw_root",
    "manifest",
    "conversion_report",
    "official_metric_script",
    "golden_root",
    "golden_manifest",
    "golden_manifest_sha256",
    "calibration_policy",
    "output_root",
}
_OPTIONAL_FIELDS = {
    "config_name",
    "device",
    "hip_visible_devices",
    "n_steps",
    "seed",
}


def _path(payload: Mapping[str, object], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"reference evaluation request {key} must be a path string")
    return Path(value).expanduser()


def _text(
    payload: Mapping[str, object], key: str, *, default: str | None = None
) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"reference evaluation request {key} must be a string")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class Target10ReferenceEvaluationRequest:
    """All reproducibility inputs, without protocol-tuning overrides."""

    checkpoint: Path
    checkpoint_sha256: str
    vae: Path
    base_model: Path
    empty_embedding: Path
    empty_embedding_sha256: str
    lerobot_root: Path
    normalizer: Path
    normalizer_sha256: str
    normalizer_source_view: Path
    evaluation_view_manifest: Path
    reference_metadata: Path
    raw_root: Path
    manifest: Path
    conversion_report: Path
    official_metric_script: Path
    golden_root: Path
    golden_manifest: Path
    golden_manifest_sha256: str
    calibration_policy: str
    output_root: Path
    config_name: str = "track31_univtac"
    device: str = "cuda:0"
    hip_visible_devices: str | None = None
    n_steps: int = REFERENCE_CONTRACT.sampling_steps
    seed: int = REFERENCE_CONTRACT.evaluation_seed

    @classmethod
    def from_json_dict(
        cls, payload: Mapping[str, object]
    ) -> "Target10ReferenceEvaluationRequest":
        missing = _REQUIRED_FIELDS - set(payload)
        unexpected = set(payload) - (_REQUIRED_FIELDS | _OPTIONAL_FIELDS)
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing=" + ",".join(sorted(missing)))
            if unexpected:
                details.append("unexpected=" + ",".join(sorted(unexpected)))
            raise ValueError(
                "invalid reference evaluation request: " + "; ".join(details)
            )
        if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
            raise ValueError("unsupported reference evaluation request schema")
        hip = payload.get("hip_visible_devices")
        if hip is not None and (not isinstance(hip, str) or not hip):
            raise ValueError("hip_visible_devices must be a non-empty string or null")
        n_steps = _positive_int(
            payload.get("n_steps", REFERENCE_CONTRACT.sampling_steps),
            label="n_steps",
        )
        seed = _nonnegative_int(
            payload.get("seed", REFERENCE_CONTRACT.evaluation_seed),
            label="seed",
        )
        if (
            n_steps != REFERENCE_CONTRACT.sampling_steps
            or seed != REFERENCE_CONTRACT.evaluation_seed
        ):
            raise ValueError("reference evaluation fixes n_steps=50 and seed=2026")
        calibration_policy = _text(payload, "calibration_policy")
        if calibration_policy not in CALIBRATION_POLICIES:
            raise ValueError(
                "calibration_policy must be one of "
                + ", ".join(sorted(CALIBRATION_POLICIES))
            )
        return cls(
            checkpoint=_path(payload, "checkpoint"),
            checkpoint_sha256=validate_sha256(
                payload.get("checkpoint_sha256"), label="checkpoint_sha256"
            ),
            vae=_path(payload, "vae"),
            base_model=_path(payload, "base_model"),
            empty_embedding=_path(payload, "empty_embedding"),
            empty_embedding_sha256=validate_sha256(
                payload.get("empty_embedding_sha256"),
                label="empty_embedding_sha256",
            ),
            lerobot_root=_path(payload, "lerobot_root"),
            normalizer=_path(payload, "normalizer"),
            normalizer_sha256=validate_sha256(
                payload.get("normalizer_sha256"), label="normalizer_sha256"
            ),
            normalizer_source_view=_path(payload, "normalizer_source_view"),
            evaluation_view_manifest=_path(payload, "evaluation_view_manifest"),
            reference_metadata=_path(payload, "reference_metadata"),
            raw_root=_path(payload, "raw_root"),
            manifest=_path(payload, "manifest"),
            conversion_report=_path(payload, "conversion_report"),
            official_metric_script=_path(payload, "official_metric_script"),
            golden_root=_path(payload, "golden_root"),
            golden_manifest=_path(payload, "golden_manifest"),
            golden_manifest_sha256=validate_sha256(
                payload.get("golden_manifest_sha256"),
                label="golden_manifest_sha256",
            ),
            calibration_policy=calibration_policy,
            output_root=_path(payload, "output_root"),
            config_name=_text(payload, "config_name", default="track31_univtac"),
            device=_text(payload, "device", default="cuda:0"),
            hip_visible_devices=hip,
            n_steps=n_steps,
            seed=seed,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "vae": str(self.vae),
            "base_model": str(self.base_model),
            "empty_embedding": str(self.empty_embedding),
            "empty_embedding_sha256": self.empty_embedding_sha256,
            "lerobot_root": str(self.lerobot_root),
            "normalizer": str(self.normalizer),
            "normalizer_sha256": self.normalizer_sha256,
            "normalizer_source_view": str(self.normalizer_source_view),
            "evaluation_view_manifest": str(self.evaluation_view_manifest),
            "reference_metadata": str(self.reference_metadata),
            "raw_root": str(self.raw_root),
            "manifest": str(self.manifest),
            "conversion_report": str(self.conversion_report),
            "official_metric_script": str(self.official_metric_script),
            "golden_root": str(self.golden_root),
            "golden_manifest": str(self.golden_manifest),
            "golden_manifest_sha256": self.golden_manifest_sha256,
            "calibration_policy": self.calibration_policy,
            "output_root": str(self.output_root),
            "config_name": self.config_name,
            "device": self.device,
            "hip_visible_devices": self.hip_visible_devices,
            "n_steps": self.n_steps,
            "seed": self.seed,
        }


def load_target10_reference_request(path: Path) -> Target10ReferenceEvaluationRequest:
    return Target10ReferenceEvaluationRequest.from_json_dict(
        read_json_object(Path(path).resolve(strict=True))
    )


__all__ = (
    "CALIBRATION_POLICIES",
    "CALIBRATION_POLICY_ALLOW_PROTOCOL_ALIGNED_REPORT",
    "CALIBRATION_POLICY_REQUIRE_PUBLISHED_GOLDEN",
    "REQUEST_SCHEMA_VERSION",
    "Target10ReferenceEvaluationRequest",
    "load_target10_reference_request",
)
