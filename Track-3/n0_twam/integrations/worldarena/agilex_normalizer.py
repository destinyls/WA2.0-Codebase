# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Manifest-bound q01/q99 statistics for AgileX qpos14 actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import numpy.typing as npt

from .agilex_actions import QPOS14_ACTION_SCHEMA, validate_qpos14
from .agilex_manifest import canonical_sha256, sha256_file


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _vector(value: object, *, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (14,) or not np.isfinite(array).all():
        raise ValueError(f"AgileX normalizer {label} must contain 14 finite values")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class AgileXQpos14Normalizer:
    q01: tuple[float, ...]
    q99: tuple[float, ...]
    sample_count: int
    source_manifest_sha256: str
    repo_route_manifest_sha256: str
    action_schema: str = QPOS14_ACTION_SCHEMA
    schema_version: int = 1

    def __post_init__(self) -> None:
        canonical_q01 = _vector(self.q01, label="q01")
        canonical_q99 = _vector(self.q99, label="q99")
        object.__setattr__(self, "q01", canonical_q01)
        object.__setattr__(self, "q99", canonical_q99)
        lower = np.asarray(canonical_q01, dtype=np.float32)
        upper = np.asarray(canonical_q99, dtype=np.float32)
        if np.any(upper <= lower):
            raise ValueError("AgileX normalizer requires q99 > q01 for every channel")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError(
                "AgileX normalizer sample_count must be a positive integer"
            )
        _sha256(self.source_manifest_sha256, label="source manifest SHA-256")
        _sha256(
            self.repo_route_manifest_sha256,
            label="repo route manifest SHA-256",
        )
        if self.action_schema != QPOS14_ACTION_SCHEMA or self.schema_version != 1:
            raise ValueError("AgileX normalizer action/schema contract mismatch")

    def _contract_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_schema": self.action_schema,
            "normalization": "q01q99",
            "q01": list(self.q01),
            "q99": list(self.q99),
            "sample_count": self.sample_count,
            "source_manifest_sha256": self.source_manifest_sha256,
            "repo_route_manifest_sha256": self.repo_route_manifest_sha256,
        }

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self._contract_payload())

    def to_json_dict(self) -> dict[str, object]:
        payload = self._contract_payload()
        payload["contract_sha256"] = self.contract_sha256
        return payload

    @staticmethod
    def file_sha256(path: Path) -> str:
        return sha256_file(path)

    def normalize(self, values: npt.ArrayLike) -> npt.NDArray[np.float32]:
        action = validate_qpos14(values, label="AgileX action")
        q01 = np.asarray(self.q01, dtype=np.float32)
        q99 = np.asarray(self.q99, dtype=np.float32)
        return ((action - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).astype(
            np.float32,
            copy=False,
        )

    def denormalize(self, values: npt.ArrayLike) -> npt.NDArray[np.float32]:
        action = validate_qpos14(values, label="AgileX normalized action")
        q01 = np.asarray(self.q01, dtype=np.float32)
        q99 = np.asarray(self.q99, dtype=np.float32)
        return ((action + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01).astype(
            np.float32,
            copy=False,
        )


def _valid_rows(
    action_batches: Iterable[npt.ArrayLike],
    valid_mask_batches: Iterable[npt.ArrayLike],
) -> npt.NDArray[np.float32]:
    batches = list(action_batches)
    masks = list(valid_mask_batches)
    if len(batches) != len(masks):
        raise ValueError("AgileX action and valid-mask batch counts differ")
    rows: list[npt.NDArray[np.float32]] = []
    for index, (values, raw_mask) in enumerate(zip(batches, masks, strict=True)):
        actions = np.asarray(values, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"AgileX normalizer batch {index} must be [T,14]")
        mask = np.asarray(raw_mask)
        if mask.dtype != np.bool_ or mask.shape != (actions.shape[0],):
            raise ValueError(f"AgileX normalizer mask {index} must be boolean [T]")
        selected = actions[mask]
        if selected.size and not np.isfinite(selected).all():
            raise ValueError(
                "AgileX valid action population contains non-finite values"
            )
        if selected.size:
            rows.append(selected)
    if not rows:
        raise ValueError("AgileX normalizer has no valid action rows")
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def fit_agilex_normalizer(
    *,
    action_batches: Iterable[npt.ArrayLike],
    valid_mask_batches: Iterable[npt.ArrayLike],
    source_manifest_sha256: str,
    repo_route_manifest_sha256: str,
) -> AgileXQpos14Normalizer:
    """Fit only on valid official/engineering rows selected by the caller."""

    population = _valid_rows(action_batches, valid_mask_batches)
    quantiles = np.quantile(population, (0.01, 0.99), axis=0, method="linear")
    return AgileXQpos14Normalizer(
        q01=tuple(float(value) for value in quantiles[0]),
        q99=tuple(float(value) for value in quantiles[1]),
        sample_count=int(population.shape[0]),
        source_manifest_sha256=_sha256(
            source_manifest_sha256,
            label="source manifest SHA-256",
        ),
        repo_route_manifest_sha256=_sha256(
            repo_route_manifest_sha256,
            label="repo route manifest SHA-256",
        ),
    )


def load_agilex_normalizer(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_source_manifest_sha256: str,
    expected_repo_route_manifest_sha256: str,
) -> AgileXQpos14Normalizer:
    source = Path(path).expanduser().resolve(strict=True)
    if source.is_symlink() or sha256_file(source) != _sha256(
        expected_file_sha256,
        label="normalizer file SHA-256",
    ):
        raise ValueError("AgileX normalizer file identity mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid AgileX normalizer: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("AgileX normalizer must contain an object")
    expected_fields = {
        "schema_version",
        "action_schema",
        "normalization",
        "q01",
        "q99",
        "sample_count",
        "source_manifest_sha256",
        "repo_route_manifest_sha256",
        "contract_sha256",
    }
    if set(payload) != expected_fields or payload.get("normalization") != "q01q99":
        raise ValueError("AgileX normalizer has a noncanonical field contract")
    raw_sample_count = payload.get("sample_count")
    raw_schema_version = payload.get("schema_version")
    if type(raw_sample_count) is not int:
        raise ValueError("AgileX normalizer sample_count must be an integer")
    if type(raw_schema_version) is not int:
        raise ValueError("AgileX normalizer schema_version must be an integer")
    normalizer = AgileXQpos14Normalizer(
        q01=_vector(payload.get("q01"), label="q01"),
        q99=_vector(payload.get("q99"), label="q99"),
        sample_count=raw_sample_count,
        source_manifest_sha256=str(payload.get("source_manifest_sha256", "")),
        repo_route_manifest_sha256=str(payload.get("repo_route_manifest_sha256", "")),
        action_schema=str(payload.get("action_schema", "")),
        schema_version=raw_schema_version,
    )
    if payload.get("contract_sha256") != normalizer.contract_sha256:
        raise ValueError("AgileX normalizer canonical contract hash mismatch")
    if normalizer.source_manifest_sha256 != expected_source_manifest_sha256:
        raise ValueError("AgileX normalizer source manifest mismatch")
    if normalizer.repo_route_manifest_sha256 != expected_repo_route_manifest_sha256:
        raise ValueError("AgileX normalizer repo route manifest mismatch")
    return normalizer


__all__ = (
    "AgileXQpos14Normalizer",
    "fit_agilex_normalizer",
    "load_agilex_normalizer",
)
