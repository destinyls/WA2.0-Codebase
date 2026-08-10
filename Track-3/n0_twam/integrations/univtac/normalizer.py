# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Train-split-only qpos8 quantile statistics."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .dataset_view import DatasetView
from .hdf5_reader import read_qpos8, verify_source_record
from .manifest import MANIFEST_SCHEMA_VERSION, UniVTACDatasetManifest
from .schema import UniVTACEpisodeRecord

LEGACY_NORMALIZER_SCHEMA_VERSION = 1
NORMALIZER_SCHEMA_VERSION = 2


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Qpos8Normalizer:
    """Quantiles for current qpos state and strict next-step action targets."""

    action_q01: tuple[float, ...]
    action_q99: tuple[float, ...]
    state_q01: tuple[float, ...]
    state_q99: tuple[float, ...]
    observed_action_min: tuple[float, ...]
    observed_action_max: tuple[float, ...]
    sample_count: int
    train_paths_sha256: str
    source_manifest_sha256: str
    source_view_id: str | None = None
    source_view_sha256: str | None = None

    def __post_init__(self) -> None:
        if (self.source_view_id is None) != (self.source_view_sha256 is None):
            raise ValueError("normalizer source view identity must be complete")

    def _canonical_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": (
                LEGACY_NORMALIZER_SCHEMA_VERSION
                if self.source_view_id is None
                else NORMALIZER_SCHEMA_VERSION
            ),
            "action_schema": "qpos8_next_step",
            "action_q01": list(self.action_q01),
            "action_q99": list(self.action_q99),
            "state_q01": list(self.state_q01),
            "state_q99": list(self.state_q99),
            "observed_action_min": list(self.observed_action_min),
            "observed_action_max": list(self.observed_action_max),
            "sample_count": self.sample_count,
            "train_paths_sha256": self.train_paths_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
        }
        if self.source_view_id is not None:
            assert self.source_view_sha256 is not None
            payload.update(
                {
                    "source_view_id": self.source_view_id,
                    "source_view_sha256": self.source_view_sha256,
                }
            )
        return payload

    @property
    def normalizer_sha256(self) -> str:
        return _canonical_sha256(self._canonical_payload())

    def to_json_dict(self) -> dict[str, object]:
        payload = self._canonical_payload()
        payload["normalizer_sha256"] = self.normalizer_sha256
        return payload

    def to_action_norm_stat(self) -> dict[str, list[float]]:
        return {"q01": list(self.action_q01), "q99": list(self.action_q99)}


def _tuple(values: npt.NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _records_for_training_view(
    manifest: UniVTACDatasetManifest,
    view: DatasetView,
) -> tuple[UniVTACEpisodeRecord, ...]:
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION or manifest.is_read_only:
        raise ValueError("view-bound normalizers require a schema-v4 manifest")
    if view.role != "training":
        raise ValueError("normalizer source must be a training view")
    if view.physical_split != "train759":
        raise ValueError("normalizer training view must use physical train759")
    if view.source_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("normalizer view does not belong to the dataset manifest")

    train_records = tuple(entry for entry in manifest.entries if entry.split == "train")
    record_by_path = {
        entry.relative_path: (episode_id, entry)
        for episode_id, entry in enumerate(train_records)
    }
    records: list[UniVTACEpisodeRecord] = []
    for view_entry in view.entries:
        indexed_record = record_by_path.get(view_entry.relative_path)
        if indexed_record is None:
            raise ValueError(
                "normalizer view contains an episode outside physical train"
            )
        expected_lerobot_id, record = indexed_record
        identity = (
            record.realpath,
            record.sha256,
            record.task,
            record.split,
            record.episode_id,
            expected_lerobot_id,
        )
        view_identity = (
            view_entry.realpath,
            view_entry.source_sha256,
            view_entry.task,
            view_entry.source_split,
            view_entry.source_episode_id,
            view_entry.lerobot_episode_id,
        )
        if identity != view_identity:
            raise ValueError("normalizer view episode identity does not match manifest")
        records.append(record)
    return tuple(records)


def compute_qpos8_normalizer(
    manifest: UniVTACDatasetManifest,
    *,
    view: DatasetView | None = None,
) -> Qpos8Normalizer:
    """Compute exact q01/q99 statistics from a training view or legacy train split."""

    train_entries = (
        tuple(entry for entry in manifest.entries if entry.split == "train")
        if view is None
        else _records_for_training_view(manifest, view)
    )
    if not train_entries:
        raise ValueError("manifest contains no training episodes")
    states: list[npt.NDArray[np.float64]] = []
    actions: list[npt.NDArray[np.float64]] = []
    for entry in train_entries:
        verify_source_record(entry)
        trajectory = read_qpos8(Path(entry.absolute_path))
        assert entry.usable_end is not None
        usable_trajectory = trajectory[entry.usable_start : entry.usable_end]
        states.append(usable_trajectory[:-1].astype(np.float64))
        actions.append(usable_trajectory[1:].astype(np.float64))
    state_array = np.concatenate(states, axis=0)
    action_array = np.concatenate(actions, axis=0)
    if state_array.shape != action_array.shape or state_array.shape[1] != 8:
        raise RuntimeError("internal qpos8 state/action alignment mismatch")

    train_paths_payload = "".join(
        f"{entry.relative_path}\t{entry.sha256}\n" for entry in train_entries
    ).encode("utf-8")
    return Qpos8Normalizer(
        action_q01=_tuple(np.quantile(action_array, 0.01, axis=0)),
        action_q99=_tuple(np.quantile(action_array, 0.99, axis=0)),
        state_q01=_tuple(np.quantile(state_array, 0.01, axis=0)),
        state_q99=_tuple(np.quantile(state_array, 0.99, axis=0)),
        observed_action_min=_tuple(action_array.min(axis=0)),
        observed_action_max=_tuple(action_array.max(axis=0)),
        sample_count=int(action_array.shape[0]),
        train_paths_sha256=hashlib.sha256(train_paths_payload).hexdigest(),
        source_manifest_sha256=manifest.manifest_sha256,
        source_view_id=None if view is None else view.view_id,
        source_view_sha256=None if view is None else view.view_sha256,
    )
