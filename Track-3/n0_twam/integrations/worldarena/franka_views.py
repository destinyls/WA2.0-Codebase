# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Deterministic development/final views over the official Franka episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .franka_manifest import OFFICIAL_RECORDS_SHA256, OFFICIAL_TASKS, canonical_sha256

VIEW_SCHEMA_VERSION = 1
DEVELOPMENT_TRAIN_VIEW = "franka_dev_train540_v1"
DEVELOPMENT_VALIDATION_VIEW = "franka_dev_validation60_v1"
FINAL_REFIT_VIEW = "franka_final_refit600_v1"
TASK_FINETUNE_VIEWS = {
    task: f"franka_task_{task}200_v1" for task in OFFICIAL_TASKS
}


@dataclass(frozen=True)
class FrankaViewEntry:
    lerobot_episode_id: int
    task: str
    source_episode_id: int

    @property
    def source_sha256(self) -> str:
        """Stable content-address key within the pinned official inventory."""

        return canonical_sha256(
            {
                "source_records_sha256": OFFICIAL_RECORDS_SHA256,
                "task": self.task,
                "source_episode_id": self.source_episode_id,
            }
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "lerobot_episode_id": self.lerobot_episode_id,
            "task": self.task,
            "source_episode_id": self.source_episode_id,
        }


@dataclass(frozen=True)
class FrankaDatasetView:
    view_id: str
    role: str
    entries: tuple[FrankaViewEntry, ...]
    source_records_sha256: str = OFFICIAL_RECORDS_SHA256

    @property
    def view_sha256(self) -> str:
        return canonical_sha256(self._canonical_payload())

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": VIEW_SCHEMA_VERSION,
            "view_id": self.view_id,
            "role": self.role,
            "source_records_sha256": self.source_records_sha256,
            "entries": [entry.to_json_dict() for entry in self.entries],
        }

    def to_json_dict(self) -> dict[str, object]:
        return {**self._canonical_payload(), "view_sha256": self.view_sha256}


def build_standard_franka_views() -> dict[str, FrankaDatasetView]:
    entries = tuple(
        FrankaViewEntry(
            lerobot_episode_id=task_index * 200 + source_episode_id,
            task=task,
            source_episode_id=source_episode_id,
        )
        for task_index, task in enumerate(OFFICIAL_TASKS)
        for source_episode_id in range(200)
    )
    train = tuple(entry for entry in entries if entry.source_episode_id < 180)
    validation = tuple(entry for entry in entries if entry.source_episode_id >= 180)
    return {
        DEVELOPMENT_TRAIN_VIEW: FrankaDatasetView(
            view_id=DEVELOPMENT_TRAIN_VIEW,
            role="development_train",
            entries=train,
        ),
        DEVELOPMENT_VALIDATION_VIEW: FrankaDatasetView(
            view_id=DEVELOPMENT_VALIDATION_VIEW,
            role="development_validation",
            entries=validation,
        ),
        FINAL_REFIT_VIEW: FrankaDatasetView(
            view_id=FINAL_REFIT_VIEW,
            role="final_refit",
            entries=entries,
        ),
    }


def build_task_franka_views() -> dict[str, FrankaDatasetView]:
    """Build one immutable all-episode view for each official task."""

    return {
        TASK_FINETUNE_VIEWS[task]: FrankaDatasetView(
            view_id=TASK_FINETUNE_VIEWS[task],
            role="task_finetune",
            entries=tuple(
                FrankaViewEntry(
                    lerobot_episode_id=task_index * 200 + source_episode_id,
                    task=task,
                    source_episode_id=source_episode_id,
                )
                for source_episode_id in range(200)
            ),
        )
        for task_index, task in enumerate(OFFICIAL_TASKS)
    }


def load_franka_view(path: Path) -> FrankaDatasetView:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Franka dataset view: {source}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Franka dataset view must be a schema-v1 object")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Franka dataset view entries must be non-empty")
    entries: list[FrankaViewEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, dict) or set(raw) != {
            "lerobot_episode_id",
            "task",
            "source_episode_id",
        }:
            raise ValueError("Franka dataset view entry schema mismatch")
        entry = FrankaViewEntry(
            lerobot_episode_id=int(raw["lerobot_episode_id"]),
            task=str(raw["task"]),
            source_episode_id=int(raw["source_episode_id"]),
        )
        if entry.task not in OFFICIAL_TASKS or not 0 <= entry.source_episode_id < 200:
            raise ValueError("Franka dataset view entry is outside the official set")
        task_index = OFFICIAL_TASKS.index(entry.task)
        if entry.lerobot_episode_id != task_index * 200 + entry.source_episode_id:
            raise ValueError("Franka view LeRobot/source IDs disagree")
        entries.append(entry)
    if len({entry.lerobot_episode_id for entry in entries}) != len(entries):
        raise ValueError("Franka dataset view contains duplicate episodes")
    view = FrankaDatasetView(
        view_id=str(payload.get("view_id")),
        role=str(payload.get("role")),
        source_records_sha256=str(payload.get("source_records_sha256")),
        entries=tuple(entries),
    )
    if view.source_records_sha256 != OFFICIAL_RECORDS_SHA256:
        raise ValueError("Franka dataset view has a different source inventory")
    if payload.get("view_sha256") != view.view_sha256:
        raise ValueError("Franka dataset view hash mismatch")
    return view


__all__ = (
    "DEVELOPMENT_TRAIN_VIEW",
    "DEVELOPMENT_VALIDATION_VIEW",
    "FINAL_REFIT_VIEW",
    "TASK_FINETUNE_VIEWS",
    "FrankaDatasetView",
    "FrankaViewEntry",
    "build_standard_franka_views",
    "build_task_franka_views",
    "load_franka_view",
)
