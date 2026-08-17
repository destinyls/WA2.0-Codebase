# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Self-hashed ordered sample roster for AgileX offline evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping

from n0_twam.evaluation.franka_atomic_io import publish_atomic_file

from n0_twam.integrations.worldarena.agilex_manifest import (
    canonical_sha256,
    sha256_file,
)


@dataclass(frozen=True)
class AgileXEvaluationEntry:
    repo_id: str
    episode_id: int
    task_id: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value or Path(value).name != value
            for value in (self.repo_id, self.task_id)
        ):
            raise ValueError("AgileX evaluation repo/task IDs must be safe names")
        if type(self.episode_id) is not int or self.episode_id < 0:
            raise ValueError("AgileX evaluation episode ID must be non-negative")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "repo_id": self.repo_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
        }


@dataclass(frozen=True)
class AgileXEvaluationView:
    view_id: str
    entries: tuple[AgileXEvaluationEntry, ...]
    view_sha256: str
    source_path: Path
    source_file_sha256: str


def load_agilex_evaluation_view(path: Path) -> AgileXEvaluationView:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError("AgileX evaluation view must be a regular non-symlink file")
    source = source.resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid AgileX evaluation view JSON") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "view_id",
        "entries",
        "view_sha256",
    }:
        raise ValueError("AgileX evaluation view has an invalid schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported AgileX evaluation view schema")
    view_id = payload["view_id"]
    raw_entries = payload["entries"]
    if not isinstance(view_id, str) or not view_id or not isinstance(raw_entries, list):
        raise ValueError("AgileX evaluation view ID/entries are invalid")
    entries: list[AgileXEvaluationEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, Mapping) or set(raw) != {
            "repo_id",
            "episode_id",
            "task_id",
        }:
            raise ValueError("AgileX evaluation entry has an invalid schema")
        entries.append(
            AgileXEvaluationEntry(
                repo_id=raw["repo_id"],  # type: ignore[arg-type]
                episode_id=raw["episode_id"],  # type: ignore[arg-type]
                task_id=raw["task_id"],  # type: ignore[arg-type]
            )
        )
    roster = tuple(entries)
    keys = tuple((item.repo_id, item.episode_id) for item in roster)
    if not roster or len(keys) != len(set(keys)):
        raise ValueError("AgileX evaluation view entries must be non-empty and unique")
    core = {key: value for key, value in payload.items() if key != "view_sha256"}
    digest = canonical_sha256(core)
    if payload["view_sha256"] != digest:
        raise ValueError("AgileX evaluation view self hash mismatch")
    return AgileXEvaluationView(
        view_id=view_id,
        entries=roster,
        view_sha256=digest,
        source_path=source,
        source_file_sha256=sha256_file(source),
    )


def publish_agilex_evaluation_view(
    *,
    output: Path,
    view_id: str,
    entries: tuple[AgileXEvaluationEntry, ...],
) -> dict[str, object]:
    """Publish one ordered evaluation roster as an immutable JSON file."""

    if not entries:
        raise ValueError("AgileX evaluation view requires at least one entry")
    core: dict[str, object] = {
        "schema_version": 1,
        "view_id": view_id,
        "entries": [entry.to_json_dict() for entry in entries],
    }
    payload = {**core, "view_sha256": canonical_sha256(core)}
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )

    def writer(handle: BinaryIO) -> None:
        handle.write(raw)

    published = publish_atomic_file(
        output=output,
        writer=writer,
        validator=lambda value: load_agilex_evaluation_view_bytes(value),
        label="AgileX evaluation view",
    )
    return {
        **payload,
        "output": str(published.path),
        "output_file_sha256": published.sha256,
    }


def load_agilex_evaluation_view_bytes(raw: bytes) -> dict[str, object]:
    """Validate staged view bytes before exclusive publication."""

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid AgileX evaluation view bytes") from error
    if not isinstance(payload, dict):
        raise ValueError("AgileX evaluation view bytes must contain an object")
    core = {key: value for key, value in payload.items() if key != "view_sha256"}
    if set(payload) != {"schema_version", "view_id", "entries", "view_sha256"}:
        raise ValueError("AgileX evaluation view staged schema mismatch")
    if payload["view_sha256"] != canonical_sha256(core):
        raise ValueError("AgileX evaluation view staged self hash mismatch")
    return payload


__all__ = (
    "AgileXEvaluationEntry",
    "AgileXEvaluationView",
    "load_agilex_evaluation_view",
    "publish_agilex_evaluation_view",
)
