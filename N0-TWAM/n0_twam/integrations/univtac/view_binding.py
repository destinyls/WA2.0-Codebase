# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Bind active DatasetView episodes to stable latent-segment identities."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .dataset_view import DatasetView, DatasetViewEntry


def _sample_id(
    *,
    view_sha256: str,
    entry: DatasetViewEntry,
    start_frame: int,
    end_frame: int,
    crop_window_policy_id: str,
) -> str:
    payload = {
        "view_sha256": view_sha256,
        "source_sha256": entry.source_sha256,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "crop_window_policy_id": crop_window_policy_id,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_view_sample_metadata(
    *,
    view: DatasetView,
    datasets: list[Any],
    crop_window_policy_id: str,
) -> dict[str, tuple[object, ...]]:
    """Bind every valid latent segment to one immutable source episode."""
    if len(datasets) != 1:
        raise ValueError("standard UniVTAC views require one physical repository")
    entries_by_episode = {entry.lerobot_episode_id: entry for entry in view.entries}
    raw_metas = getattr(datasets[0], "new_metas", None)
    if not isinstance(raw_metas, list):
        raise ValueError("qpos8 dataset does not expose valid segment metadata")
    tasks: list[object] = []
    sample_ids: list[object] = []
    source_episode_ids: list[object] = []
    relative_paths: list[object] = []
    seen_episodes: set[int] = set()
    for meta in raw_metas:
        if not isinstance(meta, dict):
            raise ValueError("qpos8 segment metadata must be a mapping")
        episode_id = int(meta.get("episode_index", -1))
        entry = entries_by_episode.get(episode_id)
        if entry is None:
            raise ValueError("qpos8 segment is outside the active dataset view")
        raw_tasks = meta.get("tasks")
        if not isinstance(raw_tasks, list) or raw_tasks != [entry.task]:
            raise ValueError("qpos8 segment task differs from the active view")
        start_frame = int(meta.get("start_frame", -1))
        end_frame = int(meta.get("end_frame", -1))
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError("qpos8 segment has invalid frame bounds")
        tasks.append(entry.task)
        sample_ids.append(
            _sample_id(
                view_sha256=view.view_sha256,
                entry=entry,
                start_frame=start_frame,
                end_frame=end_frame,
                crop_window_policy_id=crop_window_policy_id,
            )
        )
        source_episode_ids.append((view.physical_split, episode_id))
        relative_paths.append(entry.relative_path)
        seen_episodes.add(episode_id)
    if seen_episodes != set(entries_by_episode):
        missing = sorted(set(entries_by_episode) - seen_episodes)
        raise ValueError(f"active dataset view has unusable episodes: {missing}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("view-derived qpos8 sample IDs must be unique")
    return {
        "sample_tasks": tuple(tasks),
        "sample_ids": tuple(sample_ids),
        "sample_source_episode_ids": tuple(source_episode_ids),
        "sample_relative_paths": tuple(relative_paths),
    }


__all__ = ["build_view_sample_metadata"]
