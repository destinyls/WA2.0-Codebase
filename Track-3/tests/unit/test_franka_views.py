# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import json

import pytest

from n0_twam.integrations.worldarena.franka_views import (
    DEVELOPMENT_TRAIN_VIEW,
    DEVELOPMENT_VALIDATION_VIEW,
    FINAL_REFIT_VIEW,
    TASK_FINETUNE_VIEWS,
    build_task_franka_views,
    build_standard_franka_views,
    load_franka_view,
)


def test_standard_franka_views_are_complete_balanced_and_disjoint() -> None:
    views = build_standard_franka_views()
    train = views[DEVELOPMENT_TRAIN_VIEW]
    validation = views[DEVELOPMENT_VALIDATION_VIEW]
    final = views[FINAL_REFIT_VIEW]

    assert len(train.entries) == 540
    assert len(validation.entries) == 60
    assert len(final.entries) == 600
    assert {entry.lerobot_episode_id for entry in train.entries}.isdisjoint(
        entry.lerobot_episode_id for entry in validation.entries
    )
    assert {
        task: sum(entry.task == task for entry in train.entries)
        for task in ("clear_up", "pour", "wipe")
    } == {"clear_up": 180, "pour": 180, "wipe": 180}
    assert {
        task: sum(entry.task == task for entry in validation.entries)
        for task in ("clear_up", "pour", "wipe")
    } == {"clear_up": 20, "pour": 20, "wipe": 20}
    assert tuple(train.entries) + tuple(validation.entries) != tuple(final.entries)
    assert len(train.view_sha256) == len(validation.view_sha256) == 64


def test_franka_view_loader_rejects_resealed_roster_tampering(tmp_path) -> None:
    view = build_standard_franka_views()[DEVELOPMENT_VALIDATION_VIEW]
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(view.to_json_dict()), encoding="utf-8")
    assert load_franka_view(path) == view

    payload = view.to_json_dict()
    payload["entries"][0]["source_episode_id"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="IDs disagree|hash mismatch"):
        load_franka_view(path)


def test_task_franka_views_select_all_200_episodes_per_task() -> None:
    views = build_task_franka_views()

    assert set(views) == set(TASK_FINETUNE_VIEWS.values())
    for task, view_id in TASK_FINETUNE_VIEWS.items():
        view = views[view_id]
        assert view.role == "task_finetune"
        assert len(view.entries) == 200
        assert {entry.task for entry in view.entries} == {task}
        assert {entry.source_episode_id for entry in view.entries} == set(range(200))
