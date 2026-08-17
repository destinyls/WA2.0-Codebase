# Copyright 2025-2026 NeoteAI Team. All rights reserved.
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import n0_twam.integrations.worldarena.agilex_official_reader as reader
from n0_twam.integrations.worldarena.agilex_official_contracts import (
    build_official_routes,
)
from n0_twam.integrations.worldarena.agilex_official_schema import (
    RGB_FILES,
    TACTILE_FILES,
    VISION_ONLY_REPO_ID,
    VISION_TACTILE_REPO_ID,
    WRENCH_DATASETS,
)


def _episode(root: Path, *, length: int, contact: bool) -> Path:
    episode = root / "episode_0"
    episode.mkdir(parents=True)
    (episode / "meta.json").write_text(
        json.dumps({"device_name": "recap" if contact else "cobot-magic-max"}),
        encoding="utf-8",
    )
    qpos = np.arange(length * 14, dtype=np.float32).reshape(length, 14) / 100
    actions = qpos + 0.25
    with h5py.File(episode / "episode.hdf5", "w") as handle:
        handle.attrs["fps"] = 30
        handle.attrs["sim"] = False
        handle.create_dataset("observations/qpos", data=qpos)
        handle.create_dataset("action", data=actions)
    for name in RGB_FILES.values():
        (episode / name).write_bytes(b"mp4")
    if contact:
        for name in TACTILE_FILES.values():
            (episode / name).write_bytes(b"mp4")
        with h5py.File(episode / "tactile_information.hdf5", "w") as handle:
            for dataset in WRENCH_DATASETS.values():
                handle.create_dataset(dataset, data=np.ones((length, 6)))
    return episode


@pytest.mark.parametrize(
    ("contact", "repo_id"),
    ((False, VISION_ONLY_REPO_ID), (True, VISION_TACTILE_REPO_ID)),
)
def test_official_reader_streams_same_row_commands_and_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contact: bool,
    repo_id: str,
) -> None:
    length = 4
    root = _episode(tmp_path, length=length, contact=contact)
    frame_count = length + int(contact)
    monkeypatch.setattr(
        reader,
        "_video_header",
        lambda path: reader.VideoHeader(path, frame_count, 8, 10, 30.0),
    )
    routes = build_official_routes()
    audited = reader.audit_official_agilex_episode(
        root=root,
        raw_root=tmp_path,
        task="insert" if contact else "clean_table",
        source_episode_id=0,
        route=routes[repo_id],
        task_prompt="do task",
    )
    monkeypatch.setattr(
        reader,
        "_video_frames",
        lambda header: iter(
            np.full((8, 10, 3), index, dtype=np.uint8)
            for index in range(header.frame_count)
        ),
    )
    frames = list(audited.iter_converted_frames())
    assert len(frames) == length
    assert audited.terminal_video_frame_trimmed is contact
    np.testing.assert_array_equal(frames[0].action, audited.actions[0])
    assert frames[0].action_label_source == "commanded"
    assert tuple(frames[0].tactile) == (tuple(TACTILE_FILES) if contact else ())
    assert tuple(frames[0].wrench) == (tuple(WRENCH_DATASETS) if contact else ())


def test_official_reader_rejects_inconsistent_video_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _episode(tmp_path, length=4, contact=False)
    counts = iter((4, 4, 6))
    monkeypatch.setattr(
        reader,
        "_video_header",
        lambda path: reader.VideoHeader(path, next(counts), 8, 10, 30.0),
    )
    with pytest.raises(ValueError, match="consistently contain T or T\\+1"):
        reader.audit_official_agilex_episode(
            root=root,
            raw_root=tmp_path,
            task="clean_table",
            source_episode_id=0,
            route=build_official_routes()[VISION_ONLY_REPO_ID],
            task_prompt="clean",
        )


def test_official_reader_rejects_measured_qpos_as_missing_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _episode(tmp_path, length=4, contact=False)
    with h5py.File(root / "episode.hdf5", "a") as handle:
        del handle["action"]
    monkeypatch.setattr(
        reader,
        "_video_header",
        lambda path: reader.VideoHeader(path, 4, 8, 10, 30.0),
    )
    with pytest.raises(KeyError):
        reader.audit_official_agilex_episode(
            root=root,
            raw_root=tmp_path,
            task="clean_table",
            source_episode_id=0,
            route=build_official_routes()[VISION_ONLY_REPO_ID],
            task_prompt="clean",
        )
