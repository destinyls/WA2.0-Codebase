"""Unit tests for immutable DatasetView-to-qpos8 sample bindings."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from n0_twam.integrations.univtac.dataset_view import (
    DatasetView,
    DatasetViewEntry,
)
from n0_twam.integrations.univtac.view_binding import build_view_sample_metadata


def _view(tmp_path: Path) -> DatasetView:
    entries = tuple(
        DatasetViewEntry(
            relative_path=f"insert_HDMI/clean/{episode_id}.hdf5",
            realpath=str(tmp_path / f"{episode_id}.hdf5"),
            source_sha256=character * 64,
            task="insert_HDMI",
            source_split="train",
            source_episode_id=episode_id,
            lerobot_episode_id=lerobot_id,
        )
        for episode_id, lerobot_id, character in ((10, 0, "a"), (11, 1, "b"))
    )
    return DatasetView(
        view_id="test_train2_v1",
        role="training",
        physical_split="train759",
        source_manifest_sha256="c" * 64,
        tasks=("insert_HDMI",),
        entries=entries,
        selection_method="test",
    )


def _dataset() -> SimpleNamespace:
    return SimpleNamespace(
        new_metas=[
            {
                "episode_index": 0,
                "tasks": ["insert_HDMI"],
                "start_frame": 0,
                "end_frame": 45,
            },
            {
                "episode_index": 1,
                "tasks": ["insert_HDMI"],
                "start_frame": 0,
                "end_frame": 45,
            },
        ]
    )


def test_view_sample_metadata_is_stable_and_complete(tmp_path: Path) -> None:
    metadata = build_view_sample_metadata(
        view=_view(tmp_path),
        datasets=[_dataset()],
        crop_window_policy_id="epoch_content_addressed_crop_v1",
    )

    assert metadata["sample_tasks"] == ("insert_HDMI", "insert_HDMI")
    assert metadata["sample_source_episode_ids"] == (
        ("train759", 0),
        ("train759", 1),
    )
    assert metadata["sample_relative_paths"] == (
        "insert_HDMI/clean/10.hdf5",
        "insert_HDMI/clean/11.hdf5",
    )
    sample_ids = metadata["sample_ids"]
    assert len(sample_ids) == len(set(sample_ids)) == 2
    assert all(isinstance(value, str) and len(value) == 64 for value in sample_ids)


def test_view_sample_metadata_rejects_unusable_declared_episode(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    dataset.new_metas.pop()

    with pytest.raises(ValueError, match="unusable episodes"):
        build_view_sample_metadata(
            view=_view(tmp_path),
            datasets=[dataset],
            crop_window_policy_id="epoch_content_addressed_crop_v1",
        )


def test_view_sample_metadata_rejects_segment_outside_view(tmp_path: Path) -> None:
    dataset = _dataset()
    dataset.new_metas.append(
        {
            "episode_index": 2,
            "tasks": ["insert_HDMI"],
            "start_frame": 0,
            "end_frame": 45,
        }
    )

    with pytest.raises(ValueError, match="outside the active dataset view"):
        build_view_sample_metadata(
            view=_view(tmp_path),
            datasets=[dataset],
            crop_window_policy_id="epoch_content_addressed_crop_v1",
        )


def test_view_sample_metadata_binds_crop_policy(tmp_path: Path) -> None:
    first = build_view_sample_metadata(
        view=_view(tmp_path),
        datasets=[_dataset()],
        crop_window_policy_id="policy_a",
    )
    second = build_view_sample_metadata(
        view=_view(tmp_path),
        datasets=[_dataset()],
        crop_window_policy_id="policy_b",
    )

    assert first["sample_ids"] != second["sample_ids"]
