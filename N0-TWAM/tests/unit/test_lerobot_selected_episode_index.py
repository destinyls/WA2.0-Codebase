"""Regression coverage for non-contiguous immutable DatasetView selections."""

from __future__ import annotations

import pytest

from n0_twam.dataset.episode_indexing import (
    build_episode_data_index_positions,
    resolve_episode_data_index_position,
)


def test_selected_episode_uses_compact_data_index_position() -> None:
    positions = build_episode_data_index_positions([5, 6, 7, 8, 9, 20, 21, 22, 23, 24])

    assert resolve_episode_data_index_position(positions, 5) == 0
    assert resolve_episode_data_index_position(positions, 20) == 5
    assert resolve_episode_data_index_position(positions, 24) == 9


def test_selected_episode_rejects_ids_outside_the_immutable_view() -> None:
    positions = build_episode_data_index_positions([5, 20])

    with pytest.raises(ValueError, match="absent from the selected LeRobot view"):
        resolve_episode_data_index_position(positions, 10)


def test_unfiltered_episode_keeps_its_original_position() -> None:
    assert resolve_episode_data_index_position(None, 20) == 20
