"""Translate immutable LeRobot episode IDs to compact subset positions."""

from __future__ import annotations

from collections.abc import Sequence


def build_episode_data_index_positions(
    episode_ids: Sequence[int] | None,
) -> dict[int, int] | None:
    """Return positions in a compact selected-episode data-index table."""

    if episode_ids is None:
        return None
    return {
        int(episode_id): position for position, episode_id in enumerate(episode_ids)
    }


def resolve_episode_data_index_position(
    positions: dict[int, int] | None,
    episode_index: int,
) -> int:
    """Resolve an original LeRobot episode ID without reindexing its identity."""

    resolved_episode_index = int(episode_index)
    if positions is None:
        return resolved_episode_index
    try:
        return positions[resolved_episode_index]
    except KeyError as exc:
        raise ValueError(
            "episode is absent from the selected LeRobot view: "
            f"{resolved_episode_index}"
        ) from exc
