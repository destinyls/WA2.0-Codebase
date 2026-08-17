# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Canonical AgileX source-to-LeRobot conversion tests."""

from __future__ import annotations

import numpy as np

from n0_twam.actions.qpos14 import CHANNEL_NAMES, CHANNEL_UNITS, GRIPPER_ENCODING
from n0_twam.embodiments import AGILEX_RGB_KEYS
from n0_twam.integrations.worldarena.agilex_actions import QPOS14_ACTION_SCHEMA
from n0_twam.integrations.worldarena.agilex_convert import (
    build_lerobot_features,
    iter_episode_frames,
)
from n0_twam.integrations.worldarena.agilex_manifest import build_repo_route
from n0_twam.integrations.worldarena.agilex_source import audit_episode


def _episode():
    route = build_repo_route(
        "repo",
        {
            "embodiment": "agilex_dual_qpos14_v1",
            "action_schema": QPOS14_ACTION_SCHEMA,
            "action_label_source": "executed",
            "formal": True,
            "rgb_keys": list(AGILEX_RGB_KEYS),
            "tactile_keys": [],
            "wrench_keys": [
                "observation.wrench.left",
                "observation.wrench.right",
            ],
            "channel_names": list(CHANNEL_NAMES),
            "channel_units": list(CHANNEL_UNITS),
            "gripper_encoding": GRIPPER_ENCODING,
            "temporal_alignment_identity": "b" * 64,
        },
    )
    qpos = np.arange(42, dtype=np.float32).reshape(3, 14)
    actions = qpos + 100.0
    return audit_episode(
        route=route,
        source_relative_path="repo/episode_002.npz",
        episode_id=2,
        task="wipe",
        timestamps=np.asarray((1.0, 1.1, 1.2), dtype=np.float64),
        joint_qpos=qpos,
        actions=actions,
        action_valid_mask=np.asarray((True, True, False)),
        rgb={
            key: np.full((3, 8, 6, 3), index, dtype=np.uint8)
            for index, key in enumerate(AGILEX_RGB_KEYS)
        },
        tactile={},
        wrench={
            "observation.wrench.left": np.ones((3, 6), dtype=np.float32),
            "observation.wrench.right": np.full((3, 6), 2.0, dtype=np.float32),
        },
    )


def test_conversion_keeps_official_action_on_the_same_source_row() -> None:
    episode = _episode()

    frames = tuple(iter_episode_frames(episode))

    assert len(frames) == 3
    np.testing.assert_array_equal(frames[0].action, episode.actions[0])
    np.testing.assert_array_equal(frames[1].action, episode.actions[1])
    assert [frame.action_valid for frame in frames] == [True, True, False]
    assert frames[0].source_row_index == 0


def test_converted_frame_and_features_publish_label_provenance_and_wrench() -> None:
    episode = _episode()
    frame = next(iter_episode_frames(episode))

    payload = frame.to_lerobot_dict()
    features = build_lerobot_features(frame, route=episode.route)

    assert payload["action.label_source"] == "executed"
    assert payload["action.schema"] == QPOS14_ACTION_SCHEMA
    assert payload["action.valid"].dtype == np.bool_
    assert payload["observation.joint_qpos"].shape == (14,)
    assert payload["observation.wrench.left"].shape == (6,)
    assert features["action"]["shape"] == (14,)
    assert features["action"]["names"] == list(episode.route.channel_names)
    assert features["action.valid"]["dtype"] == "bool"


def test_converted_arrays_are_detached_from_episode_storage() -> None:
    episode = _episode()
    payload = next(iter_episode_frames(episode)).to_lerobot_dict()

    payload["action"][0] = -999.0

    assert episode.actions[0, 0] != -999.0
