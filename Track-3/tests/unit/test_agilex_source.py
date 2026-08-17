# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed AgileX source and label-provenance contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from n0_twam.actions.qpos14 import CHANNEL_NAMES, CHANNEL_UNITS, GRIPPER_ENCODING
from n0_twam.embodiments import AGILEX_RGB_KEYS
from n0_twam.integrations.worldarena.agilex_actions import (
    ENGINEERING_MEASURED_NEXT_CONTRACT,
    MEASURED_NEXT_QPOS_SCHEMA,
    QPOS14_ACTION_SCHEMA,
)
from n0_twam.integrations.worldarena.agilex_manifest import (
    AgileXFileRecord,
    build_repo_route,
    load_agilex_manifest,
    sha256_file,
)
from n0_twam.integrations.worldarena.agilex_source import audit_episode


def _route(*, formal: bool = True):
    return build_repo_route(
        "agilex_fixture",
        {
            "embodiment": "agilex_dual_qpos14_v1",
            "action_schema": (
                QPOS14_ACTION_SCHEMA if formal else MEASURED_NEXT_QPOS_SCHEMA
            ),
            "action_label_source": "commanded" if formal else "measured_next",
            "formal": formal,
            "rgb_keys": list(AGILEX_RGB_KEYS),
            "tactile_keys": ["observation.images.tactile_l"],
            "wrench_keys": ["observation.wrench.left"],
            "channel_names": list(CHANNEL_NAMES),
            "channel_units": list(CHANNEL_UNITS),
            "gripper_encoding": GRIPPER_ENCODING,
            "temporal_alignment_identity": "a" * 64,
        },
        allow_engineering=not formal,
    )


def _arrays(length: int = 4):
    qpos = np.arange(length * 14, dtype=np.float32).reshape(length, 14) / 100.0
    commanded = qpos + np.float32(0.25)
    images = {
        key: np.full((length, 4, 5, 3), index, dtype=np.uint8)
        for index, key in enumerate(AGILEX_RGB_KEYS)
    }
    tactile = {
        "observation.images.tactile_l": np.ones((length, 3, 2, 3), dtype=np.uint8)
    }
    wrench = {
        "observation.wrench.left": np.arange(length * 6, dtype=np.float32).reshape(
            length, 6
        )
    }
    return qpos, commanded, images, tactile, wrench


def test_formal_route_rejects_noncanonical_qpos14_layout() -> None:
    payload = {
        "embodiment": "agilex_dual_qpos14_v1",
        "action_schema": QPOS14_ACTION_SCHEMA,
        "action_label_source": "commanded",
        "formal": True,
        "rgb_keys": list(AGILEX_RGB_KEYS),
        "tactile_keys": [],
        "wrench_keys": [],
        "channel_names": list(CHANNEL_NAMES),
        "channel_units": list(CHANNEL_UNITS),
        "gripper_encoding": GRIPPER_ENCODING,
        "temporal_alignment_identity": "a" * 64,
    }
    payload["channel_names"][0] = "unexpected_channel"
    with pytest.raises(ValueError, match="canonical qpos14 layout"):
        build_repo_route("agilex_fixture", payload)

    payload["channel_names"] = list(CHANNEL_NAMES)
    payload["rgb_keys"] = [AGILEX_RGB_KEYS[0]]
    with pytest.raises(ValueError, match="canonical ordered RGB"):
        build_repo_route("agilex_fixture", payload)


def test_formal_episode_preserves_same_row_commanded_actions() -> None:
    qpos, commanded, images, tactile, wrench = _arrays()

    episode = audit_episode(
        route=_route(),
        source_relative_path="agilex_fixture/episode_000.npz",
        episode_id=0,
        task="insert_plug",
        timestamps=np.asarray((0.0, 0.1, 0.2, 0.3), dtype=np.float64),
        joint_qpos=qpos,
        actions=commanded,
        action_valid_mask=np.ones(4, dtype=np.bool_),
        rgb=images,
        tactile=tactile,
        wrench=wrench,
    )

    np.testing.assert_array_equal(episode.actions, commanded)
    np.testing.assert_array_equal(episode.joint_qpos, qpos)
    assert episode.formal is True
    assert episode.action_label_source == "commanded"


def test_formal_episode_rejects_missing_action_instead_of_inferring_qpos_next() -> None:
    qpos, _, images, tactile, wrench = _arrays()

    with pytest.raises(ValueError, match="formal.*commanded/executed"):
        audit_episode(
            route=_route(),
            source_relative_path="agilex_fixture/episode_000.npz",
            episode_id=0,
            task="insert_plug",
            timestamps=np.arange(4, dtype=np.float64),
            joint_qpos=qpos,
            actions=None,
            action_valid_mask=None,
            rgb=images,
            tactile=tactile,
            wrench=wrench,
        )


def test_measured_next_labels_require_explicit_engineering_contract() -> None:
    qpos, _, images, tactile, wrench = _arrays()
    kwargs = {
        "route": _route(formal=False),
        "source_relative_path": "agilex_fixture/episode_000.npz",
        "episode_id": 0,
        "task": "engineering_smoke",
        "timestamps": np.arange(4, dtype=np.float64),
        "joint_qpos": qpos,
        "actions": None,
        "action_valid_mask": None,
        "rgb": images,
        "tactile": tactile,
        "wrench": wrench,
    }

    with pytest.raises(ValueError, match="explicit engineering contract"):
        audit_episode(**kwargs)

    episode = audit_episode(
        **kwargs,
        engineering_label_contract=ENGINEERING_MEASURED_NEXT_CONTRACT,
    )
    np.testing.assert_array_equal(episode.actions[:-1], qpos[1:])
    np.testing.assert_array_equal(episode.actions[-1], qpos[-1])
    np.testing.assert_array_equal(
        episode.action_valid_mask,
        np.asarray((True, True, True, False)),
    )
    assert episode.formal is False


def test_episode_rejects_non_monotonic_time_and_missing_declared_sensor() -> None:
    qpos, commanded, images, tactile, wrench = _arrays()
    with pytest.raises(ValueError, match="timestamps.*increasing"):
        audit_episode(
            route=_route(),
            source_relative_path="agilex_fixture/episode_000.npz",
            episode_id=0,
            task="insert_plug",
            timestamps=np.asarray((0.0, 0.1, 0.1, 0.2)),
            joint_qpos=qpos,
            actions=commanded,
            action_valid_mask=np.ones(4, dtype=np.bool_),
            rgb=images,
            tactile={},
            wrench=wrench,
        )


def test_source_record_rejects_symlink_parent_escape(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    payload = external / "episode.bin"
    payload.write_bytes(b"outside")
    (root / "alias").symlink_to(external, target_is_directory=True)
    record = AgileXFileRecord(
        relative_path="alias/episode.bin",
        size_bytes=payload.stat().st_size,
        sha256=sha256_file(payload),
    )

    with pytest.raises(ValueError, match="contains a symlink"):
        record.verify(root)


def test_manifest_loader_rejects_leaf_symlink(tmp_path: Path) -> None:
    source = tmp_path / "manifest.json"
    source.write_text("{}", encoding="utf-8")
    link = tmp_path / "manifest-link.json"
    link.symlink_to(source)

    with pytest.raises(ValueError, match="non-symlink"):
        load_agilex_manifest(
            link,
            expected_file_sha256=sha256_file(source),
        )
