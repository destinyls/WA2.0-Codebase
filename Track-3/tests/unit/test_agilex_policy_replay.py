# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from n0_twam.integrations.worldarena.agilex_policy_replay import (
    publish_agilex_replay_observation,
    run_agilex_policy_replay,
)
from tests.unit.test_agilex_policy_io import _policy_fixture


class _Backend:
    def __init__(self) -> None:
        self.reset_count = 0
        self.commit_count = 0

    def reset(self, *, task_id: str, prompt: str, seed: int, profile: str) -> None:
        assert (task_id, prompt, profile) == (
            "wipe",
            "wipe the table",
            "vision_tactile",
        )
        assert seed >= 20260811
        self.reset_count += 1

    def infer(
        self,
        *,
        images,
        current_qpos14,
        tactile_images,
        wrench,
    ) -> np.ndarray:
        assert set(images) == {"top", "wrist_l", "wrist_r"}
        assert set(tactile_images) == {
            "observation.images.tactile_l",
            "observation.images.tactile_r",
        }
        assert set(wrench) == {
            "observation.wrench.left",
            "observation.wrench.right",
        }
        return np.repeat((current_qpos14 + 1.0)[None], 12, axis=0).astype(np.float32)

    def commit_executed_chunk(self, **kwargs) -> None:
        assert np.asarray(kwargs["actions_qpos14_cfh"]).shape == (14, 1, 12)
        assert len(kwargs["image_history"]) == 4
        assert len(kwargs["tactile_history"]) == 4
        assert len(kwargs["wrench_history"]) == 4
        self.commit_count += 1

    def close(self) -> None:
        pass


def _observation(tmp_path: Path) -> Path:
    output = tmp_path / "observation.npz"
    np.savez(
        output,
        schema_version=np.asarray(1, dtype=np.int64),
        task_id=np.asarray("wipe"),
        images=np.zeros((3, 8, 8, 3), dtype=np.uint8),
        joint_qpos=np.zeros(14, dtype=np.float32),
        execution_dt_s=np.asarray(0.1, dtype=np.float64),
        tactile_keys=np.asarray(
            (
                "observation.images.tactile_l",
                "observation.images.tactile_r",
            )
        ),
        tactile_images=np.zeros((2, 8, 8, 3), dtype=np.uint8),
        wrench_keys=np.asarray(("observation.wrench.left", "observation.wrench.right")),
        wrench=np.zeros((2, 6), dtype=np.float32),
    )
    return output


def test_replay_observation_publisher_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "published.npz"
    result = publish_agilex_replay_observation(
        output=output,
        task_id="wipe",
        images={
            key: np.zeros((8, 8, 3), dtype=np.uint8)
            for key in ("top", "wrist_l", "wrist_r")
        },
        joint_qpos=np.zeros(14, dtype=np.float32),
        execution_dt_s=0.1,
        tactile=None,
        wrench=None,
    )

    assert output.is_file()
    assert len(result["sha256"]) == 64


def test_agilex_policy_replay_seals_actions_safety_and_latency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _ = _policy_fixture(tmp_path)
    monkeypatch.setattr(
        "n0_twam.integrations.worldarena.agilex_policy_io."
        "verify_agilex_policy_artifacts",
        lambda _: {},
    )
    output = tmp_path / "replay.json"
    backend = _Backend()

    result = run_agilex_policy_replay(
        config_path=config,
        observation_path=_observation(tmp_path),
        steps=13,
        control_hz=10.0,
        output=output,
        backend=backend,
    )

    assert result["status"] == "complete"
    assert result["execution_tier"] == "offline_engineering_smoke"
    assert result["organizer_evaluation_completed"] is False
    assert result["real_robot_evaluation_completed"] is False
    assert result["steps"] == 13
    assert result["action_schema"] == "qpos14_joint_absolute_v1"
    assert result["tactile_profile"] == "vision_tactile"
    assert result["safety_intervention_count"] > 0
    assert result["grounded_chunk_count"] == 1
    assert len(result["executed_actions"]) == 13
    assert result["realtime_assessment"]["sample_count"] == 13
    assert backend.reset_count == 1
    assert backend.commit_count == 1
    assert len(result["receipt_file_sha256"]) == 64
    assert output.is_file()


def test_agilex_policy_replay_rejects_bad_modality_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _ = _policy_fixture(tmp_path)
    monkeypatch.setattr(
        "n0_twam.integrations.worldarena.agilex_policy_io."
        "verify_agilex_policy_artifacts",
        lambda _: {},
    )
    bad = tmp_path / "bad.npz"
    np.savez(bad, task_id=np.asarray("wipe"))
    output = tmp_path / "replay.json"

    with pytest.raises(ValueError, match="keys differ"):
        run_agilex_policy_replay(
            config_path=config,
            observation_path=bad,
            steps=4,
            output=output,
            backend=_Backend(),
        )
    assert not output.exists()
