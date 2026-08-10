# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from n0_twam.integrations.worldarena.franka_actions import (
    embed_ee10_in_ee20,
    end_pose8_to_ee10,
)
from n0_twam.integrations.worldarena.franka_manifest import canonical_sha256
from n0_twam.integrations.worldarena.franka_policy_replay import (
    run_franka_policy_replay,
)


class _Backend:
    def __init__(self) -> None:
        self.commit_count = 0

    def reset(self, *, prompt: str, seed: int) -> None:
        assert prompt == "clear the table"
        assert seed >= 11

    def infer(self, *, images, current_ee20) -> np.ndarray:
        assert set(images) == {
            "observation.images.top",
            "observation.images.wrist_l",
        }
        assert current_ee20.shape == (20,)
        poses = np.repeat(
            np.asarray([[0, 0, 0, 1, 0, 0, 0, 0.5]], dtype=np.float32),
            12,
            axis=0,
        )
        poses[:, 0] = np.linspace(0.0, 2.2, 12, dtype=np.float32)
        ee20 = embed_ee10_in_ee20(end_pose8_to_ee10(poses))
        return ee20.reshape(2, 6, 20).transpose(2, 0, 1)

    def commit_executed_chunk(
        self, *, actions_ee20_cfh, image_history, action_anchor_ee20
    ) -> None:
        assert actions_ee20_cfh.shape == (20, 2, 6)
        assert len(image_history) == 4
        assert action_anchor_ee20.shape == (20,)
        self.commit_count += 1


class _MutatingBackend(_Backend):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.mutated = False

    def infer(self, *, images, current_ee20) -> np.ndarray:
        if not self.mutated:
            self.path.write_bytes(self.path.read_bytes() + b"\n")
            self.mutated = True
        return super().infer(images=images, current_ee20=current_ee20)


def _config(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    payload = {
        "schema_version": 2,
        "policy_id": "n0-twam-franka-replay-test",
        "serve_bundle": str(bundle),
        "serve_bundle_receipt_sha256": "a" * 64,
        "serve_output": str(tmp_path / "serve-output"),
        "cuda_visible_device": "0",
        "distributed_port": 29642,
        "episode_seed": 11,
        "max_chunk_actions": 12,
        "external_chunk_actions": 1,
        "video_inference_steps": 3,
        "action_inference_steps": 4,
        "safety": {
            "workspace_min": [-1.0, -1.0, -1.0],
            "workspace_max": [1.0, 1.0, 1.0],
            "max_translation_step_m": 0.2,
            "max_rotation_step_rad": 0.5,
            "gripper_min": 0.0,
            "gripper_max": 1.0,
            "max_gripper_step": 0.2,
        },
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _observation(tmp_path: Path) -> Path:
    path = tmp_path / "observation.npz"
    np.savez(
        path,
        cam_high=np.zeros((8, 8, 3), dtype=np.uint8),
        cam_left_wrist=np.ones((8, 8, 3), dtype=np.uint8),
        left_end_pose=np.asarray((0, 0, 0, 1, 0, 0, 0), dtype=np.float32),
        joint_qpos=np.asarray((0, 0, 0, 0, 0, 0, 0, 0.5), dtype=np.float32),
    )
    return path


def test_offline_policy_replay_commits_cold_chunk_and_seals_receipt(
    tmp_path: Path,
) -> None:
    backend = _Backend()
    output = tmp_path / "replay.json"

    result = run_franka_policy_replay(
        config_path=_config(tmp_path),
        observation_path=_observation(tmp_path),
        prompt="clear the table",
        steps=7,
        output=output,
        backend=backend,
    )

    assert backend.commit_count == 1
    assert result["status"] == "complete"
    assert result["execution_tier"] == "offline_engineering_smoke"
    assert result["organizer_evaluation_completed"] is False
    assert result["real_robot_evaluation_completed"] is False
    assert result["steps"] == 7
    assert len(result["actions"]) == 7
    assert result["safety_intervention_count"] > 0
    assert result["actions"][0][0] == pytest.approx(0.2)
    payload = json.loads(output.read_text(encoding="utf-8"))
    core = {
        key: value for key, value in payload.items() if key != "replay_identity_sha256"
    }
    assert payload["replay_identity_sha256"] == canonical_sha256(core)
    assert len(result["receipt_file_sha256"]) == 64


def test_offline_policy_replay_rejects_bad_inputs_without_output(
    tmp_path: Path,
) -> None:
    observation = _observation(tmp_path)
    bad = tmp_path / "bad.npz"
    np.savez(bad, cam_high=np.zeros((8, 8, 3), dtype=np.uint8))
    output = tmp_path / "replay.json"

    with pytest.raises(ValueError, match="keys differ"):
        run_franka_policy_replay(
            config_path=_config(tmp_path),
            observation_path=bad,
            prompt="clear the table",
            steps=7,
            output=output,
            backend=_Backend(),
        )
    assert not output.exists()

    with pytest.raises(ValueError, match="steps"):
        run_franka_policy_replay(
            config_path=tmp_path / "policy.json",
            observation_path=observation,
            prompt="clear the table",
            steps=6,
            output=output,
            backend=_Backend(),
        )
    assert not output.exists()


def test_offline_policy_replay_rejects_leaf_symlink(tmp_path: Path) -> None:
    observation = _observation(tmp_path)
    alias = tmp_path / "observation-alias.npz"
    alias.symlink_to(observation)

    with pytest.raises(ValueError, match="non-symlink"):
        run_franka_policy_replay(
            config_path=_config(tmp_path),
            observation_path=alias,
            prompt="clear the table",
            steps=7,
            output=tmp_path / "replay.json",
            backend=_Backend(),
        )


@pytest.mark.parametrize("mutated_input", ("config", "observation"))
def test_offline_policy_replay_rejects_input_drift_before_receipt(
    tmp_path: Path,
    mutated_input: str,
) -> None:
    config = _config(tmp_path)
    observation = _observation(tmp_path)
    mutated = config if mutated_input == "config" else observation
    output = tmp_path / "replay.json"

    with pytest.raises(ValueError, match="changed during Policy replay"):
        run_franka_policy_replay(
            config_path=config,
            observation_path=observation,
            prompt="clear the table",
            steps=7,
            output=output,
            backend=_MutatingBackend(mutated),
        )
    assert not output.exists()
