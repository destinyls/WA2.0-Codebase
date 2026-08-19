# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from n0_twam.integrations.worldarena.franka_actions import (
    ee10_to_end_pose8,
    embed_ee10_in_ee20,
    end_pose8_to_ee10,
    extract_ee10_from_ee20,
)
from n0_twam.integrations.worldarena.franka_policy import Policy


class _Backend:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.reset_calls: list[tuple[str, int]] = []
        self.committed: list[dict[str, object]] = []

    def reset(self, *, prompt: str, seed: int) -> None:
        self.reset_calls.append((prompt, seed))

    def infer(self, *, images, current_ee20) -> np.ndarray:
        assert set(images) == {
            "observation.images.top",
            "observation.images.wrist_l",
        }
        assert current_ee20.shape == (20,)
        return self.output.copy()

    def commit_executed_chunk(
        self, *, actions_ee20_cfh, image_history, action_anchor_ee20
    ) -> None:
        self.committed.append(
            {
                "actions": actions_ee20_cfh.copy(),
                "images": tuple(
                    {name: image.copy() for name, image in row.items()}
                    for row in image_history
                ),
                "anchor": action_anchor_ee20.copy(),
            }
        )


class _OrderedBackend(_Backend):
    def __init__(self, output: np.ndarray) -> None:
        super().__init__(output)
        self.events: list[str] = []

    def infer(self, *, images, current_ee20) -> np.ndarray:
        self.events.append("infer")
        return super().infer(images=images, current_ee20=current_ee20)

    def commit_executed_chunk(
        self, *, actions_ee20_cfh, image_history, action_anchor_ee20
    ) -> None:
        self.events.append("commit")
        super().commit_executed_chunk(
            actions_ee20_cfh=actions_ee20_cfh,
            image_history=image_history,
            action_anchor_ee20=action_anchor_ee20,
        )


def _config(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    payload = {
        "schema_version": 2,
        "policy_id": "n0-twam-franka-test",
        "serve_bundle": str(bundle),
        "serve_bundle_receipt_sha256": "a" * 64,
        "serve_output": str(tmp_path / "output"),
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


def _backend_output() -> np.ndarray:
    poses = np.repeat(
        np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5]], np.float32),
        12,
        axis=0,
    )
    poses[:, 0] = np.linspace(0.0, 0.11, 12, dtype=np.float32)
    ee20 = embed_ee10_in_ee20(end_pose8_to_ee10(poses))
    return ee20.reshape(2, 6, 20).transpose(2, 0, 1)


def _unsafe_backend_output() -> np.ndarray:
    poses = np.repeat(
        np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5]], np.float32),
        12,
        axis=0,
    )
    ee20 = embed_ee10_in_ee20(end_pose8_to_ee10(poses))
    return ee20.reshape(2, 6, 20).transpose(2, 0, 1)


def _observation(frame_value: int = 0) -> dict[str, object]:
    return {
        "prompt": "clear the table",
        "task_id": "clear_up",
        "images": {
            "cam_high": np.full((8, 8, 3), frame_value, dtype=np.uint8),
            "cam_left_wrist": np.full((8, 8, 3), frame_value + 1, dtype=np.uint8),
        },
        "left_end_pose": np.asarray((0, 0, 0, 0, 0, 0, 1), np.float32),
        "joint_qpos": np.asarray((0, 0, 0, 0, 0, 0, 0, 0.5), np.float32),
    }


def test_policy_emits_one_action_and_grounds_cold_chunk(tmp_path: Path) -> None:
    backend = _Backend(_backend_output())
    policy = Policy(str(_config(tmp_path)), backend=backend)
    policy.reset({"prompt": "clear the table"})

    result = policy.infer(_observation(0))

    actions = result["actions"]
    assert actions.shape == (1, 8)
    assert actions.dtype == np.float32
    assert np.isfinite(actions).all()
    np.testing.assert_allclose(np.linalg.norm(actions[:, 3:7], axis=1), 1.0)
    assert actions[0, 0] == pytest.approx(0.06)
    assert result["policy_metadata"]["action_format"] == "end_pose_base"
    assert result["policy_metadata"]["control_arm"] == "right"
    assert result["policy_metadata"]["chunk_size"] == 1
    assert result["policy_metadata"]["quaternion_order"] == "xyzw"
    assert result["policy_metadata"]["wire_action_schema"] == (
        "franka_end_pose_base_xyzw8_v2"
    )
    assert result["policy_metadata"]["derived_action_schema"] == (
        "franka_ee10_rot6d_columns_from_xyzw_v2"
    )
    assert result["policy_metadata"]["tactile_mode"] == "disabled"
    assert result["policy_metadata"]["tactile_profile"] == "vision_only"
    assert result["policy_timing"]["kind"] == "cold_generation"
    assert result["policy_timing"]["generated"] is True
    assert result["policy_timing"]["grounded"] is False
    assert result["policy_timing"]["queue_depth_after"] == 5
    assert backend.reset_calls == [("clear the table", 11)]
    assert backend.committed == []

    for index in range(1, 6):
        queued = policy.infer(_observation(index))
        assert queued["actions"].shape == (1, 8)
        assert queued["policy_timing"]["kind"] == "queue_hit"
        assert queued["policy_timing"]["generated"] is False
    next_chunk = policy.infer(_observation(6))

    assert next_chunk["actions"].shape == (1, 8)
    assert next_chunk["actions"][0, 0] == pytest.approx(0.0)
    assert next_chunk["policy_timing"]["kind"] == "grounding_refill"
    assert next_chunk["policy_timing"]["generated"] is True
    assert next_chunk["policy_timing"]["grounded"] is True
    assert next_chunk["policy_timing"]["queue_depth_after"] == 11
    assert len(backend.committed) == 1
    committed = backend.committed[0]
    assert np.asarray(committed["actions"]).shape == (20, 2, 6)
    assert not np.asarray(committed["actions"])[10:].any()
    assert np.asarray(committed["anchor"]).shape == (20,)
    history = committed["images"]
    assert isinstance(history, tuple)
    assert len(history) == 4
    assert [int(row["observation.images.top"][0, 0, 0]) for row in history] == [
        2,
        3,
        4,
        6,
    ]


def test_policy_grounds_all_twelve_regular_actions(tmp_path: Path) -> None:
    backend = _Backend(_backend_output())
    policy = Policy(str(_config(tmp_path)), backend=backend)
    policy.reset({"prompt": "clear the table"})

    for index in range(7):
        policy.infer(_observation(index))
    assert len(backend.committed) == 1

    for index in range(7, 19):
        policy.infer(_observation(index))

    assert len(backend.committed) == 2
    regular_history = backend.committed[1]["images"]
    assert isinstance(regular_history, tuple)
    assert len(regular_history) == 8
    assert [int(row["observation.images.top"][0, 0, 0]) for row in regular_history] == [
        8,
        9,
        10,
        12,
        14,
        15,
        16,
        18,
    ]


def test_policy_reprojects_queued_actions_against_measured_pose_and_commits_them(
    tmp_path: Path,
) -> None:
    backend = _OrderedBackend(_unsafe_backend_output())
    policy = Policy(str(_config(tmp_path)), backend=backend)
    policy.reset({"prompt": "clear the table"})

    emitted = [policy.infer(_observation(index))["actions"][0] for index in range(6)]

    # The fixture deliberately reports that the arm stayed at x=0 after every
    # command.  Every newly emitted command must therefore be re-projected to
    # the 0.2 m single-step limit instead of following the stale planned path.
    np.testing.assert_allclose(np.asarray(emitted)[:, 0], 0.2, atol=1e-6)
    assert backend.events == ["infer"]

    refill = policy.infer(_observation(6))

    assert refill["policy_timing"]["grounded"] is True
    assert backend.events == ["infer", "commit", "infer"]
    assert len(backend.committed) == 1
    committed_cfh = np.asarray(backend.committed[0]["actions"])
    committed_flat = np.moveaxis(committed_cfh, 0, -1).reshape(-1, 20)
    committed_poses = ee10_to_end_pose8(extract_ee10_from_ee20(committed_flat[6:]))
    np.testing.assert_allclose(committed_poses, np.asarray(emitted), atol=1e-6)


def test_policy_reset_discards_incomplete_chunk(tmp_path: Path) -> None:
    backend = _Backend(_backend_output())
    policy = Policy(str(_config(tmp_path)), backend=backend)
    policy.reset({"prompt": "clear the table"})
    policy.infer(_observation())

    policy.reset({"prompt": "clear the table"})
    restarted = policy.infer(_observation())

    assert restarted["actions"][0, 0] == pytest.approx(0.06)
    assert backend.committed == []


def test_policy_rejects_tactile_and_prompt_drift(tmp_path: Path) -> None:
    policy = Policy(str(_config(tmp_path)), backend=_Backend(_backend_output()))
    observation = _observation()
    observation["tactile"] = np.zeros((1,), dtype=np.float32)
    with pytest.raises(ValueError, match="rejects tactile"):
        policy.infer(observation)

    policy.reset({"prompt": "clear the table"})
    observation = _observation()
    observation["prompt"] = "pour water"
    with pytest.raises(ValueError, match="changed without"):
        policy.infer(observation)


def test_vision_only_policy_rejects_nested_tactile_before_backend_mutation(
    tmp_path: Path,
) -> None:
    backend = _Backend(_backend_output())
    policy = Policy(str(_config(tmp_path)), backend=backend)
    observation = _observation()
    images = observation["images"]
    assert isinstance(images, dict)
    images["tactile_left"] = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="rejects tactile"):
        policy.infer(observation)

    assert backend.reset_calls == []


def test_policy_rejects_current_pose_outside_signed_workspace(tmp_path: Path) -> None:
    policy = Policy(str(_config(tmp_path)), backend=_Backend(_backend_output()))
    observation = _observation()
    observation["left_end_pose"] = np.asarray((2.0, 0, 0, 1, 0, 0, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="outside the signed workspace"):
        policy.infer(observation)
