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
        self.conditioning: list[dict[str, object]] = []

    def reset(self, *, prompt: str, seed: int) -> None:
        self.reset_calls.append((prompt, seed))

    def infer(
        self,
        *,
        images,
        current_ee20,
        precomputed_video_latent=None,
        training_aligned_video_history=None,
    ) -> np.ndarray:
        assert set(images) == {
            "observation.images.top",
            "observation.images.wrist_l",
        }
        assert current_ee20.shape == (20,)
        self.conditioning.append(
            {
                "latent": precomputed_video_latent,
                "history": training_aligned_video_history,
            }
        )
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

    def infer(
        self,
        *,
        images,
        current_ee20,
        precomputed_video_latent=None,
        training_aligned_video_history=None,
    ) -> np.ndarray:
        self.events.append("infer")
        return super().infer(
            images=images,
            current_ee20=current_ee20,
            precomputed_video_latent=precomputed_video_latent,
            training_aligned_video_history=training_aligned_video_history,
        )

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


def _strict_config(tmp_path: Path, *, external_chunk_actions: int = 1) -> Path:
    path = _config(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 4 if external_chunk_actions == 6 else 3
    payload["external_chunk_actions"] = external_chunk_actions
    payload["live_contract"] = {
        "conditioning_mode": "training_aligned_rgb_replan_v1",
        "target_fps": 10,
        "frame_interval_tolerance_ms": 15.0,
        "max_state_image_skew_ms": 50.0,
        "max_history_frames": 5,
    }
    if payload["schema_version"] == 4:
        payload["live_contract"].update(
            {
                "action_hz": 15,
                "actions_per_replan": 6,
                "future_start_index": 6,
                "require_dual_camera_history": True,
                "require_fresh_observation_after_chunk": True,
            }
        )
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


def _video_history(length: int) -> list[dict[str, np.ndarray]]:
    return [
        {
            "cam_high": np.full((8, 8, 3), index, dtype=np.uint8),
            "cam_left_wrist": np.full((8, 8, 3), index + 1, dtype=np.uint8),
        }
        for index in range(length)
    ]


def _strict_observation(
    *, sequence_id: int, first_frame: int, state_skew_ms: int = 20
) -> dict[str, object]:
    history = [
        {
            "cam_high": np.full((8, 8, 3), first_frame + index, np.uint8),
            "cam_left_wrist": np.full(
                (8, 8, 3), first_frame + index + 1, np.uint8
            ),
        }
        for index in range(5)
    ]
    timestamps = [
        (first_frame + index) * 100_000_000 for index in range(len(history))
    ]
    observation = _observation(first_frame + 4)
    observation.update(
        {
            "training_aligned_video_history": history,
            "training_aligned_video_timestamps_ns": timestamps,
            "image_timestamp_ns": timestamps[-1],
            "state_timestamp_ns": timestamps[-1] + state_skew_ms * 1_000_000,
            "observation_sequence_id": sequence_id,
        }
    )
    return observation


def test_strict_live_policy_replans_once_and_discards_unexecuted_actions(
    tmp_path: Path,
) -> None:
    backend = _Backend(_backend_output())
    policy = Policy(str(_strict_config(tmp_path)), backend=backend)
    policy.reset({"prompt": "clear the table"})

    first = policy.infer(_strict_observation(sequence_id=0, first_frame=0))
    second = policy.infer(_strict_observation(sequence_id=1, first_frame=1))

    assert first["actions"][0, 0] == pytest.approx(0.06)
    assert second["actions"][0, 0] == pytest.approx(0.06)
    assert backend.reset_calls == [("clear the table", 11), ("clear the table", 12)]
    assert len(backend.conditioning) == 2
    assert backend.committed == []
    for result in (first, second):
        assert result["policy_timing"]["kind"] == "strict_live_generation"
        assert result["policy_timing"]["generated"] is True
        assert result["policy_timing"]["grounded"] is False
        assert result["policy_timing"]["queue_depth_after"] == 0
        metadata = result["policy_metadata"]
        assert metadata["execution_mode"] == "replan_each_observation"
        assert metadata["selected_prediction_index"] == 6
        assert metadata["discarded_prediction_count"] == 11
        assert metadata["live_contract"]["status"] == "verified"


def test_strict_live_policy_returns_future_six_then_replans_from_new_observation(
    tmp_path: Path,
) -> None:
    backend = _Backend(_backend_output())
    policy = Policy(
        str(_strict_config(tmp_path, external_chunk_actions=6)), backend=backend
    )
    policy.reset({"prompt": "clear the table"})

    first = policy.infer(_strict_observation(sequence_id=0, first_frame=0))
    second = policy.infer(_strict_observation(sequence_id=1, first_frame=1))
    third = policy.infer(_strict_observation(sequence_id=2, first_frame=2))

    assert first["actions"].shape == (6, 8)
    assert second["actions"].shape == (6, 8)
    assert third["actions"].shape == (6, 8)
    assert first["actions"][:, 0] == pytest.approx(np.linspace(0.06, 0.11, 6))
    assert second["actions"][:, 0] == pytest.approx(np.linspace(0.06, 0.11, 6))
    assert third["actions"][:, 0] == pytest.approx(np.linspace(0.06, 0.11, 6))
    assert backend.reset_calls == [
        ("clear the table", 11),
        ("clear the table", 12),
        ("clear the table", 13),
    ]
    assert len(backend.conditioning) == 3
    assert backend.committed == []

    for result in (first, second, third):
        assert result["policy_timing"]["generated"] is True
        assert result["policy_timing"]["kind"] == "strict_live_generation"
        assert result["policy_timing"]["queue_depth_after"] == 0
        metadata = result["policy_metadata"]
        assert metadata["execution_mode"] == "future6_then_fresh_replan"
        assert metadata["chunk_size"] == 6
        assert metadata["selected_prediction_index"] == 6
        assert metadata["selected_prediction_end"] == 12
        assert metadata["selected_prediction_range"] == [6, 12]
        assert metadata["discarded_prediction_count"] == 6
        assert metadata["discarded_future_prediction_count"] == 0
        assert metadata["conditioning_slots_skipped"] == 6
        assert metadata["returned_future_actions"] == 6
        assert metadata["requested_action_hz"] == 15
        assert metadata["expected_execution_duration_ms"] == pytest.approx(400.0)
        assert metadata["requires_fresh_observation_after_chunk"] is True
        assert metadata["live_contract"]["status"] == "verified"


def test_strict_live_policy_fails_closed_on_missing_or_stale_timing(
    tmp_path: Path,
) -> None:
    backend = _Backend(_backend_output())
    policy = Policy(str(_strict_config(tmp_path)), backend=backend)

    missing = _observation(4)
    missing["training_aligned_video_history"] = _video_history(5)
    with pytest.raises(ValueError, match="timestamps_ns length mismatch"):
        policy.infer(missing)

    policy.infer(_strict_observation(sequence_id=0, first_frame=0))
    with pytest.raises(ValueError, match="sequence_id must increase"):
        policy.infer(_strict_observation(sequence_id=0, first_frame=1))
    assert len(backend.conditioning) == 1


def test_strict_offline_policy_can_score_last_future_action(tmp_path: Path) -> None:
    backend = _Backend(_backend_output())
    policy = Policy(str(_strict_config(tmp_path)), backend=backend)
    observation = _strict_observation(sequence_id=0, first_frame=0)
    observation.update(
        {
            "evaluation_mode": "offline_action_selection",
            "evaluation_action_selection": "last_future_action",
        }
    )

    result = policy.infer(observation)

    assert result["actions"][0, 0] == pytest.approx(0.11)
    metadata = result["policy_metadata"]
    assert metadata["action_selection"] == "last_future_action"
    assert metadata["selected_prediction_index"] == 11
    assert metadata["discarded_prediction_count"] == 11
    assert result["policy_timing"]["queue_depth_after"] == 0


def test_policy_passes_training_aligned_rgb_prefix_to_real_backend(
    tmp_path: Path,
) -> None:
    backend = _Backend(_backend_output())
    policy = Policy(str(_config(tmp_path)), backend=backend)
    observation = _observation(4)
    observation["training_aligned_video_history"] = _video_history(5)

    result = policy.infer(observation)

    assert result["policy_timing"]["generated"] is True
    assert result["policy_metadata"]["conditioning_source"] == (
        "training_aligned_raw_rgb"
    )
    history = backend.conditioning[0]["history"]
    assert isinstance(history, tuple)
    assert len(history) == 5
    assert set(history[-1]) == {
        "observation.images.top",
        "observation.images.wrist_l",
    }
    assert np.array_equal(
        history[-1]["observation.images.top"],
        observation["images"]["cam_high"],
    )


def test_policy_cached_latent_uses_same_public_policy_path(tmp_path: Path) -> None:
    backend = _Backend(_backend_output())
    policy = Policy(str(_config(tmp_path)), backend=backend)
    latent = object()
    observation = _observation()
    observation["precomputed_video_latent"] = latent

    result = policy.infer(observation)

    assert result["policy_timing"]["generated"] is True
    assert result["policy_metadata"]["conditioning_source"] == (
        "precomputed_video_latent"
    )
    assert backend.conditioning[0]["latent"] is latent
    assert backend.conditioning[0]["history"] is None


def test_policy_rejects_ambiguous_or_unused_explicit_conditioning(
    tmp_path: Path,
) -> None:
    backend = _Backend(_backend_output())
    policy = Policy(str(_config(tmp_path)), backend=backend)
    ambiguous = _observation()
    ambiguous["precomputed_video_latent"] = object()
    ambiguous["training_aligned_video_history"] = _video_history(1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        policy.infer(ambiguous)
    assert backend.reset_calls == []

    policy.infer(_observation())
    queued = _observation(1)
    queued["training_aligned_video_history"] = _video_history(1)
    with pytest.raises(ValueError, match="only valid when a new plan"):
        policy.infer(queued)
    assert len(backend.conditioning) == 1


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
