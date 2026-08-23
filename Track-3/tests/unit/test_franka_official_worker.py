# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from n0_twam.integrations.worldarena.franka_official_worker import (
    BRIDGE_RELATIVE_PATH,
    WORKER_RELATIVE_PATH,
    _capture_identity,
    _physical_width_output_to_open_ratio,
    _probe_loaded_bridge,
    _validated_hub_url,
)
from n0_twam.integrations.worldarena.franka_live_contract import (
    FrankaLiveContractConfig,
    validate_live_observation,
)
from n0_twam.integrations.worldarena.franka_live_observation_adapter import (
    FrankaLiveObservationAdapter,
    LiveAdaptedPolicy,
)


@dataclass
class _Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class _Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class _Pose:
    position_m: _Vector3 = field(default_factory=_Vector3)
    orientation_xyzw: _Quaternion = field(default_factory=_Quaternion)
    frame: str = "base"


@dataclass
class _SessionContext:
    session_id: str = ""
    episode_id: str = ""
    task_id: str = ""
    task_instruction: str = ""


class _Schema:
    Vector3 = _Vector3
    Quaternion = _Quaternion
    Pose = _Pose
    SessionContext = _SessionContext


class _GoodBridge:
    @staticmethod
    def _arm_end_pose_7d(arm):
        pose = arm.ee_pose_base
        quaternion = pose.orientation_xyzw
        return np.asarray(
            (
                pose.position_m.x,
                pose.position_m.y,
                pose.position_m.z,
                quaternion.x,
                quaternion.y,
                quaternion.z,
                quaternion.w,
            ),
            dtype=np.float32,
        )

    @staticmethod
    def actions_array_to_action_packet(actions, **kwargs):
        row = np.asarray(actions)[0]
        quaternion = _Quaternion(
            x=float(row[3]),
            y=float(row[4]),
            z=float(row[5]),
            w=float(row[6]),
        )
        action = SimpleNamespace(
            arm_id=kwargs.get("control_arm"),
            target_pose_base=_Pose(orientation_xyzw=quaternion),
        )
        step = SimpleNamespace(arm_actions=[action])
        return SimpleNamespace(action_chunk=[step])

    @classmethod
    def infer_output_to_action_packet(
        cls, output, *, context, observation_timestamp_ns
    ):
        metadata = output["policy_metadata"]
        return cls.actions_array_to_action_packet(
            output["actions"],
            context=context,
            observation_timestamp_ns=observation_timestamp_ns,
            action_format=metadata["action_format"],
            control_arm=metadata["control_arm"],
        )


class _BrokenInputBridge(_GoodBridge):
    @staticmethod
    def _arm_end_pose_7d(arm):
        pose = arm.ee_pose_base
        quaternion = pose.orientation_xyzw
        return np.asarray(
            (
                pose.position_m.x,
                pose.position_m.y,
                pose.position_m.z,
                quaternion.w,
                quaternion.x,
                quaternion.y,
                quaternion.z,
            ),
            dtype=np.float32,
        )


class _BrokenOutputBridge(_GoodBridge):
    @staticmethod
    def actions_array_to_action_packet(actions, **kwargs):
        row = np.asarray(actions)[0]
        quaternion = _Quaternion(
            x=float(row[4]),
            y=float(row[5]),
            z=float(row[6]),
            w=float(row[3]),
        )
        action = SimpleNamespace(
            arm_id=kwargs.get("control_arm"),
            target_pose_base=_Pose(orientation_xyzw=quaternion),
        )
        return SimpleNamespace(action_chunk=[SimpleNamespace(arm_actions=[action])])


class _WrongArmBridge(_GoodBridge):
    @staticmethod
    def actions_array_to_action_packet(actions, **kwargs):
        packet = _GoodBridge.actions_array_to_action_packet(actions, **kwargs)
        packet.action_chunk[0].arm_actions[0].arm_id = "left"
        return packet


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _identity_repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "worldarena"
    bridge = root / BRIDGE_RELATIVE_PATH
    worker = root / WORKER_RELATIVE_PATH
    bridge.parent.mkdir(parents=True)
    worker.parent.mkdir(parents=True, exist_ok=True)
    (root / "assets").mkdir()
    bridge.write_text("bridge-v1\n", encoding="utf-8")
    worker.write_text("worker-v1\n", encoding="utf-8")
    (root / "assets" / "large.png").write_text("pointer\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "N0 tests")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def test_xyzw_bridge_probe_checks_both_directions() -> None:
    report = _probe_loaded_bridge(_GoodBridge, _Schema)

    assert report["status"] == "pass"
    assert report["control_arm"] == "right"
    expected = np.asarray((0.1, 0.2, 0.3, np.sqrt(0.86)))
    np.testing.assert_allclose(report["probe_new_obs_pose7"][3:], expected)
    np.testing.assert_allclose(report["probe_action_packet_xyzw"], expected)


@pytest.mark.parametrize(
    ("bridge", "message"),
    (
        (_BrokenInputBridge, "canonical-xyzw to Franka-new_obs-xyzw"),
        (_BrokenOutputBridge, "Franka-action-xyzw to canonical-xyzw"),
    ),
)
def test_xyzw_bridge_probe_rejects_wxyz_reordering(
    bridge: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _probe_loaded_bridge(bridge, _Schema)


def test_xyzw_bridge_probe_rejects_wrong_control_arm() -> None:
    with pytest.raises(ValueError, match="control-arm routing"):
        _probe_loaded_bridge(_WrongArmBridge, _Schema)


def test_identity_requires_clean_unmodified_bridge_but_ignores_unrelated_lfs_tree(
    tmp_path: Path,
) -> None:
    root, revision = _identity_repo(tmp_path)
    bridge = root / BRIDGE_RELATIVE_PATH
    (root / "assets" / "large.png").write_text("smudged bytes\n", encoding="utf-8")
    bridge_sha = hashlib.sha256(bridge.read_bytes()).hexdigest()

    identity = _capture_identity(
        root,
        expected_revision=revision,
        expected_bridge_sha256=bridge_sha,
    )

    assert identity["changed_paths"] == []
    assert identity["bridge_sha256"] == bridge_sha

    bridge.write_text("bridge-v2\n", encoding="utf-8")
    changed_sha = hashlib.sha256(bridge.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unmodified"):
        _capture_identity(
            root,
            expected_revision=revision,
            expected_bridge_sha256=changed_sha,
        )

    _git(root, "checkout", "--", BRIDGE_RELATIVE_PATH.as_posix())
    extra = root / "real_world_benchmark" / "shadow.py"
    extra.write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unmodified"):
        _capture_identity(
            root,
            expected_revision=revision,
            expected_bridge_sha256=bridge_sha,
        )


def test_hub_url_is_https_or_explicit_localhost_only() -> None:
    assert (
        _validated_hub_url("https://hub.example.invalid/policy", allow_local_http=False)
        == "https://hub.example.invalid/policy"
    )
    assert (
        _validated_hub_url("http://127.0.0.1:18000/policy", allow_local_http=True)
        == "http://127.0.0.1:18000/policy"
    )
    with pytest.raises(ValueError, match="HTTPS"):
        _validated_hub_url("http://hub.example.invalid/policy", allow_local_http=True)
    with pytest.raises(ValueError, match="credentials"):
        _validated_hub_url(
            "https://secret@hub.example.invalid/policy", allow_local_http=False
        )


@dataclass
class _Camera:
    camera_role: str
    timestamp_ns: int
    frames: tuple[np.ndarray, ...]
    frame_history_timestamps_ns: tuple[int, ...]


@dataclass
class _JointState:
    position_rad: tuple[float, ...] = (0.0,) * 7


@dataclass
class _GripperState:
    width_m: float | None = 0.04
    open_ratio: float = 0.5


@dataclass
class _ArmState:
    arm_id: str = "franka"
    joint_state: _JointState = field(default_factory=_JointState)
    ee_pose_base: _Pose = field(default_factory=_Pose)
    gripper: _GripperState = field(default_factory=_GripperState)


@dataclass
class _RobotState:
    arms: tuple[_ArmState, ...] = field(default_factory=lambda: (_ArmState(),))


@dataclass
class _ObservationPacket:
    observation_timestamp_ns: int
    camera_observations: tuple[_Camera, ...]
    robot_state: _RobotState = field(default_factory=_RobotState)
    step_index: int = 0


def _live_config() -> FrankaLiveContractConfig:
    return FrankaLiveContractConfig(
        conditioning_mode="training_aligned_rgb_replan_v1",
        target_fps=10,
        frame_interval_tolerance_ms=15.0,
        max_state_image_skew_ms=50.0,
        max_history_frames=5,
    )


def _camera(
    role: str,
    timestamps: tuple[int, ...],
    *,
    offset: int,
) -> _Camera:
    frames = tuple(
        np.full((4, 6, 3), index + offset, dtype=np.uint8)
        for index in range(len(timestamps))
    )
    return _Camera(
        camera_role=role,
        timestamp_ns=timestamps[-1],
        frames=frames,
        frame_history_timestamps_ns=timestamps[:-1],
    )


def _decode(camera: object) -> tuple[np.ndarray, ...]:
    assert isinstance(camera, _Camera)
    return camera.frames


def test_live_adapter_builds_dual_camera_training_aligned_history() -> None:
    adapter = FrankaLiveObservationAdapter(_live_config())
    base = 1_000_000_000
    high_timestamps = tuple(base + index * 100_000_000 for index in range(5))
    wrist_timestamps = tuple(value + 5_000_000 for value in high_timestamps)
    packet = _ObservationPacket(
        observation_timestamp_ns=high_timestamps[-1] + 10_000_000,
        camera_observations=(
            _camera("global", high_timestamps, offset=10),
            _camera("wrist", wrist_timestamps, offset=20),
        ),
    )

    result = adapter.augment(
        packet,
        {"images": {}, "left_end_pose": np.zeros(7), "joint_qpos": np.zeros(8)},
        decode_camera_frames=_decode,
        role_to_model_key={"global": "cam_high", "wrist": "cam_left_wrist"},
    )

    history = result["training_aligned_video_history"]
    assert isinstance(history, tuple) and len(history) == 5
    assert result["training_aligned_video_timestamps_ns"] == high_timestamps
    assert result["image_timestamp_ns"] == high_timestamps[-1]
    assert result["state_timestamp_ns"] == packet.observation_timestamp_ns
    assert result["observation_sequence_id"] == 0
    assert result["observation_packet_step_index"] == 0
    assert result["observation_packet_timestamp_ns"] == (
        packet.observation_timestamp_ns
    )
    assert result["state_timestamp_provenance"] == (
        "observation_packet.observation_timestamp_ns"
    )
    assert result["camera_latest_timestamps_ns"] == {
        "cam_high": high_timestamps[-1],
        "cam_left_wrist": wrist_timestamps[-1],
    }
    assert result["history_coverage_start_ns"] == high_timestamps[0]
    assert result["history_coverage_end_ns"] == high_timestamps[-1]
    assert result["history_coverage_duration_ms"] == pytest.approx(400.0)
    assert result["active_arm_id"] == "franka"
    assert result["gripper_observation_unit"] == "width_m"
    assert result["joint_qpos"].shape == (8,)
    assert result["joint_qpos"][-1] == pytest.approx(0.04)
    assert np.array_equal(result["images"]["cam_high"], history[-1]["cam_high"])
    assert np.array_equal(
        result["images"]["cam_left_wrist"], history[-1]["cam_left_wrist"]
    )
    assert result["training_aligned_video_source_timestamps_ns"] == {
        "cam_high": high_timestamps,
        "cam_left_wrist": wrist_timestamps,
    }
    validated = validate_live_observation(
        result,
        config=_live_config(),
        history=history,
        current_state=np.zeros(8, dtype=np.float32),
        previous_sequence_id=None,
        previous_image_timestamp_ns=None,
    )
    assert validated.report["status"] == "verified"


def test_live_adapter_uses_one_frame_only_during_causal_warmup() -> None:
    adapter = FrankaLiveObservationAdapter(_live_config())
    timestamp = 2_000_000_000
    packet = _ObservationPacket(
        observation_timestamp_ns=timestamp + 1_000_000,
        camera_observations=(
            _camera("global", (timestamp,), offset=10),
            _camera("wrist", (timestamp,), offset=20),
        ),
    )

    result = adapter.augment(
        packet,
        {"images": {}},
        decode_camera_frames=_decode,
        role_to_model_key={"global": "cam_high", "wrist": "cam_left_wrist"},
    )

    assert len(result["training_aligned_video_history"]) == 1
    assert result["training_aligned_video_timestamps_ns"] == (timestamp,)


def test_live_adapter_grows_history_on_four_frame_causal_boundaries() -> None:
    config = FrankaLiveContractConfig(
        conditioning_mode="training_aligned_rgb_replan_v1",
        target_fps=10,
        frame_interval_tolerance_ms=15.0,
        max_state_image_skew_ms=50.0,
        max_history_frames=4097,
    )
    adapter = FrankaLiveObservationAdapter(config)
    base = 2_500_000_000
    timestamps = tuple(base + index * 100_000_000 for index in range(9))
    packet = _ObservationPacket(
        observation_timestamp_ns=timestamps[-1] + 1_000_000,
        camera_observations=(
            _camera("global", timestamps, offset=10),
            _camera("wrist", timestamps, offset=20),
        ),
    )

    result = adapter.augment(
        packet,
        {"images": {}},
        decode_camera_frames=_decode,
        role_to_model_key={"global": "cam_high", "wrist": "cam_left_wrist"},
    )

    assert len(result["training_aligned_video_history"]) == 9
    assert result["training_aligned_video_timestamps_ns"] == timestamps


def test_live_action_converts_width_m_to_open_ratio_without_mutation() -> None:
    model_actions = np.asarray(((0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.04),))
    output = {
        "actions": model_actions,
        "policy_metadata": {"action_format": "end_pose_base"},
    }

    adapted = _physical_width_output_to_open_ratio(
        output,
        gripper_max_width_m=0.08,
    )

    assert model_actions[0, 7] == pytest.approx(0.04)
    assert adapted["actions"][0, 7] == pytest.approx(0.5)
    assert adapted["policy_metadata"]["model_gripper_unit"] == "width_m"
    assert adapted["policy_metadata"]["wire_gripper_unit"] == "open_ratio"


def test_live_adapter_calibrates_open_ratio_when_width_is_absent() -> None:
    adapter = FrankaLiveObservationAdapter(_live_config())
    timestamp = 2_900_000_000
    packet = _ObservationPacket(
        observation_timestamp_ns=timestamp + 1_000_000,
        camera_observations=(
            _camera("global", (timestamp,), offset=10),
            _camera("wrist", (timestamp,), offset=20),
        ),
        robot_state=_RobotState(
            arms=(_ArmState(gripper=_GripperState(width_m=None, open_ratio=0.8)),)
        ),
    )

    result = adapter.augment(
        packet,
        {"images": {}},
        decode_camera_frames=_decode,
        role_to_model_key={"global": "cam_high", "wrist": "cam_left_wrist"},
    )

    assert result["joint_qpos"][-1] == pytest.approx(0.064)
    assert result["gripper_observation_source"] == "open_ratio_calibrated"


def test_live_adapter_rejects_missing_10hz_history_after_coverage() -> None:
    adapter = FrankaLiveObservationAdapter(_live_config())
    first = 3_000_000_000
    packet = _ObservationPacket(
        observation_timestamp_ns=first + 1_000_000,
        camera_observations=(
            _camera("global", (first,), offset=10),
            _camera("wrist", (first,), offset=20),
        ),
    )
    adapter.augment(
        packet,
        {"images": {}},
        decode_camera_frames=_decode,
        role_to_model_key={"global": "cam_high", "wrist": "cam_left_wrist"},
    )
    second = first + 500_000_000
    packet = _ObservationPacket(
        observation_timestamp_ns=second + 1_000_000,
        camera_observations=(
            _camera("global", (second,), offset=30),
            _camera("wrist", (second,), offset=40),
        ),
    )

    with pytest.raises(ValueError, match="required 10 Hz causal grid"):
        adapter.augment(
            packet,
            {"images": {}},
            decode_camera_frames=_decode,
            role_to_model_key={"global": "cam_high", "wrist": "cam_left_wrist"},
        )


def test_live_adapter_and_policy_reset_together() -> None:
    class _Policy:
        def __init__(self) -> None:
            self.reset_calls: list[dict[str, object] | None] = []

        def reset(self, reset_info: dict[str, object] | None = None) -> None:
            self.reset_calls.append(reset_info)

        def infer(self, new_obs: dict[str, object]) -> dict[str, object]:
            return new_obs

    core = _Policy()
    adapter = FrankaLiveObservationAdapter(_live_config())
    wrapped = LiveAdaptedPolicy(core, adapter)

    wrapped.reset({"episode_id": "next"})

    assert core.reset_calls == [{"episode_id": "next"}]
