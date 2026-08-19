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
    _probe_loaded_bridge,
    _validated_hub_url,
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
