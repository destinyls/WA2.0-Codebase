# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Offline protocol replay for the official Franka Track 3.2 Policy."""

from __future__ import annotations

import hashlib
import io
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt

from .franka_actions import (
    DERIVED_ACTION_SCHEMA,
    FRANKA_ACTION_SCHEMA,
    FRANKA_QUATERNION_ORDER,
)
from .franka_manifest import canonical_sha256
from .franka_policy import FrankaActionBackend, FrankaSafetyConfig, Policy
from .franka_realtime import (
    OFFICIAL_FRANKA_CONTROL_HZ,
    require_franka_realtime_pass,
    summarize_franka_policy_latency,
)

REPLAY_SCHEMA_VERSION = 3
_OBSERVATION_KEYS = frozenset(
    ("cam_high", "cam_left_wrist", "left_end_pose", "joint_qpos")
)


@dataclass(frozen=True)
class _StableBytes:
    path: Path
    raw: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int


def _capture_stable_bytes(path: Path, *, label: str) -> _StableBytes:
    lexical = Path(path).expanduser()
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    if lexical.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    source = lexical.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    with source.open("rb") as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        after = os.fstat(handle.fileno())
    current = source.stat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or identity != (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ):
        raise ValueError(f"{label} changed while it was read")
    if len(raw) != before.st_size:
        raise ValueError(f"{label} byte count changed while it was read")
    return _StableBytes(
        path=source,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
    )


def _require_unchanged(snapshot: _StableBytes, *, label: str) -> None:
    current = _capture_stable_bytes(snapshot.path, label=label)
    if (
        current.sha256,
        current.device,
        current.inode,
        current.size,
        current.mtime_ns,
    ) != (
        snapshot.sha256,
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mtime_ns,
    ):
        raise ValueError(f"{label} changed during Policy replay")


@contextmanager
def _frozen_config(snapshot: _StableBytes) -> Iterator[Path]:
    temporary = snapshot.path.parent / (
        f".{snapshot.path.name}.n0-replay-{os.getpid()}-{os.urandom(8).hex()}.incomplete"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(snapshot.raw)
            handle.flush()
            os.fsync(handle.fileno())
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, payload: object) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    digest = hashlib.sha256(raw).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return digest


def _image(value: npt.ArrayLike, *, label: str) -> npt.NDArray[np.uint8]:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{label} must be uint8 HWC RGB")
    return np.ascontiguousarray(array)


def _vector(
    value: npt.ArrayLike,
    *,
    length: int,
    label: str,
) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite vector with length {length}")
    return np.ascontiguousarray(array)


def _load_observation(raw: bytes) -> dict[str, npt.NDArray[np.generic]]:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != _OBSERVATION_KEYS:
                raise ValueError(
                    "replay observation keys differ from the fixed Franka schema"
                )
            return {
                "cam_high": _image(archive["cam_high"], label="cam_high"),
                "cam_left_wrist": _image(
                    archive["cam_left_wrist"], label="cam_left_wrist"
                ),
                "left_end_pose": _vector(
                    archive["left_end_pose"], length=7, label="left_end_pose"
                ),
                "joint_qpos": _vector(
                    archive["joint_qpos"], length=8, label="joint_qpos"
                ),
            }
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "replay observation"
        ):
            raise
        raise ValueError("invalid replay observation NPZ") from error


def _quaternion_angle(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(2.0 * np.arccos(cosine))


def _require_safe_action(
    action: npt.NDArray[np.float32],
    *,
    previous: npt.NDArray[np.float32],
    safety: FrankaSafetyConfig,
) -> None:
    tolerance = 1e-5
    lower = np.asarray(safety.workspace_min, dtype=np.float32)
    upper = np.asarray(safety.workspace_max, dtype=np.float32)
    if np.any(action[:3] < lower - tolerance) or np.any(action[:3] > upper + tolerance):
        raise ValueError("Policy replay action is outside the signed workspace")
    if (
        float(np.linalg.norm(action[:3] - previous[:3]))
        > safety.max_translation_step_m + tolerance
    ):
        raise ValueError("Policy replay action exceeds the translation step limit")
    if (
        _quaternion_angle(action[3:7], previous[3:7])
        > safety.max_rotation_step_rad + tolerance
    ):
        raise ValueError("Policy replay action exceeds the rotation step limit")
    if (
        not safety.gripper_min - tolerance
        <= float(action[7])
        <= safety.gripper_max + tolerance
    ):
        raise ValueError("Policy replay action is outside the gripper limits")
    if abs(float(action[7] - previous[7])) > safety.max_gripper_step + tolerance:
        raise ValueError("Policy replay action exceeds the gripper step limit")


def run_franka_policy_replay(
    *,
    config_path: Path,
    observation_path: Path,
    prompt: str,
    steps: int,
    output: Path,
    backend: FrankaActionBackend | None = None,
    control_hz: float = OFFICIAL_FRANKA_CONTROL_HZ,
    require_realtime: bool = False,
    minimum_refill_samples: int = 2,
) -> dict[str, object]:
    """Run a perfect-proprioception replay and publish a fail-closed receipt.

    This is an engineering protocol/safety smoke. It is deliberately marked as
    not being an organizer or real-robot evaluation.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("replay prompt must be non-empty")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 7 <= steps <= 256:
        raise ValueError("replay steps must be an integer in [7,256]")
    config_snapshot = _capture_stable_bytes(config_path, label="replay config")
    observation_snapshot = _capture_stable_bytes(
        observation_path, label="replay observation"
    )
    destination_input = Path(output).expanduser()
    if destination_input.exists() or destination_input.is_symlink():
        raise FileExistsError(f"replay receipt already exists: {destination_input}")
    destination = destination_input.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"replay receipt already exists: {destination}")
    observation = _load_observation(observation_snapshot.raw)

    pose7 = np.asarray(observation["left_end_pose"], dtype=np.float32).copy()
    qpos8 = np.asarray(observation["joint_qpos"], dtype=np.float32).copy()
    previous_pose8 = np.concatenate((pose7, qpos8[-1:])).astype(np.float32)
    actions: list[list[float]] = []
    intervention_count = 0
    infer_times_ms: list[float] = []
    policy_timings: list[dict[str, object]] = []
    with _frozen_config(config_snapshot) as frozen_config:
        policy = Policy(str(frozen_config), backend=backend)
        policy.reset({"prompt": prompt})
        try:
            for _ in range(steps):
                result = policy.infer(
                    {
                        "prompt": prompt,
                        "task_id": "offline_protocol_replay",
                        "images": {
                            "cam_high": observation["cam_high"],
                            "cam_left_wrist": observation["cam_left_wrist"],
                        },
                        "left_end_pose": pose7,
                        "joint_qpos": qpos8,
                    }
                )
                action = np.asarray(result.get("actions"), dtype=np.float32)
                metadata = result.get("policy_metadata")
                timing = result.get("policy_timing")
                if (
                    action.shape != (1, 8)
                    or not np.isfinite(action).all()
                    or not isinstance(metadata, dict)
                    or metadata.get("action_format") != "end_pose_base"
                    or metadata.get("action_dim") != 8
                    or metadata.get("chunk_size") != 1
                    or metadata.get("quaternion_order") != FRANKA_QUATERNION_ORDER
                    or metadata.get("wire_action_schema") != FRANKA_ACTION_SCHEMA
                    or metadata.get("derived_action_schema") != DERIVED_ACTION_SCHEMA
                    or metadata.get("tactile_mode") != "disabled"
                    or type(metadata.get("safety_intervened")) is not bool
                    or type(metadata.get("safety_intervention_count")) is not int
                    or metadata.get("safety_intervention_count") not in {0, 1}
                    or metadata.get("safety_intervened")
                    != bool(metadata.get("safety_intervention_count"))
                    or not isinstance(timing, dict)
                ):
                    raise ValueError("Policy replay output contract mismatch")
                quaternion_norm = float(np.linalg.norm(action[0, 3:7]))
                if not np.isclose(quaternion_norm, 1.0, atol=1e-4):
                    raise ValueError("Policy replay returned a non-unit quaternion")
                _require_safe_action(
                    action[0], previous=previous_pose8, safety=policy.config.safety
                )
                infer_ms = timing.get("infer_ms")
                if not isinstance(infer_ms, (int, float)) or not np.isfinite(infer_ms):
                    raise ValueError("Policy replay timing must be finite")
                required_timing = {
                    "kind",
                    "infer_ms",
                    "input_ms",
                    "grounding_ms",
                    "generation_ms",
                    "postprocess_ms",
                    "grounded",
                    "generated",
                    "queue_depth_after",
                }
                if set(timing) != required_timing:
                    raise ValueError("Policy replay timing fields differ from schema")
                if (
                    type(timing.get("grounded")) is not bool
                    or type(timing.get("generated")) is not bool
                    or type(timing.get("queue_depth_after")) is not int
                    or int(timing["queue_depth_after"]) < 0
                ):
                    raise ValueError("Policy replay timing metadata is invalid")
                actions.append([float(value) for value in action[0]])
                intervention_count += int(metadata["safety_intervention_count"])
                infer_times_ms.append(float(infer_ms))
                policy_timings.append(dict(timing))
                pose7 = action[0, :7].copy()
                qpos8[-1] = action[0, 7]
                previous_pose8 = action[0].copy()
        finally:
            policy.reset({"prompt": prompt})

    _require_unchanged(config_snapshot, label="replay config")
    _require_unchanged(observation_snapshot, label="replay observation")
    realtime_assessment = summarize_franka_policy_latency(
        policy_timings,
        control_hz=control_hz,
        minimum_refill_samples=minimum_refill_samples,
    )
    if require_realtime:
        require_franka_realtime_pass(realtime_assessment)
    core: dict[str, object] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "complete",
        "execution_tier": "offline_engineering_smoke",
        "organizer_evaluation_completed": False,
        "real_robot_evaluation_completed": False,
        "config": str(config_snapshot.path),
        "config_sha256": config_snapshot.sha256,
        "observation": str(observation_snapshot.path),
        "observation_sha256": observation_snapshot.sha256,
        "prompt": prompt,
        "steps": steps,
        "action_format": "end_pose_base",
        "action_dim": 8,
        "quaternion_order": FRANKA_QUATERNION_ORDER,
        "wire_action_schema": FRANKA_ACTION_SCHEMA,
        "derived_action_schema": DERIVED_ACTION_SCHEMA,
        "tactile_mode": "disabled",
        "safety_intervention_count": intervention_count,
        "mean_infer_ms": float(np.mean(infer_times_ms)),
        "max_infer_ms": float(np.max(infer_times_ms)),
        "policy_timings": policy_timings,
        "realtime_assessment": realtime_assessment,
        "actions": actions,
    }
    receipt = {**core, "replay_identity_sha256": canonical_sha256(core)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_file_sha256 = _write_new_json(destination, receipt)
    return {
        **receipt,
        "receipt": str(destination.resolve(strict=True)),
        "receipt_file_sha256": receipt_file_sha256,
    }


__all__ = ("REPLAY_SCHEMA_VERSION", "run_franka_policy_replay")
