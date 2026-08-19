# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict bridges from coherent live snapshots to existing Policy wires."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .contracts import CoherentSnapshot, SensorPacket, require_text

ImageArray: TypeAlias = NDArray[np.uint8]
FloatArray: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True)
class AgileXObservationSpec:
    """Stream IDs used to build one official AgileX observation."""

    top_rgb: str = "rgb.top"
    left_wrist_rgb: str = "rgb.wrist_l"
    right_wrist_rgb: str = "rgb.wrist_r"
    qpos14: str = "qpos14"
    left_tactile: str = "tactile.left"
    right_tactile: str = "tactile.right"
    left_wrench: str = "wrench.left"
    right_wrench: str = "wrench.right"

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        for value in values:
            require_text("AgileX stream ID", value)
        if len(set(values)) != len(values):
            raise ValueError("AgileX stream IDs must be unique")


@dataclass(frozen=True)
class FrankaObservationSpec:
    """Stream IDs used to build one existing Franka Policy observation."""

    top_rgb: str = "rgb.top"
    left_wrist_rgb: str = "rgb.wrist_l"
    pose7: str = "pose7"
    qpos8: str = "qpos8"

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        for value in values:
            require_text("Franka stream ID", value)
        if len(set(values)) != len(values):
            raise ValueError("Franka stream IDs must be unique")


def _packet(
    snapshot: CoherentSnapshot,
    stream_id: str,
    *,
    modality: str,
) -> SensorPacket:
    value = snapshot.packets.get(stream_id)
    if value is None:
        raise ValueError(
            f"snapshot is missing required {modality} stream {stream_id!r}"
        )
    if value.modality != modality:
        raise ValueError(
            f"stream {stream_id!r} modality must be {modality!r}, "
            f"got {value.modality!r}"
        )
    return value


def _image(packet: SensorPacket, *, label: str) -> ImageArray:
    value = np.asarray(packet.payload)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{label} must be uint8 HWC RGB")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{label} must have non-empty spatial dimensions")
    return cast(ImageArray, np.ascontiguousarray(value))


def _float_vector(
    packet: SensorPacket,
    *,
    label: str,
    size: int,
) -> FloatArray:
    value = np.asarray(packet.payload)
    if value.dtype != np.float32 or value.shape != (size,):
        raise ValueError(f"{label} must be float32[{size}]")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} must be finite")
    return cast(FloatArray, np.ascontiguousarray(value))


def _positive_seconds(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return result


def _agilex_rgb(
    snapshot: CoherentSnapshot,
    spec: AgileXObservationSpec,
) -> dict[str, ImageArray]:
    images = {
        "cam_high": _image(
            _packet(snapshot, spec.top_rgb, modality="rgb"),
            label="images.cam_high",
        ),
        "cam_wrist_left": _image(
            _packet(snapshot, spec.left_wrist_rgb, modality="rgb"),
            label="images.cam_wrist_left",
        ),
        "cam_wrist_right": _image(
            _packet(snapshot, spec.right_wrist_rgb, modality="rgb"),
            label="images.cam_wrist_right",
        ),
    }
    spatial_shapes = {image.shape[:2] for image in images.values()}
    if len(spatial_shapes) != 1:
        raise ValueError("AgileX RGB streams must share one spatial shape")
    return images


def _agilex_contact(
    snapshot: CoherentSnapshot,
    spec: AgileXObservationSpec,
    *,
    tactile_required: bool,
    wrench_required: bool,
) -> dict[str, dict[str, NDArray[Any]]] | None:
    if not isinstance(tactile_required, bool) or not isinstance(wrench_required, bool):
        raise TypeError("contact-required flags must be bool")
    if wrench_required and not tactile_required:
        raise ValueError("wrench-required AgileX routes must also require tactile")
    if not tactile_required:
        return None

    contact: dict[str, dict[str, NDArray[Any]]] = {
        "left_gripper": {
            "rectify": _image(
                _packet(snapshot, spec.left_tactile, modality="tactile"),
                label="tactile.left_gripper.rectify",
            )
        },
        "right_gripper": {
            "rectify": _image(
                _packet(snapshot, spec.right_tactile, modality="tactile"),
                label="tactile.right_gripper.rectify",
            )
        },
    }
    if wrench_required:
        contact.update(
            {
                "left_wrist_force": {
                    "wrench_6d": _float_vector(
                        _packet(snapshot, spec.left_wrench, modality="wrench"),
                        label="tactile.left_wrist_force.wrench_6d",
                        size=6,
                    )
                },
                "right_wrist_force": {
                    "wrench_6d": _float_vector(
                        _packet(snapshot, spec.right_wrench, modality="wrench"),
                        label="tactile.right_wrist_force.wrench_6d",
                        size=6,
                    )
                },
            }
        )
    return contact


def build_agilex_live_observation(
    snapshot: CoherentSnapshot,
    *,
    spec: AgileXObservationSpec,
    task_id: str,
    prompt: str,
    tactile_profile: str,
    execution_dt_s: float,
    tactile_required: bool,
    wrench_required: bool,
) -> dict[str, object]:
    """Build the documented AgileX wire without timestamp fallback."""

    if not isinstance(snapshot, CoherentSnapshot):
        raise TypeError("snapshot must be a CoherentSnapshot")
    if not isinstance(spec, AgileXObservationSpec):
        raise TypeError("spec must be an AgileXObservationSpec")
    qpos_packet = _packet(snapshot, spec.qpos14, modality="qpos")
    qpos = _float_vector(qpos_packet, label="joint_qpos", size=14)
    observation: dict[str, object] = {
        "images": _agilex_rgb(snapshot, spec),
        "joint_qpos": qpos,
        "state": qpos,
        "left_arm_joint_state": qpos[:7],
        "right_arm_joint_state": qpos[7:],
        "task_id": require_text("task_id", task_id),
        "prompt": require_text("prompt", prompt),
        "tactile_profile": require_text("tactile_profile", tactile_profile),
        "timestamp": qpos_packet.clock.capture_wall_ns / 1_000_000_000.0,
        "execution_dt_s": _positive_seconds(
            execution_dt_s,
            label="execution_dt_s",
        ),
    }
    contact = _agilex_contact(
        snapshot,
        spec,
        tactile_required=tactile_required,
        wrench_required=wrench_required,
    )
    if contact is not None:
        observation["tactile"] = contact
    return observation


def build_franka_live_observation(
    snapshot: CoherentSnapshot,
    *,
    spec: FrankaObservationSpec,
    task_id: str,
    prompt: str,
) -> dict[str, object]:
    """Build the existing two-camera Franka Policy wire."""

    if not isinstance(snapshot, CoherentSnapshot):
        raise TypeError("snapshot must be a CoherentSnapshot")
    if not isinstance(spec, FrankaObservationSpec):
        raise TypeError("spec must be a FrankaObservationSpec")
    images = {
        "cam_high": _image(
            _packet(snapshot, spec.top_rgb, modality="rgb"),
            label="images.cam_high",
        ),
        "cam_left_wrist": _image(
            _packet(snapshot, spec.left_wrist_rgb, modality="rgb"),
            label="images.cam_left_wrist",
        ),
    }
    if len({image.shape[:2] for image in images.values()}) != 1:
        raise ValueError("Franka RGB streams must share one spatial shape")
    return {
        "images": images,
        "left_end_pose": _float_vector(
            _packet(snapshot, spec.pose7, modality="pose"),
            label="left_end_pose",
            size=7,
        ),
        "joint_qpos": _float_vector(
            _packet(snapshot, spec.qpos8, modality="qpos"),
            label="joint_qpos",
            size=8,
        ),
        "task_id": require_text("task_id", task_id),
        "prompt": require_text("prompt", prompt),
    }
