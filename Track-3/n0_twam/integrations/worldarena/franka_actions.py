# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Franka end-pose geometry at the WorldArena/N0-TWAM boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

FRANKA_ACTION_SCHEMA = "franka_end_pose_base_xyzw8_v2"
DERIVED_ACTION_SCHEMA = "franka_ee10_rot6d_columns_from_xyzw_v2"
MODEL_ACTION_SCHEMA = "ee20_absee"
ACTIVE_ACTION_CHANNEL_IDS = tuple(range(10))
FRANKA_QUATERNION_ORDER = "xyzw"
TRACK32_PROFILE_ID = "franka_track32_vision_only_xyzw_v2"
FRANKA_ACTION_ROUTE_ID = "franka_pose8_xyzw_to_ee20_v2"
FRANKA_EMBODIMENT_PROFILE_ID = "franka_pose8_xyzw_ee20_v2"


@dataclass(frozen=True)
class FrankaActionContract:
    """Immutable action-representation contract used by receipts."""

    source_schema: str = FRANKA_ACTION_SCHEMA
    derived_schema: str = DERIVED_ACTION_SCHEMA
    model_schema: str = MODEL_ACTION_SCHEMA
    source_frame: str = "robot_base"
    quaternion_order: str = FRANKA_QUATERNION_ORDER
    rot6d_layout: str = "column0_then_column1"
    active_model_channels: tuple[int, ...] = ACTIVE_ACTION_CHANNEL_IDS


ACTION_CONTRACT = FrankaActionContract()


def _finite_array(
    values: npt.ArrayLike,
    *,
    last_dim: int,
    label: str,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != last_dim:
        got = None if array.ndim == 0 else array.shape[-1]
        raise ValueError(f"{label} last dimension must be {last_dim}, got {got}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def normalize_quaternion_xyzw(values: npt.ArrayLike) -> FloatArray:
    """Normalize scalar-last quaternions and reject degenerate rotations."""

    quaternion = _finite_array(
        values,
        last_dim=4,
        label="xyzw quaternion",
    )
    norm = np.linalg.norm(quaternion.astype(np.float64), axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("xyzw quaternion norm is too small")
    return (quaternion / norm).astype(np.float32, copy=False)


def quaternion_xyzw_to_matrix(values: npt.ArrayLike) -> FloatArray:
    """Convert scalar-last unit quaternions to rotation matrices."""

    quaternion = normalize_quaternion_xyzw(values).astype(np.float64)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    matrix = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix.astype(np.float32)


def _matrix_to_quaternion_wxyz_one(matrix: npt.NDArray[np.float64]) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        values = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ),
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            values = np.asarray(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                )
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            values = np.asarray(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                )
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            values = np.asarray(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    return values / np.linalg.norm(values)


def matrix_to_quaternion_xyzw(values: npt.ArrayLike) -> FloatArray:
    """Convert proper 3x3 rotation matrices to scalar-last quaternions."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end with shape (3, 3)")
    if not np.isfinite(matrix).all():
        raise ValueError("rotation matrix contains non-finite values")
    flat = matrix.reshape(-1, 3, 3)
    wxyz = np.stack([_matrix_to_quaternion_wxyz_one(item) for item in flat])
    xyzw = wxyz[:, (1, 2, 3, 0)]
    return xyzw.reshape(matrix.shape[:-2] + (4,)).astype(np.float32)


def end_pose8_to_ee10(values: npt.ArrayLike) -> FloatArray:
    """Map verified ``[xyz,xyzw,gripper]`` to N0's one-arm EE10 layout."""

    pose = _finite_array(values, last_dim=8, label="Franka end_pose_base")
    matrix = quaternion_xyzw_to_matrix(pose[..., 3:7])
    rot6d = np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)
    return np.concatenate((pose[..., :3], rot6d, pose[..., 7:8]), axis=-1).astype(
        np.float32,
        copy=False,
    )


def rot6d_columns_to_matrix(values: npt.ArrayLike) -> FloatArray:
    """Re-orthonormalize N0's two rotation-matrix columns."""

    rot6d = _finite_array(values, last_dim=6, label="rot6d")
    first = rot6d[..., :3].astype(np.float64)
    second = rot6d[..., 3:6].astype(np.float64)
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("rot6d first column is degenerate")
    basis1 = first / first_norm
    second = second - np.sum(basis1 * second, axis=-1, keepdims=True) * basis1
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("rot6d columns are collinear")
    basis2 = second / second_norm
    basis3 = np.cross(basis1, basis2)
    return np.stack((basis1, basis2, basis3), axis=-1).astype(np.float32)


def ee10_to_end_pose8(
    values: npt.ArrayLike,
    *,
    quaternion_reference: npt.ArrayLike | None = None,
) -> FloatArray:
    """Decode one-arm EE10 and optionally preserve quaternion sign continuity."""

    action = _finite_array(values, last_dim=10, label="Franka EE10")
    quaternion = matrix_to_quaternion_xyzw(rot6d_columns_to_matrix(action[..., 3:9]))
    if quaternion_reference is not None:
        reference = normalize_quaternion_xyzw(quaternion_reference)
        try:
            reference = np.broadcast_to(reference, quaternion.shape)
        except ValueError as exc:
            raise ValueError(
                "quaternion reference is not broadcast-compatible"
            ) from exc
        flip = np.sum(quaternion * reference, axis=-1, keepdims=True) < 0.0
        quaternion = np.where(flip, -quaternion, quaternion)
    return np.concatenate(
        (action[..., :3], quaternion, action[..., 9:10]), axis=-1
    ).astype(np.float32, copy=False)


def slerp_quaternion_xyzw(
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    fraction: npt.ArrayLike,
) -> FloatArray:
    """Shortest-arc scalar-last quaternion interpolation."""

    first = normalize_quaternion_xyzw(start).astype(np.float64)
    second = normalize_quaternion_xyzw(end).astype(np.float64)
    if first.shape != second.shape:
        raise ValueError("SLERP quaternion shapes must match")
    alpha = np.asarray(fraction, dtype=np.float64)
    target_shape = first.shape[:-1]
    try:
        alpha = np.broadcast_to(alpha, target_shape)[..., None]
    except ValueError as exc:
        raise ValueError("SLERP fraction is not broadcast-compatible") from exc
    if not np.isfinite(alpha).all() or np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("SLERP fraction must be finite and within [0, 1]")
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0.0, -second, second)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    close = sine < 1e-7
    safe_sine = np.where(close, 1.0, sine)
    interpolated = (
        np.sin((1.0 - alpha) * angle) / safe_sine * first
        + np.sin(alpha * angle) / safe_sine * second
    )
    linear = (1.0 - alpha) * first + alpha * second
    output = np.where(close, linear, interpolated)
    return normalize_quaternion_xyzw(output)


def interpolate_end_pose8(
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    fraction: npt.ArrayLike,
) -> FloatArray:
    """Interpolate translation/gripper linearly and orientation by SLERP."""

    first = _finite_array(start, last_dim=8, label="Franka start pose")
    second = _finite_array(end, last_dim=8, label="Franka end pose")
    if first.shape != second.shape:
        raise ValueError("Franka interpolation pose shapes must match")
    alpha = np.asarray(fraction, dtype=np.float32)
    try:
        alpha = np.broadcast_to(alpha, first.shape[:-1])[..., None]
    except ValueError as exc:
        raise ValueError("Franka interpolation fraction is invalid") from exc
    if not np.isfinite(alpha).all() or np.any(alpha < 0.0) or np.any(alpha > 1.0):
        raise ValueError("Franka interpolation fraction must be within [0, 1]")
    linear = first + alpha * (second - first)
    quaternion = slerp_quaternion_xyzw(first[..., 3:7], second[..., 3:7], alpha[..., 0])
    return np.concatenate(
        (linear[..., :3], quaternion, linear[..., 7:8]), axis=-1
    ).astype(np.float32)


def embed_ee10_in_ee20(values: npt.ArrayLike) -> FloatArray:
    """Embed one-arm EE10 into the released dual-arm 20D action layout."""

    action = _finite_array(values, last_dim=10, label="Franka EE10")
    output = np.zeros(action.shape[:-1] + (20,), dtype=np.float32)
    output[..., :10] = action
    return output


def extract_ee10_from_ee20(values: npt.ArrayLike) -> FloatArray:
    """Read only the active Franka half of a released 20D action."""

    action = _finite_array(values, last_dim=20, label="N0 EE20")
    return action[..., :10].copy()


__all__ = (
    "ACTION_CONTRACT",
    "ACTIVE_ACTION_CHANNEL_IDS",
    "DERIVED_ACTION_SCHEMA",
    "FRANKA_ACTION_SCHEMA",
    "FRANKA_ACTION_ROUTE_ID",
    "FRANKA_EMBODIMENT_PROFILE_ID",
    "FRANKA_QUATERNION_ORDER",
    "MODEL_ACTION_SCHEMA",
    "TRACK32_PROFILE_ID",
    "FrankaActionContract",
    "ee10_to_end_pose8",
    "embed_ee10_in_ee20",
    "end_pose8_to_ee10",
    "extract_ee10_from_ee20",
    "interpolate_end_pose8",
    "matrix_to_quaternion_xyzw",
    "normalize_quaternion_xyzw",
    "quaternion_xyzw_to_matrix",
    "rot6d_columns_to_matrix",
    "slerp_quaternion_xyzw",
)
