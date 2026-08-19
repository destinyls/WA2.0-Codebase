# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Audited canonical AgileX episodes independent of the raw storage backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from .agilex_actions import (
    ENGINEERING_MEASURED_NEXT_CONTRACT,
    AgileXActionLabelContract,
    derive_measured_next_qpos,
    validate_qpos14,
)
from .agilex_manifest import AgileXRepoRoute

FloatArray: TypeAlias = npt.NDArray[np.float32]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
DType = TypeVar("DType", bound=np.generic)


def _readonly(array: npt.NDArray[DType]) -> npt.NDArray[DType]:
    output: npt.NDArray[DType] = np.ascontiguousarray(array).copy()
    output.setflags(write=False)
    return output


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError("AgileX source relative path is unsafe")
    return path.as_posix()


def _validate_timestamps(
    values: npt.ArrayLike, *, length: int
) -> npt.NDArray[np.float64]:
    timestamps = np.asarray(values, dtype=np.float64)
    if timestamps.shape != (length,) or not np.isfinite(timestamps).all():
        raise ValueError("AgileX timestamps must be a finite [T] vector")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("AgileX timestamps must be strictly increasing")
    return _readonly(timestamps)


def _validate_images(
    values: Mapping[str, npt.ArrayLike],
    *,
    expected_keys: tuple[str, ...],
    length: int,
    label: str,
) -> Mapping[str, npt.NDArray[np.uint8]]:
    if set(values) != set(expected_keys):
        raise ValueError(
            f"AgileX {label} roster mismatch: "
            f"expected={sorted(expected_keys)}, actual={sorted(values)}"
        )
    output: dict[str, npt.NDArray[np.uint8]] = {}
    for key in expected_keys:
        image = np.asarray(values[key])
        if (
            image.dtype != np.uint8
            or image.ndim != 4
            or image.shape[0] != length
            or image.shape[-1] != 3
            or image.shape[1] <= 0
            or image.shape[2] <= 0
        ):
            raise ValueError(
                f"AgileX {label} stream has invalid uint8 THWC shape: {key}"
            )
        output[key] = _readonly(image)
    return MappingProxyType(output)


def _validate_wrench(
    values: Mapping[str, npt.ArrayLike],
    *,
    expected_keys: tuple[str, ...],
    length: int,
) -> Mapping[str, FloatArray]:
    if set(values) != set(expected_keys):
        raise ValueError(
            "AgileX wrench roster mismatch: "
            f"expected={sorted(expected_keys)}, actual={sorted(values)}"
        )
    output: dict[str, FloatArray] = {}
    for key in expected_keys:
        wrench = np.asarray(values[key], dtype=np.float32)
        if wrench.shape != (length, 6) or not np.isfinite(wrench).all():
            raise ValueError(f"AgileX wrench must be finite [T,6]: {key}")
        output[key] = _readonly(wrench)
    return MappingProxyType(output)


@dataclass(frozen=True)
class AgileXEpisode:
    """One post-audit episode with immutable, row-aligned arrays."""

    route: AgileXRepoRoute
    source_relative_path: str
    episode_id: int
    task: str
    timestamps: npt.NDArray[np.float64]
    joint_qpos: FloatArray
    actions: FloatArray
    action_valid_mask: BoolArray
    rgb: Mapping[str, npt.NDArray[np.uint8]]
    tactile: Mapping[str, npt.NDArray[np.uint8]]
    wrench: Mapping[str, FloatArray]

    @property
    def length(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def formal(self) -> bool:
        return self.route.formal

    @property
    def action_label_source(self) -> str:
        return self.route.action_label_source


def _validate_valid_mask(values: npt.ArrayLike, *, length: int) -> BoolArray:
    raw = np.asarray(values)
    if raw.dtype != np.bool_ or raw.shape != (length,):
        raise ValueError("AgileX action_valid_mask must be boolean [T]")
    if not raw.any():
        raise ValueError("AgileX episode must contain at least one valid action")
    valid_count = int(raw.sum())
    expected = np.arange(length) < valid_count
    if not np.array_equal(raw, expected):
        raise ValueError("AgileX invalid actions must be a terminal suffix")
    return _readonly(raw)


def audit_episode(
    *,
    route: AgileXRepoRoute,
    source_relative_path: str,
    episode_id: int,
    task: str,
    timestamps: npt.ArrayLike,
    joint_qpos: npt.ArrayLike,
    actions: npt.ArrayLike | None,
    action_valid_mask: npt.ArrayLike | None,
    rgb: Mapping[str, npt.ArrayLike],
    tactile: Mapping[str, npt.ArrayLike],
    wrench: Mapping[str, npt.ArrayLike],
    engineering_label_contract: AgileXActionLabelContract | None = None,
) -> AgileXEpisode:
    """Audit a storage-reader output without ever inferring formal actions."""

    if (
        isinstance(episode_id, bool)
        or not isinstance(episode_id, int)
        or episode_id < 0
    ):
        raise ValueError("AgileX episode_id must be a non-negative integer")
    if not isinstance(task, str) or not task:
        raise ValueError("AgileX task must be a non-empty string")
    relative_path = _safe_relative_path(source_relative_path)
    qpos = validate_qpos14(joint_qpos, label="AgileX joint_qpos")
    if qpos.ndim != 2 or qpos.shape[0] < 2:
        raise ValueError("AgileX joint_qpos must have shape [T>=2,14]")
    length = int(qpos.shape[0])
    audited_timestamps = _validate_timestamps(timestamps, length=length)

    if route.formal:
        if actions is None:
            raise ValueError(
                "formal AgileX episode requires official commanded/executed actions"
            )
        if engineering_label_contract is not None:
            raise ValueError("formal AgileX episode cannot use an engineering contract")
        if action_valid_mask is None:
            raise ValueError(
                "formal AgileX episode requires explicit action_valid_mask"
            )
        audited_actions = validate_qpos14(actions, label="AgileX official action")
        audited_mask = _validate_valid_mask(action_valid_mask, length=length)
    else:
        if engineering_label_contract != ENGINEERING_MEASURED_NEXT_CONTRACT:
            raise ValueError(
                "measured qpos[t+1] requires an explicit engineering contract"
            )
        if actions is not None or action_valid_mask is not None:
            raise ValueError(
                "engineering measured-next labels must be derived by the audited source"
            )
        labels = derive_measured_next_qpos(
            qpos,
            contract=engineering_label_contract,
        )
        audited_actions = labels.actions
        audited_mask = labels.valid_mask
    if audited_actions.shape != (length, 14):
        raise ValueError("AgileX actions must have shape [T,14]")

    return AgileXEpisode(
        route=route,
        source_relative_path=relative_path,
        episode_id=episode_id,
        task=task,
        timestamps=audited_timestamps,
        joint_qpos=_readonly(qpos),
        actions=_readonly(audited_actions),
        action_valid_mask=_readonly(audited_mask),
        rgb=_validate_images(
            rgb,
            expected_keys=route.rgb_keys,
            length=length,
            label="RGB",
        ),
        tactile=_validate_images(
            tactile,
            expected_keys=route.tactile_keys,
            length=length,
            label="tactile",
        ),
        wrench=_validate_wrench(
            wrench,
            expected_keys=route.wrench_keys,
            length=length,
        ),
    )


__all__ = (
    "AgileXEpisode",
    "audit_episode",
)
