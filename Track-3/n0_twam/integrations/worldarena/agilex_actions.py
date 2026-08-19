# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""AgileX action-label provenance at the WorldArena data boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

QPOS14_ACTION_SCHEMA = "qpos14_joint_absolute_v1"
MEASURED_NEXT_QPOS_SCHEMA = "measured_next_qpos_v1"
OFFICIAL_ACTION_LABEL_SOURCES = frozenset(("commanded", "executed"))
ACTION_DIM = 14
ACTIVE_ACTION_CHANNEL_IDS = tuple(range(ACTION_DIM))
GRIPPER_CHANNEL_IDS = (6, 13)

FloatArray: TypeAlias = npt.NDArray[np.float32]
BoolArray: TypeAlias = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class AgileXActionLabelContract:
    """Immutable distinction between official labels and engineering proxies."""

    schema: str
    label_source: str
    formal: bool
    label_offset: int

    def __post_init__(self) -> None:
        if self.formal:
            if (
                self.schema != QPOS14_ACTION_SCHEMA
                or self.label_source not in OFFICIAL_ACTION_LABEL_SOURCES
                or self.label_offset != 0
            ):
                raise ValueError(
                    "formal AgileX labels must be same-row commanded/executed "
                    "qpos14 targets"
                )
        elif (
            self.schema != MEASURED_NEXT_QPOS_SCHEMA
            or self.label_source != "measured_next"
            or self.label_offset != 1
        ):
            raise ValueError("engineering AgileX labels must use measured_next_qpos_v1")


ENGINEERING_MEASURED_NEXT_CONTRACT = AgileXActionLabelContract(
    schema=MEASURED_NEXT_QPOS_SCHEMA,
    label_source="measured_next",
    formal=False,
    label_offset=1,
)


@dataclass(frozen=True)
class MeasuredNextQposLabels:
    actions: FloatArray
    valid_mask: BoolArray
    contract: AgileXActionLabelContract = ENGINEERING_MEASURED_NEXT_CONTRACT


def validate_qpos14(values: npt.ArrayLike, *, label: str) -> FloatArray:
    """Return a contiguous finite qpos14 array without guessing semantics."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != ACTION_DIM:
        got = None if array.ndim == 0 else array.shape[-1]
        raise ValueError(f"{label} last dimension must be 14, got {got}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array)


def official_action_contract(label_source: str) -> AgileXActionLabelContract:
    """Build a formal same-row command/execution label contract."""

    return AgileXActionLabelContract(
        schema=QPOS14_ACTION_SCHEMA,
        label_source=str(label_source),
        formal=True,
        label_offset=0,
    )


def derive_measured_next_qpos(
    joint_qpos: npt.ArrayLike,
    *,
    contract: AgileXActionLabelContract,
) -> MeasuredNextQposLabels:
    """Build a masked engineering proxy; never usable as a formal label."""

    if contract != ENGINEERING_MEASURED_NEXT_CONTRACT:
        raise ValueError(
            "measured qpos[t+1] requires the explicit engineering contract"
        )
    qpos = validate_qpos14(joint_qpos, label="AgileX measured joint_qpos")
    if qpos.ndim != 2 or qpos.shape[0] < 2:
        raise ValueError("engineering qpos trajectory must have shape [T>=2, 14]")
    actions = np.empty_like(qpos)
    actions[:-1] = qpos[1:]
    actions[-1] = qpos[-1]
    valid = np.ones(qpos.shape[0], dtype=np.bool_)
    valid[-1] = False
    actions.setflags(write=False)
    valid.setflags(write=False)
    return MeasuredNextQposLabels(actions=actions, valid_mask=valid)


__all__ = (
    "ACTION_DIM",
    "ACTIVE_ACTION_CHANNEL_IDS",
    "AgileXActionLabelContract",
    "ENGINEERING_MEASURED_NEXT_CONTRACT",
    "GRIPPER_CHANNEL_IDS",
    "MEASURED_NEXT_QPOS_SCHEMA",
    "MeasuredNextQposLabels",
    "OFFICIAL_ACTION_LABEL_SOURCES",
    "QPOS14_ACTION_SCHEMA",
    "derive_measured_next_qpos",
    "official_action_contract",
    "validate_qpos14",
)
