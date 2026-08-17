# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Executed-action history for synchronous AgileX KV grounding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from n0_twam.embodiments import AGILEX_RGB_KEYS
from n0_twam.tactile_profiles import VISION_ONLY

from .agilex_policy_contracts import (
    RGB_KEYS,
    FloatArray,
    ImageArray,
    AgileXSafetyContract,
    AgileXTaskRoute,
)

ImageRow: TypeAlias = dict[str, ImageArray]
WrenchRow: TypeAlias = dict[str, FloatArray]
AGILEX_CHUNK_ACTIONS = 12
AGILEX_SENSOR_KEYFRAMES = 4


def mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, object], value)


def _image(value: object, *, label: str) -> ImageArray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{label} must be uint8 HWC RGB")
    return np.ascontiguousarray(array)


def rgb_images(value: object) -> dict[str, ImageArray]:
    payload = mapping(value, label="images")
    if set(payload) != set(RGB_KEYS):
        raise ValueError(f"images must contain exactly {list(RGB_KEYS)}")
    images = {key: _image(payload[key], label=f"images.{key}") for key in RGB_KEYS}
    if len({image.shape[:2] for image in images.values()}) != 1:
        raise ValueError("all AgileX RGB images must share one HxW shape")
    return images


def _validated_keys(
    payload: Mapping[str, object], *, keys: tuple[str, ...] | None, label: str
) -> tuple[str, ...]:
    if keys is None:
        if any(not isinstance(key, str) or not key for key in payload):
            raise ValueError(f"{label} keys must be non-empty strings")
        return tuple(sorted(payload))
    if set(payload) != set(keys):
        raise ValueError(f"{label} keys differ from the activated task route")
    return keys


def tactile_images(
    value: object | None, *, keys: tuple[str, ...] | None
) -> dict[str, ImageArray] | None:
    if value is None:
        return None
    payload = mapping(value, label="tactile")
    ordered = _validated_keys(payload, keys=keys, label="tactile")
    return {key: _image(payload[key], label=f"tactile.{key}") for key in ordered}


def wrench_values(
    value: object | None, *, keys: tuple[str, ...] | None
) -> dict[str, FloatArray] | None:
    if value is None:
        return None
    payload = mapping(value, label="wrench")
    ordered = _validated_keys(payload, keys=keys, label="wrench")
    result: dict[str, FloatArray] = {}
    for key in ordered:
        array = np.asarray(payload[key])
        if (
            array.dtype != np.float32
            or array.shape != (6,)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"wrench.{key} must be finite float32[6]")
        result[key] = np.ascontiguousarray(array)
    return result


def _contact_history(
    history: tuple[Mapping[str, npt.NDArray[np.generic]], ...] | None,
    *,
    required: bool,
    profile: str,
    label: str,
) -> None:
    if required and history is None:
        raise ValueError(f"active task route requires {label} grounding history")
    if not required and history is not None:
        if profile == VISION_ONLY:
            raise ValueError("vision_only grounding does not accept contact inputs")
        raise ValueError(f"active task route forbids {label} grounding history")
    if history is not None and len(history) != AGILEX_SENSOR_KEYFRAMES:
        raise ValueError(f"{label} history must contain exactly four sensor keyframes")


def build_grounding_observation(
    *,
    actions_qpos14_cfh: FloatArray,
    image_history: tuple[Mapping[str, ImageArray], ...],
    tactile_history: tuple[Mapping[str, ImageArray], ...] | None,
    wrench_history: tuple[Mapping[str, FloatArray], ...] | None,
    action_anchor_qpos14: FloatArray,
    route: AgileXTaskRoute,
    profile: str,
    prompt: str,
) -> dict[str, object]:
    """Validate and build one real ``[14,1,12]`` KV-cache update."""

    actions = np.asarray(actions_qpos14_cfh)
    if (
        actions.dtype != np.float32
        or actions.shape != (14, 1, AGILEX_CHUNK_ACTIONS)
        or not np.isfinite(actions).all()
    ):
        raise ValueError(
            "committed AgileX action state must be finite float32[14,1,12]"
        )
    anchor = np.asarray(action_anchor_qpos14)
    if (
        anchor.dtype != np.float32
        or anchor.shape != (14,)
        or not np.isfinite(anchor).all()
    ):
        raise ValueError("action anchor must be finite float32[14]")
    if len(image_history) != AGILEX_SENSOR_KEYFRAMES:
        raise ValueError("AgileX grounding requires exactly four RGB keyframes")
    rgb_history = [rgb_images(row) for row in image_history]
    reference_shapes = {key: rgb_history[0][key].shape for key in RGB_KEYS}
    if any(
        row[key].shape != reference_shapes[key]
        for row in rgb_history[1:]
        for key in RGB_KEYS
    ):
        raise ValueError("AgileX RGB history shapes must remain stable")

    _contact_history(
        tactile_history,
        required=route.tactile_required,
        profile=profile,
        label="tactile",
    )
    _contact_history(
        wrench_history,
        required=route.wrench_required,
        profile=profile,
        label="wrench",
    )
    tactile = (
        [tactile_images(row, keys=route.tactile_keys) for row in tactile_history]
        if tactile_history is not None
        else None
    )
    if tactile is not None and any(row is None for row in tactile):
        raise RuntimeError("validated tactile history unexpectedly became empty")
    tactile_rows = cast(list[dict[str, ImageArray]] | None, tactile)
    force = (
        [wrench_values(row, keys=route.wrench_keys) for row in wrench_history]
        if wrench_history is not None
        else None
    )
    if tactile_rows is not None:
        tactile_shapes = {key: tactile_rows[0][key].shape for key in route.tactile_keys}
        if any(
            row[key].shape != tactile_shapes[key]
            for row in tactile_rows[1:]
            for key in route.tactile_keys
        ):
            raise ValueError("AgileX tactile history shapes must remain stable")
    server_rgb = [
        {
            full_key: row[wire_key]
            for wire_key, full_key in zip(RGB_KEYS, AGILEX_RGB_KEYS, strict=True)
        }
        for row in rgb_history
    ]
    observation: dict[str, object] = {
        "obs": server_rgb,
        "state": np.ascontiguousarray(actions),
        "current_state": np.ascontiguousarray(anchor),
        "action_anchor_state": np.ascontiguousarray(anchor),
        "state_action_format": "absolute",
        "compute_kv_cache": True,
        "imagine": False,
        "prompt": prompt,
        "tactile_keys": list(route.tactile_keys),
        "wrench_keys": list(route.wrench_keys),
    }
    if tactile_rows is not None:
        observation["tactile"] = tactile_rows
    if force is not None:
        observation["wrench"] = force
        observation["wrench_available_mask"] = [
            dict.fromkeys(route.wrench_keys, True) for _ in force
        ]
    return observation


def _copy_mapping(
    values: Mapping[str, npt.NDArray[np.generic]],
) -> dict[str, npt.NDArray[np.generic]]:
    return {name: np.ascontiguousarray(value.copy()) for name, value in values.items()}


def grounding_indices(action_count: int) -> tuple[int, ...]:
    """Map 30-Hz post-action observations onto the real 10-Hz sensor grid."""

    if action_count != AGILEX_CHUNK_ACTIONS:
        raise ValueError("AgileX grounding requires exactly 12 executed actions")
    return (2, 5, 8, 11)


def project_qpos14_actions(
    raw: object,
    *,
    current_qpos14: FloatArray,
    dt: float,
    expected_actions: int,
    safety: AgileXSafetyContract,
) -> tuple[FloatArray, int, list[str]]:
    """Apply signed position and per-step limits to an exact-size action batch."""

    array = np.asarray(raw)
    if array.dtype != np.float32:
        raise ValueError("backend action must have dtype float32")
    if array.ndim != 2 or array.shape != (expected_actions, 14):
        raise ValueError(f"backend action must have shape [{expected_actions},14]")
    if not np.isfinite(array).all():
        raise ValueError("backend action contains non-finite values")
    lower = np.asarray(safety.lower_bounds, dtype=np.float32)
    upper = np.asarray(safety.upper_bounds, dtype=np.float32)
    max_step = np.asarray(safety.max_step_per_second, dtype=np.float32) * dt
    previous = current_qpos14.copy()
    output = np.empty_like(array)
    intervention_count = 0
    reasons: list[str] = []
    for index, candidate in enumerate(array):
        bounded = np.clip(candidate, lower, upper)
        position_count = int(np.count_nonzero(bounded != candidate))
        if position_count:
            intervention_count += position_count
            if "joint_position_limit" not in reasons:
                reasons.append("joint_position_limit")
        stepped = np.clip(bounded, previous - max_step, previous + max_step)
        step_count = int(np.count_nonzero(stepped != bounded))
        if step_count:
            intervention_count += step_count
            if "joint_step_limit" not in reasons:
                reasons.append("joint_step_limit")
        output[index] = stepped
        previous = stepped
    return np.ascontiguousarray(output), intervention_count, reasons


@dataclass
class AgileXExecutedChunk:
    """Mutable episode-local queue; it never invents unexecuted actions."""

    actions: FloatArray | None = None
    executed: list[FloatArray] = field(default_factory=list)
    interventions: npt.NDArray[np.bool_] | None = None
    index: int = 0
    anchor: FloatArray | None = None
    images: list[ImageRow] = field(default_factory=list)
    tactile: list[ImageRow] = field(default_factory=list)
    wrench: list[WrenchRow] = field(default_factory=list)
    awaiting_observation: bool = False

    def clear(self) -> None:
        self.actions = None
        self.executed.clear()
        self.interventions = None
        self.index = 0
        self.anchor = None
        self.images.clear()
        self.tactile.clear()
        self.wrench.clear()
        self.awaiting_observation = False

    def start(
        self,
        *,
        actions: FloatArray,
        interventions: npt.NDArray[np.bool_],
        anchor: FloatArray,
    ) -> None:
        if self.actions is not None:
            raise RuntimeError("cannot replace an active AgileX action chunk")
        self.actions = np.ascontiguousarray(actions.copy())
        self.interventions = np.ascontiguousarray(interventions.copy())
        self.anchor = np.ascontiguousarray(anchor.copy())

    def record(
        self,
        *,
        images: Mapping[str, ImageArray],
        tactile: Mapping[str, ImageArray] | None,
        wrench: Mapping[str, FloatArray] | None,
    ) -> None:
        if not self.awaiting_observation:
            return
        self.images.append(_copy_mapping(images))
        if tactile is not None:
            self.tactile.append(_copy_mapping(tactile))
        if wrench is not None:
            self.wrench.append(_copy_mapping(wrench))
        self.awaiting_observation = False

    def dequeue(self) -> tuple[FloatArray, bool]:
        if self.actions is None or self.interventions is None:
            raise RuntimeError("no AgileX action is queued")
        if self.index >= len(self.actions):
            raise RuntimeError("AgileX action queue is exhausted")
        index = self.index
        action = np.ascontiguousarray(self.actions[index : index + 1].copy())
        intervened = bool(self.interventions[index])
        self.index += 1
        self.awaiting_observation = True
        return action, intervened

    def record_executed(self, action: FloatArray) -> None:
        if not self.awaiting_observation:
            raise RuntimeError("cannot record an AgileX action before dequeue")
        array = np.asarray(action)
        if (
            array.dtype != np.float32
            or array.shape != (14,)
            or not np.isfinite(array).all()
        ):
            raise ValueError("executed AgileX action must be finite float32[14]")
        self.executed.append(np.ascontiguousarray(array.copy()))

    @property
    def finished(self) -> bool:
        return self.actions is not None and self.index >= len(self.actions)

    @property
    def depth(self) -> int:
        return 0 if self.actions is None else len(self.actions) - self.index


__all__ = (
    "AGILEX_CHUNK_ACTIONS",
    "AgileXExecutedChunk",
    "build_grounding_observation",
    "grounding_indices",
    "mapping",
    "project_qpos14_actions",
    "rgb_images",
    "tactile_images",
    "wrench_values",
)
