# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Immutable contracts consumed by the AgileX Policy core."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from n0_twam.embodiments import AGILEX_ACTION_SCHEMA
from n0_twam.tactile_profiles import MIXED, VISION_ONLY, VISION_TACTILE

ACTION_SCHEMA = AGILEX_ACTION_SCHEMA
RGB_KEYS = ("top", "wrist_l", "wrist_r")
_PROFILES = frozenset((VISION_TACTILE, MIXED, VISION_ONLY))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

FloatArray = npt.NDArray[np.float32]
ImageArray = npt.NDArray[np.uint8]


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _names(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a tuple of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    return value


def _vector14(value: object, *, label: str, positive: bool = False) -> FloatArray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (14,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain 14 finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{label} must contain 14 finite positive values")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


@dataclass(frozen=True)
class AgileXTaskRoute:
    """One upstream-authenticated task-to-modality route."""

    task_id: str
    prompt: str
    tactile_required: bool
    wrench_required: bool
    tactile_keys: tuple[str, ...]
    wrench_keys: tuple[str, ...]
    contract_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_id, str)
            or not self.task_id
            or not isinstance(self.prompt, str)
            or not self.prompt
        ):
            raise ValueError("AgileX task route requires task_id and prompt")
        if (
            type(self.tactile_required) is not bool
            or type(self.wrench_required) is not bool
        ):
            raise ValueError("AgileX modality requirements must be booleans")
        tactile_keys = _names(self.tactile_keys, label="tactile_keys")
        wrench_keys = _names(self.wrench_keys, label="wrench_keys")
        if self.tactile_required != bool(tactile_keys):
            raise ValueError("tactile_required must exactly match tactile_keys")
        if self.wrench_required != bool(wrench_keys):
            raise ValueError("wrench_required must exactly match wrench_keys")
        if self.wrench_required and not self.tactile_required:
            raise ValueError("wrench input requires an active tactile route")
        _sha256(self.contract_sha256, label="task route contract_sha256")


@dataclass(frozen=True)
class AgileXSafetyContract:
    """Upstream-authenticated per-joint limits used by the local projector."""

    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    max_step_per_second: tuple[float, ...]
    min_execution_dt_s: float
    max_execution_dt_s: float
    max_state_age_s: float
    max_inference_latency_s: float
    contract_sha256: str

    def __post_init__(self) -> None:
        lower = _vector14(self.lower_bounds, label="lower_bounds")
        upper = _vector14(self.upper_bounds, label="upper_bounds")
        max_step = _vector14(
            self.max_step_per_second,
            label="max_step_per_second",
            positive=True,
        )
        if np.any(lower >= upper):
            raise ValueError("each lower bound must be below its upper bound")
        minimum_dt = _positive(self.min_execution_dt_s, label="min_execution_dt_s")
        maximum_dt = _positive(self.max_execution_dt_s, label="max_execution_dt_s")
        if minimum_dt > maximum_dt:
            raise ValueError("min_execution_dt_s cannot exceed max_execution_dt_s")
        _positive(self.max_state_age_s, label="max_state_age_s")
        _positive(self.max_inference_latency_s, label="max_inference_latency_s")
        _sha256(self.contract_sha256, label="safety contract_sha256")
        object.__setattr__(self, "lower_bounds", tuple(map(float, lower)))
        object.__setattr__(self, "upper_bounds", tuple(map(float, upper)))
        object.__setattr__(self, "max_step_per_second", tuple(map(float, max_step)))


@dataclass(frozen=True)
class AgileXPolicyConfig:
    """Immutable Policy-core config bound to one tactile profile."""

    policy_id: str
    tactile_profile: str
    task_routes: Mapping[str, AgileXTaskRoute]
    episode_seed: int
    max_chunk_actions: int
    safety: AgileXSafetyContract

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must be non-empty")
        if self.tactile_profile not in _PROFILES:
            raise ValueError(f"unknown tactile profile {self.tactile_profile!r}")
        if type(self.episode_seed) is not int or self.episode_seed < 0:
            raise ValueError("episode_seed must be a non-negative integer")
        if (
            type(self.max_chunk_actions) is not int
            or not 1 <= self.max_chunk_actions <= 128
        ):
            raise ValueError("max_chunk_actions must be an integer in [1,128]")
        routes = dict(self.task_routes)
        if not routes or any(key != route.task_id for key, route in routes.items()):
            raise ValueError("task route keys must exactly match route task_id")
        if self.tactile_profile == VISION_TACTILE and any(
            not route.tactile_required or not route.wrench_required
            for route in routes.values()
        ):
            raise ValueError(
                "vision_tactile profile requires tactile and wrench on every route"
            )
        if self.tactile_profile == VISION_ONLY and any(
            route.tactile_required or route.wrench_required for route in routes.values()
        ):
            raise ValueError("vision_only routes cannot require tactile or wrench")
        object.__setattr__(self, "task_routes", MappingProxyType(routes))


__all__ = (
    "ACTION_SCHEMA",
    "RGB_KEYS",
    "FloatArray",
    "ImageArray",
    "AgileXPolicyConfig",
    "AgileXSafetyContract",
    "AgileXTaskRoute",
)
