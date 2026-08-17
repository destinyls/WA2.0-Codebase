# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""WorldArena ``Policy`` wire adapter for the local AgileX qpos14 policy.

This module implements only the organizer-documented in-process Python
interface.  It deliberately contains no Hub URL, worker identity, credential,
or network-transport assumptions.  The organizer guide defines
``tactile_profile`` only as a non-empty label; contact presence is therefore
governed exclusively by the signed task route, never inferred from that label.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from .agilex_backend import DirectN0AgileXBackend
from .agilex_policy import AgileXPolicy
from .agilex_policy_contracts import (
    AgileXPolicyConfig,
    FloatArray,
    ImageArray,
)
from .agilex_policy_io import AgileXDirectPolicyConfig, load_agilex_policy_config

POLICY_CONFIG_ENV = "N0_TRACK3_AGILEX_POLICY_CONFIG"

_WIRE_RGB_TO_INTERNAL = {
    "cam_high": "top",
    "cam_wrist_left": "wrist_l",
    "cam_wrist_right": "wrist_r",
}
_WIRE_TACTILE_TO_INTERNAL = {
    "left_gripper": "observation.images.tactile_l",
    "right_gripper": "observation.images.tactile_r",
}
_WIRE_WRENCH_TO_INTERNAL = {
    "left_wrist_force": "observation.wrench.left",
    "right_wrist_force": "observation.wrench.right",
}


class _PolicyCore(Protocol):
    config: AgileXPolicyConfig

    def reset(self, reset_info: Mapping[str, Any] | None = None) -> None:
        pass

    def infer(self, new_obs: Mapping[str, Any]) -> dict[str, object]:
        pass


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return cast(Mapping[str, object], value)


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite seconds")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite seconds")
    return result


def _qpos14(value: object, *, label: str) -> FloatArray:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.shape != (14,)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{label} must be finite float32[14]")
    return cast(FloatArray, np.ascontiguousarray(array))


def _joint7(value: object, *, label: str) -> FloatArray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != (7,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite float32[7]")
    return cast(FloatArray, np.ascontiguousarray(array))


def _image(value: object, *, label: str) -> ImageArray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{label} must be uint8 HWC RGB")
    return cast(ImageArray, np.ascontiguousarray(array))


def _wrench(value: object, *, label: str) -> FloatArray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite float32[6]")
    return cast(FloatArray, np.ascontiguousarray(array))


def _policy_config(
    config: AgileXDirectPolicyConfig | AgileXPolicyConfig,
) -> AgileXPolicyConfig:
    if isinstance(config, AgileXDirectPolicyConfig):
        return config.policy
    if isinstance(config, AgileXPolicyConfig):
        return config
    raise TypeError("config must be an AgileX policy config")


def _validate_routes(config: AgileXPolicyConfig) -> None:
    tactile = frozenset(_WIRE_TACTILE_TO_INTERNAL.values())
    wrench = frozenset(_WIRE_WRENCH_TO_INTERNAL.values())
    for task_id, route in config.task_routes.items():
        unknown_tactile = set(route.tactile_keys) - tactile
        unknown_wrench = set(route.wrench_keys) - wrench
        if unknown_tactile or unknown_wrench:
            raise ValueError(
                f"task route {task_id!r} contains contact keys unsupported by "
                "the official AgileX wire adapter"
            )


def _resolve_config_path(config_path: str | None) -> Path:
    selected = (
        config_path if config_path is not None else os.environ.get(POLICY_CONFIG_ENV)
    )
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError(f"{POLICY_CONFIG_ENV} must name the signed policy config")
    return Path(selected).expanduser()


class Policy:
    """Organizer-facing adapter backed by the strict local AgileX policy.

    Normal deployment uses ``Policy()`` and reads
    :data:`POLICY_CONFIG_ENV`.  ``core`` and ``config`` are explicit test and
    embedding seams; they do not weaken the production artifact loader.
    """

    def __init__(
        self,
        config_path: str | None = None,
        *,
        core: _PolicyCore | None = None,
        config: AgileXDirectPolicyConfig | AgileXPolicyConfig | None = None,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if core is None:
            if config is None:
                config = load_agilex_policy_config(_resolve_config_path(config_path))
            elif config_path is not None:
                raise ValueError(
                    "config_path and injected config are mutually exclusive"
                )
            if not isinstance(config, AgileXDirectPolicyConfig):
                raise TypeError(
                    "a direct policy config is required to construct backend"
                )
            _validate_routes(config.policy)
            backend = DirectN0AgileXBackend(config)
            core = AgileXPolicy(
                config.policy,
                backend=backend,
                wall_clock=wall_clock,
            )
            self._owned_backend: DirectN0AgileXBackend | None = backend
        else:
            if config_path is not None:
                raise ValueError("config_path cannot be combined with injected core")
            if config is not None and _policy_config(config) != core.config:
                raise ValueError("injected core and config differ")
            config = config or core.config
            self._owned_backend = None
        self._core = core
        self._config = _policy_config(config)
        self._wall_clock = wall_clock
        self._closed = False
        _validate_routes(self._config)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Policy is closed")

    def reset(self, reset_info: Mapping[str, Any] | None = None) -> None:
        """Reset the local episode, optionally binding ``task_id``/``prompt``."""

        self._ensure_open()
        if reset_info is None:
            self._core.reset(None)
            return
        payload = _mapping(reset_info, label="reset_info")
        self._core.reset(
            {
                "task_id": payload.get("task_id"),
                "prompt": payload.get("prompt"),
            }
        )

    def _images(self, value: object) -> dict[str, ImageArray]:
        payload = _mapping(value, label="images")
        if set(payload) != set(_WIRE_RGB_TO_INTERNAL):
            raise ValueError(
                f"images must contain exactly {list(_WIRE_RGB_TO_INTERNAL)}"
            )
        return {
            internal: _image(payload[wire], label=f"images.{wire}")
            for wire, internal in _WIRE_RGB_TO_INTERNAL.items()
        }

    def _contact(
        self, value: object
    ) -> tuple[dict[str, ImageArray] | None, dict[str, FloatArray] | None]:
        payload = _mapping(value, label="tactile")
        allowed = {*_WIRE_TACTILE_TO_INTERNAL, *_WIRE_WRENCH_TO_INTERNAL}
        if not payload or set(payload) - allowed:
            raise ValueError(
                "tactile must contain only documented AgileX contact groups"
            )
        tactile: dict[str, ImageArray] = {}
        wrench: dict[str, FloatArray] = {}
        for wire, internal in _WIRE_TACTILE_TO_INTERNAL.items():
            if wire not in payload:
                continue
            group = _mapping(payload[wire], label=f"tactile.{wire}")
            if set(group) != {"rectify"}:
                raise ValueError(f"tactile.{wire} must contain exactly rectify")
            tactile[internal] = _image(
                group["rectify"], label=f"tactile.{wire}.rectify"
            )
        for wire, internal in _WIRE_WRENCH_TO_INTERNAL.items():
            if wire not in payload:
                continue
            group = _mapping(payload[wire], label=f"tactile.{wire}")
            if set(group) != {"wrench_6d"}:
                raise ValueError(f"tactile.{wire} must contain exactly wrench_6d")
            wrench[internal] = _wrench(
                group["wrench_6d"], label=f"tactile.{wire}.wrench_6d"
            )
        return tactile or None, wrench or None

    @staticmethod
    def _validate_wire_profile(value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("tactile_profile must be a non-empty string")

    def infer(self, new_obs: Mapping[str, Any]) -> dict[str, object]:
        """Map the documented WorldArena observation to the local policy core."""

        self._ensure_open()
        observation = _mapping(new_obs, label="observation")
        qpos = _qpos14(observation.get("joint_qpos"), label="joint_qpos")
        state = _qpos14(observation.get("state"), label="state")
        if not np.array_equal(qpos, state):
            raise ValueError("state and joint_qpos must be identical")
        left = _joint7(
            observation.get("left_arm_joint_state"),
            label="left_arm_joint_state",
        )
        right = _joint7(
            observation.get("right_arm_joint_state"),
            label="right_arm_joint_state",
        )
        if not np.array_equal(left, qpos[:7]):
            raise ValueError("left_arm_joint_state differs from joint_qpos[:7]")
        if not np.array_equal(right, qpos[7:]):
            raise ValueError("right_arm_joint_state differs from joint_qpos[7:]")
        self._validate_wire_profile(observation.get("tactile_profile"))
        timestamp = (
            _finite_number(observation["timestamp"], label="timestamp")
            if "timestamp" in observation
            else _finite_number(self._wall_clock(), label="runtime clock")
        )
        dt = (
            _finite_number(observation["execution_dt_s"], label="execution_dt_s")
            if "execution_dt_s" in observation
            else self._config.safety.min_execution_dt_s
        )
        internal: dict[str, object] = {
            "images": self._images(observation.get("images")),
            "joint_qpos": qpos,
            "task_id": observation.get("task_id"),
            "prompt": observation.get("prompt"),
            "timestamp": timestamp,
            "execution_dt_s": dt,
        }
        if "tactile" in observation:
            tactile, wrench = self._contact(observation["tactile"])
            if tactile is not None:
                internal["tactile"] = tactile
            if wrench is not None:
                internal["wrench"] = wrench
        result = self._core.infer(internal)
        if not isinstance(result, dict):
            raise ValueError("policy core output must be a dict")
        actions = np.asarray(result.get("actions"))
        if (
            actions.dtype != np.float32
            or actions.shape != (1, 14)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("policy output actions must be finite float32[1,14]")
        return {**result, "actions": np.ascontiguousarray(actions)}

    def close(self) -> None:
        """Release owned model/distributed resources exactly once."""

        if self._closed:
            return
        self._closed = True
        if self._owned_backend is not None:
            self._owned_backend.close()
            return
        close = getattr(self._core, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "Policy":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


__all__ = ("POLICY_CONFIG_ENV", "Policy")
