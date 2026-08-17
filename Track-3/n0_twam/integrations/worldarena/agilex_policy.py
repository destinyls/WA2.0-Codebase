# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Reusable AgileX qpos14 core behind an authenticated deployment adapter.

The core validates signed task, observation, safety, and backend contracts.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import numpy as np

from n0_twam.tactile_profiles import VISION_ONLY

from .agilex_policy_contracts import (
    ACTION_SCHEMA,
    RGB_KEYS,
    FloatArray,
    ImageArray,
    AgileXPolicyConfig,
    AgileXSafetyContract,
    AgileXTaskRoute,
)
from .agilex_sync_grounding import (
    AGILEX_CHUNK_ACTIONS,
    AgileXExecutedChunk,
    grounding_indices,
    mapping as _mapping,
    project_qpos14_actions,
    rgb_images as _rgb_images,
    tactile_images as _tactile_images,
    wrench_values as _wrench,
)


class AgileXActionBackend(Protocol):
    """Local backend boundary; no official transport assumptions are made."""

    def reset(self, *, task_id: str, prompt: str, seed: int, profile: str) -> None:
        pass

    def infer(
        self,
        *,
        images: Mapping[str, ImageArray],
        current_qpos14: FloatArray,
        tactile_images: Mapping[str, ImageArray] | None,
        wrench: Mapping[str, FloatArray] | None,
    ) -> FloatArray:
        pass

    def commit_executed_chunk(
        self,
        *,
        actions_qpos14_cfh: FloatArray,
        image_history: tuple[Mapping[str, ImageArray], ...],
        tactile_history: tuple[Mapping[str, ImageArray], ...] | None,
        wrench_history: tuple[Mapping[str, FloatArray], ...] | None,
        action_anchor_qpos14: FloatArray,
    ) -> None:
        pass


class AgileXPolicy:
    """Profile-bound qpos14 Policy with strict observation and safety gates."""

    def __init__(
        self,
        config: AgileXPolicyConfig,
        *,
        backend: AgileXActionBackend,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if config.max_chunk_actions != AGILEX_CHUNK_ACTIONS:
            raise ValueError("AgileX max_chunk_actions must be exactly 12")
        self.config = config
        self._backend = backend
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._lock = threading.Lock()
        self._episode_index = -1
        self._task_id: str | None = None
        self._prompt: str | None = None
        self._backend_reset = False
        self._last_timestamp: float | None = None
        self._chunk = AgileXExecutedChunk()

    def reset(self, reset_info: Mapping[str, Any] | None = None) -> None:
        """Start a fresh episode and discard all session/dynamics state."""

        with self._lock:
            route: AgileXTaskRoute | None = None
            if reset_info is not None:
                task_id = reset_info.get("task_id")
                prompt = reset_info.get("prompt")
                route = self._resolve_route(task_id, prompt)
            self._episode_index += 1
            self._task_id = None
            self._prompt = None
            self._backend_reset = False
            self._last_timestamp = None
            self._chunk.clear()
            if route is not None:
                self._activate(route)

    def _commit_finished_chunk(self) -> bool:
        if not self._chunk.finished:
            return False
        if self._chunk.awaiting_observation:
            raise RuntimeError("final action observation is missing before grounding")
        if self._chunk.actions is None or self._chunk.anchor is None:
            raise RuntimeError("completed AgileX action queue has no anchor")
        count = len(self._chunk.actions)
        if len(self._chunk.images) != count:
            raise RuntimeError("executed action and RGB observation counts differ")
        if len(self._chunk.executed) != count:
            raise RuntimeError("planned and actually returned action counts differ")
        route = self.config.task_routes[cast(str, self._task_id)]
        if route.tactile_required and len(self._chunk.tactile) != count:
            raise RuntimeError("executed action and tactile observation counts differ")
        if route.wrench_required and len(self._chunk.wrench) != count:
            raise RuntimeError("executed action and wrench observation counts differ")
        indices = grounding_indices(count)
        action_state = np.ascontiguousarray(
            np.stack(self._chunk.executed).T[:, None, :], dtype=np.float32
        )
        self._backend.commit_executed_chunk(
            actions_qpos14_cfh=action_state,
            image_history=tuple(self._chunk.images[index] for index in indices),
            tactile_history=(
                tuple(self._chunk.tactile[index] for index in indices)
                if route.tactile_required
                else None
            ),
            wrench_history=(
                tuple(self._chunk.wrench[index] for index in indices)
                if route.wrench_required
                else None
            ),
            action_anchor_qpos14=self._chunk.anchor,
        )
        self._chunk.clear()
        return True

    def _record_and_ground(
        self,
        *,
        images: Mapping[str, ImageArray],
        tactile: Mapping[str, ImageArray] | None,
        wrench: Mapping[str, FloatArray] | None,
    ) -> bool:
        self._chunk.record(images=images, tactile=tactile, wrench=wrench)
        return self._commit_finished_chunk()

    def _resolve_route(self, task_id: object, prompt: object) -> AgileXTaskRoute:
        if not isinstance(task_id, str) or task_id not in self.config.task_routes:
            raise ValueError("task_id is not present in the verified task routes")
        route = self.config.task_routes[task_id]
        if prompt != route.prompt:
            raise ValueError("prompt does not match the verified task route")
        return route

    def _activate(self, route: AgileXTaskRoute) -> None:
        if self._episode_index < 0:
            self._episode_index = 0
        self._backend.reset(
            task_id=route.task_id,
            prompt=route.prompt,
            seed=self.config.episode_seed + self._episode_index,
            profile=self.config.tactile_profile,
        )
        self._task_id = route.task_id
        self._prompt = route.prompt
        self._backend_reset = True

    def _timestamp_and_dt(
        self, observation: Mapping[str, object]
    ) -> tuple[float, float]:
        raw_timestamp = observation.get("timestamp")
        raw_dt = observation.get("execution_dt_s")
        if isinstance(raw_timestamp, bool) or not isinstance(
            raw_timestamp, (int, float)
        ):
            raise ValueError("timestamp must be finite seconds")
        if isinstance(raw_dt, bool) or not isinstance(raw_dt, (int, float)):
            raise ValueError("execution_dt_s must be finite seconds")
        timestamp = float(raw_timestamp)
        dt = float(raw_dt)
        if not np.isfinite((timestamp, dt)).all():
            raise ValueError("timestamp and execution_dt_s must be finite")
        now = float(self._wall_clock())
        if not np.isfinite(now):
            raise ValueError("runtime clock must be finite")
        age = now - timestamp
        if age < -1e-9:
            raise ValueError("observation timestamp is in the future")
        if age > self.config.safety.max_state_age_s:
            raise ValueError("observation state is stale")
        if (
            not self.config.safety.min_execution_dt_s
            <= dt
            <= self.config.safety.max_execution_dt_s
        ):
            raise ValueError("execution_dt_s is outside the signed interval")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("observation timestamps must be strictly monotonic")
        return timestamp, dt

    def _qpos14(self, value: object) -> FloatArray:
        array = np.asarray(value)
        if (
            array.dtype != np.float32
            or array.shape != (14,)
            or not np.isfinite(array).all()
        ):
            raise ValueError("joint_qpos must be finite float32[14]")
        lower = np.asarray(self.config.safety.lower_bounds, dtype=np.float32)
        upper = np.asarray(self.config.safety.upper_bounds, dtype=np.float32)
        if np.any(array < lower) or np.any(array > upper):
            raise ValueError("current qpos14 is outside the signed joint bounds")
        return np.ascontiguousarray(array)

    def _modalities(
        self, observation: Mapping[str, object], route: AgileXTaskRoute
    ) -> tuple[
        Mapping[str, ImageArray] | None,
        Mapping[str, FloatArray] | None,
        list[str],
    ]:
        tactile_present = "tactile" in observation
        wrench_present = "wrench" in observation
        if self.config.tactile_profile == VISION_ONLY:
            if tactile_present:
                _tactile_images(observation["tactile"], keys=None)
            if wrench_present:
                _wrench(observation["wrench"], keys=None)
            dropped = [
                name
                for name, present in (
                    ("tactile", tactile_present),
                    ("wrench", wrench_present),
                )
                if present
            ]
            return None, None, dropped
        if route.tactile_required and not tactile_present:
            raise ValueError(f"task route {route.task_id!r} requires tactile")
        if not route.tactile_required and tactile_present:
            raise ValueError(f"task route {route.task_id!r} forbids tactile")
        if route.wrench_required and not wrench_present:
            raise ValueError(f"task route {route.task_id!r} requires wrench")
        if not route.wrench_required and wrench_present:
            raise ValueError(f"task route {route.task_id!r} forbids wrench")
        tactile = (
            _tactile_images(observation["tactile"], keys=route.tactile_keys)
            if tactile_present
            else None
        )
        wrench = (
            _wrench(observation["wrench"], keys=route.wrench_keys)
            if wrench_present
            else None
        )
        return tactile, wrench, []

    def _safe_chunk(
        self,
        raw: object,
        *,
        current_qpos14: FloatArray,
        dt: float,
        expected_actions: int | None = None,
    ) -> tuple[FloatArray, int, list[str]]:
        expected = (
            self.config.max_chunk_actions
            if expected_actions is None
            else expected_actions
        )
        return project_qpos14_actions(
            raw,
            current_qpos14=current_qpos14,
            dt=dt,
            expected_actions=expected,
            safety=self.config.safety,
        )

    def infer(self, new_obs: Mapping[str, Any]) -> dict[str, object]:
        """Return one qpos14 target and ground every completed executed chunk."""

        with self._lock:
            observation = _mapping(new_obs, label="observation")
            route = self._resolve_route(
                observation.get("task_id"), observation.get("prompt")
            )
            if self._task_id is not None and (
                route.task_id != self._task_id or route.prompt != self._prompt
            ):
                raise ValueError("task or prompt changed without Policy.reset()")
            timestamp, dt = self._timestamp_and_dt(observation)
            qpos = self._qpos14(observation.get("joint_qpos"))
            images = _rgb_images(observation.get("images"))
            tactile, wrench, dropped = self._modalities(observation, route)
            self._last_timestamp = timestamp
            if not self._backend_reset:
                self._activate(route)
            backend_started = float(self._monotonic_clock())
            if not np.isfinite(backend_started):
                raise ValueError("monotonic clock must be finite")
            grounded = self._record_and_ground(
                images=images, tactile=tactile, wrench=wrench
            )
            generated = False
            reasons: list[str] = []
            intervention_count = 0
            if self._chunk.actions is None:
                generated = True
                raw = self._backend.infer(
                    images=images,
                    current_qpos14=qpos,
                    tactile_images=tactile,
                    wrench=wrench,
                )
                finished_wall = float(self._wall_clock())
                if not np.isfinite(finished_wall):
                    raise ValueError("runtime clock must be finite")
                if finished_wall - timestamp > self.config.safety.max_state_age_s:
                    raise ValueError("observation state became stale during inference")
                actions, _, _ = self._safe_chunk(raw, current_qpos14=qpos, dt=dt)
                interventions = np.any(np.abs(actions - np.asarray(raw)) > 1e-6, axis=1)
                self._chunk.start(
                    actions=actions,
                    interventions=interventions,
                    anchor=qpos,
                )
            backend_finished = float(self._monotonic_clock())
            elapsed = backend_finished - backend_started
            if not np.isfinite((backend_finished, elapsed)).all() or elapsed < 0.0:
                raise ValueError("monotonic clock must be finite and nondecreasing")
            if elapsed > self.config.safety.max_inference_latency_s:
                raise TimeoutError(
                    "AgileX inference exceeded the signed replan deadline"
                )
            queued, initially_intervened = self._chunk.dequeue()
            action, step_count, step_reasons = self._safe_chunk(
                queued, current_qpos14=qpos, dt=dt, expected_actions=1
            )
            intervention_count = step_count + int(initially_intervened)
            reasons = list(step_reasons)
            if initially_intervened:
                reasons.insert(0, "initial_chunk_projection")
            action_intervened = intervention_count > 0
            self._chunk.record_executed(action[0])
            queue_depth_after = self._chunk.depth
            return {
                "actions": action,
                "policy_metadata": {
                    "action_schema": ACTION_SCHEMA,
                    "action_format": "joint_absolute",
                    "chunk_size": 1,
                    "dropped_modalities": dropped,
                    "execution_dt_s": dt,
                    "observation_timestamp": timestamp,
                    "queue_policy": "one_action_then_ground_executed_chunk",
                    "safety_intervened": action_intervened,
                    "safety_intervention_count": intervention_count,
                    "safety_intervention_reasons": reasons,
                    "tactile_profile": self.config.tactile_profile,
                    "task_id": route.task_id,
                },
                "policy_timing": {
                    "backend_ms": elapsed * 1000.0,
                    "deadline_s": self.config.safety.max_inference_latency_s,
                    "generated": generated,
                    "grounded": grounded,
                    "kind": (
                        "grounding_refill"
                        if grounded
                        else ("cold_generation" if generated else "queue_hit")
                    ),
                    "queue_depth_after": queue_depth_after,
                },
            }


Policy = AgileXPolicy

__all__ = (
    "ACTION_SCHEMA",
    "RGB_KEYS",
    "AgileXActionBackend",
    "AgileXPolicy",
    "AgileXPolicyConfig",
    "AgileXSafetyContract",
    "AgileXTaskRoute",
    "Policy",
)
