# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""AgileX qpos14 Policy-core routing and safety contracts."""

from __future__ import annotations

import numpy as np
import pytest

from n0_twam.integrations.worldarena.agilex_policy import (
    ACTION_SCHEMA,
    AgileXPolicy,
    AgileXPolicyConfig,
    AgileXSafetyContract,
    AgileXTaskRoute,
)


class _Backend:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.reset_calls: list[dict[str, object]] = []
        self.infer_calls: list[dict[str, object]] = []
        self.commit_calls: list[dict[str, object]] = []

    def reset(self, *, task_id: str, prompt: str, seed: int, profile: str) -> None:
        self.reset_calls.append(
            {"task_id": task_id, "prompt": prompt, "seed": seed, "profile": profile}
        )

    def infer(
        self,
        *,
        images,
        current_qpos14,
        tactile_images,
        wrench,
    ) -> np.ndarray:
        self.infer_calls.append(
            {
                "images": images,
                "current_qpos14": current_qpos14.copy(),
                "tactile_images": tactile_images,
                "wrench": wrench,
            }
        )
        return self.output.copy()

    def commit_executed_chunk(
        self,
        *,
        actions_qpos14_cfh,
        image_history,
        tactile_history,
        wrench_history,
        action_anchor_qpos14,
    ) -> None:
        self.commit_calls.append(
            {
                "actions": actions_qpos14_cfh.copy(),
                "images": image_history,
                "tactile": tactile_history,
                "wrench": wrench_history,
                "anchor": action_anchor_qpos14.copy(),
            }
        )


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _SequenceClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _route(*, task_id: str, tactile: bool, wrench: bool) -> AgileXTaskRoute:
    return AgileXTaskRoute(
        task_id=task_id,
        prompt=f"perform {task_id}",
        tactile_required=tactile,
        wrench_required=wrench,
        tactile_keys=("left", "right") if tactile else (),
        wrench_keys=("left", "right") if wrench else (),
        contract_sha256="a" * 64,
    )


def _config(
    profile: str = "mixed",
    *,
    routes: tuple[AgileXTaskRoute, ...] | None = None,
) -> AgileXPolicyConfig:
    selected = routes or (
        _route(task_id="wipe", tactile=True, wrench=True),
        _route(task_id="place", tactile=False, wrench=False),
    )
    return AgileXPolicyConfig(
        policy_id="agilex-test",
        tactile_profile=profile,
        task_routes={route.task_id: route for route in selected},
        episode_seed=17,
        max_chunk_actions=12,
        safety=AgileXSafetyContract(
            lower_bounds=(-1.0,) * 14,
            upper_bounds=(1.0,) * 14,
            max_step_per_second=(0.5,) * 14,
            min_execution_dt_s=0.01,
            max_execution_dt_s=0.2,
            max_state_age_s=0.25,
            max_inference_latency_s=0.1,
            contract_sha256="b" * 64,
        ),
    )


def _images(value: int = 0) -> dict[str, np.ndarray]:
    return {
        "top": np.full((8, 10, 3), value, dtype=np.uint8),
        "wrist_l": np.full((8, 10, 3), value + 1, dtype=np.uint8),
        "wrist_r": np.full((8, 10, 3), value + 2, dtype=np.uint8),
    }


def _tactile(value: int = 0) -> dict[str, np.ndarray]:
    return {
        "left": np.full((6, 7, 3), value, dtype=np.uint8),
        "right": np.full((6, 7, 3), value + 1, dtype=np.uint8),
    }


def _wrench(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "left": np.full((6,), value, dtype=np.float32),
        "right": np.full((6,), value + 1.0, dtype=np.float32),
    }


def _observation(
    *,
    task_id: str = "wipe",
    timestamp: float = 99.9,
    dt: float = 0.1,
    tactile: bool = True,
    wrench: bool = True,
) -> dict[str, object]:
    observation: dict[str, object] = {
        "task_id": task_id,
        "prompt": f"perform {task_id}",
        "timestamp": timestamp,
        "execution_dt_s": dt,
        "joint_qpos": np.zeros((14,), dtype=np.float32),
        "images": _images(),
    }
    if tactile:
        observation["tactile"] = _tactile()
    if wrench:
        observation["wrench"] = _wrench()
    return observation


def _policy(
    config: AgileXPolicyConfig,
    backend: _Backend,
    *,
    wall: _Clock | None = None,
    monotonic: _Clock | None = None,
) -> AgileXPolicy:
    return AgileXPolicy(
        config,
        backend=backend,
        wall_clock=wall or _Clock(100.0),
        monotonic_clock=monotonic or _Clock(10.0),
    )


def test_mixed_tactile_route_passes_exact_modalities_and_emits_qpos14() -> None:
    backend = _Backend(np.full((12, 14), 0.02, dtype=np.float32))
    policy = _policy(_config(), backend)
    result = policy.infer(_observation())
    assert result["actions"].shape == (1, 14)
    assert result["actions"].dtype == np.float32
    assert result["policy_metadata"] == {
        "action_schema": ACTION_SCHEMA,
        "action_format": "joint_absolute",
        "chunk_size": 1,
        "dropped_modalities": [],
        "execution_dt_s": 0.1,
        "observation_timestamp": 99.9,
        "queue_policy": "one_action_then_ground_executed_chunk",
        "safety_intervened": False,
        "safety_intervention_count": 0,
        "safety_intervention_reasons": [],
        "tactile_profile": "mixed",
        "task_id": "wipe",
    }
    assert backend.reset_calls == [
        {
            "task_id": "wipe",
            "prompt": "perform wipe",
            "seed": 17,
            "profile": "mixed",
        }
    ]
    call = backend.infer_calls[0]
    assert set(call["images"]) == {"top", "wrist_l", "wrist_r"}
    assert set(call["tactile_images"]) == {"left", "right"}
    assert set(call["wrench"]) == {"left", "right"}


@pytest.mark.parametrize("missing", ["tactile", "wrench"])
def test_mixed_route_rejects_missing_required_modality(missing: str) -> None:
    backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    policy = _policy(_config(), backend)
    observation = _observation()
    observation.pop(missing)
    with pytest.raises(ValueError, match=f"requires {missing}"):
        policy.infer(observation)
    assert backend.infer_calls == []


def test_mixed_no_tactile_route_rejects_unexpected_modalities() -> None:
    backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    policy = _policy(_config(), backend)
    with pytest.raises(ValueError, match="forbids tactile"):
        policy.infer(_observation(task_id="place"))
    assert backend.infer_calls == []


def test_vision_only_valid_extra_modalities_are_validated_then_dropped() -> None:
    route = _route(task_id="place", tactile=False, wrench=False)
    config = _config("vision_only", routes=(route,))
    output = np.linspace(0.0, 0.02, 168, dtype=np.float32).reshape(12, 14)
    backend_with = _Backend(output)
    backend_without = _Backend(output)
    with_extra = _policy(config, backend_with).infer(
        _observation(task_id="place", tactile=True, wrench=True)
    )
    without_extra = _policy(config, backend_without).infer(
        _observation(task_id="place", tactile=False, wrench=False)
    )
    np.testing.assert_array_equal(with_extra["actions"], without_extra["actions"])
    assert with_extra["policy_metadata"]["dropped_modalities"] == [
        "tactile",
        "wrench",
    ]
    assert backend_with.infer_calls[0]["tactile_images"] is None
    assert backend_with.infer_calls[0]["wrench"] is None
    assert backend_without.infer_calls[0]["tactile_images"] is None
    assert backend_without.infer_calls[0]["wrench"] is None


def test_vision_only_rejects_malformed_extra_tactile_before_drop() -> None:
    route = _route(task_id="place", tactile=False, wrench=False)
    policy = _policy(
        _config("vision_only", routes=(route,)),
        _Backend(np.zeros((12, 14), dtype=np.float32)),
    )
    observation = _observation(task_id="place", wrench=False)
    observation["tactile"] = {
        "left": np.zeros((6, 7, 3), dtype=np.float32),
        "right": np.zeros((6, 7, 3), dtype=np.uint8),
    }
    with pytest.raises(ValueError, match="uint8 HWC RGB"):
        policy.infer(observation)


def test_signed_limits_and_actual_dt_project_each_chunk_step() -> None:
    output = np.zeros((12, 14), dtype=np.float32)
    output[0] = 2.0
    output[1] = -2.0
    backend = _Backend(output)
    policy = _policy(_config(), backend)
    result = policy.infer(_observation(dt=0.1))
    np.testing.assert_allclose(result["actions"][0], 0.05, atol=1e-7)
    assert result["actions"].shape == (1, 14)
    metadata = result["policy_metadata"]
    assert metadata["safety_intervened"] is True
    assert metadata["safety_intervention_count"] == 1
    assert metadata["safety_intervention_reasons"] == [
        "initial_chunk_projection",
    ]


@pytest.mark.parametrize(
    ("timestamp", "dt", "message"),
    [
        (99.0, 0.1, "stale"),
        (100.1, 0.1, "future"),
        (99.9, 0.0, "execution_dt_s"),
        (99.9, 0.3, "execution_dt_s"),
    ],
)
def test_stale_future_and_invalid_dt_fail_closed(
    timestamp: float, dt: float, message: str
) -> None:
    backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    policy = _policy(_config(), backend)

    with pytest.raises(ValueError, match=message):
        policy.infer(_observation(timestamp=timestamp, dt=dt))

    assert backend.infer_calls == []


def test_nonfinite_runtime_clock_fails_closed_before_backend() -> None:
    backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    policy = _policy(_config(), backend, wall=_Clock(float("nan")))

    with pytest.raises(ValueError, match="runtime clock"):
        policy.infer(_observation())

    assert backend.infer_calls == []


def test_nonfinite_or_backward_monotonic_clock_fails_closed() -> None:
    for values in ((10.0, float("nan")), (10.0, 9.0)):
        backend = _Backend(np.zeros((12, 14), dtype=np.float32))
        policy = _policy(_config(), backend, monotonic=_SequenceClock(values))

        with pytest.raises(ValueError, match="monotonic clock"):
            policy.infer(_observation())


def test_timestamps_are_strictly_monotonic_and_reset_clears_session_state() -> None:
    backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    policy = _policy(_config(), backend)
    policy.infer(_observation(timestamp=99.8))

    with pytest.raises(ValueError, match="strictly monotonic"):
        policy.infer(_observation(timestamp=99.8))

    policy.reset({"task_id": "wipe", "prompt": "perform wipe"})
    restarted = policy.infer(_observation(timestamp=99.8))
    assert restarted["actions"].shape == (1, 14)
    assert [call["seed"] for call in backend.reset_calls] == [17, 18]


def test_prompt_or_task_change_requires_reset() -> None:
    backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    policy = _policy(_config(), backend)
    policy.infer(_observation(timestamp=99.8))
    changed = _observation(task_id="place", timestamp=99.9, tactile=False, wrench=False)

    with pytest.raises(ValueError, match="changed without Policy.reset"):
        policy.infer(changed)


def test_backend_shape_nonfinite_and_deadline_fail_closed() -> None:
    malformed = _Backend(np.zeros((1, 13), dtype=np.float32))
    with pytest.raises(ValueError, match=r"shape \[12,14\]"):
        _policy(_config(), malformed).infer(_observation())

    nonfinite = _Backend(np.full((12, 14), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        _policy(_config(), nonfinite).infer(_observation())

    monotonic = _Clock(10.0)
    deadline = _Backend(np.zeros((12, 14), dtype=np.float32))
    policy = _policy(_config(), deadline, monotonic=monotonic)

    def slow_infer(**kwargs):
        monotonic.value += 0.2
        return deadline.output.copy()

    deadline.infer = slow_infer  # type: ignore[method-assign]
    with pytest.raises(TimeoutError, match="deadline"):
        policy.infer(_observation())
