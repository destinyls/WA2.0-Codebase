# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Synchronous AgileX executed-action grounding contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from n0_twam.integrations.worldarena.agilex_backend import DirectN0AgileXBackend
from n0_twam.integrations.worldarena.agilex_policy import (
    AgileXSafetyContract,
    AgileXTaskRoute,
)
from n0_twam.integrations.worldarena.agilex_sync_grounding import grounding_indices

from .test_agilex_backend import (
    _RuntimeConfig,
    _Server,
    _images as _backend_images,
    _touch,
    _wrench as _backend_wrench,
)
from .test_agilex_policy import (
    _Backend,
    _Clock,
    _SequenceClock,
    _config,
    _images,
    _observation,
    _policy,
    _route,
    _tactile,
    _wrench,
)


def _direct_backend(config: _RuntimeConfig, server: _Server) -> DirectN0AgileXBackend:
    return DirectN0AgileXBackend(
        config,  # type: ignore[arg-type]
        server_factory=lambda _: server,
        artifact_verifier=lambda _: {},
    )


def _commit_kwargs(*, contact: bool) -> dict[str, Any]:
    return {
        "actions_qpos14_cfh": np.zeros((14, 1, 12), dtype=np.float32),
        "image_history": tuple(_backend_images() for _ in range(4)),
        "tactile_history": tuple(_touch() for _ in range(4)) if contact else None,
        "wrench_history": (
            tuple(_backend_wrench() for _ in range(4)) if contact else None
        ),
        "action_anchor_qpos14": np.zeros(14, dtype=np.float32),
    }


def test_grounding_indices_map_30hz_actions_to_four_real_10hz_keyframes() -> None:
    assert grounding_indices(12) == (2, 5, 8, 11)


def test_policy_commits_only_returned_safe_actions_and_aligned_contact() -> None:
    raw = np.full((12, 14), 2.0, dtype=np.float32)
    backend = _Backend(raw)
    policy = _policy(_config(), backend, wall=_Clock(100.0))
    returned: list[np.ndarray] = []
    actual_qpos = np.zeros(14, dtype=np.float32)

    for index in range(13):
        observation = _observation(timestamp=99.76 + index * 0.01)
        observation["joint_qpos"] = actual_qpos.copy()
        observation["images"] = _images(index)
        observation["tactile"] = _tactile(index)
        observation["wrench"] = _wrench(float(index))
        result = policy.infer(observation)
        actual_qpos = np.asarray(result["actions"])[0].copy()
        if index < 12:
            returned.append(actual_qpos.copy())

    assert len(backend.infer_calls) == 2
    assert len(backend.commit_calls) == 1
    committed = backend.commit_calls[0]
    state = np.asarray(committed["actions"])
    assert state.shape == (14, 1, 12)
    np.testing.assert_array_equal(state[0, 0], [action[0] for action in returned])
    np.testing.assert_allclose(state[0, 0], np.arange(1, 13) * 0.05, atol=1e-6)
    assert np.asarray(committed["anchor"]).shape == (14,)

    images = committed["images"]
    tactile = committed["tactile"]
    wrench = committed["wrench"]
    assert isinstance(images, tuple) and isinstance(tactile, tuple)
    assert isinstance(wrench, tuple)
    assert [int(row["top"][0, 0, 0]) for row in images] == [3, 6, 9, 12]
    assert [int(row["left"][0, 0, 0]) for row in tactile] == [3, 6, 9, 12]
    assert [float(row["left"][0]) for row in wrench] == [3.0, 6.0, 9.0, 12.0]


def test_policy_reprojects_queued_action_against_each_real_qpos() -> None:
    backend = _Backend(np.full((12, 14), 2.0, dtype=np.float32))
    policy = _policy(_config(), backend, wall=_Clock(100.0))

    first = policy.infer(_observation(timestamp=99.76))
    second = policy.infer(_observation(timestamp=99.77))

    np.testing.assert_allclose(first["actions"], 0.05, atol=1e-7)
    np.testing.assert_allclose(second["actions"], 0.05, atol=1e-7)


def test_grounding_and_refill_share_one_signed_latency_deadline() -> None:
    backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    clock_values = [0.0, 0.01] * 12 + [0.0, 0.2]
    policy = _policy(
        _config(),
        backend,
        wall=_Clock(100.0),
        monotonic=_SequenceClock(tuple(clock_values)),
    )
    for index in range(12):
        policy.infer(_observation(timestamp=99.76 + index * 0.01))

    with pytest.raises(TimeoutError, match="deadline"):
        policy.infer(_observation(timestamp=99.88))
    assert len(backend.commit_calls) == 1


def test_mixed_rgb_only_route_commits_without_contact_history() -> None:
    backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    policy = _policy(_config(), backend, wall=_Clock(100.0))

    for index in range(13):
        observation = _observation(
            task_id="place",
            timestamp=99.76 + index * 0.01,
            tactile=False,
            wrench=False,
        )
        observation["images"] = _images(index)
        policy.infer(observation)

    assert len(backend.commit_calls) == 1
    assert backend.commit_calls[0]["tactile"] is None
    assert backend.commit_calls[0]["wrench"] is None


def test_reset_discards_partial_chunk_and_vision_only_commit_has_no_contact() -> None:
    route = _route(task_id="place", tactile=False, wrench=False)
    backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    policy = _policy(
        _config("vision_only", routes=(route,)), backend, wall=_Clock(100.0)
    )
    for index in range(2):
        policy.infer(
            _observation(
                task_id="place",
                timestamp=99.76 + index * 0.01,
                tactile=False,
                wrench=False,
            )
        )
    policy.reset({"task_id": "place", "prompt": "perform place"})
    assert backend.commit_calls == []

    for index in range(13):
        observation = _observation(
            task_id="place",
            timestamp=99.76 + index * 0.01,
            tactile=False,
            wrench=False,
        )
        observation["images"] = _images(index)
        policy.infer(observation)

    assert len(backend.commit_calls) == 1
    assert backend.commit_calls[0]["tactile"] is None
    assert backend.commit_calls[0]["wrench"] is None


def test_backend_commits_one_real_action_frame_and_four_contact_frames() -> None:
    server = _Server(np.zeros((14, 2, 12), dtype=np.float32))
    backend = _direct_backend(_RuntimeConfig(), server)
    backend.reset(
        task_id="wipe", prompt="wipe the table", seed=7, profile="vision_tactile"
    )
    actions = np.arange(14 * 12, dtype=np.float32).reshape(14, 1, 12)
    kwargs = _commit_kwargs(contact=True)
    kwargs["actions_qpos14_cfh"] = actions
    backend.commit_executed_chunk(**kwargs)

    observation = server.calls[-1]
    assert observation["compute_kv_cache"] is True
    assert observation["imagine"] is False
    np.testing.assert_array_equal(observation["state"], actions)
    assert np.asarray(observation["state"]).shape == (14, 1, 12)
    assert len(observation["obs"]) == 4
    assert len(observation["tactile"]) == 4
    assert len(observation["wrench"]) == 4
    assert observation["wrench_available_mask"] == [
        dict.fromkeys(_backend_wrench(), True) for _ in range(4)
    ]


def test_backend_mixed_rgb_only_route_uses_no_contact_history() -> None:
    config = _RuntimeConfig("mixed", contact=False)
    server = _Server(np.zeros((14, 2, 12), dtype=np.float32))
    backend = _direct_backend(config, server)
    backend.reset(task_id="wipe", prompt="wipe the table", seed=7, profile="mixed")

    backend.commit_executed_chunk(**_commit_kwargs(contact=False))

    assert "tactile" not in server.calls[-1]
    assert "wrench" not in server.calls[-1]


def test_backend_vision_only_commit_rejects_contact() -> None:
    config = _RuntimeConfig("vision_only")
    server = _Server(np.zeros((14, 2, 12), dtype=np.float32))
    backend = _direct_backend(config, server)
    backend.reset(
        task_id="wipe", prompt="wipe the table", seed=7, profile="vision_only"
    )
    backend.commit_executed_chunk(**_commit_kwargs(contact=False))
    assert "tactile" not in server.calls[-1]
    assert "wrench" not in server.calls[-1]

    with pytest.raises(ValueError, match="vision_only"):
        backend.commit_executed_chunk(**_commit_kwargs(contact=True))


@pytest.mark.parametrize(
    "actions",
    (
        np.zeros((14, 2, 12), dtype=np.float32),
        np.zeros((14, 1, 11), dtype=np.float32),
        np.zeros((14, 1, 12), dtype=np.float64),
        np.full((14, 1, 12), np.nan, dtype=np.float32),
    ),
)
def test_backend_commit_rejects_malformed_action_state(actions: np.ndarray) -> None:
    server = _Server(np.zeros((14, 2, 12), dtype=np.float32))
    backend = _direct_backend(_RuntimeConfig(), server)
    backend.reset(
        task_id="wipe", prompt="wipe the table", seed=7, profile="vision_tactile"
    )
    kwargs = _commit_kwargs(contact=True)
    kwargs["actions_qpos14_cfh"] = actions
    with pytest.raises(ValueError, match=r"float32\[14,1,12\]"):
        backend.commit_executed_chunk(**kwargs)


def test_direct_backend_rejects_non_twelve_action_runtime_contract() -> None:
    config = _RuntimeConfig()
    object.__setattr__(config.policy, "max_chunk_actions", 8)

    with pytest.raises(ValueError, match="max_chunk_actions.*12"):
        _direct_backend(config, _Server(np.zeros((14, 2, 12), dtype=np.float32)))


def test_contracts_reject_invalid_profile_and_unsigned_safety_shapes() -> None:
    route = _route(task_id="place", tactile=False, wrench=False)
    with pytest.raises(ValueError, match="unknown tactile profile"):
        _config("other", routes=(route,))
    with pytest.raises(ValueError, match="14 finite"):
        replace(_config(routes=(route,)).safety, lower_bounds=(0.0,) * 13)
    with pytest.raises(ValueError, match="requires tactile"):
        _config("vision_tactile", routes=(route,))
    with pytest.raises(ValueError, match="exactly match tactile_keys"):
        replace(
            _route(task_id="wipe", tactile=True, wrench=True), tactile_required=False
        )
    with pytest.raises(ValueError, match="wrench input requires"):
        AgileXTaskRoute(
            task_id="wipe",
            prompt="perform wipe",
            tactile_required=False,
            wrench_required=True,
            tactile_keys=(),
            wrench_keys=("left",),
            contract_sha256="a" * 64,
        )


def test_contracts_are_deeply_immutable_and_require_string_route_identity() -> None:
    lower = [-1.0] * 14
    safety = AgileXSafetyContract(
        lower_bounds=lower,  # type: ignore[arg-type]
        upper_bounds=[1.0] * 14,  # type: ignore[arg-type]
        max_step_per_second=[0.5] * 14,  # type: ignore[arg-type]
        min_execution_dt_s=0.01,
        max_execution_dt_s=0.2,
        max_state_age_s=0.25,
        max_inference_latency_s=0.1,
        contract_sha256="c" * 64,
    )
    lower[0] = 99.0
    assert safety.lower_bounds == (-1.0,) * 14
    assert isinstance(safety.upper_bounds, tuple)
    assert isinstance(safety.max_step_per_second, tuple)
    with pytest.raises(ValueError, match="task_id and prompt"):
        AgileXTaskRoute(
            task_id=7,  # type: ignore[arg-type]
            prompt="perform task",
            tactile_required=False,
            wrench_required=False,
            tactile_keys=("left", "right"),
            wrench_keys=("left", "right"),
            contract_sha256="d" * 64,
        )
