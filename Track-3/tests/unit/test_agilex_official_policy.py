# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Organizer-wire tests for the AgileX qpos14 Policy adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from n0_twam.integrations.worldarena.agilex_official_policy import (
    POLICY_CONFIG_ENV,
    Policy,
)
from n0_twam.integrations.worldarena.agilex_policy import AgileXPolicy
from n0_twam.integrations.worldarena.agilex_policy_contracts import (
    AgileXPolicyConfig,
    AgileXSafetyContract,
    AgileXTaskRoute,
    FloatArray,
    ImageArray,
)


class _Core:
    def __init__(
        self, config: AgileXPolicyConfig, actions: object | None = None
    ) -> None:
        self.config = config
        self.actions = (
            np.zeros((1, 14), dtype=np.float32) if actions is None else actions
        )
        self.reset_calls: list[object] = []
        self.infer_calls: list[dict[str, object]] = []
        self.close_calls = 0

    def reset(self, reset_info: Mapping[str, Any] | None = None) -> None:
        self.reset_calls.append(reset_info)

    def infer(self, new_obs: Mapping[str, Any]) -> dict[str, object]:
        self.infer_calls.append(dict(new_obs))
        return {
            "actions": self.actions,
            "policy_metadata": {"action_format": "joint_absolute"},
        }

    def close(self) -> None:
        self.close_calls += 1


class _Backend:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.reset_calls: list[dict[str, object]] = []
        self.infer_calls: list[dict[str, object]] = []

    def reset(self, *, task_id: str, prompt: str, seed: int, profile: str) -> None:
        self.reset_calls.append(
            {"task_id": task_id, "prompt": prompt, "seed": seed, "profile": profile}
        )

    def infer(
        self,
        *,
        images: Mapping[str, ImageArray],
        current_qpos14: FloatArray,
        tactile_images: Mapping[str, ImageArray] | None,
        wrench: Mapping[str, FloatArray] | None,
    ) -> np.ndarray:
        self.infer_calls.append(
            {
                "images": images,
                "current_qpos14": current_qpos14,
                "tactile_images": tactile_images,
                "wrench": wrench,
            }
        )
        return self.actions.copy()


def _route(
    *,
    tactile: tuple[str, ...] = (
        "observation.images.tactile_l",
        "observation.images.tactile_r",
    ),
    wrench: tuple[str, ...] = (
        "observation.wrench.left",
        "observation.wrench.right",
    ),
) -> AgileXTaskRoute:
    return AgileXTaskRoute(
        task_id="fold",
        prompt="fold the towel",
        tactile_required=bool(tactile),
        wrench_required=bool(wrench),
        tactile_keys=tactile,
        wrench_keys=wrench,
        contract_sha256="1" * 64,
    )


def _config(
    profile: str = "vision_tactile",
    *,
    route: AgileXTaskRoute | None = None,
) -> AgileXPolicyConfig:
    selected = route or _route()
    return AgileXPolicyConfig(
        policy_id="official-wire-test",
        tactile_profile=profile,
        task_routes={selected.task_id: selected},
        episode_seed=20260811,
        max_chunk_actions=12,
        safety=AgileXSafetyContract(
            lower_bounds=(-2.0,) * 14,
            upper_bounds=(2.0,) * 14,
            max_step_per_second=(1.0,) * 14,
            min_execution_dt_s=0.02,
            max_execution_dt_s=0.2,
            max_state_age_s=0.5,
            max_inference_latency_s=1.0,
            contract_sha256="2" * 64,
        ),
    )


def _images() -> dict[str, ImageArray]:
    return {
        "cam_high": np.zeros((8, 10, 3), dtype=np.uint8),
        "cam_wrist_left": np.ones((8, 10, 3), dtype=np.uint8),
        "cam_wrist_right": np.full((8, 10, 3), 2, dtype=np.uint8),
    }


def _contact() -> dict[str, dict[str, ImageArray | FloatArray]]:
    return {
        "left_gripper": {
            "rectify": np.zeros((6, 7, 3), dtype=np.uint8),
        },
        "right_gripper": {
            "rectify": np.ones((6, 7, 3), dtype=np.uint8),
        },
        "left_wrist_force": {
            "wrench_6d": np.zeros((6,), dtype=np.float32),
        },
        "right_wrist_force": {
            "wrench_6d": np.ones((6,), dtype=np.float32),
        },
    }


def _observation(*, tactile: bool = True) -> dict[str, Any]:
    qpos = np.linspace(-0.1, 0.1, 14, dtype=np.float32)
    result: dict[str, Any] = {
        "images": _images(),
        "joint_qpos": qpos,
        "state": qpos.copy(),
        "right_arm_joint_state": qpos[7:].copy(),
        "left_arm_joint_state": qpos[:7].copy(),
        "prompt": "fold the towel",
        "task_id": "fold",
        "tactile_profile": "tactile_raw",
    }
    if tactile:
        result["tactile"] = _contact()
    return result


def test_exact_official_wire_maps_to_canonical_internal_contract() -> None:
    config = _config()
    core = _Core(config)
    policy = Policy(core=core, config=config, wall_clock=lambda: 20.0)

    result = policy.infer(_observation())

    actions = np.asarray(result["actions"])
    assert actions.shape == (1, 14)
    assert actions.dtype == np.float32
    call = core.infer_calls[0]
    assert set(cast(Mapping[str, object], call["images"])) == {
        "top",
        "wrist_l",
        "wrist_r",
    }
    assert set(cast(Mapping[str, object], call["tactile"])) == {
        "observation.images.tactile_l",
        "observation.images.tactile_r",
    }
    assert set(cast(Mapping[str, object], call["wrench"])) == {
        "observation.wrench.left",
        "observation.wrench.right",
    }
    assert call["timestamp"] == 20.0
    assert call["execution_dt_s"] == 0.02


def test_official_boundary_rejects_multi_action_core_drift() -> None:
    config = _config()
    policy = Policy(
        core=_Core(config, np.zeros((2, 14), dtype=np.float32)),
        config=config,
        wall_clock=lambda: 20.0,
    )

    with pytest.raises(ValueError, match=r"float32\[1,14\]"):
        policy.infer(_observation())


def test_optional_time_fields_are_forwarded_and_reset_is_transparent() -> None:
    config = _config()
    core = _Core(config)
    policy = Policy(core=core)
    policy.reset({"task_id": "fold", "prompt": "fold the towel", "ignored": 1})
    observation = _observation()
    observation["timestamp"] = 7.5
    observation["execution_dt_s"] = 0.1

    policy.infer(observation)

    assert core.reset_calls == [{"task_id": "fold", "prompt": "fold the towel"}]
    assert core.infer_calls[0]["timestamp"] == 7.5
    assert core.infer_calls[0]["execution_dt_s"] == 0.1


def test_vision_only_extra_contact_is_validated_mapped_then_core_drops_it() -> None:
    route = _route(tactile=(), wrench=())
    config = _config("vision_only", route=route)
    backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    core = AgileXPolicy(
        config,
        backend=backend,
        wall_clock=lambda: 30.0,
        monotonic_clock=lambda: 1.0,
    )
    policy = Policy(core=core, config=config, wall_clock=lambda: 30.0)

    observation = _observation()
    observation["tactile_profile"] = "vision_only"
    result = policy.infer(observation)

    metadata = cast(Mapping[str, object], result["policy_metadata"])
    assert metadata["dropped_modalities"] == [
        "tactile",
        "wrench",
    ]
    assert backend.infer_calls[0]["tactile_images"] is None
    assert backend.infer_calls[0]["wrench"] is None
    assert backend.reset_calls[0]["task_id"] == "fold"


def test_state_must_match_joint_qpos() -> None:
    config = _config()
    core = _Core(config)
    policy = Policy(core=core, wall_clock=lambda: 20.0)
    observation = _observation()
    observation["state"] = np.ones((14,), dtype=np.float32)

    with pytest.raises(ValueError, match="state and joint_qpos"):
        policy.infer(observation)
    assert core.infer_calls == []


def test_state_is_required_by_the_official_wire_contract() -> None:
    config = _config()
    core = _Core(config)
    observation = _observation()
    del observation["state"]

    with pytest.raises(ValueError, match="state must be finite"):
        Policy(core=core, wall_clock=lambda: 20.0).infer(observation)
    assert core.infer_calls == []


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "left_arm_joint_state",
            np.ones((7,), dtype=np.float32),
            "left_arm_joint_state differs",
        ),
        (
            "right_arm_joint_state",
            np.ones((7,), dtype=np.float32),
            "right_arm_joint_state differs",
        ),
    ),
)
def test_split_arm_states_must_match_joint_qpos(
    field: str, replacement: npt.NDArray[np.float32], message: str
) -> None:
    config = _config()
    core = _Core(config)
    observation = _observation()
    observation[field] = replacement
    with pytest.raises(ValueError, match=message):
        Policy(core=core, wall_clock=lambda: 20.0).infer(observation)
    assert core.infer_calls == []


@pytest.mark.parametrize("profile", (None, "", "   ", 7))
def test_wire_tactile_profile_must_be_a_nonempty_string(profile: object) -> None:
    config = _config()
    core = _Core(config)
    observation = _observation()
    observation["tactile_profile"] = profile
    with pytest.raises(ValueError, match="tactile_profile"):
        Policy(core=core, wall_clock=lambda: 20.0).infer(observation)
    assert core.infer_calls == []


def test_vision_only_accepts_opaque_nonempty_profile_label_with_extra_contact() -> None:
    route = _route(tactile=(), wrench=())
    config = _config("vision_only", route=route)
    core = _Core(config)
    observation = _observation()
    observation["tactile_profile"] = "organizer_defined_profile_v7"
    Policy(core=core, wall_clock=lambda: 20.0).infer(observation)
    assert "tactile" in core.infer_calls[0]
    assert "wrench" in core.infer_calls[0]


def test_contact_presence_is_governed_by_signed_task_route_not_profile_label() -> None:
    no_contact = _route(tactile=(), wrench=())
    config = _config("mixed", route=no_contact)
    backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    core = AgileXPolicy(
        config,
        backend=backend,
        wall_clock=lambda: 20.0,
        monotonic_clock=lambda: 1.0,
    )
    observation = _observation()
    observation["tactile_profile"] = "tactile_raw"
    with pytest.raises(ValueError, match="forbids tactile"):
        Policy(core=core, wall_clock=lambda: 20.0).infer(observation)
    assert backend.infer_calls == []

    contact_config = _config()
    contact_backend = _Backend(np.zeros((12, 14), dtype=np.float32))
    contact_core = AgileXPolicy(
        contact_config,
        backend=contact_backend,
        wall_clock=lambda: 20.0,
        monotonic_clock=lambda: 1.0,
    )
    missing = _observation(tactile=False)
    missing["tactile_profile"] = "organizer_defined_profile_v7"
    with pytest.raises(ValueError, match="requires tactile"):
        Policy(core=contact_core, wall_clock=lambda: 20.0).infer(missing)
    assert contact_backend.infer_calls == []


def test_unknown_internal_route_fails_during_adapter_startup() -> None:
    route = _route(
        tactile=("observation.images.vendor_specific",),
        wrench=(),
    )
    config = _config("mixed", route=route)
    with pytest.raises(ValueError, match="unsupported"):
        Policy(core=_Core(config))


def test_default_constructor_requires_policy_config_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(POLICY_CONFIG_ENV, raising=False)
    with pytest.raises(ValueError, match=POLICY_CONFIG_ENV):
        Policy()


def test_close_and_context_manager_cleanup_are_idempotent() -> None:
    config = _config()
    core = _Core(config)
    with Policy(core=core) as policy:
        assert policy is not None
    policy.close()
    assert core.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        policy.reset()


def test_core_and_injected_config_must_match() -> None:
    config = _config()
    changed = replace(config, episode_seed=config.episode_seed + 1)
    with pytest.raises(ValueError, match="differ"):
        Policy(core=_Core(config), config=changed)
