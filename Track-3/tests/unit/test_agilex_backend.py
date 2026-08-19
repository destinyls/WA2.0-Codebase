# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Single-rank full-replan backend for AgileX qpos14 Policy."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch

from n0_twam.integrations.worldarena.agilex_backend import (
    DirectN0AgileXBackend,
    _configure_environment,
    _sealed_task_routes,
)
from n0_twam.integrations.worldarena.agilex_manifest import canonical_sha256
from n0_twam.integrations.worldarena.agilex_policy_contracts import (
    AgileXPolicyConfig,
    AgileXSafetyContract,
    AgileXTaskRoute,
)


class _RuntimeConfig:
    def __init__(
        self,
        profile: str = "vision_tactile",
        *,
        contact: bool | None = None,
    ) -> None:
        contact = profile != "vision_only" if contact is None else contact
        route = AgileXTaskRoute(
            task_id="wipe",
            prompt="wipe the table",
            tactile_required=contact,
            wrench_required=contact,
            tactile_keys=(
                (
                    "observation.images.tactile_l",
                    "observation.images.tactile_r",
                )
                if contact
                else ()
            ),
            wrench_keys=(
                (
                    "observation.wrench.left",
                    "observation.wrench.right",
                )
                if contact
                else ()
            ),
            contract_sha256="1" * 64,
        )
        self.policy = AgileXPolicyConfig(
            policy_id="test",
            tactile_profile=profile,
            task_routes={"wipe": route},
            episode_seed=1,
            max_chunk_actions=12,
            safety=AgileXSafetyContract(
                lower_bounds=(-2.0,) * 14,
                upper_bounds=(2.0,) * 14,
                max_step_per_second=(1.0,) * 14,
                min_execution_dt_s=0.01,
                max_execution_dt_s=0.2,
                max_state_age_s=1.0,
                max_inference_latency_s=1.0,
                contract_sha256="2" * 64,
            ),
        )
        self.serve_bundle = Path("/tmp/agilex-bundle")
        self.serve_output = Path("/tmp/agilex-output")
        self.cuda_visible_device = "0"
        self.distributed_port = 29643
        self.video_inference_steps = 3
        self.action_inference_steps = 4
        self.repo_route_manifest_sha256 = "3" * 64
        self.repo_route_manifest_file_sha256 = "4" * 64
        self.normalizer_file_sha256 = "5" * 64
        self.normalizer_contract_sha256 = "6" * 64
        self.contact_profile_contract_sha256 = "7" * 64
        self.source_path = Path("/tmp/agilex-policy.json")
        self.policy_config_file_sha256 = "8" * 64
        route_payload = {
            "wipe": {
                "task_id": route.task_id,
                "prompt": route.prompt,
                "tactile_required": route.tactile_required,
                "wrench_required": route.wrench_required,
                "tactile_keys": list(route.tactile_keys),
                "wrench_keys": list(route.wrench_keys),
                "contract_sha256": route.contract_sha256,
            }
        }
        self.task_routes_sha256 = canonical_sha256(route_payload)


class _Server:
    def __init__(self, action: object) -> None:
        self.action = action
        self.calls: list[dict[str, object]] = []

    def infer(self, observation: dict[str, object]) -> dict[str, object]:
        self.calls.append(observation)
        if observation.get("reset"):
            return {}
        return {"action": self.action}

    def _infer(self, observation: dict[str, object], frame_st_id: int):
        self.calls.append(observation)
        assert frame_st_id == 0
        return self.action, torch.zeros((1, 48, 2, 1, 3))


def _images() -> dict[str, npt.NDArray[np.uint8]]:
    return {
        "top": np.zeros((8, 9, 3), dtype=np.uint8),
        "wrist_l": np.ones((8, 9, 3), dtype=np.uint8),
        "wrist_r": np.full((8, 9, 3), 2, dtype=np.uint8),
    }


def _touch() -> dict[str, npt.NDArray[np.uint8]]:
    return {
        "observation.images.tactile_l": np.zeros((6, 7, 3), dtype=np.uint8),
        "observation.images.tactile_r": np.ones((6, 7, 3), dtype=np.uint8),
    }


def _wrench() -> dict[str, npt.NDArray[np.float32]]:
    return {
        "observation.wrench.left": np.zeros((6,), dtype=np.float32),
        "observation.wrench.right": np.ones((6,), dtype=np.float32),
    }


def test_backend_runs_one_cold_full_replan_and_flattens_qpos14() -> None:
    raw = np.arange(14 * 1 * 12, dtype=np.float32).reshape(14, 1, 12)
    server = _Server(raw)
    backend = DirectN0AgileXBackend(
        _RuntimeConfig(),  # type: ignore[arg-type]
        server_factory=lambda _: server,
        artifact_verifier=lambda _: {},
    )
    backend.reset(
        task_id="wipe", prompt="wipe the table", seed=7, profile="vision_tactile"
    )
    result = backend.infer(
        images=_images(),
        current_qpos14=np.zeros((14,), dtype=np.float32),
        tactile_images=_touch(),
        wrench=_wrench(),
    )

    assert result.dtype == np.float32
    assert result.shape == (12, 14)
    np.testing.assert_array_equal(result, raw.transpose(1, 2, 0).reshape(12, 14))
    assert server.calls[0] == {
        "reset": True,
        "task_id": "wipe",
        "prompt": "wipe the table",
        "seed": 7,
        "tactile_profile": "vision_tactile",
        "tactile_keys": [
            "observation.images.tactile_l",
            "observation.images.tactile_r",
        ],
        "wrench_keys": [
            "observation.wrench.left",
            "observation.wrench.right",
        ],
    }
    observation = server.calls[1]
    assert observation["full_replan"] is True
    assert observation["compute_kv_cache"] is False
    assert observation["tactile_keys"] == list(_touch())
    assert observation["wrench_keys"] == list(_wrench())
    obs_history = observation["obs"]
    assert isinstance(obs_history, list)
    assert isinstance(obs_history[0], Mapping)
    assert set(obs_history[0]) == {
        "observation.images.top",
        "observation.images.wrist_l",
        "observation.images.wrist_r",
    }
    tactile = observation["tactile"]
    wrench = observation["wrench"]
    assert isinstance(tactile, Mapping)
    assert isinstance(wrench, Mapping)
    assert set(tactile) == set(_touch())
    assert set(wrench) == set(_wrench())


def test_backend_exposes_only_the_signed_frame_major_action_prefix() -> None:
    raw = np.arange(14 * 2 * 12, dtype=np.float32).reshape(14, 2, 12)
    server = _Server(raw)
    backend = DirectN0AgileXBackend(
        _RuntimeConfig(),  # type: ignore[arg-type]
        server_factory=lambda _: server,
        artifact_verifier=lambda _: {},
    )
    backend.reset(
        task_id="wipe", prompt="wipe the table", seed=7, profile="vision_tactile"
    )

    actions = backend.infer(
        images=_images(),
        current_qpos14=np.zeros((14,), dtype=np.float32),
        tactile_images=_touch(),
        wrench=_wrench(),
    )

    expected = np.moveaxis(raw, 0, -1).reshape(-1, 14)[:12]
    assert actions.shape == (12, 14)
    np.testing.assert_array_equal(actions, expected)


def test_backend_offline_prediction_retains_joint_rgb_latent() -> None:
    raw = np.arange(14 * 1 * 12, dtype=np.float32).reshape(14, 1, 12)
    server = _Server(raw)
    backend = DirectN0AgileXBackend(
        _RuntimeConfig(),  # type: ignore[arg-type]
        server_factory=lambda _: server,
        artifact_verifier=lambda _: {},
    )
    backend.reset(
        task_id="wipe", prompt="wipe the table", seed=7, profile="vision_tactile"
    )

    actions, latent = backend.infer_prediction_latent_chunk(
        images=_images(),
        current_qpos14=np.zeros(14, dtype=np.float32),
        tactile_images=_touch(),
        wrench=_wrench(),
    )

    assert actions.shape == (12, 14)
    assert isinstance(latent, torch.Tensor)
    assert latent.shape == (1, 48, 2, 1, 3)


def test_backend_prediction_uses_the_same_signed_action_prefix() -> None:
    raw = np.arange(14 * 2 * 12, dtype=np.float32).reshape(14, 2, 12)
    server = _Server(raw)
    backend = DirectN0AgileXBackend(
        _RuntimeConfig(),  # type: ignore[arg-type]
        server_factory=lambda _: server,
        artifact_verifier=lambda _: {},
    )
    backend.reset(
        task_id="wipe", prompt="wipe the table", seed=7, profile="vision_tactile"
    )

    actions, _ = backend.infer_prediction_latent_chunk(
        images=_images(),
        current_qpos14=np.zeros(14, dtype=np.float32),
        tactile_images=_touch(),
        wrench=_wrench(),
    )

    expected = np.moveaxis(raw, 0, -1).reshape(-1, 14)[:12]
    assert actions.shape == (12, 14)
    np.testing.assert_array_equal(actions, expected)


def test_injected_server_factory_does_not_mutate_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "caller-owned")
    monkeypatch.setenv("N0_TRACK3_AGILEX_SIGNED_TASK_ROUTE", "caller-route")
    server = _Server(np.zeros((14, 1, 1), dtype=np.float32))

    DirectN0AgileXBackend(
        _RuntimeConfig(),  # type: ignore[arg-type]
        server_factory=lambda _: server,
        artifact_verifier=lambda _: {},
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "caller-owned"
    assert os.environ["N0_TRACK3_AGILEX_SIGNED_TASK_ROUTE"] == "caller-route"


def test_production_environment_binds_the_sealed_policy_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_names = (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "N0_TRACK3_AGILEX_TACTILE_PROFILE",
        "N0_TRACK3_AGILEX_SERVE_BUNDLE",
        "N0_TRACK3_AGILEX_SERVE_OUTPUT",
        "N0_TRACK3_AGILEX_NORMALIZER",
        "N0_TRACK3_AGILEX_REPO_ROUTE_MANIFEST",
        "N0_TRACK3_AGILEX_VIDEO_STEPS",
        "N0_TRACK3_AGILEX_ACTION_STEPS",
        "N0_TRACK3_AGILEX_SIGNED_TASK_ROUTE",
        "N0_TRACK3_AGILEX_SIGNED_TASK_ROUTE_SHA256",
    )
    for name in configured_names:
        monkeypatch.setenv(name, "caller-owned")
    config = _RuntimeConfig()

    _configure_environment(config)  # type: ignore[arg-type]

    assert os.environ["N0_TRACK3_AGILEX_SIGNED_TASK_ROUTE"] == str(config.source_path)
    assert (
        os.environ["N0_TRACK3_AGILEX_SIGNED_TASK_ROUTE_SHA256"]
        == config.policy_config_file_sha256
    )
    assert set(_sealed_task_routes(config)) == {"wipe"}  # type: ignore[arg-type]


def test_sealed_task_routes_reject_aggregate_identity_drift() -> None:
    config = _RuntimeConfig()
    config.task_routes_sha256 = "f" * 64

    with pytest.raises(ValueError, match="sealed policy identity"):
        _sealed_task_routes(config)  # type: ignore[arg-type]


def test_backend_is_fail_closed_before_reset_and_on_route_mismatch() -> None:
    server = _Server(np.zeros((14, 2, 3), dtype=np.float32))
    backend = DirectN0AgileXBackend(
        _RuntimeConfig(),  # type: ignore[arg-type]
        server_factory=lambda _: server,
        artifact_verifier=lambda _: {},
    )
    with pytest.raises(RuntimeError, match="reset"):
        backend.infer(
            images=_images(),
            current_qpos14=np.zeros((14,), dtype=np.float32),
            tactile_images=_touch(),
            wrench=_wrench(),
        )
    with pytest.raises(ValueError, match="profile"):
        backend.reset(task_id="wipe", prompt="wipe the table", seed=1, profile="mixed")
    with pytest.raises(ValueError, match="prompt"):
        backend.reset(task_id="wipe", prompt="wrong", seed=1, profile="vision_tactile")


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "action",
    [
        np.zeros((13, 2, 3), dtype=np.float32),
        np.zeros((14, 0, 12), dtype=np.float32),
        np.zeros((14, 2, 0), dtype=np.float32),
        np.full((14, 2, 3), np.nan, dtype=np.float32),
    ],
)
def test_backend_rejects_invalid_server_output(
    action: npt.NDArray[np.float32],
) -> None:
    server = _Server(action)
    backend = DirectN0AgileXBackend(
        _RuntimeConfig(),  # type: ignore[arg-type]
        server_factory=lambda _: server,
        artifact_verifier=lambda _: {},
    )
    backend.reset(
        task_id="wipe", prompt="wipe the table", seed=1, profile="vision_tactile"
    )
    with pytest.raises(ValueError, match="qpos14 chunk"):
        backend.infer(
            images=_images(),
            current_qpos14=np.zeros((14,), dtype=np.float32),
            tactile_images=_touch(),
            wrench=_wrench(),
        )


def test_backend_revalidates_input_dtype_and_exact_modalities() -> None:
    server = _Server(np.zeros((14, 2, 3), dtype=np.float32))
    backend = DirectN0AgileXBackend(
        _RuntimeConfig(),  # type: ignore[arg-type]
        server_factory=lambda _: server,
        artifact_verifier=lambda _: {},
    )
    backend.reset(
        task_id="wipe", prompt="wipe the table", seed=1, profile="vision_tactile"
    )
    with pytest.raises(ValueError, match=r"float32\[14\]"):
        backend.infer(
            images=_images(),
            current_qpos14=np.zeros((14,), dtype=np.float64),
            tactile_images=_touch(),
            wrench=_wrench(),
        )
