# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""AgileX route and wrench injection at the production server boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from n0_twam.integrations.worldarena.agilex_manifest import canonical_sha256
from n0_twam.n0_twam_server import TWAM_Server


def _server(*, profile: str = "mixed") -> TWAM_Server:
    server = object.__new__(TWAM_Server)
    server.job_config = SimpleNamespace(
        tactile_profile=profile,
        tactile_keys=["touch_l", "touch_r"],
        wrench_keys=["force_l", "force_r"],
        wrench_sensor_id_map={"force_l": 0, "force_r": 1},
        wrench_arm_count=2,
    )
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.use_cfg = False
    server.prompt_embeds = torch.zeros(1, 2, 4)
    server.action_mask = torch.ones(14, dtype=torch.bool)
    return server


def _signed_server() -> tuple[TWAM_Server, dict[str, object]]:
    server = _server(profile="vision_tactile")
    route_core: dict[str, object] = {
        "task_id": "wipe",
        "prompt": "wipe the table",
        "tactile_required": True,
        "wrench_required": True,
        "tactile_keys": ["touch_l", "touch_r"],
        "wrench_keys": ["force_l", "force_r"],
    }
    route = {**route_core, "contract_sha256": canonical_sha256(route_core)}
    routes = {"wipe": route}
    server.job_config.require_signed_task_route = True
    server.job_config.signed_task_routes = routes
    server.job_config.signed_task_route_contract_sha256 = canonical_sha256(routes)
    observation = {
        "task_id": "wipe",
        "prompt": "wipe the table",
        "tactile_profile": "vision_tactile",
        "tactile_keys": ["touch_l", "touch_r"],
        "wrench_keys": ["force_l", "force_r"],
    }
    return server, observation


def test_mixed_route_can_bind_a_contact_free_task() -> None:
    server = _server()
    server._bind_active_contact_route({"tactile_keys": [], "wrench_keys": []})

    assert server._active_contact_keys("tactile") == ()
    assert server._active_contact_keys("wrench") == ()
    assert server._build_wrench_condition({}) is None


def test_wrench_is_mapped_to_signed_sensor_ids_and_action_input() -> None:
    server = _server(profile="vision_tactile")
    server._bind_active_contact_route(
        {
            "tactile_keys": ["touch_l", "touch_r"],
            "wrench_keys": ["force_l", "force_r"],
        }
    )
    left = np.arange(6, dtype=np.float32)
    right = np.arange(6, dtype=np.float32) + 10.0
    condition = server._build_wrench_condition(
        {
            "wrench": {"force_l": left, "force_r": right},
            "wrench_available_mask": {"force_l": True, "force_r": True},
        }
    )
    assert condition is not None
    assert condition["wrench"].shape == (1, 1, 2, 6)
    assert torch.equal(condition["wrench"][0, 0, 0], torch.from_numpy(left))
    assert torch.equal(condition["wrench"][0, 0, 1], torch.from_numpy(right))
    assert condition["wrench_available_mask"].all()

    prepared = server._prepare_latent_input(
        None,
        torch.zeros(1, 14, 1, 2, 1),
        tactile_latents={
            "tactile_global_latent": torch.zeros(1, 2, 48, 1, 1, 1),
            "tactile_local_latent": torch.zeros(1, 2, 48, 1, 1, 1),
            "tactile_sensor_ids": torch.tensor([[0, 1]]),
        },
        wrench_condition=condition,
    )["action_res_lst"]
    for key, expected in condition.items():
        assert torch.equal(prepared[key], expected)


def test_cfg_repeats_every_wrench_contract_tensor() -> None:
    server = _server(profile="vision_tactile")
    server.use_cfg = True
    server.negative_prompt_embeds = torch.ones(1, 2, 4)
    payload = {
        "noisy_latents": torch.zeros(1, 14, 1, 2, 1),
        "text_emb": torch.zeros(1, 2, 4),
        "grid_id": torch.zeros(1, 3),
        "timesteps": torch.zeros(1),
        "wrench": torch.zeros(1, 1, 2, 6),
        "wrench_available_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "temporal_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        "contact_cond_drop": torch.zeros(1, dtype=torch.bool),
    }

    repeated = server._repeat_input_for_cfg(payload)

    assert repeated["wrench"].shape == (2, 1, 2, 6)
    assert repeated["wrench_available_mask"].shape == (2, 1, 2)
    assert repeated["temporal_valid_mask"].shape == (2, 1)
    assert repeated["contact_cond_drop"].shape == (2,)


def test_route_and_wrench_validation_is_fail_closed() -> None:
    server = _server(profile="vision_only")
    with pytest.raises(ValueError, match="vision_only"):
        server._bind_active_contact_route(
            {"tactile_keys": ["touch_l"], "wrench_keys": []}
        )

    server = _server(profile="vision_tactile")
    server._bind_active_contact_route(
        {"tactile_keys": ["touch_l"], "wrench_keys": ["force_l"]}
    )
    with pytest.raises(ValueError, match=r"float32\[6\]"):
        server._build_wrench_condition(
            {
                "wrench": {"force_l": np.ones(6, dtype=np.float64)},
                "wrench_available_mask": {"force_l": True},
            }
        )


def test_contact_route_cannot_change_without_reset() -> None:
    server = _server()
    server._bind_active_contact_route(
        {"tactile_keys": ["touch_l"], "wrench_keys": ["force_l"]}
    )

    server._require_active_contact_route(
        {"tactile_keys": ["touch_l"], "wrench_keys": ["force_l"]}
    )
    with pytest.raises(ValueError, match="tactile task route changed"):
        server._require_active_contact_route(
            {"tactile_keys": ["touch_r"], "wrench_keys": ["force_l"]}
        )
    with pytest.raises(ValueError, match="provided together"):
        server._require_active_contact_route({"tactile_keys": ["touch_l"]})


def test_signed_task_route_is_bound_before_activation() -> None:
    server, observation = _signed_server()

    server._bind_active_contact_route(observation)

    assert server._active_contact_keys("tactile") == ("touch_l", "touch_r")
    assert server._active_contact_keys("wrench") == ("force_l", "force_r")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"prompt": "wrong"}, "task identity"),
        ({"tactile_profile": "mixed"}, "tactile profile"),
        ({"tactile_keys": ["touch_l"]}, "active contact keys"),
    ],
)
def test_signed_task_route_rejects_reset_identity_drift(
    mutation: dict[str, object],
    message: str,
) -> None:
    server, observation = _signed_server()
    observation.update(mutation)

    with pytest.raises(ValueError, match=message):
        server._bind_active_contact_route(observation)


def test_signed_task_route_rejects_contract_tampering() -> None:
    server, observation = _signed_server()
    server.job_config.signed_task_routes["wipe"]["prompt"] = "tampered"

    with pytest.raises(ValueError, match="aggregate identity"):
        server._bind_active_contact_route(observation)


def test_signed_task_route_requires_explicit_reset_keys() -> None:
    server, observation = _signed_server()
    observation.pop("tactile_keys")

    with pytest.raises(ValueError, match="explicit tactile_keys"):
        server._bind_active_contact_route(observation)
