# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from dataclasses import FrozenInstanceError

import pytest

from n0_twam.actions.spec import ActionSpec


def _spec(**overrides: object) -> ActionSpec:
    values = {
        "name": "test",
        "revision": "1",
        "dim": 2,
        "state_dim": 2,
        "semantics": "absolute",
        "normalization": "q01q99",
        "padding_policy": "repeat_last_with_mask",
        "wire_layout": "FHC",
        "server_output_format": "absolute",
        "channel_names": ("a", "b"),
        "active_channel_ids": (0, 1),
        "gripper_indices": (1,),
        "lower_bounds": (-1.0, -1.0),
        "upper_bounds": (1.0, 1.0),
    }
    values.update(overrides)
    return ActionSpec(**values)


def test_action_spec_is_frozen() -> None:
    spec = _spec()

    with pytest.raises(FrozenInstanceError):
        spec.dim = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    (
        {"channel_names": ("a",)},
        {"channel_names": ("a", "a")},
        {"active_channel_ids": (0, 2)},
        {"gripper_indices": (2,)},
        {"lower_bounds": (-1.0,), "upper_bounds": (1.0,)},
        {"lower_bounds": (1.0, -1.0), "upper_bounds": (0.0, 1.0)},
    ),
)
def test_action_spec_rejects_invalid_contracts(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _spec(**overrides)
