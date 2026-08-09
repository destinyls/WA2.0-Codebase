# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from types import SimpleNamespace

import pytest

from n0_twam.actions import build_action_codec_from_config
from n0_twam.actions.config import resolve_action_codec_name


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("none", "ee20_absee"),
        ("", "ee20_absee"),
        ("pi05_delta", "ee20_pi05"),
        ("openpi_delta", "ee20_pi05"),
        ("pi0.5_delta", "ee20_pi05"),
    ),
)
def test_legacy_config_resolution_preserves_existing_modes(
    mode: str,
    expected: str,
) -> None:
    config = SimpleNamespace(action_delta_mode=mode, action_dim=20)

    assert resolve_action_codec_name(config) == expected


def test_action_dim_eight_does_not_implicitly_select_qpos8() -> None:
    config = SimpleNamespace(action_delta_mode="none", action_dim=8)

    assert resolve_action_codec_name(config) == "ee20_absee"


def test_qpos8_requires_explicit_schema_opt_in() -> None:
    config = SimpleNamespace(
        action_schema="qpos8_next_step",
        action_delta_mode="none",
        action_dim=8,
    )

    assert resolve_action_codec_name(config) == "qpos8_next_step"


def test_config_builder_uses_recorded_normalizer_and_bounds() -> None:
    config = SimpleNamespace(
        action_schema="qpos8_next_step",
        action_dim=8,
        norm_stat={"q01": [-2.0] * 8, "q99": [2.0] * 8},
        action_lower_bounds=[-1.0] * 8,
        action_upper_bounds=[1.0] * 8,
    )

    codec = build_action_codec_from_config(config)

    assert codec.spec.name == "qpos8_next_step"
    assert codec.spec.lower_bounds == (-1.0,) * 8
    assert codec.spec.upper_bounds == (1.0,) * 8


def test_explicit_unknown_schema_fails_without_legacy_fallback() -> None:
    config = SimpleNamespace(
        action_schema="act",
        action_dim=8,
        norm_stat={"q01": [-1.0] * 8, "q99": [1.0] * 8},
    )

    with pytest.raises(KeyError, match="Unknown action codec"):
        build_action_codec_from_config(config)


def test_config_action_dim_must_match_explicit_schema() -> None:
    config = SimpleNamespace(
        action_schema="qpos8_next_step",
        action_dim=20,
        norm_stat={"q01": [-1.0] * 8, "q99": [1.0] * 8},
    )

    with pytest.raises(ValueError, match="action_dim=20 does not match"):
        build_action_codec_from_config(config)
