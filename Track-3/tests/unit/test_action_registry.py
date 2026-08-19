# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import numpy as np
import pytest

from n0_twam.actions import build_action_codec, list_action_codecs


def _stats(dim: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return ((-1.0,) * dim, (1.0,) * dim)


def test_builtin_action_codecs_are_registered_explicitly() -> None:
    assert list_action_codecs() == (
        "ee20_absee",
        "ee20_pi05",
        "qpos14_joint_absolute_v1",
        "qpos8_next_step",
    )


@pytest.mark.parametrize(
    ("name", "semantics"),
    (
        ("ee20_absee", "absolute_end_effector"),
        ("ee20_pi05", "delta_end_effector"),
    ),
)
def test_legacy_ee20_specs_remain_twenty_dimensional(
    name: str,
    semantics: str,
) -> None:
    q01, q99 = _stats(20)
    codec = build_action_codec(name, q01=q01, q99=q99)

    assert codec.spec.dim == 20
    assert codec.spec.state_dim == 20
    assert codec.spec.semantics == semantics
    assert codec.spec.gripper_indices == (9, 19)


def test_unknown_action_codec_fails_closed() -> None:
    with pytest.raises(KeyError, match="Unknown action codec"):
        build_action_codec("act")


def test_quantile_round_trip_preserves_values() -> None:
    q01 = (0.0,) * 8
    q99 = (2.0,) * 8
    codec = build_action_codec("qpos8_next_step", q01=q01, q99=q99)
    values = np.stack(
        (
            np.zeros(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
            np.full(8, 2.0, dtype=np.float32),
        )
    )

    normalized = codec.normalize(values)

    np.testing.assert_allclose(normalized[0], -1.0)
    np.testing.assert_allclose(normalized[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized[2], 1.0, atol=1e-6)
    np.testing.assert_allclose(codec.denormalize(normalized), values, atol=1e-6)


def test_invalid_quantile_span_is_rejected() -> None:
    with pytest.raises(ValueError, match="q99 must be greater than q01"):
        build_action_codec(
            "qpos8_next_step",
            q01=(0.0,) * 8,
            q99=(-1.0,) * 8,
        )


def test_constant_legacy_quantile_channel_is_supported() -> None:
    codec = build_action_codec(
        "qpos8_next_step",
        q01=(0.0,) * 8,
        q99=(0.0,) * 8,
    )

    normalized = codec.normalize(np.zeros((1, 8), dtype=np.float32))

    np.testing.assert_array_equal(normalized, -np.ones((1, 8), dtype=np.float32))
    np.testing.assert_array_equal(codec.denormalize(normalized), 0.0)


def test_action_core_does_not_import_simulator_stack() -> None:
    import sys

    forbidden = ("isaaclab", "tacex", "curobo", "UniVTAC")
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in sys.modules
        for prefix in forbidden
    )
