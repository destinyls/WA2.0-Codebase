# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import numpy as np

from n0_twam.actions import build_action_codec
from n0_twam.actions.ee20 import absolute_to_pi05_delta, pi05_delta_to_absolute


def test_ee20_absee_matches_legacy_quantile_formula_and_layout() -> None:
    q01 = np.linspace(-2.0, -0.1, 20, dtype=np.float32)
    q99 = np.linspace(0.2, 3.0, 20, dtype=np.float32)
    data_grid = np.linspace(
        -0.05,
        0.15,
        2 * 3 * 20,
        dtype=np.float32,
    ).reshape(2, 3, 20)
    codec = build_action_codec(
        "ee20_absee",
        q01=tuple(float(value) for value in q01),
        q99=tuple(float(value) for value in q99),
    )

    normalized = codec.normalize(data_grid)
    expected = (data_grid - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    legacy_wire = np.transpose(normalized, (2, 0, 1))
    model = codec.wire_to_model(legacy_wire)

    np.testing.assert_allclose(normalized, expected, atol=1e-6)
    assert model.shape == (1, 20, 2, 3, 1)
    np.testing.assert_allclose(
        codec.denormalize(np.transpose(codec.model_to_wire(model), (1, 2, 0))),
        data_grid,
        atol=2e-6,
    )


def test_ee20_pi05_per_frame_anchor_round_trip() -> None:
    current_state = np.zeros(20, dtype=np.float32)
    current_state[0] = 10.0
    absolute = np.zeros((2, 2, 20), dtype=np.float32)
    absolute[0, :, 0] = (11.0, 12.0)
    absolute[1, :, 0] = (20.0, 21.0)
    absolute[..., 9] = 0.25
    original = absolute.copy()

    delta = absolute_to_pi05_delta(absolute, current_state=current_state)

    np.testing.assert_array_equal(delta[0, :, 0], (1.0, 2.0))
    np.testing.assert_array_equal(delta[1, :, 0], (8.0, 9.0))
    np.testing.assert_array_equal(delta[..., 9], 0.25)
    np.testing.assert_array_equal(
        pi05_delta_to_absolute(delta, current_state=current_state),
        absolute,
    )
    np.testing.assert_array_equal(absolute, original)
