# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import numpy as np
import pytest

from n0_twam.actions import build_action_codec
from n0_twam.actions.qpos8 import build_next_step_windows


def _codec_with_bounds():
    return build_action_codec(
        "qpos8_next_step",
        q01=(-2.0,) * 8,
        q99=(2.0,) * 8,
        lower_bounds=(-1.0,) * 8,
        upper_bounds=(1.0,) * 8,
    )


def test_qpos8_schema_is_native_joint_space() -> None:
    codec = _codec_with_bounds()

    assert codec.spec.dim == 8
    assert codec.spec.state_dim == 8
    assert codec.spec.semantics == "joint_absolute_next_step"
    assert codec.spec.normalization == "q01q99"
    assert codec.spec.padding_policy == "repeat_last_with_mask"
    assert codec.spec.wire_layout == "HC"
    assert codec.spec.server_output_format == "absolute"
    assert codec.spec.gripper_indices == (7,)
    assert codec.spec.channel_names == (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
    )


def test_next_step_windows_have_no_double_shift() -> None:
    joint = np.arange(5 * 9, dtype=np.float32).reshape(5, 9)

    windows = build_next_step_windows(joint, horizon=3)

    np.testing.assert_array_equal(windows.states, joint[:-1, :8])
    np.testing.assert_array_equal(windows.actions[0, 0], joint[1, :8])
    np.testing.assert_array_equal(windows.actions[0, 1], joint[2, :8])
    np.testing.assert_array_equal(windows.actions[0, 2], joint[3, :8])
    np.testing.assert_array_equal(windows.actions[3, 0], joint[4, :8])
    np.testing.assert_array_equal(windows.actions[3, 1], joint[4, :8])
    np.testing.assert_array_equal(
        windows.is_pad,
        np.asarray(
            (
                (False, False, False),
                (False, False, False),
                (False, False, True),
                (False, True, True),
            )
        ),
    )


def test_next_step_windows_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least two timesteps"):
        build_next_step_windows(np.zeros((1, 9), dtype=np.float32), horizon=3)
    with pytest.raises(ValueError, match="at least 8 channels"):
        build_next_step_windows(np.zeros((3, 7), dtype=np.float32), horizon=3)
    with pytest.raises(ValueError, match="horizon must be positive"):
        build_next_step_windows(np.zeros((3, 9), dtype=np.float32), horizon=0)


def test_wire_and_model_layout_round_trip() -> None:
    codec = _codec_with_bounds()
    wire = np.arange(2 * 3 * 8, dtype=np.float32).reshape(6, 8) / 100.0

    model = codec.wire_to_model(wire, frame_count=2, slots_per_frame=3)

    assert model.shape == (1, 8, 2, 3, 1)
    np.testing.assert_array_equal(codec.model_to_wire(model), wire)


def test_validation_rejects_wrong_dim_nonfinite_and_out_of_bounds() -> None:
    codec = _codec_with_bounds()

    with pytest.raises(ValueError, match="last dimension must be 8"):
        codec.validate(np.zeros((2, 7), dtype=np.float32))

    nonfinite = np.zeros((2, 8), dtype=np.float32)
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        codec.validate(nonfinite)

    out_of_bounds = np.zeros((2, 8), dtype=np.float32)
    out_of_bounds[0, 3] = 1.1
    with pytest.raises(ValueError, match="physical bounds"):
        codec.validate(out_of_bounds, check_bounds=True)
