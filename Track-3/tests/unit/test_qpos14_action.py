# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import numpy as np
import pytest

from n0_twam.actions import ActionCodec, build_action_codec


def _codec_with_bounds() -> ActionCodec:
    return build_action_codec(
        "qpos14_joint_absolute_v1",
        q01=(-2.0,) * 14,
        q99=(2.0,) * 14,
        lower_bounds=(-1.0,) * 14,
        upper_bounds=(1.0,) * 14,
    )


def test_qpos14_schema_preserves_official_dual_arm_channel_order() -> None:
    codec = _codec_with_bounds()

    assert codec.spec.name == "qpos14_joint_absolute_v1"
    assert codec.spec.revision == "1"
    assert codec.spec.dim == 14
    assert codec.spec.state_dim == 14
    assert codec.spec.semantics == "joint_absolute"
    assert codec.spec.normalization == "q01q99"
    assert codec.spec.padding_policy == "repeat_last_with_mask"
    assert codec.spec.wire_layout == "HC"
    assert codec.spec.server_output_format == "absolute"
    assert codec.spec.active_channel_ids == tuple(range(14))
    assert codec.spec.gripper_indices == (6, 13)
    assert codec.spec.channel_names == (
        "left_joint1",
        "left_joint2",
        "left_joint3",
        "left_joint4",
        "left_joint5",
        "left_joint6",
        "left_gripper",
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "right_joint6",
        "right_gripper",
    )


def test_qpos14_quantile_and_layout_round_trips_preserve_values() -> None:
    codec = _codec_with_bounds()
    wire = np.arange(6 * 14, dtype=np.float32).reshape(6, 14) / 100.0

    normalized = codec.normalize(wire)
    model = codec.wire_to_model(wire, frame_count=2, slots_per_frame=3)

    assert model.shape == (1, 14, 2, 3, 1)
    np.testing.assert_allclose(codec.denormalize(normalized), wire, atol=2e-6)
    np.testing.assert_array_equal(codec.model_to_wire(model), wire)


def test_qpos14_validation_rejects_wrong_dim_nonfinite_and_bounds() -> None:
    codec = _codec_with_bounds()

    with pytest.raises(ValueError, match="last dimension must be 14"):
        codec.validate(np.zeros((2, 13), dtype=np.float32))

    nonfinite = np.zeros((2, 14), dtype=np.float32)
    nonfinite[0, 4] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        codec.validate(nonfinite)

    out_of_bounds = np.zeros((2, 14), dtype=np.float32)
    out_of_bounds[1, 13] = 1.01
    with pytest.raises(ValueError, match="physical bounds"):
        codec.validate(out_of_bounds, check_bounds=True)


def test_qpos14_quantile_and_bounds_vectors_must_have_fourteen_channels() -> None:
    with pytest.raises(ValueError, match=r"q01 must have shape \(14,\)"):
        build_action_codec(
            "qpos14_joint_absolute_v1",
            q01=(-1.0,) * 13,
            q99=(1.0,) * 14,
        )

    with pytest.raises(ValueError, match="physical bounds must contain 14"):
        build_action_codec(
            "qpos14_joint_absolute_v1",
            q01=(-1.0,) * 14,
            q99=(1.0,) * 14,
            lower_bounds=(-2.0,) * 13,
            upper_bounds=(2.0,) * 13,
        )
