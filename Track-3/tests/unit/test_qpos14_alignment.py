# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import numpy as np
import pytest

from n0_twam.actions import ActionCodec, build_action_codec
from n0_twam.data.qpos14_alignment import build_qpos14_latent_targets


def _identity_like_codec() -> ActionCodec:
    return build_action_codec(
        "qpos14_joint_absolute_v1",
        q01=(0.0,) * 14,
        q99=(2.0,) * 14,
    )


def test_explicit_action_map_is_consumed_without_an_implicit_shift() -> None:
    actions = np.repeat(np.arange(6, dtype=np.float32)[:, None], 14, axis=1)

    targets = build_qpos14_latent_targets(
        converted_actions=actions,
        codec=_identity_like_codec(),
        action_indices_per_anchor=np.asarray(((0, 2, 3), (4, 5, 5))),
        action_valid_mask=np.asarray(((True, True, True), (True, True, False))),
        latent_anchor_row_ids=np.asarray((10, 17)),
    )

    assert targets.actions.shape == (14, 2, 3, 1)
    assert targets.action_valid_mask.shape == (2, 3)
    assert targets.model_valid_mask.shape == targets.actions.shape
    np.testing.assert_allclose(
        targets.actions[0, 0, :, 0],
        np.asarray((-1.0, 1.0, 2.0), dtype=np.float32),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        targets.actions[0, 1, :, 0],
        np.asarray((3.0, 4.0, 0.0), dtype=np.float32),
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        targets.action_valid_mask,
        np.asarray(((True, True, True), (True, True, False))),
    )
    assert targets.model_valid_mask[:, 0, :, :].all()
    assert targets.model_valid_mask[:, 1, :2, :].all()
    assert not targets.model_valid_mask[:, 1, 2:, :].any()
    assert targets.slots_per_frame == 3


def test_alignment_accepts_nonuniform_indices_and_local_origin() -> None:
    local_actions = np.repeat(
        np.arange(10, 16, dtype=np.float32)[:, None],
        14,
        axis=1,
    )

    targets = build_qpos14_latent_targets(
        converted_actions=local_actions,
        codec=build_action_codec(
            "qpos14_joint_absolute_v1",
            q01=(0.0,) * 14,
            q99=(20.0,) * 14,
        ),
        action_indices_per_anchor=np.asarray(((10, 12), (13, 15))),
        action_valid_mask=np.ones((2, 2), dtype=np.bool_),
        latent_anchor_row_ids=np.asarray((4, 11)),
        action_index_origin=10,
    )

    np.testing.assert_array_equal(
        targets.action_indices_per_anchor,
        np.asarray(((10, 12), (13, 15))),
    )
    np.testing.assert_allclose(
        targets.actions[0, :, :, 0],
        np.asarray(((0.0, 0.2), (0.3, 0.5)), dtype=np.float32),
        atol=2e-6,
    )
    np.testing.assert_array_equal(targets.latent_anchor_row_ids, (4, 11))


@pytest.mark.parametrize(
    ("indices", "valid_mask", "error"),
    (
        (
            ((0, 1), (2, 3)),
            ((True, False), (True, True)),
            "only the terminal latent anchor",
        ),
        (
            ((0, 1), (2, 3)),
            ((False, True), (True, True)),
            "valid prefix",
        ),
        (
            ((0, 1), (2, 3)),
            ((True, True), (False, False)),
            "at least one valid action",
        ),
        (
            ((0, 1), (2, 3)),
            ((True, True), (True, False)),
            "repeat the last valid action index",
        ),
        (
            ((0, 0), (2, 3)),
            ((True, True), (True, True)),
            "strictly increasing",
        ),
    ),
)
def test_alignment_rejects_invalid_validity_or_padding_contract(
    indices: tuple[tuple[int, ...], ...],
    valid_mask: tuple[tuple[bool, ...], ...],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        build_qpos14_latent_targets(
            converted_actions=np.zeros((4, 14), dtype=np.float32),
            codec=_identity_like_codec(),
            action_indices_per_anchor=np.asarray(indices),
            action_valid_mask=np.asarray(valid_mask),
            latent_anchor_row_ids=np.asarray((0, 4)),
        )


def test_alignment_rejects_schema_shape_index_and_anchor_mismatches() -> None:
    actions = np.zeros((4, 14), dtype=np.float32)
    kwargs = {
        "converted_actions": actions,
        "codec": _identity_like_codec(),
        "action_indices_per_anchor": np.asarray(((0, 1), (2, 3))),
        "action_valid_mask": np.ones((2, 2), dtype=np.bool_),
        "latent_anchor_row_ids": np.asarray((0, 4)),
    }

    qpos8 = build_action_codec(
        "qpos8_next_step",
        q01=(0.0,) * 8,
        q99=(1.0,) * 8,
    )
    with pytest.raises(ValueError, match="qpos14_joint_absolute_v1 codec"):
        build_qpos14_latent_targets(**{**kwargs, "codec": qpos8})
    with pytest.raises(ValueError, match=r"shape \[timesteps, 14\]"):
        build_qpos14_latent_targets(
            **{**kwargs, "converted_actions": np.zeros((4, 13), dtype=np.float32)}
        )
    with pytest.raises(ValueError, match="within converted_actions"):
        build_qpos14_latent_targets(
            **{
                **kwargs,
                "action_indices_per_anchor": np.asarray(((0, 1), (2, 4))),
            }
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        build_qpos14_latent_targets(
            **{**kwargs, "latent_anchor_row_ids": np.asarray((4, 4))}
        )
    with pytest.raises(ValueError, match="boolean dtype"):
        build_qpos14_latent_targets(
            **{**kwargs, "action_valid_mask": np.ones((2, 2), dtype=np.int64)}
        )
