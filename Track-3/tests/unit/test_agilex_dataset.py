# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""AgileX dataset alignment and canonical mixed-mask tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import n0_twam.dataset.lerobot_latent_dataset_agilex as agilex_dataset_module
from n0_twam.dataset.agilex_sample_contract import (
    build_agilex_sample_shape_signature,
)
from n0_twam.dataset.lerobot_latent_dataset_agilex import (
    MultiLatentLeRobotAgileXDataset,
    build_action_index_grid,
    canonicalize_agilex_sample,
    content_addressed_contact_drop,
)
from n0_twam.dataset.sample_shape_signature import SampleShapeSignature

LEFT_WRENCH_KEY = "observation.wrench.left"
RIGHT_WRENCH_KEY = "observation.wrench.right"


def _strict_dataset_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "dataset_adapter": "worldarena_agilex_qpos14",
        "action_schema": "qpos14_joint_absolute_v1",
        "action_dim": 14,
        "used_action_channel_ids": list(range(14)),
        "action_per_frame": 1,
        "per_repo_action_offsets_per_anchor": {
            "touch": [0],
            "right_only": [0],
        },
        "tactile_profile": "vision_tactile",
        "repo_route_manifest_sha256": "a" * 64,
        "temporal_alignment_contract_sha256": "b" * 64,
        "selected_repo_names": ["touch", "right_only"],
        "per_repo_route_identity": {
            "touch": "c" * 64,
            "right_only": "d" * 64,
        },
        "per_repo_temporal_alignment_identity": {
            "touch": "e" * 64,
            "right_only": "f" * 64,
        },
        "per_repo_wrench_keys": {
            "touch": [LEFT_WRENCH_KEY, RIGHT_WRENCH_KEY],
            "right_only": [RIGHT_WRENCH_KEY],
        },
        "wrench_sensor_id_map": {
            LEFT_WRENCH_KEY: 9,
            RIGHT_WRENCH_KEY: 4,
        },
        "wrench_arm_count": 10,
        "max_wrench_streams": 10,
        "norm_stat": {
            "q01": [-1.0] * 14,
            "q99": [1.0] * 14,
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _base_signature(*, sensors: int = 2) -> SampleShapeSignature:
    entries = [
        ("actions", (14, 3, 4, 1), "torch.float32"),
        ("actions_mask", (14, 3, 4, 1), "torch.bool"),
        ("latents", (48, 3, 4, 8), "torch.float32"),
        ("tactile_cond_drop", (), "torch.bool"),
        ("text_emb", (16, 32), "torch.float32"),
    ]
    if sensors:
        entries.extend(
            [
                (
                    "tactile_global_latent",
                    (sensors, 8, 3, 2, 2),
                    "torch.float32",
                ),
                (
                    "tactile_local_latent",
                    (sensors, 8, 3, 2, 2),
                    "torch.float32",
                ),
                ("tactile_sensor_ids", (sensors,), "torch.int64"),
            ]
        )
    return tuple(sorted(entries))


def test_action_grid_has_no_implicit_next_step_shift_and_terminal_is_masked() -> None:
    indices, valid = build_action_index_grid(
        latent_anchor_row_ids=np.asarray((0, 2, 4), dtype=np.int64),
        action_offsets_per_anchor=(0, 1),
        action_row_valid=np.asarray((True, True, True, True, True, False)),
        action_row_count=6,
    )

    np.testing.assert_array_equal(indices, ((0, 1), (2, 3), (4, 4)))
    np.testing.assert_array_equal(valid, ((True, True), (True, True), (True, False)))


def test_canonical_sample_separates_target_availability_and_condition_drop() -> None:
    frames, horizon, sensors = 2, 3, 2
    sample = {
        "actions": torch.ones(14, frames, horizon, 1),
        "actions_mask": torch.ones(14, frames, horizon, 1, dtype=torch.bool),
        "latents": torch.zeros(48, frames, 4, 4),
        "tactile_global_latent": torch.ones(sensors, 8, frames, 2, 2),
        "tactile_local_latent": torch.ones(sensors, 8, frames, 2, 2),
        "tactile_sensor_ids": torch.tensor((3, 4), dtype=torch.int64),
        "tactile_cond_drop": torch.tensor(False),
    }
    tactile_available = torch.tensor(((True, False), (True, True)))
    wrench = torch.arange(frames * 2 * 6, dtype=torch.float32).reshape(frames, 2, 6)
    wrench_available = torch.tensor(((True, False), (True, True)))

    result = canonicalize_agilex_sample(
        sample,
        tactile_available_mask=tactile_available,
        wrench=wrench,
        wrench_available_mask=wrench_available,
        contact_cond_drop=True,
        repo_route_identity="a" * 64,
        temporal_alignment_identity="b" * 64,
    )

    assert result["action_valid_mask"].shape == (frames, horizon)
    assert result["temporal_valid_mask"].tolist() == [True, True]
    assert result["tactile_available_mask"].equal(tactile_available)
    assert result["wrench_available_mask"].equal(wrench_available)
    assert result["contact_cond_drop"].item() is True
    assert "actions_mask" not in result
    assert "tactile_cond_drop" not in result
    # CFG drop removes conditioning downstream, not the real tactile target.
    assert torch.count_nonzero(result["tactile_global_latent"]) > 0
    # Availability masking prevents a placeholder wrench side channel.
    assert torch.count_nonzero(result["wrench"][0, 1]) == 0


def test_canonical_sample_rejects_channel_disagreement_in_legacy_mask() -> None:
    mask = torch.ones(14, 1, 2, 1, dtype=torch.bool)
    mask[13, 0, 1, 0] = False
    with pytest.raises(ValueError, match="channel-invariant"):
        canonicalize_agilex_sample(
            {
                "actions": torch.zeros(14, 1, 2, 1),
                "actions_mask": mask,
                "latents": torch.zeros(48, 1, 2, 2),
            },
            tactile_available_mask=torch.zeros((1, 0), dtype=torch.bool),
            wrench=torch.zeros((1, 2, 6)),
            wrench_available_mask=torch.zeros((1, 2), dtype=torch.bool),
            contact_cond_drop=True,
            repo_route_identity="a" * 64,
            temporal_alignment_identity="b" * 64,
        )


def test_canonical_sample_rejects_coerced_mask_and_drop_types() -> None:
    sample = {
        "actions": torch.zeros(14, 1, 1, 1),
        "actions_mask": torch.ones(14, 1, 1, 1, dtype=torch.bool),
        "latents": torch.zeros(48, 1, 2, 2),
    }
    kwargs = {
        "tactile_available_mask": torch.zeros((1, 0), dtype=torch.bool),
        "wrench": torch.zeros((1, 2, 6)),
        "wrench_available_mask": torch.zeros((1, 2), dtype=torch.bool),
        "repo_route_identity": "a" * 64,
        "temporal_alignment_identity": "b" * 64,
    }

    with pytest.raises(ValueError, match="temporal_valid_mask.*boolean"):
        canonicalize_agilex_sample(
            sample,
            contact_cond_drop=True,
            temporal_valid_mask=torch.ones(1),
            **kwargs,
        )
    with pytest.raises(ValueError, match="contact_cond_drop.*boolean"):
        canonicalize_agilex_sample(
            sample,
            contact_cond_drop=1,  # type: ignore[arg-type]
            **kwargs,
        )


def test_content_addressed_drop_is_stable_and_validates_probability() -> None:
    first = content_addressed_contact_drop(
        seed=7,
        epoch=3,
        repo_id="repo",
        episode_id=11,
        frame_anchor=20,
        probability=0.4,
    )
    second = content_addressed_contact_drop(
        seed=7,
        epoch=3,
        repo_id="repo",
        episode_id=11,
        frame_anchor=20,
        probability=0.4,
    )
    assert first is second
    assert (
        content_addressed_contact_drop(
            seed=7,
            epoch=3,
            repo_id="repo",
            episode_id=11,
            frame_anchor=20,
            probability=0.0,
        )
        is False
    )
    assert (
        content_addressed_contact_drop(
            seed=7,
            epoch=3,
            repo_id="repo",
            episode_id=11,
            frame_anchor=20,
            probability=1.0,
        )
        is True
    )
    with pytest.raises(ValueError, match="probability"):
        content_addressed_contact_drop(
            seed=7,
            epoch=3,
            repo_id="repo",
            episode_id=11,
            frame_anchor=20,
            probability=1.1,
        )


def test_agilex_wrench_config_accepts_sparse_signed_ids_and_route_subsets() -> None:
    agilex_dataset_module._validate_config(_strict_dataset_config())


@pytest.mark.parametrize(
    ("sensor_map", "match"),
    (
        ({LEFT_WRENCH_KEY: 9}, "exactly cover"),
        (
            {LEFT_WRENCH_KEY: True, RIGHT_WRENCH_KEY: 4},
            "integers within wrench_arm_count",
        ),
        (
            {LEFT_WRENCH_KEY: 4, RIGHT_WRENCH_KEY: 4},
            "must be unique",
        ),
        (
            {LEFT_WRENCH_KEY: 10, RIGHT_WRENCH_KEY: 4},
            "integers within wrench_arm_count",
        ),
        (
            {LEFT_WRENCH_KEY: -1, RIGHT_WRENCH_KEY: 4},
            "integers within wrench_arm_count",
        ),
    ),
)
def test_agilex_wrench_config_rejects_invalid_signed_rosters(
    sensor_map: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        agilex_dataset_module._validate_config(
            _strict_dataset_config(wrench_sensor_id_map=sensor_map)
        )


@pytest.mark.parametrize(
    "used_keys",
    (
        (LEFT_WRENCH_KEY, RIGHT_WRENCH_KEY),
        (RIGHT_WRENCH_KEY,),
    ),
)
def test_agilex_wrench_alignment_uses_signed_sparse_slots_not_route_order(
    monkeypatch: pytest.MonkeyPatch,
    used_keys: tuple[str, ...],
) -> None:
    target = SimpleNamespace(
        actions=np.zeros((14, 2, 1, 1), dtype=np.float32),
        model_valid_mask=np.ones((14, 2, 1, 1), dtype=np.bool_),
    )
    monkeypatch.setattr(
        agilex_dataset_module,
        "build_qpos14_latent_targets",
        lambda **_kwargs: target,
    )
    dataset = object.__new__(agilex_dataset_module.AgileXLatentLeRobotDataset)
    dataset.action_offsets_per_anchor = (0,)
    dataset.qpos14_codec = object()
    dataset.max_wrench_streams = 10
    dataset.wrench_sensor_id_map = {
        LEFT_WRENCH_KEY: 9,
        RIGHT_WRENCH_KEY: 4,
    }
    dataset.used_wrench_keys = used_keys
    left = torch.arange(12, dtype=torch.float32).reshape(2, 6) + 100
    right = torch.arange(12, dtype=torch.float32).reshape(2, 6) + 200
    dataset._range_context = {
        "action.valid": np.ones(2, dtype=np.bool_),
        LEFT_WRENCH_KEY: left,
        RIGHT_WRENCH_KEY: right,
    }
    dataset._aligned_context = None

    dataset._action_post_process(
        0,
        2,
        np.repeat(np.asarray((0, 1), dtype=np.int64), 4),
        np.zeros((2, 14), dtype=np.float32),
    )

    aligned = dataset._aligned_context
    assert aligned is not None
    wrench = aligned["wrench"]
    available = aligned["wrench_available_mask"]
    assert wrench.shape == (2, 10, 6)
    assert available.shape == (2, 10)
    assert torch.equal(wrench[:, 4], right)
    assert torch.equal(available[:, 4], torch.ones(2, dtype=torch.bool))
    if LEFT_WRENCH_KEY in used_keys:
        assert torch.equal(wrench[:, 9], left)
        assert torch.equal(available[:, 9], torch.ones(2, dtype=torch.bool))
    else:
        assert torch.count_nonzero(wrench[:, 9]) == 0
        assert torch.count_nonzero(available[:, 9]) == 0
    unused_slots = (0, 1, 2, 3, 5, 6, 7, 8)
    assert torch.count_nonzero(wrench[:, unused_slots]) == 0
    assert torch.count_nonzero(available[:, unused_slots]) == 0


@pytest.mark.parametrize("sensors", (0, 2))
def test_agilex_shape_signature_matches_the_canonical_sample_roster(
    sensors: int,
) -> None:
    signature = build_agilex_sample_shape_signature(
        _base_signature(sensors=sensors),
        max_wrench_streams=2,
    )
    fields = {name: (shape, dtype) for name, shape, dtype in signature}

    assert "actions_mask" not in fields
    assert "tactile_cond_drop" not in fields
    assert fields["action_valid_mask"] == ((3, 4), "torch.bool")
    assert fields["temporal_valid_mask"] == ((3,), "torch.bool")
    assert fields["tactile_available_mask"] == ((3, sensors), "torch.bool")
    assert fields["tactile_condition_mask"] == ((3, sensors), "torch.bool")
    assert fields["tactile_target_mask"] == ((3, sensors), "torch.bool")
    assert fields["wrench"] == ((3, 2, 6), "torch.float32")
    assert fields["wrench_available_mask"] == ((3, 2), "torch.bool")
    assert fields["wrench_condition_mask"] == ((3, 2), "torch.bool")
    assert fields["contact_cond_drop"] == ((), "torch.bool")
    assert fields["repo_route_identity"] == ((), "python.str")
    assert fields["temporal_alignment_identity"] == ((), "python.str")


def test_agilex_multi_dataset_forwards_one_exact_signature_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signatures = {
        "touch": _base_signature(sensors=2),
        "rgb": _base_signature(sensors=0),
    }

    class _FakeDataset:
        def __init__(self, repo_id: str) -> None:
            self.repo_name = Path(repo_id).name
            self.sample_signatures = (signatures[self.repo_name],) * 2

        def __len__(self) -> int:
            return 2

        def set_epoch(self, _epoch: int) -> None:
            return None

    monkeypatch.setattr(agilex_dataset_module, "_validate_config", lambda _: None)
    monkeypatch.setattr(
        agilex_dataset_module,
        "recursive_find_file",
        lambda *_: [
            "/dataset/touch/meta/info.json",
            "/dataset/rgb/meta/info.json",
        ],
    )
    monkeypatch.setattr(
        agilex_dataset_module,
        "_construct",
        lambda repo_id, config: _FakeDataset(repo_id),
    )
    monkeypatch.setattr(
        agilex_dataset_module,
        "validate_tactile_profile_config",
        lambda *_args, **_kwargs: None,
    )
    dataset = MultiLatentLeRobotAgileXDataset(
        SimpleNamespace(
            dataset_path="/dataset",
            num_init_worker=1,
            max_wrench_streams=2,
        )
    )

    assert len(dataset.sample_signatures) == len(dataset) == 4
    expected = tuple(
        build_agilex_sample_shape_signature(
            signatures[child.repo_name],
            max_wrench_streams=2,
        )
        for child in dataset._datasets
        for _ in range(len(child))
    )
    assert dataset.sample_signatures == expected
