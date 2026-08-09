"""Tests for no-drop distributed bucket sampling."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

_SAMPLER_PATH = (
    Path(__file__).resolve().parents[2] / "n0_twam" / "dataset" / "bucket_sampler.py"
)
_SPEC = importlib.util.spec_from_file_location("n0_twam_bucket_sampler", _SAMPLER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
BucketedDistributedBatchSampler = _MODULE.BucketedDistributedBatchSampler


def _global_batches(
    *,
    sample_count: int,
    batch_size: int,
    world_size: int,
    epoch: int = 0,
) -> tuple[list[list[int]], list[BucketedDistributedBatchSampler]]:
    signatures = [(3, 2)] * sample_count
    samplers = [
        BucketedDistributedBatchSampler(
            signatures,
            batch_size=batch_size,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=17,
            coverage_mode="pad_global",
        )
        for rank in range(world_size)
    ]
    for sampler in samplers:
        sampler.set_epoch(epoch)
    return [batch for sampler in samplers for batch in sampler], samplers


@pytest.mark.parametrize(
    ("sample_count", "expected_total", "expected_padding"),
    [(759, 768, 9), (190, 192, 2)],
)
def test_pad_global_covers_every_index_before_exact_global_padding(
    sample_count: int,
    expected_total: int,
    expected_padding: int,
) -> None:
    batches, samplers = _global_batches(
        sample_count=sample_count,
        batch_size=2,
        world_size=48,
    )

    ordered = [index for batch in batches for index in batch]
    assert len(ordered) == expected_total
    assert len(batches) == expected_total // 2
    assert {len(sampler) for sampler in samplers} == {expected_total // 2 // 48}
    assert len(set(ordered[:sample_count])) == sample_count
    assert set(ordered[:sample_count]) == set(range(sample_count))
    assert len(ordered[sample_count:]) == expected_padding
    assert samplers[0].ordered_indices() == tuple(ordered)
    assert samplers[0].padding_indices == tuple(ordered[sample_count:])
    assert all(
        count > 0
        for count in samplers[0]
        .exposure_summary()["cumulative_exposure_counts"]
        .values()
    )


def test_pad_global_rotates_padding_deterministically_across_epochs() -> None:
    kwargs = dict(
        signatures=[(3, 2)] * 25,
        batch_size=2,
        num_replicas=4,
        rank=0,
        seed=23,
        coverage_mode="pad_global",
        tasks=[f"task_{index % 4}" for index in range(25)],
    )
    sampler = BucketedDistributedBatchSampler(**kwargs)

    sampler.set_epoch(0)
    epoch_zero = sampler.padding_indices
    sampler.set_epoch(1)
    epoch_one = sampler.padding_indices

    replay = BucketedDistributedBatchSampler(**kwargs)
    replay.set_epoch(1)

    assert epoch_zero != epoch_one
    assert set(epoch_zero).isdisjoint(epoch_one)
    assert replay.padding_indices == epoch_one
    task_padding = Counter(kwargs["tasks"][index] for index in epoch_zero)
    assert max(task_padding.values()) - min(task_padding.values()) <= 1


def test_pad_global_summary_is_json_serializable_and_hashes_global_order() -> None:
    tasks = ["insert_HDMI"] * 5 + ["lift_bottle"] * 5
    sample_ids = [f"episode-{index:03d}" for index in range(10)]
    sampler = BucketedDistributedBatchSampler(
        [(3, 2)] * 10,
        batch_size=2,
        num_replicas=4,
        rank=2,
        shuffle=False,
        seed=11,
        coverage_mode="pad_global",
        tasks=tasks,
        sample_ids=sample_ids,
    )
    sampler.set_epoch(3)

    summary = sampler.exposure_summary()
    encoded_order = json.dumps(
        list(sampler.ordered_indices()), separators=(",", ":")
    ).encode("utf-8")

    json.dumps(summary)
    assert summary["schema_version"] == 2
    assert summary["epoch"] == 3
    assert summary["unique_sample_count"] == 10
    assert summary["total_sample_count"] == 16
    assert summary["padding_count"] == 6
    assert summary["padding_indices"] == list(sampler.padding_indices)
    assert summary["ordered_index_sha256"] == hashlib.sha256(encoded_order).hexdigest()
    assert summary["per_task_unique_counts"] == {
        "insert_HDMI": 5,
        "lift_bottle": 5,
    }
    assert set(summary["cumulative_exposure_counts"]) == set(sample_ids)
    assert sum(summary["cumulative_exposure_counts"].values()) == 16


def test_pad_global_state_round_trip_restores_epoch_and_cumulative_exposure() -> None:
    kwargs = dict(
        signatures=[(3, 2)] * 19,
        batch_size=2,
        num_replicas=4,
        rank=1,
        shuffle=True,
        seed=101,
        coverage_mode="pad_global",
        tasks=[f"task_{index % 3}" for index in range(19)],
    )
    uninterrupted = BucketedDistributedBatchSampler(**kwargs)
    for epoch in range(3):
        uninterrupted.set_epoch(epoch)
        list(uninterrupted)

    state = json.loads(json.dumps(uninterrupted.state_dict()))
    resumed = BucketedDistributedBatchSampler(**kwargs)
    resumed.load_state_dict(state)

    assert resumed.epoch == 2
    assert resumed.ordered_indices() == uninterrupted.ordered_indices()
    assert resumed.exposure_summary() == uninterrupted.exposure_summary()

    uninterrupted.set_epoch(3)
    resumed.set_epoch(3)
    assert list(resumed) == list(uninterrupted)
    assert resumed.exposure_summary() == uninterrupted.exposure_summary()


def test_load_state_dict_rejects_a_different_sampler_contract() -> None:
    original = BucketedDistributedBatchSampler(
        [(3, 2)] * 10,
        batch_size=2,
        num_replicas=2,
        coverage_mode="pad_global",
        seed=5,
    )
    incompatible = BucketedDistributedBatchSampler(
        [(3, 2)] * 10,
        batch_size=2,
        num_replicas=2,
        coverage_mode="pad_global",
        seed=6,
    )

    with pytest.raises(ValueError, match="seed"):
        incompatible.load_state_dict(original.state_dict())


def test_pad_global_preserves_shape_homogeneity_when_one_bucket_has_a_tail() -> None:
    signatures = ["a"] * 5 + ["b"] * 4
    sampler = BucketedDistributedBatchSampler(
        signatures,
        batch_size=2,
        num_replicas=2,
        rank=0,
        shuffle=True,
        coverage_mode="pad_global",
    )
    peer = BucketedDistributedBatchSampler(
        signatures,
        batch_size=2,
        num_replicas=2,
        rank=1,
        shuffle=True,
        coverage_mode="pad_global",
    )
    batches = [*sampler, *peer]
    ordered = [index for batch in batches for index in batch]

    assert len(ordered) == 12
    assert set(ordered[:9]) == set(range(9))
    assert all(len({signatures[index] for index in batch}) == 1 for batch in batches)


def test_shape_balanced_alignment_minimizes_cross_rank_step_stragglers() -> None:
    signatures = ["short"] * 9 + ["long"] * 7
    world_size = 4
    samplers = [
        BucketedDistributedBatchSampler(
            signatures,
            batch_size=1,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=31,
            coverage_mode="pad_global",
            rank_alignment_mode="shape_balanced",
        )
        for rank in range(world_size)
    ]
    per_rank = [list(sampler) for sampler in samplers]
    signature_counts = [
        len(
            {
                signatures[per_rank[rank][step][0]]
                for rank in range(world_size)
            }
        )
        for step in range(len(samplers[0]))
    ]

    assert signature_counts.count(1) == len(signature_counts) - 1
    assert max(signature_counts) == 2
    assert samplers[0].exposure_summary()["rank_alignment"] == {
        "mode": "shape_balanced",
        "step_count": 4,
        "homogeneous_step_count": 3,
        "max_signatures_per_step": 2,
    }


def test_sampler_state_rejects_rank_alignment_drift() -> None:
    common = {
        "signatures": ["short"] * 8,
        "batch_size": 1,
        "num_replicas": 4,
        "coverage_mode": "pad_global",
    }
    contiguous = BucketedDistributedBatchSampler(
        **common,
        rank_alignment_mode="contiguous",
    )
    balanced = BucketedDistributedBatchSampler(
        **common,
        rank_alignment_mode="shape_balanced",
    )

    with pytest.raises(ValueError, match="rank_alignment_mode"):
        balanced.load_state_dict(contiguous.state_dict())


def test_pad_global_rejects_multiple_ragged_shape_buckets() -> None:
    with pytest.raises(ValueError, match="ragged shape bucket"):
        BucketedDistributedBatchSampler(
            ["a"] * 3 + ["b"] * 3,
            batch_size=2,
            num_replicas=2,
            coverage_mode="pad_global",
        )


def test_legacy_mode_retains_drop_last_behavior() -> None:
    signatures = ["a"] * 5 + ["b"] * 4
    samplers = [
        BucketedDistributedBatchSampler(
            signatures,
            batch_size=2,
            num_replicas=2,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )
        for rank in range(2)
    ]
    ordered = [index for sampler in samplers for batch in sampler for index in batch]

    assert len(ordered) == 8
    assert Counter(ordered) == Counter({index: 1 for index in range(9) if index != 4})
