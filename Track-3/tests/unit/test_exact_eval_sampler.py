# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import pytest

from n0_twam.data.exact_eval_sampler import (
    ExactDistributedEvalSampler,
    deterministic_validation_seed,
)


@pytest.mark.parametrize(
    ("dataset_size", "world_size"),
    ((10, 48), (40, 48), (10, 3), (40, 8), (1, 1)),
)
def test_exact_eval_sampler_counts_every_real_sample_once(
    dataset_size: int,
    world_size: int,
) -> None:
    dataset = list(range(dataset_size))
    rank_samples = [
        ExactDistributedEvalSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
        )
        for rank in range(world_size)
    ]

    assert len({len(sampler) for sampler in rank_samples}) == 1
    included = sorted(
        index
        for sampler in rank_samples
        for index, valid in zip(sampler, sampler.validity_mask, strict=True)
        if valid
    )
    assert included == list(range(dataset_size))
    assert sum(sum(sampler.validity_mask) for sampler in rank_samples) == dataset_size


def test_exact_eval_sampler_rejects_invalid_topology() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ExactDistributedEvalSampler([], num_replicas=1, rank=0)
    with pytest.raises(ValueError, match="num_replicas"):
        ExactDistributedEvalSampler([0], num_replicas=0, rank=0)
    with pytest.raises(ValueError, match="rank"):
        ExactDistributedEvalSampler([0], num_replicas=2, rank=2)


def test_validation_seed_is_independent_of_rank_and_world_size() -> None:
    dataset = list(range(10))
    mappings = []
    for world_size in (32, 48):
        mapping = {}
        for rank in range(world_size):
            sampler = ExactDistributedEvalSampler(
                dataset, num_replicas=world_size, rank=rank
            )
            for index, valid in zip(
                sampler.sample_indices, sampler.validity_mask, strict=True
            ):
                if valid:
                    mapping[index] = deterministic_validation_seed(
                        42, f"sample-{index}"
                    )
        mappings.append(mapping)

    assert mappings[0] == mappings[1]
    assert mappings[0][0] != deterministic_validation_seed(43, "sample-0")
