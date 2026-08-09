"""Tests for packed per-step distributed metric reductions."""

from __future__ import annotations

from unittest.mock import call, patch

import torch

from n0_twam.distributed.util import dist_mean_and_max


def test_dist_mean_and_max_uses_two_packed_collectives() -> None:
    metrics = torch.tensor([1.0, 2.0, 3.0, 4.0])

    with (
        patch("n0_twam.distributed.util.dist.is_initialized", return_value=True),
        patch("n0_twam.distributed.util.dist.all_reduce") as all_reduce,
    ):
        mean_metrics, max_metrics = dist_mean_and_max(metrics)

    assert torch.equal(mean_metrics, metrics)
    assert torch.equal(max_metrics, metrics)
    assert all_reduce.call_count == 2
    assert all_reduce.call_args_list == [
        call(mean_metrics, op=torch.distributed.ReduceOp.AVG),
        call(max_metrics, op=torch.distributed.ReduceOp.MAX),
    ]


def test_dist_mean_and_max_does_not_mutate_local_metrics() -> None:
    metrics = torch.tensor([1.0, 2.0])

    mean_metrics, max_metrics = dist_mean_and_max(metrics)

    assert mean_metrics.data_ptr() != metrics.data_ptr()
    assert max_metrics.data_ptr() != metrics.data_ptr()
    assert torch.equal(metrics, torch.tensor([1.0, 2.0]))
