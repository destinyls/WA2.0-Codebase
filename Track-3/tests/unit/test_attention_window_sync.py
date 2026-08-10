"""Tests for synchronized distributed attention-mask sampling."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from n0_twam.distributed.attention_schedule import (
    resolve_sync_attention_window,
    sample_attention_mask_schedule,
)


def test_sync_attention_window_defaults_off() -> None:
    with patch.dict(os.environ):
        os.environ.pop("N0_SYNC_ATTENTION_WINDOW", None)
        assert not resolve_sync_attention_window()


def test_sync_attention_window_rejects_ambiguous_values() -> None:
    with patch.dict(os.environ, {"N0_SYNC_ATTENTION_WINDOW": "true"}):
        with pytest.raises(ValueError, match="must be 0 or 1"):
            resolve_sync_attention_window()


def test_nonzero_rank_uses_rank_zero_mask_schedule() -> None:
    def fake_broadcast(values: torch.Tensor, *, src: int) -> None:
        assert src == 0
        values.copy_(torch.tensor([3, 17], dtype=values.dtype))

    with (
        patch.dict(os.environ, {"N0_SYNC_ATTENTION_WINDOW": "1"}),
        patch(
            "n0_twam.distributed.attention_schedule.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "n0_twam.distributed.attention_schedule.dist.get_rank",
            return_value=7,
        ),
        patch(
            "n0_twam.distributed.attention_schedule.dist.broadcast",
            side_effect=fake_broadcast,
        ),
    ):
        assert sample_attention_mask_schedule(torch.device("cpu")) == (3, 17)
