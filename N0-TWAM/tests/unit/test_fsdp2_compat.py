import pytest
import torch

from n0_twam.distributed.fsdp import (
    FSDP2_API_SOURCE,
    MixedPrecisionPolicy,
    fully_shard,
)
from script.track3_1.smoke_fsdp2_runtime import (
    _collective_raise_if_error,
    _snapshot_max_abs_error,
)


def test_fsdp2_compat_exports_composable_api() -> None:
    assert FSDP2_API_SOURCE in {
        "torch.distributed.fsdp",
        "torch.distributed._composable.fsdp",
    }
    assert callable(fully_shard)
    policy = MixedPrecisionPolicy()
    assert policy is not None


def test_smoke_collective_propagates_remote_rank_stage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _mark_failed(flag: torch.Tensor, op: object) -> None:
        del op
        flag.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", _mark_failed)

    with pytest.raises(RuntimeError, match="failed on at least one rank"):
        _collective_raise_if_error(None, "rank-zero write", torch.device("cpu"))


def test_smoke_snapshot_comparison_covers_adam_tensor_values() -> None:
    expected = {
        "weight": {
            "step": torch.tensor(2),
            "exp_avg": torch.tensor([1.0, 2.0]),
            "exp_avg_sq": torch.tensor([3.0, 4.0]),
        }
    }
    actual = {
        "weight": {
            "step": torch.tensor(2),
            "exp_avg": torch.tensor([1.0, 2.25]),
            "exp_avg_sq": torch.tensor([3.0, 4.0]),
        }
    }

    assert _snapshot_max_abs_error(expected, actual) == pytest.approx(0.25)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_smoke_snapshot_comparison_rejects_non_finite_values(
    non_finite: float,
) -> None:
    expected = {"exp_avg": torch.tensor([1.0, non_finite])}
    actual = {"exp_avg": torch.tensor([1.0, non_finite])}

    assert _snapshot_max_abs_error(expected, actual) == float("inf")
