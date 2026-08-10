"""Numerical parity tests for strict-resume DataLoader RNG handling."""

from __future__ import annotations

import importlib
import importlib.machinery
import random
import sys
import types
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class _RandomCropDataset(Dataset[torch.Tensor]):
    """Small map dataset whose sample represents three stochastic crops."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(
            (
                float(index),
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            ),
            dtype=torch.float64,
        )


def _load_trainer_class() -> type[Any]:
    """Import Trainer while stubbing optional local-test dependencies."""
    import n0_twam

    wandb_stub = types.ModuleType("wandb")
    wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    dataset_stub = types.ModuleType("dataset")
    dataset_stub.__spec__ = importlib.machinery.ModuleSpec("dataset", loader=None)
    dataset_stub.MultiLatentLeRobotDataset = object
    dataset_stub.BucketedDistributedBatchSampler = object

    stub_names = ("wandb", "dataset")
    previous_stubs = {name: sys.modules.get(name) for name in stub_names}
    previous_train_module = sys.modules.pop("n0_twam.train", None)
    missing_attribute = object()
    previous_train_attribute = getattr(n0_twam, "train", missing_attribute)
    try:
        sys.modules["wandb"] = wandb_stub
        sys.modules["dataset"] = dataset_stub
        return importlib.import_module("n0_twam.train").Trainer
    finally:
        sys.modules.pop("n0_twam.train", None)
        if previous_train_module is not None:
            sys.modules["n0_twam.train"] = previous_train_module
        if previous_train_attribute is missing_attribute:
            if hasattr(n0_twam, "train"):
                delattr(n0_twam, "train")
        else:
            n0_twam.train = previous_train_attribute
        for name, previous_module in previous_stubs.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


@pytest.fixture(scope="module")
def trainer_class() -> type[Any]:
    return _load_trainer_class()


def _capture_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _make_trainer(
    trainer_class: type[Any],
    *,
    batches_consumed: int = 0,
    pending_rng_state: dict[str, object] | None = None,
) -> Any:
    trainer = trainer_class.__new__(trainer_class)
    dataset = _RandomCropDataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=42,
    )
    trainer.train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
    )
    trainer.data_batches_consumed = batches_consumed
    trainer._pending_rng_state = pending_rng_state
    trainer.train_loader_iter = None
    return trainer


def _assert_rng_state_equal(
    actual: dict[str, object], expected: dict[str, object]
) -> None:
    assert actual["python"] == expected["python"]

    actual_numpy = actual["numpy"]
    expected_numpy = expected["numpy"]
    assert isinstance(actual_numpy, tuple)
    assert isinstance(expected_numpy, tuple)
    assert actual_numpy[0] == expected_numpy[0]
    np.testing.assert_array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]

    assert torch.equal(actual["torch"], expected["torch"])
    actual_cuda = actual["cuda"]
    expected_cuda = expected["cuda"]
    assert isinstance(actual_cuda, list)
    assert isinstance(expected_cuda, list)
    assert len(actual_cuda) == len(expected_cuda)
    for actual_state, expected_state in zip(actual_cuda, expected_cuda, strict=True):
        assert torch.equal(actual_state, expected_state)


@pytest.mark.parametrize(
    "checkpoint_batches",
    (
        pytest.param(2, id="mid_epoch"),
        pytest.param(4, id="epoch_boundary"),
    ),
)
def test_strict_resume_next_batch_and_rng_match_uninterrupted(
    trainer_class: type[Any], checkpoint_batches: int
) -> None:
    random.seed(20260801)
    np.random.seed(20260801)
    torch.manual_seed(20260801)
    uninterrupted = _make_trainer(trainer_class)
    for _ in range(checkpoint_batches):
        uninterrupted._get_next_batch()

    checkpoint_rng_state = _capture_rng_state()
    expected_batch = uninterrupted._get_next_batch().clone()
    expected_rng_state = _capture_rng_state()

    # Model/optimizer reconstruction may consume arbitrary RNG before the
    # checkpoint sidecar is applied. Strict resume must be independent of it.
    random.seed(123456)
    np.random.seed(123456)
    torch.manual_seed(123456)
    resumed = _make_trainer(
        trainer_class,
        batches_consumed=checkpoint_batches,
        pending_rng_state=checkpoint_rng_state,
    )
    actual_batch = resumed._get_next_batch().clone()
    actual_rng_state = _capture_rng_state()

    torch.testing.assert_close(actual_batch, expected_batch, rtol=0.0, atol=0.0)
    _assert_rng_state_equal(actual_rng_state, expected_rng_state)
