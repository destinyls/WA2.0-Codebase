# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Equal-step distributed evaluation without duplicate metric weight."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sized
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    _SampleT = TypeVar("_SampleT")

    class _SamplerBase(Generic[_SampleT]):
        """Typed stand-in when repository checks skip third-party imports."""

else:
    from torch.utils.data import Sampler as _SamplerBase


def deterministic_validation_seed(base_seed: int, sample_identity: str) -> int:
    """Derive one topology-invariant 32-bit seed from immutable sample identity."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("validation base seed must be a non-negative integer")
    if not isinstance(sample_identity, str) or not sample_identity:
        raise ValueError("validation sample identity must be a non-empty string")
    payload = f"n0-track31-val\0{base_seed}\0{sample_identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class ExactDistributedEvalSampler(_SamplerBase[int]):
    """Pad rank workloads for collectives while marking padding as invalid.

    Every rank receives the same number of samples, which is required by FSDP
    forward collectives. The paired ``validity_mask`` lets the evaluator remove
    deterministic padding from metric sums and counts.
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int,
        rank: int,
    ) -> None:
        dataset_size = len(dataset)
        if dataset_size <= 0:
            raise ValueError("evaluation dataset must be non-empty")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        samples_per_rank = math.ceil(dataset_size / num_replicas)
        total_size = samples_per_rank * num_replicas
        indices = list(range(dataset_size))
        padding_size = total_size - dataset_size
        padding = (
            (indices * math.ceil(padding_size / dataset_size))[:padding_size]
            if padding_size
            else []
        )
        padded_indices = indices + padding
        padded_validity = [True] * dataset_size + [False] * padding_size
        self._indices = tuple(padded_indices[rank:total_size:num_replicas])
        self._validity_mask = tuple(padded_validity[rank:total_size:num_replicas])
        if len(self._indices) != samples_per_rank:
            raise RuntimeError("distributed evaluation sampler size is inconsistent")

    @property
    def validity_mask(self) -> tuple[bool, ...]:
        """Return one metric-inclusion flag per yielded sample."""

        return self._validity_mask

    @property
    def sample_indices(self) -> tuple[int, ...]:
        """Return the dataset index paired with each validity flag."""

        return self._indices

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)
