# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Shape-bucketed distributed batch sampling with optional no-drop coverage."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterator, Mapping, Sequence
from typing import Any

import torch

_COVERAGE_MODES = frozenset({"legacy", "pad_global"})
_RANK_ALIGNMENT_MODES = frozenset({"contiguous", "shape_balanced"})
_STATE_SCHEMA_VERSION = 2


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class BucketedDistributedBatchSampler:
    """Yield equal homogeneous-batch counts on every distributed rank.

    ``pad_global`` covers every index before rotating padding. Multiple ragged
    shape buckets are rejected because they make that strict order impossible.
    """

    def __init__(
        self,
        signatures: Sequence[Hashable],
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
        coverage_mode: str = "legacy",
        rank_alignment_mode: str = "contiguous",
        tasks: Sequence[str] | None = None,
        sample_ids: Sequence[str | int] | None = None,
    ) -> None:
        self.signatures = list(signatures)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.coverage_mode = str(coverage_mode)
        if self.coverage_mode not in _COVERAGE_MODES:
            raise ValueError(f"invalid coverage_mode: {self.coverage_mode!r}")
        self.rank_alignment_mode = str(rank_alignment_mode)
        if self.rank_alignment_mode not in _RANK_ALIGNMENT_MODES:
            raise ValueError(
                f"invalid rank_alignment_mode: {self.rank_alignment_mode!r}"
            )
        if (
            self.rank_alignment_mode == "shape_balanced"
            and self.coverage_mode != "pad_global"
        ):
            raise ValueError("shape_balanced rank alignment requires pad_global")
        if self.coverage_mode == "pad_global" and not (
            0 <= self.rank < self.num_replicas
        ):
            raise ValueError(f"rank must be in [0, {self.num_replicas})")
        self.tasks = None if tasks is None else [str(task) for task in tasks]
        if self.tasks is not None and len(self.tasks) != len(self.signatures):
            raise ValueError("tasks must contain one label per signature")
        if sample_ids is None:
            self.sample_ids = [str(index) for index in range(len(self.signatures))]
        else:
            if len(sample_ids) != len(self.signatures):
                raise ValueError("sample_ids must contain one ID per signature")
            self.sample_ids = [str(sample_id) for sample_id in sample_ids]
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique after string conversion")
        buckets: dict[Hashable, list[int]] = defaultdict(list)
        for index, signature in enumerate(self.signatures):
            buckets[signature].append(index)
        self._buckets = dict(buckets)
        self._bucket_keys = sorted(self._buckets, key=str)
        self._ragged_keys = [
            key
            for key in self._bucket_keys
            if len(self._buckets[key]) % self.batch_size
        ]
        if self.coverage_mode == "pad_global" and len(self._ragged_keys) > 1:
            raise ValueError("pad_global rejects multiple ragged shape buckets")
        if self.coverage_mode == "pad_global":
            quantum = self.batch_size * self.num_replicas
            padded = (len(self.signatures) + quantum - 1) // quantum * quantum
            self._num_batches = padded // quantum
        else:
            if self.drop_last:
                total = sum(
                    len(indices) // self.batch_size
                    for indices in self._buckets.values()
                )
            else:
                total = sum(
                    (len(indices) + self.batch_size - 1) // self.batch_size
                    for indices in self._buckets.values()
                )
            self._num_batches = total // self.num_replicas
        self.epoch = 0
        self._padding_cycles = {
            key: self._build_padding_cycle(key) for key in self._bucket_keys
        }
        self._cumulative_exposure_counts = [0] * len(self.signatures)
        self._accounted_epochs: set[int] = set()
        self._cached_epoch: int | None = None
        self._cached_batches: tuple[tuple[int, ...], ...] = ()
        self._cached_ordered: tuple[int, ...] = ()
        self._cached_padding: tuple[int, ...] = ()
        self._input_sha256 = _sha256_json(
            (
                [repr(value) for value in self.signatures],
                self.tasks,
                self.sample_ids,
                self.rank_alignment_mode,
            )
        )

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic logical epoch."""
        value = int(epoch)
        if self.coverage_mode == "pad_global" and value < 0:
            raise ValueError(f"epoch must be >= 0, got {value}")
        self.epoch = value
        if self._cached_epoch != value:
            self._cached_epoch = None

    def _legacy_batches(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []
        for key in self._bucket_keys:
            indices = self._buckets[key]
            order = (
                torch.randperm(len(indices), generator=generator).tolist()
                if self.shuffle
                else list(range(len(indices)))
            )
            limit = (
                len(indices) // self.batch_size * self.batch_size
                if self.drop_last
                else len(indices)
            )
            for start in range(0, limit, self.batch_size):
                stop = min(start + self.batch_size, len(indices))
                batches.append(
                    [indices[order[position]] for position in range(start, stop)]
                )
        if self.shuffle and batches:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[position] for position in order]
        return batches

    def _build_padding_cycle(self, key: Hashable) -> tuple[int, ...]:
        indices = self._buckets[key]
        digest = hashlib.sha256(f"{self.seed}:{key!r}".encode()).digest()
        generator = torch.Generator().manual_seed(
            int.from_bytes(digest[:8], "big") % (2**63 - 1)
        )
        if self.tasks is None:
            order = torch.randperm(len(indices), generator=generator).tolist()
            return tuple(indices[position] for position in order)
        groups: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            groups[self.tasks[index]].append(index)
        names = sorted(groups)
        order = torch.randperm(len(names), generator=generator).tolist()
        names = [names[position] for position in order]
        for name in names:
            order = torch.randperm(len(groups[name]), generator=generator).tolist()
            groups[name] = [groups[name][position] for position in order]
        positions = {name: 0 for name in names}
        cycle: list[int] = []
        while len(cycle) < len(indices):
            for name in names:
                position = positions[name]
                if position < len(groups[name]):
                    cycle.append(groups[name][position])
                    positions[name] += 1
        return tuple(cycle)

    @staticmethod
    def _cycle_slice(cycle: tuple[int, ...], start: int, count: int) -> list[int]:
        return [cycle[(start + offset) % len(cycle)] for offset in range(count)]

    def _padding_owner(self) -> Hashable:
        bucket_count = len(self._bucket_keys)
        return self._bucket_keys[(self.seed + self.epoch) % bucket_count]

    def _prior_owner_epochs(self, key: Hashable) -> int:
        bucket_count = len(self._bucket_keys)
        position = self._bucket_keys.index(key)
        phase = self.seed % bucket_count
        complete, remainder = divmod(self.epoch, bucket_count)
        return complete + int((position - phase) % bucket_count < remainder)

    def _pad_global_plan(
        self,
    ) -> tuple[list[list[int]], tuple[int, ...], tuple[int, ...]]:
        if not self.signatures:
            return [], (), ()
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        full_batches: list[list[int]] = []
        tail: list[int] = []
        tail_key = self._ragged_keys[0] if self._ragged_keys else None
        for key in self._bucket_keys:
            indices = self._buckets[key]
            order = (
                torch.randperm(len(indices), generator=generator).tolist()
                if self.shuffle
                else list(range(len(indices)))
            )
            limit = len(indices) // self.batch_size * self.batch_size
            full_batches.extend(
                [
                    indices[order[position]]
                    for position in range(start, start + self.batch_size)
                ]
                for start in range(0, limit, self.batch_size)
            )
            if limit < len(indices):
                tail = [
                    indices[order[position]] for position in range(limit, len(indices))
                ]
        if self.shuffle and full_batches:
            order = torch.randperm(len(full_batches), generator=generator).tolist()
            full_batches = [full_batches[position] for position in order]
        unique = [index for batch in full_batches for index in batch] + tail
        quantum = self.batch_size * self.num_replicas
        padded = (len(unique) + quantum - 1) // quantum * quantum
        padding_count = padded - len(unique)
        tail_padding = self.batch_size - len(tail) if tail else 0
        extra_padding = padding_count - tail_padding
        offsets = {
            key: self._prior_owner_epochs(key) * extra_padding
            + (self.epoch * tail_padding if tail and key == tail_key else 0)
            for key in self._bucket_keys
        }
        padding: list[int] = []
        if tail:
            padding.extend(
                self._cycle_slice(
                    self._padding_cycles[tail_key], offsets[tail_key], tail_padding
                )
            )
            offsets[tail_key] += tail_padding
        if extra_padding:
            key = self._padding_owner()
            padding.extend(
                self._cycle_slice(
                    self._padding_cycles[key], offsets[key], extra_padding
                )
            )
        ordered = tuple(unique + padding)
        batches = [
            list(ordered[start : start + self.batch_size])
            for start in range(0, len(ordered), self.batch_size)
        ]
        if any(
            len({self.signatures[index] for index in batch}) != 1 for batch in batches
        ):
            raise RuntimeError("pad_global produced a heterogeneous shape batch")
        if self.rank_alignment_mode == "shape_balanced":
            batches = self._shape_balanced_rank_major_batches(
                batches,
                generator=generator,
            )
            ordered = tuple(index for batch in batches for index in batch)
        return batches, ordered, tuple(padding)

    def _shape_balanced_rank_major_batches(
        self,
        batches: list[list[int]],
        *,
        generator: torch.Generator,
    ) -> list[list[int]]:
        """Group similar shapes at the same optimizer step on every rank."""
        if len(batches) % self.num_replicas:
            raise RuntimeError("rank-aligned batches must divide across all ranks")
        sorted_batches = sorted(
            batches,
            key=lambda batch: repr(self.signatures[batch[0]]),
        )
        step_groups = [
            sorted_batches[start : start + self.num_replicas]
            for start in range(0, len(sorted_batches), self.num_replicas)
        ]
        if self.shuffle and step_groups:
            order = torch.randperm(
                len(step_groups),
                generator=generator,
            ).tolist()
            step_groups = [step_groups[position] for position in order]
        return [
            step_groups[step][rank]
            for rank in range(self.num_replicas)
            for step in range(len(step_groups))
        ]

    def _ensure_plan(self) -> None:
        if self._cached_epoch == self.epoch:
            return
        if self.coverage_mode == "pad_global":
            batches, ordered, padding = self._pad_global_plan()
        else:
            batches = self._legacy_batches()
            used = batches[: self._num_batches * self.num_replicas]
            ordered = tuple(index for batch in used for index in batch)
            padding = ()
        self._cached_epoch = self.epoch
        self._cached_batches = tuple(tuple(batch) for batch in batches)
        self._cached_ordered = ordered
        self._cached_padding = padding

    def _all_batches(self) -> list[list[int]]:
        self._ensure_plan()
        return [list(batch) for batch in self._cached_batches]

    def ordered_indices(self) -> tuple[int, ...]:
        self._ensure_plan()
        return self._cached_ordered

    @property
    def padding_indices(self) -> tuple[int, ...]:
        self._ensure_plan()
        return self._cached_padding

    @property
    def ordered_index_sha256(self) -> str:
        return _sha256_json(list(self.ordered_indices()))

    def _record_exposure(self) -> None:
        if self.epoch not in self._accounted_epochs:
            for index in self.ordered_indices():
                self._cumulative_exposure_counts[index] += 1
            self._accounted_epochs.add(self.epoch)

    def _exposure_counts(self) -> dict[str, int]:
        return dict(zip(self.sample_ids, self._cumulative_exposure_counts, strict=True))

    def exposure_summary(self) -> dict[str, Any]:
        self._record_exposure()
        ordered = self.ordered_indices()
        unique = set(ordered)
        unique_tasks: Counter[str] = Counter()
        if self.tasks is not None:
            unique_tasks.update(self.tasks[index] for index in unique)
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "epoch": self.epoch,
            "unique_sample_count": len(unique),
            "total_sample_count": len(ordered),
            "padding_count": len(self.padding_indices),
            "ordered_index_sha256": self.ordered_index_sha256,
            "padding_indices": list(self.padding_indices),
            "per_task_unique_counts": dict(sorted(unique_tasks.items())),
            "cumulative_exposure_counts": self._exposure_counts(),
            "rank_alignment": self._rank_alignment_summary(),
        }

    def _rank_alignment_summary(self) -> dict[str, int | str]:
        """Summarize cross-rank shape diversity at corresponding steps."""
        self._ensure_plan()
        signature_counts = []
        for step in range(self._num_batches):
            signatures = {
                self.signatures[
                    self._cached_batches[rank * self._num_batches + step][0]
                ]
                for rank in range(self.num_replicas)
            }
            signature_counts.append(len(signatures))
        return {
            "mode": self.rank_alignment_mode,
            "step_count": self._num_batches,
            "homogeneous_step_count": sum(
                count == 1 for count in signature_counts
            ),
            "max_signatures_per_step": max(signature_counts, default=0),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "epoch": self.epoch,
            "seed": self.seed,
            "coverage_mode": self.coverage_mode,
            "rank_alignment_mode": self.rank_alignment_mode,
            "batch_size": self.batch_size,
            "num_replicas": self.num_replicas,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "dataset_size": len(self.signatures),
            "input_sha256": self._input_sha256,
            "ordered_index_sha256": self.ordered_index_sha256,
            "accounted_epochs": sorted(self._accounted_epochs),
            "cumulative_exposure_counts": self._exposure_counts(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate and restore :meth:`state_dict` output."""
        if not isinstance(state, Mapping):
            raise TypeError("sampler state must be a mapping")
        if state.get("schema_version") != _STATE_SCHEMA_VERSION:
            raise ValueError("unsupported sampler state schema_version")
        bindings = {
            "seed": self.seed,
            "coverage_mode": self.coverage_mode,
            "rank_alignment_mode": self.rank_alignment_mode,
            "batch_size": self.batch_size,
            "num_replicas": self.num_replicas,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "dataset_size": len(self.signatures),
            "input_sha256": self._input_sha256,
        }
        for name, expected in bindings.items():
            if state.get(name) != expected:
                raise ValueError(f"sampler state {name} mismatch")
        epoch = state.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise ValueError("sampler state epoch is invalid")
        if self.coverage_mode == "pad_global" and epoch < 0:
            raise ValueError("sampler state epoch is invalid")
        raw_counts = state.get("cumulative_exposure_counts")
        if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(
            self.sample_ids
        ):
            raise ValueError("sampler exposure keys do not match sample_ids")
        raw_count_values = [raw_counts[sample_id] for sample_id in self.sample_ids]
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in raw_count_values
        ):
            raise ValueError("sampler exposure counts must be non-negative integers")
        counts = [int(count) for count in raw_count_values]
        raw_epochs = state.get("accounted_epochs")
        if not isinstance(raw_epochs, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in raw_epochs
        ):
            raise ValueError("sampler accounted_epochs must be non-negative integers")
        accounted = set(raw_epochs)
        if len(accounted) != len(raw_epochs):
            raise ValueError("sampler accounted_epochs must be unique")
        previous_epoch = self.epoch
        self.set_epoch(epoch)
        if state.get("ordered_index_sha256") != self.ordered_index_sha256:
            self.set_epoch(previous_epoch)
            raise ValueError("sampler state ordered_index_sha256 mismatch")
        self._cumulative_exposure_counts = counts
        self._accounted_epochs = accounted

    def __iter__(self) -> Iterator[list[int]]:
        self._record_exposure()
        batches = self._all_batches()
        start = self.rank * self._num_batches
        return iter(batches[start : start + self._num_batches])

    def __len__(self) -> int:
        return self._num_batches


__all__ = ["BucketedDistributedBatchSampler"]
