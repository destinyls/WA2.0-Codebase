# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Content-addressed temporal cleanup contracts for UniVTAC episodes."""

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import numpy.typing as npt

STRICT_MONOTONIC_POLICY: Final[str] = "strict_monotonic_v1"
PINNED_MAXIMAL_PREFIX_POLICY: Final[str] = "pinned_maximal_monotonic_prefix_v1"


@dataclass(frozen=True)
class TemporalSelection:
    """Audited source-row interval used for next-step supervision."""

    usable_start: int
    usable_end: int
    discontinuities_after_rows: tuple[int, ...]
    policy: str

    @property
    def usable_length(self) -> int:
        """Return the number of retained raw source rows."""

        return self.usable_end - self.usable_start

    @property
    def converted_length(self) -> int:
        """Return the number of valid next-step pairs in the interval."""

        return self.usable_length - 1


@dataclass(frozen=True)
class PinnedTemporalContract:
    """Exact exception for one immutable source artifact."""

    relative_path: str
    sha256: str
    raw_length: int
    usable_start: int
    usable_end: int
    discontinuity_after_row: int
    left_step: int
    right_step: int


LIFT_CAN_62_TEMPORAL_CONTRACT: Final[PinnedTemporalContract] = PinnedTemporalContract(
    relative_path="lift_can/clean/62.hdf5",
    sha256=("335f6c32947b1d37acb638e67a756b7c37204df4c849121822234c7931fd63f2"),
    raw_length=256,
    usable_start=0,
    usable_end=228,
    discontinuity_after_row=227,
    left_step=594,
    right_step=586,
)

PINNED_TEMPORAL_CONTRACTS: Final[tuple[PinnedTemporalContract, ...]] = (
    LIFT_CAN_62_TEMPORAL_CONTRACT,
)


def _discontinuities(steps: npt.NDArray[np.int64]) -> tuple[int, ...]:
    """Return raw row indices whose outgoing step delta is non-positive."""

    if steps.size < 2:
        return ()
    return tuple(int(index) for index in np.flatnonzero(np.diff(steps) <= 0))


def validate_temporal_selection(
    steps: npt.NDArray[np.int64],
    *,
    selection: TemporalSelection,
    source_label: str,
) -> None:
    """Fail closed if persisted temporal metadata no longer matches the source."""

    length = int(steps.shape[0])
    if not 0 <= selection.usable_start < selection.usable_end <= length:
        raise ValueError(f"invalid usable source range for {source_label}")
    if selection.usable_length < 2:
        raise ValueError(f"usable source range is too short for {source_label}")
    observed = _discontinuities(steps)
    if observed != selection.discontinuities_after_rows:
        raise ValueError(
            "step discontinuities changed for "
            f"{source_label}: {selection.discontinuities_after_rows} -> {observed}"
        )
    retained = steps[selection.usable_start : selection.usable_end]
    if retained.size > 1 and np.any(np.diff(retained) <= 0):
        raise ValueError(
            f"usable step values must be strictly increasing for {source_label}"
        )
    if observed and selection.policy != PINNED_MAXIMAL_PREFIX_POLICY:
        raise ValueError(f"non-monotonic source lacks a pinned policy: {source_label}")
    if not observed and selection.policy != STRICT_MONOTONIC_POLICY:
        raise ValueError(f"monotonic source has an invalid policy: {source_label}")


def resolve_temporal_selection(
    steps: npt.NDArray[np.int64],
    *,
    relative_path: str,
    source_sha256: str | None,
    contracts: Sequence[PinnedTemporalContract] | None = None,
) -> TemporalSelection:
    """Resolve a strict full episode or one exact content-addressed prefix."""

    active_contracts = PINNED_TEMPORAL_CONTRACTS if contracts is None else contracts
    observed = _discontinuities(steps)
    matching_paths = tuple(
        contract
        for contract in active_contracts
        if contract.relative_path == relative_path
    )
    if not matching_paths and not observed:
        selection = TemporalSelection(
            usable_start=0,
            usable_end=int(steps.shape[0]),
            discontinuities_after_rows=(),
            policy=STRICT_MONOTONIC_POLICY,
        )
        validate_temporal_selection(
            steps,
            selection=selection,
            source_label=relative_path,
        )
        return selection
    if not matching_paths:
        raise ValueError(f"step values must be strictly increasing for {relative_path}")
    if len(matching_paths) != 1:
        raise ValueError(f"ambiguous temporal contracts for {relative_path}")
    contract = matching_paths[0]
    if source_sha256 != contract.sha256:
        raise ValueError(f"pinned temporal source sha256 mismatch for {relative_path}")
    expected_discontinuities = (contract.discontinuity_after_row,)
    if int(steps.shape[0]) != contract.raw_length:
        raise ValueError(f"pinned temporal source length mismatch for {relative_path}")
    if observed != expected_discontinuities:
        raise ValueError(
            f"pinned temporal discontinuity mismatch for {relative_path}: {observed}"
        )
    boundary = contract.discontinuity_after_row
    if (
        int(steps[boundary]) != contract.left_step
        or int(steps[boundary + 1]) != contract.right_step
    ):
        raise ValueError(f"pinned temporal step values mismatch for {relative_path}")

    selection = TemporalSelection(
        usable_start=contract.usable_start,
        usable_end=contract.usable_end,
        discontinuities_after_rows=expected_discontinuities,
        policy=PINNED_MAXIMAL_PREFIX_POLICY,
    )
    validate_temporal_selection(
        steps,
        selection=selection,
        source_label=relative_path,
    )
    return selection


def validate_declared_temporal_selection(
    *,
    relative_path: str,
    source_sha256: str | None,
    raw_length: int,
    split: str,
    selection: TemporalSelection,
    contracts: Sequence[PinnedTemporalContract] | None = None,
) -> None:
    """Validate persisted metadata without reopening a content-addressed source."""

    active_contracts = PINNED_TEMPORAL_CONTRACTS if contracts is None else contracts
    matches = tuple(
        contract
        for contract in active_contracts
        if contract.relative_path == relative_path
    )
    if selection.policy == STRICT_MONOTONIC_POLICY:
        if matches:
            raise ValueError(
                f"pinned temporal source lacks its policy: {relative_path}"
            )
        if (
            selection.usable_start != 0
            or selection.usable_end != raw_length
            or selection.discontinuities_after_rows
        ):
            raise ValueError(f"strict temporal selection is not full: {relative_path}")
        return

    if selection.policy != PINNED_MAXIMAL_PREFIX_POLICY:
        raise ValueError(f"unsupported temporal policy: {selection.policy}")
    if split != "train":
        raise ValueError("pinned temporal exceptions are restricted to train")
    if len(matches) != 1:
        raise ValueError(f"undeclared pinned temporal source: {relative_path}")
    contract = matches[0]
    expected = TemporalSelection(
        usable_start=contract.usable_start,
        usable_end=contract.usable_end,
        discontinuities_after_rows=(contract.discontinuity_after_row,),
        policy=PINNED_MAXIMAL_PREFIX_POLICY,
    )
    if (
        source_sha256 != contract.sha256
        or raw_length != contract.raw_length
        or selection != expected
    ):
        raise ValueError(f"pinned temporal metadata mismatch: {relative_path}")
