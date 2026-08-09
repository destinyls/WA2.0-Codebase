# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Immutable action-space contracts shared by training and deployment."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ActionSpec:
    """Versioned description of one physical action representation."""

    name: str
    revision: str
    dim: int
    state_dim: int
    semantics: str
    normalization: str
    padding_policy: str
    wire_layout: str
    server_output_format: str
    channel_names: tuple[str, ...]
    active_channel_ids: tuple[int, ...]
    gripper_indices: tuple[int, ...]
    lower_bounds: tuple[float, ...] | None = None
    upper_bounds: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ActionSpec.name must be non-empty")
        if not self.revision:
            raise ValueError("ActionSpec.revision must be non-empty")
        if self.dim <= 0:
            raise ValueError("ActionSpec.dim must be positive")
        if self.state_dim <= 0:
            raise ValueError("ActionSpec.state_dim must be positive")
        for field_name in (
            "semantics",
            "normalization",
            "padding_policy",
            "wire_layout",
            "server_output_format",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"ActionSpec.{field_name} must be non-empty")
        if len(self.channel_names) != self.dim:
            raise ValueError(
                f"channel_names must contain {self.dim} entries, "
                f"got {len(self.channel_names)}"
            )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be unique")
        self._validate_indices("active_channel_ids", self.active_channel_ids)
        self._validate_indices("gripper_indices", self.gripper_indices)
        if not set(self.gripper_indices).issubset(self.active_channel_ids):
            raise ValueError("gripper_indices must be active channels")
        self._validate_bounds()

    def _validate_indices(self, field_name: str, values: tuple[int, ...]) -> None:
        if len(set(values)) != len(values):
            raise ValueError(f"{field_name} must not contain duplicates")
        if any(value < 0 or value >= self.dim for value in values):
            raise ValueError(f"{field_name} must be within [0, {self.dim})")

    def _validate_bounds(self) -> None:
        if (self.lower_bounds is None) != (self.upper_bounds is None):
            raise ValueError("lower_bounds and upper_bounds must be set together")
        if self.lower_bounds is None or self.upper_bounds is None:
            return
        if len(self.lower_bounds) != self.dim or len(self.upper_bounds) != self.dim:
            raise ValueError(f"physical bounds must contain {self.dim} entries")
        for lower, upper in zip(self.lower_bounds, self.upper_bounds):
            if not isfinite(lower) or not isfinite(upper):
                raise ValueError("physical bounds must be finite")
            if lower > upper:
                raise ValueError("each lower bound must not exceed its upper bound")

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-safe action contract for checkpoints and reports."""

        return {
            "name": self.name,
            "revision": self.revision,
            "dim": self.dim,
            "state_dim": self.state_dim,
            "semantics": self.semantics,
            "normalization": self.normalization,
            "padding_policy": self.padding_policy,
            "wire_layout": self.wire_layout,
            "server_output_format": self.server_output_format,
            "channel_names": list(self.channel_names),
            "active_channel_ids": list(self.active_channel_ids),
            "gripper_indices": list(self.gripper_indices),
            "lower_bounds": (
                None if self.lower_bounds is None else list(self.lower_bounds)
            ),
            "upper_bounds": (
                None if self.upper_bounds is None else list(self.upper_bounds)
            ),
        }
