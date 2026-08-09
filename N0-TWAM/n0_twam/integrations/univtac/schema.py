# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Frozen UniVTAC source and converted feature names for Track 3.1."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping, Sequence

from .temporal_contract import (
    PINNED_MAXIMAL_PREFIX_POLICY,
    STRICT_MONOTONIC_POLICY,
    TemporalSelection,
)

UNIVTAC_ALL_TASKS: Final[tuple[str, ...]] = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)
TRACK31_TARGET_TASKS: Final[tuple[str, ...]] = ("insert_HDMI", "lift_bottle")
# Deprecated compatibility alias for the original two-task artifacts. New
# eight-task code must name ``UNIVTAC_ALL_TASKS`` explicitly.
TRACK31_TASKS = TRACK31_TARGET_TASKS
LEGACY_TRACK31_TASKS = TRACK31_TARGET_TASKS

CANONICAL_TASK_PROMPT_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "grasp_classify": (
            "The robotic arm grasps an object and classifies it using tactile "
            "feedback, inferring object properties from the contact signals."
        ),
        "insert_HDMI": (
            "The robotic arm aligns an HDMI connector and inserts it into the "
            "port with careful tactile guidance and fine pose correction."
        ),
        "insert_hole": (
            "The robotic arm performs precise peg-in-hole insertion, using "
            "tactile feedback to align the peg and complete the insertion "
            "accurately."
        ),
        "insert_tube": (
            "The robotic arm inserts a tube into a fixture, adjusting alignment "
            "and contact force to achieve a secure fit."
        ),
        "lift_bottle": (
            "The robotic arm grasps a bottle and lifts it off a surface near a "
            "wall, maintaining careful contact control to avoid collisions."
        ),
        "lift_can": (
            "The robotic arm grasps a cylindrical can and lifts it smoothly from "
            "the surface with precise force and grasp stabilization."
        ),
        "pull_out_key": (
            "The robotic arm extracts a key from a lock by applying controlled "
            "pulling force while maintaining stable grasp and alignment."
        ),
        "put_bottle_in_shelf": (
            "The robotic arm places a bottle onto a shelf with careful "
            "positioning, controlled release, and stable placement."
        ),
    }
)
# Less verbose alias for callers that already carry the UniVTAC namespace.
UNIVTAC_TASK_PROMPTS = CANONICAL_TASK_PROMPT_MAP
TASK_PROMPT_MAP_SHA256: Final[str] = hashlib.sha256(
    json.dumps(
        dict(CANONICAL_TASK_PROMPT_MAP),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

ACTION_SCHEMA = "qpos8_next_step"
DEFAULT_SOURCE_FPS = 10.0
LEGACY_UNIVTAC_JPEG_CONTRACT = "opencv_imencode_rgb_input_v1"
STANDARD_JPEG_RGB_CONTRACT = "standard_jpeg_rgb_v2"
SOURCE_IMAGE_ENCODING_CONTRACT = LEGACY_UNIVTAC_JPEG_CONTRACT
OUTPUT_COLOR_SPACE = "RGB"

JOINT_PATH = "embodiment/joint"
STEP_PATH = "step"
IMAGE_PATHS = (
    ("observation.images.top", "observation/head/rgb"),
    ("observation.images.wrist_l", "observation/wrist/rgb"),
    ("observation.images.tactile_a", "tactile/left_gsmini/rgb_marker"),
    ("observation.images.tactile_b", "tactile/right_gsmini/rgb_marker"),
)
REQUIRED_HDF5_PATHS = (JOINT_PATH,) + tuple(path for _, path in IMAGE_PATHS)


def canonical_task_scope(tasks: Sequence[str]) -> tuple[str, ...]:
    """Return a unique supported task scope in canonical UniVTAC order."""

    requested = tuple(tasks)
    if not requested:
        raise ValueError("task scope must be non-empty")
    if len(requested) != len(set(requested)):
        raise ValueError("task scope contains duplicates")
    unsupported = sorted(set(requested) - set(UNIVTAC_ALL_TASKS))
    if unsupported:
        raise ValueError(f"unsupported UniVTAC tasks: {unsupported}")
    selected = set(requested)
    return tuple(task for task in UNIVTAC_ALL_TASKS if task in selected)


@dataclass(frozen=True)
class UniVTACEpisodeRecord:
    """Audited metadata for one immutable source HDF5 episode."""

    relative_path: str
    absolute_path: Path
    task: str
    split: str
    length: int
    length_source: str
    joint_shape: tuple[int, ...]
    image_shapes: tuple[tuple[str, tuple[int, int, int]], ...]
    size_bytes: int
    sha256: str | None
    usable_start: int = 0
    usable_end: int | None = None
    step_discontinuities_after_rows: tuple[int, ...] = ()
    temporal_policy: str = STRICT_MONOTONIC_POLICY

    def __post_init__(self) -> None:
        usable_end = self.length if self.usable_end is None else self.usable_end
        object.__setattr__(self, "usable_end", usable_end)
        if not 0 <= self.usable_start < usable_end <= self.length:
            raise ValueError("episode usable source range is invalid")
        if usable_end - self.usable_start < 2:
            raise ValueError("episode usable source range must contain two rows")
        if tuple(sorted(set(self.step_discontinuities_after_rows))) != (
            self.step_discontinuities_after_rows
        ):
            raise ValueError("episode step discontinuities must be unique and sorted")
        if any(
            index < 0 or index >= self.length - 1
            for index in self.step_discontinuities_after_rows
        ):
            raise ValueError("episode step discontinuity index is out of bounds")
        if self.temporal_policy == STRICT_MONOTONIC_POLICY:
            if (
                self.usable_start != 0
                or usable_end != self.length
                or self.step_discontinuities_after_rows
            ):
                raise ValueError("strict episode must retain the full monotonic source")
        elif self.temporal_policy == PINNED_MAXIMAL_PREFIX_POLICY:
            if self.usable_start != 0 or self.step_discontinuities_after_rows != (
                usable_end - 1,
            ):
                raise ValueError("pinned prefix must end at its sole discontinuity")
        else:
            raise ValueError(f"unsupported temporal policy: {self.temporal_policy}")

    @property
    def usable_length(self) -> int:
        """Return retained raw rows after temporal contract enforcement."""

        assert self.usable_end is not None
        return self.usable_end - self.usable_start

    @property
    def converted_length(self) -> int:
        """Return valid next-step rows emitted by the converter."""

        return self.usable_length - 1

    @property
    def temporal_selection(self) -> TemporalSelection:
        """Return the immutable selection consumed by downstream readers."""

        assert self.usable_end is not None
        return TemporalSelection(
            usable_start=self.usable_start,
            usable_end=self.usable_end,
            discontinuities_after_rows=self.step_discontinuities_after_rows,
            policy=self.temporal_policy,
        )

    @property
    def episode_id(self) -> int:
        """Return the numeric source episode ID encoded by the filename."""

        stem = Path(self.relative_path).stem
        if not stem.isdecimal():
            raise ValueError(
                f"source episode filename must be numeric: {self.relative_path}"
            )
        return int(stem)

    @property
    def realpath(self) -> str:
        """Return the normalized audited source path without following it anew."""

        return str(self.absolute_path.resolve(strict=False))

    def to_json_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "realpath": self.realpath,
            "episode_id": self.episode_id,
            "task": self.task,
            "split": self.split,
            "length": self.length,
            "length_source": self.length_source,
            "joint_shape": list(self.joint_shape),
            "image_shapes": {name: list(shape) for name, shape in self.image_shapes},
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "usable_source_range": [self.usable_start, self.usable_end],
            "step_discontinuities_after_rows": list(
                self.step_discontinuities_after_rows
            ),
            "temporal_policy": self.temporal_policy,
        }
