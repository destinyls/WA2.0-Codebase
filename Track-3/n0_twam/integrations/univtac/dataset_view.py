# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Immutable, content-addressed UniVTAC dataset views."""

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .manifest import MANIFEST_SCHEMA_VERSION, UniVTACDatasetManifest
from .schema import (
    TRACK31_TARGET_TASKS,
    UNIVTAC_ALL_TASKS,
    canonical_task_scope,
)

DATASET_VIEW_SCHEMA_VERSION = 1
DEFAULT_INTERNAL_SELECTION_SEED = "n0-twam-internal-dev-v1"

STAGE_A_DEV_VIEW_ID = "stage_a_dev719_v1"
INTERNAL_DEV_VIEW_ID = "internal_dev40_v1"
STAGE_A_FINAL_VIEW_ID = "stage_a_final759_v1"
STAGE_B_DEV_VIEW_ID = "stage_b_dev180_v1"
INTERNAL_TARGET_DEV_VIEW_ID = "internal_target_dev10_v1"
STAGE_B_FINAL_VIEW_ID = "stage_b_final190_v1"
FROZEN_TARGET_VIEW_ID = "frozen_target10_v1"
FROZEN_OTHER_VIEW_ID = "frozen_other30_v1"
QUARANTINE_VIEW_ID = "quarantine1_v1"

# The official public visuo-tactile leaderboard names Insert HDMI and Lift
# Bottle. Their exact five-episode projection is the single default evaluation
# cohort; the remaining frozen episodes are opt-in diagnostics only.
DEFAULT_UNIFIED_EVALUATION_VIEW_ID = FROZEN_TARGET_VIEW_ID
DEFAULT_UNIFIED_EVALUATION_TASKS = TRACK31_TARGET_TASKS
DEFAULT_UNIFIED_EVALUATION_EPISODE_IDS = (0, 1, 2, 3, 5)
DEFAULT_UNIFIED_EVALUATION_EPISODE_COUNT = 10
DIAGNOSTIC_EVALUATION_VIEW_IDS = (FROZEN_OTHER_VIEW_ID,)

STANDARD_VIEW_IDS = (
    STAGE_A_DEV_VIEW_ID,
    INTERNAL_DEV_VIEW_ID,
    STAGE_A_FINAL_VIEW_ID,
    STAGE_B_DEV_VIEW_ID,
    INTERNAL_TARGET_DEV_VIEW_ID,
    STAGE_B_FINAL_VIEW_ID,
    FROZEN_TARGET_VIEW_ID,
    FROZEN_OTHER_VIEW_ID,
    QUARANTINE_VIEW_ID,
)

_PHYSICAL_SPLITS = {
    "train759": "train",
    "frozen40": "validation",
    "quarantine1": "quarantine",
}
_VIEW_ROLES = frozenset(
    ("training", "internal_development", "frozen_evaluation", "quarantine")
)
_OTHER_TASKS = tuple(
    task for task in UNIVTAC_ALL_TASKS if task not in TRACK31_TARGET_TASKS
)


@dataclass(frozen=True)
class _StandardViewSpec:
    role: str
    physical_split: str
    tasks: tuple[str, ...]
    per_task_counts: tuple[tuple[str, int], ...]
    selection_method: str
    uses_selection_seed: bool


def _counts(**values: int) -> tuple[tuple[str, int], ...]:
    return tuple((task, values[task]) for task in UNIVTAC_ALL_TASKS if task in values)


_STANDARD_VIEW_SPECS: Mapping[str, _StandardViewSpec] = MappingProxyType(
    {
        STAGE_A_DEV_VIEW_ID: _StandardViewSpec(
            "training",
            "train759",
            UNIVTAC_ALL_TASKS,
            _counts(
                grasp_classify=89,
                insert_HDMI=90,
                insert_hole=90,
                insert_tube=90,
                lift_bottle=90,
                lift_can=90,
                pull_out_key=90,
                put_bottle_in_shelf=90,
            ),
            "exclude_internal_dev40",
            True,
        ),
        INTERNAL_DEV_VIEW_ID: _StandardViewSpec(
            "internal_development",
            "train759",
            UNIVTAC_ALL_TASKS,
            _counts(**{task: 5 for task in UNIVTAC_ALL_TASKS}),
            "sha256_per_task_lowest_5",
            True,
        ),
        STAGE_A_FINAL_VIEW_ID: _StandardViewSpec(
            "training",
            "train759",
            UNIVTAC_ALL_TASKS,
            _counts(
                grasp_classify=94,
                insert_HDMI=95,
                insert_hole=95,
                insert_tube=95,
                lift_bottle=95,
                lift_can=95,
                pull_out_key=95,
                put_bottle_in_shelf=95,
            ),
            "all_eligible_train",
            False,
        ),
        STAGE_B_DEV_VIEW_ID: _StandardViewSpec(
            "training",
            "train759",
            TRACK31_TARGET_TASKS,
            _counts(insert_HDMI=90, lift_bottle=90),
            "target_filter_of_stage_a_dev719",
            True,
        ),
        INTERNAL_TARGET_DEV_VIEW_ID: _StandardViewSpec(
            "internal_development",
            "train759",
            TRACK31_TARGET_TASKS,
            _counts(insert_HDMI=5, lift_bottle=5),
            "target_filter_of_internal_dev40",
            True,
        ),
        STAGE_B_FINAL_VIEW_ID: _StandardViewSpec(
            "training",
            "train759",
            TRACK31_TARGET_TASKS,
            _counts(insert_HDMI=95, lift_bottle=95),
            "target_filter_of_stage_a_final759",
            False,
        ),
        FROZEN_TARGET_VIEW_ID: _StandardViewSpec(
            "frozen_evaluation",
            "frozen40",
            TRACK31_TARGET_TASKS,
            _counts(insert_HDMI=5, lift_bottle=5),
            "target_filter_of_frozen40",
            False,
        ),
        FROZEN_OTHER_VIEW_ID: _StandardViewSpec(
            "frozen_evaluation",
            "frozen40",
            _OTHER_TASKS,
            _counts(**{task: 5 for task in _OTHER_TASKS}),
            "non_target_filter_of_frozen40",
            False,
        ),
        QUARANTINE_VIEW_ID: _StandardViewSpec(
            "quarantine",
            "quarantine1",
            ("grasp_classify",),
            _counts(grasp_classify=1),
            "fixed_quarantine_identity",
            False,
        ),
    }
)


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class DatasetViewEntry:
    """One source episode and its stable physical LeRobot episode ID."""

    relative_path: str
    realpath: str
    source_sha256: str
    task: str
    source_split: str
    source_episode_id: int
    lerobot_episode_id: int

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if (
            path.is_absolute()
            or len(path.parts) != 3
            or path.parts[0] != self.task
            or path.parts[1] != "clean"
            or not path.stem.isdecimal()
            or int(path.stem) != self.source_episode_id
        ):
            raise ValueError(f"invalid dataset view source path: {self.relative_path}")
        if not Path(self.realpath).is_absolute():
            raise ValueError("dataset view realpath must be absolute")
        if not _valid_sha256(self.source_sha256):
            raise ValueError("dataset view entry requires a lowercase source sha256")
        if self.source_split not in {"train", "validation", "quarantine"}:
            raise ValueError(f"invalid source split: {self.source_split!r}")
        if self.source_episode_id < 0 or self.lerobot_episode_id < 0:
            raise ValueError("dataset view episode IDs must be non-negative")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "realpath": self.realpath,
            "source_sha256": self.source_sha256,
            "task": self.task,
            "source_split": self.source_split,
            "source_episode_id": self.source_episode_id,
            "lerobot_episode_id": self.lerobot_episode_id,
        }


@dataclass(frozen=True)
class DatasetView:
    """A typed immutable view whose digest excludes storage location metadata."""

    view_id: str
    role: str
    physical_split: str
    source_manifest_sha256: str
    tasks: tuple[str, ...]
    entries: tuple[DatasetViewEntry, ...]
    selection_method: str
    selection_seed: str | int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", canonical_task_scope(self.tasks))
        if not self.view_id or not self.selection_method:
            raise ValueError("dataset view identity and selection method are required")
        if self.role not in _VIEW_ROLES:
            raise ValueError(f"unsupported dataset view role: {self.role!r}")
        if self.physical_split not in _PHYSICAL_SPLITS:
            raise ValueError(f"unsupported physical split: {self.physical_split!r}")
        if not _valid_sha256(self.source_manifest_sha256):
            raise ValueError("dataset view requires a source manifest sha256")
        if not self.entries:
            raise ValueError("dataset view entries must be non-empty")
        expected_source_split = _PHYSICAL_SPLITS[self.physical_split]
        for entry in self.entries:
            if entry.task not in self.tasks:
                raise ValueError(f"view entry task is outside scope: {entry.task!r}")
            if entry.source_split != expected_source_split:
                raise ValueError("view entry does not belong to its physical split")
        dimensions: dict[str, list[object]] = {
            "relative_path": [entry.relative_path for entry in self.entries],
            "realpath": [entry.realpath for entry in self.entries],
            "source_sha256": [entry.source_sha256 for entry in self.entries],
            "source_episode_id": [
                (entry.task, entry.source_episode_id) for entry in self.entries
            ],
            "lerobot_episode_id": [entry.lerobot_episode_id for entry in self.entries],
        }
        for label, values in dimensions.items():
            if len(values) != len(set(values)):
                raise ValueError(f"dataset view has duplicate {label}")

    @property
    def per_task_counts(self) -> dict[str, int]:
        return {
            task: sum(entry.task == task for entry in self.entries)
            for task in self.tasks
        }

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": DATASET_VIEW_SCHEMA_VERSION,
            "view_id": self.view_id,
            "role": self.role,
            "physical_split": self.physical_split,
            "source_manifest_sha256": self.source_manifest_sha256,
            "tasks": list(self.tasks),
            "entries": [entry.to_json_dict() for entry in self.entries],
            "per_task_counts": self.per_task_counts,
            "selection_method": self.selection_method,
            "selection_seed": self.selection_seed,
        }

    @property
    def view_sha256(self) -> str:
        return _sha256(self._canonical_payload())

    def to_json_dict(self) -> dict[str, object]:
        return {**self._canonical_payload(), "view_sha256": self.view_sha256}


def load_dataset_view(path: Path) -> DatasetView:
    """Load one view and reject any post-hoc edit."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read dataset view: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("dataset view must be a JSON object")
    canonical_keys = (
        "schema_version",
        "view_id",
        "role",
        "physical_split",
        "source_manifest_sha256",
        "tasks",
        "entries",
        "per_task_counts",
        "selection_method",
        "selection_seed",
    )
    if set(payload) != set(canonical_keys) | {"view_sha256"}:
        raise ValueError("dataset view fields do not match schema v1")
    canonical = {key: payload[key] for key in canonical_keys}
    if payload["schema_version"] != DATASET_VIEW_SCHEMA_VERSION:
        raise ValueError("unsupported dataset view schema version")
    if payload["view_sha256"] != _sha256(canonical):
        raise ValueError("dataset view sha256 verification failed")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("dataset view entries must be a list")
    entries = tuple(DatasetViewEntry(**item) for item in raw_entries)
    raw_tasks = payload["tasks"]
    if not isinstance(raw_tasks, list):
        raise ValueError("dataset view tasks must be a list")
    view = DatasetView(
        view_id=str(payload["view_id"]),
        role=str(payload["role"]),
        physical_split=str(payload["physical_split"]),
        source_manifest_sha256=str(payload["source_manifest_sha256"]),
        tasks=tuple(str(task) for task in raw_tasks),
        entries=entries,
        selection_method=str(payload["selection_method"]),
        selection_seed=payload["selection_seed"],
    )
    if view.to_json_dict() != payload:
        raise ValueError("dataset view canonical payload is inconsistent")
    return view


def content_addressed_sample_seed(
    entry: DatasetViewEntry,
    *,
    seed: int,
    epoch: int,
) -> int:
    """Derive an order-independent uint64 seed from immutable episode content."""

    if seed < 0 or epoch < 0:
        raise ValueError("sample seed and epoch must be non-negative")
    payload = f"{seed}\0{epoch}\0{entry.task}\0{entry.source_sha256}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_content_addressed_crop_start(
    entry: DatasetViewEntry,
    *,
    seed: int,
    epoch: int,
    num_latent_frames: int,
    max_latent_frames: int,
) -> int:
    """Choose one order-independent crop per source episode and epoch."""
    if num_latent_frames <= max_latent_frames:
        raise ValueError("content-addressed crop requires a truncated sequence")
    if max_latent_frames <= 0:
        raise ValueError("max_latent_frames must be positive")
    crop_count = num_latent_frames - max_latent_frames + 1
    return (
        content_addressed_sample_seed(
            entry,
            seed=seed,
            epoch=epoch,
        )
        % crop_count
    )


def _selection_score(entry: DatasetViewEntry, seed: str | int) -> str:
    payload = f"{seed}\0{entry.task}\0{entry.source_sha256}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_view_entries(
    manifest: UniVTACDatasetManifest,
    *,
    verify_sources: bool,
) -> dict[str, tuple[DatasetViewEntry, ...]]:
    root = Path(manifest.data_root).resolve(strict=verify_sources)
    source_hashes = [entry.sha256 for entry in manifest.entries]
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError("standard source universe has duplicate content sha256")
    task_rank = {task: index for index, task in enumerate(manifest.tasks)}
    result: dict[str, tuple[DatasetViewEntry, ...]] = {}
    for split in ("train", "validation", "quarantine"):
        records = sorted(
            (entry for entry in manifest.entries if entry.split == split),
            key=lambda entry: (
                task_rank[entry.task],
                entry.episode_id,
                entry.relative_path,
            ),
        )
        view_entries: list[DatasetViewEntry] = []
        for episode_index, record in enumerate(records):
            expected = (root / record.relative_path).resolve(strict=verify_sources)
            actual = record.absolute_path.resolve(strict=verify_sources)
            if expected != actual or root not in actual.parents:
                raise ValueError(f"manifest realpath drift: {record.relative_path}")
            if verify_sources:
                from .hdf5_reader import verify_source_record

                verify_source_record(record)
            assert record.sha256 is not None
            view_entries.append(
                DatasetViewEntry(
                    relative_path=record.relative_path,
                    realpath=str(actual),
                    source_sha256=record.sha256,
                    task=record.task,
                    source_split=split,
                    source_episode_id=record.episode_id,
                    lerobot_episode_id=episode_index,
                )
            )
        result[split] = tuple(view_entries)
    return result


def _make_view(
    manifest: UniVTACDatasetManifest,
    *,
    view_id: str,
    role: str,
    physical_split: str,
    tasks: tuple[str, ...],
    entries: Sequence[DatasetViewEntry],
    selection_method: str,
    selection_seed: str | int | None = None,
) -> DatasetView:
    task_rank = {task: index for index, task in enumerate(UNIVTAC_ALL_TASKS)}
    ordered = tuple(
        sorted(
            entries,
            key=lambda entry: (
                task_rank[entry.task],
                entry.source_episode_id,
                entry.relative_path,
            ),
        )
    )
    return DatasetView(
        view_id=view_id,
        role=role,
        physical_split=physical_split,
        source_manifest_sha256=manifest.manifest_sha256,
        tasks=tasks,
        entries=ordered,
        selection_method=selection_method,
        selection_seed=selection_seed,
    )


def build_standard_dataset_views(
    manifest: UniVTACDatasetManifest,
    *,
    selection_seed: str | int = DEFAULT_INTERNAL_SELECTION_SEED,
    verify_sources: bool = True,
) -> dict[str, DatasetView]:
    """Build the frozen 719/40, 759, 180/10, 190, 10/30, and quarantine views."""

    if manifest.schema_version != MANIFEST_SCHEMA_VERSION or manifest.is_read_only:
        raise ValueError(
            "standard dataset views require a writeable schema-v4 manifest"
        )
    if manifest.tasks != UNIVTAC_ALL_TASKS:
        raise ValueError("standard dataset views require the explicit eight-task scope")
    expected_counts = {
        "train": {
            task: 94 if task == "grasp_classify" else 95 for task in UNIVTAC_ALL_TASKS
        },
        "validation": {task: 5 for task in UNIVTAC_ALL_TASKS},
        "quarantine": {
            task: 1 if task == "grasp_classify" else 0 for task in UNIVTAC_ALL_TASKS
        },
    }
    if manifest.task_counts != expected_counts:
        raise ValueError("source universe must have exact 759/40/1 per-task counts")
    by_split = _manifest_view_entries(manifest, verify_sources=verify_sources)
    quarantine = by_split["quarantine"]
    if len(quarantine) != 1 or quarantine[0].relative_path != (
        "grasp_classify/clean/90.hdf5"
    ):
        raise ValueError("quarantine identity must be grasp_classify/clean/90.hdf5")
    internal: list[DatasetViewEntry] = []
    for task in UNIVTAC_ALL_TASKS:
        candidates = [entry for entry in by_split["train"] if entry.task == task]
        internal.extend(
            sorted(
                candidates,
                key=lambda entry: (
                    _selection_score(entry, selection_seed),
                    entry.source_sha256,
                ),
            )[:5]
        )
    internal_paths = {entry.relative_path for entry in internal}
    stage_a_dev = [
        entry
        for entry in by_split["train"]
        if entry.relative_path not in internal_paths
    ]
    target_internal = [
        entry for entry in internal if entry.task in TRACK31_TARGET_TASKS
    ]
    stage_b_dev = [entry for entry in stage_a_dev if entry.task in TRACK31_TARGET_TASKS]
    target_final = [
        entry for entry in by_split["train"] if entry.task in TRACK31_TARGET_TASKS
    ]
    frozen_target = [
        entry for entry in by_split["validation"] if entry.task in TRACK31_TARGET_TASKS
    ]
    frozen_other = [
        entry for entry in by_split["validation"] if entry.task in _OTHER_TASKS
    ]
    specs = (
        (
            STAGE_A_DEV_VIEW_ID,
            "training",
            "train759",
            UNIVTAC_ALL_TASKS,
            stage_a_dev,
            "exclude_internal_dev40",
            selection_seed,
        ),
        (
            INTERNAL_DEV_VIEW_ID,
            "internal_development",
            "train759",
            UNIVTAC_ALL_TASKS,
            internal,
            "sha256_per_task_lowest_5",
            selection_seed,
        ),
        (
            STAGE_A_FINAL_VIEW_ID,
            "training",
            "train759",
            UNIVTAC_ALL_TASKS,
            by_split["train"],
            "all_eligible_train",
            None,
        ),
        (
            STAGE_B_DEV_VIEW_ID,
            "training",
            "train759",
            TRACK31_TARGET_TASKS,
            stage_b_dev,
            "target_filter_of_stage_a_dev719",
            selection_seed,
        ),
        (
            INTERNAL_TARGET_DEV_VIEW_ID,
            "internal_development",
            "train759",
            TRACK31_TARGET_TASKS,
            target_internal,
            "target_filter_of_internal_dev40",
            selection_seed,
        ),
        (
            STAGE_B_FINAL_VIEW_ID,
            "training",
            "train759",
            TRACK31_TARGET_TASKS,
            target_final,
            "target_filter_of_stage_a_final759",
            None,
        ),
        (
            FROZEN_TARGET_VIEW_ID,
            "frozen_evaluation",
            "frozen40",
            TRACK31_TARGET_TASKS,
            frozen_target,
            "target_filter_of_frozen40",
            None,
        ),
        (
            FROZEN_OTHER_VIEW_ID,
            "frozen_evaluation",
            "frozen40",
            _OTHER_TASKS,
            frozen_other,
            "non_target_filter_of_frozen40",
            None,
        ),
        (
            QUARANTINE_VIEW_ID,
            "quarantine",
            "quarantine1",
            ("grasp_classify",),
            quarantine,
            "fixed_quarantine_identity",
            None,
        ),
    )
    views = {
        view_id: _make_view(
            manifest,
            view_id=view_id,
            role=role,
            physical_split=physical_split,
            tasks=tasks,
            entries=entries,
            selection_method=method,
            selection_seed=seed,
        )
        for view_id, role, physical_split, tasks, entries, method, seed in specs
    }
    verify_standard_view_set(views)
    return views


_PAIR_ORDER = tuple(combinations(STANDARD_VIEW_IDS, 2))
_EXPECTED_INTERSECTIONS: dict[tuple[str, str], str | None] = {
    pair: None for pair in _PAIR_ORDER
}
_EXPECTED_INTERSECTIONS.update(
    {
        (STAGE_A_DEV_VIEW_ID, STAGE_A_FINAL_VIEW_ID): STAGE_A_DEV_VIEW_ID,
        (STAGE_A_DEV_VIEW_ID, STAGE_B_DEV_VIEW_ID): STAGE_B_DEV_VIEW_ID,
        (STAGE_A_DEV_VIEW_ID, STAGE_B_FINAL_VIEW_ID): STAGE_B_DEV_VIEW_ID,
        (INTERNAL_DEV_VIEW_ID, STAGE_A_FINAL_VIEW_ID): INTERNAL_DEV_VIEW_ID,
        (
            INTERNAL_DEV_VIEW_ID,
            INTERNAL_TARGET_DEV_VIEW_ID,
        ): INTERNAL_TARGET_DEV_VIEW_ID,
        (INTERNAL_DEV_VIEW_ID, STAGE_B_FINAL_VIEW_ID): INTERNAL_TARGET_DEV_VIEW_ID,
        (STAGE_A_FINAL_VIEW_ID, STAGE_B_DEV_VIEW_ID): STAGE_B_DEV_VIEW_ID,
        (
            STAGE_A_FINAL_VIEW_ID,
            INTERNAL_TARGET_DEV_VIEW_ID,
        ): INTERNAL_TARGET_DEV_VIEW_ID,
        (STAGE_A_FINAL_VIEW_ID, STAGE_B_FINAL_VIEW_ID): STAGE_B_FINAL_VIEW_ID,
        (STAGE_B_DEV_VIEW_ID, STAGE_B_FINAL_VIEW_ID): STAGE_B_DEV_VIEW_ID,
        (
            INTERNAL_TARGET_DEV_VIEW_ID,
            STAGE_B_FINAL_VIEW_ID,
        ): INTERNAL_TARGET_DEV_VIEW_ID,
    }
)
ALLOWED_OVERLAP_MATRIX: Mapping[tuple[str, str], str | None] = MappingProxyType(
    _EXPECTED_INTERSECTIONS
)


def _identity_set(view: DatasetView, dimension: str) -> set[object]:
    if dimension == "relative_path":
        return {entry.relative_path for entry in view.entries}
    if dimension == "realpath":
        return {entry.realpath for entry in view.entries}
    if dimension == "source_sha256":
        return {entry.source_sha256 for entry in view.entries}
    return {(view.physical_split, entry.lerobot_episode_id) for entry in view.entries}


def verify_standard_view_set(views: Mapping[str, DatasetView]) -> None:
    """Enforce exact counts and every allowed/disallowed cross-view overlap."""

    if set(views) != set(STANDARD_VIEW_IDS):
        raise ValueError("standard dataset view set is incomplete or has extras")
    for pair, intersection_view_id in ALLOWED_OVERLAP_MATRIX.items():
        left, right = (views[view_id] for view_id in pair)
        for dimension in ("relative_path", "realpath", "source_sha256", "episode_id"):
            actual = _identity_set(left, dimension) & _identity_set(right, dimension)
            expected = (
                set()
                if intersection_view_id is None
                else _identity_set(views[intersection_view_id], dimension)
            )
            if actual != expected:
                raise ValueError(
                    f"view overlap for {dimension} violates allowed matrix: {pair}"
                )
        actual_entries = set(left.entries) & set(right.entries)
        expected_entries = (
            set()
            if intersection_view_id is None
            else set(views[intersection_view_id].entries)
        )
        if actual_entries != expected_entries:
            raise ValueError(
                "view overlap for composite episode identity violates allowed "
                f"matrix: {pair}"
            )

    seeded_values: set[str | int] = set()
    for view_id, spec in _STANDARD_VIEW_SPECS.items():
        view = views[view_id]
        if view.view_id != view_id:
            raise ValueError(f"dataset view count/identity mismatch: {view_id}")
        if (
            view.role != spec.role
            or view.physical_split != spec.physical_split
            or view.tasks != spec.tasks
            or view.selection_method != spec.selection_method
        ):
            raise ValueError(f"dataset view declarative contract mismatch: {view_id}")
        expected_counts = dict(spec.per_task_counts)
        if view.per_task_counts != expected_counts:
            raise ValueError(f"dataset view per-task counts mismatch: {view_id}")
        if spec.uses_selection_seed:
            if view.selection_seed is None:
                raise ValueError(f"dataset view selection seed is missing: {view_id}")
            seeded_values.add(view.selection_seed)
        elif view.selection_seed is not None:
            raise ValueError(f"dataset view must not carry a selection seed: {view_id}")
    if len(seeded_values) != 1:
        raise ValueError("seeded standard views must share one selection seed")

    manifest_hashes = {view.source_manifest_sha256 for view in views.values()}
    if len(manifest_hashes) != 1:
        raise ValueError("standard dataset views do not share one source manifest")

    final_entries = set(views[STAGE_A_FINAL_VIEW_ID].entries)
    internal_entries = set(views[INTERNAL_DEV_VIEW_ID].entries)
    selection_seed = views[INTERNAL_DEV_VIEW_ID].selection_seed
    assert selection_seed is not None
    expected_internal = {
        entry
        for task in UNIVTAC_ALL_TASKS
        for entry in sorted(
            (candidate for candidate in final_entries if candidate.task == task),
            key=lambda candidate: (
                _selection_score(candidate, selection_seed),
                candidate.source_sha256,
            ),
        )[:5]
    }
    if internal_entries != expected_internal:
        raise ValueError("internal_dev40 violates deterministic per-task selection")

    dev_entries = set(views[STAGE_A_DEV_VIEW_ID].entries)
    if dev_entries != final_entries - internal_entries:
        raise ValueError("stage_a_dev719 must exactly complement internal_dev40")
    expected_filters = {
        STAGE_B_DEV_VIEW_ID: {
            entry for entry in dev_entries if entry.task in TRACK31_TARGET_TASKS
        },
        INTERNAL_TARGET_DEV_VIEW_ID: {
            entry for entry in internal_entries if entry.task in TRACK31_TARGET_TASKS
        },
        STAGE_B_FINAL_VIEW_ID: {
            entry for entry in final_entries if entry.task in TRACK31_TARGET_TASKS
        },
    }
    for view_id, expected_entries in expected_filters.items():
        if set(views[view_id].entries) != expected_entries:
            raise ValueError(
                f"dataset child view is not an exact parent filter: {view_id}"
            )
    default_roster = tuple(
        (entry.task, entry.source_episode_id)
        for entry in views[DEFAULT_UNIFIED_EVALUATION_VIEW_ID].entries
    )
    expected_default_roster = tuple(
        (task, episode_id)
        for task in DEFAULT_UNIFIED_EVALUATION_TASKS
        for episode_id in DEFAULT_UNIFIED_EVALUATION_EPISODE_IDS
    )
    if default_roster != expected_default_roster:
        raise ValueError("default unified evaluation roster is not canonical Target-10")
    quarantine_entries = views[QUARANTINE_VIEW_ID].entries
    if (
        len(quarantine_entries) != 1
        or quarantine_entries[0].relative_path != "grasp_classify/clean/90.hdf5"
    ):
        raise ValueError("quarantine view does not contain its fixed episode")
