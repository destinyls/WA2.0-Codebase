# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Versioned, content-addressed UniVTAC split manifests."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .schema import (
    ACTION_SCHEMA,
    OUTPUT_COLOR_SPACE,
    SOURCE_IMAGE_ENCODING_CONTRACT,
    TRACK31_TASKS,
    UniVTACEpisodeRecord,
    canonical_task_scope,
)
from .temporal_contract import validate_declared_temporal_selection

MANIFEST_SCHEMA_VERSION = 4
READABLE_MANIFEST_SCHEMA_VERSIONS = frozenset((1, 2, 3, 4))
_SPLIT_ORDER = {"train": 0, "validation": 1, "quarantine": 2}
_CANONICAL_KEYS = (
    "schema_version",
    "action_schema",
    "source_image_encoding_contract",
    "output_color_space",
    "tasks",
    "entries",
)
_SUPPLEMENTAL_KEYS = frozenset(
    ("data_root", "manifest_sha256", "split_counts", "task_counts")
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_sort_key(relative_path: str) -> tuple[str, int, str]:
    path = Path(relative_path)
    episode_id = int(path.stem) if path.stem.isdecimal() else 2**31 - 1
    return path.parts[0], episode_id, relative_path


def _validate_declared_relative_path(relative_path: str) -> tuple[str, int]:
    path = Path(relative_path)
    if path.is_absolute() or len(path.parts) != 3 or path.parts[1] != "clean":
        raise ValueError(
            "UniVTAC path must match <task>/clean/<episode_id>.hdf5: "
            f"{relative_path}"
        )
    if path.suffix not in {".hdf5", ".h5"} or not path.stem.isdecimal():
        raise ValueError(f"invalid UniVTAC episode path: {relative_path}")
    return path.parts[0], int(path.stem)


def load_declared_paths(
    manifest_path: Path,
    *,
    tasks: Sequence[str] = TRACK31_TASKS,
) -> tuple[str, ...]:
    """Read UniVTAC-style ``[{hdf5_path, task}, ...]`` declarations."""

    task_scope = canonical_task_scope(tasks)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read manifest JSON: {manifest_path}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must contain a JSON list: {manifest_path}")
    selected: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry {index} must be an object")
        relative_path = item.get("hdf5_path")
        declared_task = item.get("task")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Manifest entry {index} has invalid hdf5_path")
        path_task, _ = _validate_declared_relative_path(relative_path)
        if declared_task is not None and str(declared_task) != path_task:
            raise ValueError(
                f"Manifest entry {index} task does not match its path: "
                f"{declared_task!r} vs {path_task!r}"
            )
        if path_task in task_scope:
            selected.append(relative_path)
    if len(selected) != len(set(selected)):
        raise ValueError(f"Manifest contains duplicate selected paths: {manifest_path}")
    return tuple(sorted(selected, key=_path_sort_key))


def discover_source_paths(
    data_root: Path,
    *,
    tasks: Sequence[str] = TRACK31_TASKS,
) -> tuple[str, ...]:
    """Discover HDF5 episodes for an explicit supported task scope."""

    root = data_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    task_scope = canonical_task_scope(tasks)
    discovered: list[str] = []
    for task in task_scope:
        clean_dir = root / task / "clean"
        if not clean_dir.is_dir():
            raise NotADirectoryError(clean_dir)
        for suffix in ("*.hdf5", "*.h5"):
            for path in clean_dir.glob(suffix):
                resolved = path.resolve(strict=True)
                if root not in resolved.parents:
                    raise ValueError(f"source path escapes data root: {path}")
                relative_path = resolved.relative_to(root).as_posix()
                _validate_declared_relative_path(relative_path)
                discovered.append(relative_path)
    unique_paths = tuple(sorted(set(discovered), key=_path_sort_key))
    if len(unique_paths) != len(discovered):
        raise ValueError("source discovery resolves multiple paths to one episode")
    if not unique_paths:
        raise ValueError(f"No UniVTAC HDF5 episodes found below {root}")
    return unique_paths


def derive_split_paths(
    *,
    data_root: Path,
    validation_manifest_path: Path,
    quarantine_manifest_path: Path | None = None,
    tasks: Sequence[str] = TRACK31_TASKS,
) -> dict[str, tuple[str, ...]]:
    """Derive train paths as source minus declared validation/quarantine paths."""

    task_scope = canonical_task_scope(tasks)
    source_paths = discover_source_paths(data_root, tasks=task_scope)
    validation_paths = load_declared_paths(
        validation_manifest_path,
        tasks=task_scope,
    )
    quarantine_paths = (
        ()
        if quarantine_manifest_path is None
        else load_declared_paths(quarantine_manifest_path, tasks=task_scope)
    )
    source_set = set(source_paths)
    missing = sorted((set(validation_paths) | set(quarantine_paths)) - source_set)
    if missing:
        raise ValueError(
            "Declared split paths are absent from source: " + ", ".join(missing)
        )
    if set(validation_paths) & set(quarantine_paths):
        raise ValueError("validation and quarantine paths overlap")
    train_paths = tuple(
        sorted(
            source_set - set(validation_paths) - set(quarantine_paths),
            key=_path_sort_key,
        )
    )
    return {
        "train": train_paths,
        "validation": validation_paths,
        "quarantine": quarantine_paths,
    }


def _identity_values(
    record: UniVTACEpisodeRecord,
) -> tuple[str, str, str, tuple[str, int]]:
    if record.sha256 is None or len(record.sha256) != 64:
        raise ValueError(f"record requires a source sha256: {record.relative_path}")
    path_task, episode_id = _validate_declared_relative_path(record.relative_path)
    if path_task != record.task:
        raise ValueError(
            f"record task does not match path: {record.task!r} vs {path_task!r}"
        )
    return (
        record.relative_path,
        record.realpath,
        record.sha256,
        (record.task, episode_id),
    )


def _validate_record_identities(
    records: Sequence[UniVTACEpisodeRecord],
    *,
    tasks: Sequence[str],
) -> None:
    task_scope = canonical_task_scope(tasks)
    dimensions: dict[str, list[object]] = {
        "relative_path": [],
        "realpath": [],
        "episode ID": [],
        "content SHA-256": [],
    }
    for record in records:
        if record.split not in _SPLIT_ORDER:
            raise ValueError(f"unsupported record split: {record.split!r}")
        if record.task not in task_scope:
            raise ValueError(f"record task is outside manifest scope: {record.task!r}")
        relative_path, realpath, source_sha256, episode_identity = _identity_values(
            record
        )
        for values, identity in zip(
            dimensions.values(),
            (relative_path, realpath, episode_identity, source_sha256),
        ):
            values.append(identity)
    for label, values in dimensions.items():
        if len(values) != len(set(values)):
            if label == "content SHA-256":
                raise ValueError(
                    "dataset content overlap is non-empty (content SHA-256 overlap)"
                )
            raise ValueError(f"dataset {label} overlap is non-empty")


@dataclass(frozen=True)
class UniVTACDatasetManifest:
    """Frozen source universe with explicit tasks and four-way identities."""

    data_root: str
    entries: tuple[UniVTACEpisodeRecord, ...]
    tasks: tuple[str, ...] = TRACK31_TASKS
    schema_version: int = MANIFEST_SCHEMA_VERSION
    _persisted_manifest_sha256: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        canonical_tasks = canonical_task_scope(self.tasks)
        object.__setattr__(self, "tasks", canonical_tasks)
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "legacy dataset manifests are read-only and must be loaded "
                "from a verified artifact"
            )
        if not self.entries:
            raise ValueError("dataset manifest entries must be non-empty")
        _validate_record_identities(self.entries, tasks=canonical_tasks)
        for entry in self.entries:
            validate_declared_temporal_selection(
                relative_path=entry.relative_path,
                source_sha256=entry.sha256,
                raw_length=entry.length,
                split=entry.split,
                selection=entry.temporal_selection,
            )
        if self._persisted_manifest_sha256 is not None:
            raise ValueError("schema-v4 manifest cannot reuse a legacy identity")

    @property
    def is_read_only(self) -> bool:
        return self.schema_version < MANIFEST_SCHEMA_VERSION

    def _canonical_payload(self) -> dict[str, object]:
        if self.is_read_only:
            raise ValueError("legacy dataset manifests are read-only")
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "action_schema": ACTION_SCHEMA,
            "source_image_encoding_contract": SOURCE_IMAGE_ENCODING_CONTRACT,
            "output_color_space": OUTPUT_COLOR_SPACE,
            "tasks": list(self.tasks),
            "entries": [entry.to_json_dict() for entry in self.entries],
        }

    @property
    def manifest_sha256(self) -> str:
        if self.is_read_only:
            assert self._persisted_manifest_sha256 is not None
            return self._persisted_manifest_sha256
        return _canonical_sha256(self._canonical_payload())

    @property
    def task_counts(self) -> dict[str, dict[str, int]]:
        return {
            split: {
                task: sum(
                    entry.split == split and entry.task == task
                    for entry in self.entries
                )
                for task in self.tasks
            }
            for split in _SPLIT_ORDER
        }

    def to_json_dict(self) -> dict[str, object]:
        payload = self._canonical_payload()
        payload["data_root"] = self.data_root
        payload["manifest_sha256"] = self.manifest_sha256
        payload["split_counts"] = dict(
            sorted(Counter(entry.split for entry in self.entries).items())
        )
        payload["task_counts"] = self.task_counts
        return payload


class _ReadOnlyLegacyUniVTACDatasetManifest(UniVTACDatasetManifest):
    """Load-only legacy representation whose constructor always fails closed."""

    def __post_init__(self) -> None:
        raise ValueError("legacy dataset manifests are read-only")


def _build_read_only_legacy_manifest(
    *,
    data_root: str,
    entries: tuple[UniVTACEpisodeRecord, ...],
    tasks: tuple[str, ...],
    schema_version: int,
    persisted_manifest_sha256: str,
) -> UniVTACDatasetManifest:
    """Construct a verified legacy instance without exposing an init token."""

    if schema_version not in {1, 2, 3}:
        raise ValueError("read-only manifest factory accepts only schema v1/v2/v3")
    if len(persisted_manifest_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in persisted_manifest_sha256
    ):
        raise ValueError("legacy manifest requires its persisted sha256")
    canonical_tasks = canonical_task_scope(tasks)
    if not entries:
        raise ValueError("dataset manifest entries must be non-empty")
    _validate_record_identities(entries, tasks=canonical_tasks)
    manifest = object.__new__(_ReadOnlyLegacyUniVTACDatasetManifest)
    object.__setattr__(manifest, "data_root", data_root)
    object.__setattr__(manifest, "entries", entries)
    object.__setattr__(manifest, "tasks", canonical_tasks)
    object.__setattr__(manifest, "schema_version", schema_version)
    object.__setattr__(
        manifest,
        "_persisted_manifest_sha256",
        persisted_manifest_sha256,
    )
    return manifest


def _parse_manifest_entry(
    item: object,
    *,
    root: Path,
    schema_version: int,
    verify_sources: bool,
) -> UniVTACEpisodeRecord:
    if not isinstance(item, dict):
        raise ValueError("dataset manifest entry must be an object")
    temporal_keys = {
        "usable_source_range",
        "step_discontinuities_after_rows",
        "temporal_policy",
    }
    if schema_version < MANIFEST_SCHEMA_VERSION and temporal_keys & set(item):
        raise ValueError("legacy manifest cannot declare schema-v4 temporal fields")
    required = {
        "relative_path",
        "task",
        "split",
        "length",
        "length_source",
        "joint_shape",
        "image_shapes",
        "size_bytes",
        "sha256",
    }
    if schema_version == MANIFEST_SCHEMA_VERSION:
        required.update(
            {
                "usable_source_range",
                "step_discontinuities_after_rows",
                "temporal_policy",
            }
        )
    if not required.issubset(item):
        raise ValueError("dataset manifest entry is incomplete")
    relative_path = str(item["relative_path"])
    task, episode_id = _validate_declared_relative_path(relative_path)
    absolute_path = (root / relative_path).resolve(strict=verify_sources)
    if root not in absolute_path.parents:
        raise ValueError(f"manifest entry escapes data root: {relative_path}")
    if schema_version >= 3:
        if item.get("realpath") != str(absolute_path):
            raise ValueError(f"manifest entry realpath mismatch: {relative_path}")
        if item.get("episode_id") != episode_id:
            raise ValueError(f"manifest entry episode ID mismatch: {relative_path}")
    image_shapes = item["image_shapes"]
    if not isinstance(image_shapes, dict):
        raise ValueError("manifest entry image_shapes must be an object")
    parsed_image_shapes: list[tuple[str, tuple[int, int, int]]] = []
    for name, raw_shape in image_shapes.items():
        shape = tuple(int(value) for value in raw_shape)
        if len(shape) != 3:
            raise ValueError("manifest image shape must contain three dimensions")
        parsed_image_shapes.append((str(name), (shape[0], shape[1], shape[2])))
    raw_usable_range = item.get("usable_source_range")
    if schema_version == MANIFEST_SCHEMA_VERSION:
        if not isinstance(raw_usable_range, list) or len(raw_usable_range) != 2:
            raise ValueError("manifest usable_source_range must contain two values")
        usable_start, usable_end = (int(value) for value in raw_usable_range)
        raw_discontinuities = item.get("step_discontinuities_after_rows")
        if not isinstance(raw_discontinuities, list):
            raise ValueError("manifest step discontinuities must be a list")
        discontinuities = tuple(int(value) for value in raw_discontinuities)
        temporal_policy = item.get("temporal_policy")
        if not isinstance(temporal_policy, str) or not temporal_policy:
            raise ValueError("manifest temporal_policy must be a string")
    else:
        usable_start = 0
        usable_end = int(item["length"])
        discontinuities = ()
        temporal_policy = "strict_monotonic_v1"
    record = UniVTACEpisodeRecord(
        relative_path=relative_path,
        absolute_path=absolute_path,
        task=str(item["task"]),
        split=str(item["split"]),
        length=int(item["length"]),
        length_source=str(item["length_source"]),
        joint_shape=tuple(int(value) for value in item["joint_shape"]),
        image_shapes=tuple(parsed_image_shapes),
        size_bytes=int(item["size_bytes"]),
        sha256=str(item["sha256"]),
        usable_start=usable_start,
        usable_end=usable_end,
        step_discontinuities_after_rows=discontinuities,
        temporal_policy=temporal_policy,
    )
    if record.task != task:
        raise ValueError(f"manifest entry task mismatch: {relative_path}")
    if verify_sources:
        from .hdf5_reader import verify_source_record

        verify_source_record(record)
    return record


def load_dataset_manifest(
    manifest_path: Path,
    *,
    verify_sources: bool = True,
) -> UniVTACDatasetManifest:
    """Load v1-v3 read-only or current schema-v4 manifest artifacts."""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read dataset manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be a JSON object")
    try:
        schema_version = int(payload["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("dataset manifest has invalid schema_version") from exc
    if schema_version not in READABLE_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported dataset manifest schema version: {schema_version}"
        )
    allowed_keys = set(_CANONICAL_KEYS) | set(_SUPPLEMENTAL_KEYS)
    if set(payload) - allowed_keys:
        raise ValueError("dataset manifest contains unsupported fields")
    if schema_version >= 2 and not set(_CANONICAL_KEYS).issubset(payload):
        raise ValueError("dataset manifest canonical payload is incomplete")
    canonical_payload = {key: payload[key] for key in _CANONICAL_KEYS if key in payload}
    declared_sha256 = payload.get("manifest_sha256")
    if (
        not isinstance(declared_sha256, str)
        or _canonical_sha256(canonical_payload) != declared_sha256
    ):
        raise ValueError("dataset manifest sha256 verification failed")
    if payload.get("action_schema") != ACTION_SCHEMA:
        raise ValueError("dataset manifest has an unsupported action schema")
    if "source_image_encoding_contract" in payload and (
        payload["source_image_encoding_contract"] != SOURCE_IMAGE_ENCODING_CONTRACT
    ):
        raise ValueError("dataset manifest has an unsupported image contract")
    if "output_color_space" in payload and payload["output_color_space"] != "RGB":
        raise ValueError("dataset manifest output color space must be RGB")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("dataset manifest entries must be a non-empty list")
    raw_tasks = payload.get("tasks")
    if raw_tasks is None and schema_version == 1:
        raw_tasks = list(
            dict.fromkeys(
                str(item.get("task")) for item in raw_entries if isinstance(item, dict)
            )
        )
    if not isinstance(raw_tasks, list) or not all(
        isinstance(task, str) for task in raw_tasks
    ):
        raise ValueError("dataset manifest tasks must be a list of strings")
    tasks = canonical_task_scope(raw_tasks)
    data_root = payload.get("data_root")
    if not isinstance(data_root, str) or not data_root:
        raise ValueError("dataset manifest data_root must be a path string")
    root = Path(data_root).resolve(strict=verify_sources)
    if verify_sources and not root.is_dir():
        raise NotADirectoryError(root)
    entries = tuple(
        _parse_manifest_entry(
            item,
            root=root,
            schema_version=schema_version,
            verify_sources=verify_sources,
        )
        for item in raw_entries
    )
    manifest = (
        _build_read_only_legacy_manifest(
            data_root=str(root),
            entries=entries,
            tasks=tasks,
            schema_version=schema_version,
            persisted_manifest_sha256=declared_sha256,
        )
        if schema_version < MANIFEST_SCHEMA_VERSION
        else UniVTACDatasetManifest(
            data_root=str(root),
            entries=entries,
            tasks=tasks,
            schema_version=schema_version,
        )
    )
    if schema_version == MANIFEST_SCHEMA_VERSION:
        if manifest.manifest_sha256 != declared_sha256:
            raise ValueError(
                "dataset manifest v4 entry canonicalization changed sha256"
            )
    return manifest


def _validate_expected_counts(
    entries: Sequence[UniVTACEpisodeRecord],
    expected: Mapping[str, Mapping[str, int]],
    *,
    tasks: Sequence[str],
) -> None:
    task_scope = canonical_task_scope(tasks)
    for split, task_counts in expected.items():
        if split not in _SPLIT_ORDER:
            raise ValueError(f"Unsupported expected split: {split!r}")
        if set(task_counts) != set(task_scope):
            raise ValueError(
                f"expected {split} count map must name every manifest task"
            )
        for task in task_scope:
            expected_count = task_counts[task]
            if isinstance(expected_count, bool) or expected_count < 0:
                raise ValueError("expected task counts must be non-negative integers")
            actual = sum(
                entry.split == split and entry.task == task for entry in entries
            )
            if actual != expected_count:
                raise ValueError(
                    f"Expected {split}/{task}={expected_count}, found {actual}"
                )


def build_dataset_manifest(
    *,
    data_root: Path,
    train_paths: Sequence[str],
    validation_paths: Sequence[str],
    quarantine_paths: Sequence[str] = (),
    expected_task_counts: Mapping[str, Mapping[str, int]] | None = None,
    tasks: Sequence[str] = TRACK31_TASKS,
) -> UniVTACDatasetManifest:
    """Audit explicit split lists and build a writeable schema-v4 manifest."""

    from .hdf5_reader import audit_episode

    task_scope = canonical_task_scope(tasks)
    split_paths = {
        "train": tuple(train_paths),
        "validation": tuple(validation_paths),
        "quarantine": tuple(quarantine_paths),
    }
    raw_paths = [path for paths in split_paths.values() for path in paths]
    if len(raw_paths) != len(set(raw_paths)):
        raise ValueError("dataset split path overlap is non-empty")
    if not train_paths:
        raise ValueError("train_paths must be non-empty")
    if not validation_paths:
        raise ValueError("validation_paths must be non-empty")
    records = tuple(
        audit_episode(
            data_root,
            relative_path,
            split=split,
            hash_file=True,
            allowed_tasks=task_scope,
        )
        for split, paths in split_paths.items()
        for relative_path in paths
    )
    _validate_record_identities(records, tasks=task_scope)
    shape_oracle = dict(records[0].image_shapes)
    for record in records[1:]:
        if dict(record.image_shapes) != shape_oracle:
            raise ValueError(
                "inconsistent image shapes across episodes: "
                f"{records[0].relative_path} vs {record.relative_path}"
            )
    if expected_task_counts is not None:
        _validate_expected_counts(
            records,
            expected_task_counts,
            tasks=task_scope,
        )
    task_rank = {task: index for index, task in enumerate(task_scope)}
    ordered = tuple(
        sorted(
            records,
            key=lambda record: (
                _SPLIT_ORDER[record.split],
                task_rank[record.task],
                record.episode_id,
                record.relative_path,
            ),
        )
    )
    return UniVTACDatasetManifest(
        data_root=str(data_root.resolve(strict=True)),
        entries=ordered,
        tasks=task_scope,
    )
