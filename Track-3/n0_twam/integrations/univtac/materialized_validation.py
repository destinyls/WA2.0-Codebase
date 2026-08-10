# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Validate materialized LeRobot episode tables against temporal contracts."""

import json
from numbers import Integral
from pathlib import Path
from typing import Any, Sequence


def _read_episode_metadata(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"unable to read LeRobot episode metadata: {path}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid LeRobot episode metadata at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError("LeRobot episode metadata must contain objects")
        records.append(record)
    return records


def _expected_converted_length(entry: dict[str, Any]) -> int:
    raw_range = entry.get("usable_source_range")
    if not isinstance(raw_range, list) or len(raw_range) != 2:
        raise ValueError("manifest entry lacks a usable source range")
    usable_start, usable_end = (int(value) for value in raw_range)
    length = usable_end - usable_start - 1
    if length <= 0:
        raise ValueError("manifest entry has an empty converted timeline")
    return length


def _unwrap_scalar(value: object, *, label: str) -> object:
    current = value
    while isinstance(current, (list, tuple)):
        if len(current) != 1:
            raise ValueError(f"{label} must contain exactly one scalar")
        current = current[0]
    return current


def _require_integer_scalar(value: object, *, label: str) -> int:
    """Require a true scalar integer without accepting list-shaped drift."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer scalar")
    return int(value)


def _require_integer_tensor_1x1(value: object, *, label: str) -> int:
    """Require the explicit ``shape=(1, 1)`` source-provenance schema."""

    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError(f"{label} must have shape (1, 1)")
    row = value[0]
    if not isinstance(row, (list, tuple)) or len(row) != 1:
        raise ValueError(f"{label} must have shape (1, 1)")
    scalar = row[0]
    if isinstance(scalar, bool) or not isinstance(scalar, Integral):
        raise ValueError(f"{label} must contain an integer scalar")
    return int(scalar)


def _verify_metadata(
    root: Path,
    expected_entries: Sequence[dict[str, Any]],
) -> tuple[int, ...]:
    records = _read_episode_metadata(root / "meta" / "episodes.jsonl")
    if len(records) != len(expected_entries):
        raise ValueError("LeRobot episode metadata count does not match manifest")
    expected_lengths = tuple(
        _expected_converted_length(entry) for entry in expected_entries
    )
    for episode_index, (record, expected_length, entry) in enumerate(
        zip(records, expected_lengths, expected_entries)
    ):
        recorded_episode_index = _require_integer_scalar(
            record.get("episode_index"),
            label="episode_index",
        )
        recorded_length = _require_integer_scalar(
            record.get("length"),
            label="length",
        )
        if recorded_episode_index != episode_index:
            raise ValueError("LeRobot episode indices are not canonical")
        if recorded_length != expected_length:
            raise ValueError(
                f"LeRobot episode {episode_index} length does not match manifest"
            )
        tasks = record.get("tasks")
        expected_task = str(entry["task"])
        if tasks != [expected_task]:
            raise ValueError(
                f"LeRobot episode {episode_index} task does not match manifest"
            )
        action_config = [
            {
                "start_frame": 0,
                "end_frame": expected_length,
                "action_text": expected_task,
            }
        ]
        if record.get("action_config") != action_config:
            raise ValueError(
                f"LeRobot episode {episode_index} action_config does not match"
            )
    return expected_lengths


def _read_parquet_rows(root: Path) -> list[dict[str, object]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - required production dependency
        raise ImportError("pyarrow is required to validate LeRobot tables") from exc

    required_columns = (
        "episode_index",
        "frame_index",
        "source.relative_path",
        "source.row_index",
        "source.frame_index",
    )
    parquet_paths = sorted((root / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise ValueError("LeRobot data directory contains no Parquet tables")
    rows: list[dict[str, object]] = []
    for path in parquet_paths:
        try:
            table = parquet.read_table(path, columns=list(required_columns))
        except Exception as exc:
            raise ValueError(f"unable to read LeRobot Parquet table: {path}") from exc
        if tuple(table.column_names) != required_columns:
            raise ValueError(f"LeRobot table lacks provenance columns: {path}")
        columns = {name: table[name].to_pylist() for name in required_columns}
        for row_index in range(table.num_rows):
            rows.append({name: values[row_index] for name, values in columns.items()})
    return rows


def _verify_rows(
    root: Path,
    expected_entries: Sequence[dict[str, Any]],
    expected_lengths: Sequence[int],
) -> None:
    rows = _read_parquet_rows(root)
    grouped: dict[int, list[dict[str, object]]] = {
        episode_index: [] for episode_index in range(len(expected_entries))
    }
    for row in rows:
        episode_index = _require_integer_scalar(
            row["episode_index"], label="episode_index"
        )
        if episode_index not in grouped:
            raise ValueError("LeRobot table contains an unknown episode index")
        grouped[episode_index].append(row)

    for episode_index, (entry, expected_length) in enumerate(
        zip(expected_entries, expected_lengths)
    ):
        episode_rows = grouped[episode_index]
        if len(episode_rows) != expected_length:
            raise ValueError(
                f"LeRobot episode {episode_index} Parquet row count does not match"
            )
        frame_indices = [
            _require_integer_scalar(row["frame_index"], label="frame_index")
            for row in episode_rows
        ]
        if frame_indices != list(range(expected_length)):
            raise ValueError(
                f"LeRobot episode {episode_index} frame indices are not contiguous"
            )
        source_paths = {
            str(
                _unwrap_scalar(
                    row["source.relative_path"],
                    label="source.relative_path",
                )
            )
            for row in episode_rows
        }
        if source_paths != {str(entry["relative_path"])}:
            raise ValueError(
                f"LeRobot episode {episode_index} source path does not match"
            )
        source_steps = [
            _require_integer_tensor_1x1(
                row["source.frame_index"],
                label="source.frame_index",
            )
            for row in episode_rows
        ]
        usable_start = int(entry["usable_source_range"][0])
        source_rows = [
            _require_integer_tensor_1x1(
                row["source.row_index"],
                label="source.row_index",
            )
            for row in episode_rows
        ]
        if source_rows != list(range(usable_start, usable_start + expected_length)):
            raise ValueError(
                f"LeRobot episode {episode_index} source row indices do not match"
            )
        if any(right <= left for left, right in zip(source_steps, source_steps[1:])):
            raise ValueError(
                f"LeRobot episode {episode_index} source steps are not increasing"
            )


def verify_materialized_episode_tables(
    root: Path,
    *,
    expected_entries: Sequence[dict[str, Any]],
) -> None:
    """Bind actual episode metadata and Parquet rows to manifest ranges."""

    expected_lengths = _verify_metadata(root, expected_entries)
    _verify_rows(root, expected_entries, expected_lengths)
