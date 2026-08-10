# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import json
from pathlib import Path

import pytest

from n0_twam.integrations.univtac import materialized_validation
from n0_twam.integrations.univtac.materialized_validation import (
    _verify_metadata,
    _verify_rows,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_index", 0.0),
        ("episode_index", "0"),
        ("episode_index", True),
        ("episode_index", [0]),
        ("episode_index", [[0]]),
        ("length", 5.0),
        ("length", "5"),
        ("length", True),
        ("length", [5]),
        ("length", [[5]]),
    ],
)
def test_materialized_metadata_rejects_coercive_integer_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = tmp_path / "train759"
    (root / "meta").mkdir(parents=True)
    record: dict[str, object] = {
        "episode_index": 0,
        "length": 5,
        "tasks": ["insert_HDMI"],
        "action_config": [
            {
                "start_frame": 0,
                "end_frame": 5,
                "action_text": "insert_HDMI",
            }
        ],
    }
    record[field] = value
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"{field} must be an integer scalar"):
        _verify_metadata(
            root,
            expected_entries=(
                {
                    "task": "insert_HDMI",
                    "usable_source_range": [0, 6],
                },
            ),
        )


@pytest.mark.parametrize("field", ["episode_index", "frame_index"])
def test_materialized_rows_reject_list_typed_scalar_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    rows: list[dict[str, object]] = []
    for frame_index in range(5):
        row: dict[str, object] = {
            "episode_index": 0,
            "frame_index": frame_index,
            "source.relative_path": ["insert_HDMI/clean/0.hdf5"],
            "source.row_index": [[frame_index]],
            "source.frame_index": [[100 + frame_index]],
        }
        row[field] = [row[field]]
        rows.append(row)
    monkeypatch.setattr(materialized_validation, "_read_parquet_rows", lambda _: rows)

    with pytest.raises(ValueError, match=f"{field} must be an integer scalar"):
        _verify_rows(
            tmp_path,
            expected_entries=(
                {
                    "relative_path": "insert_HDMI/clean/0.hdf5",
                    "usable_source_range": [0, 6],
                },
            ),
            expected_lengths=(5,),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source.row_index", [0], "source.row_index must have shape"),
        ("source.row_index", [[[0]]], "source.row_index must contain"),
        ("source.frame_index", [[True]], "source.frame_index must contain"),
    ],
)
def test_materialized_rows_require_exact_source_integer_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    rows: list[dict[str, object]] = []
    for frame_index in range(5):
        row: dict[str, object] = {
            "episode_index": 0,
            "frame_index": frame_index,
            "source.relative_path": ["insert_HDMI/clean/0.hdf5"],
            "source.row_index": [[frame_index]],
            "source.frame_index": [[100 + frame_index]],
        }
        if frame_index == 0:
            row[field] = value
        rows.append(row)
    monkeypatch.setattr(materialized_validation, "_read_parquet_rows", lambda _: rows)

    with pytest.raises(ValueError, match=message):
        _verify_rows(
            tmp_path,
            expected_entries=(
                {
                    "relative_path": "insert_HDMI/clean/0.hdf5",
                    "usable_source_range": [0, 6],
                },
            ),
            expected_lengths=(5,),
        )
