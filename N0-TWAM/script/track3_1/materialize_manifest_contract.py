# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Formal schema-v4 manifest contract for UniVTAC materialization."""

from typing import Final

from n0_twam.integrations.univtac.manifest import (
    MANIFEST_SCHEMA_VERSION,
    UniVTACDatasetManifest,
)
from n0_twam.integrations.univtac.schema import UNIVTAC_ALL_TASKS

EXPECTED_SPLIT_COUNTS: Final = {
    "train": 759,
    "validation": 40,
    "quarantine": 1,
}
EXPECTED_TASK_COUNTS: Final = {
    "train": {
        task: 94 if task == "grasp_classify" else 95 for task in UNIVTAC_ALL_TASKS
    },
    "validation": {task: 5 for task in UNIVTAC_ALL_TASKS},
    "quarantine": {
        task: 1 if task == "grasp_classify" else 0 for task in UNIVTAC_ALL_TASKS
    },
}


def validate_formal_manifest(manifest: UniVTACDatasetManifest) -> None:
    """Require the exact eight-task 759/40/1 formal universe."""

    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError("formal materialization requires a schema-v4 manifest")
    if manifest.tasks != UNIVTAC_ALL_TASKS:
        raise ValueError("formal materialization requires the eight-task scope")
    split_counts = {
        split: sum(entry.split == split for entry in manifest.entries)
        for split in EXPECTED_SPLIT_COUNTS
    }
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "formal manifest split counts must be 759/40/1, found " f"{split_counts}"
        )
    if manifest.task_counts != EXPECTED_TASK_COUNTS:
        raise ValueError("formal manifest per-task counts do not match 759/40/1")
    quarantine = [entry for entry in manifest.entries if entry.split == "quarantine"]
    if (
        len(quarantine) != 1
        or quarantine[0].relative_path != "grasp_classify/clean/90.hdf5"
        or quarantine[0].length != 43
    ):
        raise ValueError(
            "quarantine identity must be the fixed 43-frame "
            "grasp_classify/clean/90.hdf5 episode"
        )
