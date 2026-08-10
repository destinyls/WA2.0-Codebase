# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""UniVTAC-specific bridge code isolated from the N0 core import graph."""

from .dataset_view import (
    ALLOWED_OVERLAP_MATRIX,
    DEFAULT_UNIFIED_EVALUATION_EPISODE_COUNT,
    DEFAULT_UNIFIED_EVALUATION_EPISODE_IDS,
    DEFAULT_UNIFIED_EVALUATION_TASKS,
    DEFAULT_UNIFIED_EVALUATION_VIEW_ID,
    DIAGNOSTIC_EVALUATION_VIEW_IDS,
    DatasetView,
    DatasetViewEntry,
    build_standard_dataset_views,
    content_addressed_sample_seed,
    load_dataset_view,
    select_content_addressed_crop_start,
    verify_standard_view_set,
)
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    UniVTACDatasetManifest,
    build_dataset_manifest,
    load_dataset_manifest,
)
from .schema import (
    CANONICAL_TASK_PROMPT_MAP,
    TRACK31_TARGET_TASKS,
    UNIVTAC_ALL_TASKS,
)

__all__ = (
    "ALLOWED_OVERLAP_MATRIX",
    "CANONICAL_TASK_PROMPT_MAP",
    "DEFAULT_UNIFIED_EVALUATION_EPISODE_COUNT",
    "DEFAULT_UNIFIED_EVALUATION_EPISODE_IDS",
    "DEFAULT_UNIFIED_EVALUATION_TASKS",
    "DEFAULT_UNIFIED_EVALUATION_VIEW_ID",
    "DIAGNOSTIC_EVALUATION_VIEW_IDS",
    "DatasetView",
    "DatasetViewEntry",
    "MANIFEST_SCHEMA_VERSION",
    "TRACK31_TARGET_TASKS",
    "UNIVTAC_ALL_TASKS",
    "UniVTACDatasetManifest",
    "build_dataset_manifest",
    "build_standard_dataset_views",
    "content_addressed_sample_seed",
    "load_dataset_manifest",
    "load_dataset_view",
    "select_content_addressed_crop_start",
    "verify_standard_view_set",
)
