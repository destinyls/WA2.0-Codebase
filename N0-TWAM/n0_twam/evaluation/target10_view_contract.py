# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Canonical dataset-view binding for the frozen Target-10 evaluation."""

from __future__ import annotations

from pathlib import Path

from n0_twam.integrations.univtac.dataset_view import (
    DEFAULT_UNIFIED_EVALUATION_VIEW_ID,
    DatasetView,
    build_standard_dataset_views,
    load_dataset_view,
)
from n0_twam.integrations.univtac.manifest import load_dataset_manifest


def load_canonical_target10_view(
    *,
    view_path: Path,
    manifest_path: Path,
) -> DatasetView:
    """Load Target-10 and require the unique standard view derived from its manifest."""

    view = load_dataset_view(Path(view_path).resolve(strict=True))
    manifest = load_dataset_manifest(
        Path(manifest_path).resolve(strict=True),
        verify_sources=False,
    )
    canonical_view = build_standard_dataset_views(
        manifest,
        verify_sources=False,
    )[DEFAULT_UNIFIED_EVALUATION_VIEW_ID]
    if view.to_json_dict() != canonical_view.to_json_dict():
        raise ValueError(
            "evaluation view is not the canonical frozen_target10_v1 derived "
            "from the sealed dataset manifest"
        )
    return view


__all__ = ("load_canonical_target10_view",)
