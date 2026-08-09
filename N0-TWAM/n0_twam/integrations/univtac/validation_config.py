# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Build an isolated validation dataset config from a training config."""

from __future__ import annotations

import copy
from typing import Any


def build_validation_dataset_config(config: Any) -> Any:
    """Rebind generic dataset fields to the declared validation view."""

    validation_path = getattr(config, "val_dataset_path", None)
    if not isinstance(validation_path, str) or not validation_path:
        raise ValueError("validation dataset path must be a non-empty string")
    validation_view_path = getattr(config, "val_dataset_view_path", None)
    validation_view_id = getattr(config, "validation_view_id", None)
    if (validation_view_path is None) != (validation_view_id is None):
        raise ValueError("validation view path and ID must be declared together")
    validation_config = copy.copy(config)
    validation_config.dataset_path = validation_path
    # Never inherit the training view through the generic dataset adapter.
    validation_config.dataset_view_path = validation_view_path
    validation_config.train_view_id = validation_view_id
    return validation_config
