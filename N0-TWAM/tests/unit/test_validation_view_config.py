# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from types import SimpleNamespace

import pytest

from n0_twam.integrations.univtac.validation_config import (
    build_validation_dataset_config,
)


def test_validation_config_rebinds_generic_dataset_view_fields() -> None:
    training = SimpleNamespace(
        dataset_path="/data/train759",
        dataset_view_path="/views/stage_a_dev719_v1.json",
        train_view_id="stage_a_dev719_v1",
        val_dataset_path="/data/train759",
        val_dataset_view_path="/views/internal_dev40_v1.json",
        validation_view_id="internal_dev40_v1",
    )

    validation = build_validation_dataset_config(training)

    assert validation is not training
    assert validation.dataset_path == "/data/train759"
    assert validation.dataset_view_path == "/views/internal_dev40_v1.json"
    assert validation.train_view_id == "internal_dev40_v1"
    assert training.dataset_view_path == "/views/stage_a_dev719_v1.json"


def test_validation_config_clears_inherited_view_for_legacy_dataset() -> None:
    training = SimpleNamespace(
        dataset_path="/data/train",
        dataset_view_path="/views/train.json",
        train_view_id="train_view",
        val_dataset_path="/data/validation",
    )

    validation = build_validation_dataset_config(training)

    assert validation.dataset_path == "/data/validation"
    assert validation.dataset_view_path is None
    assert validation.train_view_id is None


def test_validation_config_rejects_partial_view_binding() -> None:
    training = SimpleNamespace(
        val_dataset_path="/data/train759",
        val_dataset_view_path="/views/internal_dev40_v1.json",
        validation_view_id=None,
    )

    with pytest.raises(ValueError, match="declared together"):
        build_validation_dataset_config(training)
