"""Fail-closed contracts for Track 3.1 Stage A/B training profiles."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from n0_twam.configs.twam_track31_training_profiles import (
    build_track31_training_profile_contract,
    resolve_track31_run_role,
    resolve_track31_train_profile,
)

STAGE_A = "multitask_pretrain_v1"
STAGE_B = "target_finetune_v1"


@pytest.mark.parametrize(
    ("environment_name", "invalid_value", "resolver"),
    (
        ("N0_TRACK31_TRAIN_PROFILE", "stage_a", resolve_track31_train_profile),
        ("N0_TRACK31_RUN_ROLE", "evaluation", resolve_track31_run_role),
    ),
)
def test_profile_and_run_role_reject_unknown_values(
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    invalid_value: str,
    resolver: Callable[[], str],
) -> None:
    monkeypatch.setenv(environment_name, invalid_value)

    with pytest.raises(ValueError, match=environment_name):
        resolver()


def test_fresh_multitask_profile_migrates_released_20d_weights() -> None:
    contract = build_track31_training_profile_contract(
        training_profile_id=STAGE_A,
        run_role="development",
        released_checkpoint="/checkpoints/released20d",
        resume_from=None,
        init_from=None,
        parent_train_meta=None,
    )

    assert contract["training_profile_id"] == STAGE_A
    assert contract["train_view_id"] == "stage_a_dev719_v1"
    assert contract["validation_view_id"] == "internal_dev40_v1"
    assert contract["normalizer_source_view_id"] == "stage_a_dev719_v1"
    assert contract["initialization_mode"] == "released_action_migration"
    assert contract["checkpoint_compatibility"] == "migrate_action"
    lineage = contract["training_lineage"]
    assert isinstance(lineage, Mapping)
    assert lineage == {
        "source_kind": "released_20d",
        "source_checkpoint": "/checkpoints/released20d",
        "parent_training_profile_id": None,
        "parent_checkpoint_identity": None,
    }


def test_fresh_target_finetune_requires_stage_a_weights_only_parent() -> None:
    parent_identity = {"sha256": "a" * 64, "action_dim": 8}
    parent_train_meta = {
        "training_profile_id": STAGE_A,
        "transformer_identity": parent_identity,
    }

    contract = build_track31_training_profile_contract(
        training_profile_id=STAGE_B,
        run_role="development",
        released_checkpoint=None,
        resume_from=None,
        init_from="/runs/stage-a/checkpoint-5000",
        parent_train_meta=parent_train_meta,
    )

    assert contract["train_view_id"] == "stage_b_dev180_v1"
    assert contract["validation_view_id"] == "internal_target_dev10_v1"
    assert contract["normalizer_source_view_id"] == "stage_a_dev719_v1"
    assert contract["initialization_mode"] == "stage_a_weights_only"
    assert contract["checkpoint_compatibility"] == "strict"
    assert contract["inherit_action_migration_report"] is True
    assert contract["training_lineage"] == {
        "source_kind": "stage_a_checkpoint",
        "source_checkpoint": "/runs/stage-a/checkpoint-5000",
        "parent_training_profile_id": STAGE_A,
        "parent_checkpoint_identity": parent_identity,
    }


def test_target_finetune_rejects_stage_a_strict_resume() -> None:
    with pytest.raises(ValueError, match="same training profile"):
        build_track31_training_profile_contract(
            training_profile_id=STAGE_B,
            run_role="development",
            released_checkpoint=None,
            resume_from="/runs/stage-a/checkpoint-5000",
            init_from=None,
            parent_train_meta={"training_profile_id": STAGE_A},
        )


def test_final_refit_requires_no_validation_view() -> None:
    contract = build_track31_training_profile_contract(
        training_profile_id=STAGE_A,
        run_role="final_refit",
        released_checkpoint="/checkpoints/released20d",
        resume_from=None,
        init_from=None,
        parent_train_meta=None,
    )
    assert contract["validation_view_id"] is None
    assert contract["normalizer_source_view_id"] == "stage_a_final759_v1"
    assert contract["has_validation_loader"] is False
