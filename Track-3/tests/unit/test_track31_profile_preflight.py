"""Profile-specific Track 3.1 preflight tests."""

from pathlib import Path

import pytest

from script.track3_1.preflight_train import (
    _audit_checkpoint,
    _resolve_profile_contract,
)
from tests.unit.test_track31_preflight import (
    _STAGE_A,
    _set_current_runtime_provenance_env,
    _write_resume_sidecars,
    _write_training_state,
    _write_transformer_checkpoint,
)

_STAGE_B = "target_finetune_v1"


@pytest.fixture(autouse=True)
def _clean_profile_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N0_TRACK31_TRAIN_PROFILE", _STAGE_A)
    monkeypatch.setenv("N0_TRACK31_RUN_ROLE", "final_refit")
    for name in (
        "N0_TRACK31_INIT_FROM",
        "N0_TRACK31_RESUME_FROM",
        "N0_TRACK31_VALIDATION_VIEW_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    _set_current_runtime_provenance_env(monkeypatch)


def test_preflight_accepts_stage_b_weights_only_stage_a_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "stage_a_parent"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(
        checkpoint,
        world_size=1,
        training_profile_id=_STAGE_A,
        run_role="development",
    )
    _write_training_state(checkpoint, strict_fields, world_size=1)
    monkeypatch.setenv("N0_TRACK31_TRAIN_PROFILE", _STAGE_B)
    monkeypatch.setenv("N0_TRACK31_RUN_ROLE", "development")
    monkeypatch.setenv("N0_TRACK31_INIT_FROM", str(checkpoint))
    monkeypatch.delenv("N0_RELEASED_CHECKPOINT", raising=False)
    monkeypatch.delenv("N0_RELEASED_TRANSFORMER_SHA256", raising=False)

    report = _audit_checkpoint()

    assert report["mode"] == "stage_a_weights_only"
    assert report["parent_training_profile_id"] == _STAGE_A
    assert (
        report["runtime_training_lineage"]["parent_transformer_identity"]
        == report["parent_transformer_identity"]
    )
    assert report["reset_optimizer"] is True
    assert report["reset_scheduler"] is True
    assert report["reset_rng"] is True
    assert report["reset_data_cursor"] is True


def test_preflight_rejects_stage_b_parent_with_wrong_run_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "stage_a_parent"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(
        checkpoint,
        world_size=1,
        training_profile_id=_STAGE_A,
        run_role="final_refit",
    )
    _write_training_state(checkpoint, strict_fields, world_size=1)
    monkeypatch.setenv("N0_TRACK31_TRAIN_PROFILE", _STAGE_B)
    monkeypatch.setenv("N0_TRACK31_RUN_ROLE", "development")
    monkeypatch.setenv("N0_TRACK31_INIT_FROM", str(checkpoint))

    with pytest.raises(ValueError, match="run role"):
        _audit_checkpoint()


def test_preflight_rejects_cross_profile_strict_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "stage_a_resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    _write_resume_sidecars(
        checkpoint,
        world_size=1,
        training_profile_id=_STAGE_A,
        run_role="final_refit",
    )
    monkeypatch.setenv("N0_TRACK31_TRAIN_PROFILE", _STAGE_B)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))

    with pytest.raises(ValueError, match="same training profile"):
        _audit_checkpoint()


def test_preflight_final_refit_rejects_validation_view_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("N0_TRACK31_VALIDATION_VIEW_PATH", "/not/used/frozen_view.json")

    with pytest.raises(ValueError, match="must not use a validation view"):
        _resolve_profile_contract()
