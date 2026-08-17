# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Trainer-side tests for the canonical AgileX batch contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.unit.test_track31_invocation_contract import train_module  # noqa: F401


def _trainer(
    train_module: Any,  # noqa: F811
    *,
    profile: str = "mixed",
) -> Any:
    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.config = SimpleNamespace(
        dataset_adapter="worldarena_agilex_qpos14",
        tactile_profile=profile,
        tactile_mode="disabled" if profile == "vision_only" else "enabled",
        contact_cond_drop_strategy="content_addressed_v1",
        action_dim=14,
        wrench_arm_count=2,
        tactile_global_zero=False,
        noisy_cond_prob_tactile=0.0,
    )
    trainer.device = torch.device("cpu")
    return trainer


def _batch(
    *,
    batch_size: int = 2,
    contact_drop: tuple[bool, ...] = (False, False),
) -> dict[str, object]:
    frames, horizon, sensors, arms = 2, 3, 2, 2
    assert len(contact_drop) == batch_size
    temporal = torch.tensor([[True, True]] * batch_size)
    action_valid = torch.tensor(
        [[[True, True, True], [True, False, False]]] * batch_size
    )
    tactile_available = torch.tensor([[[True, False], [True, True]]] * batch_size)
    wrench_available = torch.tensor([[[True, False], [True, True]]] * batch_size)
    return {
        "actions": torch.ones(batch_size, 14, frames, horizon, 1),
        "action_valid_mask": action_valid,
        "temporal_valid_mask": temporal,
        "tactile_available_mask": tactile_available,
        "wrench": torch.ones(batch_size, frames, arms, 6),
        "wrench_available_mask": wrench_available,
        "contact_cond_drop": torch.tensor(contact_drop, dtype=torch.bool),
        "latents": torch.zeros(batch_size, 48, frames, 2, 2),
        "text_emb": torch.zeros(batch_size, 4, 8),
        "tactile_global_latent": torch.ones(batch_size, sensors, 8, frames, 1, 1),
        "tactile_local_latent": torch.ones(batch_size, sensors, 8, frames, 1, 1),
        "tactile_sensor_ids": torch.tensor([[0, 1]] * batch_size),
        "repo_route_identity": ["a" * 64] * batch_size,
        "temporal_alignment_identity": ["b" * 64] * batch_size,
    }


def _install_prepare_stubs(trainer: Any) -> None:
    def add_noise(
        *, latent: torch.Tensor, action_mask: torch.Tensor | None, **_: object
    ) -> dict[str, torch.Tensor]:
        masked = latent if action_mask is None else latent * action_mask
        frames = latent.shape[2]
        return {
            "noisy_latents": masked.clone(),
            "targets": masked.clone(),
            "latent": masked.clone(),
            "timesteps": torch.zeros(latent.shape[0], frames),
            "cond_timesteps": torch.zeros(latent.shape[0], frames),
            "grid_id": torch.zeros(latent.shape[0], 4, 1),
        }

    trainer._add_noise = add_noise
    trainer._sample_attention_mask_schedule = lambda: (1, 1)
    trainer.train_scheduler_latent = object()
    trainer.train_scheduler_action = object()
    trainer.train_scheduler_tactile = object()


def test_agilex_adapter_builds_dense_action_mask_without_rng_redraw(
    train_module: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer(train_module)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("AgileX contact drop must come from the dataset")
        ),
    )

    converted = trainer.convert_input_format(_batch())

    expected = converted["action_valid_mask"][:, None, :, :, None].expand(
        -1, 14, -1, -1, -1
    )
    assert converted["actions_mask"].dtype is torch.bool
    assert torch.equal(converted["actions_mask"], expected)
    assert torch.equal(
        converted["tactile_target_mask"],
        converted["tactile_available_mask"],
    )
    assert torch.equal(
        converted["tactile_condition_mask"],
        converted["tactile_available_mask"],
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "action_valid_mask",
            torch.ones(2, 2, 3),
            "action_valid_mask.*bool",
        ),
        (
            "temporal_valid_mask",
            torch.ones(2, 2, dtype=torch.int64),
            "temporal_valid_mask.*bool",
        ),
        (
            "wrench",
            torch.ones(2, 2, 2, 6, dtype=torch.int64),
            "wrench.*float32",
        ),
    ),
)
def test_agilex_adapter_rejects_wrong_canonical_dtypes(
    train_module: Any,  # noqa: F811
    field: str,
    replacement: torch.Tensor,
    message: str,
) -> None:
    trainer = _trainer(train_module)
    batch = _batch()
    batch[field] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        trainer.convert_input_format(batch)


def test_agilex_adapter_rejects_inconsistent_batch_drop_flags(
    train_module: Any,  # noqa: F811
) -> None:
    trainer = _trainer(train_module)

    with pytest.raises(ValueError, match="contact_cond_drop.*uniform"):
        trainer.convert_input_format(_batch(contact_drop=(False, True)))


def test_agilex_condition_drop_does_not_remove_tactile_target(
    train_module: Any,  # noqa: F811
) -> None:
    trainer = _trainer(train_module)
    batch = _batch(contact_drop=(True, True))
    converted = trainer.convert_input_format(batch)
    assert converted["tactile_target_mask"].any()
    assert not converted["tactile_condition_mask"].any()
    assert not converted["wrench_condition_mask"].any()

    _install_prepare_stubs(trainer)
    prepared = trainer._prepare_input_dict(converted)
    action = prepared["action_dict"]

    assert action["contact_cond_drop"].tolist() == [True, True]
    assert action["tactile_target_mask"].any()
    assert "tactile_global_targets" in action
    assert torch.count_nonzero(action["tactile_global_targets"]) > 0
    assert torch.count_nonzero(action["tactile_global_clean_latent"]) == 0
    assert torch.count_nonzero(action["tactile_local_latent"]) == 0
    assert torch.equal(action["wrench"], converted["wrench"])


def test_vision_only_validates_then_drops_extra_contact_inputs(
    train_module: Any,  # noqa: F811
) -> None:
    trainer = _trainer(train_module, profile="vision_only")
    converted = trainer.convert_input_format(_batch(contact_drop=(True, True)))

    for field in (
        "tactile_global_latent",
        "tactile_local_latent",
        "tactile_sensor_ids",
        "wrench",
    ):
        assert field not in converted
    assert not converted["tactile_available_mask"].any()
    assert not converted["wrench_available_mask"].any()
    assert converted["contact_cond_drop"].all()

    _install_prepare_stubs(trainer)
    action = trainer._prepare_input_dict(converted)["action_dict"]
    assert "wrench" not in action
    assert "tactile_global_clean_latent" not in action
    assert "tactile_local_latent" not in action
    assert not action["tactile_target_mask"].any()


def test_tactile_loss_uses_target_availability_not_condition_drop(
    train_module: Any,  # noqa: F811
) -> None:
    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.config = SimpleNamespace(tactile_diffusion_loss_weight=1.0)
    trainer.patch_size = [1, 1, 1]
    trainer.gradient_accumulation_steps = 1
    trainer._compute_latent_loss = lambda *_: torch.tensor(0.0)
    trainer._compute_action_loss = lambda *_: torch.tensor(0.0)
    tactile_target = torch.zeros(2, 1, 1, 2, 1, 1)
    tactile_pred = torch.tensor([[[1.0], [100.0]], [[100.0], [100.0]]])
    input_dict = {
        "action_dict": {
            "tactile_global_targets": tactile_target,
            "tactile_target_mask": torch.tensor(
                [[[True], [False]], [[False], [False]]]
            ),
        }
    }

    losses = trainer.compute_loss(
        input_dict,
        (torch.zeros(2, 0, 1), torch.zeros(2, 0, 1), tactile_pred),
    )

    assert losses["tactile_loss"].item() == pytest.approx(1.0)


def test_legacy_batch_keeps_existing_actions_mask_and_cfg_rng(
    train_module: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.config = SimpleNamespace(
        dataset_adapter="worldarena_franka_ee10",
        tactile_cfg_prob=0.5,
    )
    trainer.device = torch.device("cpu")
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.0]))
    mask = torch.ones(1, 20, 1, 1, 1, dtype=torch.bool)
    batch = {
        "actions": torch.zeros(1, 20, 1, 1, 1),
        "actions_mask": mask,
        "latents": torch.zeros(1, 48, 1, 1, 1),
        "text_emb": torch.zeros(1, 1, 1),
        "tactile_global_latent": torch.ones(1, 1, 8, 1, 1, 1),
    }

    converted = trainer.convert_input_format(batch)

    assert converted["actions_mask"] is mask
    assert trainer._tactile_cond_drop is True


def test_strict_resume_lineage_is_exact_across_all_three_sidecars(
    train_module: Any,  # noqa: F811
) -> None:
    current = {"schema_version": 1, "action_schema": "qpos14_joint_absolute_v1"}
    payloads = (
        ("train metadata", {"training_lineage": dict(current)}),
        ("training state", {"training_lineage": dict(current)}),
        ("completion marker", {"training_lineage": dict(current)}),
    )
    train_module._validate_strict_resume_training_lineage(
        current_training_lineage=current,
        sidecars=payloads,
    )

    changed = list(payloads)
    changed[1] = (
        "training state",
        {"training_lineage": {**current, "action_schema": "wrong"}},
    )
    with pytest.raises(ValueError, match="training state.*training_lineage"):
        train_module._validate_strict_resume_training_lineage(
            current_training_lineage=current,
            sidecars=tuple(changed),
        )


def test_legacy_resume_without_current_lineage_skips_new_sidecar_requirement(
    train_module: Any,  # noqa: F811
) -> None:
    train_module._validate_strict_resume_training_lineage(
        current_training_lineage=None,
        sidecars=(
            ("train metadata", {}),
            ("training state", {}),
            ("completion marker", {}),
        ),
    )


def test_agilex_strict_resume_binds_seed_intervals_and_run_role(
    train_module: Any,  # noqa: F811
) -> None:
    from n0_twam.configs.twam_track3_agilex_recipe import (
        build_agilex_resume_recipe_contract,
    )

    config = SimpleNamespace(
        dataset_adapter="worldarena_agilex_qpos14",
        run_role="development",
        seed=20260811,
        save_interval=300,
        val_interval=100,
    )
    recipe = build_agilex_resume_recipe_contract(config)
    lineage = {"resume_recipe_contract": recipe}
    sidecars = (
        (
            "train metadata",
            {"run_role": "development", "training_lineage": lineage},
        ),
        ("training state", {"training_lineage": lineage}),
        ("completion marker", {"training_lineage": lineage}),
    )
    train_module._validate_agilex_strict_resume_recipe(
        config=config,
        sidecars=sidecars,
    )

    changed_seed = SimpleNamespace(**vars(config))
    changed_seed.seed += 1
    with pytest.raises(ValueError, match="AgileX resume recipe"):
        train_module._validate_agilex_strict_resume_recipe(
            config=changed_seed,
            sidecars=sidecars,
        )

    changed_role = list(sidecars)
    changed_role[0] = (
        "train metadata",
        {"run_role": "final_refit", "training_lineage": lineage},
    )
    with pytest.raises(ValueError, match="train metadata AgileX run_role"):
        train_module._validate_agilex_strict_resume_recipe(
            config=config,
            sidecars=tuple(changed_role),
        )


def test_non_agilex_resume_does_not_require_agilex_recipe(
    train_module: Any,  # noqa: F811
) -> None:
    train_module._validate_agilex_strict_resume_recipe(
        config=SimpleNamespace(dataset_adapter="worldarena_franka_ee10"),
        sidecars=(("train metadata", {}),),
    )


def test_qpos14_initial_migration_is_revalidated_after_identity_attachment(
    train_module: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def validate(report: dict[str, object], **kwargs: object) -> dict[str, object]:
        captured.update(report=report, kwargs=kwargs)
        assert "source_transformer_identity" in report
        return report

    monkeypatch.setattr(
        train_module,
        "validate_action_migration_report_for_contract",
        validate,
    )
    identity = {"sha256": "a" * 64}
    report = {"source_checkpoint_sha256": "a" * 64}
    config = SimpleNamespace(
        checkpoint_source_action_dim=20,
        checkpoint_source_action_schema="ee20_pi05",
        initialized_target_only_prefixes=("agilex_wrench_",),
    )
    codec = SimpleNamespace(
        spec=SimpleNamespace(name="qpos14_joint_absolute_v1", dim=14)
    )

    result = train_module._validate_initial_action_migration_report(
        report=report,
        released_transformer_identity=identity,
        released_transformer_sha256="a" * 64,
        config=config,
        action_codec=codec,
    )

    assert result["source_transformer_identity"] is identity
    assert captured["kwargs"] == {
        "source_action_dim": 20,
        "source_action_schema": "ee20_pi05",
        "target_action_dim": 14,
        "target_action_schema": "qpos14_joint_absolute_v1",
        "initialized_target_only_prefixes": ("agilex_wrench_",),
    }


def test_agilex_qpos14_strict_weights_only_init_inherits_migration_report(
    train_module: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    expected = {"source_action_dim": 20, "target_action_dim": 14}

    def load(checkpoint: Path, **kwargs: object) -> dict[str, object]:
        captured.update(checkpoint=checkpoint, kwargs=kwargs)
        return expected

    monkeypatch.setattr(
        train_module,
        "load_validated_action_migration_report_for_contract",
        load,
    )
    config = SimpleNamespace(
        dataset_adapter="worldarena_agilex_qpos14",
        initialized_target_only_prefixes=("local_tactile_", "agilex_wrench_"),
    )
    codec = SimpleNamespace(
        spec=SimpleNamespace(name="qpos14_joint_absolute_v1", dim=14)
    )

    result = train_module._load_agilex_qpos14_strict_init_migration_report(
        init_from=tmp_path / "checkpoint",
        resume_from=None,
        checkpoint_compatibility="strict",
        config=config,
        action_codec=codec,
    )

    assert result == expected
    assert captured == {
        "checkpoint": tmp_path / "checkpoint",
        "kwargs": {
            "source_action_dim": 20,
            "source_action_schema": "ee20_pi05",
            "target_action_dim": 14,
            "target_action_schema": "qpos14_joint_absolute_v1",
            "initialized_target_only_prefixes": (
                "local_tactile_",
                "agilex_wrench_",
            ),
        },
    }


@pytest.mark.parametrize(
    ("dataset_adapter", "compatibility", "resume"),
    (
        ("worldarena_franka_ee10", "strict", None),
        ("worldarena_agilex_qpos14", "migrate_action", None),
        ("worldarena_agilex_qpos14", "strict", "resume"),
    ),
)
def test_strict_init_migration_inheritance_is_agilex_stage_b_only(
    train_module: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_adapter: str,
    compatibility: str,
    resume: str | None,
) -> None:
    monkeypatch.setattr(
        train_module,
        "load_validated_action_migration_report_for_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unrelated initialization must not load AgileX lineage")
        ),
    )

    result = train_module._load_agilex_qpos14_strict_init_migration_report(
        init_from=tmp_path / "checkpoint",
        resume_from=resume,
        checkpoint_compatibility=compatibility,
        config=SimpleNamespace(dataset_adapter=dataset_adapter),
        action_codec=SimpleNamespace(
            spec=SimpleNamespace(name="qpos14_joint_absolute_v1", dim=14)
        ),
    )

    assert result is None
