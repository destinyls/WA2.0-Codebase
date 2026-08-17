# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import pytest
import torch
from torch import nn

from n0_twam.configs.twam_track32_franka_cfg import twam_track32_franka_cfg
from n0_twam.models.trainability import configure_parameter_trainability


class _ToyMoT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleDict(
            {
                "video": nn.Linear(2, 2),
                "action": nn.Linear(2, 2),
                "tactile": nn.Linear(2, 2),
            }
        )


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_patch_embed = nn.Linear(2, 2)
        self.action_proj = nn.Linear(2, 2)
        self.tactile_patch_embed = nn.Linear(2, 2)
        self.sensor_id_embed = nn.Embedding(2, 2)
        self.local_tactile_proj = nn.Linear(2, 2)
        self.contact_gate = nn.Linear(2, 2)
        self.agilex_wrench_conditioner = nn.Linear(2, 2)
        self.mot = _ToyMoT()


def test_franka_vision_only_preserves_pretrained_tactile_expert_shape() -> None:
    assert twam_track32_franka_cfg.use_mot is True
    assert twam_track32_franka_cfg.mot_warmstart_experts == (
        "video",
        "action",
        "tactile",
    )
    assert twam_track32_franka_cfg.mot_expert_hidden_dim["tactile"] == 1024
    assert twam_track32_franka_cfg.mot_expert_ffn_dim["tactile"] == 4096


def test_disabled_tactile_parameters_are_excluded_from_adamw() -> None:
    torch.manual_seed(7)
    model = _ToyModel()
    contract = configure_parameter_trainability(
        model,
        tactile_mode="disabled",
        freeze_tactile_parameters=True,
        tactile_profile="vision_only",
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.1,
        weight_decay=0.1,
    )
    loss = model.video_patch_embed(torch.ones(1, 2)).sum()
    loss += model.action_proj(torch.ones(1, 2)).sum()
    loss += model.mot.experts["action"](torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()

    assert contract.policy == "freeze_tactile_only_v1"
    assert contract.tactile_profile == "vision_only"
    assert contract.frozen_parameter_count == len(before)
    assert contract.trainable_parameter_count > 0
    assert len(contract.contract_sha256) == 64
    assert any(name.startswith("agilex_wrench_conditioner") for name in before)
    for name, parameter in model.named_parameters():
        if name in before:
            assert not parameter.requires_grad
            torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)


def test_disabled_tactile_requires_an_explicit_freeze() -> None:
    model = _ToyModel()

    try:
        configure_parameter_trainability(
            model,
            tactile_mode="disabled",
            freeze_tactile_parameters=False,
            tactile_profile="vision_only",
        )
    except ValueError as error:
        assert "must freeze" in str(error)
    else:  # pragma: no cover
        raise AssertionError("disabled tactile mode accepted trainable tactile weights")


@pytest.mark.parametrize("profile", ["vision_tactile", "mixed"])
def test_enabled_profiles_keep_tactile_parameters_trainable(profile: str) -> None:
    model = _ToyModel()

    contract = configure_parameter_trainability(
        model,
        tactile_mode="enabled",
        freeze_tactile_parameters=False,
        tactile_profile=profile,
    )

    assert contract.policy == "train_all_v1"
    assert contract.tactile_profile == profile
    assert contract.frozen_parameter_count == 0
    assert all(parameter.requires_grad for parameter in model.parameters())
