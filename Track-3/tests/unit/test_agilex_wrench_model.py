# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""AgileX-only wrench topology tests for the shared WAM model."""

from __future__ import annotations

import torch

from n0_twam.models.model import WanTransformer3DModel
from n0_twam.models.trainability import configure_parameter_trainability


def _model(
    *,
    enabled: bool,
    instantiate: bool | None = None,
) -> WanTransformer3DModel:
    return WanTransformer3DModel(
        patch_size=[1, 1, 1],
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=4,
        out_channels=4,
        action_dim=14,
        text_dim=8,
        freq_dim=4,
        ffn_dim=16,
        num_layers=1,
        rope_max_seq_len=8,
        tactile_in_channels=3,
        tactile_num_tokens=1,
        tactile_encoder_dim=8,
        max_tactile_streams=2,
        use_local_tactile=False,
        instantiate_local_tactile=True,
        use_contact_gate=False,
        use_wrench_conditioner=enabled,
        instantiate_wrench_conditioner=instantiate,
        wrench_arm_count=2,
        wrench_max_frames=4,
    )


def test_agilex_wrench_topology_is_opt_in() -> None:
    legacy = _model(enabled=False)
    agilex = _model(enabled=True)

    assert not any(key.startswith("agilex_wrench") for key in legacy.state_dict())
    assert any(key.startswith("agilex_wrench") for key in agilex.state_dict())


def test_agilex_wrench_branch_is_identity_at_initialization() -> None:
    model = _model(enabled=True)
    tokens = model._encode_agilex_wrench_condition(
        torch.randn(1, 2, 2, 6),
        wrench_available_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        temporal_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        contact_cond_drop=torch.zeros(1, dtype=torch.bool),
    )
    action_hidden = torch.randn(1, 6, 8)

    conditioned = model._apply_agilex_wrench_cross_attn(action_hidden, tokens)

    assert tokens.shape == (1, 4, 8)
    torch.testing.assert_close(conditioned, action_hidden)


def test_vision_only_preserves_but_freezes_contact_topology() -> None:
    model = _model(enabled=False, instantiate=True)
    wrench_names = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("agilex_wrench_")
    }
    local_names = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("local_tactile_")
    }

    contract = configure_parameter_trainability(
        model,
        tactile_mode="disabled",
        freeze_tactile_parameters=True,
        tactile_profile="vision_only",
    )

    assert wrench_names
    assert local_names
    assert wrench_names <= set(contract.frozen_parameter_names)
    assert local_names <= set(contract.frozen_parameter_names)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name in wrench_names | local_names
    )
