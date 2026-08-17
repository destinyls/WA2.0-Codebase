# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Masking and initialization tests for AgileX wrench conditioning."""

from __future__ import annotations

import torch
import pytest

from n0_twam.models.wrench_conditioner import WrenchConditioner


def _module() -> WrenchConditioner:
    return WrenchConditioner(hidden_dim=16, max_frames=8, arm_count=2)


def test_wrench_conditioner_is_zero_initialized_and_shape_stable() -> None:
    module = _module()
    wrench = torch.randn(2, 3, 2, 6)
    available = torch.ones(2, 3, 2, dtype=torch.bool)
    temporal = torch.ones(2, 3, dtype=torch.bool)
    cond_drop = torch.zeros(2, dtype=torch.bool)

    output = module(
        wrench,
        wrench_available_mask=available,
        temporal_valid_mask=temporal,
        contact_cond_drop=cond_drop,
    )

    assert output.shape == (2, 6, 16)
    assert torch.count_nonzero(output) == 0


def test_wrench_conditioner_masked_placeholders_do_not_leak() -> None:
    module = _module()
    with torch.no_grad():
        module.output_projection.weight.fill_(0.25)
        module.output_projection.bias.fill_(0.5)
    base = torch.zeros(1, 3, 2, 6)
    random_placeholder = torch.randn_like(base) * 1000.0
    available = torch.tensor([[[True, False], [False, False], [True, True]]])
    temporal = torch.tensor([[True, False, True]])
    cond_drop = torch.tensor([False])

    first = module(
        base,
        wrench_available_mask=available,
        temporal_valid_mask=temporal,
        contact_cond_drop=cond_drop,
    )
    second = module(
        torch.where(available[..., None], base, random_placeholder),
        wrench_available_mask=available,
        temporal_valid_mask=temporal,
        contact_cond_drop=cond_drop,
    )

    torch.testing.assert_close(first, second)
    flattened_mask = (available & temporal[:, :, None]).reshape(1, -1)
    assert torch.count_nonzero(first[~flattened_mask]) == 0


def test_wrench_conditioner_contact_drop_removes_all_conditioning() -> None:
    module = _module()
    with torch.no_grad():
        module.output_projection.weight.fill_(1.0)
        module.output_projection.bias.fill_(1.0)
    output = module(
        torch.randn(1, 2, 2, 6),
        wrench_available_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        temporal_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        contact_cond_drop=torch.ones(1, dtype=torch.bool),
    )

    assert torch.count_nonzero(output) == 0


def test_wrench_conditioner_rejects_noncanonical_shapes() -> None:
    module = _module()

    with pytest.raises(ValueError, match=r"\[B,F,A,6\]"):
        module(
            torch.zeros(1, 2, 6),
            wrench_available_mask=torch.ones(1, 2, 2, dtype=torch.bool),
            temporal_valid_mask=torch.ones(1, 2, dtype=torch.bool),
            contact_cond_drop=torch.zeros(1, dtype=torch.bool),
        )
