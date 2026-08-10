# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from n0_twam.evaluation.tactile_checkpoint import audit_tactile_head_contract
from n0_twam.evaluation.tactile_model_contract import (
    audit_loaded_tactile_runtime,
)


def _contract() -> dict[str, object]:
    return {
        "patch_size": [1, 2, 2],
        "snr_shift": 5.0,
        "use_local_tactile": True,
        "max_tactile_streams": 2,
        "tactile_in_channels": 3,
        "tactile_num_tokens": 4,
        "tactile_encoder_dim": 256,
        "tactile_latent_channels": 48,
    }


def _checkpoint_config() -> dict[str, object]:
    return {
        **{
            key: value
            for key, value in _contract().items()
            if key not in {"snr_shift", "tactile_latent_channels"}
        },
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "rope_max_seq_len": 16,
    }


def _head_tensors() -> dict[str, torch.Tensor]:
    inner_dim = 8
    patch_channels = 48 * 4
    return {
        "tactile_patch_embed.weight": torch.zeros(inner_dim, patch_channels),
        "tactile_patch_embed.bias": torch.zeros(inner_dim),
        "sensor_id_embed.weight": torch.zeros(2, inner_dim),
        "tactile_norm.weight": torch.zeros(inner_dim),
        "tactile_norm.bias": torch.zeros(inner_dim),
        "tactile_proj_out.weight": torch.zeros(patch_channels, inner_dim),
        "tactile_proj_out.bias": torch.zeros(patch_channels),
        "local_tactile_patch_embed.weight": torch.zeros(inner_dim, patch_channels),
        "local_tactile_patch_embed.bias": torch.zeros(inner_dim),
        "local_tactile_sensor_embed.weight": torch.zeros(2, inner_dim),
        "local_tactile_frame_embed.weight": torch.zeros(16, inner_dim),
        "local_tactile_h_embed.weight": torch.zeros(16, inner_dim),
        "local_tactile_w_embed.weight": torch.zeros(16, inner_dim),
        "local_tactile_norm.weight": torch.zeros(inner_dim),
        "local_tactile_norm.bias": torch.zeros(inner_dim),
        "local_tactile_cross_attn.to_q.weight": torch.zeros(inner_dim, inner_dim),
        "local_tactile_cross_attn.to_q.bias": torch.zeros(inner_dim),
        "local_tactile_cross_attn.to_k.weight": torch.zeros(inner_dim, inner_dim),
        "local_tactile_cross_attn.to_k.bias": torch.zeros(inner_dim),
        "local_tactile_cross_attn.to_v.weight": torch.zeros(inner_dim, inner_dim),
        "local_tactile_cross_attn.to_v.bias": torch.zeros(inner_dim),
        "local_tactile_cross_attn.to_out.0.weight": torch.zeros(inner_dim, inner_dim),
        "local_tactile_cross_attn.to_out.0.bias": torch.zeros(inner_dim),
        "local_tactile_cross_attn.norm_q.weight": torch.zeros(inner_dim),
        "local_tactile_cross_attn.norm_k.weight": torch.zeros(inner_dim),
    }


def test_tactile_head_audit_binds_global_and_local_sentinels(tmp_path) -> None:
    weights_path = tmp_path / "model.safetensors"
    save_file(_head_tensors(), weights_path)

    result = audit_tactile_head_contract(
        weights_path,
        model_contract=_contract(),
        config_payload=_checkpoint_config(),
    )

    assert result["inner_dim"] == 8
    assert result["patch_volume"] == 4
    assert result["required_tensor_shapes"]["tactile_proj_out.weight"] == [
        192,
        8,
    ]
    assert (
        "local_tactile_cross_attn.to_out.0.weight" in result["required_tensor_shapes"]
    )


def test_tactile_head_audit_rejects_missing_local_sentinel(tmp_path) -> None:
    weights_path = tmp_path / "model.safetensors"
    tensors = _head_tensors()
    tensors.pop("local_tactile_cross_attn.to_out.0.weight")
    save_file(tensors, weights_path)

    with pytest.raises(ValueError, match="missing required tactile tensors"):
        audit_tactile_head_contract(
            weights_path,
            model_contract=_contract(),
            config_payload=_checkpoint_config(),
        )


def test_loaded_runtime_contract_checks_schedulers_and_model() -> None:
    config = SimpleNamespace(**_contract())
    model_config = SimpleNamespace(**_checkpoint_config())
    transformer = SimpleNamespace(
        config=model_config,
        patch_size=[1, 2, 2],
        use_local_tactile=True,
        max_tactile_streams=2,
        tactile_latent_channels=48,
        state_dict=lambda: _head_tensors(),
    )
    trainer = SimpleNamespace(
        config=config,
        patch_size=[1, 2, 2],
        train_scheduler_latent=SimpleNamespace(shift=5.0),
        train_scheduler_tactile=SimpleNamespace(shift=5.0),
        transformer=transformer,
    )

    result = audit_loaded_tactile_runtime(trainer, _contract())

    assert result["model_contract"] == _contract()
    assert result["latent_scheduler_shift"] == 5.0
    assert result["tactile_scheduler_shift"] == 5.0

    trainer.train_scheduler_tactile.shift = 3.0
    with pytest.raises(ValueError, match="tactile scheduler shift"):
        audit_loaded_tactile_runtime(trainer, _contract())
