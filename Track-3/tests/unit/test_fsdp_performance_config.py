"""Unit tests for explicit Stage A FSDP performance controls."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from n0_twam.distributed.fsdp import (
    _activation_checkpointing_enabled,
    _build_fsdp_device_mesh,
    _keep_expert_params_between_pre_post,
    _resolve_expert_reshard_policy,
    _resolve_fsdp_reduce_dtype,
    _resolve_fsdp_topology,
    apply_ac,
    capture_fsdp_execution_contract,
    shard_model,
)


def test_fsdp_performance_controls_preserve_legacy_defaults() -> None:
    with patch.dict(os.environ):
        os.environ.pop("N0_FSDP_REDUCE_DTYPE", None)
        os.environ.pop("N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST", None)
        os.environ.pop("N0_FSDP_EXPERT_RESHARD_POLICY", None)
        os.environ.pop("N0_FSDP_TOPOLOGY", None)
        os.environ.pop("N0_FSDP_SHARD_SIZE", None)
        os.environ.pop("N0_MOT_ACTIVATION_CHECKPOINTING", None)

        assert _resolve_fsdp_reduce_dtype() is torch.float32
        assert not _keep_expert_params_between_pre_post()
        assert _resolve_expert_reshard_policy() == "after_call"
        assert _resolve_fsdp_topology(
            world_size=32,
            local_world_size=8,
        ).mode == "global_shard"
        assert _activation_checkpointing_enabled()


def test_fsdp_performance_controls_enable_hcu_recipe() -> None:
    with patch.dict(
        os.environ,
        {
            "N0_FSDP_REDUCE_DTYPE": "bfloat16",
            "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "1",
            "N0_MOT_ACTIVATION_CHECKPOINTING": "0",
        },
    ):
        assert _resolve_fsdp_reduce_dtype() is torch.bfloat16
        assert _keep_expert_params_between_pre_post()
        assert not _activation_checkpointing_enabled()


def test_hsdp_topology_is_node_local_four_by_eight() -> None:
    with patch.dict(
        os.environ,
        {
            "N0_FSDP_TOPOLOGY": "hsdp",
            "N0_FSDP_SHARD_SIZE": "8",
            "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "1",
            "N0_FSDP_EXPERT_RESHARD_POLICY": "after_backward",
            "N0_FSDP_REDUCE_DTYPE": "bfloat16",
        },
    ):
        topology = _resolve_fsdp_topology(
            world_size=32,
            local_world_size=8,
        )
        contract = capture_fsdp_execution_contract(
            world_size=32,
            local_world_size=8,
        )

    assert topology.mode == "hsdp"
    assert topology.mesh_shape == (4, 8)
    assert topology.replicate_size == 4
    assert topology.shard_size == 8
    assert contract == {
        "schema_version": 1,
        "topology": "hsdp",
        "mesh_shape": [4, 8],
        "replicate_size": 4,
        "shard_size": 8,
        "expert_reshard_policy": "after_backward",
        "reduce_dtype": "bfloat16",
    }


@pytest.mark.parametrize(
    ("world_size", "local_world_size", "shard_size"),
    (
        (30, 8, "8"),
        (32, 4, "8"),
        (32, 8, "0"),
    ),
)
def test_hsdp_topology_rejects_non_node_local_mesh(
    world_size: int,
    local_world_size: int,
    shard_size: str,
) -> None:
    with patch.dict(
        os.environ,
        {
            "N0_FSDP_TOPOLOGY": "hsdp",
            "N0_FSDP_SHARD_SIZE": shard_size,
        },
    ):
        with pytest.raises(ValueError):
            _resolve_fsdp_topology(
                world_size=world_size,
                local_world_size=local_world_size,
            )


def test_hsdp_builds_named_device_mesh() -> None:
    topology = SimpleNamespace(mode="hsdp", mesh_shape=(4, 8))
    sentinel = object()

    with patch(
        "n0_twam.distributed.fsdp.init_device_mesh",
        return_value=sentinel,
    ) as init_mesh:
        mesh = _build_fsdp_device_mesh(topology)

    assert mesh is sentinel
    init_mesh.assert_called_once_with(
        "cuda",
        (4, 8),
        mesh_dim_names=("replicate", "shard"),
    )


def test_shard_model_reuses_hsdp_mesh_and_defers_expert_reshard() -> None:
    first_block = SimpleNamespace()
    second_block = SimpleNamespace()
    model = SimpleNamespace(
        mot=SimpleNamespace(
            expert_names=("video", "tactile"),
            experts={
                "video": SimpleNamespace(blocks=[first_block]),
                "tactile": SimpleNamespace(blocks=[second_block]),
            },
            manual_reshard_experts=True,
        )
    )
    mesh = object()

    with (
        patch.dict(
            os.environ,
            {
                "WORLD_SIZE": "32",
                "LOCAL_WORLD_SIZE": "8",
                "N0_FSDP_TOPOLOGY": "hsdp",
                "N0_FSDP_SHARD_SIZE": "8",
                "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "1",
                "N0_FSDP_EXPERT_RESHARD_POLICY": "after_backward",
            },
        ),
        patch(
            "n0_twam.distributed.fsdp._build_fsdp_device_mesh",
            return_value=mesh,
        ),
        patch("n0_twam.distributed.fsdp.fully_shard") as fully_shard_mock,
    ):
        result = shard_model(model)

    assert result is model
    assert not model.mot.manual_reshard_experts
    assert fully_shard_mock.call_count == 3
    for call in fully_shard_mock.call_args_list:
        assert call.kwargs["mesh"] is mesh
        assert call.kwargs["reshard_after_forward"] is False or call.args[0] is model


def test_after_backward_policy_requires_parameter_reuse() -> None:
    with patch.dict(
        os.environ,
        {
            "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "0",
            "N0_FSDP_EXPERT_RESHARD_POLICY": "after_backward",
        },
    ):
        with pytest.raises(ValueError, match="requires"):
            _resolve_expert_reshard_policy()


def test_apply_ac_disables_mot_layer_recomputation() -> None:
    model = SimpleNamespace(mot=SimpleNamespace(gradient_checkpointing=True))

    with patch.dict(os.environ, {"N0_MOT_ACTIVATION_CHECKPOINTING": "0"}):
        apply_ac(model)

    assert not model.mot.gradient_checkpointing


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("N0_FSDP_REDUCE_DTYPE", "float16"),
        ("N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST", "true"),
        ("N0_FSDP_EXPERT_RESHARD_POLICY", "after_epoch"),
        ("N0_FSDP_TOPOLOGY", "ddp"),
        ("N0_FSDP_SHARD_SIZE", "eight"),
        ("N0_MOT_ACTIVATION_CHECKPOINTING", "false"),
    ),
)
def test_fsdp_performance_controls_fail_closed(name: str, value: str) -> None:
    with patch.dict(os.environ, {name: value}):
        with pytest.raises(ValueError):
            if name == "N0_FSDP_REDUCE_DTYPE":
                _resolve_fsdp_reduce_dtype()
            elif name == "N0_FSDP_EXPERT_RESHARD_POLICY":
                _resolve_expert_reshard_policy()
            elif name in ("N0_FSDP_TOPOLOGY", "N0_FSDP_SHARD_SIZE"):
                _resolve_fsdp_topology(world_size=32, local_world_size=8)
            elif name == "N0_MOT_ACTIVATION_CHECKPOINTING":
                _activation_checkpointing_enabled()
            else:
                _keep_expert_params_between_pre_post()
