# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from n0_twam.checkpointing.runtime_provenance import LOCAL_EXECUTION_TIER
from n0_twam.track31.local_provenance import LocalProvenance, package_import_root
from n0_twam.track32.preflight import _require_accelerator_profile
from n0_twam.track32.request import (
    load_track32_train_request,
    track32_train_request_template,
)
from n0_twam.track32.runner import (
    build_launch_plan,
    build_training_command,
    build_training_environment,
)


def _payload() -> dict[str, object]:
    payload = track32_train_request_template()
    paths = payload["paths"]
    artifacts = payload["artifacts"]
    assert isinstance(paths, dict) and isinstance(artifacts, dict)
    paths["empty_embedding_sha256"] = "a" * 64
    paths["init_transformer_sha256"] = "b" * 64
    for digest, field in zip(
        ("c", "d", "e", "f", "0", "1", "2"),
        (
            "prepare_receipt_sha256",
            "conversion_report_sha256",
            "latent_inventory_file_sha256",
            "train_view_sha256",
            "validation_view_sha256",
            "normalizer_source_view_sha256",
            "normalizer_sha256",
        ),
    ):
        artifacts[field] = digest * 64
    return payload


def _write_request(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    path = tmp_path / "track32.train.json"
    path.write_text(json.dumps(payload or _payload()), encoding="utf-8")
    return path


def _provenance(tmp_path: Path) -> LocalProvenance:
    return LocalProvenance(
        invocation_id="track32-franka",
        code_manifest_path=tmp_path / "code.json",
        environment_manifest_path=tmp_path / "env.json",
        launch_receipt_path=tmp_path / "launch.json",
        runtime_source_identity={
            "schema_version": 2,
            "execution_tier": LOCAL_EXECUTION_TIER,
            "code_manifest_sha256": "1" * 64,
            "environment_manifest_sha256": "2" * 64,
            "empty_embedding_sha256": "a" * 64,
            "package_version": "0.1.0",
        },
        checkpoint_invocation_identity={
            "schema_version": 2,
            "execution_tier": LOCAL_EXECUTION_TIER,
            "invocation_id": "track32-franka",
            "launch_receipt_sha256": "3" * 64,
        },
    )


def test_track32_request_and_plan_pin_franka_contract(tmp_path: Path) -> None:
    request = load_track32_train_request(_write_request(tmp_path))
    plan = build_launch_plan(request)

    assert request.runtime.devices == tuple(range(8))
    assert request.runtime.accelerator_profile == "portable"
    assert plan["world_size"] == 8
    assert plan["recipe"]["action_loss_profile"] == "legacy_v1"
    assert plan["model_contract"] == {
        "wire_action_schema": "franka_end_pose_base_xyzw8_v2",
        "derived_action_schema": "franka_ee10_rot6d_columns_from_xyzw_v2",
        "model_action_schema": "ee20_absee",
        "active_action_channels": list(range(10)),
        "tactile_profile": "vision_only",
        "tactile_mode": "disabled",
        "frozen_tactile_parameters": True,
    }
    command = build_training_command(request)
    assert "torch.distributed.run" in command
    assert "n0_twam.train" in command
    assert "track32_franka" in command


def test_track32_environment_is_clean_and_profile_bound(tmp_path: Path) -> None:
    request = load_track32_train_request(_write_request(tmp_path))
    environment = build_training_environment(
        request,
        _provenance(tmp_path),
        environ={
            "PATH": "/usr/bin",
            "PYTHONPATH": "/untrusted",
            "PYTHONHOME": "/untrusted",
            "LD_PRELOAD": "/untrusted.so",
            "CUDA_VISIBLE_DEVICES": "7",
            "GLOO_SOCKET_IFNAME": "eno1",
            "NCCL_SOCKET_IFNAME": "eno1",
            "N0_ROGUE": "1",
            "N0_TRACK32_ACTION_LOSS_PROFILE": "franka_trajectory_fit_v1",
        },
    )

    assert environment["PATH"] == "/usr/bin"
    assert environment["PYTHONPATH"] == str(package_import_root())
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert environment["N0_TRACK32_ACCELERATOR_PROFILE"] == "portable"
    assert environment["N0_TRACK32_ACTION_LOSS_PROFILE"] == "legacy_v1"
    assert environment["N0_TRACK32_TRAIN_VIEW_ID"] == "franka_dev_train540_v1"
    assert (
        environment["N0_TRACK32_NORMALIZER_SOURCE_VIEW_ID"]
        == "franka_dev_train540_v1"
    )
    assert environment["N0_MOT_ACTIVATION_CHECKPOINTING"] == "1"
    assert environment["N0_FLEX_ATTENTION_BACKEND"] == "grouped_sdpa"
    assert "PYTHONHOME" not in environment
    assert "LD_PRELOAD" not in environment
    assert "GLOO_SOCKET_IFNAME" not in environment
    assert "NCCL_SOCKET_IFNAME" not in environment
    assert "N0_ROGUE" not in environment


def test_completed_weights_init_maps_receipt_without_rank_local_rehash(
    tmp_path: Path,
) -> None:
    payload = _payload()
    paths = payload["paths"]
    artifacts = payload["artifacts"]
    assert isinstance(paths, dict) and isinstance(artifacts, dict)
    paths["init_checkpoint_complete_sha256"] = "4" * 64
    artifacts["full_verification_receipt_sha256"] = "5" * 64
    request = load_track32_train_request(_write_request(tmp_path, payload))

    environment = build_training_environment(
        request,
        _provenance(tmp_path),
        environ={"PATH": "/usr/bin"},
    )

    assert environment["N0_TRACK32_INIT_CHECKPOINT_COMPLETE_SHA256"] == "4" * 64
    assert environment["N0_TRACK32_INIT_RECEIPT_VALIDATED"] == "1"
    assert environment["N0_TRACK32_FULL_VERIFICATION_RECEIPT_SHA256"] == "5" * 64


def test_hcu_profile_binds_a_verified_collective_interface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    runtime["accelerator_profile"] = "hcu_performance"
    runtime["collective_network_interface"] = "bond1"
    request = load_track32_train_request(_write_request(tmp_path, payload))
    monkeypatch.setattr(
        "n0_twam.track32.runner._available_network_interfaces",
        lambda: frozenset({"lo", "bond1"}),
    )

    environment = build_training_environment(
        request,
        _provenance(tmp_path),
        environ={
            "PATH": "/usr/bin",
            "GLOO_SOCKET_IFNAME": "eno1",
            "NCCL_SOCKET_IFNAME": "eno1",
        },
    )

    assert environment["GLOO_SOCKET_IFNAME"] == "bond1"
    assert environment["NCCL_SOCKET_IFNAME"] == "bond1"
    assert environment["N0_TRACK32_COLLECTIVE_NETWORK_INTERFACE"] == "bond1"
    assert environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert environment["TORCH_NCCL_BLOCKING_WAIT"] == "1"


def test_hcu_profile_rejects_an_unavailable_collective_interface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    runtime["accelerator_profile"] = "hcu_performance"
    runtime["collective_network_interface"] = "eno1"
    request = load_track32_train_request(_write_request(tmp_path, payload))
    monkeypatch.setattr(
        "n0_twam.track32.runner._available_network_interfaces",
        lambda: frozenset({"lo", "bond1"}),
    )

    with pytest.raises(ValueError, match="unavailable.*eno1"):
        build_training_environment(request, _provenance(tmp_path), environ={})


def test_hcu_preflight_rechecks_profile_and_collective_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "N0_TRACK32_ACCELERATOR_PROFILE": "hcu_performance",
        "N0_FLEX_ATTENTION_BACKEND": "grouped_flash_attn",
        "N0_MOT_CROSS_ATTENTION_BACKEND": "flash_attn",
        "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "1",
        "N0_FSDP_REDUCE_DTYPE": "bfloat16",
        "N0_FSDP_EXPERT_RESHARD_POLICY": "after_layer",
        "N0_MOT_ACTIVATION_CHECKPOINTING": "0",
        "N0_TRACK32_COLLECTIVE_NETWORK_INTERFACE": "bond1",
        "NCCL_SOCKET_IFNAME": "bond1",
        "GLOO_SOCKET_IFNAME": "bond1",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "n0_twam.track32.preflight.socket.if_nameindex",
        lambda: [(1, "lo"), (2, "bond1")],
    )

    _require_accelerator_profile()

    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eno1")
    with pytest.raises(ValueError, match="collective socket interface mismatch"):
        _require_accelerator_profile()


def test_portable_preflight_rejects_ambient_hcu_network_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "N0_TRACK32_ACCELERATOR_PROFILE": "portable",
        "N0_FLEX_ATTENTION_BACKEND": "grouped_sdpa",
        "N0_MOT_CROSS_ATTENTION_BACKEND": "sdpa",
        "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "0",
        "N0_FSDP_REDUCE_DTYPE": "float32",
        "N0_FSDP_EXPERT_RESHARD_POLICY": "after_call",
        "N0_MOT_ACTIVATION_CHECKPOINTING": "1",
        "NCCL_SOCKET_IFNAME": "eno1",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("N0_TRACK32_COLLECTIVE_NETWORK_INTERFACE", raising=False)
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)

    with pytest.raises(ValueError, match="inherited HCU network bindings"):
        _require_accelerator_profile()


def test_track32_request_rejects_unknown_profile_and_output_overlap(
    tmp_path: Path,
) -> None:
    payload = _payload()
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    runtime["accelerator_profile"] = "automatic"
    with pytest.raises(ValueError, match="portable or hcu_performance"):
        load_track32_train_request(_write_request(tmp_path, payload))

    payload = _payload()
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    runtime["accelerator_profile"] = "hcu_performance"
    with pytest.raises(ValueError, match="requires.*collective_network_interface"):
        load_track32_train_request(_write_request(tmp_path, payload))

    payload = _payload()
    runtime = payload["runtime"]
    assert isinstance(runtime, dict)
    runtime["collective_network_interface"] = "bond1"
    with pytest.raises(ValueError, match="portable requires"):
        load_track32_train_request(_write_request(tmp_path, payload))

    payload = _payload()
    paths = payload["paths"]
    assert isinstance(paths, dict)
    paths["output_root"] = "./artifacts/franka/output"
    with pytest.raises(ValueError, match="disjoint"):
        load_track32_train_request(_write_request(tmp_path, payload))


def test_track32_request_binds_trajectory_action_loss_profile(tmp_path: Path) -> None:
    payload = _payload()
    train = payload["train"]
    assert isinstance(train, dict)
    train["action_loss_profile"] = "franka_trajectory_fit_v1"
    request = load_track32_train_request(_write_request(tmp_path, payload))

    environment = build_training_environment(
        request,
        _provenance(tmp_path),
        environ={"N0_TRACK32_ACTION_LOSS_PROFILE": "legacy_v1"},
    )

    assert request.train.action_loss_profile == "franka_trajectory_fit_v1"
    assert environment["N0_TRACK32_ACTION_LOSS_PROFILE"] == (
        "franka_trajectory_fit_v1"
    )


def test_track32_request_rejects_unknown_action_loss_profile(tmp_path: Path) -> None:
    payload = _payload()
    train = payload["train"]
    assert isinstance(train, dict)
    train["action_loss_profile"] = "ambient_override"

    with pytest.raises(ValueError, match="train.action_loss_profile"):
        load_track32_train_request(_write_request(tmp_path, payload))
