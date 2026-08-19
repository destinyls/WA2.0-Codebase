"""Regression tests for the six-node Track 3.2 HSDP launch contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from n0_twam.track32 import preflight
from n0_twam.track32.multinode import (
    _multinode_execution_contract,
    _secret_free_environment,
)

_HCA_ROSTER = "shca_0:1,shca_1:1,shca_2:1,shca_3:1"
_PLUGIN_DIR = "/opt/n0_twam/rccl_plugin"
_PLUGIN_HOST_DIR = "/opt/hpc/software/app/rccl/shca_rdma_plugins/v8/lib"
_PLUGIN_FILENAME = "librccl-net-shca.so.0.0.0"
_PLUGIN_SHA256 = (
    "20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"
)


def _install_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "N0_TRACK32_EXPECTED_LOCAL_WORLD_SIZE": "8",
        "N0_FSDP_TOPOLOGY": "hsdp",
        "N0_FSDP_SHARD_SIZE": "8",
        "N0_TRACK32_EXPECTED_IB_UVERBS": "4",
        "N0_TRACK32_RCCL_PLUGIN_DIR": _PLUGIN_DIR,
        "N0_TRACK32_RCCL_PLUGIN_HOST_DIR": _PLUGIN_HOST_DIR,
        "N0_TRACK32_RCCL_PLUGIN_FILENAME": _PLUGIN_FILENAME,
        "N0_TRACK32_RCCL_PLUGIN_SHA256": _PLUGIN_SHA256,
        "LD_LIBRARY_PATH": f"{_PLUGIN_DIR}:/opt/dtk/lib",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": _HCA_ROSTER,
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_IB_QPS_PER_CONNECTION": "4",
        "NCCL_IB_TC": "160",
        "NCCL_IB_TIMEOUT": "22",
        "NCCL_NET_PLUGIN": "shca",
        "NCCL_NET_GDR_LEVEL": "PHB",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_multinode_environment_drops_credentials_before_persistence() -> None:
    filtered = _secret_free_environment(
        {
            "PATH": "/opt/runtime/bin",
            "LD_LIBRARY_PATH": "/opt/runtime/lib",
            "HUB_TOKEN": "do-not-persist",
            "AWS_SECRET_ACCESS_KEY": "do-not-persist",
            "SSH_AUTH_SOCK": "/run/user/1000/ssh-agent.socket",
        }
    )

    assert filtered == {
        "PATH": "/opt/runtime/bin",
        "LD_LIBRARY_PATH": "/opt/runtime/lib",
    }


def test_multinode_execution_contract_builds_six_by_eight_hsdp() -> None:
    environment, contract = _multinode_execution_contract(
        node_count=6,
        local_world_size=8,
        nccl_ib_hca=_HCA_ROSTER,
    )

    assert environment["N0_FSDP_TOPOLOGY"] == "hsdp"
    assert environment["N0_FSDP_SHARD_SIZE"] == "8"
    assert environment["NCCL_NET_PLUGIN"] == "shca"
    assert environment["NCCL_NET_GDR_LEVEL"] == "PHB"
    assert "NCCL_DMABUF_ENABLE" not in environment
    assert contract["mesh_shape"] == [6, 8]
    assert contract["replicate_size"] == 6
    assert contract["shard_size"] == 8
    assert contract["required_rdma_devices"] == [
        "rdma_cm",
        "uverbs0",
        "uverbs1",
        "uverbs2",
        "uverbs3",
    ]


def test_multinode_preflight_accepts_exact_shca_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_environment(monkeypatch)
    devices: list[Path] = []
    bindings: list[tuple[str, int]] = []
    monkeypatch.setattr(preflight, "_require_character_device", devices.append)
    monkeypatch.setattr(preflight, "_require_rccl_network_plugin", lambda: None)
    monkeypatch.setattr(
        preflight,
        "_require_hca_binding",
        lambda entry, *, uverbs_index: bindings.append((entry, uverbs_index)),
    )

    preflight._require_multinode_hsdp_and_ib()

    assert devices == [
        Path("/dev/infiniband/rdma_cm"),
        Path("/dev/infiniband/uverbs0"),
        Path("/dev/infiniband/uverbs1"),
        Path("/dev/infiniband/uverbs2"),
        Path("/dev/infiniband/uverbs3"),
    ]
    assert bindings == [
        ("shca_0:1", 0),
        ("shca_1:1", 1),
        ("shca_2:1", 2),
        ("shca_3:1", 3),
    ]


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("N0_FSDP_TOPOLOGY", "global_shard"),
        ("N0_FSDP_SHARD_SIZE", "48"),
        ("NCCL_IB_DISABLE", "1"),
        ("NCCL_NET_PLUGIN", "none"),
        ("NCCL_NET_GDR_LEVEL", "2"),
    ),
)
def test_multinode_preflight_rejects_transport_drift(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    _install_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(preflight, "_require_character_device", lambda _path: None)
    monkeypatch.setattr(preflight, "_require_rccl_network_plugin", lambda: None)
    monkeypatch.setattr(
        preflight,
        "_require_hca_binding",
        lambda _entry, *, uverbs_index: None,
    )

    with pytest.raises(ValueError, match="environment mismatch"):
        preflight._require_multinode_hsdp_and_ib()


def test_multinode_preflight_rejects_dmabuf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_environment(monkeypatch)
    monkeypatch.setenv("NCCL_DMABUF_ENABLE", "1")
    monkeypatch.setattr(preflight, "_require_character_device", lambda _path: None)
    monkeypatch.setattr(preflight, "_require_rccl_network_plugin", lambda: None)
    monkeypatch.setattr(
        preflight,
        "_require_hca_binding",
        lambda _entry, *, uverbs_index: None,
    )

    with pytest.raises(ValueError, match="must remain unset"):
        preflight._require_multinode_hsdp_and_ib()
