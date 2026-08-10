"""Tests for time-sampled HCU utilization acceptance evidence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "script"
    / "track3_1"
    / "measure_hcu_utilization.py"
)
_SPEC = importlib.util.spec_from_file_location("measure_hcu_utilization", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _hy_smi(hcu_values: list[float], *, vram: float = 55.0) -> str:
    rows = [
        "HCU Temp AvgPwr Perf PwrCap VRAM HCU% Dec% Enc% Mode",
    ]
    rows.extend(
        f"{index} 55.0C 450.0W auto 1000.0W {vram}% {value}% 0.0% 0.0% Normal"
        for index, value in enumerate(hcu_values)
    )
    return "\n".join(rows)


def test_parse_hy_smi_requires_exact_normal_eight_hcu_roster() -> None:
    readings = _MODULE.parse_hy_smi(_hy_smi([80.0] * 8), node="n1")

    assert len(readings) == 8
    assert readings[0].hcu_index == 0
    assert readings[0].utilization_percent == 80.0
    assert readings[0].vram_percent == 55.0
    assert readings[0].mode == "Normal"

    with pytest.raises(ValueError, match="exact HCU roster"):
        _MODULE.parse_hy_smi(_hy_smi([80.0] * 7), node="n1")


def test_hy_smi_command_uses_local_proxy_jump_and_strict_known_hosts(
    tmp_path: Path,
) -> None:
    known_hosts = tmp_path / "known_hosts"
    command = _MODULE._hy_smi_command(
        node="10.0.0.2",
        port=36000,
        known_hosts=known_hosts,
        jump_host="hpu",
    )

    assert command[:3] == ["ssh", "-J", "hpu"]
    assert f"UserKnownHostsFile={known_hosts}" in command
    assert command[-2:] == ["root@10.0.0.2", "/opt/hyhal/bin/hy-smi"]


def test_utilization_report_passes_only_a_time_sampled_eighty_percent_run() -> None:
    snapshots = []
    for sample_index in range(3):
        readings = []
        for node in ("n1", "n2"):
            readings.extend(
                _MODULE.parse_hy_smi(
                    _hy_smi([80.0 + sample_index] * 8),
                    node=node,
                )
            )
        snapshots.append(
            _MODULE.UtilizationSnapshot(
                timestamp_unix_ns=sample_index + 1,
                readings=tuple(readings),
            )
        )

    report = _MODULE.build_utilization_report(
        snapshots=tuple(snapshots),
        nodes=("n1", "n2"),
        expected_hcus_per_node=8,
        phase="steady_optimizer_steps",
        minimum_cluster_mean_percent=80.0,
        minimum_device_mean_percent=70.0,
        maximum_vram_percent=90.0,
    )

    assert report["status"] == "PASS"
    assert report["cluster"]["mean_hcu_percent"] == 81.0
    assert report["cluster"]["sample_count"] == 3
    assert report["cluster"]["device_count"] == 16
    assert report["failures"] == []


def test_utilization_report_rejects_high_single_snapshot_and_idle_device() -> None:
    readings = []
    for node in ("n1", "n2"):
        values = [100.0] * 8
        if node == "n2":
            values[-1] = 0.0
        readings.extend(_MODULE.parse_hy_smi(_hy_smi(values), node=node))

    report = _MODULE.build_utilization_report(
        snapshots=(
            _MODULE.UtilizationSnapshot(
                timestamp_unix_ns=1,
                readings=tuple(readings),
            ),
        ),
        nodes=("n1", "n2"),
        expected_hcus_per_node=8,
        phase="steady_optimizer_steps",
        minimum_cluster_mean_percent=80.0,
        minimum_device_mean_percent=70.0,
        maximum_vram_percent=90.0,
    )

    assert report["status"] == "FAIL"
    assert any("sample count" in failure for failure in report["failures"])
    assert any("device mean" in failure for failure in report["failures"])
