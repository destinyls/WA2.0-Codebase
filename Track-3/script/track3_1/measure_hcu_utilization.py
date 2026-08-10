#!/usr/bin/env python3
"""Collect time-sampled hy-smi evidence and enforce an HCU utilization gate."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

_ROW_PATTERN = re.compile(
    r"^\s*(?P<index>\d+)\s+"
    r"(?P<temperature>\d+(?:\.\d+)?)C\s+"
    r"(?P<power>\d+(?:\.\d+)?)W\s+"
    r"\S+\s+\d+(?:\.\d+)?W\s+"
    r"(?P<vram>\d+(?:\.\d+)?)%\s+"
    r"(?P<hcu>\d+(?:\.\d+)?)%\s+"
    r"\d+(?:\.\d+)?%\s+\d+(?:\.\d+)?%\s+"
    r"(?P<mode>\S+)\s*$"
)
_SAFE_NODE_PATTERN = re.compile(r"^[A-Za-z0-9.-]+$")
_MINIMUM_TIME_SAMPLES = 3


@dataclass(frozen=True)
class HcuReading:
    """One device row from a single hy-smi snapshot."""

    node: str
    hcu_index: int
    utilization_percent: float
    vram_percent: float
    power_watts: float
    temperature_celsius: float
    mode: str


@dataclass(frozen=True)
class UtilizationSnapshot:
    """One synchronized cluster-wide measurement."""

    timestamp_unix_ns: int
    readings: tuple[HcuReading, ...]


def parse_hy_smi(
    output: str,
    *,
    node: str,
    expected_hcus: int = 8,
) -> tuple[HcuReading, ...]:
    """Parse one hy-smi table and reject incomplete or unhealthy rosters."""
    readings = []
    for line in output.splitlines():
        match = _ROW_PATTERN.match(line)
        if match is None:
            continue
        readings.append(
            HcuReading(
                node=node,
                hcu_index=int(match.group("index")),
                utilization_percent=float(match.group("hcu")),
                vram_percent=float(match.group("vram")),
                power_watts=float(match.group("power")),
                temperature_celsius=float(match.group("temperature")),
                mode=match.group("mode"),
            )
        )
    expected_indices = set(range(expected_hcus))
    actual_indices = {reading.hcu_index for reading in readings}
    if len(readings) != expected_hcus or actual_indices != expected_indices:
        raise ValueError(
            f"{node} does not contain the exact HCU roster 0..{expected_hcus - 1}"
        )
    if any(reading.mode != "Normal" for reading in readings):
        raise ValueError(f"{node} contains a non-Normal HCU")
    return tuple(sorted(readings, key=lambda reading: reading.hcu_index))


def _finite_mean(values: Sequence[float], *, label: str) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain finite observations")
    return float(statistics.fmean(values))


def build_utilization_report(
    *,
    snapshots: tuple[UtilizationSnapshot, ...],
    nodes: tuple[str, ...],
    expected_hcus_per_node: int,
    phase: str,
    minimum_cluster_mean_percent: float,
    minimum_device_mean_percent: float,
    maximum_vram_percent: float,
) -> dict[str, object]:
    """Build a fail-closed temporal utilization acceptance report."""
    expected_devices = {
        (node, index)
        for node in nodes
        for index in range(expected_hcus_per_node)
    }
    device_samples: dict[tuple[str, int], list[float]] = {
        device: [] for device in expected_devices
    }
    all_hcu_values: list[float] = []
    all_vram_values: list[float] = []
    for snapshot in snapshots:
        roster = {
            (reading.node, reading.hcu_index) for reading in snapshot.readings
        }
        if roster != expected_devices or len(snapshot.readings) != len(
            expected_devices
        ):
            raise ValueError("snapshot HCU roster does not match the formal topology")
        for reading in snapshot.readings:
            device = (reading.node, reading.hcu_index)
            device_samples[device].append(reading.utilization_percent)
            all_hcu_values.append(reading.utilization_percent)
            all_vram_values.append(reading.vram_percent)

    failures: list[str] = []
    if len(snapshots) < _MINIMUM_TIME_SAMPLES:
        failures.append(
            f"time sample count {len(snapshots)} is below {_MINIMUM_TIME_SAMPLES}"
        )
    cluster_mean = _finite_mean(all_hcu_values, label="HCU utilization")
    maximum_vram = max(all_vram_values)
    device_means = {
        f"{node}:{index}": _finite_mean(
            device_samples[(node, index)],
            label=f"{node}:{index} utilization",
        )
        for node, index in sorted(expected_devices)
    }
    minimum_observed_device_mean = min(device_means.values())
    if cluster_mean < minimum_cluster_mean_percent:
        failures.append(
            f"cluster mean {cluster_mean:.2f}% is below "
            f"{minimum_cluster_mean_percent:.2f}%"
        )
    if minimum_observed_device_mean < minimum_device_mean_percent:
        failures.append(
            f"minimum device mean {minimum_observed_device_mean:.2f}% is below "
            f"{minimum_device_mean_percent:.2f}%"
        )
    if maximum_vram > maximum_vram_percent:
        failures.append(
            f"maximum VRAM {maximum_vram:.2f}% exceeds "
            f"{maximum_vram_percent:.2f}%"
        )
    return {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "phase": phase,
        "thresholds": {
            "minimum_cluster_mean_percent": minimum_cluster_mean_percent,
            "minimum_device_mean_percent": minimum_device_mean_percent,
            "maximum_vram_percent": maximum_vram_percent,
            "minimum_time_samples": _MINIMUM_TIME_SAMPLES,
        },
        "cluster": {
            "sample_count": len(snapshots),
            "device_count": len(expected_devices),
            "mean_hcu_percent": cluster_mean,
            "minimum_device_mean_percent": minimum_observed_device_mean,
            "maximum_vram_percent": maximum_vram,
        },
        "device_mean_hcu_percent": device_means,
        "failures": failures,
        "snapshots": [
            {
                "timestamp_unix_ns": snapshot.timestamp_unix_ns,
                "readings": [asdict(reading) for reading in snapshot.readings],
            }
            for snapshot in snapshots
        ],
    }


def _parse_nodes(value: str) -> tuple[str, ...]:
    nodes = tuple(item.strip() for item in value.split(","))
    if not nodes or len(set(nodes)) != len(nodes) or any(
        not _SAFE_NODE_PATTERN.fullmatch(node) for node in nodes
    ):
        raise ValueError("nodes must be a duplicate-free safe CSV")
    return nodes


def _hy_smi_command(
    *,
    node: str,
    port: int,
    known_hosts: Path,
    jump_host: str | None,
) -> list[str]:
    command = [
        "ssh",
    ]
    if jump_host is not None:
        command.extend(["-J", jump_host])
    command.extend([
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        f"root@{node}",
        "/opt/hyhal/bin/hy-smi",
    ])
    return command


def _read_node(
    *,
    node: str,
    port: int,
    known_hosts: Path,
    jump_host: str | None,
    expected_hcus: int,
) -> tuple[HcuReading, ...]:
    command = _hy_smi_command(
        node=node,
        port=port,
        known_hosts=known_hosts,
        jump_host=jump_host,
    )
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return parse_hy_smi(
        completed.stdout,
        node=node,
        expected_hcus=expected_hcus,
    )


def collect_snapshots(
    *,
    nodes: tuple[str, ...],
    port: int,
    known_hosts: Path,
    jump_host: str | None,
    expected_hcus: int,
    sample_count: int,
    interval_seconds: float,
) -> tuple[UtilizationSnapshot, ...]:
    """Collect concurrent per-node readings at fixed wall-clock intervals."""
    snapshots = []
    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        for sample_index in range(sample_count):
            started = time.monotonic()
            futures = [
                executor.submit(
                    _read_node,
                    node=node,
                    port=port,
                    known_hosts=known_hosts,
                    jump_host=jump_host,
                    expected_hcus=expected_hcus,
                )
                for node in nodes
            ]
            readings = tuple(
                reading for future in futures for reading in future.result()
            )
            snapshots.append(
                UtilizationSnapshot(
                    timestamp_unix_ns=time.time_ns(),
                    readings=readings,
                )
            )
            remaining = interval_seconds - (time.monotonic() - started)
            if sample_index + 1 < sample_count and remaining > 0:
                time.sleep(remaining)
    return tuple(snapshots)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--jump-host")
    parser.add_argument("--ssh-port", type=int, default=36000)
    parser.add_argument("--expected-hcus-per-node", type=int, default=8)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--phase", default="steady_optimizer_steps")
    parser.add_argument("--minimum-cluster-mean-percent", type=float, default=80.0)
    parser.add_argument("--minimum-device-mean-percent", type=float, default=70.0)
    parser.add_argument("--maximum-vram-percent", type=float, default=90.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    nodes = _parse_nodes(args.nodes)
    if args.samples < _MINIMUM_TIME_SAMPLES:
        raise ValueError(f"--samples must be at least {_MINIMUM_TIME_SAMPLES}")
    if args.expected_hcus_per_node <= 0 or args.interval_seconds <= 0:
        raise ValueError("HCU count and interval must be positive")
    if not args.known_hosts.is_file() or args.known_hosts.is_symlink():
        raise ValueError("--known-hosts must be a regular non-symlink file")
    if args.jump_host is not None and not _SAFE_NODE_PATTERN.fullmatch(
        args.jump_host
    ):
        raise ValueError("--jump-host is not a safe SSH host identity")
    if args.warmup_seconds > 0:
        time.sleep(args.warmup_seconds)
    snapshots = collect_snapshots(
        nodes=nodes,
        port=args.ssh_port,
        known_hosts=args.known_hosts,
        jump_host=args.jump_host,
        expected_hcus=args.expected_hcus_per_node,
        sample_count=args.samples,
        interval_seconds=args.interval_seconds,
    )
    report = build_utilization_report(
        snapshots=snapshots,
        nodes=nodes,
        expected_hcus_per_node=args.expected_hcus_per_node,
        phase=args.phase,
        minimum_cluster_mean_percent=args.minimum_cluster_mean_percent,
        minimum_device_mean_percent=args.minimum_device_mean_percent,
        maximum_vram_percent=args.maximum_vram_percent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["cluster"], sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
