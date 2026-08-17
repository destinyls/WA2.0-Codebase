# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Public single-command publication of AgileX training artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from n0_twam.actions.qpos14 import CHANNEL_NAMES, CHANNEL_UNITS, GRIPPER_ENCODING
from n0_twam.cli import run_cli
from n0_twam.embodiments import AGILEX_RGB_KEYS, build_agilex_repo_route_manifest
from n0_twam.integrations.worldarena.agilex_manifest import (
    build_repo_route,
    canonical_sha256,
    sha256_file,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[list[str], dict[str, Path]]:
    dataset = tmp_path / "dataset"
    repo = dataset / "vision"
    (repo / "meta").mkdir(parents=True)
    (repo / "data" / "chunk-000").mkdir(parents=True)
    (repo / "videos" / "chunk-000").mkdir(parents=True)
    _write_json(repo / "meta" / "info.json", {"total_episodes": 1})
    (repo / "data" / "chunk-000" / "episode.parquet").write_bytes(b"table")
    (repo / "videos" / "chunk-000" / "episode.mp4").write_bytes(b"video")
    latent = repo / "latents" / "chunk-000" / AGILEX_RGB_KEYS[0]
    latent.mkdir(parents=True)
    (latent / "episode_000000_0_4.pth").write_bytes(b"vision-latent")

    temporal_identity = "a" * 64
    route_payload: dict[str, object] = {
        "embodiment": "agilex_dual_qpos14_v1",
        "action_schema": "qpos14_joint_absolute_v1",
        "action_label_source": "executed",
        "formal": True,
        "rgb_keys": list(AGILEX_RGB_KEYS),
        "tactile_keys": [],
        "wrench_keys": [],
        "channel_names": list(CHANNEL_NAMES),
        "channel_units": list(CHANNEL_UNITS),
        "gripper_encoding": GRIPPER_ENCODING,
        "temporal_alignment_identity": temporal_identity,
    }
    source_route = build_repo_route("vision", route_payload)
    record = {"path": "vision/episode.bin", "size": 7, "sha256": "b" * 64}
    source = _write_json(
        tmp_path / "contracts" / "source.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "dataset_id": "agilex-cli-fixture",
            "revision": "fixture-v1",
            "records": [record],
            "canonical_records_sha256": canonical_sha256([record]),
            "repos": {"vision": route_payload},
        },
    )
    simple_route = {
        "embodiment": route_payload["embodiment"],
        "action_schema": route_payload["action_schema"],
        "rgb_keys": route_payload["rgb_keys"],
        "tactile_keys": [],
        "wrench_keys": [],
    }
    route_contract = build_agilex_repo_route_manifest(
        {"vision": simple_route}
    ).to_json_dict()
    route = _write_json(tmp_path / "contracts" / "routes.json", route_contract)
    temporal_core: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "repo_route_manifest_sha256": route_contract["contract_sha256"],
        "per_repo_bindings": {
            "vision": {
                "action_offsets_per_anchor": [0, 1, 2],
                "repo_route_identity": source_route.route_identity,
                "temporal_alignment_identity": temporal_identity,
            }
        },
    }
    temporal = _write_json(
        tmp_path / "contracts" / "temporal.json",
        {**temporal_core, "contract_sha256": canonical_sha256(temporal_core)},
    )
    outputs = tmp_path / "outputs"
    conversion, inventory = outputs / "conversion.json", outputs / "latents.json"
    command = [
        "track32",
        "agilex-build-artifacts",
        "--dataset-root",
        str(dataset),
        "--source-manifest",
        str(source),
        "--source-manifest-sha256",
        sha256_file(source),
        "--repo-route-manifest",
        str(route),
        "--repo-route-manifest-file-sha256",
        sha256_file(route),
        "--temporal-alignment",
        str(temporal),
        "--temporal-alignment-file-sha256",
        sha256_file(temporal),
        "--conversion-output",
        str(conversion),
        "--latent-output",
        str(inventory),
    ]
    return command, {
        "dataset": dataset,
        "source": source,
        "route": route,
        "temporal": temporal,
        "conversion": conversion,
        "inventory": inventory,
    }


def test_agilex_build_artifacts_publishes_hash_bound_pair(tmp_path: Path) -> None:
    command, paths = _fixture(tmp_path)

    result = run_cli(command)

    conversion = json.loads(paths["conversion"].read_text(encoding="utf-8"))
    inventory = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["conversion_file_sha256"] == sha256_file(paths["conversion"])
    assert (
        result["conversion_identity_sha256"] == conversion["conversion_identity_sha256"]
    )
    assert result["latent_file_sha256"] == sha256_file(paths["inventory"])
    assert result["latent_inventory_sha256"] == inventory["inventory_sha256"]
    assert result["record_count"] == inventory["record_count"] == 1
    assert inventory["conversion_receipt_sha256"] == sha256_file(paths["conversion"])
    assert not paths["conversion"].is_symlink()
    with pytest.raises(FileExistsError, match="must be new"):
        run_cli(command)


def test_agilex_build_artifacts_builds_both_before_publication(tmp_path: Path) -> None:
    command, paths = _fixture(tmp_path)
    next((paths["dataset"] / "vision" / "latents").rglob("*.pth")).unlink()

    with pytest.raises(ValueError, match="contains no .pth"):
        run_cli(command)

    assert not paths["conversion"].exists()
    assert not paths["inventory"].exists()


def test_agilex_build_artifacts_rejects_hash_and_disjoint_output(
    tmp_path: Path,
) -> None:
    command, paths = _fixture(tmp_path)
    bad_hash = list(command)
    index = bad_hash.index("--repo-route-manifest-file-sha256") + 1
    bad_hash[index] = "c" * 64
    with pytest.raises(ValueError, match="repo-route manifest SHA256 mismatch"):
        run_cli(bad_hash)

    disjoint = list(command)
    index = disjoint.index("--conversion-output") + 1
    disjoint[index] = str(paths["dataset"] / "conversion.json")
    with pytest.raises(ValueError, match="disjoint"):
        run_cli(disjoint)

    nested = list(command)
    parent = tmp_path / "nested-output"
    nested[nested.index("--conversion-output") + 1] = str(parent)
    nested[nested.index("--latent-output") + 1] = str(parent / "latents.json")
    with pytest.raises(ValueError, match="disjoint"):
        run_cli(nested)


def test_agilex_build_artifacts_preserves_published_conversion_on_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command, paths = _fixture(tmp_path)
    real_link = os.link
    calls = 0

    def race(source: Path, destination: Path, *, follow_symlinks: bool = True) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileExistsError(destination)
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr("n0_twam.cli.os.link", race)
    with pytest.raises(FileExistsError):
        run_cli(command)

    assert paths["conversion"].is_file()
    assert not paths["inventory"].exists()
    assert not list(paths["conversion"].parent.glob(".*.tmp"))
