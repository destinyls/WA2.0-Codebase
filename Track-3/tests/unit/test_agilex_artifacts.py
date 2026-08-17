# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Regression tests for formal AgileX aggregate data artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from n0_twam.actions.qpos14 import CHANNEL_NAMES, CHANNEL_UNITS, GRIPPER_ENCODING
from n0_twam.embodiments import AGILEX_RGB_KEYS
from n0_twam.integrations.worldarena.agilex_artifacts import (
    build_agilex_conversion_receipt,
    build_agilex_latent_inventory,
    verify_agilex_conversion_receipt,
    verify_agilex_latent_inventory,
)
from n0_twam.integrations.worldarena.agilex_manifest import (
    AgileXRepoRoute,
    build_repo_route,
    canonical_sha256,
    sha256_file,
)

SOURCE_SHA = "1" * 64
ROUTE_SHA = "2" * 64
TEMPORAL_SHA = "3" * 64


def _route(
    repo_id: str = "touch", *, tactile: bool = True, formal: bool = True
) -> AgileXRepoRoute:
    return build_repo_route(
        repo_id,
        {
            "embodiment": "agilex_dual_qpos14_v1",
            "action_schema": (
                "qpos14_joint_absolute_v1" if formal else "measured_next_qpos_v1"
            ),
            "action_label_source": "commanded" if formal else "measured_next",
            "formal": formal,
            "rgb_keys": list(AGILEX_RGB_KEYS),
            "tactile_keys": (["observation.images.tactile_l"] if tactile else []),
            "wrench_keys": ["observation.wrench.left"] if tactile else [],
            "channel_names": list(CHANNEL_NAMES),
            "channel_units": list(CHANNEL_UNITS),
            "gripper_encoding": GRIPPER_ENCODING,
            "temporal_alignment_identity": "4" * 64,
        },
        allow_engineering=not formal,
    )


def _dataset(tmp_path: Path, routes: tuple[AgileXRepoRoute, ...]) -> Path:
    root = tmp_path / "dataset"
    for route in routes:
        repo = root / route.repo_id
        (repo / "meta").mkdir(parents=True)
        (repo / "data" / "chunk-000").mkdir(parents=True)
        (repo / "videos" / "chunk-000").mkdir(parents=True)
        (repo / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": 1}), encoding="utf-8"
        )
        (repo / "data" / "chunk-000" / "episode.parquet").write_bytes(b"table")
        (repo / "videos" / "chunk-000" / "episode.mp4").write_bytes(b"video")
        video = repo / "latents" / "chunk-000" / "observation.images.top"
        video.mkdir(parents=True)
        (video / "episode_000000_0_4.pth").write_bytes(b"video-latent")
        if route.tactile_keys:
            tactile = (
                repo
                / "latents_tactile"
                / "global"
                / "chunk-000"
                / route.tactile_keys[0]
            )
            tactile.mkdir(parents=True)
            (tactile / "episode_000000_0_4.pth").write_bytes(b"touch-latent")
    return root


def _conversion(root: Path, routes: tuple[AgileXRepoRoute, ...]) -> dict[str, object]:
    return build_agilex_conversion_receipt(
        dataset_root=root,
        routes=routes,
        source_manifest_sha256=SOURCE_SHA,
        repo_route_manifest_sha256=ROUTE_SHA,
        temporal_alignment_contract_sha256=TEMPORAL_SHA,
    )


def _write_receipt(tmp_path: Path, payload: object) -> tuple[Path, str]:
    path = tmp_path / "conversion.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, sha256_file(path)


def _reseal(payload: dict[str, object], identity_key: str) -> None:
    core = {key: value for key, value in payload.items() if key != identity_key}
    payload[identity_key] = canonical_sha256(core)


def test_build_and_verify_conversion_and_exact_latent_inventory(tmp_path: Path) -> None:
    routes = (_route(), _route("vision", tactile=False))
    root = _dataset(tmp_path, routes)
    conversion = _conversion(root, routes)
    verified_conversion = verify_agilex_conversion_receipt(
        conversion,
        dataset_root=root,
        routes=routes,
        source_manifest_sha256=SOURCE_SHA,
        repo_route_manifest_sha256=ROUTE_SHA,
        temporal_alignment_contract_sha256=TEMPORAL_SHA,
    )
    _, receipt_sha = _write_receipt(tmp_path, conversion)
    inventory = build_agilex_latent_inventory(
        dataset_root=root,
        routes=routes,
        conversion_receipt=conversion,
        conversion_receipt_sha256=receipt_sha,
    )
    verified_latents = verify_agilex_latent_inventory(
        inventory,
        dataset_root=root,
        routes=routes,
        conversion_receipt=conversion,
        conversion_receipt_sha256=receipt_sha,
    )

    assert verified_conversion.repo_ids == ("touch", "vision")
    assert verified_latents.record_count == 3
    assert [record["path"] for record in inventory["records"]] == [
        "touch/latents/chunk-000/observation.images.top/episode_000000_0_4.pth",
        "touch/latents_tactile/global/chunk-000/observation.images.tactile_l/episode_000000_0_4.pth",
        "vision/latents/chunk-000/observation.images.top/episode_000000_0_4.pth",
    ]


def test_conversion_rehashes_current_table_bytes(tmp_path: Path) -> None:
    routes = (_route(),)
    root = _dataset(tmp_path, routes)
    conversion = _conversion(root, routes)
    (root / "touch" / "data" / "chunk-000" / "episode.parquet").write_bytes(b"changed")

    with pytest.raises(ValueError, match="live data/routes"):
        verify_agilex_conversion_receipt(
            conversion,
            dataset_root=root,
            routes=routes,
            source_manifest_sha256=SOURCE_SHA,
            repo_route_manifest_sha256=ROUTE_SHA,
            temporal_alignment_contract_sha256=TEMPORAL_SHA,
        )


@pytest.mark.parametrize(
    "field,value,error",
    (
        ("schema_version", True, "conversion schema"),
        ("status", "ready", "not complete schema"),
    ),
)
def test_conversion_rejects_bool_schema_and_incomplete_status(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    routes = (_route(),)
    root = _dataset(tmp_path, routes)
    conversion = _conversion(root, routes)
    conversion[field] = value
    _reseal(conversion, "conversion_identity_sha256")

    with pytest.raises(ValueError, match=error):
        verify_agilex_conversion_receipt(
            conversion,
            dataset_root=root,
            routes=routes,
            source_manifest_sha256=SOURCE_SHA,
            repo_route_manifest_sha256=ROUTE_SHA,
            temporal_alignment_contract_sha256=TEMPORAL_SHA,
        )


def test_conversion_rejects_bool_size_duplicate_and_missing_repo(
    tmp_path: Path,
) -> None:
    route = _route()
    root = _dataset(tmp_path, (route,))
    conversion = _conversion(root, (route,))
    bad_size = copy.deepcopy(conversion)
    bad_size["repositories"][0]["table_inventory"][0]["size_bytes"] = True
    _reseal(bad_size, "conversion_identity_sha256")
    with pytest.raises(ValueError, match="size"):
        verify_agilex_conversion_receipt(
            bad_size,
            dataset_root=root,
            routes=(route,),
            source_manifest_sha256=SOURCE_SHA,
            repo_route_manifest_sha256=ROUTE_SHA,
            temporal_alignment_contract_sha256=TEMPORAL_SHA,
        )

    duplicate = copy.deepcopy(conversion)
    duplicate["repositories"].append(copy.deepcopy(duplicate["repositories"][0]))
    _reseal(duplicate, "conversion_identity_sha256")
    with pytest.raises(ValueError, match="duplicate repos"):
        verify_agilex_conversion_receipt(
            duplicate,
            dataset_root=root,
            routes=(route,),
            source_manifest_sha256=SOURCE_SHA,
            repo_route_manifest_sha256=ROUTE_SHA,
            temporal_alignment_contract_sha256=TEMPORAL_SHA,
        )

    with pytest.raises(ValueError, match="unavailable"):
        _conversion(root, (route, _route("missing", tactile=False)))


def test_formal_builder_rejects_engineering_action_route(tmp_path: Path) -> None:
    route = _route(formal=False)
    root = _dataset(tmp_path, (route,))
    with pytest.raises(ValueError, match="formal action routes"):
        _conversion(root, (route,))


def test_latent_verifier_rejects_changed_or_unexpected_payload(tmp_path: Path) -> None:
    routes = (_route(),)
    root = _dataset(tmp_path, routes)
    conversion = _conversion(root, routes)
    _, receipt_sha = _write_receipt(tmp_path, conversion)
    inventory = build_agilex_latent_inventory(
        dataset_root=root,
        routes=routes,
        conversion_receipt=conversion,
        conversion_receipt_sha256=receipt_sha,
    )
    payload = root / str(inventory["records"][0]["path"])
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="live payload bytes"):
        verify_agilex_latent_inventory(
            inventory,
            dataset_root=root,
            routes=routes,
            conversion_receipt=conversion,
            conversion_receipt_sha256=receipt_sha,
        )

    inventory = build_agilex_latent_inventory(
        dataset_root=root,
        routes=routes,
        conversion_receipt=conversion,
        conversion_receipt_sha256=receipt_sha,
    )
    (payload.parent / "unexpected.pth").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="live payload bytes"):
        verify_agilex_latent_inventory(
            inventory,
            dataset_root=root,
            routes=routes,
            conversion_receipt=conversion,
            conversion_receipt_sha256=receipt_sha,
        )


@pytest.mark.parametrize("mutation", ("escape", "duplicate", "bool_size"))
def test_latent_inventory_rejects_unsafe_duplicate_and_bool_records(
    tmp_path: Path, mutation: str
) -> None:
    routes = (_route(),)
    root = _dataset(tmp_path, routes)
    conversion = _conversion(root, routes)
    _, receipt_sha = _write_receipt(tmp_path, conversion)
    inventory = build_agilex_latent_inventory(
        dataset_root=root,
        routes=routes,
        conversion_receipt=conversion,
        conversion_receipt_sha256=receipt_sha,
    )
    if mutation == "escape":
        inventory["records"][0]["path"] = "../escaped.pth"
    elif mutation == "duplicate":
        inventory["records"].append(copy.deepcopy(inventory["records"][0]))
        inventory["record_count"] = len(inventory["records"])
    else:
        inventory["records"][0]["size_bytes"] = True
    _reseal(inventory, "inventory_sha256")

    with pytest.raises(ValueError):
        verify_agilex_latent_inventory(
            inventory,
            dataset_root=root,
            routes=routes,
            conversion_receipt=conversion,
            conversion_receipt_sha256=receipt_sha,
        )


def test_latent_tree_rejects_symlink(tmp_path: Path) -> None:
    routes = (_route(),)
    root = _dataset(tmp_path, routes)
    conversion = _conversion(root, routes)
    _, receipt_sha = _write_receipt(tmp_path, conversion)
    external = tmp_path / "external.pth"
    external.write_bytes(b"outside")
    (root / "touch" / "latents" / "escape.pth").symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        build_agilex_latent_inventory(
            dataset_root=root,
            routes=routes,
            conversion_receipt=conversion,
            conversion_receipt_sha256=receipt_sha,
        )


def test_dataset_layout_rejects_nested_duplicate_info(tmp_path: Path) -> None:
    routes = (_route(),)
    root = _dataset(tmp_path, routes)
    nested = root / "archive" / "touch" / "meta"
    nested.mkdir(parents=True)
    (nested / "info.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one canonical repo root"):
        _conversion(root, routes)


def test_dataset_layout_rejects_nested_repo_id_and_repo_symlink(tmp_path: Path) -> None:
    nested_route = _route("nested/touch")
    nested_root = _dataset(tmp_path / "nested-case", (nested_route,))
    with pytest.raises(ValueError, match="one dataset-root child"):
        _conversion(nested_root, (nested_route,))

    route = _route()
    root = _dataset(tmp_path / "symlink-case", (route,))
    alias = _route("alias", tactile=False)
    (root / "alias").symlink_to(root / "touch", target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink directory"):
        _conversion(root, (route, alias))


def test_builders_reject_zero_digest_and_empty_payload(tmp_path: Path) -> None:
    routes = (_route(),)
    root = _dataset(tmp_path, routes)
    with pytest.raises(ValueError, match="SHA-256"):
        build_agilex_conversion_receipt(
            dataset_root=root,
            routes=routes,
            source_manifest_sha256="0" * 64,
            repo_route_manifest_sha256=ROUTE_SHA,
            temporal_alignment_contract_sha256=TEMPORAL_SHA,
        )

    conversion = _conversion(root, routes)
    _, receipt_sha = _write_receipt(tmp_path, conversion)
    latent = next((root / "touch" / "latents").rglob("*.pth"))
    latent.write_bytes(b"")
    with pytest.raises(ValueError, match="regular file"):
        build_agilex_latent_inventory(
            dataset_root=root,
            routes=routes,
            conversion_receipt=conversion,
            conversion_receipt_sha256=receipt_sha,
        )
