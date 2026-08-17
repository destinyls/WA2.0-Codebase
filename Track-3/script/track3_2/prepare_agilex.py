#!/usr/bin/env python3
"""Audit and convert the pinned official AgileX mixed-profile dataset."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from n0_twam.integrations.worldarena.agilex_convert import (
    write_agilex_streaming_lerobot_dataset,
)
from n0_twam.integrations.worldarena.agilex_manifest import sha256_file
from n0_twam.integrations.worldarena.agilex_official_contracts import (
    build_normalizer_payload,
    build_official_routes,
    build_route_manifest,
    build_source_manifest,
    build_temporal_contract,
    write_new_json,
)
from n0_twam.integrations.worldarena.agilex_official_reader import (
    discover_official_agilex_episodes,
)
from n0_twam.integrations.worldarena.agilex_official_schema import (
    OFFICIAL_EPISODE_COUNT,
    OFFICIAL_SOURCE_FPS,
    VISION_ONLY_REPO_ID,
    VISION_TACTILE_REPO_ID,
)


def prepare(args: argparse.Namespace) -> dict[str, object]:
    raw_root = args.raw_root.expanduser().resolve(strict=True)
    artifact_root = args.artifact_root.expanduser().resolve(strict=False)
    dataset_root = args.dataset_root.expanduser().resolve(strict=False)
    for path, label in (
        (artifact_root, "artifact-root"),
        (dataset_root, "dataset-root"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"AgileX {label} must be new: {path}")
    if (
        artifact_root == dataset_root
        or artifact_root in dataset_root.parents
        or dataset_root in artifact_root.parents
    ):
        raise ValueError("AgileX artifact and dataset roots must be disjoint")

    routes = build_official_routes()
    episodes = discover_official_agilex_episodes(raw_root=raw_root, routes=routes)
    if len(episodes) != OFFICIAL_EPISODE_COUNT:
        raise RuntimeError("AgileX preparation did not audit all 983 episodes")

    staging_token = uuid.uuid4().hex
    artifact_staging = artifact_root.with_name(
        f".{artifact_root.name}.prepare-{staging_token}.partial"
    )
    dataset_staging = dataset_root.with_name(
        f".{dataset_root.name}.prepare-{staging_token}.partial"
    )
    for path in (artifact_staging, dataset_staging):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"AgileX preparation staging path exists: {path}")
    artifact_staging.mkdir(parents=True)
    dataset_staging.mkdir(parents=True)

    route_payload = build_route_manifest(routes)
    route_path = artifact_staging / "repo_routes.json"
    route_file_sha = write_new_json(route_path, route_payload)
    route_contract_sha = str(route_payload["contract_sha256"])
    temporal_payload = build_temporal_contract(
        routes, route_manifest_sha256=route_contract_sha
    )
    temporal_path = artifact_staging / "temporal_alignment.json"
    temporal_file_sha = write_new_json(temporal_path, temporal_payload)

    source_payload = build_source_manifest(
        raw_root=raw_root, episodes=episodes, routes=routes
    )
    source_path = artifact_staging / "source_manifest.json"
    source_file_sha = write_new_json(source_path, source_payload)
    normalizer_payload = build_normalizer_payload(
        episodes,
        source_manifest_sha256=source_file_sha,
        route_manifest_contract_sha256=route_contract_sha,
    )
    normalizer_path = artifact_staging / "qpos14_normalizer.json"
    normalizer_file_sha = write_new_json(normalizer_path, normalizer_payload)

    reports: dict[str, object] = {}
    repo_ids = (VISION_ONLY_REPO_ID, VISION_TACTILE_REPO_ID)
    for repo_id in repo_ids:
        repo_episodes = tuple(
            episode for episode in episodes if episode.route.repo_id == repo_id
        )
        reports[repo_id] = write_agilex_streaming_lerobot_dataset(
            repo_episodes,
            output_root=dataset_staging / repo_id,
            repo_id=repo_id,
            fps=OFFICIAL_SOURCE_FPS,
            image_writer_threads=args.image_writer_threads,
        )
    conversion_path = artifact_staging / "per_repo_conversion.json"
    conversion_file_sha = write_new_json(
        conversion_path,
        {
            "schema_version": 1,
            "status": "complete",
            "profile": "mixed",
            "episode_count": len(episodes),
            "repositories": reports,
        },
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "profile": "mixed",
        "raw_root": str(raw_root),
        "dataset_root": str(dataset_root),
        "artifact_root": str(artifact_root),
        "episode_count": len(episodes),
        "repo_ids": list(repo_ids),
        "source_manifest": "source_manifest.json",
        "source_manifest_sha256": source_file_sha,
        "repo_route_manifest": "repo_routes.json",
        "repo_route_manifest_file_sha256": route_file_sha,
        "repo_route_manifest_contract_sha256": route_contract_sha,
        "temporal_alignment": "temporal_alignment.json",
        "temporal_alignment_file_sha256": temporal_file_sha,
        "temporal_alignment_contract_sha256": temporal_payload["contract_sha256"],
        "normalizer": "qpos14_normalizer.json",
        "normalizer_file_sha256": normalizer_file_sha,
        "normalizer_contract_sha256": normalizer_payload["contract_sha256"],
        "per_repo_conversion": "per_repo_conversion.json",
        "per_repo_conversion_sha256": conversion_file_sha,
    }
    receipt_path = artifact_staging / "prepare_receipt.json"
    write_new_json(receipt_path, receipt)
    os.replace(dataset_staging, dataset_root)
    os.replace(artifact_staging, artifact_root)
    return {
        **receipt,
        "prepare_receipt": str(artifact_root / "prepare_receipt.json"),
        "prepare_receipt_sha256": sha256_file(artifact_root / "prepare_receipt.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    return parser


def main() -> int:
    result = prepare(_parser().parse_args())
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
