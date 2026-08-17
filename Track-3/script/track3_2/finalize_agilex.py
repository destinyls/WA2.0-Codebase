#!/usr/bin/env python3
"""Publish aggregate AgileX conversion and latent artifacts after encoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from n0_twam.data.encoder_source_identity import load_encoder_source_identity
from n0_twam.configs.twam_track3_agilex_contracts import (
    load_repo_route_binding,
    load_temporal_binding,
)
from n0_twam.integrations.worldarena.agilex_artifacts import (
    build_agilex_conversion_receipt,
    build_agilex_latent_inventory,
    verify_agilex_conversion_receipt,
    verify_agilex_latent_inventory,
)
from n0_twam.integrations.worldarena.agilex_manifest import (
    load_agilex_manifest,
    sha256_file,
)
from n0_twam.integrations.worldarena.agilex_official_contracts import write_new_json
from n0_twam.integrations.worldarena.agilex_official_schema import (
    RGB_FILES,
    TACTILE_FILES,
    VISION_TACTILE_REPO_ID,
)
from script.track3_2.encode_agilex_latents import _validate_existing_payload


def _episode_rows(repo: Path) -> tuple[tuple[int, int], ...]:
    jsonl = repo / "meta" / "episodes.jsonl"
    if jsonl.is_file():
        values = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                values.append((int(payload["episode_index"]), int(payload["length"])))
        return tuple(values)
    import pandas as pd  # type: ignore[import-untyped]

    files = sorted((repo / "meta" / "episodes").rglob("*.parquet"))
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    return tuple(
        (int(row.episode_index), int(row.length))
        for row in frame.itertuples(index=False)
    )


def _require_complete_latents(
    dataset_root: Path,
    repo_ids: tuple[str, ...],
    *,
    encoder_identity_sha256: str,
) -> None:
    expected: set[Path] = set()
    for repo_id in repo_ids:
        repo = dataset_root / repo_id
        for episode_index, length in _episode_rows(repo):
            name = f"episode_{episode_index:06d}_0_{length}.pth"
            for key in RGB_FILES:
                path = repo / "latents" / "chunk-000" / key / name
                _validate_existing_payload(
                    path,
                    length=length,
                    encoder_identity_sha256=encoder_identity_sha256,
                    kind="video",
                    tactile_mode=None,
                )
                expected.add(path)
            if repo_id == VISION_TACTILE_REPO_ID:
                for mode in ("global", "local"):
                    for key in TACTILE_FILES:
                        path = (
                            repo / "latents_tactile" / mode / "chunk-000" / key / name
                        )
                        _validate_existing_payload(
                            path,
                            length=length,
                            encoder_identity_sha256=encoder_identity_sha256,
                            kind="tactile",
                            tactile_mode=mode,
                        )
                        expected.add(path)
    actual = {
        path.resolve(strict=True)
        for repo_id in repo_ids
        for root_name in ("latents", "latents_tactile")
        if (dataset_root / repo_id / root_name).is_dir()
        for path in (dataset_root / repo_id / root_name).rglob("*.pth")
    }
    resolved_expected = {path.resolve(strict=True) for path in expected}
    if actual != resolved_expected:
        missing = sorted(str(path) for path in expected if not path.is_file())[:5]
        raise ValueError(
            f"AgileX latent payload roster is incomplete: "
            f"expected={len(expected)} actual={len(actual)} missing={missing}"
        )


def finalize(args: argparse.Namespace) -> dict[str, object]:
    artifact_root = args.artifact_root.expanduser().resolve(strict=True)
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    source_path = artifact_root / "source_manifest.json"
    route_path = artifact_root / "repo_routes.json"
    temporal_path = artifact_root / "temporal_alignment.json"
    source_sha = sha256_file(source_path)
    route_file_sha = sha256_file(route_path)
    temporal_file_sha = sha256_file(temporal_path)
    route_binding = load_repo_route_binding(route_path)
    repo_ids = tuple(sorted(repo for repo, _ in route_binding.manifest.routes))
    temporal = load_temporal_binding(
        temporal_path,
        repo_names=repo_ids,
        repo_route_manifest_sha256=route_binding.manifest.contract_sha256,
    )
    source = load_agilex_manifest(
        source_path,
        expected_file_sha256=source_sha,
        selected_repo_ids=repo_ids,
    )
    encoder_identity = load_encoder_source_identity(
        artifact_root / "encoder_source_identity.json"
    )
    _require_complete_latents(
        dataset_root,
        repo_ids,
        encoder_identity_sha256=str(encoder_identity["identity_sha256"]),
    )
    conversion = build_agilex_conversion_receipt(
        dataset_root=dataset_root,
        routes=source.routes,
        source_manifest_sha256=source_sha,
        repo_route_manifest_sha256=route_binding.manifest.contract_sha256,
        temporal_alignment_contract_sha256=temporal.contract_sha256,
    )
    conversion_path = artifact_root / "conversion_receipt.json"
    conversion_file_sha = write_new_json(conversion_path, conversion)
    inventory = build_agilex_latent_inventory(
        dataset_root=dataset_root,
        routes=source.routes,
        conversion_receipt=conversion,
        conversion_receipt_sha256=conversion_file_sha,
    )
    inventory_path = artifact_root / "latent_inventory.json"
    inventory_file_sha = write_new_json(inventory_path, inventory)
    verify_agilex_conversion_receipt(
        conversion,
        dataset_root=dataset_root,
        routes=source.routes,
        source_manifest_sha256=source_sha,
        repo_route_manifest_sha256=route_binding.manifest.contract_sha256,
        temporal_alignment_contract_sha256=temporal.contract_sha256,
    )
    verify_agilex_latent_inventory(
        inventory,
        dataset_root=dataset_root,
        routes=source.routes,
        conversion_receipt=conversion,
        conversion_receipt_sha256=conversion_file_sha,
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "source_manifest_sha256": source_sha,
        "repo_route_manifest_file_sha256": route_file_sha,
        "repo_route_manifest_contract_sha256": route_binding.manifest.contract_sha256,
        "temporal_alignment_file_sha256": temporal_file_sha,
        "temporal_alignment_contract_sha256": temporal.contract_sha256,
        "conversion_receipt": str(conversion_path),
        "conversion_receipt_sha256": conversion_file_sha,
        "latent_inventory": str(inventory_path),
        "latent_inventory_sha256": inventory_file_sha,
        "latent_record_count": inventory["record_count"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser


def main() -> int:
    result = finalize(_parser().parse_args())
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
