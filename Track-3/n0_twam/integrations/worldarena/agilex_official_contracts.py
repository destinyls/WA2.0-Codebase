# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Build immutable source, route, time, and normalizer contracts for AgileX."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from n0_twam.actions.qpos14 import CHANNEL_NAMES, CHANNEL_UNITS, GRIPPER_ENCODING
from n0_twam.embodiments import (
    AGILEX_ACTION_SCHEMA,
    AGILEX_EMBODIMENT_PROFILE_ID,
    AGILEX_RGB_KEYS,
    AGILEX_TACTILE_KEYS,
    AGILEX_WRENCH_KEYS,
    build_agilex_repo_route_manifest,
)

from .agilex_manifest import (
    AgileXRepoRoute,
    build_repo_route,
    canonical_sha256,
    sha256_file,
)
from .agilex_normalizer import fit_agilex_normalizer
from .agilex_official_reader import OfficialAgileXEpisode
from .agilex_official_schema import (
    ACTION_OFFSETS_PER_LATENT_ANCHOR,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    VISION_ONLY_REPO_ID,
    VISION_TACTILE_REPO_ID,
)


def write_new_json(path: Path, payload: object) -> str:
    destination = Path(path).expanduser().resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite AgileX artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode()
        + b"\n"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(destination)


def temporal_identity(repo_id: str) -> str:
    return canonical_sha256(
        {
            "schema_version": 1,
            "repo_id": repo_id,
            "source_fps": 30,
            "sampled_video_fps": 10,
            "video_sample_stride": 3,
            "vae_temporal_stride": 4,
            "action_offsets_per_anchor": list(ACTION_OFFSETS_PER_LATENT_ANCHOR),
            "policy": "same_row_commanded_qpos14_manifest_index_map_v1",
        }
    )


def build_official_routes() -> dict[str, AgileXRepoRoute]:
    output: dict[str, AgileXRepoRoute] = {}
    for repo_id, contact in (
        (VISION_ONLY_REPO_ID, False),
        (VISION_TACTILE_REPO_ID, True),
    ):
        output[repo_id] = build_repo_route(
            repo_id,
            {
                "embodiment": AGILEX_EMBODIMENT_PROFILE_ID,
                "formal": True,
                "action_schema": AGILEX_ACTION_SCHEMA,
                "action_label_source": "commanded",
                "rgb_keys": list(AGILEX_RGB_KEYS),
                "tactile_keys": list(AGILEX_TACTILE_KEYS if contact else ()),
                "wrench_keys": list(AGILEX_WRENCH_KEYS if contact else ()),
                "channel_names": list(CHANNEL_NAMES),
                "channel_units": list(CHANNEL_UNITS),
                "gripper_encoding": GRIPPER_ENCODING,
                "temporal_alignment_identity": temporal_identity(repo_id),
            },
        )
    return output


def build_route_manifest(routes: Mapping[str, AgileXRepoRoute]) -> dict[str, object]:
    compact = {
        task: {
            "embodiment": route.embodiment,
            "action_schema": route.action_schema,
            "rgb_keys": list(route.rgb_keys),
            "tactile_keys": list(route.tactile_keys),
            "wrench_keys": list(route.wrench_keys),
        }
        for task, route in routes.items()
    }
    return build_agilex_repo_route_manifest(
        compact,
        tactile_sensor_id_map={
            key: index for index, key in enumerate(AGILEX_TACTILE_KEYS)
        },
        wrench_sensor_id_map={
            key: index for index, key in enumerate(AGILEX_WRENCH_KEYS)
        },
    ).to_json_dict()


def build_temporal_contract(
    routes: Mapping[str, AgileXRepoRoute], *, route_manifest_sha256: str
) -> dict[str, object]:
    raw: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "repo_route_manifest_sha256": route_manifest_sha256,
        "per_repo_bindings": {
            task: {
                "action_offsets_per_anchor": list(ACTION_OFFSETS_PER_LATENT_ANCHOR),
                "repo_route_identity": route.route_identity,
                "temporal_alignment_identity": route.temporal_alignment_identity,
            }
            for task, route in sorted(routes.items())
        },
    }
    return {**raw, "contract_sha256": canonical_sha256(raw)}


def _source_records(
    raw_root: Path, episodes: Sequence[OfficialAgileXEpisode]
) -> list[dict[str, object]]:
    root = Path(raw_root).resolve(strict=True)
    paths = {root / "prompt.json"}
    for episode in episodes:
        paths.update(episode.consumed_files)
    records: list[dict[str, object]] = []
    for path in sorted(paths):
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return records


def build_source_manifest(
    *,
    raw_root: Path,
    episodes: Sequence[OfficialAgileXEpisode],
    routes: Mapping[str, AgileXRepoRoute],
) -> dict[str, object]:
    records = _source_records(raw_root, episodes)
    return {
        "schema_version": 1,
        "status": "frozen",
        "dataset_id": OFFICIAL_REPO_ID,
        "revision": OFFICIAL_REVISION,
        "canonical_records_sha256": canonical_sha256(records),
        "records": records,
        "repos": {task: route.to_json_dict() for task, route in sorted(routes.items())},
    }


def build_normalizer_payload(
    episodes: Iterable[OfficialAgileXEpisode],
    *,
    source_manifest_sha256: str,
    route_manifest_contract_sha256: str,
) -> dict[str, object]:
    cached = tuple(episodes)
    normalizer = fit_agilex_normalizer(
        action_batches=(episode.actions for episode in cached),
        valid_mask_batches=(episode.action_valid_mask for episode in cached),
        source_manifest_sha256=source_manifest_sha256,
        repo_route_manifest_sha256=route_manifest_contract_sha256,
    )
    return normalizer.to_json_dict()


__all__ = (
    "build_normalizer_payload",
    "build_official_routes",
    "build_route_manifest",
    "build_source_manifest",
    "build_temporal_contract",
    "temporal_identity",
    "write_new_json",
)
