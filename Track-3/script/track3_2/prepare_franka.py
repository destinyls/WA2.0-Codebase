#!/usr/bin/env python3
"""Audit, convert, split, and normalize the official no-tactile Franka data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from n0_twam.integrations.worldarena.franka_actions import end_pose8_to_ee10
from n0_twam.integrations.worldarena.franka_convert import (
    write_franka_lerobot_dataset,
)
from n0_twam.integrations.worldarena.franka_manifest import (
    OFFICIAL_TASKS,
    canonical_sha256,
    load_franka_inventory,
    sha256_file,
)
from n0_twam.integrations.worldarena.franka_source import (
    FrankaEpisode,
    audit_episode,
    read_end_pose8,
)
from n0_twam.integrations.worldarena.franka_views import (
    DEVELOPMENT_TRAIN_VIEW,
    FINAL_REFIT_VIEW,
    FrankaDatasetView,
    build_standard_franka_views,
)


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _audit_all(
    data_root: Path, inventory_path: Path
) -> tuple[object, tuple[FrankaEpisode, ...]]:
    inventory = load_franka_inventory(inventory_path)
    episodes = tuple(
        audit_episode(
            data_root=data_root,
            inventory=inventory,
            task=task,
            source_episode_id=episode_id,
        )
        for task in OFFICIAL_TASKS
        for episode_id in range(200)
    )
    if len(episodes) != 600:
        raise RuntimeError("Franka audit did not produce 600 episodes")
    return inventory, episodes


def _normalizer(
    episodes: Sequence[FrankaEpisode],
    view: FrankaDatasetView,
    *,
    source_records_sha256: str,
) -> dict[str, object]:
    by_identity = {
        (episode.task, episode.source_episode_id): episode for episode in episodes
    }
    action_chunks = []
    for entry in view.entries:
        episode = by_identity[(entry.task, entry.source_episode_id)]
        action_chunks.append(end_pose8_to_ee10(read_end_pose8(episode))[1:])
    action = np.concatenate(action_chunks, axis=0)
    q01 = np.quantile(action, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(action, 0.99, axis=0).astype(np.float32)
    if q01.shape != (10,) or q99.shape != (10,) or np.any(q99 < q01):
        raise RuntimeError("computed Franka normalizer is invalid")
    model_q01 = np.concatenate((q01, np.full(10, -1.0, dtype=np.float32)))
    model_q99 = np.concatenate((q99, np.full(10, 1.0, dtype=np.float32)))
    canonical = {
        "schema_version": 1,
        "normalizer_id": f"franka_ee20_{view.view_id}",
        "method": "q01q99",
        "source_action_schema": "franka_end_pose_base_wxyz8_v1",
        "derived_action_schema": "franka_ee10_rot6d_columns_v1",
        "model_action_schema": "ee20_absee",
        "active_action_channel_ids": list(range(10)),
        "source_records_sha256": source_records_sha256,
        "source_view_id": view.view_id,
        "source_view_sha256": view.view_sha256,
        "sample_count": int(action.shape[0]),
        "action_q01": model_q01.tolist(),
        "action_q99": model_q99.tolist(),
    }
    return {**canonical, "normalizer_sha256": canonical_sha256(canonical)}


def prepare(args: argparse.Namespace) -> dict[str, object]:
    data_root = args.data_root.expanduser().resolve(strict=True)
    artifact_root = args.artifact_root.expanduser().resolve(strict=False)
    lerobot_root = args.lerobot_root.expanduser().resolve(strict=False)
    if artifact_root.exists() or lerobot_root.exists():
        raise FileExistsError("artifact-root and lerobot-root must both be new")
    if (
        artifact_root == lerobot_root
        or artifact_root in lerobot_root.parents
        or lerobot_root in artifact_root.parents
    ):
        raise ValueError("artifact-root and lerobot-root must be disjoint")
    inventory, episodes = _audit_all(data_root, args.inventory)
    # Do not publish even an empty artifact tree until every immutable source
    # episode has passed the complete HDF5/JSON/video-sidecar audit.
    artifact_root.mkdir(parents=True)
    views = build_standard_franka_views()
    view_hashes: dict[str, str] = {}
    for view_id, view in views.items():
        view_path = artifact_root / "views" / f"{view_id}.json"
        _write_json(view_path, view.to_json_dict())
        view_hashes[view_id] = view.view_sha256

    normalizer_hashes: dict[str, str] = {}
    for view_id in (DEVELOPMENT_TRAIN_VIEW, FINAL_REFIT_VIEW):
        payload = _normalizer(
            episodes,
            views[view_id],
            source_records_sha256=inventory.records_sha256,
        )
        path = artifact_root / "normalizers" / f"{view_id}.json"
        _write_json(path, payload)
        normalizer_hashes[view_id] = str(payload["normalizer_sha256"])

    conversion = write_franka_lerobot_dataset(
        episodes,
        output_root=lerobot_root / "all600",
        repo_id=args.repo_id,
        image_writer_threads=args.image_writer_threads,
    )
    conversion_canonical = {
        **conversion,
        "official_repo_id": inventory.repo_id,
        "official_revision": inventory.revision,
        "official_inventory_sha256": inventory.manifest_sha256,
        "official_records_sha256": inventory.records_sha256,
        "view_sha256": view_hashes,
        "normalizer_sha256": normalizer_hashes,
    }
    conversion_report = {
        **conversion_canonical,
        "conversion_sha256": canonical_sha256(conversion_canonical),
    }
    conversion_path = artifact_root / "conversion_report.json"
    conversion_file_sha = _write_json(conversion_path, conversion_report)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "tactile_mode": "disabled",
        "artifact_root": str(artifact_root),
        "lerobot_root": str(lerobot_root),
        "official_records_sha256": inventory.records_sha256,
        "conversion_report": str(conversion_path),
        "conversion_report_sha256": conversion_file_sha,
        "conversion_sha256": conversion_report["conversion_sha256"],
        "view_sha256": view_hashes,
        "normalizer_sha256": normalizer_hashes,
    }
    receipt_path = artifact_root / "prepare_receipt.json"
    receipt_sha = _write_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": receipt_sha}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="worldarena_franka_fr3_all600")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    return parser


def main() -> int:
    result = prepare(_parser().parse_args())
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
