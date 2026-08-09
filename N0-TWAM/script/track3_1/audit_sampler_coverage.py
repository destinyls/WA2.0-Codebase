#!/usr/bin/env python3
# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Dry-run the exact Stage A sampler across every requested training rank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

from n0_twam.dataset.bucket_sampler import BucketedDistributedBatchSampler


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _load_self_hashed_json(
    path: Path,
    *,
    digest_field: str,
    ensure_ascii: bool,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"identity artifact must be a JSON object: {path}")
    recorded = _require_sha256(payload.get(digest_field), label=digest_field)
    core = {key: value for key, value in payload.items() if key != digest_field}
    encoded = json.dumps(
        core,
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != recorded:
        raise ValueError(f"identity artifact digest mismatch: {path}")
    return payload


def resolve_exact_world_size(
    cli_value: int | None,
    environment_value: str | None,
) -> int:
    """Require the CLI diagnostic value to agree with the launcher contract."""

    if environment_value is None:
        raise ValueError("N0_TRACK31_EXPECTED_WORLD_SIZE is required")
    try:
        world_size = int(environment_value)
    except ValueError as exc:
        raise ValueError("N0_TRACK31_EXPECTED_WORLD_SIZE must be an integer") from exc
    if cli_value is not None and cli_value != world_size:
        raise ValueError("--world-size conflicts with the launcher world size")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return world_size


def resolve_exact_config_value(
    cli_value: int | None,
    configured_value: int,
    *,
    label: str,
) -> int:
    """Reject a coverage-only override that differs from the training config."""

    if cli_value is not None and cli_value != configured_value:
        raise ValueError(f"--{label} conflicts with the training config")
    return configured_value


def build_sampler_coverage_core(
    signatures: Sequence[Hashable],
    *,
    tasks: Sequence[str],
    sample_ids: Sequence[str | int],
    batch_size: int,
    world_size: int,
    seed: int,
) -> dict[str, Any]:
    """Prove full coverage and equal homogeneous work for one logical epoch."""

    if not signatures:
        raise ValueError("sampler coverage requires at least one sample")
    if batch_size <= 0 or world_size <= 0:
        raise ValueError("batch_size and world_size must be positive")
    if len(tasks) != len(signatures) or len(sample_ids) != len(signatures):
        raise ValueError("tasks/sample_ids must align with signatures")

    samplers = [
        BucketedDistributedBatchSampler(
            signatures,
            batch_size=batch_size,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
            coverage_mode="pad_global",
            tasks=tasks,
            sample_ids=sample_ids,
        )
        for rank in range(world_size)
    ]
    global_orders = {sampler.ordered_indices() for sampler in samplers}
    if len(global_orders) != 1:
        raise RuntimeError("ranks disagree on the global sampler order")
    global_order = global_orders.pop()
    rank_batches = [list(sampler) for sampler in samplers]
    batch_counts = {len(batches) for batches in rank_batches}
    if len(batch_counts) != 1:
        raise RuntimeError("ranks receive different batch counts")
    flattened = [
        index for batches in rank_batches for batch in batches for index in batch
    ]
    if tuple(flattened) != global_order:
        raise RuntimeError("rank slicing does not reconstruct the global order")
    expected_indices = set(range(len(signatures)))
    if set(flattened) != expected_indices:
        raise RuntimeError("rank assignment does not cover every dataset index")
    if any(
        len(batch) != batch_size or len({signatures[index] for index in batch}) != 1
        for batches in rank_batches
        for batch in batches
    ):
        raise RuntimeError("rank assignment contains an invalid shape bucket")

    quantum = batch_size * world_size
    expected_slots = (len(signatures) + quantum - 1) // quantum * quantum
    if len(flattened) != expected_slots:
        raise RuntimeError("sampler padding does not match the global quantum")
    unique_prefix = flattened[: len(signatures)]
    if (
        len(set(unique_prefix)) != len(signatures)
        or set(unique_prefix) != expected_indices
    ):
        raise RuntimeError("sampler must cover every sample before any padding")
    if tuple(flattened[len(signatures) :]) != samplers[0].padding_indices:
        raise RuntimeError("sampler padding is not an exact trailing suffix")
    counts = Counter(flattened)
    padding_count = expected_slots - len(signatures)
    if sum(count - 1 for count in counts.values()) != padding_count:
        raise RuntimeError("sampler duplicate count does not equal declared padding")
    per_task = Counter(tasks[index] for index in expected_indices)
    per_rank_batches = batch_counts.pop()
    return {
        "schema_version": 1,
        "status": "coverage_ready",
        "coverage_mode": "pad_global",
        "dataset_size": len(signatures),
        "unique_sample_count": len(expected_indices),
        "total_slot_count": expected_slots,
        "padding_count": padding_count,
        "world_size": world_size,
        "batch_size": batch_size,
        "global_batch_size": quantum,
        "per_rank_batch_count": per_rank_batches,
        "per_rank_slot_count": per_rank_batches * batch_size,
        "seed": seed,
        "ordered_index_sha256": _sha256_json(list(global_order)),
        "rank_assignment_sha256": _sha256_json(rank_batches),
        "sampler_input_sha256": _sha256_json(
            ([repr(value) for value in signatures], list(tasks), list(sample_ids))
        ),
        "per_task_unique_counts": dict(sorted(per_task.items())),
    }


def seal_report(core: Mapping[str, object]) -> dict[str, object]:
    """Attach a digest after all dataset/profile identities are present."""

    if "coverage_report_sha256" in core:
        raise ValueError("coverage core is already sealed")
    report = dict(core)
    report["coverage_report_sha256"] = _sha256_json(report)
    return report


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from n0_twam.configs.twam_track31_univtac_cfg import twam_track31_univtac_cfg
    from n0_twam.dataset import MultiLatentLeRobotDataset

    config = twam_track31_univtac_cfg
    if (
        config.training_profile_id != "multitask_pretrain_v1"
        or config.run_role != "development"
        or config.train_view_id != "stage_a_dev719_v1"
    ):
        raise ValueError("coverage audit requires Stage A development profile")
    world_size = resolve_exact_world_size(
        args.world_size,
        os.environ.get("N0_TRACK31_EXPECTED_WORLD_SIZE"),
    )
    batch_size = resolve_exact_config_value(
        args.batch_size,
        int(config.batch_size),
        label="batch-size",
    )
    seed = resolve_exact_config_value(
        args.seed,
        int(config.seed),
        label="seed",
    )
    dataset = MultiLatentLeRobotDataset(config=config)
    core = build_sampler_coverage_core(
        dataset.sample_signatures,
        tasks=dataset.sample_tasks,
        sample_ids=dataset.sample_ids,
        batch_size=batch_size,
        world_size=world_size,
        seed=seed,
    )
    manifest_sha256 = _require_sha256(
        config.source_manifest_sha256,
        label="source_manifest_sha256",
    )
    normalizer_sha256 = _require_sha256(
        config.normalizer_sha256,
        label="normalizer_sha256",
    )
    conversion = _load_self_hashed_json(
        Path(config.conversion_report_path),
        digest_field="conversion_report_sha256",
        ensure_ascii=False,
    )
    conversion_sha256 = _require_sha256(
        conversion["conversion_report_sha256"],
        label="conversion_report_sha256",
    )
    dataset_root = Path(config.dataset_path).resolve(strict=True)
    inventories = {
        kind: _load_self_hashed_json(
            dataset_root / filename,
            digest_field="inventory_sha256",
            ensure_ascii=True,
        )
        for kind, filename in {
            "video": "latent_video_inventory.json",
            "tactile": "latent_tactile_inventory.json",
        }.items()
    }
    for kind, inventory in inventories.items():
        if (
            inventory.get("status") != "ready"
            or inventory.get("kind") != kind
            or inventory.get("split") != "train759"
            or inventory.get("manifest_sha256") != manifest_sha256
            or inventory.get("conversion_report_sha256") != conversion_sha256
        ):
            raise ValueError(f"{kind} latent inventory identity mismatch")
    encoder_identities: set[str] = set()
    for kind, inventory in inventories.items():
        encoder_identity = inventory.get("encoder_source_identity")
        if not isinstance(encoder_identity, dict):
            raise ValueError(f"{kind} inventory has no encoder source identity")
        encoder_identities.add(
            _require_sha256(
                encoder_identity.get("identity_sha256"),
                label=f"{kind} encoder identity",
            )
        )
    if len(encoder_identities) != 1:
        raise ValueError("video/tactile encoder source identities differ")
    if dataset.dataset_view is None:
        raise ValueError("coverage dataset has no bound training view")
    core.update(
        {
            "training_profile_id": config.training_profile_id,
            "run_role": config.run_role,
            "train_view_id": config.train_view_id,
            "dataset_path": str(dataset_root),
            "source_manifest_sha256": manifest_sha256,
            "train_view_sha256": _require_sha256(
                dataset.dataset_view.view_sha256,
                label="train_view_sha256",
            ),
            "normalizer_sha256": normalizer_sha256,
            "conversion_report_sha256": conversion_sha256,
            "video_inventory_sha256": inventories["video"]["inventory_sha256"],
            "tactile_inventory_sha256": inventories["tactile"]["inventory_sha256"],
            "encoder_source_identity_sha256": encoder_identities.pop(),
            "sample_ids_sha256": _sha256_json(list(dataset.sample_ids)),
        }
    )
    report = seal_report(core)
    if args.output is not None:
        _write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
