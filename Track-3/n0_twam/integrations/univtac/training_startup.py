# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Rank-safe formal Track 3.1 artifact and latent startup verification."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist

from n0_twam.data.latent_inventory import build_encoder_source_identity

from .artifact_contracts import (
    VerifiedTrack31Artifacts,
    verify_track31_training_bundle,
)


def _optional_path(value: object) -> Path | None:
    return None if value is None else Path(str(value))


def _verify_on_rank_zero(config: object) -> VerifiedTrack31Artifacts:
    encoder_source_identity = build_encoder_source_identity(
        Path(str(getattr(config, "wan22_pretrained_model_name_or_path")))
    )
    normalizer_source_view_path = _optional_path(
        getattr(config, "normalizer_source_view_path", None)
    )
    parent_validation_view_path = None
    if (
        getattr(config, "training_profile_id", None) == "target_finetune_v1"
        and getattr(config, "run_role", None) == "development"
        and normalizer_source_view_path is not None
    ):
        parent_validation_view_path = (
            normalizer_source_view_path.parent / "internal_dev40_v1.json"
        )
    artifacts = verify_track31_training_bundle(
        manifest_path=Path(str(getattr(config, "dataset_manifest_path"))),
        normalizer_path=Path(str(getattr(config, "norm_stat_path"))),
        conversion_report_path=Path(str(getattr(config, "conversion_report_path"))),
        dataset_root=Path(str(getattr(config, "lerobot_root"))),
        train_view_path=_optional_path(getattr(config, "dataset_view_path", None)),
        validation_view_path=_optional_path(
            getattr(config, "val_dataset_view_path", None)
        ),
        normalizer_source_view_path=normalizer_source_view_path,
        parent_validation_view_path=parent_validation_view_path,
        encoder_source_identity=encoder_source_identity,
    )
    artifacts.latent_inventory_report()
    return artifacts


def _rank_contract_status(
    config: object, *, process_group_rank: int
) -> dict[str, object]:
    configured_rank = getattr(config, "rank", None)
    if isinstance(configured_rank, bool) or not isinstance(configured_rank, int):
        return {
            "status": "error",
            "rank": process_group_rank,
            "error": f"configured rank is not an integer: {configured_rank!r}",
        }
    if configured_rank != process_group_rank:
        return {
            "status": "error",
            "rank": process_group_rank,
            "error": (
                f"configured rank {configured_rank} disagrees with process group "
                f"rank {process_group_rank}"
            ),
        }
    return {"status": "ok", "rank": process_group_rank}


def _gather_rank_contract(
    config: object,
    *,
    process_group_rank: int,
    world_size: int,
) -> None:
    local_status = _rank_contract_status(
        config,
        process_group_rank=process_group_rank,
    )
    gathered: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_status)
    errors: list[str] = []
    for expected_rank, status in enumerate(gathered):
        if not isinstance(status, Mapping):
            errors.append(f"rank {expected_rank}: invalid rank-contract status")
            continue
        if status.get("rank") != expected_rank:
            errors.append(f"rank {expected_rank}: non-canonical status rank")
        if status.get("status") != "ok":
            errors.append(
                f"rank {expected_rank}: {status.get('error', 'unknown error')}"
            )
    if errors:
        raise ValueError(
            "distributed Track 3.1 rank contract failed: " + "; ".join(errors)
        )


def verify_track31_training_startup(
    config: object,
    *,
    device: torch.device | None,
) -> VerifiedTrack31Artifacts:
    """Agree on process-group ranks, then rehash once and broadcast the result.

    Distributed ranks first aggregate local rank-contract failures, so no peer
    can raise before the first collective.  The authoritative process-group rank
    selects the sole storage auditor; every rank then enters the result broadcast.
    """

    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _gather_rank_contract(
            config,
            process_group_rank=rank,
            world_size=world_size,
        )
    else:
        rank = 0
        local_status = _rank_contract_status(config, process_group_rank=rank)
        if local_status["status"] != "ok":
            raise ValueError(str(local_status["error"]))

    message: dict[str, object] | None = None
    if rank == 0:
        try:
            message = {
                "status": "ok",
                "artifacts": _verify_on_rank_zero(config),
            }
        except Exception as error:
            message = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }

    messages: list[object] = [message]
    if distributed:
        if device is None:
            dist.broadcast_object_list(messages, src=0)
        else:
            dist.broadcast_object_list(messages, src=0, device=device)
    received = messages[0]
    if not isinstance(received, Mapping):
        raise RuntimeError("rank-0 Track 3.1 verification broadcast is invalid")
    if received.get("status") != "ok":
        error_type = str(received.get("error_type", "ValidationError"))
        error_message = str(received.get("error", "unknown validation failure"))
        raise ValueError(
            f"rank-0 Track 3.1 artifact verification failed "
            f"({error_type}): {error_message}"
        )
    artifacts = received.get("artifacts")
    if not isinstance(artifacts, VerifiedTrack31Artifacts):
        raise RuntimeError("rank-0 Track 3.1 artifact identity is invalid")
    artifacts.latent_inventory_report()
    return artifacts


__all__ = ("verify_track31_training_startup",)
