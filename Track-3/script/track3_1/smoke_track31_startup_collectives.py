#!/usr/bin/env python3
"""Exercise Track 3.1 startup object collectives on two real ranks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.integrations.univtac.training_startup import (  # noqa: E402
    verify_track31_training_startup,
)

SmokeMode = Literal[
    "rank_mismatch",
    "rank0_audit_failure",
    "object_broadcast_success",
]

_MODES: tuple[SmokeMode, ...] = (
    "rank_mismatch",
    "rank0_audit_failure",
    "object_broadcast_success",
)
_BACKENDS = ("nccl", "gloo")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument(
        "--backend",
        choices=_BACKENDS,
        default="nccl",
        help="Use nccl for the vendor HCU smoke and gloo only for CPU tests.",
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument(
        "--init-method",
        default="env://",
        help="Keep env:// under torchrun; file:// is supported for CPU tests.",
    )
    parser.add_argument(
        "--missing-artifact-root",
        type=Path,
        help="A path that must not exist; required by rank0_audit_failure.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=45)
    return parser.parse_args(argv)


def _environment_integer(name: str) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        raise ValueError(f"torchrun environment is missing {name}")
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"torchrun environment has invalid {name}: {raw_value!r}"
        ) from error
    if value < 0:
        raise ValueError(f"torchrun environment has negative {name}: {value}")
    return value


def _initialize_process_group(
    *,
    backend: str,
    init_method: str,
    timeout_seconds: int,
) -> tuple[int, int, torch.device | None]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    rank = _environment_integer("RANK")
    local_rank = _environment_integer("LOCAL_RANK")
    world_size = _environment_integer("WORLD_SIZE")
    if world_size != 2:
        raise ValueError(
            "Track 3.1 startup collective smoke requires exactly two ranks"
        )
    device: torch.device | None = None
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "NCCL startup smoke requires the vendor CUDA-compatible API"
            )
        if local_rank >= torch.cuda.device_count():
            raise ValueError(f"LOCAL_RANK {local_rank} has no visible HCU")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    return rank, world_size, device


def _artifact_failure_config(
    *,
    rank: int,
    mode: SmokeMode,
    missing_root: Path,
) -> SimpleNamespace:
    configured_rank = 0 if mode == "rank_mismatch" else rank
    return SimpleNamespace(
        rank=configured_rank,
        wan22_pretrained_model_name_or_path=missing_root / "model",
        dataset_manifest_path=missing_root / "dataset_manifest.json",
        norm_stat_path=missing_root / "normalizer.json",
        conversion_report_path=missing_root / "conversion_report.json",
        lerobot_root=missing_root / "dataset",
        dataset_view_path=missing_root / "train_view.json",
        val_dataset_view_path=None,
        normalizer_source_view_path=missing_root / "normalizer_view.json",
        training_profile_id="multitask_pretrain_v1",
        run_role="development",
    )


def _gather_observations(
    local_observation: dict[str, object],
    *,
    world_size: int,
) -> list[dict[str, object]]:
    gathered: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_observation)
    observations: list[dict[str, object]] = []
    for expected_rank, item in enumerate(gathered):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"rank {expected_rank} returned an invalid observation")
        observation = {str(key): value for key, value in item.items()}
        if observation.get("rank") != expected_rank:
            raise RuntimeError(f"rank {expected_rank} returned a non-canonical rank")
        observations.append(observation)
    return observations


def _run_expected_failure(
    *,
    rank: int,
    world_size: int,
    device: torch.device | None,
    mode: SmokeMode,
    missing_root: Path,
) -> dict[str, object]:
    config = _artifact_failure_config(
        rank=rank,
        mode=mode,
        missing_root=missing_root,
    )
    try:
        verify_track31_training_startup(config, device=device)
    except Exception as error:
        local_observation: dict[str, object] = {
            "rank": rank,
            "outcome": "expected_failure",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    else:
        local_observation = {
            "rank": rank,
            "outcome": "unexpected_success",
            "error_type": None,
            "error": None,
        }
    observations = _gather_observations(
        local_observation,
        world_size=world_size,
    )
    if any(item.get("outcome") != "expected_failure" for item in observations):
        raise RuntimeError(f"{mode} did not fail on every rank")
    errors = [item.get("error") for item in observations]
    if len(set(errors)) != 1 or not isinstance(errors[0], str):
        raise RuntimeError(f"{mode} produced inconsistent rank errors")
    error_message = errors[0]
    if mode == "rank_mismatch":
        if "configured rank 0" not in error_message or "rank 1" not in error_message:
            raise RuntimeError("rank_mismatch did not exercise the rank contract")
        startup_collectives_completed = 1
    else:
        if (
            "rank-0 Track 3.1 artifact verification failed" not in error_message
            or str(missing_root) not in error_message
        ):
            raise RuntimeError("rank0_audit_failure did not broadcast the audit error")
        startup_collectives_completed = 2
    return {
        "all_ranks_agree": True,
        "startup_collectives_completed": startup_collectives_completed,
        "rank_observations": observations,
    }


def _pickle_payload() -> tuple[tuple[str, object], ...]:
    return (
        ("manifest_sha256", "a" * 64),
        ("normalizer_sha256", "b" * 64),
        ("conversion_report_sha256", "c" * 64),
        ("train_episode_count", 759),
        ("validation_episode_count", 40),
        ("normalizer_sample_count", 144484),
        ("action_q01", (0.0,) * 8),
        ("action_q99", (1.0,) * 8),
        ("train_view_identity", ("stage_a_final759_v1", "d" * 64)),
        ("validation_view_identity", (None, None)),
        ("latent_inventories", ("e" * 64, "f" * 64, 759, 2277, 4554)),
    )


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _run_object_broadcast(
    *,
    rank: int,
    world_size: int,
    device: torch.device | None,
) -> dict[str, object]:
    expected = _pickle_payload()
    objects: list[object] = [expected if rank == 0 else None]
    if device is None:
        dist.broadcast_object_list(objects, src=0)
    else:
        dist.broadcast_object_list(objects, src=0, device=device)
    received = objects[0]
    digest = _payload_sha256(received)
    local_observation: dict[str, object] = {
        "rank": rank,
        "outcome": "success" if received == expected else "payload_mismatch",
        "payload_sha256": digest,
    }
    observations = _gather_observations(
        local_observation,
        world_size=world_size,
    )
    expected_digest = _payload_sha256(expected)
    if any(
        item.get("outcome") != "success"
        or item.get("payload_sha256") != expected_digest
        for item in observations
    ):
        raise RuntimeError("object broadcast payload differed across ranks")
    return {
        "all_ranks_agree": True,
        "payload_sha256": expected_digest,
        "rank_observations": observations,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    parent = path.parent.resolve(strict=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
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
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _publish_report(
    *,
    report_path: Path,
    report: dict[str, object],
    rank: int,
    device: torch.device | None,
) -> None:
    message: dict[str, object] | None = None
    if rank == 0:
        try:
            _write_json_atomic(report_path, report)
            message = {"status": "ok"}
        except Exception as error:
            message = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    messages: list[object] = [message]
    if device is None:
        dist.broadcast_object_list(messages, src=0)
    else:
        dist.broadcast_object_list(messages, src=0, device=device)
    received = messages[0]
    if not isinstance(received, Mapping) or received.get("status") != "ok":
        raise RuntimeError(f"rank-0 report publication failed: {received!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 1 <= args.timeout_seconds <= 300:
        raise ValueError("--timeout-seconds must be in [1, 300]")
    if args.backend == "nccl" and args.init_method != "env://":
        raise ValueError("the vendor NCCL smoke requires --init-method=env://")
    if not (
        args.init_method == "env://"
        or (args.backend == "gloo" and str(args.init_method).startswith("file://"))
    ):
        raise ValueError("unsupported --init-method")
    mode = cast(SmokeMode, args.mode)
    if mode == "rank0_audit_failure" and args.missing_artifact_root is None:
        raise ValueError("rank0_audit_failure requires --missing-artifact-root")
    missing_root = Path(
        args.missing_artifact_root or ".n0_track31_unused_missing_artifacts"
    ).resolve(strict=False)
    rank, world_size, device = _initialize_process_group(
        backend=str(args.backend),
        init_method=str(args.init_method),
        timeout_seconds=int(args.timeout_seconds),
    )
    try:
        if mode == "rank0_audit_failure":
            existence: dict[str, object] = {
                "rank": rank,
                "exists": os.path.lexists(missing_root),
            }
            existence_observations = _gather_observations(
                existence,
                world_size=world_size,
            )
            if any(item.get("exists") is not False for item in existence_observations):
                raise ValueError(
                    "--missing-artifact-root must not exist on any rank: "
                    f"{missing_root}"
                )
        if mode == "object_broadcast_success":
            details = _run_object_broadcast(
                rank=rank,
                world_size=world_size,
                device=device,
            )
        else:
            details = _run_expected_failure(
                rank=rank,
                world_size=world_size,
                device=device,
                mode=mode,
                missing_root=missing_root,
            )
        report = {
            "schema_version": 1,
            "status": "PASS",
            "mode": mode,
            "backend": str(dist.get_backend()),
            "world_size": world_size,
            "timeout_seconds": int(args.timeout_seconds),
            **details,
        }
        _publish_report(
            report_path=Path(args.report_path),
            report=report,
            rank=rank,
            device=device,
        )
        dist.barrier()
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
