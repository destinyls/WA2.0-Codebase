# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict portable request schema for Franka Track 3.2 post-training."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REQUEST_SCHEMA_VERSION = 2
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,50}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_NETWORK_INTERFACE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,14}$")
_ROOT_FIELDS = frozenset(
    {"schema_version", "run_id", "runtime", "paths", "artifacts", "train"}
)
_RUNTIME_FIELDS = frozenset(
    {
        "devices",
        "master_port",
        "accelerator_profile",
        "collective_network_interface",
    }
)
_PATH_FIELDS = frozenset(
    {
        "artifact_root",
        "lerobot_root",
        "base_model",
        "empty_embedding",
        "empty_embedding_sha256",
        "init_from",
        "init_transformer_sha256",
        "resume_from",
        "resume_checkpoint_identity_sha256",
        "output_root",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "prepare_receipt_sha256",
        "conversion_report_sha256",
        "latent_inventory_file_sha256",
        "train_view_sha256",
        "validation_view_sha256",
        "normalizer_sha256",
    }
)
_TRAIN_FIELDS = frozenset(
    {
        "run_role",
        "num_steps",
        "stop_after_step",
        "save_interval",
        "val_interval",
        "batch_size",
        "gradient_accumulation_steps",
        "max_latent_frames",
        "seed",
    }
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(
    payload: Mapping[str, object], expected: frozenset[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"invalid {label}: missing={sorted(expected - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected)}"
        )


def _positive(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _sha(value: object, *, label: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _path(
    value: object,
    *,
    base: Path,
    label: str,
    optional: bool = False,
) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve(
        strict=False
    )


@dataclass(frozen=True)
class Track32RuntimeRequest:
    devices: tuple[int, ...]
    master_port: int
    accelerator_profile: str
    collective_network_interface: str | None


@dataclass(frozen=True)
class Track32PathRequest:
    artifact_root: Path
    lerobot_root: Path
    base_model: Path
    empty_embedding: Path
    empty_embedding_sha256: str
    init_from: Path | None
    init_transformer_sha256: str | None
    resume_from: Path | None
    resume_checkpoint_identity_sha256: str | None
    output_root: Path


@dataclass(frozen=True)
class Track32ArtifactRequest:
    prepare_receipt_sha256: str
    conversion_report_sha256: str
    latent_inventory_file_sha256: str
    train_view_sha256: str
    validation_view_sha256: str | None
    normalizer_sha256: str


@dataclass(frozen=True)
class Track32TrainRecipe:
    run_role: str
    num_steps: int
    stop_after_step: int
    save_interval: int
    val_interval: int
    batch_size: int
    gradient_accumulation_steps: int
    max_latent_frames: int
    seed: int


@dataclass(frozen=True)
class Track32TrainRequest:
    source_path: Path
    run_id: str
    runtime: Track32RuntimeRequest
    paths: Track32PathRequest
    artifacts: Track32ArtifactRequest
    train: Track32TrainRecipe


def _parse_runtime(payload: Mapping[str, object]) -> Track32RuntimeRequest:
    _exact(payload, _RUNTIME_FIELDS, label="runtime request")
    raw_devices = payload.get("devices")
    if not isinstance(raw_devices, list) or not raw_devices:
        raise ValueError("runtime.devices must be a non-empty list")
    devices = tuple(
        _nonnegative(value, label="runtime device") for value in raw_devices
    )
    if len(devices) != len(set(devices)):
        raise ValueError("runtime.devices contains duplicates")
    port = _positive(payload.get("master_port"), label="runtime.master_port")
    if port > 65535:
        raise ValueError("runtime.master_port must be at most 65535")
    accelerator_profile = payload.get("accelerator_profile")
    if accelerator_profile not in {"portable", "hcu_performance"}:
        raise ValueError(
            "runtime.accelerator_profile must be portable or hcu_performance"
        )
    raw_interface = payload.get("collective_network_interface")
    if raw_interface is not None and (
        not isinstance(raw_interface, str)
        or not _NETWORK_INTERFACE_PATTERN.fullmatch(raw_interface)
    ):
        raise ValueError(
            "runtime.collective_network_interface must be null or one Linux "
            "interface name"
        )
    if accelerator_profile == "hcu_performance" and raw_interface is None:
        raise ValueError(
            "hcu_performance requires runtime.collective_network_interface"
        )
    if accelerator_profile == "portable" and raw_interface is not None:
        raise ValueError("portable requires runtime.collective_network_interface=null")
    return Track32RuntimeRequest(
        devices=devices,
        master_port=port,
        accelerator_profile=str(accelerator_profile),
        collective_network_interface=(
            None if raw_interface is None else str(raw_interface)
        ),
    )


def _parse_paths(payload: Mapping[str, object], *, base: Path) -> Track32PathRequest:
    _exact(payload, _PATH_FIELDS, label="paths request")
    required = {
        name: _path(payload.get(name), base=base, label=f"paths.{name}")
        for name in (
            "artifact_root",
            "lerobot_root",
            "base_model",
            "empty_embedding",
            "output_root",
        )
    }
    init_from = _path(
        payload.get("init_from"), base=base, label="paths.init_from", optional=True
    )
    resume_from = _path(
        payload.get("resume_from"),
        base=base,
        label="paths.resume_from",
        optional=True,
    )
    assert all(value is not None for value in required.values())
    return Track32PathRequest(
        artifact_root=required["artifact_root"],  # type: ignore[arg-type]
        lerobot_root=required["lerobot_root"],  # type: ignore[arg-type]
        base_model=required["base_model"],  # type: ignore[arg-type]
        empty_embedding=required["empty_embedding"],  # type: ignore[arg-type]
        empty_embedding_sha256=str(
            _sha(payload.get("empty_embedding_sha256"), label="empty embedding")
        ),
        init_from=init_from,
        init_transformer_sha256=_sha(
            payload.get("init_transformer_sha256"),
            label="init transformer",
            optional=True,
        ),
        resume_from=resume_from,
        resume_checkpoint_identity_sha256=_sha(
            payload.get("resume_checkpoint_identity_sha256"),
            label="resume checkpoint identity",
            optional=True,
        ),
        output_root=required["output_root"],  # type: ignore[arg-type]
    )


def _parse_artifacts(payload: Mapping[str, object]) -> Track32ArtifactRequest:
    _exact(payload, _ARTIFACT_FIELDS, label="artifacts request")
    validation = _sha(
        payload.get("validation_view_sha256"),
        label="validation view",
        optional=True,
    )
    return Track32ArtifactRequest(
        prepare_receipt_sha256=str(
            _sha(payload.get("prepare_receipt_sha256"), label="prepare receipt")
        ),
        conversion_report_sha256=str(
            _sha(payload.get("conversion_report_sha256"), label="conversion report")
        ),
        latent_inventory_file_sha256=str(
            _sha(
                payload.get("latent_inventory_file_sha256"),
                label="latent inventory file",
            )
        ),
        train_view_sha256=str(
            _sha(payload.get("train_view_sha256"), label="train view")
        ),
        validation_view_sha256=validation,
        normalizer_sha256=str(
            _sha(payload.get("normalizer_sha256"), label="normalizer")
        ),
    )


def _parse_train(payload: Mapping[str, object]) -> Track32TrainRecipe:
    _exact(payload, _TRAIN_FIELDS, label="train request")
    role = payload.get("run_role")
    if role not in {"development", "final_refit"}:
        raise ValueError("train.run_role must be development or final_refit")
    total = _positive(payload.get("num_steps"), label="train.num_steps")
    stop = _positive(payload.get("stop_after_step"), label="train.stop_after_step")
    if stop > total:
        raise ValueError("train.stop_after_step cannot exceed num_steps")
    return Track32TrainRecipe(
        run_role=str(role),
        num_steps=total,
        stop_after_step=stop,
        save_interval=_positive(payload.get("save_interval"), label="save_interval"),
        val_interval=_positive(payload.get("val_interval"), label="val_interval"),
        batch_size=_positive(payload.get("batch_size"), label="batch_size"),
        gradient_accumulation_steps=_positive(
            payload.get("gradient_accumulation_steps"), label="gradient accumulation"
        ),
        max_latent_frames=_positive(
            payload.get("max_latent_frames"), label="max_latent_frames"
        ),
        seed=_nonnegative(payload.get("seed"), label="seed"),
    )


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_route(request: Track32TrainRequest) -> None:
    paths = request.paths
    if (paths.init_from is None) == (paths.resume_from is None):
        raise ValueError("exactly one of paths.init_from/resume_from is required")
    if (paths.init_from is None) != (paths.init_transformer_sha256 is None):
        raise ValueError("fresh init requires exactly one transformer SHA256")
    if (paths.resume_from is None) != (paths.resume_checkpoint_identity_sha256 is None):
        raise ValueError("resume requires exactly one checkpoint identity SHA256")
    protected = [
        paths.artifact_root,
        paths.lerobot_root,
        paths.base_model,
        paths.empty_embedding,
        *(value for value in (paths.init_from, paths.resume_from) if value is not None),
    ]
    if any(_overlap(paths.output_root, value) for value in protected):
        raise ValueError("paths.output_root must be disjoint from every input")
    if request.train.run_role == "development":
        if request.artifacts.validation_view_sha256 is None:
            raise ValueError("development requires a validation view SHA256")
    elif request.artifacts.validation_view_sha256 is not None:
        raise ValueError("final_refit must not declare a validation view SHA256")


def load_track32_train_request(path: Path) -> Track32TrainRequest:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Track 3.2 training request: {source}") from error
    root = _mapping(payload, label="Track 3.2 training request")
    _exact(root, _ROOT_FIELDS, label="Track 3.2 training request")
    if root.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported Track 3.2 request schema")
    run_id = root.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must match [a-z0-9][a-z0-9._-]{0,50}")
    request = Track32TrainRequest(
        source_path=source,
        run_id=run_id,
        runtime=_parse_runtime(_mapping(root.get("runtime"), label="runtime")),
        paths=_parse_paths(
            _mapping(root.get("paths"), label="paths"), base=source.parent
        ),
        artifacts=_parse_artifacts(_mapping(root.get("artifacts"), label="artifacts")),
        train=_parse_train(_mapping(root.get("train"), label="train")),
    )
    _validate_route(request)
    return request


def track32_train_request_template() -> dict[str, object]:
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "run_id": "track32-franka-dev",
        "runtime": {
            "devices": list(range(8)),
            "master_port": 29632,
            "accelerator_profile": "portable",
            "collective_network_interface": None,
        },
        "paths": {
            "artifact_root": "./artifacts/franka",
            "lerobot_root": "./data/franka_lerobot",
            "base_model": "./models/n0-twam-base",
            "empty_embedding": "./models/n0-twam-base/empty_emb.pt",
            "empty_embedding_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
            "init_from": "./models/n0-twam-base",
            "init_transformer_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
            "resume_from": None,
            "resume_checkpoint_identity_sha256": None,
            "output_root": "./outputs/track32-franka-dev",
        },
        "artifacts": {
            "prepare_receipt_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
            "conversion_report_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
            "latent_inventory_file_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
            "train_view_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
            "validation_view_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
            "normalizer_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        },
        "train": {
            "run_role": "development",
            "num_steps": 1500,
            "stop_after_step": 1500,
            "save_interval": 300,
            "val_interval": 100,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_latent_frames": 5,
            "seed": 20260810,
        },
    }


__all__ = (
    "Track32TrainRequest",
    "load_track32_train_request",
    "track32_train_request_template",
)
