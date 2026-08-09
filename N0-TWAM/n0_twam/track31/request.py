# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict, portable request schema for public Track 3.1 training."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REQUEST_SCHEMA_VERSION = 1
MULTITASK_PRETRAIN_PROFILE = "multitask_pretrain_v1"
TARGET_FINETUNE_PROFILE = "target_finetune_v1"
DEVELOPMENT_RUN_ROLE = "development"
FINAL_REFIT_RUN_ROLE = "final_refit"
# Local invocations append ``-`` plus twelve hex characters. Keeping the
# user-controlled prefix at 51 characters preserves the shared 64-character
# checkpoint invocation contract.
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,50}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ROOT_FIELDS = frozenset({"schema_version", "run_id", "runtime", "paths", "train"})
_RUNTIME_FIELDS = frozenset({"devices", "master_port"})
_PATH_FIELDS = frozenset(
    {
        "artifact_root",
        "lerobot_root",
        "base_model",
        "empty_embedding",
        "empty_embedding_sha256",
        "released_checkpoint",
        "released_transformer_sha256",
        "output_root",
        "resume_from",
        "init_from",
    }
)
_TRAIN_FIELDS = frozenset(
    {
        "profile",
        "run_role",
        "num_steps",
        "stop_after_step",
        "save_interval",
        "val_interval",
        "batch_size",
        "gradient_accumulation_steps",
        "max_latent_frames",
        "seed",
        "action_init_seed",
    }
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    missing = expected - set(payload)
    unexpected = set(payload) - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if unexpected:
            details.append("unexpected=" + ",".join(sorted(unexpected)))
        raise ValueError(f"invalid {label}: " + "; ".join(details))


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _path(
    value: object,
    *,
    base_dir: Path,
    label: str,
    optional: bool = False,
) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        suffix = " or null" if optional else ""
        raise ValueError(f"{label} must be a non-empty path string{suffix}")
    path = Path(value).expanduser()
    resolved = path if path.is_absolute() else (base_dir / path)
    return resolved.resolve(strict=False)


def _optional_sha256(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, label=label)


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class Track31RuntimeRequest:
    """Single-node device selection for the public launcher."""

    devices: tuple[int, ...]
    master_port: int


@dataclass(frozen=True)
class Track31PathRequest:
    """All filesystem and byte-identity inputs used by training."""

    artifact_root: Path
    lerobot_root: Path
    base_model: Path
    empty_embedding: Path
    empty_embedding_sha256: str
    released_checkpoint: Path | None
    released_transformer_sha256: str | None
    output_root: Path
    resume_from: Path | None
    init_from: Path | None


@dataclass(frozen=True)
class Track31TrainRecipe:
    """Scientific training recipe, independent of the cluster launcher."""

    profile: str
    run_role: str
    num_steps: int
    stop_after_step: int
    save_interval: int
    val_interval: int
    batch_size: int
    gradient_accumulation_steps: int
    max_latent_frames: int
    seed: int
    action_init_seed: int


@dataclass(frozen=True)
class Track31TrainRequest:
    """Normalized request consumed by the public training runner."""

    source_path: Path
    run_id: str
    runtime: Track31RuntimeRequest
    paths: Track31PathRequest
    train: Track31TrainRecipe


def _parse_runtime(payload: Mapping[str, object]) -> Track31RuntimeRequest:
    _require_exact_fields(payload, _RUNTIME_FIELDS, label="runtime request")
    devices = payload.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ValueError("runtime.devices must be a non-empty integer list")
    normalized = tuple(
        _nonnegative_int(value, label="runtime.devices item") for value in devices
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("runtime.devices must not contain duplicates")
    port = _positive_int(payload.get("master_port"), label="runtime.master_port")
    if port > 65535:
        raise ValueError("runtime.master_port must be at most 65535")
    return Track31RuntimeRequest(devices=normalized, master_port=port)


def _parse_paths(
    payload: Mapping[str, object], *, base_dir: Path
) -> Track31PathRequest:
    _require_exact_fields(payload, _PATH_FIELDS, label="paths request")
    released_checkpoint = _path(
        payload.get("released_checkpoint"),
        base_dir=base_dir,
        label="paths.released_checkpoint",
        optional=True,
    )
    resume_from = _path(
        payload.get("resume_from"),
        base_dir=base_dir,
        label="paths.resume_from",
        optional=True,
    )
    init_from = _path(
        payload.get("init_from"),
        base_dir=base_dir,
        label="paths.init_from",
        optional=True,
    )
    artifact_root = _path(
        payload.get("artifact_root"), base_dir=base_dir, label="paths.artifact_root"
    )
    lerobot_root = _path(
        payload.get("lerobot_root"), base_dir=base_dir, label="paths.lerobot_root"
    )
    base_model = _path(
        payload.get("base_model"), base_dir=base_dir, label="paths.base_model"
    )
    empty_embedding = _path(
        payload.get("empty_embedding"),
        base_dir=base_dir,
        label="paths.empty_embedding",
    )
    output_root = _path(
        payload.get("output_root"), base_dir=base_dir, label="paths.output_root"
    )
    assert artifact_root is not None
    assert lerobot_root is not None
    assert base_model is not None
    assert empty_embedding is not None
    assert output_root is not None
    return Track31PathRequest(
        artifact_root=artifact_root,
        lerobot_root=lerobot_root,
        base_model=base_model,
        empty_embedding=empty_embedding,
        empty_embedding_sha256=_sha256(
            payload.get("empty_embedding_sha256"),
            label="paths.empty_embedding_sha256",
        ),
        released_checkpoint=released_checkpoint,
        released_transformer_sha256=_optional_sha256(
            payload.get("released_transformer_sha256"),
            label="paths.released_transformer_sha256",
        ),
        output_root=output_root,
        resume_from=resume_from,
        init_from=init_from,
    )


def _parse_train(payload: Mapping[str, object]) -> Track31TrainRecipe:
    _require_exact_fields(payload, _TRAIN_FIELDS, label="train request")
    profile = payload.get("profile")
    if profile not in {MULTITASK_PRETRAIN_PROFILE, TARGET_FINETUNE_PROFILE}:
        raise ValueError("train.profile is not a supported Track 3.1 profile")
    run_role = payload.get("run_role")
    if run_role not in {DEVELOPMENT_RUN_ROLE, FINAL_REFIT_RUN_ROLE}:
        raise ValueError("train.run_role must be development or final_refit")
    num_steps = _positive_int(payload.get("num_steps"), label="train.num_steps")
    stop_after_step = _positive_int(
        payload.get("stop_after_step"), label="train.stop_after_step"
    )
    if stop_after_step > num_steps:
        raise ValueError("train.stop_after_step must not exceed train.num_steps")
    return Track31TrainRecipe(
        profile=str(profile),
        run_role=str(run_role),
        num_steps=num_steps,
        stop_after_step=stop_after_step,
        save_interval=_positive_int(
            payload.get("save_interval"), label="train.save_interval"
        ),
        val_interval=_positive_int(
            payload.get("val_interval"), label="train.val_interval"
        ),
        batch_size=_positive_int(payload.get("batch_size"), label="train.batch_size"),
        gradient_accumulation_steps=_positive_int(
            payload.get("gradient_accumulation_steps"),
            label="train.gradient_accumulation_steps",
        ),
        max_latent_frames=_positive_int(
            payload.get("max_latent_frames"), label="train.max_latent_frames"
        ),
        seed=_nonnegative_int(payload.get("seed"), label="train.seed"),
        action_init_seed=_nonnegative_int(
            payload.get("action_init_seed"), label="train.action_init_seed"
        ),
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_route(request: Track31TrainRequest) -> None:
    paths = request.paths
    profile = request.train.profile
    protected_inputs = {
        "paths.artifact_root": paths.artifact_root,
        "paths.lerobot_root": paths.lerobot_root,
        "paths.base_model": paths.base_model,
        "paths.empty_embedding": paths.empty_embedding,
    }
    if paths.released_checkpoint is not None:
        protected_inputs["paths.released_checkpoint"] = paths.released_checkpoint
    if paths.resume_from is not None:
        protected_inputs["paths.resume_from"] = paths.resume_from
    if paths.init_from is not None:
        protected_inputs["paths.init_from"] = paths.init_from
    for label, protected_path in protected_inputs.items():
        if _paths_overlap(paths.output_root, protected_path):
            raise ValueError(f"paths.output_root must be disjoint from {label}")
    if paths.resume_from is not None and paths.init_from is not None:
        raise ValueError("paths.resume_from and paths.init_from are mutually exclusive")
    if paths.resume_from is not None:
        return
    if profile == MULTITASK_PRETRAIN_PROFILE:
        if paths.init_from is not None:
            raise ValueError("fresh Stage A must not set paths.init_from")
        if paths.released_checkpoint is None:
            raise ValueError("fresh Stage A requires paths.released_checkpoint")
        if paths.released_transformer_sha256 is None:
            raise ValueError("fresh Stage A requires paths.released_transformer_sha256")
        return
    if paths.init_from is None:
        raise ValueError("fresh Stage B requires paths.init_from")


def load_track31_train_request(path: Path) -> Track31TrainRequest:
    """Load one strict request and resolve relative paths beside the JSON file."""

    source = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Track 3.1 training request: {source}") from error
    root = _mapping(payload, label="Track 3.1 training request")
    _require_exact_fields(root, _ROOT_FIELDS, label="Track 3.1 training request")
    if root.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported Track 3.1 training request schema")
    run_id = root.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must match [a-z0-9][a-z0-9._-]{0,50}")
    request = Track31TrainRequest(
        source_path=source,
        run_id=run_id,
        runtime=_parse_runtime(_mapping(root.get("runtime"), label="runtime")),
        paths=_parse_paths(
            _mapping(root.get("paths"), label="paths"), base_dir=source.parent
        ),
        train=_parse_train(_mapping(root.get("train"), label="train")),
    )
    _validate_route(request)
    return request


def track31_train_request_template() -> dict[str, object]:
    """Return the portable, development-safe public training template."""

    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "run_id": "track31-stage-a-dev",
        "runtime": {"devices": [0], "master_port": 29631},
        "paths": {
            "artifact_root": "./artifacts",
            "lerobot_root": "./lerobot",
            "base_model": "./models/n0-twam-base",
            "empty_embedding": "./models/n0-twam-base/empty_emb.pt",
            "empty_embedding_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
            "released_checkpoint": "./models/n0-twam-base",
            "released_transformer_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
            "output_root": "./outputs/track31-stage-a-dev",
            "resume_from": None,
            "init_from": None,
        },
        "train": {
            "profile": MULTITASK_PRETRAIN_PROFILE,
            "run_role": DEVELOPMENT_RUN_ROLE,
            "num_steps": 1500,
            "stop_after_step": 1500,
            "save_interval": 300,
            "val_interval": 100,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_latent_frames": 5,
            "seed": 20260801,
            "action_init_seed": 0,
        },
    }


__all__ = (
    "REQUEST_SCHEMA_VERSION",
    "Track31PathRequest",
    "Track31RuntimeRequest",
    "Track31TrainRecipe",
    "Track31TrainRequest",
    "load_track31_train_request",
    "track31_train_request_template",
)
