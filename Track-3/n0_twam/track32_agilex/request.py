# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Portable, hash-complete request schema for AgileX post-training."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .request_template import build_agilex_train_request_template

REQUEST_SCHEMA_VERSION = 1
PROFILES = frozenset({"vision_tactile", "mixed", "vision_only"})
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,50}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INTERFACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,14}$")
_ROOT_FIELDS = frozenset(
    {"schema_version", "run_id", "profile", "runtime", "paths", "train"}
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
        "source_root",
        "source_manifest",
        "source_manifest_sha256",
        "dataset_root",
        "artifact_root",
        "conversion_receipt",
        "conversion_receipt_sha256",
        "latent_inventory",
        "latent_inventory_sha256",
        "repo_route_manifest",
        "repo_route_manifest_sha256",
        "temporal_alignment",
        "temporal_alignment_sha256",
        "normalizer",
        "normalizer_sha256",
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


def _object(value: object, *, label: str) -> Mapping[str, object]:
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


def _integer(value: object, *, label: str, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _digest(value: object, *, label: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _path(
    value: object, *, base: Path, label: str, optional: bool = False
) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    candidate = Path(value).expanduser()
    lexical = candidate if candidate.is_absolute() else base / candidate
    if lexical.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    return lexical.resolve(strict=False)


@dataclass(frozen=True)
class AgileXRuntimeRequest:
    devices: tuple[int, ...]
    master_port: int
    accelerator_profile: str
    collective_network_interface: str | None


@dataclass(frozen=True)
class AgileXPathRequest:
    source_root: Path
    source_manifest: Path
    source_manifest_sha256: str
    dataset_root: Path
    artifact_root: Path
    conversion_receipt: Path
    conversion_receipt_sha256: str
    latent_inventory: Path
    latent_inventory_sha256: str
    repo_route_manifest: Path
    repo_route_manifest_sha256: str
    temporal_alignment: Path
    temporal_alignment_sha256: str
    normalizer: Path
    normalizer_sha256: str
    base_model: Path
    empty_embedding: Path
    empty_embedding_sha256: str
    init_from: Path | None
    init_transformer_sha256: str | None
    resume_from: Path | None
    resume_checkpoint_identity_sha256: str | None
    output_root: Path


@dataclass(frozen=True)
class AgileXTrainRecipe:
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
class AgileXTrainRequest:
    source_path: Path
    source_sha256: str
    run_id: str
    profile: str
    runtime: AgileXRuntimeRequest
    paths: AgileXPathRequest
    train: AgileXTrainRecipe


def _runtime(payload: Mapping[str, object]) -> AgileXRuntimeRequest:
    _exact(payload, _RUNTIME_FIELDS, label="runtime")
    raw_devices = payload.get("devices")
    if not isinstance(raw_devices, list) or not raw_devices:
        raise ValueError("runtime.devices must be a non-empty list")
    devices = tuple(
        _integer(item, label="runtime device", positive=False) for item in raw_devices
    )
    if len(devices) != len(set(devices)):
        raise ValueError("runtime.devices contains duplicates")
    port = _integer(payload.get("master_port"), label="runtime.master_port")
    if port > 65535:
        raise ValueError("runtime.master_port must be at most 65535")
    accelerator = payload.get("accelerator_profile")
    if accelerator not in {"portable", "hcu_performance"}:
        raise ValueError("accelerator_profile must be portable or hcu_performance")
    interface = payload.get("collective_network_interface")
    if interface is not None and (
        not isinstance(interface, str) or not _INTERFACE.fullmatch(interface)
    ):
        raise ValueError("collective_network_interface is invalid")
    if (accelerator == "hcu_performance") != (interface is not None):
        raise ValueError(
            "hcu_performance requires one interface and portable requires null"
        )
    return AgileXRuntimeRequest(devices, port, str(accelerator), interface)


def _paths(payload: Mapping[str, object], *, base: Path) -> AgileXPathRequest:
    _exact(payload, _PATH_FIELDS, label="paths")
    required_paths = {
        name: _path(payload.get(name), base=base, label=f"paths.{name}")
        for name in (
            "source_root",
            "source_manifest",
            "dataset_root",
            "artifact_root",
            "conversion_receipt",
            "latent_inventory",
            "repo_route_manifest",
            "temporal_alignment",
            "normalizer",
            "base_model",
            "empty_embedding",
            "output_root",
        )
    }
    digests = {
        name: _digest(payload.get(name), label=f"paths.{name}")
        for name in (
            "source_manifest_sha256",
            "conversion_receipt_sha256",
            "latent_inventory_sha256",
            "repo_route_manifest_sha256",
            "temporal_alignment_sha256",
            "normalizer_sha256",
            "empty_embedding_sha256",
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
    init_sha = _digest(
        payload.get("init_transformer_sha256"),
        label="paths.init_transformer_sha256",
        optional=True,
    )
    resume_sha = _digest(
        payload.get("resume_checkpoint_identity_sha256"),
        label="paths.resume_checkpoint_identity_sha256",
        optional=True,
    )
    if (init_from is None) == (resume_from is None):
        raise ValueError("exactly one of init_from or resume_from is required")
    if (init_from is None) != (init_sha is None):
        raise ValueError("init_from and init_transformer_sha256 must be paired")
    if (resume_from is None) != (resume_sha is None):
        raise ValueError("resume_from and resume identity must be paired")
    output = required_paths["output_root"]
    assert output is not None
    inputs = (
        *(path for key, path in required_paths.items() if key != "output_root"),
        *(path for path in (init_from, resume_from) if path is not None),
    )
    for source in inputs:
        assert source is not None
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("paths.output_root must be disjoint from every input")
    return AgileXPathRequest(
        **required_paths,
        **digests,
        init_from=init_from,
        init_transformer_sha256=init_sha,
        resume_from=resume_from,
        resume_checkpoint_identity_sha256=resume_sha,
    )


def _train(payload: Mapping[str, object]) -> AgileXTrainRecipe:
    _exact(payload, _TRAIN_FIELDS, label="train")
    role = payload.get("run_role")
    if role not in {"development", "final_refit"}:
        raise ValueError("train.run_role must be development or final_refit")
    values = {
        field: _integer(payload.get(field), label=f"train.{field}")
        for field in (
            "num_steps",
            "stop_after_step",
            "save_interval",
            "val_interval",
            "batch_size",
            "gradient_accumulation_steps",
            "max_latent_frames",
        )
    }
    seed = _integer(payload.get("seed"), label="train.seed", positive=False)
    if values["stop_after_step"] > values["num_steps"]:
        raise ValueError("stop_after_step cannot exceed num_steps")
    return AgileXTrainRecipe(run_role=str(role), seed=seed, **values)


def _stable_request_bytes(path: Path) -> tuple[Path, bytes, str]:
    """Read one regular request file while rejecting replacement or drift."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("AgileX request must be a regular non-symlink file")
    source = candidate.resolve(strict=True)
    before = source.stat()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened_before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = source.stat()

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
        )

    if (
        not stat.S_ISREG(opened_before.st_mode)
        or identity(before) != identity(opened_before)
        or identity(opened_before) != identity(opened_after)
        or identity(opened_after) != identity(after)
        or len(raw) != opened_before.st_size
    ):
        raise ValueError("AgileX request changed while it was being read")
    return source, raw, hashlib.sha256(raw).hexdigest()


def require_agilex_request_unchanged(request: AgileXTrainRequest) -> None:
    """Require the on-disk request to remain the bytes loaded by the parent."""

    source, _raw, digest = _stable_request_bytes(request.source_path)
    if source != request.source_path or digest != request.source_sha256:
        raise ValueError("AgileX request changed after it was loaded")


def load_agilex_train_request(path: Path) -> AgileXTrainRequest:
    source, raw, source_sha256 = _stable_request_bytes(path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("AgileX request must contain valid UTF-8 JSON") from error
    root = _object(payload, label="request")
    _exact(root, _ROOT_FIELDS, label="request")
    if root.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported AgileX request schema_version")
    run_id = root.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id has an invalid portable format")
    profile = root.get("profile")
    if profile not in PROFILES:
        raise ValueError("profile must be vision_tactile, mixed, or vision_only")
    runtime = _runtime(_object(root.get("runtime"), label="runtime"))
    paths = _paths(_object(root.get("paths"), label="paths"), base=source.parent)
    train = _train(_object(root.get("train"), label="train"))
    if profile == "mixed" and train.batch_size != 1:
        raise ValueError(
            "mixed AgileX training requires batch_size=1 until per-sample "
            "contact conditioning is supported"
        )
    return AgileXTrainRequest(
        source_path=source,
        source_sha256=source_sha256,
        run_id=run_id,
        profile=str(profile),
        runtime=runtime,
        paths=paths,
        train=train,
    )


def agilex_train_request_template() -> dict[str, object]:
    return build_agilex_train_request_template(REQUEST_SCHEMA_VERSION)


__all__ = (
    "AgileXRuntimeRequest",
    "AgileXPathRequest",
    "AgileXTrainRecipe",
    "AgileXTrainRequest",
    "PROFILES",
    "agilex_train_request_template",
    "load_agilex_train_request",
    "require_agilex_request_unchanged",
)
