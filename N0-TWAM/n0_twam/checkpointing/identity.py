# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed identity contract for an N0 transformer safetensors file."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Mapping

from safetensors import safe_open

TRANSFORMER_IDENTITY_SCHEMA_VERSION = 1
TRANSFORMER_WEIGHTS_FILENAME = "diffusion_pytorch_model.safetensors"
TRANSFORMER_ACTION_INNER_DIM = 3072
ACTION_TENSOR_KEYS = (
    "action_embedder.weight",
    "action_embedder.bias",
    "action_proj_out.weight",
    "action_proj_out.bias",
)
TRANSFORMER_SENTINEL_KEYS = (
    "condition_embedder.text_embedder.linear_1.weight",
    "mot.experts.action.in_proj.weight",
    "mot.experts.tactile.in_proj.weight",
)
_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "file_name",
        "size_bytes",
        "sha256",
        "tensor_count",
        "action_dim",
        "action_shapes",
        "required_sentinel_keys",
    }
)


def _required_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def validate_sha256(value: object, *, label: str) -> str:
    """Return a canonical SHA256 or reject non-lowercase/ambiguous values."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be exactly 64 lowercase hex characters")
    return value


def _regular_file_stat(path: Path) -> os.stat_result:
    try:
        result = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"transformer weights do not exist: {path}") from None
    if stat.S_ISLNK(result.st_mode) or not stat.S_ISREG(result.st_mode):
        raise ValueError(f"transformer weights must be a regular non-symlink: {path}")
    if result.st_size <= 0:
        raise ValueError(f"transformer weights are empty: {path}")
    return result


def _stat_fingerprint(result: os.stat_result) -> tuple[int, ...]:
    return (
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_mode),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def _descriptor_path(descriptor: int) -> str:
    for root in (Path("/proc/self/fd"), Path("/dev/fd")):
        candidate = root / str(descriptor)
        if candidate.exists():
            return str(candidate)
    raise RuntimeError("platform has no descriptor path for atomic safetensors audit")


def _sha256_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb", closefd=False) as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_action_shapes(
    action_shapes: Mapping[str, tuple[int, ...]],
    *,
    action_dim: int,
) -> None:
    if action_dim <= 0:
        raise ValueError("checkpoint action_dim must be positive")
    expected_shapes = {
        "action_embedder.weight": (TRANSFORMER_ACTION_INNER_DIM, action_dim),
        "action_embedder.bias": (TRANSFORMER_ACTION_INNER_DIM,),
        "action_proj_out.weight": (action_dim, TRANSFORMER_ACTION_INNER_DIM),
        "action_proj_out.bias": (action_dim,),
    }
    mismatches = [
        f"{key}:{action_shapes[key]}!={expected}"
        for key, expected in expected_shapes.items()
        if action_shapes[key] != expected
    ]
    if mismatches:
        raise ValueError(
            f"transformer action projection violates action_dim={action_dim}: "
            + ", ".join(mismatches)
        )


def validate_recorded_transformer_identity(
    payload: object,
    *,
    expected_action_dim: int,
) -> dict[str, object]:
    """Validate a sidecar identity without trusting it as proof of file bytes."""

    if not isinstance(payload, dict) or set(payload) != _IDENTITY_KEYS:
        raise ValueError("transformer identity has an invalid field set")
    schema_version = _required_integer(
        payload.get("schema_version"),
        label="transformer identity schema_version",
    )
    if schema_version != TRANSFORMER_IDENTITY_SCHEMA_VERSION:
        raise ValueError("unsupported transformer identity schema")
    if payload.get("file_name") != TRANSFORMER_WEIGHTS_FILENAME:
        raise ValueError("transformer identity has an unexpected file name")
    size_bytes = _required_integer(
        payload.get("size_bytes"), label="transformer identity size_bytes"
    )
    tensor_count = _required_integer(
        payload.get("tensor_count"), label="transformer identity tensor_count"
    )
    action_dim = _required_integer(
        payload.get("action_dim"), label="transformer identity action_dim"
    )
    if size_bytes <= 0 or tensor_count < len(ACTION_TENSOR_KEYS) + len(
        TRANSFORMER_SENTINEL_KEYS
    ):
        raise ValueError("transformer identity size/tensor count is invalid")
    if action_dim != int(expected_action_dim):
        raise ValueError(
            "transformer identity action_dim mismatch: "
            f"{action_dim} vs {expected_action_dim}"
        )
    sha256 = validate_sha256(payload.get("sha256"), label="transformer SHA256")

    raw_shapes = payload.get("action_shapes")
    if not isinstance(raw_shapes, dict) or set(raw_shapes) != set(ACTION_TENSOR_KEYS):
        raise ValueError("transformer identity action_shapes field is invalid")
    action_shapes: dict[str, tuple[int, ...]] = {}
    for key in ACTION_TENSOR_KEYS:
        raw_shape = raw_shapes[key]
        if not isinstance(raw_shape, list):
            raise ValueError("transformer identity action shapes must be JSON arrays")
        action_shapes[key] = tuple(
            _required_integer(
                size,
                label=f"transformer identity shape dimension for {key}",
            )
            for size in raw_shape
        )
    _validate_action_shapes(action_shapes, action_dim=action_dim)
    if payload.get("required_sentinel_keys") != list(TRANSFORMER_SENTINEL_KEYS):
        raise ValueError("transformer identity sentinel contract is invalid")
    return {
        "schema_version": TRANSFORMER_IDENTITY_SCHEMA_VERSION,
        "file_name": TRANSFORMER_WEIGHTS_FILENAME,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "tensor_count": tensor_count,
        "action_dim": action_dim,
        "action_shapes": {key: list(action_shapes[key]) for key in ACTION_TENSOR_KEYS},
        "required_sentinel_keys": list(TRANSFORMER_SENTINEL_KEYS),
    }


def audit_transformer_checkpoint(
    weights_path: Path,
    *,
    expected_action_dim: int,
) -> dict[str, object]:
    """Parse the safetensors header and bind its exact bytes to SHA256."""

    path = Path(weights_path)
    initial_stat = _regular_file_stat(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if _stat_fingerprint(opened_stat) != _stat_fingerprint(initial_stat):
            raise RuntimeError("transformer weights changed before identity audit")
        with safe_open(
            _descriptor_path(descriptor), framework="pt", device="cpu"
        ) as checkpoint:
            tensor_keys = tuple(checkpoint.keys())
            missing = sorted(
                (set(ACTION_TENSOR_KEYS) | set(TRANSFORMER_SENTINEL_KEYS))
                - set(tensor_keys)
            )
            if missing:
                raise ValueError(
                    "transformer safetensors is missing required keys: "
                    + ", ".join(missing)
                )
            action_shapes = {
                key: tuple(int(size) for size in checkpoint.get_slice(key).get_shape())
                for key in ACTION_TENSOR_KEYS
            }
        sha256 = _sha256_descriptor(descriptor)
        final_opened_stat = os.fstat(descriptor)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid transformer safetensors file: {path}") from exc
    finally:
        os.close(descriptor)
    _validate_action_shapes(action_shapes, action_dim=int(expected_action_dim))
    final_stat = _regular_file_stat(path)
    if _stat_fingerprint(final_opened_stat) != _stat_fingerprint(
        initial_stat
    ) or _stat_fingerprint(final_stat) != _stat_fingerprint(initial_stat):
        raise RuntimeError("transformer weights changed during identity audit")
    identity = {
        "schema_version": TRANSFORMER_IDENTITY_SCHEMA_VERSION,
        "file_name": TRANSFORMER_WEIGHTS_FILENAME,
        "size_bytes": int(final_stat.st_size),
        "sha256": sha256,
        "tensor_count": len(tensor_keys),
        "action_dim": int(expected_action_dim),
        "action_shapes": {key: list(action_shapes[key]) for key in ACTION_TENSOR_KEYS},
        "required_sentinel_keys": list(TRANSFORMER_SENTINEL_KEYS),
    }
    return validate_recorded_transformer_identity(
        identity,
        expected_action_dim=expected_action_dim,
    )


def validate_transformer_identity_match(
    recorded: object,
    actual: object,
    *,
    expected_action_dim: int,
    label: str,
) -> dict[str, object]:
    """Require byte-for-byte identity metadata equality after validation."""

    recorded_identity = validate_recorded_transformer_identity(
        recorded,
        expected_action_dim=expected_action_dim,
    )
    actual_identity = validate_recorded_transformer_identity(
        actual,
        expected_action_dim=expected_action_dim,
    )
    if recorded_identity != actual_identity:
        raise ValueError(f"{label} transformer identity does not match actual bytes")
    return actual_identity
