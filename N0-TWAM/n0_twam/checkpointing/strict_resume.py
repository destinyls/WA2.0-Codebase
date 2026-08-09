"""Safe, hash-bound sidecars for exact Track 3.1 training resume."""

from __future__ import annotations

import hashlib
import math
import stat
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import torch

from ._strict_resume_state import (
    _canonical_json_bytes,
    _float_hex,
    _integer,
    _parse_float_hex,
    _regular_file,
    _strict_json,
)
from ._strict_resume_state import (  # noqa: F401 - public re-exports
    capture_rng_state as capture_rng_state,
)
from ._strict_resume_state import load_rng_state as load_rng_state  # noqa: F401
from ._strict_resume_state import (  # noqa: F401
    load_rng_state_payloads as load_rng_state_payloads,
)
from ._strict_resume_state import restore_rng_state as restore_rng_state  # noqa: F401
from ._strict_resume_state import save_rng_state as save_rng_state  # noqa: F401
from ._strict_resume_state import validate_rng_state as validate_rng_state  # noqa: F401
from ._strict_resume_state import (
    write_json_atomic,
)
from ._strict_resume_state import (  # noqa: F401
    write_safetensors_atomic as write_safetensors_atomic,
)

SCHEDULER_STATE_FILENAME = "scheduler_state.json"
SIDECAR_INVENTORY_SCHEMA_VERSION = 1
SCHEDULER_STATE_SCHEMA_VERSION = 1


def _schedule_contract(contract: Mapping[str, object]) -> dict[str, object]:
    schedule = contract.get("lr_schedule")
    if schedule not in ("constant", "cosine"):
        raise ValueError("scheduler execution contract has unsupported lr_schedule")
    warmup = _integer(contract.get("warmup_steps"), "warmup_steps")
    total = _integer(contract.get("num_steps"), "num_steps", minimum=1)
    ratio_hex = _float_hex(contract.get("lr_min_ratio"), "lr_min_ratio")
    ratio = float.fromhex(ratio_hex)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("lr_min_ratio must be in [0, 1]")
    return {
        "lr_schedule": schedule,
        "warmup_steps": warmup,
        "num_steps": total,
        "lr_min_ratio_hex": ratio_hex,
    }


def _schedule_multiplier(step: int, schedule: Mapping[str, object]) -> float:
    warmup = _integer(schedule.get("warmup_steps"), "warmup_steps")
    if schedule["lr_schedule"] == "constant":
        return float(step) / float(max(1, warmup)) if step < warmup else 1.0
    if step < warmup:
        return float(step + 1) / float(max(1, warmup))
    total = _integer(schedule.get("num_steps"), "num_steps", minimum=1)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    minimum = _parse_float_hex(schedule.get("lr_min_ratio_hex"), "lr_min_ratio")
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def capture_scheduler_state(
    state_dict: Mapping[str, object],
    *,
    completed_steps: int,
    learning_rate: float,
    execution_contract: Mapping[str, object],
) -> dict[str, object]:
    """Convert a LambdaLR state dict into non-executable semantic JSON."""
    step = _integer(completed_steps, "completed_steps")
    base_lr = float.fromhex(_float_hex(learning_rate, "learning_rate"))
    if base_lr <= 0.0:
        raise ValueError("learning_rate must be positive")
    schedule = _schedule_contract(execution_contract)
    if step > _integer(schedule.get("num_steps"), "num_steps", minimum=1):
        raise ValueError("completed_steps exceeds the configured schedule")
    saved_epoch = _integer(state_dict.get("last_epoch"), "LambdaLR last_epoch")
    saved_step_count = _integer(
        state_dict.get("_step_count"), "LambdaLR step_count", minimum=1
    )
    if saved_epoch != step or saved_step_count != step + 1:
        raise ValueError("LambdaLR progress does not match completed_steps")
    raw_base = state_dict.get("base_lrs")
    raw_last = state_dict.get("_last_lr")
    lambdas = state_dict.get("lr_lambdas")
    if (
        not isinstance(raw_base, (list, tuple))
        or not raw_base
        or not isinstance(raw_last, (list, tuple))
        or len(raw_last) != len(raw_base)
        or not isinstance(lambdas, (list, tuple))
        or len(lambdas) != len(raw_base)
        or any(value is not None for value in lambdas)
    ):
        raise ValueError("scheduler state is not a supported LambdaLR state")
    base_lrs = [float.fromhex(_float_hex(value, "base LR")) for value in raw_base]
    last_lrs = [float.fromhex(_float_hex(value, "last LR")) for value in raw_last]
    if any(value != base_lr for value in base_lrs):
        raise ValueError("LambdaLR base LR does not match learning_rate")
    multiplier = _schedule_multiplier(step, schedule)
    expected = [value * multiplier for value in base_lrs]
    if any(actual != wanted for actual, wanted in zip(last_lrs, expected, strict=True)):
        raise ValueError("LambdaLR last LR does not match schedule f(completed_steps)")
    return {
        "schema_version": SCHEDULER_STATE_SCHEMA_VERSION,
        "state_format": "lambda_lr_semantic_v1",
        "completed_steps": step,
        "last_epoch": step,
        "step_count": step + 1,
        "base_lrs_hex": [value.hex() for value in base_lrs],
        "last_lrs_hex": [value.hex() for value in last_lrs],
        "schedule": schedule,
    }


def validate_scheduler_state(
    payload: object,
    *,
    completed_steps: int,
    learning_rate: float,
    execution_contract: Mapping[str, object],
) -> dict[str, object]:
    """Validate scheduler JSON against current progress and schedule semantics."""
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "state_format",
        "completed_steps",
        "last_epoch",
        "step_count",
        "base_lrs_hex",
        "last_lrs_hex",
        "schedule",
    }:
        raise ValueError("scheduler state has an invalid field set")
    schema = _integer(payload["schema_version"], "scheduler schema_version", minimum=1)
    if (
        schema != SCHEDULER_STATE_SCHEMA_VERSION
        or payload["state_format"] != "lambda_lr_semantic_v1"
    ):
        raise ValueError("unsupported scheduler state format")
    step = _integer(completed_steps, "completed_steps")
    saved_steps = _integer(payload["completed_steps"], "saved completed_steps")
    saved_epoch = _integer(payload["last_epoch"], "saved last_epoch")
    saved_step_count = _integer(payload["step_count"], "saved step_count", minimum=1)
    if saved_steps != step or saved_epoch != step:
        raise ValueError("scheduler state step is inconsistent")
    if saved_step_count != step + 1:
        raise ValueError("scheduler state step_count is inconsistent")
    expected_schedule = _schedule_contract(execution_contract)
    if payload["schedule"] != expected_schedule:
        raise ValueError("scheduler state execution contract is inconsistent")
    base_values = payload["base_lrs_hex"]
    last_values = payload["last_lrs_hex"]
    if (
        not isinstance(base_values, list)
        or not base_values
        or not isinstance(last_values, list)
    ):
        raise ValueError("scheduler LR fields must be non-empty JSON arrays")
    base_lrs = [_parse_float_hex(value, "base LR") for value in base_values]
    last_lrs = [_parse_float_hex(value, "last LR") for value in last_values]
    if len(last_lrs) != len(base_lrs):
        raise ValueError("scheduler LR group counts differ")
    expected_base = float.fromhex(_float_hex(learning_rate, "learning_rate"))
    multiplier = _schedule_multiplier(step, expected_schedule)
    if any(value != expected_base for value in base_lrs) or any(
        actual != base * multiplier
        for actual, base in zip(last_lrs, base_lrs, strict=True)
    ):
        raise ValueError("scheduler LR values do not match schedule semantics")
    return dict(payload)


def save_scheduler_state(
    checkpoint_dir: Path,
    state_dict: Mapping[str, object],
    *,
    completed_steps: int,
    learning_rate: float,
    execution_contract: Mapping[str, object],
) -> dict[str, object]:
    payload = capture_scheduler_state(
        state_dict,
        completed_steps=completed_steps,
        learning_rate=learning_rate,
        execution_contract=execution_contract,
    )
    write_json_atomic(Path(checkpoint_dir) / SCHEDULER_STATE_FILENAME, payload)
    return payload


def load_scheduler_state(
    checkpoint_dir: Path,
    *,
    completed_steps: int,
    learning_rate: float,
    execution_contract: Mapping[str, object],
) -> dict[str, object]:
    payload = _strict_json(Path(checkpoint_dir) / SCHEDULER_STATE_FILENAME)
    return validate_scheduler_state(
        payload,
        completed_steps=completed_steps,
        learning_rate=learning_rate,
        execution_contract=execution_contract,
    )


def restore_scheduler_state(
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    payload: object,
    *,
    completed_steps: int,
    learning_rate: float,
    execution_contract: Mapping[str, object],
) -> dict[str, object]:
    """Restore four semantic fields using the fresh LambdaLR state template."""
    if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        raise TypeError("strict scheduler restore requires LambdaLR")
    safe = validate_scheduler_state(
        payload,
        completed_steps=completed_steps,
        learning_rate=learning_rate,
        execution_contract=execution_contract,
    )
    schedule = safe["schedule"]
    if not isinstance(schedule, Mapping):
        raise ValueError("scheduler state schedule must be an object")
    total_steps = _integer(schedule.get("num_steps"), "num_steps", minimum=1)
    warmup_steps = _integer(schedule.get("warmup_steps"), "warmup_steps")
    probe_steps = {
        completed_steps,
        min(completed_steps + 1, total_steps),
        0,
        total_steps,
        min(warmup_steps, total_steps),
        min(max(warmup_steps - 1, 0), total_steps),
    }
    for probe_step in probe_steps:
        expected_multiplier = _schedule_multiplier(probe_step, schedule)
        if any(
            float(function(probe_step)) != expected_multiplier
            for function in scheduler.lr_lambdas
        ):
            raise ValueError(
                "current LambdaLR callable does not implement the saved schedule: "
                f"f({probe_step})"
            )
    base_values = safe["base_lrs_hex"]
    last_values = safe["last_lrs_hex"]
    if not isinstance(base_values, list) or not isinstance(last_values, list):
        raise ValueError("scheduler LR fields must be JSON arrays")
    base_lrs = [_parse_float_hex(value, "base LR") for value in base_values]
    last_lrs = [_parse_float_hex(value, "last LR") for value in last_values]
    template = scheduler.state_dict()
    template.update(
        base_lrs=base_lrs,
        last_epoch=safe["last_epoch"],
        _step_count=safe["step_count"],
        _last_lr=last_lrs,
    )
    scheduler.load_state_dict(template)
    for group, lr in zip(scheduler.optimizer.param_groups, last_lrs, strict=True):
        group["lr"] = lr
    return safe


def expected_sidecar_paths(
    world_size: int,
    *,
    include_transformer_config: bool = True,
    include_action_migration: bool = True,
) -> tuple[str, ...]:
    world = _integer(world_size, "world_size", minimum=1)
    paths = ["train_meta.json", "training_state.json", SCHEDULER_STATE_FILENAME]
    if include_action_migration:
        paths.append("action_migration_report.json")
    if include_transformer_config:
        paths.append("transformer/config.json")
    for rank in range(world):
        paths.extend(
            (f"rng_state_rank{rank}.json", f"rng_state_rank{rank}.safetensors")
        )
    return tuple(sorted(paths))


def _validate_relative_paths(paths: Sequence[str]) -> tuple[str, ...]:
    if isinstance(paths, (str, bytes)):
        raise ValueError("sidecar paths must be a sequence of relative paths")
    normalized: list[str] = []
    for value in paths:
        if not isinstance(value, str):
            raise ValueError("sidecar path must be a string")
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or path.as_posix() != value
            or ".." in path.parts
            or "." in path.parts
        ):
            raise ValueError(f"invalid sidecar path: {value!r}")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("sidecar paths contain duplicates")
    return tuple(sorted(normalized))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_path(checkpoint_dir: Path, relative: str) -> Path:
    root = Path(checkpoint_dir)
    current = root
    for part in PurePosixPath(relative).parts[:-1]:
        current /= part
        try:
            result = current.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"missing strict-resume sidecar directory: {current}"
            ) from None
        if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
            raise ValueError(f"sidecar parent must be a regular directory: {current}")
    return root / relative


def build_sidecar_inventory(
    checkpoint_dir: Path,
    expected_paths: Sequence[str],
) -> dict[str, object]:
    paths = _validate_relative_paths(expected_paths)
    files = []
    for relative in paths:
        path = _sidecar_path(checkpoint_dir, relative)
        result = _regular_file(path)
        files.append(
            {
                "path": relative,
                "size_bytes": result.st_size,
                "sha256": _sha256_file(path),
            }
        )
    core = {"schema_version": SIDECAR_INVENTORY_SCHEMA_VERSION, "files": files}
    return {
        **core,
        "inventory_sha256": hashlib.sha256(_canonical_json_bytes(core)).hexdigest(),
    }


def validate_sidecar_inventory(
    checkpoint_dir: Path,
    payload: object,
    expected_paths: Sequence[str],
) -> dict[str, object]:
    expected = _validate_relative_paths(expected_paths)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "files",
        "inventory_sha256",
    }:
        raise ValueError("sidecar inventory has an invalid field set")
    schema = _integer(
        payload["schema_version"], "sidecar inventory schema_version", minimum=1
    )
    if schema != SIDECAR_INVENTORY_SCHEMA_VERSION or not isinstance(
        payload["files"], list
    ):
        raise ValueError("unsupported sidecar inventory schema")
    files = payload["files"]
    if any(
        not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}
        for entry in files
    ):
        raise ValueError("sidecar inventory file entry is invalid")
    recorded_paths = tuple(entry["path"] for entry in files)
    if recorded_paths != expected:
        raise ValueError("sidecar inventory path set/order differs from expected paths")
    core = {"schema_version": SIDECAR_INVENTORY_SCHEMA_VERSION, "files": files}
    if (
        payload["inventory_sha256"]
        != hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
    ):
        raise ValueError("sidecar inventory canonical digest is inconsistent")
    for entry in files:
        size = _integer(entry["size_bytes"], "sidecar size", minimum=1)
        sha = entry["sha256"]
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(character not in "0123456789abcdef" for character in sha)
        ):
            raise ValueError("sidecar inventory SHA256 is invalid")
        path = _sidecar_path(checkpoint_dir, entry["path"])
        if _regular_file(path).st_size != size or _sha256_file(path) != sha:
            raise ValueError(f"sidecar bytes differ from inventory: {entry['path']}")
    return dict(payload)
