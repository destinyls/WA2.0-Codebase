# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Pure JSON-schema parsers shared by the AgileX local policy boundary."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

from n0_twam.tactile_profiles import MIXED, VISION_ONLY, VISION_TACTILE

from .agilex_manifest import canonical_sha256
from .agilex_policy_contracts import AgileXSafetyContract, AgileXTaskRoute

POLICY_CONFIG_SCHEMA_VERSION = 1
PROFILES = frozenset((VISION_TACTILE, MIXED, VISION_ONLY))
POLICY_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
DEVICE = re.compile(r"^[0-9]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def json_object(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def exact_fields(value: Mapping[str, object], fields: set[str], *, label: str) -> None:
    missing = fields - set(value)
    unexpected = set(value) - fields
    if missing or unexpected:
        raise ValueError(
            f"invalid {label}: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def positive_int(value: object, *, label: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [1,{maximum}]")
    return value


def resolve_input(path: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    raw = Path(value).expanduser()
    return (raw if raw.is_absolute() else path.parent / raw).resolve(strict=True)


def resolve_output(path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("serve_output must be a non-empty path string")
    raw = Path(value).expanduser()
    return (raw if raw.is_absolute() else path.parent / raw).resolve(strict=False)


def load_json_file(path: Path, *, label: str) -> dict[str, object]:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        payload = json.loads(raw.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {raw}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _positive_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _vector14(value: object, *, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (14,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain 14 finite values")
    return tuple(float(item) for item in array)


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicates")
    return result


def _self_hash(payload: Mapping[str, object], *, label: str) -> str:
    digest = sha256(payload.get("contract_sha256"), label=f"{label} hash")
    core = {key: value for key, value in payload.items() if key != "contract_sha256"}
    if canonical_sha256(core) != digest:
        raise ValueError(f"{label} self hash mismatch")
    return digest


def parse_safety(value: object) -> AgileXSafetyContract:
    payload = json_object(value, label="safety")
    fields = {
        "lower_bounds",
        "upper_bounds",
        "max_step_per_second",
        "min_execution_dt_s",
        "max_execution_dt_s",
        "max_state_age_s",
        "max_inference_latency_s",
        "contract_sha256",
    }
    exact_fields(payload, fields, label="AgileX safety contract")
    digest = _self_hash(payload, label="AgileX safety contract")
    return AgileXSafetyContract(
        lower_bounds=_vector14(payload["lower_bounds"], label="lower_bounds"),
        upper_bounds=_vector14(payload["upper_bounds"], label="upper_bounds"),
        max_step_per_second=_vector14(
            payload["max_step_per_second"], label="max_step_per_second"
        ),
        min_execution_dt_s=_positive_float(
            payload["min_execution_dt_s"], label="min_execution_dt_s"
        ),
        max_execution_dt_s=_positive_float(
            payload["max_execution_dt_s"], label="max_execution_dt_s"
        ),
        max_state_age_s=_positive_float(
            payload["max_state_age_s"], label="max_state_age_s"
        ),
        max_inference_latency_s=_positive_float(
            payload["max_inference_latency_s"], label="max_inference_latency_s"
        ),
        contract_sha256=digest,
    )


def parse_routes(value: object) -> tuple[dict[str, AgileXTaskRoute], str]:
    payload = json_object(value, label="task_routes")
    if not payload:
        raise ValueError("task_routes must not be empty")
    routes: dict[str, AgileXTaskRoute] = {}
    for task_id, raw_route in payload.items():
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task route keys must be non-empty strings")
        route = json_object(raw_route, label=f"task route {task_id!r}")
        fields = {
            "task_id",
            "prompt",
            "tactile_required",
            "wrench_required",
            "tactile_keys",
            "wrench_keys",
            "contract_sha256",
        }
        exact_fields(route, fields, label=f"task route {task_id!r}")
        digest = _self_hash(route, label=f"task route {task_id!r}")
        if route["task_id"] != task_id:
            raise ValueError("task route key and task_id differ")
        if (
            type(route["tactile_required"]) is not bool
            or type(route["wrench_required"]) is not bool
        ):
            raise ValueError("task route modality flags must be booleans")
        routes[task_id] = AgileXTaskRoute(
            task_id=task_id,
            prompt=cast(str, route["prompt"]),
            tactile_required=route["tactile_required"],
            wrench_required=route["wrench_required"],
            tactile_keys=_string_tuple(route["tactile_keys"], label="tactile_keys"),
            wrench_keys=_string_tuple(route["wrench_keys"], label="wrench_keys"),
            contract_sha256=digest,
        )
    return routes, canonical_sha256(dict(payload))


__all__ = (
    "DEVICE",
    "POLICY_CONFIG_SCHEMA_VERSION",
    "POLICY_ID",
    "PROFILES",
    "exact_fields",
    "load_json_file",
    "parse_routes",
    "parse_safety",
    "positive_int",
    "resolve_input",
    "resolve_output",
    "sha256",
)
