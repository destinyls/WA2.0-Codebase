# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Offline safety, protocol, and latency replay for the AgileX Policy."""

from __future__ import annotations

import io
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Mapping, cast

import numpy as np
import numpy.typing as npt

from n0_twam.evaluation.franka_atomic_io import publish_atomic_file
from n0_twam.evaluation.franka_prediction_io import (
    StablePredictionInput,
    capture_prediction_input,
    require_prediction_unchanged,
)

from .agilex_manifest import canonical_sha256
from .agilex_policy import AgileXActionBackend, AgileXPolicy
from .agilex_policy_contracts import ACTION_SCHEMA, RGB_KEYS, FloatArray, ImageArray

REPLAY_SCHEMA_VERSION = 1
_OBSERVATION_KEYS = frozenset(
    (
        "schema_version",
        "task_id",
        "images",
        "joint_qpos",
        "execution_dt_s",
        "tactile_keys",
        "tactile_images",
        "wrench_keys",
        "wrench",
    )
)


def _string(value: object, *, label: str) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _strings(value: npt.NDArray[np.generic], *, label: str) -> tuple[str, ...]:
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{label} must be a string vector")
    result = tuple(_string(item, label=label) for item in value.tolist())
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicates")
    return result


def _observation(raw: bytes) -> dict[str, object]:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != _OBSERVATION_KEYS:
                raise ValueError("AgileX replay observation keys differ from schema")
            schema = np.asarray(archive["schema_version"])
            task_id = np.asarray(archive["task_id"])
            if (
                schema.shape != ()
                or type(schema.item()) is not int
                or schema.item() != 1
            ):
                raise ValueError("unsupported AgileX replay observation schema")
            if task_id.shape != ():
                raise ValueError("AgileX replay task_id must be scalar")
            images = np.asarray(archive["images"])
            qpos = np.asarray(archive["joint_qpos"])
            dt = np.asarray(archive["execution_dt_s"])
            tactile_keys = _strings(
                np.asarray(archive["tactile_keys"]), label="tactile_keys"
            )
            tactile = np.asarray(archive["tactile_images"])
            wrench_keys = _strings(
                np.asarray(archive["wrench_keys"]), label="wrench_keys"
            )
            wrench = np.asarray(archive["wrench"])
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("AgileX replay"):
            raise
        raise ValueError("invalid AgileX replay observation NPZ") from error
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[0] != 3:
        raise ValueError("AgileX replay images must be uint8 [3,H,W,3]")
    if images.shape[-1] != 3 or min(images.shape[1:3]) < 1:
        raise ValueError("AgileX replay images have an invalid RGB shape")
    if qpos.dtype != np.float32 or qpos.shape != (14,) or not np.isfinite(qpos).all():
        raise ValueError("AgileX replay joint_qpos must be finite float32[14]")
    if dt.shape != () or not np.issubdtype(dt.dtype, np.floating):
        raise ValueError("AgileX replay execution_dt_s must be a float scalar")
    execution_dt_s = float(dt.item())
    if not np.isfinite(execution_dt_s) or execution_dt_s <= 0.0:
        raise ValueError("AgileX replay execution_dt_s must be positive")
    if (
        tactile.dtype != np.uint8
        or tactile.ndim != 4
        or tactile.shape[0] != len(tactile_keys)
        or tactile.shape[-1] != 3
    ):
        raise ValueError("AgileX replay tactile_images must be uint8 [S,H,W,3]")
    if (
        wrench.dtype != np.float32
        or wrench.shape != (len(wrench_keys), 6)
        or not np.isfinite(wrench).all()
    ):
        raise ValueError("AgileX replay wrench must be finite float32 [W,6]")
    return {
        "task_id": _string(task_id.item(), label="task_id"),
        "images": np.ascontiguousarray(images),
        "joint_qpos": np.ascontiguousarray(qpos),
        "execution_dt_s": execution_dt_s,
        "tactile_keys": tactile_keys,
        "tactile_images": np.ascontiguousarray(tactile),
        "wrench_keys": wrench_keys,
        "wrench": np.ascontiguousarray(wrench),
    }


@contextmanager
def _frozen_config(snapshot: StablePredictionInput) -> Iterator[Path]:
    temporary = snapshot.path.parent / (
        f".{snapshot.path.name}.agilex-replay-{os.getpid()}-"
        f"{os.urandom(8).hex()}.incomplete"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(snapshot.raw)
            handle.flush()
            os.fsync(handle.fileno())
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: object) -> str:
    raw = (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )

    def writer(handle: BinaryIO) -> None:
        handle.write(raw)

    def validator(candidate: bytes) -> None:
        if not isinstance(json.loads(candidate), dict):
            raise ValueError("AgileX replay receipt must contain a JSON object")

    return str(
        publish_atomic_file(
            output=path,
            writer=writer,
            validator=validator,
            label="AgileX replay receipt",
        ).sha256
    )


def publish_agilex_replay_observation(
    *,
    output: Path,
    task_id: str,
    images: Mapping[str, ImageArray],
    joint_qpos: FloatArray,
    execution_dt_s: float,
    tactile: Mapping[str, ImageArray] | None,
    wrench: Mapping[str, FloatArray] | None,
) -> dict[str, object]:
    """Publish one pickle-free observation for the real Policy replay path."""

    if tuple(images) != RGB_KEYS:
        raise ValueError("AgileX replay RGB keys differ from the canonical route")
    tactile = {} if tactile is None else dict(tactile)
    wrench = {} if wrench is None else dict(wrench)
    rgb = np.stack([np.asarray(images[key]) for key in RGB_KEYS])
    tactile_images = (
        np.stack([np.asarray(image) for image in tactile.values()])
        if tactile
        else np.empty((0, 1, 1, 3), dtype=np.uint8)
    )
    wrench_array = (
        np.stack([np.asarray(value) for value in wrench.values()]).astype(
            np.float32, copy=False
        )
        if wrench
        else np.empty((0, 6), dtype=np.float32)
    )
    payload = {
        "schema_version": np.asarray(REPLAY_SCHEMA_VERSION, dtype=np.int64),
        "task_id": np.asarray(_string(task_id, label="task_id")),
        "images": rgb,
        "joint_qpos": np.asarray(joint_qpos, dtype=np.float32),
        "execution_dt_s": np.asarray(execution_dt_s, dtype=np.float64),
        "tactile_keys": np.asarray(tuple(tactile), dtype=np.str_),
        "tactile_images": tactile_images,
        "wrench_keys": np.asarray(tuple(wrench), dtype=np.str_),
        "wrench": wrench_array,
    }

    def writer(handle: BinaryIO) -> None:
        np.savez(handle, **payload)

    published = publish_atomic_file(
        output=output,
        writer=writer,
        validator=lambda raw: _observation(raw),
        label="AgileX replay observation",
    )
    return {
        "path": str(published.path),
        "sha256": published.sha256,
        "size_bytes": published.size,
    }


def _realtime(
    times_ms: list[float], *, control_hz: float, deadline_s: float
) -> dict[str, object]:
    budget_ms = min(1000.0 / control_hz, deadline_s * 1000.0)
    p95 = float(np.percentile(times_ms, 95))
    return {
        "sample_count": len(times_ms),
        "control_hz": control_hz,
        "budget_ms": budget_ms,
        "mean_backend_ms": float(np.mean(times_ms)),
        "p50_backend_ms": float(np.percentile(times_ms, 50)),
        "p95_backend_ms": p95,
        "max_backend_ms": float(np.max(times_ms)),
        "realtime_pass": p95 <= budget_ms,
        "failure_reasons": [] if p95 <= budget_ms else ["backend_p95_exceeds_budget"],
    }


def _modalities(
    payload: Mapping[str, object],
) -> tuple[dict[str, ImageArray] | None, dict[str, FloatArray] | None]:
    tactile_keys = cast(tuple[str, ...], payload["tactile_keys"])
    tactile_array = cast(npt.NDArray[np.uint8], payload["tactile_images"])
    wrench_keys = cast(tuple[str, ...], payload["wrench_keys"])
    wrench_array = cast(npt.NDArray[np.float32], payload["wrench"])
    tactile = (
        {key: tactile_array[index] for index, key in enumerate(tactile_keys)}
        if tactile_keys
        else None
    )
    wrench = (
        {key: wrench_array[index] for index, key in enumerate(wrench_keys)}
        if wrench_keys
        else None
    )
    return tactile, wrench


def run_agilex_policy_replay(
    *,
    config_path: Path,
    observation_path: Path,
    steps: int,
    output: Path,
    control_hz: float = 10.0,
    require_realtime: bool = False,
    backend: AgileXActionBackend | None = None,
) -> dict[str, object]:
    """Run repeated full replans with perfect-proprioception execution feedback."""

    if type(steps) is not int or not 1 <= steps <= 1024:
        raise ValueError("AgileX replay steps must be in [1,1024]")
    if not np.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("AgileX replay control_hz must be positive")
    destination = Path(output).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"AgileX replay receipt already exists: {destination}")
    config_snapshot = capture_prediction_input(config_path)
    observation_snapshot = capture_prediction_input(observation_path)
    payload = _observation(observation_snapshot.raw)

    from .agilex_backend import DirectN0AgileXBackend
    from .agilex_policy_io import load_agilex_policy_config

    with _frozen_config(config_snapshot) as frozen:
        config = load_agilex_policy_config(frozen)
        active_backend = backend or DirectN0AgileXBackend(config)
        policy = AgileXPolicy(config.policy, backend=active_backend)
        task_id = cast(str, payload["task_id"])
        if task_id not in config.policy.task_routes:
            raise ValueError("replay task_id is absent from the policy task routes")
        route = config.policy.task_routes[task_id]
        tactile, wrench = _modalities(payload)
        if tuple(() if tactile is None else tactile) != route.tactile_keys:
            raise ValueError("replay tactile keys differ from the signed task route")
        if tuple(() if wrench is None else wrench) != route.wrench_keys:
            raise ValueError("replay wrench keys differ from the signed task route")
        image_array = cast(npt.NDArray[np.uint8], payload["images"])
        images = {key: image_array[index] for index, key in enumerate(RGB_KEYS)}
        qpos = cast(npt.NDArray[np.float32], payload["joint_qpos"]).copy()
        dt = cast(float, payload["execution_dt_s"])
        actions: list[list[float]] = []
        times_ms: list[float] = []
        intervention_count = 0
        grounded_count = 0
        policy.reset({"task_id": task_id, "prompt": route.prompt})
        try:
            for _ in range(steps):
                result = policy.infer(
                    {
                        "task_id": task_id,
                        "prompt": route.prompt,
                        "timestamp": time.time(),
                        "execution_dt_s": dt,
                        "joint_qpos": qpos,
                        "images": images,
                        **({"tactile": tactile} if tactile is not None else {}),
                        **({"wrench": wrench} if wrench is not None else {}),
                    }
                )
                chunk = np.asarray(result["actions"])
                metadata = cast(Mapping[str, object], result["policy_metadata"])
                timing = cast(Mapping[str, object], result["policy_timing"])
                if chunk.dtype != np.float32 or chunk.ndim != 2 or chunk.shape[1] != 14:
                    raise ValueError("AgileX Policy replay returned invalid actions")
                if metadata.get("action_schema") != ACTION_SCHEMA:
                    raise ValueError("AgileX Policy replay action schema changed")
                raw_backend_ms = timing.get("backend_ms", -1.0)
                if isinstance(raw_backend_ms, bool) or not isinstance(
                    raw_backend_ms, (int, float)
                ):
                    raise ValueError("AgileX Policy replay timing is invalid")
                backend_ms = float(raw_backend_ms)
                if not np.isfinite(backend_ms) or backend_ms < 0.0:
                    raise ValueError("AgileX Policy replay timing is invalid")
                executed = np.ascontiguousarray(chunk[0])
                actions.append([float(value) for value in executed])
                qpos = executed
                times_ms.append(backend_ms)
                count = metadata.get("safety_intervention_count")
                if type(count) is not int or count < 0:
                    raise ValueError("AgileX Policy replay safety metadata is invalid")
                intervention_count += count
                grounded = timing.get("grounded")
                if type(grounded) is not bool:
                    raise ValueError(
                        "AgileX Policy replay grounding metadata is invalid"
                    )
                grounded_count += int(grounded)
        finally:
            close = getattr(active_backend, "close", None)
            if callable(close):
                close()

    require_prediction_unchanged(config_snapshot)
    require_prediction_unchanged(observation_snapshot)
    realtime = _realtime(
        times_ms,
        control_hz=float(control_hz),
        deadline_s=config.policy.safety.max_inference_latency_s,
    )
    if require_realtime and not realtime["realtime_pass"]:
        raise RuntimeError("AgileX Policy replay did not meet its realtime budget")
    core: dict[str, object] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "complete",
        "execution_tier": "offline_engineering_smoke",
        "safety_contract_tier": "offline_non_robot_envelope",
        "organizer_evaluation_completed": False,
        "real_robot_evaluation_completed": False,
        "config": str(config_snapshot.path),
        "config_sha256": config_snapshot.sha256,
        "observation": str(observation_snapshot.path),
        "observation_sha256": observation_snapshot.sha256,
        "task_id": task_id,
        "prompt": route.prompt,
        "steps": steps,
        "action_schema": ACTION_SCHEMA,
        "action_dim": 14,
        "action_format": "joint_absolute",
        "tactile_profile": config.policy.tactile_profile,
        "safety_intervention_count": intervention_count,
        "grounded_chunk_count": grounded_count,
        "realtime_assessment": realtime,
        "executed_actions": actions,
    }
    receipt = {**core, "replay_identity_sha256": canonical_sha256(core)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_sha256 = _write_json(destination, receipt)
    return {
        **receipt,
        "receipt": str(destination.resolve(strict=True)),
        "receipt_file_sha256": receipt_sha256,
    }


__all__ = (
    "REPLAY_SCHEMA_VERSION",
    "publish_agilex_replay_observation",
    "run_agilex_policy_replay",
)
