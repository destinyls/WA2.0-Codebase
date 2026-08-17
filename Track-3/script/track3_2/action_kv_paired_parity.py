# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""In-process semantic parity probe for AgileX Action-KV reuse."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import torch

FloatArray: TypeAlias = npt.NDArray[np.float32]
ImageArray: TypeAlias = npt.NDArray[np.uint8]


def _digest(array: FloatArray) -> str:
    payload = np.ascontiguousarray(array, dtype=np.float32).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _delta(left: FloatArray, right: FloatArray) -> dict[str, object]:
    if (
        left.shape != right.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("paired Action-KV outputs are invalid")
    absolute = np.abs(left.astype(np.float64) - right.astype(np.float64))
    atol = 1e-4
    rtol = 1e-3
    return {
        "passed": bool(np.allclose(left, right, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(absolute.max()),
        "mean_abs": float(absolute.mean()),
        "left_sha256": _digest(left),
        "right_sha256": _digest(right),
    }


def _infer_once(
    *,
    backend: Any,
    job_config: Any,
    enabled: bool,
    task_id: str,
    prompt: str,
    profile: str,
    seed: int,
    images: Mapping[str, ImageArray],
    qpos: FloatArray,
    tactile: Mapping[str, ImageArray],
    wrench: Mapping[str, FloatArray],
    synchronize: Callable[[], None],
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    job_config.action_denoise_kv_reuse = enabled
    backend.reset(task_id=task_id, prompt=prompt, seed=seed, profile=profile)
    actions, latent = backend.infer_prediction_latent_chunk(
        images=images,
        current_qpos14=qpos,
        tactile_images=tactile,
        wrench=wrench,
    )
    synchronize()
    if actions.shape != (12, 14) or not np.isfinite(actions).all():
        raise ValueError("paired Action-KV probe produced invalid actions")
    if not torch.is_tensor(latent) or not bool(torch.isfinite(latent).all()):
        raise ValueError("paired Action-KV probe produced invalid video latent")
    receipt = getattr(backend._server, "_last_action_kv_reuse_receipt", None)
    if not isinstance(receipt, dict):
        raise ValueError("paired Action-KV probe has no server receipt")
    latent_array = latent.detach().to(device="cpu", dtype=torch.float32).numpy()
    return (
        np.ascontiguousarray(actions, dtype=np.float32),
        np.ascontiguousarray(latent_array, dtype=np.float32),
        copy.deepcopy(receipt),
    )


def run_paired_action_kv_parity(
    *,
    backend: Any,
    job_config: Any,
    task_id: str,
    prompt: str,
    profile: str,
    seed: int,
    images: Mapping[str, ImageArray],
    qpos: FloatArray,
    tactile: Mapping[str, ImageArray],
    wrench: Mapping[str, FloatArray],
    synchronize: Callable[[], None],
) -> dict[str, object]:
    """Run OFF twice and ON once with one loaded model and one exact seed."""

    original = bool(getattr(job_config, "action_denoise_kv_reuse", False))
    outputs: dict[str, tuple[FloatArray, FloatArray, dict[str, Any]]] = {}
    try:
        for name, enabled in (("off_a", False), ("off_b", False), ("on", True)):
            outputs[name] = _infer_once(
                backend=backend,
                job_config=job_config,
                enabled=enabled,
                task_id=task_id,
                prompt=prompt,
                profile=profile,
                seed=seed,
                images=images,
                qpos=qpos,
                tactile=tactile,
                wrench=wrench,
                synchronize=synchronize,
            )
    finally:
        job_config.action_denoise_kv_reuse = original

    off_a, off_b, on = outputs["off_a"], outputs["off_b"], outputs["on"]
    repeat_action = _delta(off_a[0], off_b[0])
    repeat_latent = _delta(off_a[1], off_b[1])
    cached_action = _delta(off_a[0], on[0])
    cached_latent = _delta(off_a[1], on[1])
    receipt_contract = {
        "off_a_requested": off_a[2].get("requested") is False,
        "off_b_requested": off_b[2].get("requested") is False,
        "on_requested": on[2].get("requested") is True,
        "on_used_fast_path": on[2].get("used_fast_path") is True,
        "on_cached_steps": on[2].get("cached_steps"),
    }
    passed = all(
        bool(item["passed"])
        for item in (repeat_action, repeat_latent, cached_action, cached_latent)
    ) and all(
        bool(value)
        for key, value in receipt_contract.items()
        if key != "on_cached_steps"
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "execution_tier": "same_process_same_seed_semantic_probe",
        "seed": seed,
        "passed": passed,
        "off_repeat_action": repeat_action,
        "off_repeat_video_latent": repeat_latent,
        "off_vs_on_action": cached_action,
        "off_vs_on_video_latent": cached_latent,
        "receipt_contract": receipt_contract,
    }


__all__ = ("run_paired_action_kv_parity",)
