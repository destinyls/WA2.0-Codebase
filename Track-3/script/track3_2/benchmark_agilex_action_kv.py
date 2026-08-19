# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Run a sealed single-sample AgileX Action-KV HCU latency canary."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import time
from typing import Any, BinaryIO, Mapping, cast

import numpy as np
import torch

from n0_twam.evaluation.agilex_evaluation_view import (
    load_agilex_evaluation_view,
)
from n0_twam.evaluation.agilex_generation_data import (
    current_qpos14,
    uint8_camera_tiles,
)
from n0_twam.evaluation.agilex_generation_runtime import (
    dataset_index,
    sample_modalities,
    target_latent,
)
from n0_twam.evaluation.agilex_offline_generation import (
    prepare_agilex_evaluation_environment,
)
from n0_twam.evaluation.franka_atomic_io import publish_atomic_file
from n0_twam.integrations.worldarena.agilex_manifest import (
    canonical_sha256,
    sha256_file,
)
from script.track3_2.action_kv_paired_parity import run_paired_action_kv_parity

logger = logging.getLogger(__name__)


def _runtime_environment() -> dict[str, object]:
    device_name = "unavailable"
    if torch.cuda.is_available():
        device_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        hpu = getattr(torch, "hpu", None)
        available = getattr(hpu, "is_available", None)
        get_device_name = getattr(hpu, "get_device_name", None)
        if callable(available) and bool(available()) and callable(get_device_name):
            device_name = str(get_device_name())
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
        "hip_version": str(getattr(torch.version, "hip", None)),
        "device_name": device_name,
    }


def _summary(values: list[float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("latency summary requires at least one sample")
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    hpu = getattr(torch, "hpu", None)
    available = getattr(hpu, "is_available", None)
    synchronize = getattr(hpu, "synchronize", None)
    if callable(available) and callable(synchronize) and bool(available()):
        synchronize()


def _phase_summary(
    receipts: list[dict[str, Any]],
) -> dict[str, dict[str, int | float]]:
    collected: dict[str, list[float]] = {}
    for receipt in receipts:
        timing = receipt.get("phase_latency_ms")
        if not isinstance(timing, Mapping):
            raise ValueError("Action KV receipt has no phase latency payload")
        samples = timing.get("samples")
        if not isinstance(samples, Mapping):
            raise ValueError("Action KV receipt phase samples are invalid")
        for phase, raw_values in samples.items():
            if not isinstance(phase, str) or not isinstance(raw_values, list):
                raise ValueError("Action KV receipt phase sample schema is invalid")
            values = collected.setdefault(phase, [])
            for value in raw_values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("Action KV phase latency must be numeric")
                latency = float(value)
                if not np.isfinite(latency) or latency < 0:
                    raise ValueError("Action KV phase latency must be finite")
                values.append(latency)
    return {phase: _summary(values) for phase, values in collected.items() if values}


def _validate_output(raw: bytes) -> dict[str, object]:
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError("Action KV benchmark receipt is incomplete")
    identity = payload.get("benchmark_identity_sha256")
    core = {
        key: value
        for key, value in payload.items()
        if key != "benchmark_identity_sha256"
    }
    if identity != canonical_sha256(core):
        raise ValueError("Action KV benchmark receipt self hash mismatch")
    return cast(dict[str, object], payload)


def _publish(output: Path, payload: dict[str, object]) -> None:
    core = dict(payload)
    sealed = {**core, "benchmark_identity_sha256": canonical_sha256(core)}
    raw = (
        json.dumps(sealed, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )

    def writer(handle: BinaryIO) -> None:
        handle.write(raw)

    publish_atomic_file(
        output=output,
        writer=writer,
        validator=_validate_output,
        label="AgileX Action KV benchmark receipt",
    )


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.kv_reuse not in (0, 1):
        raise ValueError("kv_reuse must be exactly 0 or 1")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup/iterations are invalid")
    if (
        not isinstance(args.expected_source_manifest_sha256, str)
        or len(args.expected_source_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.expected_source_manifest_sha256
        )
    ):
        raise ValueError("expected source manifest SHA256 is invalid")
    raw_source_manifest = args.source_manifest.expanduser()
    if raw_source_manifest.is_symlink() or not raw_source_manifest.is_file():
        raise ValueError("source manifest must be a regular non-symlink file")
    source_manifest = raw_source_manifest.resolve(strict=True)
    input_hashes = {
        "train_request_file_sha256": sha256_file(args.train_request),
        "policy_config_file_sha256": sha256_file(args.policy_config),
        "evaluation_view_file_sha256": sha256_file(args.evaluation_view),
        "source_manifest_file_sha256": sha256_file(source_manifest),
    }
    if (
        input_hashes["source_manifest_file_sha256"]
        != args.expected_source_manifest_sha256
    ):
        raise ValueError("source manifest file SHA256 mismatch")
    prepare_agilex_evaluation_environment(
        train_request=args.train_request,
        device=args.device,
        seed=args.seed,
    )
    os.environ["N0_ACTION_DENOISE_KV_REUSE"] = str(args.kv_reuse)

    from n0_twam.configs import TWAM_CONFIGS
    from n0_twam.dataset.lerobot_latent_dataset_agilex import (
        MultiLatentLeRobotAgileXDataset,
    )
    from n0_twam.integrations.worldarena.agilex_backend import (
        DirectN0AgileXBackend,
    )
    from n0_twam.integrations.worldarena.agilex_policy_io import (
        load_agilex_policy_config,
    )
    from n0_twam.track32_agilex.request import load_agilex_train_request

    request = load_agilex_train_request(args.train_request)
    policy = load_agilex_policy_config(args.policy_config)
    serve_output = args.serve_output.expanduser()
    if not serve_output.is_absolute():
        serve_output = (Path.cwd() / serve_output).resolve(strict=False)
    if serve_output.exists() or serve_output.is_symlink():
        raise FileExistsError(
            f"Action KV benchmark serve output exists: {serve_output}"
        )
    policy = replace(policy, serve_output=serve_output)
    view = load_agilex_evaluation_view(args.evaluation_view)
    entry = next(
        (
            item
            for item in view.entries
            if policy.policy.task_routes[item.task_id].tactile_required
        ),
        None,
    )
    if entry is None:
        raise ValueError("evaluation view contains no tactile task")
    route = policy.policy.task_routes[entry.task_id]
    runtime = copy.copy(TWAM_CONFIGS[f"track3_agilex_{request.profile}"])
    runtime.max_latent_frames = 2
    runtime.num_init_worker = 1
    runtime.cfg_prob = 0.0
    runtime.tactile_cfg_prob = 0.0
    dataset = MultiLatentLeRobotAgileXDataset(runtime, num_init_worker=1)
    child, local_index = dataset_index(dataset)[(entry.repo_id, entry.episode_id)]
    sample = cast(dict[str, object], child[local_index])
    meta = child.new_metas[local_index]
    backend = DirectN0AgileXBackend(policy)
    total_ms: list[float] = []
    receipts: list[dict[str, Any]] = []
    action_digests: list[str] = []
    action_outputs: list[list[list[float]]] = []
    latent_digests: list[str] = []
    paired_semantic_parity: dict[str, object] | None = None
    job_config = getattr(backend._server, "job_config", None)  # noqa: SLF001
    raw_runtime_contract = getattr(job_config, "server_runtime_contract", None)
    if not isinstance(raw_runtime_contract, Mapping):
        raise ValueError("server did not expose its runtime contract")
    runtime_contract = copy.deepcopy(dict(raw_runtime_contract))
    runtime_contract_sha256 = runtime_contract.get("contract_sha256")
    runtime_core = {
        key: value
        for key, value in runtime_contract.items()
        if key != "contract_sha256"
    }
    if runtime_contract_sha256 != canonical_sha256(runtime_core):
        raise ValueError("server runtime contract self hash mismatch")
    if runtime_core.get("action_denoise_kv_reuse") is not bool(args.kv_reuse):
        raise ValueError("server runtime contract differs from the selected A/B arm")
    try:
        target_video = backend.decode_video_latent_batch(
            target_latent(sample), batch_size=1, spatial_tiles=3
        )[0]
        images, tactile, wrench = sample_modalities(
            backend=backend,
            sample=sample,
            route=route,
            target_tiles=uint8_camera_tiles(target_video),
            tactile_sensor_ids=dict(runtime.tactile_sensor_id_map),
            wrench_sensor_ids=dict(runtime.wrench_sensor_id_map),
            decode_batch_size=1,
        )
        if tactile is None or wrench is None:
            raise ValueError("selected AgileX benchmark sample is not contact-complete")
        qpos = current_qpos14(
            child,
            episode_id=entry.episode_id,
            row_id=int(meta["start_frame"]),
        )
        for ordinal in range(args.warmup + args.iterations):
            backend.reset(
                task_id=entry.task_id,
                prompt=route.prompt,
                seed=args.seed + ordinal,
                profile=request.profile,
            )
            _synchronize()
            started = time.perf_counter()
            actions, latent = backend.infer_prediction_latent_chunk(
                images=images,
                current_qpos14=qpos,
                tactile_images=tactile,
                wrench=wrench,
            )
            _synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if actions.shape != (12, 14) or not np.isfinite(actions).all():
                raise ValueError("benchmark produced invalid qpos14 actions")
            if not torch.is_tensor(latent) or not bool(torch.isfinite(latent).all()):
                raise ValueError("benchmark produced invalid video latent")
            receipt = getattr(
                backend._server, "_last_action_kv_reuse_receipt", None
            )  # noqa: SLF001
            if not isinstance(receipt, dict):
                raise ValueError("server did not publish an Action KV receipt")
            if ordinal >= args.warmup:
                total_ms.append(float(elapsed_ms))
                receipts.append(cast(dict[str, Any], copy.deepcopy(receipt)))
                action_digests.append(canonical_sha256(actions.tolist()))
                action_outputs.append(actions.astype(np.float64).tolist())
                latent_bytes = (
                    latent.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                    .numpy()
                    .tobytes()
                )
                latent_digests.append(hashlib.sha256(latent_bytes).hexdigest())
        if args.paired_parity:
            paired_semantic_parity = run_paired_action_kv_parity(
                backend=backend,
                job_config=job_config,
                task_id=entry.task_id,
                prompt=route.prompt,
                profile=request.profile,
                seed=args.seed + 1_000_000,
                images=images,
                qpos=qpos,
                tactile=tactile,
                wrench=wrench,
                synchronize=_synchronize,
            )
    finally:
        backend.close()

    final_input_hashes = {
        "train_request_file_sha256": sha256_file(args.train_request),
        "policy_config_file_sha256": sha256_file(args.policy_config),
        "evaluation_view_file_sha256": sha256_file(args.evaluation_view),
        "source_manifest_file_sha256": sha256_file(source_manifest),
    }
    if final_input_hashes != input_hashes:
        raise RuntimeError("Action KV benchmark input changed during execution")

    expected_fast = args.kv_reuse == 1
    if any(bool(item.get("used_fast_path")) != expected_fast for item in receipts):
        raise ValueError("Action KV fast-path activation differs from the selected arm")
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "execution_tier": "offline_hcu_action_kv_latency_canary",
        "real_robot_evaluation_completed": False,
        "organizer_evaluation_completed": False,
        "kv_reuse": bool(args.kv_reuse),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "checkpoint_identity_sha256": policy.checkpoint_identity_sha256,
        **input_hashes,
        "serve_output": str(serve_output),
        "evaluation_view_sha256": view.view_sha256,
        "runtime_environment": _runtime_environment(),
        "server_runtime_contract": runtime_contract,
        "server_runtime_contract_sha256": runtime_contract_sha256,
        "sample": entry.to_json_dict(),
        "total_latency_ms": _summary(total_ms),
        "action_phase_latency_ms": _phase_summary(receipts),
        "action_digests": action_digests,
        "action_outputs": action_outputs,
        "latent_digests": latent_digests,
        "action_kv_receipts": receipts,
    }
    if paired_semantic_parity is not None:
        payload["paired_semantic_parity"] = paired_semantic_parity
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-request", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--evaluation-view", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--serve-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-reuse", type=int, choices=(0, 1), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="0")
    parser.add_argument("--paired-parity", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parser().parse_args()
    result = benchmark(args)
    _publish(args.output, result)
    logger.info("published Action KV benchmark receipt: %s", args.output)


if __name__ == "__main__":
    main()
