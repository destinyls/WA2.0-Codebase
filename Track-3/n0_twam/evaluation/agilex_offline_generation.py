# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Generate sealed future RGB/qpos14 predictions from one AgileX checkpoint."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.evaluation.franka_prediction_io import (
    capture_prediction_input,
    require_prediction_unchanged,
)
from n0_twam.integrations.worldarena.agilex_manifest import canonical_sha256
from n0_twam.track32_agilex.request import (
    load_agilex_train_request,
    require_agilex_request_unchanged,
)
from n0_twam.track32_agilex.runner import build_agilex_request_environment

from .agilex_evaluation_view import load_agilex_evaluation_view
from .agilex_generation_runtime import (
    dataset_index,
    sample_modalities,
    target_latent,
)
from .agilex_generation_data import (
    current_qpos14,
    target_qpos14_rows,
    uint8_camera_tiles,
)
from .agilex_prediction_artifact import publish_agilex_predictions

FloatArray = npt.NDArray[np.float32]
ImageArray = npt.NDArray[np.uint8]


def _set_environment(request: object, *, device: str, seed: int) -> None:
    environment = build_agilex_request_environment(request, environ=os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": device,
            "HIP_VISIBLE_DEVICES": device,
            "PYTHONHASHSEED": str(seed),
            "N0_TRACK3_AGILEX_LOAD_WORKER": "0",
        }
    )
    runtime = getattr(request, "runtime", None)
    if getattr(runtime, "accelerator_profile", None) == "hcu_performance":
        environment["N0_WAN_VAE_ATTENTION_FALLBACK"] = "1"
    for name in tuple(os.environ):
        if name.startswith("N0_") or name in {
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        }:
            os.environ.pop(name, None)
    os.environ.update(environment)


def prepare_agilex_evaluation_environment(
    *, train_request: Path, device: str, seed: int
) -> dict[str, object]:
    """Install the signed request environment before importing config modules."""

    _validate_arguments(
        seed=seed,
        device=device,
        decode_batch_size=1,
        max_samples=None,
    )
    request = load_agilex_train_request(train_request)
    _set_environment(request, device=device, seed=seed)
    require_agilex_request_unchanged(request)
    return {
        "request_sha256": request.source_sha256,
        "profile": request.profile,
    }


def _validate_arguments(
    *,
    seed: int,
    device: str,
    decode_batch_size: int,
    max_samples: int | None,
) -> None:
    if type(seed) is not int or seed < 0 or not device.isdecimal():
        raise ValueError("AgileX evaluation seed/device is invalid")
    if type(decode_batch_size) is not int or decode_batch_size <= 0:
        raise ValueError("decode_batch_size must be a positive integer")
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("max_samples must be positive when supplied")


def generate_agilex_offline_predictions(
    *,
    train_request: Path,
    policy_config: Path,
    dataset_view: Path,
    output: Path,
    seed: int = 20260813,
    device: str = "0",
    decode_batch_size: int = 4,
    max_samples: int | None = None,
    replay_observation_output: Path | None = None,
) -> dict[str, object]:
    """Run the real Direct backend on an ordered all-task proxy roster."""

    _validate_arguments(
        seed=seed,
        device=device,
        decode_batch_size=decode_batch_size,
        max_samples=max_samples,
    )
    destination = Path(output).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"AgileX prediction output exists: {destination}")
    request = load_agilex_train_request(train_request)
    if request.profile not in {"mixed", "vision_tactile", "vision_only"}:
        raise ValueError("AgileX evaluation request profile is invalid")
    config_snapshot = capture_prediction_input(policy_config)
    view_snapshot = capture_prediction_input(dataset_view)
    normalizer_snapshot = capture_prediction_input(request.paths.normalizer)
    _set_environment(request, device=device, seed=seed)

    from n0_twam.configs import TWAM_CONFIGS
    from n0_twam.dataset.lerobot_latent_dataset_agilex import (
        MultiLatentLeRobotAgileXDataset,
    )
    from n0_twam.evaluation.tactile_provenance import audit_vae_decoder
    from n0_twam.integrations.worldarena.agilex_backend import DirectN0AgileXBackend
    from n0_twam.integrations.worldarena.agilex_policy_io import (
        load_agilex_policy_config,
    )

    policy = load_agilex_policy_config(config_snapshot.path)
    require_prediction_unchanged(config_snapshot)
    if policy.policy.tactile_profile != request.profile:
        raise ValueError("policy and training request tactile profiles differ")
    if (
        policy.source_manifest_sha256 != request.paths.source_manifest_sha256
        or policy.repo_route_manifest_file_sha256
        != request.paths.repo_route_manifest_sha256
        or policy.normalizer_file_sha256 != request.paths.normalizer_sha256
    ):
        raise ValueError("policy and training request artifact identities differ")
    receipt = policy.serve_bundle / "serve_bundle_receipt.json"
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    checkpoint_root = Path(receipt_payload["checkpoint_root"])
    checkpoint_before = build_strict_checkpoint_identity(
        capture_strict_checkpoint_snapshot(checkpoint_root)
    )
    if checkpoint_before["identity_sha256"] != policy.checkpoint_identity_sha256:
        raise ValueError("AgileX policy checkpoint identity differs")
    decoder_before = audit_vae_decoder(request.paths.base_model / "vae")
    decoder_sha256 = canonical_sha256(decoder_before)

    view = load_agilex_evaluation_view(dataset_view)
    entries = view.entries if max_samples is None else view.entries[:max_samples]
    runtime = copy.copy(TWAM_CONFIGS[f"track3_agilex_{request.profile}"])
    runtime.max_latent_frames = 2
    runtime.num_init_worker = 1
    runtime.cfg_prob = 0.0
    runtime.tactile_cfg_prob = 0.0
    dataset = MultiLatentLeRobotAgileXDataset(runtime, num_init_worker=1)
    index = dataset_index(dataset)
    selected = []
    for entry in entries:
        key = (entry.repo_id, entry.episode_id)
        if key not in index or entry.task_id not in policy.policy.task_routes:
            raise ValueError("AgileX evaluation view is absent from dataset/policy")
        child, local_index = index[key]
        meta = child.new_metas[local_index]
        if policy.policy.task_routes[entry.task_id].prompt not in meta["tasks"]:
            raise ValueError("AgileX evaluation task differs from converted metadata")
        selected.append((entry, child, local_index, child[local_index]))

    backend = DirectN0AgileXBackend(policy)
    import torch

    target_latents = torch.cat(
        [target_latent(cast(dict[str, object], item[3])) for item in selected], dim=0
    )
    target_video = backend.decode_video_latent_batch(
        target_latents,
        batch_size=decode_batch_size,
        spatial_tiles=3,
    )
    target_tiles = [uint8_camera_tiles(video) for video in target_video]
    predicted_latents: list[torch.Tensor] = []
    predicted_qpos: list[FloatArray] = []
    target_qpos: list[FloatArray] = []
    action_valid: list[npt.NDArray[np.bool_]] = []
    contact: list[bool] = []
    action_offsets: tuple[int, ...] | None = None
    replay_payload: (
        tuple[
            object,
            dict[str, ImageArray],
            FloatArray,
            dict[str, ImageArray] | None,
            dict[str, FloatArray] | None,
        ]
        | None
    ) = None
    replay_has_contact = False
    try:
        for ordinal, (entry, child, local_index, raw_sample) in enumerate(selected):
            sample = cast(dict[str, object], raw_sample)
            meta = child.new_metas[local_index]
            route = policy.policy.task_routes[entry.task_id]
            offsets = tuple(runtime.per_repo_action_offsets_per_anchor[entry.repo_id])
            targets, valid, future_offsets = target_qpos14_rows(
                sample,
                action_offsets=offsets,
                q01=np.asarray(runtime.norm_stat["q01"], dtype=np.float32),
                q99=np.asarray(runtime.norm_stat["q99"], dtype=np.float32),
            )
            if action_offsets is None:
                action_offsets = future_offsets
            elif action_offsets != future_offsets:
                raise ValueError(
                    "AgileX evaluation repos have different action offsets"
                )
            qpos = current_qpos14(
                child,
                episode_id=entry.episode_id,
                row_id=int(meta["start_frame"]),
            )
            images, tactile, wrench = sample_modalities(
                backend=backend,
                sample=sample,
                route=route,
                target_tiles=target_tiles[ordinal],
                tactile_sensor_ids=dict(runtime.tactile_sensor_id_map),
                wrench_sensor_ids=dict(runtime.wrench_sensor_id_map),
                decode_batch_size=decode_batch_size,
            )
            if replay_payload is None or (
                tactile is not None and not replay_has_contact
            ):
                replay_payload = (entry, images, qpos, tactile, wrench)
                replay_has_contact = tactile is not None
            backend.reset(
                task_id=entry.task_id,
                prompt=route.prompt,
                seed=seed + ordinal,
                profile=request.profile,
            )
            actions, latent = backend.infer_prediction_latent_chunk(
                images=images,
                current_qpos14=qpos,
                tactile_images=tactile,
                wrench=wrench,
            )
            if not isinstance(latent, torch.Tensor):
                raise TypeError("AgileX backend returned a non-tensor RGB latent")
            if len(actions) < len(future_offsets) + 1:
                raise ValueError("AgileX predicted action chunk is shorter than target")
            predicted_latents.append(latent.detach().cpu())
            predicted_qpos.append(np.ascontiguousarray(actions[1 : len(targets) + 1]))
            target_qpos.append(targets)
            action_valid.append(valid)
            contact.append(tactile is not None)
        predicted_video = backend.decode_video_latent_batch(
            torch.cat(predicted_latents, dim=0),
            batch_size=decode_batch_size,
            spatial_tiles=3,
        )
    finally:
        backend.close()
    predicted_tiles = [uint8_camera_tiles(video) for video in predicted_video]
    if any(
        left.shape != right.shape for left, right in zip(predicted_tiles, target_tiles)
    ):
        raise ValueError("AgileX predicted and target RGB shapes differ")
    assert action_offsets is not None
    frame_offsets = np.arange(1, target_tiles[0].shape[1], dtype=np.int64)
    arrays = {
        "view_names": np.asarray(("top", "wrist_l", "wrist_r")),
        "frame_offsets": frame_offsets,
        "action_offsets": np.asarray(action_offsets, dtype=np.int64),
        "sample_ids": np.asarray(
            [f"{item.repo_id}:episode_{item.episode_id:06d}" for item in entries]
        ),
        "task_ids": np.asarray([item.task_id for item in entries]),
        "repo_ids": np.asarray([item.repo_id for item in entries]),
        "contact_condition_present": np.asarray(contact, dtype=np.bool_),
        "predicted_rgb": np.stack([tiles[:, 1:] for tiles in predicted_tiles]),
        "target_rgb": np.stack([tiles[:, 1:] for tiles in target_tiles]),
        "video_valid": np.ones((len(entries), 3, len(frame_offsets)), dtype=np.bool_),
        "predicted_qpos14": np.stack(predicted_qpos).astype(np.float32),
        "target_qpos14": np.stack(target_qpos).astype(np.float32),
        "action_valid": np.stack(action_valid),
    }
    result = publish_agilex_predictions(
        output=destination,
        metadata={
            "checkpoint_identity_sha256": policy.checkpoint_identity_sha256,
            "dataset_view_id": view.view_id,
            "dataset_view_sha256": view.view_sha256,
            "decoder_sha256": decoder_sha256,
            "seed": seed,
            "run_role": request.train.run_role,
            "prediction_mode": "policy_action",
            "tactile_profile": request.profile,
        },
        arrays=arrays,
    )
    replay_result: dict[str, object] | None = None
    if replay_observation_output is not None:
        from n0_twam.integrations.worldarena.agilex_policy_replay import (
            publish_agilex_replay_observation,
        )

        assert replay_payload is not None
        entry, images, qpos, tactile, wrench = replay_payload
        safety = policy.policy.safety
        execution_dt_s = min(
            max(1.0 / 30.0, safety.min_execution_dt_s),
            safety.max_execution_dt_s,
        )
        replay_result = publish_agilex_replay_observation(
            output=replay_observation_output,
            task_id=entry.task_id,
            images=images,
            joint_qpos=qpos,
            execution_dt_s=execution_dt_s,
            tactile=tactile,
            wrench=wrench,
        )
    require_prediction_unchanged(config_snapshot)
    require_prediction_unchanged(view_snapshot)
    require_prediction_unchanged(normalizer_snapshot)
    require_agilex_request_unchanged(request)
    if (
        build_strict_checkpoint_identity(
            capture_strict_checkpoint_snapshot(checkpoint_root)
        )
        != checkpoint_before
        or audit_vae_decoder(request.paths.base_model / "vae") != decoder_before
    ):
        raise RuntimeError(
            "AgileX evaluation checkpoint/decoder changed during generation"
        )
    return {
        "status": "complete",
        "execution_tier": "training_distribution_offline_proxy",
        "organizer_evaluation_completed": False,
        "real_robot_evaluation_completed": False,
        "sample_count": len(entries),
        "checkpoint_identity_sha256": policy.checkpoint_identity_sha256,
        "dataset_view_id": view.view_id,
        "dataset_view_sha256": view.view_sha256,
        "decoder_sha256": decoder_sha256,
        "replay_observation": replay_result,
        **result,
    }


__all__ = (
    "generate_agilex_offline_predictions",
    "prepare_agilex_evaluation_environment",
)
