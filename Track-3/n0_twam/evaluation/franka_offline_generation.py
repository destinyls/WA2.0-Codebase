# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Generate sealed Track 3.2 future RGB/action predictions from one checkpoint."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import numpy.typing as npt
import torch

from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.evaluation.franka_prediction_artifact import (
    publish_franka_predictions,
)
from n0_twam.evaluation.franka_prediction_io import (
    capture_prediction_input,
    require_prediction_unchanged,
)
from n0_twam.evaluation.franka_training_aligned_history import (
    build_training_aligned_video_history,
)
from n0_twam.evaluation.sealed_artifact_io import validate_sha256
from n0_twam.integrations.worldarena.franka_actions import (
    DERIVED_ACTION_SCHEMA,
    FRANKA_ACTION_SCHEMA,
    FRANKA_QUATERNION_ORDER,
    ee10_to_end_pose8,
    embed_ee10_in_ee20,
    extract_ee10_from_ee20,
)
from n0_twam.integrations.worldarena.franka_manifest import canonical_sha256
from n0_twam.integrations.worldarena.franka_serve_bundle import (
    verify_franka_serve_bundle,
)
from n0_twam.integrations.worldarena.franka_source import TASK_PROMPTS


def _set_runtime_environment(
    *, artifact_root: Path, lerobot_root: Path, base_model: Path, normalizer: Path
) -> None:
    os.environ["N0_TRACK32_ARTIFACT_ROOT"] = str(artifact_root)
    os.environ["N0_TRACK32_LEROBOT_ROOT"] = str(lerobot_root)
    os.environ["N0_BASE_MODEL"] = str(base_model)
    os.environ["N0_EMPTY_EMBEDDING"] = str(base_model / "empty_emb.pt")
    os.environ["N0_TRACK32_NORMALIZER_PATH"] = str(normalizer)
    os.environ["N0_TRACK32_RUN_ROLE"] = "development"


def _decode_actions(
    values: npt.NDArray[np.float32],
    *,
    quaternion_reference: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    output: list[npt.NDArray[np.float32]] = []
    reference = quaternion_reference
    for value in values:
        pose = ee10_to_end_pose8(value, quaternion_reference=reference).reshape(8)
        output.append(pose)
        reference = pose[3:7]
    return np.ascontiguousarray(np.stack(output), dtype=np.float32)


def _uint8_tiles(video: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("decoded video must be uint8 [T,H,W,3]")
    if video.shape[2] % 2 != 0:
        raise ValueError("two-camera decoded video width must be even")
    width = video.shape[2] // 2
    if width < 7 or video.shape[1] < 7:
        raise ValueError("decoded video resolution is too small for SSIM")
    return np.ascontiguousarray(
        np.stack((video[:, :, :width], video[:, :, width:]), axis=0)
    )


def _target_action_rows(
    sample: Mapping[str, object],
    *,
    q01: npt.NDArray[np.float32],
    q99: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    actions_value = sample.get("actions")
    mask_value = sample.get("actions_mask")
    actions = np.asarray(getattr(actions_value, "numpy", lambda: actions_value)())
    mask = np.asarray(getattr(mask_value, "numpy", lambda: mask_value)())
    if actions.shape != (20, 2, 6, 1) or mask.shape != actions.shape:
        raise ValueError("evaluation sample must contain a two-frame EE20 target")
    if mask.dtype != np.bool_:
        raise ValueError("evaluation action mask must be boolean")
    normalized = actions[:10, 1, :, 0].T.astype(np.float32, copy=False)
    valid = np.all(mask[:10, 1, :, 0], axis=0)
    raw = (normalized + 1.0) / 2.0 * (q99[:10] - q01[:10] + 1e-6) + q01[:10]
    return np.ascontiguousarray(raw), np.ascontiguousarray(valid)


def _current_state_row(
    sample: Mapping[str, object],
    *,
    q01: npt.NDArray[np.float32],
    q99: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Recover the observed EE10 state stored in the cold conditioning frame."""

    actions_value = sample.get("actions")
    mask_value = sample.get("actions_mask")
    actions = np.asarray(getattr(actions_value, "numpy", lambda: actions_value)())
    mask = np.asarray(getattr(mask_value, "numpy", lambda: mask_value)())
    if actions.shape != (20, 2, 6, 1) or mask.shape != actions.shape:
        raise ValueError("evaluation sample must contain a two-frame EE20 target")
    cold = actions[:10, 0, :, 0].T.astype(np.float32, copy=False)
    cold_valid = mask[:10, 0, :, 0].T
    if cold_valid.dtype != np.bool_ or not bool(cold_valid.all()):
        raise ValueError("evaluation cold state must mark every EE10 channel valid")
    if not np.allclose(cold, cold[:1], rtol=0.0, atol=1e-6):
        raise ValueError("evaluation cold state must repeat one observed EE10 row")
    raw = (cold[0] + 1.0) / 2.0 * (q99[:10] - q01[:10] + 1e-6) + q01[:10]
    if not np.isfinite(raw).all():
        raise ValueError("evaluation cold state contains non-finite values")
    return np.ascontiguousarray(raw, dtype=np.float32)


def generate_franka_offline_predictions(
    *,
    checkpoint: Path,
    checkpoint_identity_sha256: str,
    serve_bundle: Path,
    serve_bundle_receipt_sha256: str,
    serve_output: Path,
    artifact_root: Path,
    lerobot_root: Path,
    base_model: Path,
    normalizer: Path,
    dataset_view: Path,
    output: Path,
    seed: int = 20260810,
    device: str = "0",
    distributed_port: int = 29651,
    video_inference_steps: int = 3,
    action_inference_steps: int = 4,
    decode_batch_size: int = 4,
    max_samples: int | None = None,
    conditioning_source: str = "precomputed_video_latent",
) -> dict[str, object]:
    """Run the real cold Direct backend on the frozen validation60 roster."""

    expected_checkpoint_sha = validate_sha256(
        checkpoint_identity_sha256, label="checkpoint identity SHA256"
    )
    expected_bundle_sha = validate_sha256(
        serve_bundle_receipt_sha256, label="serve-bundle receipt SHA256"
    )
    if seed < 0 or distributed_port <= 0 or not device.isdecimal():
        raise ValueError("evaluation seed/device/port is invalid")
    if video_inference_steps <= 0 or action_inference_steps <= 0:
        raise ValueError("inference step counts must be positive")
    if (
        isinstance(decode_batch_size, bool)
        or not isinstance(decode_batch_size, int)
        or decode_batch_size <= 0
    ):
        raise ValueError("decode_batch_size must be a positive integer")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when supplied")
    if conditioning_source not in {
        "precomputed_video_latent",
        "training_aligned_raw_rgb",
    }:
        raise ValueError("unsupported Franka conditioning source")
    destination = Path(output).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"prediction output already exists: {destination}")
    serve_destination = Path(serve_output).expanduser()
    if serve_destination.exists() or serve_destination.is_symlink():
        raise FileExistsError(f"serve output already exists: {serve_destination}")

    checkpoint_root = Path(checkpoint).expanduser().resolve(strict=True)
    artifact_path = Path(artifact_root).expanduser().resolve(strict=True)
    lerobot_path = Path(lerobot_root).expanduser().resolve(strict=True)
    base_path = Path(base_model).expanduser().resolve(strict=True)
    normalizer_path = Path(normalizer).expanduser().resolve(strict=True)
    view_snapshot = capture_prediction_input(dataset_view)
    normalizer_snapshot = capture_prediction_input(normalizer_path)
    checkpoint_before = build_strict_checkpoint_identity(
        capture_strict_checkpoint_snapshot(checkpoint_root)
    )
    if checkpoint_before.get("identity_sha256") != expected_checkpoint_sha:
        raise ValueError("checkpoint identity differs from the evaluation request")
    bundle_before = verify_franka_serve_bundle(
        serve_bundle, expected_receipt_file_sha256=expected_bundle_sha
    )
    if bundle_before.get("checkpoint_identity") != checkpoint_before:
        raise ValueError("serve bundle does not belong to the requested checkpoint")

    from n0_twam.evaluation.tactile_provenance import audit_vae_decoder

    decoder_before = audit_vae_decoder(base_path / "vae")
    decoder_sha256 = canonical_sha256(decoder_before)
    _set_runtime_environment(
        artifact_root=artifact_path,
        lerobot_root=lerobot_path,
        base_model=base_path,
        normalizer=normalizer_path,
    )

    from n0_twam.configs import TWAM_CONFIGS
    from n0_twam.dataset.lerobot_latent_dataset_franka import (
        MultiLatentLeRobotFrankaDataset,
    )
    from n0_twam.integrations.worldarena.franka_policy import (
        DirectN0FrankaBackend,
        FrankaPolicyConfig,
        FrankaSafetyConfig,
    )
    from n0_twam.integrations.worldarena.franka_views import load_franka_view

    view = load_franka_view(Path(dataset_view))
    if view.view_sha256 != canonical_sha256(
        {
            key: value
            for key, value in view.to_json_dict().items()
            if key != "view_sha256"
        }
    ):
        raise ValueError("validation view canonical identity changed")
    config = copy.copy(TWAM_CONFIGS["track32_franka"])
    config.train_view_id = view.view_id
    config.dataset_view_path = str(Path(dataset_view).resolve(strict=True))
    config.max_latent_frames = 2
    config.seed = seed
    config.cfg_prob = 0.0
    dataset = MultiLatentLeRobotFrankaDataset(config=config, num_init_worker=1)
    if len(dataset) != len(view.entries):
        raise ValueError("validation dataset and frozen view episode counts differ")
    sample_count = (
        len(dataset) if max_samples is None else min(max_samples, len(dataset))
    )
    q01 = np.asarray(config.norm_stat["q01"], dtype=np.float32)
    q99 = np.asarray(config.norm_stat["q99"], dtype=np.float32)

    evaluation_samples = [dataset[index] for index in range(sample_count)]
    target_ee10_rows: list[npt.NDArray[np.float32]] = []
    current_poses: list[npt.NDArray[np.float32]] = []
    current_ee20_rows: list[npt.NDArray[np.float32]] = []
    action_valid: list[npt.NDArray[np.bool_]] = []
    target_latents: list[torch.Tensor] = []
    for sample in evaluation_samples:
        target_ee10, valid = _target_action_rows(sample, q01=q01, q99=q99)
        current_ee10 = _current_state_row(sample, q01=q01, q99=q99)
        latent = sample.get("latents")
        if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
            raise ValueError("evaluation sample must contain a four-dimensional latent")
        target_ee10_rows.append(target_ee10)
        current_poses.append(ee10_to_end_pose8(current_ee10).reshape(8))
        current_ee20_rows.append(embed_ee10_in_ee20(current_ee10).reshape(20))
        action_valid.append(valid)
        target_latents.append(latent[:, :2].unsqueeze(0).detach().cpu())

    raw_conditioning_latents: list[torch.Tensor] = []
    if conditioning_source == "training_aligned_raw_rgb":
        from n0_twam.evaluation.franka_training_aligned_encoder import (
            TrainingAlignedFrankaVideoEncoder,
        )

        server_config = TWAM_CONFIGS["track32_franka_server"]
        encoder = TrainingAlignedFrankaVideoEncoder(
            base_model=base_path,
            device=device,
            camera_keys=tuple(server_config.obs_cam_keys),
            height=int(server_config.height),
            width=int(server_config.width),
            dtype=server_config.param_dtype,
        )
        try:
            for index in range(sample_count):
                history = build_training_aligned_video_history(
                    dataset,
                    sample_index=index,
                    max_latent_frames=config.max_latent_frames,
                )
                raw_conditioning_latents.append(encoder.encode(history.frames))
        finally:
            encoder.close()

    backend = DirectN0FrankaBackend(
        FrankaPolicyConfig(
            policy_id="n0-twam-franka-offline-eval",
            serve_bundle=Path(serve_bundle).resolve(strict=True),
            serve_bundle_receipt_sha256=expected_bundle_sha,
            serve_output=serve_destination.resolve(strict=False),
            cuda_visible_device=device,
            distributed_port=distributed_port,
            episode_seed=seed,
            max_chunk_actions=12,
            external_chunk_actions=1,
            video_inference_steps=video_inference_steps,
            action_inference_steps=action_inference_steps,
            safety=FrankaSafetyConfig(
                workspace_min=(-10.0, -10.0, -10.0),
                workspace_max=(10.0, 10.0, 10.0),
                max_translation_step_m=10.0,
                max_rotation_step_rad=float(np.pi),
                gripper_min=-10.0,
                gripper_max=10.0,
                max_gripper_step=10.0,
            ),
        )
    )

    predicted_actions: list[npt.NDArray[np.float32]] = []
    target_actions: list[npt.NDArray[np.float32]] = []
    predicted_latents: list[torch.Tensor] = []
    conditioning_cosines: list[float] = []
    conditioning_absolute_errors: list[float] = []
    for index in range(sample_count):
        target_ee10 = target_ee10_rows[index]
        current_pose = current_poses[index]
        task = dataset.sample_tasks[index]
        backend.reset(prompt=TASK_PROMPTS[task], seed=seed + index)
        inference_inputs: dict[str, object]
        if conditioning_source == "training_aligned_raw_rgb":
            inference_inputs = {
                "images": {},
                "precomputed_video_latent": raw_conditioning_latents[index],
            }
        else:
            inference_inputs = {
                "images": {},
                "precomputed_video_latent": target_latents[index][:, :, :1],
            }
        raw_action, generated_latent = backend.infer_prediction_latent_chunk(
            current_ee20=current_ee20_rows[index],
            **inference_inputs,
        )
        if conditioning_source == "training_aligned_raw_rgb":
            online = raw_conditioning_latents[index].float().reshape(-1)
            cached = target_latents[index][:, :, :1].float().reshape(-1)
            conditioning_cosines.append(
                float(torch.nn.functional.cosine_similarity(online, cached, dim=0))
            )
            conditioning_absolute_errors.append(float(torch.mean(torch.abs(online - cached))))
        if not isinstance(generated_latent, torch.Tensor):
            raise TypeError("Direct backend returned a non-tensor video latent")
        predicted_latents.append(generated_latent.detach().cpu())
        prediction_ee10 = extract_ee10_from_ee20(raw_action[:, 1, :].T)
        predicted_actions.append(
            _decode_actions(
                prediction_ee10,
                quaternion_reference=current_pose[3:7],
            )
        )
        target_actions.append(
            _decode_actions(target_ee10, quaternion_reference=current_pose[3:7])
        )

    # Raw-prefix conditioning uses the streaming VAE encoder. Decode videos only
    # after every Policy forward so decoder allocations cannot starve the encoder.
    backend.release_video_conditioning_cache()
    target_videos = backend.decode_video_latent_batch(
        torch.cat(target_latents, dim=0), batch_size=decode_batch_size
    )
    target_tiles = [_uint8_tiles(video) for video in target_videos]
    if any(tiles.shape[1] != 5 for tiles in target_tiles):
        raise ValueError("target cold video must decode to five RGB frames")

    predicted_videos = backend.decode_video_latent_batch(
        torch.cat(predicted_latents, dim=0), batch_size=decode_batch_size
    )
    backend.close()
    predicted_tiles = [_uint8_tiles(video) for video in predicted_videos]
    if any(
        prediction.shape != target.shape
        for prediction, target in zip(predicted_tiles, target_tiles, strict=True)
    ):
        raise ValueError("predicted and target cold videos have different shapes")

    arrays = {
        "view_names": np.asarray(("cam_high", "cam_left_wrist")),
        "frame_offsets": np.arange(1, 5, dtype=np.int64),
        "action_offsets": np.arange(1, 7, dtype=np.int64),
        "sample_ids": np.asarray(dataset.sample_ids[:sample_count]),
        "task_ids": np.asarray(dataset.sample_tasks[:sample_count]),
        "lerobot_episode_ids": np.asarray(
            dataset.sample_episode_ids[:sample_count], dtype=np.int64
        ),
        "predicted_rgb": np.stack([tiles[:, 1:5] for tiles in predicted_tiles]),
        "target_rgb": np.stack([tiles[:, 1:5] for tiles in target_tiles]),
        "video_valid": np.ones((sample_count, 2, 4), dtype=np.bool_),
        "predicted_end_pose": np.stack(predicted_actions),
        "target_end_pose": np.stack(target_actions),
        "action_valid": np.stack(action_valid),
    }
    result = publish_franka_predictions(
        output=destination,
        metadata={
            "checkpoint_identity_sha256": expected_checkpoint_sha,
            "dataset_view_id": view.view_id,
            "dataset_view_sha256": view.view_sha256,
            "decoder_sha256": decoder_sha256,
            "seed": seed,
            "run_role": "development",
            "prediction_mode": "policy_action",
            "wire_action_schema": FRANKA_ACTION_SCHEMA,
            "derived_action_schema": DERIVED_ACTION_SCHEMA,
            "quaternion_order": FRANKA_QUATERNION_ORDER,
        },
        arrays=arrays,
    )

    require_prediction_unchanged(view_snapshot)
    require_prediction_unchanged(normalizer_snapshot)
    checkpoint_after = build_strict_checkpoint_identity(
        capture_strict_checkpoint_snapshot(checkpoint_root)
    )
    decoder_after = audit_vae_decoder(base_path / "vae")
    bundle_after = verify_franka_serve_bundle(
        serve_bundle, expected_receipt_file_sha256=expected_bundle_sha
    )
    if (
        checkpoint_after != checkpoint_before
        or decoder_after != decoder_before
        or bundle_after != bundle_before
    ):
        raise RuntimeError("evaluation inputs changed during prediction generation")
    return {
        "status": "complete",
        "sample_count": sample_count,
        "checkpoint_identity_sha256": expected_checkpoint_sha,
        "dataset_view_id": view.view_id,
        "dataset_view_sha256": view.view_sha256,
        "decoder_sha256": decoder_sha256,
        "prediction_mode": "policy_action",
        "conditioning_source": conditioning_source,
        "conditioning_latent_cosine_mean": (
            None if not conditioning_cosines else float(np.mean(conditioning_cosines))
        ),
        "conditioning_latent_mean_absolute_error": (
            None
            if not conditioning_absolute_errors
            else float(np.mean(conditioning_absolute_errors))
        ),
        "current_state_source": "observation_state_cold_slot",
        "video_inference_steps": video_inference_steps,
        "action_inference_steps": action_inference_steps,
        **result,
    }


__all__ = ("generate_franka_offline_predictions",)
