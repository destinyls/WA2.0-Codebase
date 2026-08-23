# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Compare raw RGB and cached-latent conditioning through ``Policy.infer``."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import numpy.typing as npt
import torch

from n0_twam.evaluation.franka_offline_generation import (
    _current_state_row,
    _decode_actions,
    _set_runtime_environment,
    _target_action_rows,
)
from n0_twam.evaluation.franka_training_aligned_history import (
    build_training_aligned_video_history,
)
from n0_twam.integrations.worldarena.franka_policy import (
    DirectN0FrankaBackend,
    Policy,
    load_franka_policy_config,
)
from n0_twam.integrations.worldarena.franka_source import TASK_PROMPTS

PARITY_SCHEMA_VERSION = 1
ACTION_SELECTIONS = ("first_future_action", "last_future_action")


def _rotation_error_deg(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    left = prediction[:, 3:7].astype(np.float64, copy=False)
    right = target[:, 3:7].astype(np.float64, copy=False)
    cosine = np.abs(np.sum(left * right, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(cosine, 0.0, 1.0)))


def _position_error_cm(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> npt.NDArray[np.float64]:
    return np.linalg.norm(
        prediction[:, :3].astype(np.float64)
        - target[:, :3].astype(np.float64),
        axis=1,
    ) * 100.0


def _summary(
    *,
    raw_actions: npt.NDArray[np.float32],
    latent_actions: npt.NDArray[np.float32],
    target_actions: npt.NDArray[np.float32],
) -> dict[str, object]:
    if not (
        raw_actions.shape == latent_actions.shape == target_actions.shape
        and raw_actions.ndim == 2
        and raw_actions.shape[1] == 8
        and np.isfinite(raw_actions).all()
        and np.isfinite(latent_actions).all()
        and np.isfinite(target_actions).all()
    ):
        raise ValueError("Policy parity actions must be finite [N,8] arrays")
    raw_latent_position = _position_error_cm(raw_actions, latent_actions)
    raw_latent_rotation = _rotation_error_deg(raw_actions, latent_actions)
    raw_target_position = _position_error_cm(raw_actions, target_actions)
    latent_target_position = _position_error_cm(latent_actions, target_actions)
    raw_target_rotation = _rotation_error_deg(raw_actions, target_actions)
    latent_target_rotation = _rotation_error_deg(latent_actions, target_actions)
    position_metric_delta = abs(
        float(np.mean(raw_target_position))
        - float(np.mean(latent_target_position))
    )
    rotation_metric_delta = abs(
        float(np.mean(raw_target_rotation))
        - float(np.mean(latent_target_rotation))
    )
    passed = bool(
        float(np.max(raw_latent_position)) <= 0.01
        and float(np.max(raw_latent_rotation)) <= 0.05
        and position_metric_delta <= 0.01
        and rotation_metric_delta <= 0.05
    )
    return {
        "sample_count": int(raw_actions.shape[0]),
        "raw_vs_cached": {
            "action_max_absolute_error": float(
                np.max(np.abs(raw_actions[:, :7] - latent_actions[:, :7]))
            ),
            "position_error_cm_mean": float(np.mean(raw_latent_position)),
            "position_error_cm_max": float(np.max(raw_latent_position)),
            "rotation_error_deg_mean": float(np.mean(raw_latent_rotation)),
            "rotation_error_deg_max": float(np.max(raw_latent_rotation)),
        },
        "training_corpus_regression": {
            "raw_rgb_position_error_cm_mean": float(np.mean(raw_target_position)),
            "cached_latent_position_error_cm_mean": float(
                np.mean(latent_target_position)
            ),
            "position_metric_delta_cm": position_metric_delta,
            "raw_rgb_rotation_error_deg_mean": float(np.mean(raw_target_rotation)),
            "cached_latent_rotation_error_deg_mean": float(
                np.mean(latent_target_rotation)
            ),
            "rotation_metric_delta_deg": rotation_metric_delta,
        },
        "acceptance": {
            "position_parity_max_cm": 0.01,
            "rotation_parity_max_deg": 0.05,
            "metric_delta_max_cm": 0.01,
            "metric_delta_max_deg": 0.05,
            "passed": passed,
        },
    }


def _external_history(
    frames: tuple[Mapping[str, npt.NDArray[np.uint8]], ...],
) -> tuple[dict[str, npt.NDArray[np.uint8]], ...]:
    return tuple(
        {
            "cam_high": np.ascontiguousarray(frame["observation.images.top"]),
            "cam_left_wrist": np.ascontiguousarray(
                frame["observation.images.wrist_l"]
            ),
        }
        for frame in frames
    )


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new_samples(
    path: Path,
    *,
    raw_actions: npt.NDArray[np.float32],
    latent_actions: npt.NDArray[np.float32],
    target_actions: npt.NDArray[np.float32],
    current_actions: npt.NDArray[np.float32],
    cam_high: npt.NDArray[np.uint8],
    cam_wrist: npt.NDArray[np.uint8],
    sample_indices: npt.NDArray[np.int64],
    episode_ids: npt.NDArray[np.int64],
    history_frame_counts: npt.NDArray[np.int64],
    action_selection: str,
) -> str:
    """Write immutable per-sample arrays used by visualization-only tooling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            raw_actions=raw_actions,
            latent_actions=latent_actions,
            target_actions=target_actions,
            current_actions=current_actions,
            cam_high=cam_high,
            cam_wrist=cam_wrist,
            sample_indices=sample_indices,
            episode_ids=episode_ids,
            history_frame_counts=history_frame_counts,
            action_selection=np.asarray(action_selection),
            selected_future_offset=np.asarray(
                0 if action_selection == "first_future_action" else 5,
                dtype=np.int64,
            ),
        )
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_franka_policy_conditioning_parity(
    *,
    policy_config: Path,
    artifact_root: Path,
    lerobot_root: Path,
    base_model: Path,
    normalizer: Path,
    dataset_view: Path,
    output: Path,
    samples_output: Path | None = None,
    max_samples: int | None = None,
    action_selection: str = "first_future_action",
) -> dict[str, object]:
    """Run paired cold generations through the deployable Franka ``Policy``."""

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when supplied")
    if action_selection not in ACTION_SELECTIONS:
        raise ValueError(f"action_selection must be one of {ACTION_SELECTIONS}")
    destination = Path(output).expanduser().resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Policy parity output already exists: {destination}")
    samples_destination = (
        None
        if samples_output is None
        else Path(samples_output).expanduser().resolve(strict=False)
    )
    if samples_destination is not None and (
        samples_destination.exists() or samples_destination.is_symlink()
    ):
        raise FileExistsError(
            f"Policy parity samples output already exists: {samples_destination}"
        )
    _set_runtime_environment(
        artifact_root=Path(artifact_root).resolve(strict=True),
        lerobot_root=Path(lerobot_root).resolve(strict=True),
        base_model=Path(base_model).resolve(strict=True),
        normalizer=Path(normalizer).resolve(strict=True),
    )

    from n0_twam.configs import TWAM_CONFIGS
    from n0_twam.dataset.lerobot_latent_dataset_franka import (
        MultiLatentLeRobotFrankaDataset,
    )
    from n0_twam.integrations.worldarena.franka_views import load_franka_view

    view = load_franka_view(Path(dataset_view).resolve(strict=True))
    training_config = copy.copy(TWAM_CONFIGS["track32_franka"])
    training_config.train_view_id = view.view_id
    training_config.dataset_view_path = str(Path(dataset_view).resolve(strict=True))
    training_config.max_latent_frames = 2
    training_config.cfg_prob = 0.0
    dataset = MultiLatentLeRobotFrankaDataset(
        config=training_config,
        num_init_worker=1,
    )
    sample_count = (
        len(dataset) if max_samples is None else min(max_samples, len(dataset))
    )
    if sample_count <= 0:
        raise ValueError("Policy parity dataset is empty")
    q01 = np.asarray(training_config.norm_stat["q01"], dtype=np.float32)
    q99 = np.asarray(training_config.norm_stat["q99"], dtype=np.float32)

    config_path = Path(policy_config).resolve(strict=True)
    loaded_policy_config = load_franka_policy_config(config_path)
    backend = DirectN0FrankaBackend(loaded_policy_config)
    raw_policy = Policy(str(config_path), backend=backend)
    latent_policy = Policy(str(config_path), backend=backend)
    raw_rows: list[npt.NDArray[np.float32]] = []
    latent_rows: list[npt.NDArray[np.float32]] = []
    target_rows: list[npt.NDArray[np.float32]] = []
    current_rows: list[npt.NDArray[np.float32]] = []
    high_images: list[npt.NDArray[np.uint8]] = []
    wrist_images: list[npt.NDArray[np.uint8]] = []
    episode_ids: list[int] = []
    history_frame_counts: list[int] = []
    strict_contract_verified = 0
    strict_zero_queue_verified = 0
    strict_execution_contract_verified = 0
    try:
        for index in range(sample_count):
            sample = dataset[index]
            current_ee10 = _current_state_row(sample, q01=q01, q99=q99)
            current_pose = _decode_actions(
                current_ee10.reshape(1, 10),
                quaternion_reference=np.asarray((0.0, 0.0, 0.0, 1.0), np.float32),
            )[0]
            target_ee10, valid = _target_action_rows(sample, q01=q01, q99=q99)
            target_index = 0 if action_selection == "first_future_action" else -1
            if not bool(valid[target_index]):
                raise ValueError("selected Policy action target is masked")
            target_pose = _decode_actions(
                target_ee10[target_index : target_index + 1]
                if target_index >= 0
                else target_ee10[-1:],
                quaternion_reference=current_pose[3:7],
            )[0]
            history = build_training_aligned_video_history(
                dataset,
                sample_index=index,
                max_latent_frames=training_config.max_latent_frames,
            )
            external_history = _external_history(history.frames)
            current_images = external_history[-1]
            task = dataset.sample_tasks[index]
            prompt = TASK_PROMPTS[task]
            common = {
                "prompt": prompt,
                "task_id": task,
                "images": current_images,
                "left_end_pose": current_pose[:7],
                "joint_qpos": np.asarray(
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, current_pose[7]),
                    dtype=np.float32,
                ),
            }
            raw_policy.reset({"prompt": prompt})
            raw_observation: dict[str, object] = {
                **common,
                "training_aligned_video_history": external_history,
            }
            if loaded_policy_config.live_contract is not None:
                frame_interval_ns = int(
                    round(1_000_000_000 / loaded_policy_config.live_contract.target_fps)
                )
                timestamps = tuple(
                    frame_index * frame_interval_ns
                    for frame_index in range(len(external_history))
                )
                raw_observation.update(
                    {
                        "training_aligned_video_timestamps_ns": timestamps,
                        "image_timestamp_ns": timestamps[-1],
                        "state_timestamp_ns": timestamps[-1],
                        "observation_sequence_id": 0,
                    }
                )
            if action_selection == "last_future_action":
                raw_observation.update(
                    {
                        "evaluation_mode": "offline_action_selection",
                        "evaluation_action_selection": action_selection,
                    }
                )
            raw_result = raw_policy.infer(raw_observation)
            if loaded_policy_config.live_contract is not None:
                metadata = raw_result.get("policy_metadata")
                timing = raw_result.get("policy_timing")
                if not isinstance(metadata, Mapping) or not isinstance(timing, Mapping):
                    raise RuntimeError("strict Policy result omitted contract metadata")
                contract = metadata.get("live_contract")
                if not isinstance(contract, Mapping) or contract.get("status") != "verified":
                    raise RuntimeError("strict Policy did not verify the RGB contract")
                strict_contract_verified += 1
                expected_queue_depth = 0
                if timing.get("queue_depth_after") != expected_queue_depth:
                    raise RuntimeError(
                        "strict Policy queue depth differs from the configured "
                        "execution contract"
                    )
                expected_discarded = (
                    loaded_policy_config.max_chunk_actions - 1
                    if loaded_policy_config.external_chunk_actions == 1
                    else loaded_policy_config.max_chunk_actions
                    - loaded_policy_config.external_chunk_actions
                )
                if metadata.get("discarded_prediction_count") != expected_discarded:
                    raise RuntimeError(
                        "strict Policy discarded predictions outside the configured "
                        "execution contract"
                    )
                strict_execution_contract_verified += 1
                if expected_queue_depth == 0:
                    strict_zero_queue_verified += 1
            latent = sample.get("latents")
            if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
                raise ValueError("evaluation sample has no cached video latent")
            latent_policy.reset({"prompt": prompt})
            latent_observation: dict[str, object] = {
                **common,
                "precomputed_video_latent": latent[:, :1].unsqueeze(0),
            }
            if loaded_policy_config.live_contract is not None:
                latent_observation["evaluation_mode"] = "cached_latent_reference"
            if action_selection == "last_future_action":
                latent_observation["evaluation_action_selection"] = action_selection
            latent_result = latent_policy.infer(latent_observation)
            raw_rows.append(np.asarray(raw_result["actions"], np.float32)[0])
            latent_rows.append(np.asarray(latent_result["actions"], np.float32)[0])
            target_rows.append(target_pose)
            current_rows.append(current_pose)
            high_images.append(np.ascontiguousarray(current_images["cam_high"]))
            wrist_images.append(
                np.ascontiguousarray(current_images["cam_left_wrist"])
            )
            episode_ids.append(history.episode_id)
            history_frame_counts.append(len(external_history))
    finally:
        backend.close()

    raw_actions = np.stack(raw_rows)
    latent_actions = np.stack(latent_rows)
    target_actions = np.stack(target_rows)
    metrics = _summary(
        raw_actions=raw_actions,
        latent_actions=latent_actions,
        target_actions=target_actions,
    )
    result: dict[str, object] = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "evaluation_type": "training-corpus regression diagnostic",
        "execution_path": "Policy.infer",
        "action_selection": action_selection,
        "selected_future_offset": 0 if action_selection == "first_future_action" else 5,
        "execution_mode": (
            "grounded_action_queue"
            if loaded_policy_config.live_contract is None
            else (
                "replan_each_observation"
                if loaded_policy_config.external_chunk_actions == 1
                else "future6_then_fresh_replan"
            )
        ),
        "conditioning_pair": [
            "training_aligned_raw_rgb",
            "precomputed_video_latent",
        ],
        "live_contract_verification": {
            "enabled": loaded_policy_config.live_contract is not None,
            "verified_sample_count": strict_contract_verified,
            "execution_contract_verified_sample_count": (
                strict_execution_contract_verified
            ),
            "expected_queue_depth_after_generation": (
                0
                if loaded_policy_config.live_contract is not None
                else None
            ),
            "zero_queue_sample_count": strict_zero_queue_verified,
        },
        "dataset_view_id": view.view_id,
        **metrics,
    }
    if samples_destination is not None:
        samples_sha256 = _write_new_samples(
            samples_destination,
            raw_actions=raw_actions,
            latent_actions=latent_actions,
            target_actions=target_actions,
            current_actions=np.stack(current_rows),
            cam_high=np.stack(high_images),
            cam_wrist=np.stack(wrist_images),
            sample_indices=np.arange(sample_count, dtype=np.int64),
            episode_ids=np.asarray(episode_ids, dtype=np.int64),
            history_frame_counts=np.asarray(history_frame_counts, dtype=np.int64),
            action_selection=action_selection,
        )
        result["visualization_samples"] = {
            "path": str(samples_destination),
            "sha256": samples_sha256,
        }
    _write_new_json(destination, result)
    return result


__all__ = ("evaluate_franka_policy_conditioning_parity",)
