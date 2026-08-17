# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Official WorldArena Franka ``Policy`` adapter for N0-TWAM."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

from .franka_actions import (
    ee10_to_end_pose8,
    embed_ee10_in_ee20,
    end_pose8_to_ee10,
    extract_ee10_from_ee20,
    interpolate_end_pose8,
    normalize_quaternion_wxyz,
)
from .franka_serve_bundle import verify_franka_serve_bundle

POLICY_CONFIG_SCHEMA_VERSION = 2
FRANKA_CONTROL_ARM: Final[str] = "right"
_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DEVICE_PATTERN = re.compile(r"^[0-9]+$")


class FrankaActionBackend(Protocol):
    """Backend boundary consumed by the official Policy adapter."""

    def reset(self, *, prompt: str, seed: int) -> None: ...

    def infer(
        self,
        *,
        images: Mapping[str, npt.NDArray[np.uint8]],
        current_ee20: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]: ...

    def commit_executed_chunk(
        self,
        *,
        actions_ee20_cfh: npt.NDArray[np.float32],
        image_history: tuple[Mapping[str, npt.NDArray[np.uint8]], ...],
        action_anchor_ee20: npt.NDArray[np.float32],
    ) -> None: ...


@dataclass(frozen=True)
class FrankaSafetyConfig:
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    max_translation_step_m: float
    max_rotation_step_rad: float
    gripper_min: float
    gripper_max: float
    max_gripper_step: float


@dataclass(frozen=True)
class FrankaPolicyConfig:
    policy_id: str
    serve_bundle: Path
    serve_bundle_receipt_sha256: str
    serve_output: Path
    cuda_visible_device: str
    distributed_port: int
    episode_seed: int
    max_chunk_actions: int
    external_chunk_actions: int
    video_inference_steps: int
    action_inference_steps: int
    safety: FrankaSafetyConfig


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _exact(payload: Mapping[str, object], fields: set[str], *, label: str) -> None:
    if set(payload) != fields:
        raise ValueError(
            f"invalid {label}: missing={sorted(fields - set(payload))}, "
            f"unexpected={sorted(set(payload) - fields)}"
        )


def _positive_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _vector3(value: object, *, label: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain three finite values")
    return tuple(float(item) for item in array)


def load_franka_policy_config(path: Path) -> FrankaPolicyConfig:
    source = Path(path).expanduser().resolve(strict=True)
    payload = _mapping(json.loads(source.read_text(encoding="utf-8")), label="policy")
    fields = {
        "schema_version",
        "policy_id",
        "serve_bundle",
        "serve_bundle_receipt_sha256",
        "serve_output",
        "cuda_visible_device",
        "distributed_port",
        "episode_seed",
        "max_chunk_actions",
        "external_chunk_actions",
        "video_inference_steps",
        "action_inference_steps",
        "safety",
    }
    _exact(payload, fields, label="Franka policy config")
    if payload.get("schema_version") != POLICY_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported Franka policy config schema")
    policy_id = payload.get("policy_id")
    if not isinstance(policy_id, str) or not re.fullmatch(
        r"[A-Za-z0-9._-]{1,64}", policy_id
    ):
        raise ValueError("policy_id contains unsupported characters")
    device = payload.get("cuda_visible_device")
    if not isinstance(device, str) or not _DEVICE_PATTERN.fullmatch(device):
        raise ValueError("cuda_visible_device must name exactly one numeric device")
    receipt_sha = payload.get("serve_bundle_receipt_sha256")
    if not isinstance(receipt_sha, str) or not _SHA_PATTERN.fullmatch(receipt_sha):
        raise ValueError("serve_bundle_receipt_sha256 must be lowercase SHA256")
    port = payload.get("distributed_port")
    seed = payload.get("episode_seed")
    chunk = payload.get("max_chunk_actions")
    external_chunk = payload.get("external_chunk_actions")
    video_steps = payload.get("video_inference_steps")
    action_steps = payload.get("action_inference_steps")
    for value, label, maximum in (
        (port, "distributed_port", 65535),
        (chunk, "max_chunk_actions", 128),
        (external_chunk, "external_chunk_actions", 128),
        (video_steps, "video_inference_steps", 100),
        (action_steps, "action_inference_steps", 100),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise ValueError(f"{label} must be an integer in [1,{maximum}]")
    if chunk != 12:
        raise ValueError(
            "max_chunk_actions must be 12 for frame_chunk_size=2 and "
            "action_per_frame=6"
        )
    if external_chunk != 1:
        raise ValueError(
            "external_chunk_actions must be 1 so every executed action is "
            "followed by a real observation before N0 cache grounding"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("episode_seed must be a non-negative integer")
    safety_raw = _mapping(payload.get("safety"), label="safety")
    _exact(
        safety_raw,
        {
            "workspace_min",
            "workspace_max",
            "max_translation_step_m",
            "max_rotation_step_rad",
            "gripper_min",
            "gripper_max",
            "max_gripper_step",
        },
        label="safety",
    )
    workspace_min = _vector3(safety_raw.get("workspace_min"), label="workspace_min")
    workspace_max = _vector3(safety_raw.get("workspace_max"), label="workspace_max")
    if any(low >= high for low, high in zip(workspace_min, workspace_max, strict=True)):
        raise ValueError("each workspace_min coordinate must be below workspace_max")
    gripper_min = float(safety_raw.get("gripper_min"))
    gripper_max = float(safety_raw.get("gripper_max"))
    if not np.isfinite((gripper_min, gripper_max)).all() or gripper_min >= gripper_max:
        raise ValueError("gripper limits are invalid")
    serve_bundle = (
        (source.parent / str(payload["serve_bundle"])).resolve(strict=True)
        if not Path(str(payload["serve_bundle"])).expanduser().is_absolute()
        else Path(str(payload["serve_bundle"])).expanduser().resolve(strict=True)
    )
    output_raw = Path(str(payload["serve_output"])).expanduser()
    serve_output = (
        output_raw if output_raw.is_absolute() else source.parent / output_raw
    ).resolve(strict=False)
    if (
        serve_output == serve_bundle
        or serve_bundle in serve_output.parents
        or serve_output in serve_bundle.parents
    ):
        raise ValueError("serve_output and immutable serve bundle must be disjoint")
    return FrankaPolicyConfig(
        policy_id=policy_id,
        serve_bundle=serve_bundle,
        serve_bundle_receipt_sha256=receipt_sha,
        serve_output=serve_output,
        cuda_visible_device=device,
        distributed_port=int(port),
        episode_seed=int(seed),
        max_chunk_actions=int(chunk),
        external_chunk_actions=int(external_chunk),
        video_inference_steps=int(video_steps),
        action_inference_steps=int(action_steps),
        safety=FrankaSafetyConfig(
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            max_translation_step_m=_positive_number(
                safety_raw.get("max_translation_step_m"),
                label="max_translation_step_m",
            ),
            max_rotation_step_rad=_positive_number(
                safety_raw.get("max_rotation_step_rad"),
                label="max_rotation_step_rad",
            ),
            gripper_min=gripper_min,
            gripper_max=gripper_max,
            max_gripper_step=_positive_number(
                safety_raw.get("max_gripper_step"), label="max_gripper_step"
            ),
        ),
    )


def franka_policy_config_template() -> dict[str, object]:
    return {
        "schema_version": POLICY_CONFIG_SCHEMA_VERSION,
        "policy_id": "n0-twam-franka",
        "serve_bundle": "./serve-bundle",
        "serve_bundle_receipt_sha256": "REPLACE_WITH_64_LOWERCASE_HEX",
        "serve_output": "./serve-output",
        "cuda_visible_device": "0",
        "distributed_port": 29642,
        "episode_seed": 20260810,
        "max_chunk_actions": 12,
        "external_chunk_actions": 1,
        "video_inference_steps": 3,
        "action_inference_steps": 4,
        "safety": {
            "workspace_min": ["CALIBRATE_X_MIN", "CALIBRATE_Y_MIN", "CALIBRATE_Z_MIN"],
            "workspace_max": ["CALIBRATE_X_MAX", "CALIBRATE_Y_MAX", "CALIBRATE_Z_MAX"],
            "max_translation_step_m": "CALIBRATE_TRANSLATION_STEP",
            "max_rotation_step_rad": "CALIBRATE_ROTATION_STEP",
            "gripper_min": "CALIBRATE_GRIPPER_MIN",
            "gripper_max": "CALIBRATE_GRIPPER_MAX",
            "max_gripper_step": "CALIBRATE_GRIPPER_STEP",
        },
    }


class DirectN0FrankaBackend:
    """In-process N0 server backend used by the official long-poll Policy."""

    def __init__(self, config: FrankaPolicyConfig) -> None:
        receipt = verify_franka_serve_bundle(
            config.serve_bundle,
            expected_receipt_file_sha256=config.serve_bundle_receipt_sha256,
        )
        run_role = receipt.get("run_role")
        if run_role not in {"development", "final_refit"}:
            raise ValueError("serve bundle has an invalid run role")
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_device
        os.environ["HIP_VISIBLE_DEVICES"] = config.cuda_visible_device
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(config.distributed_port)
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["N0_TRACK32_RUN_ROLE"] = str(run_role)
        os.environ["N0_TRACK32_SERVE_BUNDLE"] = str(config.serve_bundle)
        os.environ["N0_TRACK32_SERVE_OUTPUT"] = str(config.serve_output)
        os.environ["N0_TRACK32_NORMALIZER_PATH"] = str(
            config.serve_bundle / "normalizer.json"
        )
        os.environ["N0_TRACK32_VIDEO_STEPS"] = str(config.video_inference_steps)
        os.environ["N0_TRACK32_ACTION_STEPS"] = str(config.action_inference_steps)
        config.serve_output.mkdir(parents=True, exist_ok=True)

        import torch.distributed as dist

        from n0_twam.distributed.util import init_distributed
        from n0_twam.n0_twam_server import TWAM_CONFIGS, TWAM_Server

        if dist.is_initialized():
            if dist.get_world_size() != 1 or dist.get_rank() != 0:
                raise RuntimeError("direct Franka Policy requires a fresh world_size=1")
        else:
            init_distributed(1, 0, 0)
        server_config = TWAM_CONFIGS["track32_franka_server"]
        server_config.rank = 0
        server_config.local_rank = 0
        server_config.world_size = 1
        self._server = TWAM_Server(server_config)
        self._prompt: str | None = None
        self._started = False

    def reset(self, *, prompt: str, seed: int) -> None:
        self._server.infer({"reset": True, "prompt": prompt, "seed": seed})
        self._prompt = prompt
        self._started = True

    def close(self) -> None:
        """Release the single-rank process group after offline evaluation."""

        import torch.distributed as dist

        self._started = False
        self._prompt = None
        if dist.is_initialized():
            dist.destroy_process_group()

    def infer(
        self,
        *,
        images: Mapping[str, npt.NDArray[np.uint8]],
        current_ee20: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        if not self._started:
            raise RuntimeError("backend must be reset before infer")
        observation: dict[str, object] = {
            "obs": [dict(images)],
            "current_state": current_ee20,
            "action_anchor_state": current_ee20,
            "state_action_format": "absolute",
        }
        result = self._server.infer(observation)
        action = result.get("action")
        array = np.asarray(action, dtype=np.float32)
        if array.shape != (20, 2, 6) or not np.isfinite(array).all():
            raise ValueError(f"N0 backend returned invalid EE20 chunk: {array.shape}")
        return np.ascontiguousarray(array)

    def infer_prediction_chunk(
        self,
        *,
        images: Mapping[str, npt.NDArray[np.uint8]],
        current_ee20: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """Return the cold policy chunk and its jointly generated RGB frames.

        This evaluation-only boundary intentionally calls the same private server
        kernel used by :meth:`infer`, while retaining the video latent that the
        public robot Policy does not transmit.  The returned RGB tensor has shape
        ``[T,H,W_total,3]``; camera tiles remain concatenated in config order.
        """

        action_array, latents = self.infer_prediction_latent_chunk(
            images=images,
            current_ee20=current_ee20,
        )
        return np.ascontiguousarray(action_array), self.decode_video_latents(latents)

    def infer_prediction_latent_chunk(
        self,
        *,
        images: Mapping[str, npt.NDArray[np.uint8]],
        current_ee20: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], object]:
        """Return one cold EE20 chunk and the normalized generated latent."""

        if not self._started:
            raise RuntimeError("backend must be reset before infer")
        observation: dict[str, object] = {
            "obs": [dict(images)],
            "current_state": current_ee20,
            "action_anchor_state": current_ee20,
            "state_action_format": "absolute",
        }
        actions, latents = self._server._infer(observation, frame_st_id=0)
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (20, 2, 6) or not np.isfinite(action_array).all():
            raise ValueError(
                f"N0 backend returned invalid EE20 chunk: {action_array.shape}"
            )
        latent_shape = tuple(getattr(latents, "shape", ()))
        if len(latent_shape) != 5 or latent_shape[0] != 1 or latent_shape[2] != 2:
            raise ValueError(
                f"N0 backend returned invalid video latent: {latent_shape}"
            )
        return np.ascontiguousarray(action_array), latents

    def decode_video_latents(self, latents: object) -> npt.NDArray[np.uint8]:
        """Decode one normalized two-camera Wan latent through the bound VAE."""

        decoded = self.decode_video_latent_batch(latents, batch_size=1)
        if decoded.shape[0] != 1:
            raise ValueError("single latent decode returned a batch")
        return np.ascontiguousarray(decoded[0])

    def decode_video_latent_batch(
        self, latents: object, *, batch_size: int
    ) -> npt.NDArray[np.uint8]:
        """Decode normalized Wan latents in bounded HCU batches."""

        from diffusers.video_processor import VideoProcessor
        import torch

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("decode batch size must be a positive integer")
        if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
            raise ValueError("video latents must be a five-dimensional tensor")

        self._server.video_processor = VideoProcessor(vae_scale_factor=1)
        with torch.no_grad():
            chunks = [
                np.asarray(self._server.decode_one_video(chunk, "np"))
                for chunk in latents.split(batch_size)
            ]
        decoded = np.concatenate(chunks, axis=0)
        if decoded.ndim != 5 or decoded.shape[-1] != 3:
            raise ValueError(f"decoded RGB batch has invalid shape: {decoded.shape}")
        if decoded.dtype != np.uint8:
            if not np.isfinite(decoded).all():
                raise ValueError("decoded RGB batch contains non-finite values")
            maximum = float(decoded.max(initial=0.0))
            minimum = float(decoded.min(initial=0.0))
            if minimum < -1e-6 or maximum > 1.0 + 1e-6:
                raise ValueError("decoded RGB batch is outside the [0,1] range")
            decoded = np.rint(np.clip(decoded, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.ascontiguousarray(decoded)

    def commit_executed_chunk(
        self,
        *,
        actions_ee20_cfh: npt.NDArray[np.float32],
        image_history: tuple[Mapping[str, npt.NDArray[np.uint8]], ...],
        action_anchor_ee20: npt.NDArray[np.float32],
    ) -> None:
        if not self._started or self._prompt is None:
            raise RuntimeError("backend must be reset before cache grounding")
        array = np.asarray(actions_ee20_cfh, dtype=np.float32)
        if array.shape != (20, 2, 6) or not np.isfinite(array).all():
            raise ValueError("committed Franka action chunk must have shape [20,2,6]")
        if len(image_history) not in {4, 8}:
            raise ValueError("grounding requires exactly 4 cold or 8 regular images")
        anchor = np.asarray(action_anchor_ee20, dtype=np.float32)
        if anchor.shape != (20,) or not np.isfinite(anchor).all():
            raise ValueError("action anchor must be a finite EE20 vector")
        observations = [
            {
                name: _image(image, label=f"image_history.{name}")
                for name, image in row.items()
            }
            for row in image_history
        ]
        self._server.infer(
            {
                "obs": observations,
                "state": np.ascontiguousarray(array),
                "current_state": np.ascontiguousarray(anchor),
                "action_anchor_state": np.ascontiguousarray(anchor),
                "state_action_format": "absolute",
                "compute_kv_cache": True,
                "imagine": False,
                "prompt": self._prompt,
            }
        )


def _image(value: object, *, label: str) -> npt.NDArray[np.uint8]:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{label} must be uint8 HWC RGB")
    return np.ascontiguousarray(array)


def _subsample_grounding_history(
    history: list[dict[str, npt.NDArray[np.uint8]]],
) -> tuple[dict[str, npt.NDArray[np.uint8]], ...]:
    """Map post-action 15-Hz observations onto the training-time 10-Hz grid."""

    if len(history) not in {6, 12}:
        raise ValueError("grounding history must cover 6 cold or 12 regular actions")
    selected: list[dict[str, npt.NDArray[np.uint8]]] = []
    for start in range(0, len(history), 6):
        selected.extend(history[start + offset] for offset in (1, 2, 3, 5))
    return tuple(selected)


def _angle_wxyz(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    first = normalize_quaternion_wxyz(left).reshape(4)
    second = normalize_quaternion_wxyz(right).reshape(4)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(2.0 * np.arccos(cosine))


def _safe_chunk(
    actions: npt.NDArray[np.float32],
    *,
    current_pose: npt.NDArray[np.float32],
    safety: FrankaSafetyConfig,
) -> npt.NDArray[np.float32]:
    output = np.empty_like(actions)
    previous = np.asarray(current_pose, dtype=np.float32).reshape(8).copy()
    lower = np.asarray(safety.workspace_min, dtype=np.float32)
    upper = np.asarray(safety.workspace_max, dtype=np.float32)
    if np.any(previous[:3] < lower) or np.any(previous[:3] > upper):
        raise ValueError("current Franka pose is outside the signed workspace")
    if not safety.gripper_min <= float(previous[7]) <= safety.gripper_max:
        raise ValueError("current Franka gripper is outside the signed limits")
    for index, candidate in enumerate(actions):
        target = np.asarray(candidate, dtype=np.float32).reshape(8).copy()
        target[:3] = np.clip(target[:3], lower, upper)
        delta = target[:3] - previous[:3]
        distance = float(np.linalg.norm(delta))
        if distance > safety.max_translation_step_m:
            target[:3] = previous[:3] + delta * (
                safety.max_translation_step_m / distance
            )
        angle = _angle_wxyz(previous[3:7], target[3:7])
        if angle > safety.max_rotation_step_rad:
            fraction = safety.max_rotation_step_rad / angle
            target = interpolate_end_pose8(previous, target, fraction).reshape(8)
        target[7] = np.clip(
            target[7],
            previous[7] - safety.max_gripper_step,
            previous[7] + safety.max_gripper_step,
        )
        target[7] = np.clip(target[7], safety.gripper_min, safety.gripper_max)
        target[3:7] = normalize_quaternion_wxyz(target[3:7])
        output[index] = target
        previous = target
    return np.ascontiguousarray(output)


class Policy:
    """WorldArena Track 3.2 Franka policy with an official `(chunk,8)` output."""

    def __init__(
        self,
        config_path: str | None = None,
        *,
        backend: FrankaActionBackend | None = None,
    ) -> None:
        selected = config_path or os.environ.get("N0_TRACK32_POLICY_CONFIG")
        if not selected:
            raise ValueError("config_path or N0_TRACK32_POLICY_CONFIG is required")
        self.config = load_franka_policy_config(Path(selected))
        self._backend = backend or DirectN0FrankaBackend(self.config)
        self._lock = threading.Lock()
        self._episode_index = -1
        self._prompt: str | None = None
        self._backend_reset = False
        self._cold_chunk = True
        self._pending_actions: npt.NDArray[np.float32] | None = None
        self._pending_interventions: npt.NDArray[np.bool_] | None = None
        self._pending_index = 0
        self._pending_commit_chunk: npt.NDArray[np.float32] | None = None
        self._pending_anchor: npt.NDArray[np.float32] | None = None
        self._executed_actions: list[npt.NDArray[np.float32]] = []
        self._image_history: list[dict[str, npt.NDArray[np.uint8]]] = []
        self._awaiting_post_action_observation = False

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._episode_index += 1
            info = {} if reset_info is None else reset_info
            prompt = info.get("prompt")
            self._prompt = str(prompt) if isinstance(prompt, str) and prompt else None
            self._backend_reset = False
            self._cold_chunk = True
            self._pending_actions = None
            self._pending_interventions = None
            self._pending_index = 0
            self._pending_commit_chunk = None
            self._pending_anchor = None
            self._executed_actions = []
            self._image_history = []
            self._awaiting_post_action_observation = False
            if self._prompt is not None:
                self._reset_backend(self._prompt)

    def _reset_backend(self, prompt: str) -> None:
        seed = self.config.episode_seed + max(self._episode_index, 0)
        self._backend.reset(prompt=prompt, seed=seed)
        self._backend_reset = True

    def _record_post_action_observation(
        self, images: Mapping[str, npt.NDArray[np.uint8]]
    ) -> None:
        if not self._awaiting_post_action_observation:
            return
        self._image_history.append(
            {name: np.ascontiguousarray(image.copy()) for name, image in images.items()}
        )
        self._awaiting_post_action_observation = False

    def _commit_finished_chunk(self) -> bool:
        if self._pending_actions is None or self._pending_index < len(
            self._pending_actions
        ):
            return False
        if self._awaiting_post_action_observation:
            raise RuntimeError(
                "final action observation is missing before cache grounding"
            )
        if self._pending_commit_chunk is None or self._pending_anchor is None:
            raise RuntimeError("completed action queue has no grounding provenance")
        if len(self._image_history) != len(self._pending_actions):
            raise RuntimeError(
                "executed action and post-action observation counts differ"
            )
        if len(self._executed_actions) != len(self._pending_actions):
            raise RuntimeError("safe executed action and queued action counts differ")
        frames, horizon = self._pending_commit_chunk.shape[1:]
        start_index = frames * horizon - len(self._executed_actions)
        committed_flat = np.moveaxis(self._pending_commit_chunk, 0, -1).reshape(-1, 20)
        safe_executed = np.stack(self._executed_actions)
        committed_flat[start_index:] = embed_ee10_in_ee20(
            end_pose8_to_ee10(safe_executed)
        )
        committed_chunk = committed_flat.reshape(frames, horizon, 20).transpose(2, 0, 1)
        self._backend.commit_executed_chunk(
            actions_ee20_cfh=np.ascontiguousarray(committed_chunk, dtype=np.float32),
            image_history=_subsample_grounding_history(self._image_history),
            action_anchor_ee20=self._pending_anchor,
        )
        self._pending_actions = None
        self._pending_interventions = None
        self._pending_index = 0
        self._pending_commit_chunk = None
        self._pending_anchor = None
        self._executed_actions = []
        self._image_history = []
        return True

    def _dequeue_action(
        self, *, current_pose: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], bool]:
        if self._pending_actions is None or self._pending_interventions is None:
            raise RuntimeError("no Franka action is queued")
        if self._pending_index >= len(self._pending_actions):
            raise RuntimeError("Franka action queue is exhausted")
        planned = self._pending_actions[self._pending_index].copy()
        action = _safe_chunk(
            planned.reshape(1, 8),
            current_pose=current_pose,
            safety=self.config.safety,
        )[0]
        intervened = bool(
            self._pending_interventions[self._pending_index]
            or np.any(np.abs(action - planned) > 1e-6)
        )
        self._executed_actions.append(
            np.ascontiguousarray(action.copy(), dtype=np.float32)
        )
        self._pending_index += self.config.external_chunk_actions
        self._awaiting_post_action_observation = True
        return action.reshape(1, 8), intervened

    def infer(self, new_obs: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        input_ms = 0.0
        grounding_ms = 0.0
        generation_ms = 0.0
        postprocess_ms = 0.0
        timing_kind = "queue_hit"
        grounded = False
        generated = False
        queue_depth_after = 0
        with self._lock:
            if "tactile" in new_obs:
                raise ValueError("Franka vision-only Policy rejects tactile input")
            images_raw = _mapping(new_obs.get("images"), label="images")
            tactile_image_keys = tuple(
                key for key in images_raw if "tactile" in str(key).lower()
            )
            if tactile_image_keys:
                raise ValueError(
                    "Franka vision-only Policy rejects tactile image input"
                )
            prompt = new_obs.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("Franka observation requires a non-empty prompt")
            if self._prompt is not None and prompt != self._prompt:
                raise ValueError("task prompt changed without Policy.reset()")
            self._prompt = prompt
            if not self._backend_reset:
                self._reset_backend(prompt)
            high = _image(images_raw.get("cam_high"), label="images.cam_high")
            wrist_value = images_raw.get("cam_left_wrist", images_raw.get("cam_wrist"))
            wrist = _image(wrist_value, label="images.cam_left_wrist")
            pose7 = np.asarray(new_obs.get("left_end_pose"), dtype=np.float32)
            qpos = np.asarray(new_obs.get("joint_qpos"), dtype=np.float32)
            if (
                pose7.shape != (7,)
                or qpos.shape != (8,)
                or not np.isfinite(pose7).all()
                or not np.isfinite(qpos).all()
            ):
                raise ValueError("Franka proprioception must be finite pose7 + qpos8")
            current_pose8 = np.concatenate((pose7, qpos[-1:])).astype(np.float32)
            current_ee20 = embed_ee10_in_ee20(end_pose8_to_ee10(current_pose8)).reshape(
                20
            )
            internal_images = {
                "observation.images.top": high,
                "observation.images.wrist_l": wrist,
            }
            input_ms = (time.perf_counter() - started) * 1000.0
            self._record_post_action_observation(internal_images)
            grounding_started = time.perf_counter()
            grounded = self._commit_finished_chunk()
            grounding_ms = (time.perf_counter() - grounding_started) * 1000.0
            postprocess_started = time.perf_counter()
            if self._pending_actions is None:
                generated = True
                timing_kind = (
                    "cold_generation" if self._cold_chunk else "grounding_refill"
                )
                generation_started = time.perf_counter()
                raw = np.asarray(
                    self._backend.infer(
                        images=internal_images,
                        current_ee20=current_ee20,
                    ),
                    dtype=np.float32,
                )
                generation_ms = (time.perf_counter() - generation_started) * 1000.0
                postprocess_started = time.perf_counter()
                if raw.shape != (20, 2, 6) or not np.isfinite(raw).all():
                    raise ValueError("backend action must have shape [20,2,6]")
                frames, horizon = raw.shape[1:]
                flat_ee20 = np.moveaxis(raw, 0, -1).reshape(-1, 20)
                if flat_ee20.shape[0] != self.config.max_chunk_actions:
                    raise ValueError(
                        "backend differs from the fixed 12-action contract"
                    )
                ee10 = extract_ee10_from_ee20(flat_ee20)
                decoded: list[npt.NDArray[np.float32]] = []
                reference = current_pose8[3:7]
                for action in ee10:
                    pose = ee10_to_end_pose8(
                        action, quaternion_reference=reference
                    ).reshape(8)
                    decoded.append(pose)
                    reference = pose[3:7]
                decoded_actions = np.stack(decoded)
                start_index = horizon if self._cold_chunk else 0
                safe_executed = _safe_chunk(
                    decoded_actions[start_index:],
                    current_pose=current_pose8,
                    safety=self.config.safety,
                )
                grounded_actions = decoded_actions.copy()
                grounded_actions[start_index:] = safe_executed
                interventions = np.any(
                    np.abs(grounded_actions - decoded_actions) > 1e-6, axis=1
                )
                executed_ee20 = embed_ee10_in_ee20(end_pose8_to_ee10(grounded_actions))
                executed_cfh = executed_ee20.reshape(frames, horizon, 20).transpose(
                    2, 0, 1
                )
                self._cold_chunk = False
                self._pending_actions = np.ascontiguousarray(
                    grounded_actions[start_index:], dtype=np.float32
                )
                self._pending_interventions = np.ascontiguousarray(
                    interventions[start_index:], dtype=np.bool_
                )
                self._pending_index = 0
                self._pending_commit_chunk = np.ascontiguousarray(
                    executed_cfh, dtype=np.float32
                )
                self._pending_anchor = np.ascontiguousarray(
                    current_ee20.copy(), dtype=np.float32
                )
                self._executed_actions = []
                self._image_history = []
            actions, safety_intervened = self._dequeue_action(
                current_pose=current_pose8
            )
            safety_interventions = int(safety_intervened)
            if self._pending_actions is None:
                raise RuntimeError("Policy dequeued an action without a queue")
            queue_depth_after = len(self._pending_actions) - self._pending_index
            postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "actions": actions.astype(np.float32, copy=False),
            "policy_metadata": {
                "policy_id": self.config.policy_id,
                "platform": "franka",
                "action_format": "end_pose_base",
                "control_arm": FRANKA_CONTROL_ARM,
                "action_dim": 8,
                "chunk_size": self.config.external_chunk_actions,
                "quaternion_order": "wxyz",
                "tactile_profile": "vision_only",
                "tactile_mode": "disabled",
                "task_id": str(new_obs.get("task_id", "")),
                "safety_intervened": safety_interventions > 0,
                "safety_intervention_count": safety_interventions,
            },
            "policy_timing": {
                "kind": timing_kind,
                "infer_ms": elapsed_ms,
                "input_ms": input_ms,
                "grounding_ms": grounding_ms,
                "generation_ms": generation_ms,
                "postprocess_ms": postprocess_ms,
                "grounded": grounded,
                "generated": generated,
                "queue_depth_after": queue_depth_after,
            },
        }


__all__ = (
    "DirectN0FrankaBackend",
    "FRANKA_CONTROL_ARM",
    "FrankaActionBackend",
    "FrankaPolicyConfig",
    "FrankaSafetyConfig",
    "Policy",
    "franka_policy_config_template",
    "load_franka_policy_config",
)
