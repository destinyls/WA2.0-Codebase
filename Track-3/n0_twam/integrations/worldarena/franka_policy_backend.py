# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""In-process N0 backend used by the WorldArena Franka Policy."""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from .franka_policy_config import FrankaPolicyConfig
from .franka_policy_inputs import image, training_aligned_video_history
from .franka_serve_bundle import verify_franka_serve_bundle


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
        precomputed_video_latent: object | None = None,
        training_aligned_video_history: tuple[
            Mapping[str, npt.NDArray[np.uint8]], ...
        ]
        | None = None,
    ) -> npt.NDArray[np.float32]:
        if not self._started:
            raise RuntimeError("backend must be reset before infer")
        observation = self._observation(
            images=images,
            current_ee20=current_ee20,
            precomputed_video_latent=precomputed_video_latent,
            history=training_aligned_video_history,
        )
        result = self._server.infer(observation)
        array = np.asarray(result.get("action"), dtype=np.float32)
        if array.shape != (20, 2, 6) or not np.isfinite(array).all():
            raise ValueError(f"N0 backend returned invalid EE20 chunk: {array.shape}")
        return np.ascontiguousarray(array)

    def infer_prediction_chunk(
        self,
        *,
        images: Mapping[str, npt.NDArray[np.uint8]],
        current_ee20: npt.NDArray[np.float32],
        precomputed_video_latent: object | None = None,
        training_aligned_video_history: tuple[
            Mapping[str, npt.NDArray[np.uint8]], ...
        ]
        | None = None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """Return the cold policy chunk and its jointly generated RGB frames."""

        action_array, latents = self.infer_prediction_latent_chunk(
            images=images,
            current_ee20=current_ee20,
            precomputed_video_latent=precomputed_video_latent,
            training_aligned_video_history=training_aligned_video_history,
        )
        return np.ascontiguousarray(action_array), self.decode_video_latents(latents)

    def infer_prediction_latent_chunk(
        self,
        *,
        images: Mapping[str, npt.NDArray[np.uint8]],
        current_ee20: npt.NDArray[np.float32],
        precomputed_video_latent: object | None = None,
        training_aligned_video_history: tuple[
            Mapping[str, npt.NDArray[np.uint8]], ...
        ]
        | None = None,
    ) -> tuple[npt.NDArray[np.float32], object]:
        """Return one cold EE20 chunk and the normalized generated latent."""

        if not self._started:
            raise RuntimeError("backend must be reset before infer")
        observation = self._observation(
            images=images,
            current_ee20=current_ee20,
            precomputed_video_latent=precomputed_video_latent,
            history=training_aligned_video_history,
        )
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

        import torch
        from diffusers.video_processor import VideoProcessor

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

    def release_video_conditioning_cache(self) -> None:
        """Release streaming-encoder state before memory-heavy video decoding."""

        import torch

        self._server.streaming_vae.clear_cache()
        torch.cuda.empty_cache()

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
                name: image(value, label=f"image_history.{name}")
                for name, value in row.items()
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

    @staticmethod
    def _observation(
        *,
        images: Mapping[str, npt.NDArray[np.uint8]],
        current_ee20: npt.NDArray[np.float32],
        precomputed_video_latent: object | None,
        history: tuple[Mapping[str, npt.NDArray[np.uint8]], ...] | None,
    ) -> dict[str, object]:
        if precomputed_video_latent is not None and history is not None:
            raise ValueError(
                "precomputed latent and training-aligned history are mutually exclusive"
            )
        observation: dict[str, object] = {
            "obs": [dict(images)],
            "current_state": current_ee20,
            "action_anchor_state": current_ee20,
            "state_action_format": "absolute",
        }
        if precomputed_video_latent is not None:
            observation["precomputed_video_latent"] = precomputed_video_latent
        if history is not None:
            observation["training_aligned_video_history"] = (
                training_aligned_video_history(history, current_images=images)
            )
        return observation


__all__ = ("DirectN0FrankaBackend",)
