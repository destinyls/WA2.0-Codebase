# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Local single-rank TWAM backend for AgileX qpos14 full replanning."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from n0_twam.embodiments import (
    AGILEX_RGB_KEYS,
)
from .agilex_manifest import canonical_sha256
from .agilex_policy_artifacts import verify_agilex_policy_artifacts
from .agilex_policy_contracts import RGB_KEYS
from .agilex_policy_io import AgileXDirectPolicyConfig
from .agilex_sync_grounding import (
    AGILEX_CHUNK_ACTIONS,
    build_grounding_observation,
    rgb_images as _images,
    tactile_images as _tactile,
    wrench_values as _wrench,
)

FloatArray: TypeAlias = npt.NDArray[np.float32]
ImageArray: TypeAlias = npt.NDArray[np.uint8]


class _Server(Protocol):
    video_processor: object

    def infer(self, observation: dict[str, object]) -> Mapping[str, object]:
        pass

    def decode_one_video(self, latents: object, output_type: str) -> object:
        pass


ArtifactVerifier = Callable[[AgileXDirectPolicyConfig], Mapping[str, object]]
ServerFactory = Callable[[AgileXDirectPolicyConfig], _Server]


def _flatten_qpos14_chunk(
    value: object,
    *,
    max_chunk_actions: int,
) -> FloatArray:
    """Convert the model's ``[C,F,H]`` grid to one external action chunk."""

    raw = np.asarray(value)
    if (
        raw.dtype != np.float32
        or raw.ndim != 3
        or raw.shape[0] != 14
        or raw.shape[1] < 1
        or raw.shape[2] < 1
        or not np.isfinite(raw).all()
    ):
        raise ValueError(
            f"N0 backend returned invalid qpos14 chunk: {raw.shape} {raw.dtype}"
        )
    flattened = np.moveaxis(raw, 0, -1).reshape(-1, 14)
    if len(flattened) < max_chunk_actions:
        raise ValueError(
            "N0 backend returned fewer actions than the signed action prefix"
        )
    return cast(
        FloatArray,
        np.ascontiguousarray(flattened[:max_chunk_actions], dtype=np.float32),
    )


def _sealed_task_routes(config: AgileXDirectPolicyConfig) -> dict[str, object]:
    routes: dict[str, object] = {}
    for task_id, route in config.policy.task_routes.items():
        routes[task_id] = {
            "task_id": route.task_id,
            "prompt": route.prompt,
            "tactile_required": route.tactile_required,
            "wrench_required": route.wrench_required,
            "tactile_keys": list(route.tactile_keys),
            "wrench_keys": list(route.wrench_keys),
            "contract_sha256": route.contract_sha256,
        }
    if canonical_sha256(routes) != config.task_routes_sha256:
        raise ValueError("AgileX task routes differ from the sealed policy identity")
    return routes


def _configure_environment(config: AgileXDirectPolicyConfig) -> None:
    values = {
        "CUDA_VISIBLE_DEVICES": config.cuda_visible_device,
        "HIP_VISIBLE_DEVICES": config.cuda_visible_device,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(config.distributed_port),
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "N0_TRACK3_AGILEX_TACTILE_PROFILE": config.policy.tactile_profile,
        "N0_TRACK3_AGILEX_SERVE_BUNDLE": str(config.serve_bundle),
        "N0_TRACK3_AGILEX_SERVE_OUTPUT": str(config.serve_output),
        "N0_TRACK3_AGILEX_NORMALIZER": str(config.serve_bundle / "normalizer.json"),
        "N0_TRACK3_AGILEX_REPO_ROUTE_MANIFEST": str(
            config.serve_bundle / "repo_route_manifest.json"
        ),
        "N0_TRACK3_AGILEX_VIDEO_STEPS": str(config.video_inference_steps),
        "N0_TRACK3_AGILEX_ACTION_STEPS": str(config.action_inference_steps),
        "N0_TRACK3_AGILEX_SIGNED_TASK_ROUTE": str(config.source_path),
        "N0_TRACK3_AGILEX_SIGNED_TASK_ROUTE_SHA256": (config.policy_config_file_sha256),
    }
    os.environ.update(values)


def _default_server_factory(config: AgileXDirectPolicyConfig) -> _Server:
    """Allocate the production TWAM server only after all identity gates pass."""

    import torch.distributed as dist

    from n0_twam.distributed.util import init_distributed
    from n0_twam.n0_twam_server import TWAM_CONFIGS, TWAM_Server

    if dist.is_initialized():
        if dist.get_world_size() != 1 or dist.get_rank() != 0:
            raise RuntimeError("Direct AgileX backend requires a fresh single rank")
    else:
        init_distributed(1, 0, 0)
    job_config = deepcopy(TWAM_CONFIGS["track3_agilex_server"])
    job_config.rank = 0
    job_config.local_rank = 0
    job_config.world_size = 1
    job_config.wan22_pretrained_model_name_or_path = str(config.serve_bundle)
    job_config.save_root = str(config.serve_output)
    job_config.norm_stat_path = str(config.serve_bundle / "normalizer.json")
    job_config.norm_stat = {
        "q01": list(config.normalizer.q01),
        "q99": list(config.normalizer.q99),
    }
    job_config.normalizer_sha256 = config.normalizer_file_sha256
    job_config.repo_route_manifest_sha256 = config.repo_route_manifest_sha256
    job_config.repo_route_manifest_source_file_sha256 = (
        config.repo_route_manifest_file_sha256
    )
    job_config.require_signed_task_route = True
    job_config.allow_dynamic_signed_task_routes = True
    job_config.signed_task_route_manifest_path = str(config.source_path)
    job_config.signed_task_route_manifest_sha256 = config.policy_config_file_sha256
    job_config.signed_task_route_contract_sha256 = config.task_routes_sha256
    job_config.signed_task_routes = _sealed_task_routes(config)
    job_config.num_inference_steps = config.video_inference_steps
    job_config.action_num_inference_steps = config.action_inference_steps
    if (
        job_config.action_schema != "qpos14_joint_absolute_v1"
        or list(job_config.server_return_action_channel_ids) != list(range(14))
        or job_config.tactile_profile != config.policy.tactile_profile
    ):
        raise ValueError("track3_agilex_server config differs from policy identity")
    return cast(_Server, TWAM_Server(job_config))


class DirectN0AgileXBackend:
    """In-process qpos14 backend with executed-history KV grounding."""

    def __init__(
        self,
        config: AgileXDirectPolicyConfig,
        *,
        server_factory: ServerFactory | None = None,
        artifact_verifier: ArtifactVerifier = verify_agilex_policy_artifacts,
    ) -> None:
        if config.policy.max_chunk_actions != AGILEX_CHUNK_ACTIONS:
            raise ValueError("AgileX max_chunk_actions must be exactly 12")
        artifact_verifier(config)
        Path(config.serve_output).mkdir(parents=True, exist_ok=True)
        self.config = config
        self._default_factory = server_factory is None
        if self._default_factory:
            _configure_environment(config)
            self._server = _default_server_factory(config)
        else:
            assert server_factory is not None
            self._server = server_factory(config)
        self._active_task_id: str | None = None
        self._active_prompt: str | None = None

    def reset(self, *, task_id: str, prompt: str, seed: int, profile: str) -> None:
        if profile != self.config.policy.tactile_profile:
            raise ValueError("reset profile differs from the bound tactile profile")
        if task_id not in self.config.policy.task_routes:
            raise ValueError("reset task_id is not present in the bound task routes")
        route = self.config.policy.task_routes[task_id]
        if prompt != route.prompt:
            raise ValueError("reset prompt differs from the bound task route")
        if type(seed) is not int or seed < 0:
            raise ValueError("reset seed must be a non-negative integer")
        self._server.infer(
            {
                "reset": True,
                "task_id": task_id,
                "prompt": prompt,
                "seed": seed,
                "tactile_profile": profile,
                "tactile_keys": list(route.tactile_keys),
                "wrench_keys": list(route.wrench_keys),
            }
        )
        self._active_task_id = task_id
        self._active_prompt = prompt

    def close(self) -> None:
        """Release a process group owned by the production default factory."""

        self._active_task_id = None
        self._active_prompt = None
        close = getattr(self._server, "close", None)
        if callable(close):
            close()
        if self._default_factory:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()

    def infer(
        self,
        *,
        images: Mapping[str, ImageArray],
        current_qpos14: FloatArray,
        tactile_images: Mapping[str, ImageArray] | None,
        wrench: Mapping[str, FloatArray] | None,
    ) -> FloatArray:
        if self._active_task_id is None:
            raise RuntimeError("backend must be reset before infer")
        route = self.config.policy.task_routes[self._active_task_id]
        qpos = np.asarray(current_qpos14)
        if (
            qpos.dtype != np.float32
            or qpos.shape != (14,)
            or not np.isfinite(qpos).all()
        ):
            raise ValueError("current_qpos14 must be finite float32[14]")
        rgb = _images(images)
        tactile = _tactile(tactile_images, keys=route.tactile_keys)
        force = _wrench(wrench, keys=route.wrench_keys)
        if route.tactile_required != (tactile is not None):
            raise ValueError("tactile presence differs from the task route")
        if route.wrench_required != (force is not None):
            raise ValueError("wrench presence differs from the task route")
        server_images = {
            full_key: rgb[wire_key]
            for wire_key, full_key in zip(RGB_KEYS, AGILEX_RGB_KEYS, strict=True)
        }
        observation: dict[str, object] = {
            "obs": [server_images],
            "current_state": np.ascontiguousarray(qpos),
            "action_anchor_state": np.ascontiguousarray(qpos),
            "state_action_format": "absolute",
            "compute_kv_cache": False,
            "full_replan": True,
            "tactile_keys": list(route.tactile_keys),
            "wrench_keys": list(route.wrench_keys),
        }
        if tactile is not None:
            observation["tactile"] = tactile
        if force is not None:
            observation["wrench"] = force
            observation["wrench_available_mask"] = {
                key: True for key in route.wrench_keys
            }
        result = self._server.infer(observation)
        return _flatten_qpos14_chunk(
            result.get("action"),
            max_chunk_actions=self.config.policy.max_chunk_actions,
        )

    def commit_executed_chunk(
        self,
        *,
        actions_qpos14_cfh: FloatArray,
        image_history: tuple[Mapping[str, ImageArray], ...],
        tactile_history: tuple[Mapping[str, ImageArray], ...] | None,
        wrench_history: tuple[Mapping[str, FloatArray], ...] | None,
        action_anchor_qpos14: FloatArray,
    ) -> None:
        """Ground one real 12-slot qpos14 frame and its four RGB keyframes."""

        if self._active_task_id is None or self._active_prompt is None:
            raise RuntimeError("backend must be reset before cache grounding")
        route = self.config.policy.task_routes[self._active_task_id]
        self._server.infer(
            build_grounding_observation(
                actions_qpos14_cfh=actions_qpos14_cfh,
                image_history=image_history,
                tactile_history=tactile_history,
                wrench_history=wrench_history,
                action_anchor_qpos14=action_anchor_qpos14,
                route=route,
                profile=self.config.policy.tactile_profile,
                prompt=self._active_prompt,
            )
        )

    def infer_prediction_latent_chunk(
        self,
        *,
        images: Mapping[str, ImageArray],
        current_qpos14: FloatArray,
        tactile_images: Mapping[str, ImageArray] | None,
        wrench: Mapping[str, FloatArray] | None,
    ) -> tuple[FloatArray, object]:
        """Return one real cold replan plus its jointly generated RGB latent."""

        observation = self._prediction_observation(
            images=images,
            current_qpos14=current_qpos14,
            tactile_images=tactile_images,
            wrench=wrench,
        )
        kernel = getattr(self._server, "_infer", None)
        if not callable(kernel):
            raise RuntimeError("server does not expose the offline prediction kernel")
        actions, latents = kernel(observation, frame_st_id=0)
        flattened = _flatten_qpos14_chunk(
            actions,
            max_chunk_actions=self.config.policy.max_chunk_actions,
        )
        shape = tuple(getattr(latents, "shape", ()))
        if len(shape) != 5 or shape[0] != 1 or shape[2] < 2:
            raise ValueError(f"N0 backend returned an invalid RGB latent: {shape}")
        return flattened, latents

    def _prediction_observation(
        self,
        *,
        images: Mapping[str, ImageArray],
        current_qpos14: FloatArray,
        tactile_images: Mapping[str, ImageArray] | None,
        wrench: Mapping[str, FloatArray] | None,
    ) -> dict[str, object]:
        """Validate modalities and build the private joint-prediction request."""

        if self._active_task_id is None:
            raise RuntimeError("backend must be reset before infer")
        from .agilex_backend_prediction import prediction_observation

        return prediction_observation(
            route=self.config.policy.task_routes[self._active_task_id],
            images=images,
            current_qpos14=current_qpos14,
            tactile_images=tactile_images,
            wrench=wrench,
        )

    def decode_video_latent_batch(
        self,
        latents: object,
        *,
        batch_size: int,
        spatial_tiles: int = 1,
    ) -> ImageArray:
        """Decode normalized three-camera Wan latents in bounded HCU batches."""

        from .agilex_backend_prediction import decode_video_latent_batch

        return decode_video_latent_batch(
            self._server,
            latents,
            batch_size=batch_size,
            spatial_tiles=spatial_tiles,
        )


__all__ = ("DirectN0AgileXBackend",)
