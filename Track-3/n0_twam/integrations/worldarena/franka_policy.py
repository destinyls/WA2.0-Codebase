# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Official WorldArena Franka ``Policy`` adapter for N0-TWAM."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

from .franka_actions import (
    DERIVED_ACTION_SCHEMA,
    FRANKA_ACTION_SCHEMA,
    FRANKA_QUATERNION_ORDER,
    ee10_to_end_pose8,
    embed_ee10_in_ee20,
    end_pose8_to_ee10,
    extract_ee10_from_ee20,
    interpolate_end_pose8,
    normalize_quaternion_xyzw,
)
from .franka_live_contract import (
    TRAINING_ACTIONS_PER_REPLAN,
    validate_live_observation,
)
from .franka_policy_backend import DirectN0FrankaBackend
from .franka_policy_config import (
    LEGACY_POLICY_CONFIG_SCHEMA_VERSION,
    POLICY_CONFIG_SCHEMA_VERSION,
    TRAINING_ALIGNED_POLICY_CONFIG_SCHEMA_VERSION,
    FrankaPolicyConfig,
    FrankaSafetyConfig,
    franka_policy_config_template,
    load_franka_policy_config,
    mapping as _mapping,
)
from .franka_policy_inputs import (
    image as _image,
    training_aligned_video_history as _training_aligned_video_history,
)

FRANKA_CONTROL_ARM: Final[str] = "right"


class FrankaActionBackend(Protocol):
    """Backend boundary consumed by the official Policy adapter."""

    def reset(self, *, prompt: str, seed: int) -> None: ...

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
    ) -> npt.NDArray[np.float32]: ...

    def commit_executed_chunk(
        self,
        *,
        actions_ee20_cfh: npt.NDArray[np.float32],
        image_history: tuple[Mapping[str, npt.NDArray[np.uint8]], ...],
        action_anchor_ee20: npt.NDArray[np.float32],
    ) -> None: ...


def _policy_training_aligned_video_history(
    value: object,
    *,
    current_images: Mapping[str, npt.NDArray[np.uint8]],
) -> tuple[dict[str, npt.NDArray[np.uint8]], ...]:
    """Map a public Policy RGB prefix onto the model's canonical camera keys."""

    if not isinstance(value, (list, tuple)):
        raise ValueError("training_aligned_video_history must be a sequence")
    mapped: list[dict[str, npt.NDArray[np.uint8]]] = []
    for index, frame_value in enumerate(value):
        frame = _mapping(
            frame_value,
            label=f"training_aligned_video_history[{index}]",
        )
        high = _image(
            frame.get("cam_high"),
            label=f"training_aligned_video_history[{index}].cam_high",
        )
        wrist_value = frame.get("cam_left_wrist", frame.get("cam_wrist"))
        wrist = _image(
            wrist_value,
            label=f"training_aligned_video_history[{index}].cam_left_wrist",
        )
        mapped.append(
            {
                "observation.images.top": high,
                "observation.images.wrist_l": wrist,
            }
        )
    validated = _training_aligned_video_history(
        tuple(mapped),
        current_images=current_images,
    )
    return tuple(validated)


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


def _angle_xyzw(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    first = normalize_quaternion_xyzw(left).reshape(4)
    second = normalize_quaternion_xyzw(right).reshape(4)
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
        angle = _angle_xyzw(previous[3:7], target[3:7])
        if angle > safety.max_rotation_step_rad:
            fraction = safety.max_rotation_step_rad / angle
            target = interpolate_end_pose8(previous, target, fraction).reshape(8)
        target[7] = np.clip(
            target[7],
            previous[7] - safety.max_gripper_step,
            previous[7] + safety.max_gripper_step,
        )
        target[7] = np.clip(target[7], safety.gripper_min, safety.gripper_max)
        target[3:7] = normalize_quaternion_xyzw(target[3:7])
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
        self._pending_conditioning_source: str | None = None
        self._executed_actions: list[npt.NDArray[np.float32]] = []
        self._image_history: list[dict[str, npt.NDArray[np.uint8]]] = []
        self._awaiting_post_action_observation = False
        self._strict_plan_index = 0
        self._strict_last_observation_sequence_id: int | None = None
        self._strict_last_image_timestamp_ns: int | None = None

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
            self._pending_conditioning_source = None
            self._executed_actions = []
            self._image_history = []
            self._awaiting_post_action_observation = False
            self._strict_plan_index = 0
            self._strict_last_observation_sequence_id = None
            self._strict_last_image_timestamp_ns = None
            if self._prompt is not None and self.config.live_contract is None:
                self._reset_backend(self._prompt)

    def _reset_backend(self, prompt: str, *, plan_index: int | None = None) -> int:
        seed = self.config.episode_seed + max(self._episode_index, 0) * 1_000_000
        if plan_index is not None:
            seed += plan_index
        self._backend.reset(prompt=prompt, seed=seed)
        self._backend_reset = True
        return seed

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
        live_contract_report: Mapping[str, object] | None = None
        plan_seed: int | None = None
        selected_prediction_index: int | None = None
        selected_prediction_end: int | None = None
        discarded_prediction_count = 0
        discarded_future_prediction_count = 0
        action_selection = str(
            new_obs.get("evaluation_action_selection", "first_future_action")
        )
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
            precomputed_video_latent = new_obs.get("precomputed_video_latent")
            history_value = new_obs.get("training_aligned_video_history")
            if precomputed_video_latent is not None and history_value is not None:
                raise ValueError(
                    "precomputed latent and training-aligned history are mutually exclusive"
                )
            strict_live = self.config.live_contract is not None
            queue_will_refill = strict_live or self._pending_actions is None or (
                self._pending_index >= len(self._pending_actions)
            )
            if (
                precomputed_video_latent is not None or history_value is not None
            ) and not queue_will_refill and not (
                strict_live and history_value is not None
            ):
                raise ValueError(
                    "explicit Policy conditioning is only valid when a new plan "
                    "will be generated"
                )
            training_aligned_video_history = (
                None
                if history_value is None
                else _policy_training_aligned_video_history(
                    history_value,
                    current_images=internal_images,
                )
            )
            if action_selection not in {
                "first_future_action",
                "last_future_action",
            }:
                raise ValueError("unsupported evaluation_action_selection")
            if action_selection != "first_future_action":
                if strict_live and self.config.external_chunk_actions == 6:
                    raise ValueError(
                        "future-six live execution requires first_future_action"
                    )
                if not strict_live:
                    raise ValueError(
                        "offline action selection requires the strict Policy path"
                    )
                if new_obs.get("evaluation_mode") not in {
                    "offline_action_selection",
                    "cached_latent_reference",
                }:
                    raise ValueError(
                        "last-future-action selection is evaluation-only"
                    )
            if strict_live and precomputed_video_latent is None:
                if training_aligned_video_history is None:
                    raise ValueError(
                        "strict live Policy requires training-aligned RGB history"
                    )
                validated_live = validate_live_observation(
                    new_obs,
                    config=self.config.live_contract,
                    history=training_aligned_video_history,
                    current_state=current_pose8,
                    previous_sequence_id=(
                        self._strict_last_observation_sequence_id
                    ),
                    previous_image_timestamp_ns=(
                        self._strict_last_image_timestamp_ns
                    ),
                )
                self._strict_last_observation_sequence_id = (
                    validated_live.observation_sequence_id
                )
                self._strict_last_image_timestamp_ns = (
                    validated_live.image_timestamp_ns
                )
                live_contract_report = validated_live.report
            elif strict_live:
                if new_obs.get("evaluation_mode") != "cached_latent_reference":
                    raise ValueError(
                        "strict live Policy permits cached latent only for the "
                        "explicit offline reference path"
                    )
                live_contract_report = {
                    "status": "offline_reference",
                    "conditioning_mode": "precomputed_video_latent",
                }
            if strict_live:
                plan_seed = self._reset_backend(
                    prompt, plan_index=self._strict_plan_index
                )
            elif not self._backend_reset:
                plan_seed = self._reset_backend(prompt)
            input_ms = (time.perf_counter() - started) * 1000.0
            if not strict_live:
                self._record_post_action_observation(internal_images)
                grounding_started = time.perf_counter()
                grounded = self._commit_finished_chunk()
                grounding_ms = (time.perf_counter() - grounding_started) * 1000.0
            postprocess_started = time.perf_counter()
            if self._pending_actions is None:
                generated = True
                timing_kind = (
                    "strict_live_generation"
                    if strict_live
                    else (
                        "cold_generation" if self._cold_chunk else "grounding_refill"
                    )
                )
                generation_started = time.perf_counter()
                raw = np.asarray(
                    self._backend.infer(
                        images=internal_images,
                        current_ee20=current_ee20,
                        precomputed_video_latent=precomputed_video_latent,
                        training_aligned_video_history=(
                            training_aligned_video_history
                        ),
                    ),
                    dtype=np.float32,
                )
                self._pending_conditioning_source = (
                    "precomputed_video_latent"
                    if precomputed_video_latent is not None
                    else (
                        "training_aligned_raw_rgb"
                        if training_aligned_video_history is not None
                        else "live_streaming_rgb"
                    )
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
                start_index = horizon if strict_live or self._cold_chunk else 0
                if strict_live and self.config.live_contract is not None:
                    start_index = self.config.live_contract.future_start_index
                if strict_live and action_selection == "last_future_action":
                    start_index = len(decoded_actions) - 1
                if strict_live and self.config.external_chunk_actions == 1:
                    selected_prediction_index = start_index
                    selected_prediction_end = start_index + 1
                    planned = decoded_actions[start_index].reshape(1, 8)
                    actions = _safe_chunk(
                        planned,
                        current_pose=current_pose8,
                        safety=self.config.safety,
                    )
                    safety_interventions = int(
                        np.any(np.abs(actions[0] - planned[0]) > 1e-6)
                    )
                    discarded_prediction_count = len(decoded_actions) - 1
                    discarded_future_prediction_count = max(
                        0, len(decoded_actions) - selected_prediction_end
                    )
                    queue_depth_after = 0
                    self._strict_plan_index += 1
                    postprocess_ms = (
                        time.perf_counter() - postprocess_started
                    ) * 1000.0
                elif strict_live:
                    if self.config.live_contract is None:
                        raise RuntimeError("strict Policy has no live contract")
                    selected_prediction_index = (
                        self.config.live_contract.future_start_index
                    )
                    selected_prediction_end = (
                        selected_prediction_index
                        + self.config.live_contract.actions_per_replan
                    )
                    planned = decoded_actions[
                        selected_prediction_index:selected_prediction_end
                    ].copy()
                    if len(planned) != TRAINING_ACTIONS_PER_REPLAN:
                        raise ValueError(
                            "backend does not contain the signed six future actions"
                        )
                    actions = _safe_chunk(
                        planned,
                        current_pose=current_pose8,
                        safety=self.config.safety,
                    )
                    safety_interventions = int(
                        np.count_nonzero(
                            np.any(np.abs(actions - planned) > 1e-6, axis=1)
                        )
                    )
                    discarded_prediction_count = len(decoded_actions) - len(actions)
                    discarded_future_prediction_count = max(
                        0, len(decoded_actions) - selected_prediction_end
                    )
                    queue_depth_after = 0
                    self._strict_plan_index += 1
                    postprocess_ms = (
                        time.perf_counter() - postprocess_started
                    ) * 1000.0
                else:
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
                    executed_ee20 = embed_ee10_in_ee20(
                        end_pose8_to_ee10(grounded_actions)
                    )
                    executed_cfh = executed_ee20.reshape(
                        frames, horizon, 20
                    ).transpose(2, 0, 1)
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
            if not strict_live:
                actions, safety_intervened = self._dequeue_action(
                    current_pose=current_pose8
                )
                safety_interventions = int(safety_intervened)
                if self._pending_actions is None:
                    raise RuntimeError("Policy dequeued an action without a queue")
                queue_depth_after = len(self._pending_actions) - self._pending_index
                postprocess_ms = (
                    time.perf_counter() - postprocess_started
                ) * 1000.0
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
                "model_gripper_unit": "width_m",
                "quaternion_order": FRANKA_QUATERNION_ORDER,
                "wire_action_schema": FRANKA_ACTION_SCHEMA,
                "derived_action_schema": DERIVED_ACTION_SCHEMA,
                "tactile_profile": "vision_only",
                "tactile_mode": "disabled",
                "conditioning_source": self._pending_conditioning_source,
                "execution_mode": (
                    "future6_then_fresh_replan"
                    if self.config.live_contract is not None
                    and self.config.external_chunk_actions == 6
                    else (
                        "replan_each_observation"
                        if self.config.live_contract is not None
                        else "grounded_action_queue"
                    )
                ),
                "live_contract": live_contract_report,
                "plan_seed": plan_seed,
                "selected_prediction_index": selected_prediction_index,
                "selected_prediction_end": selected_prediction_end,
                "selected_prediction_range": (
                    None
                    if selected_prediction_index is None
                    or selected_prediction_end is None
                    else [selected_prediction_index, selected_prediction_end]
                ),
                "discarded_prediction_count": discarded_prediction_count,
                "discarded_future_prediction_count": (
                    discarded_future_prediction_count
                ),
                "model_output_actions": self.config.max_chunk_actions,
                "conditioning_slots_skipped": (
                    self.config.live_contract.future_start_index
                    if self.config.live_contract is not None
                    else 0
                ),
                "returned_future_actions": len(actions),
                "requested_action_hz": (
                    self.config.live_contract.action_hz
                    if self.config.live_contract is not None
                    else None
                ),
                "expected_execution_duration_ms": (
                    1000.0
                    * len(actions)
                    / self.config.live_contract.action_hz
                    if self.config.live_contract is not None
                    else None
                ),
                "requires_fresh_observation_after_chunk": (
                    self.config.live_contract.require_fresh_observation_after_chunk
                    if self.config.live_contract is not None
                    else False
                ),
                "action_selection": action_selection,
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
