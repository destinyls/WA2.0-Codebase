# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Frozen Target-10 contract used for the Wan2.2 score comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from n0_twam.evaluation.sealed_artifact_io import read_json_object, sha256_file
from n0_twam.integrations.univtac.dataset_view import (
    DEFAULT_UNIFIED_EVALUATION_EPISODE_COUNT,
    DEFAULT_UNIFIED_EVALUATION_EPISODE_IDS,
    DEFAULT_UNIFIED_EVALUATION_TASKS,
    DatasetView,
)

REFERENCE_CONTRACT_ID = "wan22_target10_reference9_continuous41_v3"
REFERENCE_METADATA_SHA256 = (
    "3f8e8727be72655b02b2e72001e681f360e2c515a9ea7b48e00006f3e52892ce"
)
REFERENCE_METRIC_SHA256 = (
    "33ab83b85dfa32f1495667c0f1b431d6eb604a73707158b1993cff37914d9bc3"
)
TARGET_TASKS = DEFAULT_UNIFIED_EVALUATION_TASKS
TARGET_EPISODE_IDS = DEFAULT_UNIFIED_EVALUATION_EPISODE_IDS
TARGET_RAW_ROWS = tuple(range(0, 41, 5))
CONDITIONING_RAW_ROWS = (0,)
FUTURE_FRAME_INDICES = tuple(range(1, 9))
MODEL_OUTPUT_FRAME_INDICES = tuple(range(41))


@dataclass(frozen=True)
class Target10ReferenceContract:
    """Immutable metric and output-adapter definition."""

    contract_id: str = REFERENCE_CONTRACT_ID
    target_tasks: tuple[str, ...] = TARGET_TASKS
    target_episode_ids: tuple[int, ...] = TARGET_EPISODE_IDS
    conditioning_raw_rows: tuple[int, ...] = CONDITIONING_RAW_ROWS
    target_raw_rows: tuple[int, ...] = TARGET_RAW_ROWS
    temporal_binding: str = "continuous41_then_stride5_selection_v1"
    training_max_latent_frames: int = 5
    training_decoded_frame_count: int = 17
    model_latent_frames: int = 11
    model_decoded_frame_count: int = 41
    selected_output_indices: tuple[int, ...] = TARGET_RAW_ROWS
    decoded_frame_count: int = 9
    sequence_length_status: str = "evaluation_horizon_ood_vs_training17"
    future_frame_indices: tuple[int, ...] = FUTURE_FRAME_INDICES
    native_sensor_shape: tuple[int, int, int] = (128, 128, 3)
    reference_sensor_shape: tuple[int, int, int] = (192, 256, 3)
    reference_combined_shape: tuple[int, int, int] = (192, 512, 3)
    sensor_order: tuple[str, ...] = ("left_gsmini", "right_gsmini")
    reconstruction: str = "clip_round(frame0_128_plus_255x_signed_residual)"
    prediction_resize: str = "pil_bilinear_128x128_to_192x256_v1"
    ground_truth_resize: str = "pil_bilinear_raw_to_192x256_v1"
    metric_domain: str = "decoded_mp4_rgb_uint8"
    writer_profile: str = "imageio_v3_pyav_libx264_yuv420p_v1"
    fps: int = 2
    sampling_steps: int = 50
    evaluation_seed: int = 2026
    sampler: str = "n0_flowmatch_euler_autoregressive_v1"
    published_reference_sampler: str = "unipc_shift5_v1"
    ssim_backend: str = "numpy_scipy_gaussian_ssim_v1"
    psnr_cap_db: float = 100.0
    aggregation: str = "frame_episode_task_two_task_macro_v1"
    expected_episode_count: int = DEFAULT_UNIFIED_EVALUATION_EPISODE_COUNT
    expected_future_frame_pairs: int = 80
    published_psnr_display: str = "21.26"
    published_ssim_display: str = "0.746"
    organizer_contract_confirmed: bool = False

    def to_json_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible contract."""

        return asdict(self)

    @property
    def sha256(self) -> str:
        """Return the content identity of the contract itself."""

        payload = json.dumps(
            self.to_json_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


REFERENCE_CONTRACT = Target10ReferenceContract()


def _safe_metadata_entry(value: object) -> tuple[str, str, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "hdf5_path",
        "task",
        "ep_idx",
    }:
        raise ValueError("reference metadata entry has an invalid schema")
    path_value = value.get("hdf5_path")
    task_value = value.get("task")
    episode_value = value.get("ep_idx")
    if not isinstance(path_value, str) or not isinstance(task_value, str):
        raise ValueError("reference metadata path/task must be strings")
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts or path.suffix not in {".h5", ".hdf5"}:
        raise ValueError("reference metadata contains an unsafe HDF5 path")
    if isinstance(episode_value, bool) or not isinstance(episode_value, int):
        raise ValueError("reference metadata ep_idx must be an integer")
    return path.as_posix(), task_value, episode_value


def load_target10_reference_roster(
    metadata_path: Path,
) -> tuple[tuple[str, str, int], ...]:
    """Load the pinned metadata and return its ordered Target-10 projection."""

    path = Path(metadata_path).resolve(strict=True)
    if not path.is_file() or sha256_file(path) != REFERENCE_METADATA_SHA256:
        raise ValueError("reference metadata identity is not approved")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("reference metadata is not valid JSON") from exc
    if not isinstance(raw, list):
        raise ValueError("reference metadata must contain a list")
    complete = tuple(_safe_metadata_entry(item) for item in raw)
    selected = tuple(item for item in complete if item[1] in TARGET_TASKS)
    expected = tuple(
        (f"{task}/clean/{episode_id}.hdf5", task, episode_id)
        for task in TARGET_TASKS
        for episode_id in TARGET_EPISODE_IDS
    )
    if selected != expected:
        raise ValueError("reference metadata Target-10 roster is not canonical")
    return selected


def validate_view_against_reference_roster(
    view: DatasetView,
    reference_roster: Sequence[tuple[str, str, int]],
) -> tuple[dict[str, object], ...]:
    """Bind the N0 frozen view to the independent reference roster."""

    actual = tuple(
        (entry.relative_path, entry.task, entry.source_episode_id)
        for entry in view.entries
    )
    expected = tuple(reference_roster)
    if actual != expected:
        raise ValueError("frozen Target-10 view differs from reference metadata roster")
    return tuple(
        {
            "relative_path": entry.relative_path,
            "task": entry.task,
            "episode_id": entry.source_episode_id,
            "lerobot_episode_id": entry.lerobot_episode_id,
        }
        for entry in view.entries
    )


def validate_reference_metric(path: Path) -> Path:
    """Require the exact fixed-backend Stage-1 metric implementation."""

    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file() or sha256_file(resolved) != REFERENCE_METRIC_SHA256:
        raise ValueError("reference metric implementation identity is not approved")
    return resolved


def write_contract(path: Path) -> None:
    """Write the canonical contract for operator inspection."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            REFERENCE_CONTRACT.to_json_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def verify_contract_file(path: Path) -> None:
    """Verify a persisted contract without accepting operator overrides."""

    payload = read_json_object(Path(path).resolve(strict=True))
    if payload != REFERENCE_CONTRACT.to_json_dict():
        raise ValueError("persisted Target-10 reference contract has drifted")


__all__ = (
    "CONDITIONING_RAW_ROWS",
    "FUTURE_FRAME_INDICES",
    "MODEL_OUTPUT_FRAME_INDICES",
    "REFERENCE_CONTRACT",
    "REFERENCE_CONTRACT_ID",
    "REFERENCE_METADATA_SHA256",
    "REFERENCE_METRIC_SHA256",
    "TARGET_EPISODE_IDS",
    "TARGET_RAW_ROWS",
    "TARGET_TASKS",
    "Target10ReferenceContract",
    "load_target10_reference_roster",
    "validate_reference_metric",
    "validate_view_against_reference_roster",
    "verify_contract_file",
    "write_contract",
)
