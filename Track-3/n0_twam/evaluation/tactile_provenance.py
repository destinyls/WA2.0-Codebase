# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Deterministic provenance contract for offline tactile-quality evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Sized
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from n0_twam.checkpointing.identity import validate_sha256
from n0_twam.data.encoder_source_identity import build_encoder_source_identity
from n0_twam.data.latent_inventory import validate_latent_inventory_pair
from n0_twam.evaluation.tactile_checkpoint import validate_checkpoint_provenance
from n0_twam.evaluation.tactile_quality import (
    INTERNAL_OFFLINE_PROTOCOL,
    TACTILE_METRIC_DOMAIN,
)
from n0_twam.integrations.univtac.artifact_contracts import (
    verify_track31_evaluation_bundle,
)

OFFLINE_TACTILE_PROTOCOL = INTERNAL_OFFLINE_PROTOCOL

_SOURCE_METADATA_KEYS = (
    "repo_id",
    "episode_index",
    "start_frame",
    "end_frame",
    "frame_index",
    "timestamp",
    "task",
    "tasks",
    "tactile_sensor_ids",
)


def _required_nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _required_positive_integer(value: object, *, label: str) -> int:
    result = _required_nonnegative_integer(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"source metadata value is not JSON serializable: {type(value)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def set_evaluation_seed(seed: int) -> dict[str, object]:
    """Seed every RNG used by the single-process evaluation path."""

    resolved_seed = _required_nonnegative_integer(seed, label="evaluation seed")
    if resolved_seed > np.iinfo(np.uint32).max:
        raise ValueError("evaluation seed must fit in an unsigned 32-bit integer")
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != str(resolved_seed):
        raise ValueError(
            "PYTHONHASHSEED must be set before process launch to the evaluation seed"
        )
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    torch.cuda.manual_seed_all(resolved_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": resolved_seed,
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "torch_cpu_random_seeded": True,
        "torch_cuda_random_seeded": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "torch_deterministic_algorithms": True,
        "python_hash_seed_environment": hash_seed,
        "determinism_level": (
            "seeded_single_process_best_effort_not_cross_hardware_bitwise"
        ),
    }


def set_sample_seed(base_seed: int, sample_ordinal: int) -> int:
    """Decouple each deterministic dataset crop and generation RNG stream."""

    seed = _required_nonnegative_integer(base_seed, label="base_seed")
    ordinal = _required_nonnegative_integer(sample_ordinal, label="sample_ordinal")
    payload = f"n0-twam-tactile-eval:{seed}:{ordinal}".encode("ascii")
    sample_seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    torch.cuda.manual_seed_all(sample_seed)
    return sample_seed


def mask_future_local_tactile(
    batch: Mapping[str, object],
) -> dict[str, object]:
    """Expose only frame 0 of LocalTactile while retaining the input branch."""

    required = (
        "tactile_global_latent",
        "tactile_local_latent",
        "tactile_sensor_ids",
    )
    missing = [key for key in required if not isinstance(batch.get(key), torch.Tensor)]
    if missing:
        raise ValueError("offline tactile sample is missing: " + ", ".join(missing))
    global_tactile = cast(torch.Tensor, batch["tactile_global_latent"])
    local_tactile = cast(torch.Tensor, batch["tactile_local_latent"])
    sensor_ids = cast(torch.Tensor, batch["tactile_sensor_ids"])
    if global_tactile.ndim not in (5, 6) or local_tactile.ndim != global_tactile.ndim:
        raise ValueError("global/local tactile tensors must share a 5D or 6D layout")
    if global_tactile.shape[:3] != local_tactile.shape[:3]:
        raise ValueError(
            "global/local tactile batch, sensor, and channel shapes differ"
        )
    if global_tactile.shape[-3:] != local_tactile.shape[-3:]:
        raise ValueError("global/local tactile temporal/spatial shapes differ")
    if global_tactile.shape[-3] <= 1:
        raise ValueError("offline tactile quality requires a future tactile frame")
    expected_sensors = (
        global_tactile.shape[0] * global_tactile.shape[1]
        if global_tactile.ndim == 6
        else global_tactile.shape[0]
    )
    if sensor_ids.numel() != expected_sensors:
        raise ValueError("tactile sensor IDs do not match tactile streams")
    conditioned = dict(batch)
    masked_local = local_tactile.clone()
    masked_local[..., 1:, :, :] = masked_local[..., 0:1, :, :]
    conditioned["tactile_local_latent"] = masked_local
    return conditioned


def restore_raw_video_context(
    batch: Mapping[str, object],
    input_dict: dict[str, Any],
    *,
    clean_timestep: int | float,
) -> None:
    """Replace noisy-cond video state with exact raw validation latents."""

    raw = batch.get("latents")
    latent_dict = input_dict.get("latent_dict")
    if not isinstance(raw, torch.Tensor) or not isinstance(latent_dict, dict):
        raise ValueError("raw video latents and prepared latent_dict are required")
    prepared = latent_dict.get("latent")
    if not isinstance(prepared, torch.Tensor) or prepared.shape != raw.shape:
        raise ValueError("raw and prepared video latent shapes differ")
    clean = raw.to(device=prepared.device, dtype=prepared.dtype).clone()
    latent_dict["latent"] = clean
    latent_dict["noisy_latents"] = clean.clone()
    for key in ("timesteps", "cond_timesteps"):
        value = latent_dict.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"prepared latent_dict is missing {key}")
        latent_dict[key] = torch.full_like(value, clean_timestep)


def build_deterministic_eval_loader(dataset: Dataset[Any]) -> DataLoader[Any]:
    """Build a fixed-order, single-process loader for metric generation."""

    return DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(cast(Sized, dataset)),
        num_workers=0,
        drop_last=False,
    )


def resolve_dataset_index_metadata(
    dataset: Dataset[Any], dataset_index: int
) -> dict[str, object]:
    """Resolve Track 3.1 episode/task metadata for one global dataset index."""

    index = _required_nonnegative_integer(dataset_index, label="dataset_index")
    datasets = getattr(dataset, "_datasets", None)
    index_map = getattr(dataset, "item_id_to_dataset_id", None)
    offsets = getattr(dataset, "acc_dset_num", None)
    qpos8_offsets = getattr(dataset, "_offsets", None)
    if not isinstance(datasets, list):
        raise ValueError("evaluation dataset does not expose index provenance")
    if isinstance(qpos8_offsets, list):
        dataset_id = next(
            (
                candidate
                for candidate in range(len(qpos8_offsets) - 1, -1, -1)
                if index >= int(qpos8_offsets[candidate])
            ),
            None,
        )
        if dataset_id is None:
            raise ValueError("qpos8 dataset index provenance is incomplete")
        local_index = index - int(qpos8_offsets[dataset_id])
    else:
        if not isinstance(index_map, dict) or not isinstance(offsets, dict):
            raise ValueError("evaluation dataset index provenance is incomplete")
        dataset_id = index_map.get(index)
        if not isinstance(dataset_id, int):
            raise ValueError("evaluation dataset index has no source repository")
        local_index = index - int(offsets[dataset_id])
    source_dataset = datasets[dataset_id]
    metas = getattr(source_dataset, "new_metas", None)
    if not isinstance(metas, list) or not 0 <= local_index < len(metas):
        raise ValueError("evaluation dataset segment metadata is unavailable")
    metadata = dict(metas[local_index])
    metadata["repo_id"] = str(getattr(source_dataset, "repo_id", "")) or None
    return {key: _json_value(metadata.get(key)) for key in _SOURCE_METADATA_KEYS}


def capture_source_sample_metadata(
    batch: Mapping[str, object],
    *,
    output_sample_id: str,
    dataset_index: int,
    loader_cycle: int,
    sample_seed: int | None = None,
    dataset_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Capture only metadata actually present in the pre-conversion batch."""

    if not isinstance(output_sample_id, str) or not output_sample_id:
        raise ValueError("output_sample_id must be a non-empty string")
    resolved_index = _required_nonnegative_integer(dataset_index, label="dataset_index")
    resolved_cycle = _required_nonnegative_integer(loader_cycle, label="loader_cycle")
    resolved_dataset_metadata = dict(dataset_metadata or {})
    source_metadata = {}
    metadata_sources: dict[str, str | None] = {}
    for key in _SOURCE_METADATA_KEYS:
        if key in batch:
            source_metadata[key] = _json_value(batch[key])
            metadata_sources[key] = "batch"
        elif key in resolved_dataset_metadata:
            source_metadata[key] = _json_value(resolved_dataset_metadata[key])
            metadata_sources[key] = "dataset_index"
        else:
            source_metadata[key] = None
            metadata_sources[key] = None
    unavailable = [key for key, value in source_metadata.items() if value is None]
    tensor_contract = {
        str(key): {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "content_sha256": _tensor_sha256(value),
        }
        for key, value in sorted(batch.items(), key=lambda item: str(item[0]))
        if isinstance(value, torch.Tensor)
    }
    return {
        "output_sample_id": output_sample_id,
        "dataset_index": resolved_index,
        "loader_cycle": resolved_cycle,
        "sample_seed": (
            None
            if sample_seed is None
            else _required_nonnegative_integer(sample_seed, label="sample_seed")
        ),
        "source_metadata": source_metadata,
        "source_metadata_origins": metadata_sources,
        "unavailable_source_metadata_keys": unavailable,
        "available_batch_keys": sorted(str(key) for key in batch),
        "batch_tensor_contract": tensor_contract,
    }


def audit_evaluation_dataset(
    *,
    dataset_path: Path,
    manifest_path: Path,
    conversion_report_path: Path,
    normalizer_path: Path,
    evaluation_view_path: Path,
    normalizer_source_view_path: Path,
    expected_manifest_sha256: str,
    expected_normalizer_sha256: str,
    base_model_path: Path,
) -> dict[str, object]:
    """Bind validation materialization to its verified Track 3.1 artifacts."""

    resolved_dataset = Path(dataset_path).resolve(strict=True)
    if not resolved_dataset.is_dir():
        raise NotADirectoryError(resolved_dataset)
    resolved_base_model = Path(base_model_path).resolve(strict=True)
    if not resolved_base_model.is_dir():
        raise NotADirectoryError(resolved_base_model)
    resolved_files = {
        "dataset_manifest": Path(manifest_path).resolve(strict=True),
        "conversion_report": Path(conversion_report_path).resolve(strict=True),
        "normalizer": Path(normalizer_path).resolve(strict=True),
        "evaluation_view": Path(evaluation_view_path).resolve(strict=True),
        "normalizer_source_view": Path(normalizer_source_view_path).resolve(
            strict=True
        ),
        "latent_video_inventory": (
            resolved_dataset / "latent_video_inventory.json"
        ).resolve(strict=True),
        "latent_tactile_inventory": (
            resolved_dataset / "latent_tactile_inventory.json"
        ).resolve(strict=True),
    }
    payloads: dict[str, dict[str, object]] = {}
    for label, path in resolved_files.items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid {label} JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{label} must be a JSON object")
        payloads[label] = payload
    manifest_sha256 = validate_sha256(
        expected_manifest_sha256, label="source manifest SHA256"
    )
    normalizer_sha256 = validate_sha256(
        expected_normalizer_sha256, label="normalizer SHA256"
    )
    verified = verify_track31_evaluation_bundle(
        manifest_path=resolved_files["dataset_manifest"],
        normalizer_path=resolved_files["normalizer"],
        conversion_report_path=resolved_files["conversion_report"],
        dataset_root=resolved_dataset.parent,
        evaluation_view_path=resolved_files["evaluation_view"],
        normalizer_source_view_path=resolved_files["normalizer_source_view"],
    )
    if verified.manifest_sha256 != manifest_sha256:
        raise ValueError("dataset manifest identity disagrees with config")
    if verified.normalizer_sha256 != normalizer_sha256:
        raise ValueError("normalizer identity disagrees with config")
    conversion_report_sha256 = validate_sha256(
        verified.conversion_report_sha256,
        label="verified conversion report SHA256",
    )
    encoder_source_identity = build_encoder_source_identity(resolved_base_model)
    physical_split = resolved_dataset.name
    if physical_split not in {"validation", "frozen40"}:
        raise ValueError("evaluation latent repository must be validation or frozen40")
    inventory_validation = validate_latent_inventory_pair(
        resolved_dataset,
        expected_split=physical_split,
        expected_manifest_sha256=manifest_sha256,
        expected_conversion_report_sha256=conversion_report_sha256,
        expected_encoder_source_identity=encoder_source_identity,
    )
    for label in ("latent_video_inventory", "latent_tactile_inventory"):
        inventory = payloads[label]
        if inventory.get("split") != physical_split:
            raise ValueError(f"{label} does not match the evaluation repository")
        if inventory.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"{label} does not belong to dataset manifest")
    artifact_files = {}
    for label, path in resolved_files.items():
        record: dict[str, object] = {
            "path": str(path),
            "file_sha256": _sha256_file(path),
        }
        if label.startswith("latent_"):
            segments = payloads[label].get("segments")
            if not isinstance(segments, list):
                raise ValueError(f"{label} has no segment inventory")
            record.update(
                {
                    "inventory_sha256": payloads[label].get("inventory_sha256"),
                    "segment_count": len(segments),
                    "artifact_count": sum(
                        len(segment.get("artifacts", []))
                        for segment in segments
                        if isinstance(segment, dict)
                    ),
                }
            )
        artifact_files[label] = record
    return {
        "split": "validation",
        "physical_split": physical_split,
        "dataset_path": str(resolved_dataset),
        "source_manifest_sha256": manifest_sha256,
        "normalizer_sha256": normalizer_sha256,
        "conversion_report_sha256": conversion_report_sha256,
        "evaluation_view_id": verified.evaluation_view_id,
        "evaluation_view_sha256": verified.evaluation_view_sha256,
        "evaluation_episode_count": verified.evaluation_episode_count,
        "normalizer_source_view_id": verified.normalizer_source_view_id,
        "normalizer_source_view_sha256": (verified.normalizer_source_view_sha256),
        "encoder_source_identity": encoder_source_identity,
        "latent_inventory_validation": inventory_validation,
        "artifact_files": artifact_files,
    }


def audit_vae_decoder(vae_path: Path) -> dict[str, object]:
    """Bind the exact VAE config and safetensors files used for RGB decoding."""

    root = Path(vae_path).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"VAE config does not exist: {config_path}")
    weight_paths = sorted(root.rglob("*.safetensors"))
    if not weight_paths:
        raise FileNotFoundError(f"VAE has no safetensors weights under {root}")
    return {
        "directory": str(root),
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "weights": [
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in weight_paths
        ],
    }


def _safe_path_component(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or Path(value).name != value
        or value in {"", ".", ".."}
    ):
        raise ValueError(f"{label} must be one safe path component")
    return value


def _sensor_ids(value: object) -> list[int]:
    flattened = value
    if (
        isinstance(flattened, list)
        and len(flattened) == 1
        and isinstance(flattened[0], list)
    ):
        flattened = flattened[0]
    if not isinstance(flattened, list) or not flattened:
        raise ValueError("sample tactile sensor IDs must be a non-empty list")
    result = []
    for sensor_id in flattened:
        if (
            isinstance(sensor_id, bool)
            or not isinstance(sensor_id, int)
            or sensor_id < 0
        ):
            raise ValueError("sample tactile sensor IDs must be non-negative integers")
        result.append(sensor_id)
    if len(result) != len(set(result)):
        raise ValueError("sample tactile sensor IDs must be unique")
    return result


def _validate_metric_sample_alignment(
    metric_report: Mapping[str, object],
    samples: Sequence[object],
    *,
    active_sensor_count: int,
    active_sensor_ids: Sequence[int],
    model_sensor_capacity: int,
) -> dict[str, object]:
    skip_first = _required_nonnegative_integer(
        metric_report.get("skip_first_frames_per_video"),
        label="skip_first_frames_per_video",
    )
    expected_videos: dict[str, int] = {}
    sample_ids: set[str] = set()
    for raw_sample in samples:
        if not isinstance(raw_sample, Mapping):
            raise ValueError("sample_selection entries must be objects")
        sample_id = _safe_path_component(
            raw_sample.get("output_sample_id"), label="output_sample_id"
        )
        if sample_id in sample_ids:
            raise ValueError("sample_selection output_sample_id values must be unique")
        sample_ids.add(sample_id)
        metadata = raw_sample.get("source_metadata")
        tensors = raw_sample.get("batch_tensor_contract")
        if not isinstance(metadata, Mapping) or not isinstance(tensors, Mapping):
            raise ValueError("sample selection lacks metadata or tensor contract")
        tasks = metadata.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise ValueError("sample selection must bind exactly one task")
        task = _safe_path_component(tasks[0], label="sample task")
        sensor_ids = _sensor_ids(metadata.get("tactile_sensor_ids"))
        if len(sensor_ids) != active_sensor_count:
            raise ValueError("sample active tactile sensor count mismatch")
        if set(sensor_ids) != set(active_sensor_ids):
            raise ValueError(
                "sample tactile sensor IDs differ from checkpoint metadata"
            )
        if max(sensor_ids) >= model_sensor_capacity:
            raise ValueError("sample tactile sensor ID exceeds model capacity")
        global_contract = tensors.get("tactile_global_latent")
        if not isinstance(global_contract, Mapping):
            raise ValueError("sample lacks tactile_global_latent tensor contract")
        shape = global_contract.get("shape")
        if not isinstance(shape, list) or len(shape) not in (5, 6):
            raise ValueError("sample tactile_global_latent shape is invalid")
        tensor_sensor_count = shape[1] if len(shape) == 6 else shape[0]
        if tensor_sensor_count != active_sensor_count:
            raise ValueError("sample tensor sensor count disagrees with metadata")
        decoded_streams = raw_sample.get("decoded_tactile_streams")
        if not isinstance(decoded_streams, list) or len(decoded_streams) != len(
            sensor_ids
        ):
            raise ValueError("sample decoded tactile stream contract is incomplete")
        decoded_by_sensor: dict[int, int] = {}
        for stream in decoded_streams:
            if not isinstance(stream, Mapping):
                raise ValueError("decoded tactile stream records must be objects")
            stream_sensor_ids = _sensor_ids([stream.get("sensor_id")])
            stream_sensor_id = stream_sensor_ids[0]
            if stream_sensor_id in decoded_by_sensor:
                raise ValueError("decoded tactile stream sensor IDs must be unique")
            decoded_frame_count = _required_positive_integer(
                stream.get("decoded_frame_count"),
                label="decoded tactile frame count",
            )
            written_indices = stream.get("written_frame_indices")
            if written_indices != list(range(decoded_frame_count)):
                raise ValueError("decoded tactile written frame indices are incomplete")
            scored_indices = stream.get("metric_scored_frame_indices")
            if scored_indices != list(range(skip_first, decoded_frame_count)):
                raise ValueError(
                    "decoded tactile scored frame indices are inconsistent"
                )
            scored_frame_count = len(scored_indices)
            if scored_frame_count <= 0:
                raise ValueError("sample has no decoded tactile frames after skip")
            if stream.get("reference_layout_mp4_frame_count") != scored_frame_count:
                raise ValueError("decoded tactile MP4 frame count is inconsistent")
            codecs = stream.get("verified_h264_codecs")
            if (
                not isinstance(codecs, Mapping)
                or set(codecs) != {"generate_videos", "gt_videos"}
                or not all(
                    isinstance(value, str) and value for value in codecs.values()
                )
            ):
                raise ValueError("decoded tactile MP4 verification is incomplete")
            decoded_by_sensor[stream_sensor_id] = scored_frame_count
        if decoded_by_sensor.keys() != set(sensor_ids):
            raise ValueError("decoded tactile sensors disagree with sample metadata")
        for sensor_id, scored_frames in decoded_by_sensor.items():
            expected_videos[f"{task}/{sample_id}/sensor_{sensor_id}"] = scored_frames

    per_video = metric_report.get("per_video")
    if not isinstance(per_video, list):
        raise ValueError("metric report must contain per_video records")
    actual_videos: dict[str, int] = {}
    for record in per_video:
        if not isinstance(record, Mapping):
            raise ValueError("metric per_video entries must be objects")
        video_id = record.get("video_id")
        if not isinstance(video_id, str) or video_id in actual_videos:
            raise ValueError("metric video IDs must be unique strings")
        actual_videos[video_id] = _required_positive_integer(
            record.get("frame_count"),
            label=f"metric frame count for {video_id}",
        )
    if actual_videos.keys() != expected_videos.keys():
        raise ValueError("metric video set does not match selected samples and sensors")
    if actual_videos != expected_videos:
        raise ValueError("metric per-video frame counts do not match selected samples")
    overall = metric_report.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError("metric report overall summary must be an object")
    if overall.get("video_count") != len(expected_videos):
        raise ValueError("metric overall video_count disagrees with per-video records")
    if overall.get("frame_count") != sum(expected_videos.values()):
        raise ValueError("metric overall frame_count disagrees with per-video records")
    return {
        "active_tactile_sensor_count": active_sensor_count,
        "active_tactile_sensor_ids": list(active_sensor_ids),
        "model_tactile_stream_capacity": model_sensor_capacity,
        "expected_video_count": len(expected_videos),
        "expected_scored_frame_count": sum(expected_videos.values()),
    }


def build_provenance_bound_report(
    metric_report: Mapping[str, object],
    *,
    checkpoint: Mapping[str, object],
    dataset: Mapping[str, object],
    decoder: Mapping[str, object],
    runtime: Mapping[str, object],
    rng_contract: Mapping[str, object],
    seed: int,
    n_steps: int,
    requested_n_gen: int,
    max_latent_frames: int,
    expected_active_tactile_sensor_count: int,
    metric_split: str,
    sample_selection: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Bind an offline metric report to model, sampling, and sample identities."""

    if metric_report.get("protocol") != OFFLINE_TACTILE_PROTOCOL:
        raise ValueError(
            f"generated tactile reports must use {OFFLINE_TACTILE_PROTOCOL!r}"
        )
    if metric_report.get("pixel_domain") != TACTILE_METRIC_DOMAIN:
        raise ValueError(
            "generated tactile report has the wrong metric domain: "
            f"{metric_report.get('pixel_domain')!r}"
        )
    if metric_report.get("leaderboard_compatible") is not False:
        raise ValueError(
            "internal residual evaluation cannot be leaderboard compatible"
        )
    if metric_report.get("aggregation") != "macro_over_videos":
        raise ValueError("tactile report must use macro_over_videos aggregation")
    input_pairs_sha256 = validate_sha256(
        metric_report.get("input_pairs_sha256"), label="metric input-pairs SHA256"
    )
    resolved_seed = _required_nonnegative_integer(seed, label="evaluation seed")
    resolved_steps = _required_positive_integer(n_steps, label="n_steps")
    resolved_requested = _required_positive_integer(
        requested_n_gen, label="requested_n_gen"
    )
    resolved_max_frames = _required_positive_integer(
        max_latent_frames, label="max_latent_frames"
    )
    if metric_split != "validation":
        raise ValueError("metric_split must be validation")
    samples = [_json_value(dict(sample)) for sample in sample_selection]
    if len(samples) != resolved_requested:
        raise ValueError("generated sample count must equal requested_n_gen")
    checkpoint_provenance = validate_checkpoint_provenance(checkpoint)
    active_sensor_count = _required_positive_integer(
        expected_active_tactile_sensor_count,
        label="expected_active_tactile_sensor_count",
    )
    model_contract = checkpoint_provenance["model_contract"]
    if not isinstance(model_contract, Mapping):
        raise ValueError("checkpoint has no model contract")
    model_sensor_capacity = _required_positive_integer(
        model_contract.get("max_tactile_streams"),
        label="model max_tactile_streams",
    )
    if active_sensor_count > model_sensor_capacity:
        raise ValueError("active tactile sensors exceed model stream capacity")
    training_metadata = checkpoint_provenance["training_metadata"]
    if not isinstance(training_metadata, Mapping):
        raise ValueError("checkpoint has no training metadata")
    if training_metadata.get("active_tactile_sensor_count") != active_sensor_count:
        raise ValueError("evaluator active sensor count differs from checkpoint")
    active_sensor_ids = training_metadata.get("active_tactile_sensor_ids")
    if not isinstance(active_sensor_ids, list):
        raise ValueError("checkpoint has no active tactile sensor IDs")
    metric_output_contract = _validate_metric_sample_alignment(
        metric_report,
        samples,
        active_sensor_count=active_sensor_count,
        active_sensor_ids=active_sensor_ids,
        model_sensor_capacity=model_sensor_capacity,
    )
    provenance = {
        "schema_version": 2,
        "checkpoint": checkpoint_provenance,
        "dataset": _json_value(dict(dataset)),
        "decoder": _json_value(dict(decoder)),
        "runtime": _json_value(dict(runtime)),
        "sampling": {
            "seed": resolved_seed,
            "n_steps": resolved_steps,
            "requested_n_gen": resolved_requested,
            "generated_sample_count": len(samples),
            "max_latent_frames": resolved_max_frames,
            "split": metric_split,
            "data_order": "sequential_no_shuffle",
            "data_loader_num_workers": 0,
            "rng_contract": _json_value(dict(rng_contract)),
        },
        "context_contract": {
            "name": "offline_conditional_tactile_prediction",
            "protocol": OFFLINE_TACTILE_PROTOCOL,
            "closed_loop": False,
            "ground_truth_video_context": True,
            "ground_truth_action_context": True,
            "global_tactile_observed_frame_count": 1,
            "local_tactile_observed_frame_count": 1,
            "future_global_tactile_ground_truth_visible_to_predictor": False,
            "future_local_tactile_ground_truth_visible_to_predictor": False,
        },
        "metric_output_contract": metric_output_contract,
        "metric_domain": TACTILE_METRIC_DOMAIN,
        "sample_selection": samples,
    }
    provenance["provenance_sha256"] = _canonical_json_sha256(provenance)
    bound_report = dict(metric_report)
    bound_report["schema_version"] = 2
    metric_payload_sha256 = _canonical_json_sha256(metric_report)
    bound_report["metric_payload_sha256"] = metric_payload_sha256
    bound_report["provenance"] = provenance
    bound_report["evaluation_identity_sha256"] = _canonical_json_sha256(
        {
            "input_pairs_sha256": input_pairs_sha256,
            "metric_payload_sha256": metric_payload_sha256,
            "provenance_sha256": provenance["provenance_sha256"],
        }
    )
    return bound_report


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Publish complete JSON through a same-directory atomic rename."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
        directory_descriptor = os.open(
            output_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
