# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Condition-only Target-10 bundle consumed by Stage-A HCU generation."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import numpy.typing as npt
import torch

from n0_twam.evaluation.fair_protocol import (
    CAUSAL_FUTURE_ONLY_PROTOCOL,
    causal_conditioning_sha256,
    sanitize_causal_future_only_batch,
    verify_future_gt_invariance,
)
from n0_twam.evaluation.sealed_artifact_io import (
    publish_sealed_npz_directory,
    read_json_object,
    sha256_file,
    validate_sha256,
    verify_sealed_file_inventory,
)
from n0_twam.evaluation.target10_reference_contract import REFERENCE_CONTRACT
from n0_twam.integrations.univtac.dataset_view import load_dataset_view

BUNDLE_SCHEMA_VERSION = 4
BUNDLE_TYPE = "n0_twam_target10_continuous41_causal_conditioning"
METADATA_NAME = "conditions.json"
PAYLOAD_NAME = "conditions.npz"
SEAL_NAME = "seal.json"
TENSOR_KEYS = (
    "latents",
    "actions",
    "actions_mask",
    "text_emb",
    "tactile_global_latent",
    "tactile_local_latent",
    "tactile_sensor_ids",
)


@dataclass(frozen=True)
class CausalInputSample:
    """One restored CPU tensor envelope with no future observation values."""

    metadata: dict[str, object]
    tensors: dict[str, torch.Tensor]


@dataclass(frozen=True)
class VerifiedCausalInputBundle:
    """Verified condition-only inputs and artifact identities."""

    root: Path
    metadata: dict[str, object]
    samples: tuple[CausalInputSample, ...]
    seal_sha256: str
    file_sha256: dict[str, str]


def _normalize_batch_tensor(key: str, tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu().contiguous()
    if key == "tactile_sensor_ids" and value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim == 0 or value.shape[0] != 1:
        raise ValueError(f"causal tensor {key} must have one leading batch item")
    return value


def _numpy_payload(
    tensor: torch.Tensor,
) -> tuple[npt.NDArray[np.generic], str]:
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().copy(), "bfloat16"
    try:
        return tensor.numpy().copy(), str(tensor.dtype).removeprefix("torch.")
    except TypeError as exc:
        raise ValueError(f"unsupported causal tensor dtype: {tensor.dtype}") from exc


def _torch_payload(array: npt.NDArray[np.generic], dtype_name: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(array).copy())
    if dtype_name == "bfloat16":
        if tensor.dtype != torch.uint16:
            raise ValueError("bfloat16 causal tensor storage must use uint16")
        return tensor.view(torch.bfloat16)
    expected = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bool": torch.bool,
        "int32": torch.int32,
        "int64": torch.int64,
    }.get(dtype_name)
    if expected is None or tensor.dtype != expected:
        raise ValueError(f"causal tensor storage dtype mismatch for {dtype_name}")
    return tensor


def _perturb_future(batch: Mapping[str, object]) -> dict[str, object]:
    perturbed: dict[str, object] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            perturbed[key] = value
            continue
        clone = value.clone()
        if key in {"latents", "tactile_global_latent", "tactile_local_latent"}:
            if clone.shape[-3] <= 1:
                raise ValueError(f"{key} has no future frames to perturb")
            clone[..., 1:, :, :] = clone[..., 1:, :, :] + 1
        elif key == "actions":
            clone.fill_(1)
        elif key == "actions_mask":
            clone.fill_(True)
        perturbed[key] = clone
    return perturbed


def _scalar_int(value: object, *, label: str) -> int:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"causal source {label} must be a scalar tensor")
    result = int(value.item())
    if result < 0:
        raise ValueError(f"causal source {label} must be non-negative")
    return result


def _scalar_string(value: object, *, label: str) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value:
        raise ValueError(f"causal source {label} must be a string")
    return value


def _sample_id(view_sha256: str, source_sha256: str) -> str:
    payload = f"{view_sha256}\0{source_sha256}".encode("ascii")
    return "sample_" + hashlib.sha256(payload).hexdigest()


def _empty_embedding_identity(path: Path, *, expected_sha256: str) -> dict[str, object]:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("empty embedding must not be a symlink")
    resolved = raw_path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("empty embedding must be a regular file")
    expected = validate_sha256(expected_sha256, label="empty embedding SHA256")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError("empty embedding SHA256 differs from the request")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "file_sha256": actual,
    }


def build_target10_causal_input_bundle(
    output: Path,
    *,
    config_name: str,
    evaluation_view_path: Path,
    lerobot_root: Path,
    manifest_path: Path,
    conversion_report_path: Path,
    normalizer_path: Path,
    normalizer_source_view_path: Path,
    base_model_path: Path,
    empty_embedding_path: Path,
    expected_manifest_sha256: str,
    expected_normalizer_sha256: str,
    expected_empty_embedding_sha256: str,
) -> Path:
    """Build and seal ten frame0-only causal envelopes on CPU."""

    from n0_twam.configs import TWAM_CONFIGS
    from n0_twam.dataset import MultiLatentLeRobotDataset
    from n0_twam.evaluation.tactile_provenance import (
        audit_evaluation_dataset,
        build_deterministic_eval_loader,
        resolve_dataset_index_metadata,
    )

    if config_name != "track31_univtac":
        raise ValueError("Target-10 causal input requires track31_univtac")
    view_path = Path(evaluation_view_path).resolve(strict=True)
    view = load_dataset_view(view_path)
    if view.view_id != "frozen_target10_v1" or len(view.entries) != 10:
        raise ValueError("causal input requires the complete frozen Target-10 view")
    config = deepcopy(TWAM_CONFIGS[config_name])
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.enable_wandb = False
    config.load_worker = 0
    config.lerobot_root = str(Path(lerobot_root).resolve(strict=True))
    config.dataset_path = str(
        (Path(config.lerobot_root) / view.physical_split).resolve(strict=True)
    )
    config.dataset_manifest_path = str(Path(manifest_path).resolve(strict=True))
    config.conversion_report_path = str(
        Path(conversion_report_path).resolve(strict=True)
    )
    config.norm_stat_path = str(Path(normalizer_path).resolve(strict=True))
    config.normalizer_source_view_path = str(
        Path(normalizer_source_view_path).resolve(strict=True)
    )
    config.wan22_pretrained_model_name_or_path = str(
        Path(base_model_path).resolve(strict=True)
    )
    empty_embedding = _empty_embedding_identity(
        empty_embedding_path,
        expected_sha256=expected_empty_embedding_sha256,
    )
    config.empty_emb_path = str(empty_embedding["path"])
    config.source_manifest_sha256 = validate_sha256(
        expected_manifest_sha256, label="source manifest SHA256"
    )
    config.normalizer_sha256 = validate_sha256(
        expected_normalizer_sha256, label="normalizer SHA256"
    )
    config.dataset_view_path = str(view_path)
    config.train_view_id = view.view_id
    config.val_dataset_path = None
    config.val_dataset_view_path = None
    config.raw_tactile_evaluation = True
    config.deterministic_evaluation_crop_zero = True
    config.max_latent_frames = REFERENCE_CONTRACT.model_latent_frames
    config.tactile_cfg_prob = 0.0
    config.noisy_cond_prob_tactile = 0.0

    dataset_provenance = {
        **audit_evaluation_dataset(
            dataset_path=Path(config.dataset_path),
            manifest_path=Path(config.dataset_manifest_path),
            conversion_report_path=Path(config.conversion_report_path),
            normalizer_path=Path(config.norm_stat_path),
            evaluation_view_path=view_path,
            normalizer_source_view_path=Path(config.normalizer_source_view_path),
            expected_manifest_sha256=config.source_manifest_sha256,
            expected_normalizer_sha256=config.normalizer_sha256,
            base_model_path=Path(config.wan22_pretrained_model_name_or_path),
        ),
        "empty_embedding": empty_embedding,
    }
    if (
        dataset_provenance["evaluation_view_id"] != view.view_id
        or dataset_provenance["evaluation_view_sha256"] != view.view_sha256
        or dataset_provenance["evaluation_episode_count"] != 10
    ):
        raise ValueError("audited evaluation dataset differs from Target-10 view")

    dataset: Any = MultiLatentLeRobotDataset(config=config)
    if len(dataset) != 10:
        raise ValueError("causal input dataset must contain exactly ten episodes")
    loader = iter(build_deterministic_eval_loader(dataset))
    samples: list[dict[str, object]] = []
    tensors_by_key: dict[str, list[torch.Tensor]] = {key: [] for key in TENSOR_KEYS}
    view_by_path = {entry.relative_path: entry for entry in view.entries}
    for dataset_index in range(10):
        batch = dict(next(loader))
        source_path = _scalar_string(
            batch.pop("source_relative_path", None), label="relative path"
        )
        episode_index = _scalar_int(
            batch.pop("lerobot_episode_index", None), label="episode index"
        )
        source_rows = cast(torch.Tensor, batch.pop("source_row_ids", None))
        batch.pop("source_step_ids", None)
        if not isinstance(source_rows, torch.Tensor) or source_rows.reshape(
            -1
        ).tolist() != list(range(REFERENCE_CONTRACT.model_decoded_frame_count)):
            raise ValueError("causal source must contain continuous raw rows 0..40")
        index_metadata = resolve_dataset_index_metadata(dataset, dataset_index)
        raw_tasks = index_metadata.get("tasks")
        if not isinstance(raw_tasks, list) or len(raw_tasks) != 1:
            raise ValueError("causal dataset item must contain exactly one task")
        task = str(raw_tasks[0])
        entry = view_by_path.get(source_path)
        if (
            entry is None
            or entry.task != task
            or entry.lerobot_episode_id != episode_index
            or entry.source_episode_id
            != REFERENCE_CONTRACT.target_episode_ids[
                dataset_index % len(REFERENCE_CONTRACT.target_episode_ids)
            ]
        ):
            raise ValueError("causal dataset item differs from frozen Target-10")
        proof = verify_future_gt_invariance(batch, _perturb_future(batch))
        sanitized = sanitize_causal_future_only_batch(batch)
        if causal_conditioning_sha256(sanitized) != proof["conditioning_sha256"]:
            raise RuntimeError(
                "causal conditioning hash changed after invariance proof"
            )
        for key in TENSOR_KEYS:
            tensors_by_key[key].append(_normalize_batch_tensor(key, sanitized[key]))
        samples.append(
            {
                "sample_id": _sample_id(view.view_sha256, entry.source_sha256),
                "dataset_index": dataset_index,
                "episode_index": entry.source_episode_id,
                "lerobot_episode_index": episode_index,
                "task": task,
                "source_relative_path": source_path,
                "conditioning_sha256": proof["conditioning_sha256"],
                "future_gt_invariant": True,
            }
        )
    arrays: dict[str, npt.NDArray[np.generic]] = {}
    tensor_dtypes: dict[str, str] = {}
    tensor_shapes: dict[str, list[int]] = {}
    for key in TENSOR_KEYS:
        combined = torch.cat(tensors_by_key[key], dim=0).contiguous()
        arrays[key], tensor_dtypes[key] = _numpy_payload(combined)
        tensor_shapes[key] = list(combined.shape)
    metadata: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "artifact_type": BUNDLE_TYPE,
        "contract_id": REFERENCE_CONTRACT.contract_id,
        "contract_sha256": REFERENCE_CONTRACT.sha256,
        "conditioning_protocol_id": CAUSAL_FUTURE_ONLY_PROTOCOL,
        "conditioning_raw_rows": [0],
        "target_raw_rows": list(REFERENCE_CONTRACT.target_raw_rows),
        "temporal_binding": REFERENCE_CONTRACT.temporal_binding,
        "model_latent_frames": REFERENCE_CONTRACT.model_latent_frames,
        "model_decoded_frame_count": REFERENCE_CONTRACT.model_decoded_frame_count,
        "selected_output_indices": list(REFERENCE_CONTRACT.selected_output_indices),
        "decoded_frame_count": REFERENCE_CONTRACT.decoded_frame_count,
        "sequence_length_status": REFERENCE_CONTRACT.sequence_length_status,
        "future_gt_embedded": False,
        "evaluation_view_id": view.view_id,
        "evaluation_view_sha256": view.view_sha256,
        "sample_count": 10,
        "dataset_provenance": dataset_provenance,
        "tensor_dtypes": tensor_dtypes,
        "tensor_shapes": tensor_shapes,
        "samples": samples,
    }
    return cast(
        Path,
        publish_sealed_npz_directory(
            Path(output),
            schema_version=BUNDLE_SCHEMA_VERSION,
            artifact_type=BUNDLE_TYPE,
            metadata_name=METADATA_NAME,
            payload_name=PAYLOAD_NAME,
            seal_name=SEAL_NAME,
            metadata=metadata,
            arrays=arrays,
        ),
    )


def verify_target10_causal_input_bundle(path: Path) -> VerifiedCausalInputBundle:
    """Verify and restore a sealed condition-only bundle."""

    root, seal_sha256, file_sha256 = verify_sealed_file_inventory(
        path,
        schema_version=BUNDLE_SCHEMA_VERSION,
        artifact_type=BUNDLE_TYPE,
        metadata_name=METADATA_NAME,
        payload_name=PAYLOAD_NAME,
        seal_name=SEAL_NAME,
    )
    metadata = read_json_object(root / METADATA_NAME)
    expected = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "artifact_type": BUNDLE_TYPE,
        "contract_id": REFERENCE_CONTRACT.contract_id,
        "contract_sha256": REFERENCE_CONTRACT.sha256,
        "conditioning_protocol_id": CAUSAL_FUTURE_ONLY_PROTOCOL,
        "conditioning_raw_rows": [0],
        "target_raw_rows": list(REFERENCE_CONTRACT.target_raw_rows),
        "temporal_binding": REFERENCE_CONTRACT.temporal_binding,
        "model_latent_frames": REFERENCE_CONTRACT.model_latent_frames,
        "model_decoded_frame_count": REFERENCE_CONTRACT.model_decoded_frame_count,
        "selected_output_indices": list(REFERENCE_CONTRACT.selected_output_indices),
        "decoded_frame_count": REFERENCE_CONTRACT.decoded_frame_count,
        "sequence_length_status": REFERENCE_CONTRACT.sequence_length_status,
        "future_gt_embedded": False,
        "sample_count": 10,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("causal input bundle contract is invalid")
    validate_sha256(metadata.get("evaluation_view_sha256"), label="view SHA256")
    dataset_provenance = metadata.get("dataset_provenance")
    if not isinstance(dataset_provenance, Mapping):
        raise ValueError("causal input dataset provenance is missing")
    for key in (
        "source_manifest_sha256",
        "normalizer_sha256",
        "conversion_report_sha256",
        "normalizer_source_view_sha256",
    ):
        validate_sha256(dataset_provenance.get(key), label=f"dataset {key}")
    empty_embedding = dataset_provenance.get("empty_embedding")
    if (
        not isinstance(empty_embedding, Mapping)
        or set(empty_embedding) != {"path", "size_bytes", "file_sha256"}
        or not isinstance(empty_embedding.get("path"), str)
        or not isinstance(empty_embedding.get("size_bytes"), int)
        or isinstance(empty_embedding.get("size_bytes"), bool)
        or int(empty_embedding["size_bytes"]) <= 0
    ):
        raise ValueError("causal input empty embedding provenance is invalid")
    validate_sha256(
        empty_embedding.get("file_sha256"),
        label="dataset empty embedding SHA256",
    )
    if (
        dataset_provenance.get("evaluation_view_id")
        != metadata.get("evaluation_view_id")
        or dataset_provenance.get("evaluation_view_sha256")
        != metadata.get("evaluation_view_sha256")
        or dataset_provenance.get("evaluation_episode_count") != 10
    ):
        raise ValueError("causal input dataset provenance disagrees with its view")
    raw_samples = metadata.get("samples")
    dtypes = metadata.get("tensor_dtypes")
    shapes = metadata.get("tensor_shapes")
    if (
        not isinstance(raw_samples, list)
        or len(raw_samples) != 10
        or not isinstance(dtypes, Mapping)
        or set(dtypes) != set(TENSOR_KEYS)
        or not isinstance(shapes, Mapping)
        or set(shapes) != set(TENSOR_KEYS)
    ):
        raise ValueError("causal input bundle metadata is incomplete")
    payload_path = root / PAYLOAD_NAME
    with np.load(payload_path, allow_pickle=False) as payload:
        if set(payload.files) != set(TENSOR_KEYS):
            raise ValueError("causal input tensor names are invalid")
        arrays = {key: np.asarray(payload[key]).copy() for key in TENSOR_KEYS}
    if sha256_file(payload_path) != file_sha256[PAYLOAD_NAME]:
        raise RuntimeError("causal input tensors changed while loading")
    restored = {
        key: _torch_payload(arrays[key], str(dtypes[key])) for key in TENSOR_KEYS
    }
    for key, tensor in restored.items():
        if list(tensor.shape) != shapes[key] or tensor.shape[0] != 10:
            raise ValueError(f"causal input tensor shape drift for {key}")
    samples = []
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, dict):
            raise ValueError("causal input sample metadata is invalid")
        expected_task = REFERENCE_CONTRACT.target_tasks[
            index // len(REFERENCE_CONTRACT.target_episode_ids)
        ]
        expected_episode = REFERENCE_CONTRACT.target_episode_ids[
            index % len(REFERENCE_CONTRACT.target_episode_ids)
        ]
        if (
            raw_sample.get("dataset_index") != index
            or raw_sample.get("task") != expected_task
            or raw_sample.get("episode_index") != expected_episode
            or isinstance(raw_sample.get("lerobot_episode_index"), bool)
            or not isinstance(raw_sample.get("lerobot_episode_index"), int)
            or int(raw_sample["lerobot_episode_index"]) < 0
        ):
            raise ValueError("causal input sample roster is invalid")
        tensors = {key: restored[key][index : index + 1].clone() for key in TENSOR_KEYS}
        if causal_conditioning_sha256(tensors) != raw_sample.get("conditioning_sha256"):
            raise ValueError("causal input sample hash is invalid")
        samples.append(CausalInputSample(metadata=dict(raw_sample), tensors=tensors))
    return VerifiedCausalInputBundle(
        root=root,
        metadata=metadata,
        samples=tuple(samples),
        seal_sha256=seal_sha256,
        file_sha256=file_sha256,
    )


__all__ = (
    "CausalInputSample",
    "VerifiedCausalInputBundle",
    "build_target10_causal_input_bundle",
    "verify_target10_causal_input_bundle",
)
