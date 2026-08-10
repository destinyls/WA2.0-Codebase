"""Track 3.1 latent encoding contracts and payload provenance identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import torch

LATENT_INVENTORY_SCHEMA_VERSION = 3
TRACK31_VIDEO_PAYLOAD_PROVENANCE_SCHEMA_VERSION = 3
TRACK31_TACTILE_PAYLOAD_PROVENANCE_SCHEMA_VERSION = 2
# Backward-compatible public alias; kind-aware code uses the two constants above.
TRACK31_PAYLOAD_PROVENANCE_SCHEMA_VERSION = (
    TRACK31_VIDEO_PAYLOAD_PROVENANCE_SCHEMA_VERSION
)
TRACK31_VIDEO_KEYS = ("observation.images.top", "observation.images.wrist_l")
TRACK31_TACTILE_KEYS = ("observation.images.tactile_a", "observation.images.tactile_b")
TRACK31_TACTILE_MODES = ("global", "local")
TRACK31_LATENT_CHANNELS = 48
TRACK31_VIDEO_LATENT_SIZE = (16, 16)
TRACK31_TACTILE_LATENT_SIZE = (8, 8)
TRACK31_TEXT_EMBEDDING_SHAPE = (512, 4096)
TRACK31_FPS = 10
TRACK31_DATASET_SPLITS = frozenset({"train", "validation", "train759", "frozen40"})
TRACK31_FORMAL_SPLITS = frozenset({"train759", "frozen40"})
TRACK31_VIDEO_ENCODING_CONTRACT = {
    "schema_version": 5,
    "height": 256,
    "width": 256,
    "target_fps": TRACK31_FPS,
    "max_sequence_length": 512,
    "compute_dtype": "bf16",
    "video_keys": list(TRACK31_VIDEO_KEYS),
    "prompt_source": "frozen_episode_task",
    "execution_device_policy": "explicit_no_fallback_v1",
    "text_encoder_device_policy": "dedicated_hcu_prompt_precompute_v1",
    "text_encoder_model_class": "UMT5EncoderModel",
    "text_encoder_load_policy": "low_cpu_mem_device_map_v1",
    "text_tokenizer_class": "T5TokenizerFast",
    "prompt_clean_policy": "diffusers_wan_prompt_clean_v1",
    "text_encoder_forward_policy": "all_unique_max8_eval_inference_v1",
    "text_encoder_lifecycle": "shared_read_only_cache_before_vae_workers_v1",
    "prompt_embedding_cache_schema": 1,
}
TRACK31_TACTILE_ENCODING_CONTRACT = {
    "schema_version": 3,
    "height": 128,
    "width": 128,
    "target_fps": TRACK31_FPS,
    "compute_dtype": "bf16",
    "tactile_keys": list(TRACK31_TACTILE_KEYS),
    "mode": "both",
    "local_mode": "current",
    "execution_device_policy": "explicit_no_fallback_v1",
    "vae_device_policy": "logical_cuda0_only_v1",
}

InventoryKind = Literal["video", "tactile"]


def canonical_bytes(payload: object) -> bytes:
    serialized = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return serialized.encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _normalize_source_video_identity(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != {
        "relative_path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("source video identity has an invalid schema")
    relative_path = str(payload.get("relative_path", ""))
    parsed = Path(relative_path)
    if (
        not relative_path
        or parsed.is_absolute()
        or ".." in parsed.parts
        or not relative_path.startswith("videos/")
        or not relative_path.endswith(".mp4")
    ):
        raise ValueError("source video identity has an invalid relative path")
    size_bytes = int(payload.get("size_bytes", 0))
    if size_bytes <= 0:
        raise ValueError("source video identity has an invalid size")
    return {
        "relative_path": relative_path,
        "size_bytes": size_bytes,
        "sha256": _require_sha256(
            str(payload.get("sha256", "")),
            "source video sha256",
        ),
    }


def _require_contained_regular_file(
    dataset_root: Path,
    path: Path,
    *,
    label: str,
) -> Path:
    root = Path(dataset_root).resolve(strict=True)
    candidate = Path(path)
    try:
        relative_path = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escaped the dataset root: {candidate}") from exc
    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {current}")
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(f"missing regular {label}: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} resolved outside the dataset root") from exc
    return resolved


def build_track31_source_video_identity(
    dataset_root: Path,
    source_video: Path,
) -> dict[str, object]:
    """Bind one formal payload to the exact source MP4 bytes it consumed."""

    root = Path(dataset_root).resolve(strict=True)
    resolved = _require_contained_regular_file(
        root,
        Path(source_video),
        label="source video",
    )
    relative_path = resolved.relative_to(root).as_posix()
    return _normalize_source_video_identity(
        {
            "relative_path": relative_path,
            "size_bytes": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
        }
    )


def _normalize_execution_device(value: str) -> tuple[str, str]:
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid execution device: {value!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("formal latent execution device must be cpu or cuda")
    normalized = str(device)
    if device.type == "cpu" and normalized != "cpu":
        raise ValueError("formal CPU execution device must be exactly 'cpu'")
    return normalized, device.type


def build_track31_payload_provenance(
    *,
    split: str,
    manifest_sha256: str,
    conversion_report_sha256: str,
    encoder_source_identity_sha256: str,
    kind: InventoryKind,
    episode_index: int,
    start_frame: int,
    end_frame: int,
    episode_task: str,
    stream_key: str,
    tactile_mode: str | None,
    prompt: str | None,
    source_video_identity: object,
    execution_device: str,
    text_encoder_device: str | None,
) -> dict[str, object]:
    """Construct the only provenance schema accepted for formal payloads."""

    if split not in TRACK31_FORMAL_SPLITS:
        raise ValueError("payload provenance requires a formal physical split")
    if episode_index < 0 or not 0 <= start_frame < end_frame:
        raise ValueError("payload provenance has an invalid episode segment")
    if not isinstance(episode_task, str) or not episode_task.strip():
        raise ValueError("payload provenance requires a non-empty episode task")
    execution_device, execution_device_type = _normalize_execution_device(
        execution_device
    )
    if kind == "video":
        if stream_key not in TRACK31_VIDEO_KEYS:
            raise ValueError("video provenance has an invalid stream key")
        if tactile_mode is not None:
            raise ValueError("video provenance cannot declare a tactile mode")
        if prompt != episode_task:
            raise ValueError("video provenance prompt must equal the frozen task")
        if text_encoder_device is None:
            raise ValueError("formal video provenance requires an HCU text device")
        normalized_text_device, text_encoder_device_type = _normalize_execution_device(
            text_encoder_device
        )
        if (
            execution_device_type != "cuda"
            or text_encoder_device_type != "cuda"
            or normalized_text_device != execution_device
            or execution_device != "cuda:0"
        ):
            raise ValueError(
                "formal video provenance requires HCU-only umT5 and VAE execution"
            )
        text_encoder_device = normalized_text_device
        provenance_schema_version = TRACK31_VIDEO_PAYLOAD_PROVENANCE_SCHEMA_VERSION
        contract = TRACK31_VIDEO_ENCODING_CONTRACT
        prompt_source: str | None = "frozen_episode_task"
    elif kind == "tactile":
        if stream_key not in TRACK31_TACTILE_KEYS:
            raise ValueError("tactile provenance has an invalid stream key")
        if tactile_mode not in TRACK31_TACTILE_MODES:
            raise ValueError("tactile provenance has an invalid mode")
        if prompt is not None or text_encoder_device is not None:
            raise ValueError("tactile provenance cannot declare text encoding")
        if execution_device_type != "cuda" or execution_device != "cuda:0":
            raise ValueError("formal tactile provenance requires VAE on HCU cuda:0")
        text_encoder_device_type = None
        provenance_schema_version = TRACK31_TACTILE_PAYLOAD_PROVENANCE_SCHEMA_VERSION
        contract = TRACK31_TACTILE_ENCODING_CONTRACT
        prompt_source = None
    else:
        raise ValueError(f"unsupported latent provenance kind: {kind!r}")
    return {
        "schema_version": provenance_schema_version,
        "code_schema_version": LATENT_INVENTORY_SCHEMA_VERSION,
        "physical_split": split,
        "manifest_sha256": _require_sha256(
            manifest_sha256,
            "manifest_sha256",
        ),
        "conversion_report_sha256": _require_sha256(
            conversion_report_sha256,
            "conversion_report_sha256",
        ),
        "encoder_source_identity_sha256": _require_sha256(
            encoder_source_identity_sha256,
            "encoder_source_identity_sha256",
        ),
        "encoding_contract_sha256": hashlib.sha256(
            canonical_bytes(contract)
        ).hexdigest(),
        "kind": kind,
        "episode_index": episode_index,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "episode_task": episode_task,
        "stream_key": stream_key,
        "tactile_mode": tactile_mode,
        "prompt_source": prompt_source,
        "prompt": prompt,
        "source_video": _normalize_source_video_identity(source_video_identity),
        "execution_device": execution_device,
        "execution_device_type": execution_device_type,
        "text_encoder_device": text_encoder_device,
        "text_encoder_device_type": text_encoder_device_type,
    }


def expected_track31_sampled_frame_ids(
    start_frame: int,
    end_frame: int,
) -> list[int]:
    """Return the complete same-FPS segment after deterministic Wan trimming."""

    frame_count = end_frame - start_frame
    if frame_count <= 0:
        raise ValueError("formal segment must contain at least one source frame")
    retained_count = frame_count - ((frame_count - 1) % 4)
    return list(range(start_frame, start_frame + retained_count))
