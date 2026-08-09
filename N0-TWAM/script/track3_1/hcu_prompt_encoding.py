"""HCU-only prompt preparation for formal Track 3.1 video latents."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from diffusers.pipelines.wan.pipeline_wan import prompt_clean

PROMPT_CACHE_SCHEMA_VERSION = 1
PROMPT_CACHE_EMBEDDING_WIDTH = 4096


def clean_episode_task(tasks: object) -> str:
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    if isinstance(tasks, tuple) and tasks:
        return str(tasks[0])
    if isinstance(tasks, str):
        return tasks
    return ""


def resolve_episode_prompt(row: dict[str, Any], override: str | None) -> str:
    if override is not None:
        return override
    if "tasks" not in row:
        raise KeyError(
            "episode metadata has no `tasks` (v3 episodes parquet without a "
            "tasks column and no meta/episodes.jsonl) -- pass --prompt instead"
        )
    return clean_episode_task(row["tasks"])


def normalize_prompt(prompt: str) -> str:
    return str(prompt_clean(prompt))


def build_text_embedding_cache(
    prompts: Iterable[str],
    tokenizer: Any,
    text_encoder: Any,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
    max_unique_prompts: int | None = None,
) -> dict[str, torch.Tensor]:
    """Encode first-seen unique prompts in one inference-only umT5 batch."""

    unique_prompts = list(dict.fromkeys(normalize_prompt(prompt) for prompt in prompts))
    if not unique_prompts:
        raise ValueError("at least one prompt is required for umT5 encoding")
    if max_unique_prompts is not None and len(unique_prompts) > max_unique_prompts:
        raise ValueError(
            "formal Track 3.1 prompt count exceeds the HCU batch contract: "
            f"{len(unique_prompts)} > {max_unique_prompts}"
        )

    text_inputs = tokenizer(
        unique_prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    mask = text_inputs.attention_mask
    sequence_lengths = [int(value) for value in mask.gt(0).sum(dim=1).tolist()]
    encoder_device = next(text_encoder.parameters()).device
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    with torch.inference_mode():
        embeddings = text_encoder(
            text_input_ids.to(encoder_device),
            mask.to(encoder_device),
        ).last_hidden_state.to(dtype=dtype, device=device)

    cache: dict[str, torch.Tensor] = {}
    for prompt, embedding, sequence_length in zip(
        unique_prompts,
        embeddings,
        sequence_lengths,
    ):
        trimmed = embedding[:sequence_length]
        padded = torch.cat(
            [
                trimmed,
                trimmed.new_zeros(
                    max_sequence_length - sequence_length,
                    trimmed.size(1),
                ),
            ]
        )
        cache[prompt] = (
            padded.detach().to(device="cpu", dtype=dtype).contiguous().clone()
        )
    return cache


def write_prompt_embedding_cache(
    path: Path,
    *,
    embeddings: Mapping[str, torch.Tensor],
    split: str,
    manifest_sha256: str,
    conversion_report_sha256: str,
    encoder_source_identity_sha256: str,
    encoding_contract_sha256: str,
    max_sequence_length: int,
) -> None:
    """Atomically publish one formal cache produced by umT5 on HCU."""

    cache_path = Path(path)
    if cache_path.is_symlink() or cache_path.exists():
        raise FileExistsError(f"refusing pre-existing prompt cache: {cache_path}")
    normalized = _validate_embeddings(
        embeddings,
        expected_prompts=set(embeddings),
        max_sequence_length=max_sequence_length,
    )
    payload = {
        "schema_version": PROMPT_CACHE_SCHEMA_VERSION,
        "physical_split": split,
        "manifest_sha256": _require_sha256(manifest_sha256, "manifest"),
        "conversion_report_sha256": _require_sha256(
            conversion_report_sha256,
            "conversion report",
        ),
        "encoder_source_identity_sha256": _require_sha256(
            encoder_source_identity_sha256,
            "encoder source identity",
        ),
        "encoding_contract_sha256": _require_sha256(
            encoding_contract_sha256,
            "encoding contract",
        ),
        "producer_device": "cuda:0",
        "producer_device_type": "cuda",
        "compute_dtype": "bf16",
        "max_sequence_length": max_sequence_length,
        "prompt_clean_policy": "diffusers_wan_prompt_clean_v1",
        "text_encoder_model_class": "UMT5EncoderModel",
        "text_encoder_load_policy": "low_cpu_mem_device_map_v1",
        "text_encoder_forward_policy": "all_unique_max8_eval_inference_v1",
        "embeddings": normalized,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.parent / (
        f".{cache_path.name}.pid-{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_prompt_embedding_cache(
    path: Path,
    *,
    expected_prompts: Iterable[str],
    expected_split: str,
    expected_manifest_sha256: str,
    expected_conversion_report_sha256: str,
    expected_encoder_source_identity_sha256: str,
    expected_encoding_contract_sha256: str,
    max_sequence_length: int,
) -> dict[str, torch.Tensor]:
    """Load and exactly validate an HCU-produced formal prompt cache."""

    cache_path = Path(path)
    if cache_path.is_symlink() or not cache_path.is_file():
        raise FileNotFoundError(f"missing regular prompt cache: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    expected_keys = {
        "schema_version",
        "physical_split",
        "manifest_sha256",
        "conversion_report_sha256",
        "encoder_source_identity_sha256",
        "encoding_contract_sha256",
        "producer_device",
        "producer_device_type",
        "compute_dtype",
        "max_sequence_length",
        "prompt_clean_policy",
        "text_encoder_model_class",
        "text_encoder_load_policy",
        "text_encoder_forward_policy",
        "embeddings",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("prompt cache has an invalid schema")
    expected_metadata = {
        "schema_version": PROMPT_CACHE_SCHEMA_VERSION,
        "physical_split": expected_split,
        "manifest_sha256": expected_manifest_sha256,
        "conversion_report_sha256": expected_conversion_report_sha256,
        "encoder_source_identity_sha256": expected_encoder_source_identity_sha256,
        "encoding_contract_sha256": expected_encoding_contract_sha256,
        "producer_device": "cuda:0",
        "producer_device_type": "cuda",
        "compute_dtype": "bf16",
        "max_sequence_length": max_sequence_length,
        "prompt_clean_policy": "diffusers_wan_prompt_clean_v1",
        "text_encoder_model_class": "UMT5EncoderModel",
        "text_encoder_load_policy": "low_cpu_mem_device_map_v1",
        "text_encoder_forward_policy": "all_unique_max8_eval_inference_v1",
    }
    if any(payload.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("prompt cache metadata does not match the formal run")
    normalized_prompts = {normalize_prompt(prompt) for prompt in expected_prompts}
    return _validate_embeddings(
        payload["embeddings"],
        expected_prompts=normalized_prompts,
        max_sequence_length=max_sequence_length,
    )


def _validate_embeddings(
    embeddings: object,
    *,
    expected_prompts: set[str],
    max_sequence_length: int,
) -> dict[str, torch.Tensor]:
    if (
        not isinstance(embeddings, Mapping)
        or not expected_prompts
        or set(embeddings) != expected_prompts
        or len(expected_prompts) > 8
    ):
        raise ValueError("prompt cache does not contain the exact formal prompt set")
    normalized: dict[str, torch.Tensor] = {}
    for prompt, embedding in embeddings.items():
        if (
            not isinstance(prompt, str)
            or not prompt
            or not isinstance(embedding, torch.Tensor)
            or embedding.device.type != "cpu"
            or embedding.dtype != torch.bfloat16
            or tuple(embedding.shape)
            != (max_sequence_length, PROMPT_CACHE_EMBEDDING_WIDTH)
            or embedding.requires_grad
            or not bool(torch.isfinite(embedding.float()).all())
        ):
            raise ValueError("prompt cache contains an invalid embedding")
        normalized[prompt] = embedding.detach().contiguous().clone()
    return normalized


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


__all__ = (
    "PROMPT_CACHE_SCHEMA_VERSION",
    "build_text_embedding_cache",
    "clean_episode_task",
    "load_prompt_embedding_cache",
    "normalize_prompt",
    "resolve_episode_prompt",
    "write_prompt_embedding_cache",
)
