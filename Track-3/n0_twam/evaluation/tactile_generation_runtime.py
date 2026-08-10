# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Shared runtime helpers for packaged tactile prediction generators."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Protocol, cast

import torch


class TransformerRuntime(Protocol):
    """Transformer surface required by the sealed artifact generator."""

    def eval(self) -> object: ...


class TrainerRuntime(Protocol):
    """Trainer surface used by Stage-A inference."""

    transformer: TransformerRuntime

    def convert_input_format(
        self, input_dict: dict[str, object]
    ) -> dict[str, object]: ...


class TrainerFactory(Protocol):
    def __call__(
        self, config: object, *, inference_only: bool = False
    ) -> TrainerRuntime: ...


class LoadMotCheckpoint(Protocol):
    def __call__(
        self,
        checkpoint_dir: object,
        *,
        torch_dtype: torch.dtype,
        torch_device: str,
    ) -> TransformerRuntime: ...


class LoadVae(Protocol):
    def __call__(
        self,
        vae_path: object,
        *,
        torch_dtype: torch.dtype,
        torch_device: str,
    ) -> object: ...


def load_legacy_runtime() -> tuple[
    TrainerFactory,
    LoadMotCheckpoint,
    LoadVae,
]:
    """Load the legacy modules from either a source tree or an installed wheel."""

    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    model_utils = importlib.import_module("models.utils")
    train_module = importlib.import_module("train")
    return (
        cast(TrainerFactory, getattr(train_module, "Trainer")),
        cast(LoadMotCheckpoint, getattr(model_utils, "load_mot_checkpoint")),
        cast(LoadVae, getattr(model_utils, "load_vae")),
    )


def canonical_sha256(payload: object) -> str:
    """Hash a JSON-compatible payload using the project canonical encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_device_argument(device: object) -> str:
    """Require the logical device used by the single-device evaluator."""

    resolved = str(device)
    if resolved != "cuda:0":
        raise ValueError(
            "Stage-A supports only logical cuda:0; select the physical device "
            "with CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES"
        )
    return resolved


__all__ = (
    "canonical_sha256",
    "load_legacy_runtime",
    "validate_device_argument",
)
