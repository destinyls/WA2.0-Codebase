# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Parameter-freezing contract for vision-only post-training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from torch import nn

_TACTILE_PREFIXES = (
    "tactile_",
    "sensor_id_embed.",
    "local_tactile_",
    "contact_gate.",
    "mot.experts.tactile.",
)


def is_tactile_parameter(name: str) -> bool:
    """Return whether a state-dict parameter belongs only to tactile handling."""

    normalized = name.removeprefix("module.")
    return normalized.startswith(_TACTILE_PREFIXES)


@dataclass(frozen=True)
class TrainabilityContract:
    schema_version: int
    policy: str
    tactile_mode: str
    tactile_profile: str
    frozen_parameter_names: tuple[str, ...]
    frozen_parameter_count: int
    frozen_numel: int
    trainable_parameter_count: int
    trainable_numel: int
    contract_sha256: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "tactile_mode": self.tactile_mode,
            "tactile_profile": self.tactile_profile,
            "frozen_parameter_names": list(self.frozen_parameter_names),
            "frozen_parameter_count": self.frozen_parameter_count,
            "frozen_numel": self.frozen_numel,
            "trainable_parameter_count": self.trainable_parameter_count,
            "trainable_numel": self.trainable_numel,
            "contract_sha256": self.contract_sha256,
        }


def configure_parameter_trainability(
    model: nn.Module,
    *,
    tactile_mode: str,
    freeze_tactile_parameters: bool,
    tactile_profile: str | None = None,
) -> TrainabilityContract:
    """Apply and hash the explicit trainable/frozen parameter partition."""

    if tactile_mode not in {"enabled", "disabled"}:
        raise ValueError("tactile_mode must be enabled or disabled")
    if tactile_mode == "disabled" and not freeze_tactile_parameters:
        raise ValueError("disabled tactile mode must freeze tactile-only parameters")
    resolved_profile = tactile_profile
    if resolved_profile is None:
        resolved_profile = "vision_only" if tactile_mode == "disabled" else "legacy"
    if resolved_profile == "vision_only" and (
        tactile_mode != "disabled" or not freeze_tactile_parameters
    ):
        raise ValueError("vision_only trainability must disable and freeze tactile")
    if resolved_profile in {"vision_tactile", "mixed"} and (
        tactile_mode != "enabled" or freeze_tactile_parameters
    ):
        raise ValueError(f"{resolved_profile} trainability must keep tactile trainable")

    frozen: list[tuple[str, int]] = []
    trainable: list[tuple[str, int]] = []
    for name, parameter in model.named_parameters():
        should_freeze = freeze_tactile_parameters and is_tactile_parameter(name)
        parameter.requires_grad_(not should_freeze)
        target = frozen if should_freeze else trainable
        target.append((name, int(parameter.numel())))
    if tactile_mode == "disabled" and not frozen:
        raise ValueError("no tactile-only parameters were found to freeze")
    if not trainable:
        raise ValueError("trainability policy left no trainable parameters")

    canonical = {
        "schema_version": 1,
        "policy": (
            "freeze_tactile_only_v1" if freeze_tactile_parameters else "train_all_v1"
        ),
        "tactile_mode": tactile_mode,
        "tactile_profile": resolved_profile,
        "frozen_parameter_names": [name for name, _ in frozen],
        "frozen_parameter_count": len(frozen),
        "frozen_numel": sum(size for _, size in frozen),
        "trainable_parameter_count": len(trainable),
        "trainable_numel": sum(size for _, size in trainable),
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return TrainabilityContract(
        schema_version=1,
        policy=str(canonical["policy"]),
        tactile_mode=tactile_mode,
        tactile_profile=resolved_profile,
        frozen_parameter_names=tuple(name for name, _ in frozen),
        frozen_parameter_count=len(frozen),
        frozen_numel=sum(size for _, size in frozen),
        trainable_parameter_count=len(trainable),
        trainable_numel=sum(size for _, size in trainable),
        contract_sha256=hashlib.sha256(encoded).hexdigest(),
    )


__all__ = (
    "TrainabilityContract",
    "configure_parameter_trainability",
    "is_tactile_parameter",
)
