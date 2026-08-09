# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Canonical tactile model configuration and loaded-runtime validation."""

from __future__ import annotations

import math
from collections.abc import Mapping

TACTILE_LATENT_CHANNELS = 48
MODEL_CONTRACT_FIELDS = (
    "patch_size",
    "snr_shift",
    "use_local_tactile",
    "max_tactile_streams",
    "tactile_in_channels",
    "tactile_num_tokens",
    "tactile_encoder_dim",
    "tactile_latent_channels",
)


def nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def positive_integer(value: object, *, label: str) -> int:
    result = nonnegative_integer(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def positive_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def normalize_patch_size(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("patch_size must contain three positive integers")
    return [
        positive_integer(item, label=f"patch_size[{index}]")
        for index, item in enumerate(value)
    ]


def validate_tactile_model_contract(payload: object) -> dict[str, object]:
    """Return the canonical evaluator/checkpoint tactile model contract."""

    if not isinstance(payload, Mapping) or set(payload) != set(MODEL_CONTRACT_FIELDS):
        raise ValueError("tactile model contract has an invalid field set")
    use_local_tactile = payload["use_local_tactile"]
    if not isinstance(use_local_tactile, bool):
        raise ValueError("use_local_tactile must be boolean")
    latent_channels = positive_integer(
        payload["tactile_latent_channels"],
        label="tactile_latent_channels",
    )
    if latent_channels != TACTILE_LATENT_CHANNELS:
        raise ValueError(f"tactile_latent_channels must be {TACTILE_LATENT_CHANNELS}")
    return {
        "patch_size": normalize_patch_size(payload["patch_size"]),
        "snr_shift": positive_float(payload["snr_shift"], label="snr_shift"),
        "use_local_tactile": use_local_tactile,
        "max_tactile_streams": positive_integer(
            payload["max_tactile_streams"], label="max_tactile_streams"
        ),
        "tactile_in_channels": positive_integer(
            payload["tactile_in_channels"], label="tactile_in_channels"
        ),
        "tactile_num_tokens": positive_integer(
            payload["tactile_num_tokens"], label="tactile_num_tokens"
        ),
        "tactile_encoder_dim": positive_integer(
            payload["tactile_encoder_dim"], label="tactile_encoder_dim"
        ),
        "tactile_latent_channels": latent_channels,
    }


def build_expected_tactile_model_contract(config: object) -> dict[str, object]:
    """Capture evaluator configuration fields that affect tactile inference."""

    return validate_tactile_model_contract(
        {
            "patch_size": getattr(config, "patch_size", None),
            "snr_shift": getattr(config, "snr_shift", None),
            "use_local_tactile": getattr(config, "use_local_tactile", None),
            "max_tactile_streams": getattr(config, "max_tactile_streams", None),
            "tactile_in_channels": getattr(config, "tactile_in_channels", None),
            "tactile_num_tokens": getattr(config, "tactile_num_tokens", None),
            "tactile_encoder_dim": getattr(config, "tactile_encoder_dim", None),
            "tactile_latent_channels": TACTILE_LATENT_CHANNELS,
        }
    )


def required_tactile_shapes(
    model_contract: Mapping[str, object],
    *,
    inner_dim: int,
    rope_max_seq_len: int,
) -> dict[str, tuple[int, ...]]:
    """Derive every required tactile state-dict shape from model capacity."""

    contract = validate_tactile_model_contract(model_contract)
    patch_volume = math.prod(normalize_patch_size(contract["patch_size"]))
    patch_channels = (
        positive_integer(
            contract["tactile_latent_channels"], label="tactile_latent_channels"
        )
        * patch_volume
    )
    max_streams = positive_integer(
        contract["max_tactile_streams"], label="max_tactile_streams"
    )
    shapes = {
        "tactile_patch_embed.weight": (inner_dim, patch_channels),
        "tactile_patch_embed.bias": (inner_dim,),
        "sensor_id_embed.weight": (max_streams, inner_dim),
        "tactile_norm.weight": (inner_dim,),
        "tactile_norm.bias": (inner_dim,),
        "tactile_proj_out.weight": (patch_channels, inner_dim),
        "tactile_proj_out.bias": (patch_channels,),
    }
    if bool(contract["use_local_tactile"]):
        shapes.update(
            {
                "local_tactile_patch_embed.weight": (inner_dim, patch_channels),
                "local_tactile_patch_embed.bias": (inner_dim,),
                "local_tactile_sensor_embed.weight": (max_streams, inner_dim),
                "local_tactile_frame_embed.weight": (rope_max_seq_len, inner_dim),
                "local_tactile_h_embed.weight": (rope_max_seq_len, inner_dim),
                "local_tactile_w_embed.weight": (rope_max_seq_len, inner_dim),
                "local_tactile_norm.weight": (inner_dim,),
                "local_tactile_norm.bias": (inner_dim,),
                "local_tactile_cross_attn.to_q.weight": (inner_dim, inner_dim),
                "local_tactile_cross_attn.to_q.bias": (inner_dim,),
                "local_tactile_cross_attn.to_k.weight": (inner_dim, inner_dim),
                "local_tactile_cross_attn.to_k.bias": (inner_dim,),
                "local_tactile_cross_attn.to_v.weight": (inner_dim, inner_dim),
                "local_tactile_cross_attn.to_v.bias": (inner_dim,),
                "local_tactile_cross_attn.to_out.0.weight": (
                    inner_dim,
                    inner_dim,
                ),
                "local_tactile_cross_attn.to_out.0.bias": (inner_dim,),
                "local_tactile_cross_attn.norm_q.weight": (inner_dim,),
                "local_tactile_cross_attn.norm_k.weight": (inner_dim,),
            }
        )
    return shapes


def _read_field(value: object, field: str) -> object:
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field, None)


def audit_loaded_tactile_runtime(
    trainer: object, expected_model_contract: Mapping[str, object]
) -> dict[str, object]:
    """Require loaded trainer/model/schedulers to match the audited checkpoint."""

    expected = validate_tactile_model_contract(expected_model_contract)
    runtime_contract = build_expected_tactile_model_contract(
        getattr(trainer, "config", None)
    )
    if runtime_contract != expected:
        raise ValueError("trainer tactile configuration changed after checkpoint audit")
    if (
        normalize_patch_size(getattr(trainer, "patch_size", None))
        != expected["patch_size"]
    ):
        raise ValueError("trainer patch_size disagrees with evaluator contract")
    latent_scheduler = getattr(trainer, "train_scheduler_latent", None)
    tactile_scheduler = getattr(trainer, "train_scheduler_tactile", None)
    latent_shift = positive_float(
        getattr(latent_scheduler, "shift", None), label="latent scheduler shift"
    )
    tactile_shift = positive_float(
        getattr(tactile_scheduler, "shift", None), label="tactile scheduler shift"
    )
    if latent_shift != expected["snr_shift"]:
        raise ValueError("latent scheduler shift disagrees with evaluator contract")
    if tactile_shift != expected["snr_shift"]:
        raise ValueError("tactile scheduler shift disagrees with evaluator contract")
    transformer = getattr(trainer, "transformer", None)
    model_config = getattr(transformer, "config", None)
    for field in (
        "patch_size",
        "use_local_tactile",
        "max_tactile_streams",
        "tactile_in_channels",
        "tactile_num_tokens",
        "tactile_encoder_dim",
    ):
        actual = _read_field(model_config, field)
        expected_value = expected[field]
        if field == "patch_size":
            actual = normalize_patch_size(actual)
        if actual != expected_value:
            raise ValueError(f"loaded transformer config mismatch for {field}")
    model_attributes = {
        "patch_size": normalize_patch_size(getattr(transformer, "patch_size", None)),
        "use_local_tactile": getattr(transformer, "use_local_tactile", None),
        "max_tactile_streams": getattr(transformer, "max_tactile_streams", None),
        "tactile_latent_channels": getattr(
            transformer, "tactile_latent_channels", None
        ),
    }
    for field, value in model_attributes.items():
        if value != expected[field]:
            raise ValueError(f"loaded transformer attribute mismatch for {field}")
    heads = positive_integer(
        _read_field(model_config, "num_attention_heads"),
        label="loaded num_attention_heads",
    )
    head_dim = positive_integer(
        _read_field(model_config, "attention_head_dim"),
        label="loaded attention_head_dim",
    )
    rope_max_seq_len = positive_integer(
        _read_field(model_config, "rope_max_seq_len"),
        label="loaded rope_max_seq_len",
    )
    expected_shapes = required_tactile_shapes(
        expected,
        inner_dim=heads * head_dim,
        rope_max_seq_len=rope_max_seq_len,
    )
    state_dict_fn = getattr(transformer, "state_dict", None)
    if not callable(state_dict_fn):
        raise ValueError("loaded transformer does not expose a state_dict")
    state_dict = state_dict_fn()
    missing = sorted(set(expected_shapes) - set(state_dict))
    if missing:
        raise ValueError(
            "loaded transformer is missing tactile tensors: " + ", ".join(missing)
        )
    shape_mismatches = [
        f"{key}:{tuple(state_dict[key].shape)}!={expected_shape}"
        for key, expected_shape in expected_shapes.items()
        if tuple(state_dict[key].shape) != expected_shape
    ]
    if shape_mismatches:
        raise ValueError(
            "loaded transformer tactile tensor shape mismatch: "
            + ", ".join(shape_mismatches)
        )
    return {
        "model_contract": expected,
        "latent_scheduler_shift": latent_shift,
        "tactile_scheduler_shift": tactile_shift,
        "loaded_transformer_attributes": model_attributes,
        "loaded_tactile_tensor_shapes": {
            key: list(shape) for key, shape in expected_shapes.items()
        },
    }
