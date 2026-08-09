# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed checkpoint and runtime contract for tactile evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import stat
from collections.abc import Mapping
from pathlib import Path

from safetensors import safe_open

from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
    validate_recorded_transformer_identity,
    validate_sha256,
)
from n0_twam.evaluation.tactile_model_contract import (
    MODEL_CONTRACT_FIELDS,
    nonnegative_integer,
    normalize_patch_size,
    positive_float,
    positive_integer,
    required_tactile_shapes,
    validate_tactile_model_contract,
)

_CHECKPOINT_FIELDS = {
    "transformer_directory",
    "config_path",
    "config_sha256",
    "training_metadata",
    "transformer_sha256",
    "transformer_identity",
    "model_contract",
    "tactile_head_contract",
}


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_track31_artifact_identity(payload: object) -> dict[str, str]:
    """Canonicalize the training-data identities needed by formal evaluation."""

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint has no Track 3.1 artifact identity")
    result = {
        "manifest_sha256": validate_sha256(
            payload.get("manifest_sha256"),
            label="checkpoint manifest SHA256",
        ),
        "normalizer_sha256": validate_sha256(
            payload.get("normalizer_sha256"),
            label="checkpoint normalizer SHA256",
        ),
        "conversion_report_sha256": validate_sha256(
            payload.get("conversion_report_sha256"),
            label="checkpoint conversion report SHA256",
        ),
        "train_view_id": _nonempty_string(
            payload.get("train_view_id"), label="checkpoint train_view_id"
        ),
        "train_view_sha256": validate_sha256(
            payload.get("train_view_sha256"),
            label="checkpoint train view SHA256",
        ),
        "normalizer_source_view_id": _nonempty_string(
            payload.get("normalizer_source_view_id"),
            label="checkpoint normalizer_source_view_id",
        ),
        "normalizer_source_view_sha256": validate_sha256(
            payload.get("normalizer_source_view_sha256"),
            label="checkpoint normalizer source view SHA256",
        ),
    }
    return result


def _normalize_sensor_id_map(
    payload: object,
    *,
    active_sensor_ids: list[int],
    label: str,
) -> dict[str, int]:
    """Validate a name-to-ID map before comparing its values."""

    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{label} is invalid")
    normalized: dict[str, int] = {}
    for raw_name, raw_sensor_id in payload.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(f"{label} contains an invalid sensor name")
        normalized[raw_name] = nonnegative_integer(
            raw_sensor_id, label=f"{label} sensor ID"
        )
    if sorted(normalized.values()) != sorted(active_sensor_ids):
        raise ValueError(f"{label} is inconsistent with active sensor IDs")
    return normalized


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {path}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload, raw


def _checkpoint_directories(checkpoint_path: Path) -> tuple[Path, Path]:
    candidate = Path(checkpoint_path).expanduser().resolve(strict=True)
    if not candidate.is_dir():
        raise NotADirectoryError(candidate)
    transformer_candidate = candidate / "transformer"
    if transformer_candidate.is_dir():
        return candidate, transformer_candidate.resolve(strict=True)
    return candidate.parent, candidate


def _file_fingerprint(path: Path) -> tuple[int, ...]:
    result = path.lstat()
    if stat.S_ISLNK(result.st_mode) or not stat.S_ISREG(result.st_mode):
        raise ValueError(f"tactile checkpoint must be a regular non-symlink: {path}")
    return (
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_mode),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def audit_tactile_head_contract(
    weights_path: Path,
    *,
    model_contract: Mapping[str, object],
    config_payload: Mapping[str, object],
) -> dict[str, object]:
    """Require the tactile input/output sentinel tensors and their exact shapes."""

    contract = validate_tactile_model_contract(model_contract)
    heads = positive_integer(
        config_payload.get("num_attention_heads"), label="num_attention_heads"
    )
    head_dim = positive_integer(
        config_payload.get("attention_head_dim"), label="attention_head_dim"
    )
    rope_max_seq_len = positive_integer(
        config_payload.get("rope_max_seq_len"), label="rope_max_seq_len"
    )
    inner_dim = heads * head_dim
    expected_shapes = required_tactile_shapes(
        contract,
        inner_dim=inner_dim,
        rope_max_seq_len=rope_max_seq_len,
    )
    path = Path(weights_path)
    initial_fingerprint = _file_fingerprint(path)
    try:
        with safe_open(
            str(path.resolve(strict=True)), framework="pt", device="cpu"
        ) as f:
            keys = set(f.keys())
            missing = sorted(set(expected_shapes) - keys)
            if missing:
                raise ValueError(
                    "transformer safetensors is missing required tactile tensors: "
                    + ", ".join(missing)
                )
            actual_shapes = {
                key: tuple(int(size) for size in f.get_slice(key).get_shape())
                for key in expected_shapes
            }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid tactile checkpoint safetensors: {path}") from exc
    if _file_fingerprint(path) != initial_fingerprint:
        raise RuntimeError("transformer weights changed during tactile head audit")
    mismatches = [
        f"{key}:{actual_shapes[key]}!={expected}"
        for key, expected in expected_shapes.items()
        if actual_shapes[key] != expected
    ]
    if mismatches:
        raise ValueError(
            "tactile checkpoint tensor shape mismatch: " + ", ".join(mismatches)
        )
    return {
        "schema_version": 1,
        "inner_dim": inner_dim,
        "patch_volume": math.prod(normalize_patch_size(contract["patch_size"])),
        "rope_max_seq_len": rope_max_seq_len,
        "required_tensor_shapes": {
            key: list(shape) for key, shape in expected_shapes.items()
        },
    }


def validate_recorded_tactile_head_contract(
    payload: object, *, model_contract: Mapping[str, object]
) -> dict[str, object]:
    """Validate tactile sentinel provenance without trusting arbitrary fields."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "inner_dim",
        "patch_volume",
        "rope_max_seq_len",
        "required_tensor_shapes",
    }:
        raise ValueError("tactile head contract has an invalid field set")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported tactile head contract schema")
    inner_dim = positive_integer(payload["inner_dim"], label="tactile inner_dim")
    rope_max_seq_len = positive_integer(
        payload["rope_max_seq_len"], label="tactile rope_max_seq_len"
    )
    contract = validate_tactile_model_contract(model_contract)
    expected = required_tactile_shapes(
        contract,
        inner_dim=inner_dim,
        rope_max_seq_len=rope_max_seq_len,
    )
    raw_shapes = payload["required_tensor_shapes"]
    if not isinstance(raw_shapes, Mapping) or set(raw_shapes) != set(expected):
        raise ValueError("tactile head sentinel field set is invalid")
    shapes = {}
    for key, expected_shape in expected.items():
        raw_shape = raw_shapes[key]
        if not isinstance(raw_shape, list) or tuple(raw_shape) != expected_shape:
            raise ValueError(f"tactile head sentinel shape is invalid: {key}")
        shapes[key] = list(expected_shape)
    patch_volume = positive_integer(
        payload["patch_volume"], label="tactile patch_volume"
    )
    if patch_volume != math.prod(normalize_patch_size(contract["patch_size"])):
        raise ValueError("tactile patch volume disagrees with patch_size")
    return {
        "schema_version": 1,
        "inner_dim": inner_dim,
        "patch_volume": patch_volume,
        "rope_max_seq_len": rope_max_seq_len,
        "required_tensor_shapes": shapes,
    }


def audit_evaluation_checkpoint(
    checkpoint_path: Path,
    *,
    expected_action_dim: int,
    expected_action_schema: str,
    expected_model_contract: Mapping[str, object],
) -> dict[str, object]:
    """Bind the checkpoint config, scheduler metadata, and tactile head bytes."""

    action_dim = positive_integer(expected_action_dim, label="expected_action_dim")
    if not isinstance(expected_action_schema, str) or not expected_action_schema:
        raise ValueError("expected_action_schema must be a non-empty string")
    expected = validate_tactile_model_contract(expected_model_contract)
    checkpoint_root, transformer_dir = _checkpoint_directories(checkpoint_path)
    config_path = transformer_dir / "config.json"
    config_payload, config_bytes = _read_json_object(
        config_path, label="transformer config"
    )
    if config_payload.get("is_mot") is not True:
        raise ValueError("transformer config is not an MoT checkpoint")
    if config_payload.get("action_dim") != action_dim:
        raise ValueError("transformer config action_dim mismatch")
    if config_payload.get("action_schema") != expected_action_schema:
        raise ValueError("transformer config action_schema mismatch")

    train_meta_path = checkpoint_root / "train_meta.json"
    train_meta, train_meta_bytes = _read_json_object(
        train_meta_path, label="checkpoint training metadata"
    )
    checkpoint_contract = {
        key: config_payload.get(key) for key in MODEL_CONTRACT_FIELDS
    }
    actual = validate_tactile_model_contract(checkpoint_contract)
    for field in MODEL_CONTRACT_FIELDS:
        metadata_value = train_meta.get(field)
        if field == "patch_size" and isinstance(metadata_value, (list, tuple)):
            metadata_value = list(metadata_value)
        if metadata_value != actual[field]:
            raise ValueError(
                f"checkpoint training metadata mismatch for {field}: "
                f"{metadata_value!r} vs {actual[field]!r}"
            )
    mismatches = [
        f"{key}:{actual[key]!r}!={expected[key]!r}"
        for key in MODEL_CONTRACT_FIELDS
        if actual[key] != expected[key]
    ]
    if mismatches:
        raise ValueError(
            "checkpoint/evaluator tactile model mismatch: " + ", ".join(mismatches)
        )
    active_sensor_count = positive_integer(
        train_meta.get("active_tactile_sensor_count"),
        label="checkpoint active_tactile_sensor_count",
    )
    raw_sensor_ids = train_meta.get("active_tactile_sensor_ids")
    if (
        not isinstance(raw_sensor_ids, list)
        or len(raw_sensor_ids) != active_sensor_count
    ):
        raise ValueError("checkpoint active tactile sensor IDs are incomplete")
    active_sensor_ids = [
        nonnegative_integer(sensor_id, label="active tactile sensor ID")
        for sensor_id in raw_sensor_ids
    ]
    if len(set(active_sensor_ids)) != active_sensor_count:
        raise ValueError("checkpoint active tactile sensor IDs must be unique")
    if max(active_sensor_ids) >= positive_integer(
        actual["max_tactile_streams"], label="max_tactile_streams"
    ):
        raise ValueError("checkpoint active tactile sensor ID exceeds model capacity")
    sensor_id_map = _normalize_sensor_id_map(
        train_meta.get("tactile_sensor_id_map"),
        active_sensor_ids=active_sensor_ids,
        label="checkpoint tactile sensor ID map",
    )
    raw_track31_artifacts = train_meta.get("track31_artifacts")
    track31_artifacts = (
        None
        if raw_track31_artifacts is None
        else _validate_track31_artifact_identity(raw_track31_artifacts)
    )

    weights_path = transformer_dir / TRANSFORMER_WEIGHTS_FILENAME
    tactile_head = audit_tactile_head_contract(
        weights_path,
        model_contract=actual,
        config_payload=config_payload,
    )
    transformer_identity = audit_transformer_checkpoint(
        weights_path,
        expected_action_dim=action_dim,
    )
    return {
        "transformer_directory": str(transformer_dir),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "training_metadata": {
            "path": str(train_meta_path),
            "sha256": hashlib.sha256(train_meta_bytes).hexdigest(),
            "snr_shift": actual["snr_shift"],
            "active_tactile_sensor_count": active_sensor_count,
            "active_tactile_sensor_ids": active_sensor_ids,
            "tactile_sensor_id_map": sensor_id_map,
            "track31_artifacts": track31_artifacts,
        },
        "transformer_sha256": transformer_identity["sha256"],
        "transformer_identity": transformer_identity,
        "model_contract": actual,
        "tactile_head_contract": tactile_head,
    }


def validate_checkpoint_provenance(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    """Canonicalize an audited checkpoint before embedding it in a report."""

    if set(checkpoint) != _CHECKPOINT_FIELDS:
        raise ValueError("checkpoint provenance has an invalid field set")
    transformer_directory = checkpoint["transformer_directory"]
    config_path = checkpoint["config_path"]
    if not isinstance(transformer_directory, str) or not transformer_directory:
        raise ValueError("checkpoint transformer_directory must be non-empty")
    if not isinstance(config_path, str) or not config_path:
        raise ValueError("checkpoint config_path must be non-empty")
    identity_raw = checkpoint["transformer_identity"]
    if not isinstance(identity_raw, dict):
        raise ValueError("checkpoint transformer_identity must be an object")
    action_dim = positive_integer(
        identity_raw.get("action_dim"), label="checkpoint action_dim"
    )
    identity = validate_recorded_transformer_identity(
        identity_raw, expected_action_dim=action_dim
    )
    transformer_sha256 = validate_sha256(
        checkpoint["transformer_sha256"], label="checkpoint transformer SHA256"
    )
    if transformer_sha256 != identity["sha256"]:
        raise ValueError("checkpoint transformer SHA256 disagrees with its identity")
    model_contract = validate_tactile_model_contract(checkpoint["model_contract"])
    head_contract = validate_recorded_tactile_head_contract(
        checkpoint["tactile_head_contract"], model_contract=model_contract
    )
    metadata = checkpoint["training_metadata"]
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "path",
        "sha256",
        "snr_shift",
        "active_tactile_sensor_count",
        "active_tactile_sensor_ids",
        "tactile_sensor_id_map",
        "track31_artifacts",
    }:
        raise ValueError("checkpoint training metadata provenance is invalid")
    metadata_path = metadata["path"]
    if not isinstance(metadata_path, str) or not metadata_path:
        raise ValueError("checkpoint training metadata path must be non-empty")
    metadata_snr = positive_float(
        metadata["snr_shift"], label="checkpoint metadata snr_shift"
    )
    if metadata_snr != model_contract["snr_shift"]:
        raise ValueError("checkpoint metadata snr_shift disagrees with model contract")
    active_sensor_count = positive_integer(
        metadata["active_tactile_sensor_count"],
        label="checkpoint active_tactile_sensor_count",
    )
    active_sensor_ids = metadata["active_tactile_sensor_ids"]
    raw_sensor_id_map = metadata["tactile_sensor_id_map"]
    if (
        not isinstance(active_sensor_ids, list)
        or len(active_sensor_ids) != active_sensor_count
        or any(
            isinstance(sensor_id, bool)
            or not isinstance(sensor_id, int)
            or sensor_id < 0
            for sensor_id in active_sensor_ids
        )
        or len(set(active_sensor_ids)) != active_sensor_count
        or max(active_sensor_ids)
        >= positive_integer(
            model_contract["max_tactile_streams"], label="max_tactile_streams"
        )
    ):
        raise ValueError("checkpoint active tactile sensor provenance is invalid")
    sensor_id_map = _normalize_sensor_id_map(
        raw_sensor_id_map,
        active_sensor_ids=list(active_sensor_ids),
        label="checkpoint tactile sensor ID map provenance",
    )
    raw_track31_artifacts = metadata["track31_artifacts"]
    track31_artifacts = (
        None
        if raw_track31_artifacts is None
        else _validate_track31_artifact_identity(raw_track31_artifacts)
    )
    return {
        "transformer_directory": transformer_directory,
        "config_path": config_path,
        "config_sha256": validate_sha256(
            checkpoint["config_sha256"], label="checkpoint config SHA256"
        ),
        "training_metadata": {
            "path": metadata_path,
            "sha256": validate_sha256(
                metadata["sha256"], label="checkpoint training metadata SHA256"
            ),
            "snr_shift": metadata_snr,
            "active_tactile_sensor_count": active_sensor_count,
            "active_tactile_sensor_ids": list(active_sensor_ids),
            "tactile_sensor_id_map": dict(sensor_id_map),
            "track31_artifacts": track31_artifacts,
        },
        "transformer_sha256": transformer_sha256,
        "transformer_identity": identity,
        "model_contract": model_contract,
        "tactile_head_contract": head_contract,
    }


def validate_checkpoint_dataset_binding(
    checkpoint: Mapping[str, object],
    dataset: Mapping[str, object],
) -> dict[str, object]:
    """Require formal predictions to use the checkpoint's exact data bundle."""

    canonical_checkpoint = validate_checkpoint_provenance(checkpoint)
    metadata = canonical_checkpoint["training_metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint training metadata provenance is invalid")
    artifacts = _validate_track31_artifact_identity(metadata.get("track31_artifacts"))
    dataset_identity = {
        "manifest_sha256": validate_sha256(
            dataset.get("source_manifest_sha256"),
            label="evaluation dataset manifest SHA256",
        ),
        "normalizer_sha256": validate_sha256(
            dataset.get("normalizer_sha256"),
            label="evaluation dataset normalizer SHA256",
        ),
        "conversion_report_sha256": validate_sha256(
            dataset.get("conversion_report_sha256"),
            label="evaluation dataset conversion report SHA256",
        ),
        "normalizer_source_view_id": _nonempty_string(
            dataset.get("normalizer_source_view_id"),
            label="evaluation dataset normalizer_source_view_id",
        ),
        "normalizer_source_view_sha256": validate_sha256(
            dataset.get("normalizer_source_view_sha256"),
            label="evaluation dataset normalizer source view SHA256",
        ),
    }
    mismatches = [
        field
        for field in dataset_identity
        if artifacts[field] != dataset_identity[field]
    ]
    if mismatches:
        raise ValueError(
            "checkpoint/evaluation dataset identity mismatch: " + ", ".join(mismatches)
        )
    return {
        "schema_version": 1,
        **dataset_identity,
        "train_view_id": artifacts["train_view_id"],
        "train_view_sha256": artifacts["train_view_sha256"],
    }
