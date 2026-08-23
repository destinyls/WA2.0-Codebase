# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Hash-bound prepared-data and video-latent contracts for Franka training."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from n0_twam.data.encoder_source_identity import (
    build_encoder_source_identity,
    validate_encoder_source_identity,
)
from n0_twam.integrations.univtac.convert_lerobot import (
    build_lerobot_table_inventory,
)

from .franka_actions import DERIVED_ACTION_SCHEMA, FRANKA_ACTION_SCHEMA
from .franka_manifest import OFFICIAL_RECORDS_SHA256, canonical_sha256, sha256_file
from .franka_views import (
    DEVELOPMENT_TRAIN_VIEW,
    DEVELOPMENT_VALIDATION_VIEW,
    FINAL_REFIT_VIEW,
    TASK_FINETUNE_VIEWS,
    FrankaDatasetView,
    build_task_franka_views,
    load_franka_view,
)

VIDEO_KEYS = ("observation.images.top", "observation.images.wrist_l")
MODEL_ACTION_SCHEMA = "ee20_absee"
SOURCE_ACTION_SCHEMA = FRANKA_ACTION_SCHEMA
FULL_VERIFICATION_RECEIPT = "full600_xyzw_verification.json"


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _write_new_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _episode_lengths(dataset_root: Path) -> dict[int, int]:
    path = dataset_root / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing Franka episodes.jsonl: {path}")
    lengths: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        episode_id = int(payload["episode_index"])
        length = int(payload["length"])
        if episode_id in lengths or not 0 <= episode_id < 600 or length <= 1:
            raise ValueError("Franka episode table is invalid")
        lengths[episode_id] = length
    if set(lengths) != set(range(600)):
        raise ValueError("Franka episode table must contain IDs 0..599")
    return lengths


def _validate_lerobot_info(dataset_root: Path) -> dict[str, object]:
    info = _json_object(dataset_root / "meta" / "info.json", label="LeRobot info")
    if int(round(float(info.get("fps", 0)))) != 15:
        raise ValueError("official Franka LeRobot data must remain at 15 fps")
    if int(info.get("total_episodes", 0)) != 600:
        raise ValueError("official Franka LeRobot data must contain 600 episodes")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("Franka LeRobot feature table is invalid")
    required = {*VIDEO_KEYS, "observation.state", "action"}
    if not required.issubset(features):
        raise ValueError("Franka LeRobot dataset is missing required features")
    if list(features["action"].get("shape", [])) != [10]:
        raise ValueError("converted Franka action feature must be EE10")
    if any("tactile" in str(key).lower() for key in features):
        raise ValueError("vision-only Franka LeRobot data contains tactile features")
    return info


def _validate_normalizer(
    path: Path,
    *,
    source_view: FrankaDatasetView,
) -> dict[str, object]:
    payload = _json_object(path, label="Franka normalizer")
    if (
        payload.get("schema_version") != 1
        or payload.get("method") != "q01q99"
        or payload.get("source_action_schema") != SOURCE_ACTION_SCHEMA
        or payload.get("derived_action_schema") != DERIVED_ACTION_SCHEMA
        or payload.get("model_action_schema") != MODEL_ACTION_SCHEMA
        or payload.get("active_action_channel_ids") != list(range(10))
        or payload.get("source_records_sha256") != OFFICIAL_RECORDS_SHA256
        or payload.get("source_view_id") != source_view.view_id
        or payload.get("source_view_sha256") != source_view.view_sha256
    ):
        raise ValueError("Franka normalizer provenance mismatch")
    for field in ("action_q01", "action_q99"):
        values = payload.get(field)
        if not isinstance(values, list) or len(values) != 20:
            raise ValueError(f"Franka normalizer {field} must contain 20 values")
    if payload["action_q01"][10:] != [-1.0] * 10:
        raise ValueError("Franka inactive q01 channels are not canonical")
    if payload["action_q99"][10:] != [1.0] * 10:
        raise ValueError("Franka inactive q99 channels are not canonical")
    core = {key: value for key, value in payload.items() if key != "normalizer_sha256"}
    if payload.get("normalizer_sha256") != canonical_sha256(core):
        raise ValueError("Franka normalizer canonical hash mismatch")
    return payload


@dataclass(frozen=True)
class VerifiedFrankaArtifacts:
    artifact_root: Path
    dataset_root: Path
    train_view: FrankaDatasetView
    validation_view: FrankaDatasetView | None
    normalizer_source_view: FrankaDatasetView
    normalizer: dict[str, object]
    conversion_report: dict[str, object]
    latent_inventory: dict[str, object]
    prepare_receipt_sha256: str
    conversion_report_sha256: str
    latent_inventory_sha256: str
    full_verification_receipt_sha256: str | None


def _validate_full_verification_receipt(
    *,
    artifacts: Path,
    dataset: Path,
    prepare_receipt_sha256: str,
    conversion_report_sha256: str,
    latent_inventory_sha256: str,
    normalizer_sha256: object,
) -> str:
    """Bind a task view to the previously completed all600 byte audit."""

    path = artifacts / FULL_VERIFICATION_RECEIPT
    if path.is_symlink() or not path.is_file():
        raise ValueError("task fine-tuning requires the all600 verification receipt")
    payload = _json_object(path, label="Franka all600 verification receipt")
    expected = {
        "status": "PASS",
        "dataset_root": dataset.as_posix(),
        "prepare_receipt_sha256": prepare_receipt_sha256,
        "conversion_report_sha256": conversion_report_sha256,
        "latent_inventory_file_sha256": latent_inventory_sha256,
        "normalizer_sha256": normalizer_sha256,
        "latent_record_count": 1200,
        "train_episode_count": 600,
        "train_view_id": FINAL_REFIT_VIEW,
        "validation_view": None,
    }
    mismatches = {
        field: (payload.get(field), wanted)
        for field, wanted in expected.items()
        if payload.get(field) != wanted
    }
    if mismatches:
        raise ValueError(f"Franka all600 verification receipt mismatch: {mismatches}")
    return sha256_file(path)


def _validate_latent_payload(
    path: Path,
    *,
    episode_id: int,
    length: int,
    video_key: str,
    encoder_identity_sha256: str,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Franka latent payload is not a dictionary: {path}")
    frame_ids = payload.get("frame_ids")
    latent = payload.get("latent")
    if (
        not isinstance(frame_ids, list)
        or len(frame_ids) < 5
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in frame_ids
        )
        or frame_ids[0] != 0
        or frame_ids != sorted(set(frame_ids))
        or int(frame_ids[-1]) >= length
    ):
        raise ValueError(f"Franka latent frame IDs are invalid: {path}")
    if not torch.is_tensor(latent) or latent.ndim != 2 or not latent.isfinite().all():
        raise ValueError(f"Franka latent tensor is invalid: {path}")
    latent_frames = int(payload.get("latent_num_frames", 0))
    height = int(payload.get("latent_height", 0))
    width = int(payload.get("latent_width", 0))
    if (
        payload.get("start_frame") != 0
        or payload.get("end_frame") != length
        or payload.get("fps") != 10
        or payload.get("ori_fps") != 15
        or payload.get("video_num_frames") != len(frame_ids)
        or latent_frames != (len(frame_ids) - 1) // 4 + 1
        or latent.shape[0] != latent_frames * height * width
        or payload.get("encoder_source_identity_sha256") != encoder_identity_sha256
    ):
        raise ValueError(f"Franka latent metadata mismatch: {path}")
    return {
        "path": path.as_posix(),
        "episode_id": episode_id,
        "video_key": video_key,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "frame_ids_sha256": hashlib.sha256(
            json.dumps(frame_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "latent_num_frames": latent_frames,
    }


def finalize_franka_video_latents(
    *,
    artifact_root: Path,
    lerobot_root: Path,
    base_model: Path,
) -> dict[str, object]:
    """Audit all 1,200 payloads and publish one immutable inventory."""

    artifacts = Path(artifact_root).expanduser().resolve(strict=True)
    dataset = Path(lerobot_root).expanduser().resolve(strict=True) / "all600"
    _validate_lerobot_info(dataset)
    lengths = _episode_lengths(dataset)
    encoder_identity = validate_encoder_source_identity(
        build_encoder_source_identity(Path(base_model))
    )
    records: list[dict[str, object]] = []
    for episode_id in range(600):
        length = lengths[episode_id]
        for video_key in VIDEO_KEYS:
            relative = (
                Path("latents")
                / "chunk-000"
                / video_key
                / (f"episode_{episode_id:06d}_0_{length}.pth")
            )
            path = dataset / relative
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"missing Franka latent payload: {path}")
            record = _validate_latent_payload(
                path,
                episode_id=episode_id,
                length=length,
                video_key=video_key,
                encoder_identity_sha256=str(encoder_identity["identity_sha256"]),
            )
            record["path"] = relative.as_posix()
            records.append(record)
    core = {
        "schema_version": 1,
        "kind": "franka_video_latents",
        "dataset": "all600",
        "source_records_sha256": OFFICIAL_RECORDS_SHA256,
        "encoding_contract": {
            "source_fps": 15,
            "target_fps": 10,
            "height": 256,
            "width": 256,
            "video_keys": list(VIDEO_KEYS),
        },
        "encoder_source_identity": encoder_identity,
        "record_count": len(records),
        "records": records,
    }
    inventory = {**core, "inventory_sha256": canonical_sha256(core)}
    destination = artifacts / "franka_video_latent_inventory.json"
    _write_new_json(destination, inventory)
    return inventory


def verify_franka_training_artifacts(
    *,
    artifact_root: Path,
    lerobot_root: Path,
    base_model: Path,
    run_role: str,
    train_view_id: str | None = None,
    normalizer_source_view_id: str | None = None,
) -> VerifiedFrankaArtifacts:
    """Verify the inputs needed by a training launch.

    Task fine-tuning consumes the immutable all600 verification receipt instead
    of rehashing 600 parquet files, 1,200 latents, and the base encoder again.
    Development and all600 training retain the original exhaustive path.
    """

    artifacts = Path(artifact_root).expanduser().resolve(strict=True)
    dataset = Path(lerobot_root).expanduser().resolve(strict=True) / "all600"
    task_finetune = (
        run_role == "final_refit"
        and train_view_id in frozenset(TASK_FINETUNE_VIEWS.values())
    )
    _validate_lerobot_info(dataset)
    receipt_path = artifacts / "prepare_receipt.json"
    conversion_path = artifacts / "conversion_report.json"
    receipt = _json_object(receipt_path, label="Franka prepare receipt")
    conversion = _json_object(conversion_path, label="Franka conversion report")
    if (
        receipt.get("status") != "complete"
        or receipt.get("tactile_mode") != "disabled"
        or receipt.get("official_records_sha256") != OFFICIAL_RECORDS_SHA256
        or conversion.get("episode_count") != 600
        or conversion.get("tactile_mode") != "disabled"
        or conversion.get("source_action_schema") != SOURCE_ACTION_SCHEMA
        or conversion.get("derived_action_schema") != DERIVED_ACTION_SCHEMA
        or conversion.get("official_records_sha256") != OFFICIAL_RECORDS_SHA256
        or (
            not task_finetune
            and conversion.get("table_inventory")
            != build_lerobot_table_inventory(dataset)
        )
    ):
        raise ValueError("Franka conversion/receipt provenance mismatch")
    if receipt.get("conversion_report_sha256") != sha256_file(conversion_path):
        raise ValueError("Franka prepare receipt does not bind conversion bytes")

    views = {
        name: load_franka_view(artifacts / "views" / f"{name}.json")
        for name in (
            DEVELOPMENT_TRAIN_VIEW,
            DEVELOPMENT_VALIDATION_VIEW,
            FINAL_REFIT_VIEW,
        )
    }
    if run_role == "development":
        if train_view_id not in (None, DEVELOPMENT_TRAIN_VIEW):
            raise ValueError("development must use the development train view")
        if normalizer_source_view_id not in (None, DEVELOPMENT_TRAIN_VIEW):
            raise ValueError("development must use the development normalizer")
        train_view = views[DEVELOPMENT_TRAIN_VIEW]
        validation_view: FrankaDatasetView | None = views[DEVELOPMENT_VALIDATION_VIEW]
        normalizer_source_view = train_view
    elif run_role == "final_refit":
        selected_view_id = train_view_id or FINAL_REFIT_VIEW
        allowed = {FINAL_REFIT_VIEW, *TASK_FINETUNE_VIEWS.values()}
        if selected_view_id not in allowed:
            raise ValueError("final_refit train view is not an approved Franka view")
        if selected_view_id == FINAL_REFIT_VIEW:
            train_view = views[FINAL_REFIT_VIEW]
        else:
            train_view = load_franka_view(
                artifacts / "views" / f"{selected_view_id}.json"
            )
            expected = build_task_franka_views()[selected_view_id]
            if train_view != expected:
                raise ValueError("Franka task view must contain all 200 task episodes")
        validation_view = None
        selected_normalizer_id = normalizer_source_view_id or FINAL_REFIT_VIEW
        if selected_normalizer_id != FINAL_REFIT_VIEW:
            raise ValueError("final_refit must retain the all600 normalizer")
        normalizer_source_view = views[FINAL_REFIT_VIEW]
    else:
        raise ValueError("Franka run_role must be development or final_refit")
    normalizer = _validate_normalizer(
        artifacts / "normalizers" / f"{normalizer_source_view.view_id}.json",
        source_view=normalizer_source_view,
    )

    inventory_path = artifacts / "franka_video_latent_inventory.json"
    inventory = _json_object(inventory_path, label="Franka latent inventory")
    core = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    if (
        inventory.get("inventory_sha256") != canonical_sha256(core)
        or inventory.get("record_count") != 1200
        or inventory.get("source_records_sha256") != OFFICIAL_RECORDS_SHA256
    ):
        raise ValueError("Franka latent inventory canonical identity is invalid")
    recorded_encoder = validate_encoder_source_identity(
        inventory.get("encoder_source_identity")
    )
    if not task_finetune:
        current_encoder = validate_encoder_source_identity(
            build_encoder_source_identity(Path(base_model))
        )
        if recorded_encoder != current_encoder:
            raise ValueError("Franka latent encoder source differs from the base model")
    lengths = _episode_lengths(dataset)
    records = inventory.get("records")
    if not isinstance(records, list) or len(records) != 1200:
        raise ValueError("Franka latent inventory record set is incomplete")
    expected_pairs = {
        (episode_id, video_key) for episode_id in range(600) for video_key in VIDEO_KEYS
    }
    actual_pairs: set[tuple[int, str]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Franka latent inventory record is invalid")
        episode_id = int(record.get("episode_id", -1))
        video_key = str(record.get("video_key", ""))
        pair = (episode_id, video_key)
        if pair in actual_pairs or pair not in expected_pairs:
            raise ValueError("Franka latent inventory has duplicate/unknown records")
        actual_pairs.add(pair)
        expected_relative = (
            Path("latents")
            / "chunk-000"
            / video_key
            / (f"episode_{episode_id:06d}_0_{lengths[episode_id]}.pth")
        )
        if record.get("path") != expected_relative.as_posix():
            raise ValueError("Franka latent inventory path mismatch")
        if not task_finetune:
            path = dataset / expected_relative
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != record.get("size_bytes")
                or sha256_file(path) != record.get("sha256")
            ):
                raise ValueError(f"Franka latent payload bytes changed: {path}")
    if actual_pairs != expected_pairs:
        raise ValueError("Franka latent inventory pair set is incomplete")
    prepare_receipt_sha256 = sha256_file(receipt_path)
    conversion_report_sha256 = sha256_file(conversion_path)
    latent_inventory_sha256 = sha256_file(inventory_path)
    full_verification_receipt_sha256 = (
        _validate_full_verification_receipt(
            artifacts=artifacts,
            dataset=dataset,
            prepare_receipt_sha256=prepare_receipt_sha256,
            conversion_report_sha256=conversion_report_sha256,
            latent_inventory_sha256=latent_inventory_sha256,
            normalizer_sha256=normalizer.get("normalizer_sha256"),
        )
        if task_finetune
        else None
    )
    return VerifiedFrankaArtifacts(
        artifact_root=artifacts,
        dataset_root=dataset,
        train_view=train_view,
        validation_view=validation_view,
        normalizer_source_view=normalizer_source_view,
        normalizer=normalizer,
        conversion_report=conversion,
        latent_inventory=inventory,
        prepare_receipt_sha256=prepare_receipt_sha256,
        conversion_report_sha256=conversion_report_sha256,
        latent_inventory_sha256=latent_inventory_sha256,
        full_verification_receipt_sha256=full_verification_receipt_sha256,
    )


__all__ = (
    "MODEL_ACTION_SCHEMA",
    "SOURCE_ACTION_SCHEMA",
    "FULL_VERIFICATION_RECEIPT",
    "VIDEO_KEYS",
    "VerifiedFrankaArtifacts",
    "finalize_franka_video_latents",
    "verify_franka_training_artifacts",
)
