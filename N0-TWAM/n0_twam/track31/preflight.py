# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed filesystem preflight for N0 UniVTAC Track 3.1 training."""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
    validate_recorded_transformer_identity,
    validate_sha256,
    validate_transformer_identity_match,
)
from n0_twam.checkpointing.runtime_provenance import (
    capture_checkpoint_invocation_identity,
    capture_runtime_source_identity,
    validate_checkpoint_runtime_provenance,
)
from n0_twam.checkpointing.sidecar_snapshot import (
    capture_stable_json_file,
    strict_integer,
)
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    capture_strict_checkpoint_snapshot,
)
from n0_twam.checkpointing.strict_resume import (
    validate_scheduler_state,
)
from n0_twam.checkpointing.training_lineage import (
    validate_action_migration_report,
    validate_stage_a_parent_checkpoint,
)
from n0_twam.configs.twam_track31_training_profiles import (
    MULTITASK_PRETRAIN_PROFILE,
    build_track31_training_profile_contract,
    resolve_track31_run_role,
    resolve_track31_train_profile,
)
from n0_twam.configs.twam_track31_univtac_cfg import (
    build_track31_invocation_contract,
    build_track31_tactile_training_contract,
    resolve_track31_load_worker,
    resolve_track31_max_latent_frames,
    resolve_track31_stop_after_step,
    resolve_track31_training_overrides,
    twam_track31_univtac_cfg,
    validate_track31_checkpoint_tactile_contract,
)
from n0_twam.data.latent_inventory import (
    build_encoder_source_identity,
    validate_latent_inventory_pair,
)
from n0_twam.data.track31_training_identity import (
    build_track31_profile_identity,
    validate_checkpoint_latent_inventory_binding,
)
from n0_twam.distributed.optimizer_checkpoint import (
    OPTIMIZER_STATE_FORMAT,
    build_training_execution_contract,
    validate_optimizer_checkpoint,
    validate_runtime_signature,
    validate_training_execution_contract,
)
from n0_twam.distributed.fsdp import _activation_checkpointing_enabled
from n0_twam.integrations.univtac.artifact_contracts import (
    verify_track31_training_bundle,
)
from n0_twam.models.model import capture_attention_execution_contract
from n0_twam.track31.checkpoint_preflight import (
    audit_tactile_sensor_embedding_capacity,
)


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable is unset: {name}")
    return Path(value).expanduser().resolve(strict=True)


def _required_env_sha256(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"required environment variable is unset: {name}")
    return validate_sha256(value, label=name)


def _artifact_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, str(default))
    return Path(value).expanduser().resolve(strict=True)


def _load_json_mapping(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _optional_env_checkpoint(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser().resolve(strict=True)


def _resolve_profile_contract() -> dict[str, object]:
    """Resolve the profile route from the current launch environment."""
    training_profile_id = resolve_track31_train_profile()
    run_role = resolve_track31_run_role()
    if run_role == "final_refit" and os.environ.get("N0_TRACK31_VALIDATION_VIEW_PATH"):
        raise ValueError("final_refit must not use a validation view")

    resume_from = _optional_env_checkpoint("N0_TRACK31_RESUME_FROM")
    init_from = _optional_env_checkpoint("N0_TRACK31_INIT_FROM")
    # A stale released-prior variable is irrelevant to Stage B and strict
    # resumes; only the selected checkpoint route is resolved below.
    released_from = os.environ.get("N0_RELEASED_CHECKPOINT")
    parent = resume_from or init_from
    parent_train_meta = (
        None
        if parent is None
        else _load_json_mapping(
            parent / "train_meta.json", label="parent train metadata"
        )
    )
    return cast(
        dict[str, object],
        build_track31_training_profile_contract(
            training_profile_id=training_profile_id,
            run_role=run_role,
            released_checkpoint=released_from,
            init_from=init_from,
            resume_from=resume_from,
            parent_train_meta=parent_train_meta,
        ),
    )


def _expected_profile_identity(
    profile_contract: Mapping[str, object],
    artifacts: Mapping[str, object] | None,
) -> dict[str, object]:
    if artifacts is None:
        raise ValueError("formal profile identity requires verified artifacts")
    train_view_id = profile_contract["train_view_id"]
    validation_view_id = profile_contract["validation_view_id"]
    normalizer_source_view_id = profile_contract["normalizer_source_view_id"]
    if artifacts.get("train_view_id") not in (None, train_view_id):
        raise ValueError("verified training view does not match profile")
    if artifacts.get("validation_view_id") not in (None, validation_view_id):
        raise ValueError("verified validation view does not match profile")
    if artifacts.get("normalizer_source_view_id") not in (
        None,
        normalizer_source_view_id,
    ):
        raise ValueError("verified normalizer source view does not match profile")
    identity = build_track31_profile_identity(
        {
            **profile_contract,
            "sampler_coverage_mode": "pad_global",
            "source_manifest_sha256": artifacts.get("manifest_sha256"),
            "normalizer_sha256": artifacts.get("normalizer_sha256"),
            "train_view_sha256": artifacts.get("train_view_sha256"),
            "validation_view_sha256": artifacts.get("validation_view_sha256"),
            "normalizer_source_view_sha256": artifacts.get(
                "normalizer_source_view_sha256"
            ),
            "video_inventory_sha256": artifacts.get("video_inventory_sha256"),
            "tactile_inventory_sha256": artifacts.get("tactile_inventory_sha256"),
        }
    )
    if identity is None:
        raise ValueError("formal profile identity was not constructed")
    return identity


def _require_files(root: Path, relative_paths: tuple[str, ...]) -> None:
    missing = [
        str(root / relative_path)
        for relative_path in relative_paths
        if not (root / relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))


def _audit_optimizer_dcp(
    checkpoint: Path,
    *,
    expected_inventory_sha256: str,
) -> dict[str, object]:
    optimizer_dcp = checkpoint / "optimizer_dcp"
    if not optimizer_dcp.is_dir():
        raise FileNotFoundError(
            f"missing sharded optimizer checkpoint directory: {optimizer_dcp}"
        )
    inventory = validate_optimizer_checkpoint(
        optimizer_dcp,
        expected_inventory_sha256=expected_inventory_sha256,
    )
    return cast(dict[str, object], inventory.to_json_dict())


def _audit_lerobot_split(
    root: Path,
    split: str,
    *,
    expected_inventory_split: str | None = None,
    expected_manifest_sha256: str,
    expected_conversion_report_sha256: str,
    expected_encoder_source_identity: object,
    prevalidated_latent_inventory: Mapping[str, object] | None = None,
) -> dict[str, object]:
    split_root = root / split
    _require_files(split_root, ("meta/info.json", "meta/episodes.jsonl"))
    info = json.loads((split_root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{split} is not LeRobot v2.1")
    features = info.get("features", {})
    required_features = {
        "observation.images.top",
        "observation.images.wrist_l",
        "observation.images.tactile_a",
        "observation.images.tactile_b",
        "observation.state",
        "action",
    }
    missing_features = sorted(required_features - set(features))
    if missing_features:
        raise ValueError(f"{split} is missing features: {missing_features}")
    if features["action"].get("shape") != [8]:
        raise ValueError(f"{split} action feature is not 8D qpos")
    latent_inventory = (
        dict(prevalidated_latent_inventory)
        if prevalidated_latent_inventory is not None
        else validate_latent_inventory_pair(
            split_root,
            expected_split=expected_inventory_split or split,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_conversion_report_sha256=expected_conversion_report_sha256,
            expected_encoder_source_identity=expected_encoder_source_identity,
        )
    )
    return {
        "root": str(split_root),
        "total_episodes": int(info.get("total_episodes", 0)),
        "total_frames": int(info.get("total_frames", 0)),
        "latent_inventory": latent_inventory,
    }


def _build_invocation_contract(
    checkpoint_report: Mapping[str, object],
) -> dict[str, int | bool]:
    """Build the mutable-per-launch contract after auditing resume progress."""
    training_overrides = resolve_track31_training_overrides()
    num_steps = training_overrides["num_steps"]
    raw_start_step = checkpoint_report.get("step", 0)
    if isinstance(raw_start_step, bool) or not isinstance(raw_start_step, int):
        raise ValueError("checkpoint report step must be an integer")
    start_step = raw_start_step
    stop_after_step = resolve_track31_stop_after_step(num_steps=num_steps)
    return cast(
        dict[str, int | bool],
        build_track31_invocation_contract(
            start_step=start_step,
            stop_after_step=stop_after_step,
            num_steps=num_steps,
        ),
    )


def _audit_checkpoint_tactile_contract(
    checkpoint_cfg: Mapping[str, object],
    *,
    train_meta: Mapping[str, object] | None,
    released_prior: bool,
) -> dict[str, object]:
    """Require model capacity and active UniVTAC streams to stay distinct."""
    expected = build_track31_tactile_training_contract(twam_track31_univtac_cfg)
    return cast(
        dict[str, object],
        validate_track31_checkpoint_tactile_contract(
            transformer_config=checkpoint_cfg,
            train_meta=train_meta,
            expected_contract=expected,
            released_prior=released_prior,
        ),
    )


def _validate_saved_artifact_identity(
    train_meta: Mapping[str, object],
    *,
    expected_track31_artifacts: Mapping[str, object] | None,
    label: str,
) -> Mapping[str, object]:
    saved_artifacts = train_meta.get("track31_artifacts")
    identity_keys = (
        "manifest_sha256",
        "normalizer_sha256",
        "conversion_report_sha256",
        "video_inventory_sha256",
        "tactile_inventory_sha256",
    )
    if not isinstance(saved_artifacts, Mapping):
        raise ValueError(f"{label} has no Track3.1 artifact identity")
    for key in identity_keys:
        identity_label = (
            f"{label} latent inventory {key}"
            if key.endswith("inventory_sha256")
            else f"{label} {key}"
        )
        validate_sha256(saved_artifacts.get(key), label=identity_label)
    if expected_track31_artifacts is not None and any(
        saved_artifacts.get(key) != expected_track31_artifacts.get(key)
        for key in identity_keys
    ):
        raise ValueError(f"{label} UniVTAC artifact identity mismatch")
    return saved_artifacts


def _audit_stage_a_parent(
    checkpoint: Path,
    *,
    profile_contract: Mapping[str, object],
    expected_track31_artifacts: Mapping[str, object] | None,
) -> dict[str, object]:
    """Validate a Stage A model-only parent for a fresh Stage B run."""
    run_role = profile_contract.get("run_role")
    if not isinstance(run_role, str):
        raise ValueError("Stage B profile contract has no run role")
    parent = validate_stage_a_parent_checkpoint(
        checkpoint,
        expected_run_role=run_role,
        expected_track31_artifacts=expected_track31_artifacts,
    )
    train_meta = parent["train_meta"]
    if not isinstance(train_meta, Mapping):
        raise ValueError("Stage A parent train metadata is invalid")
    checkpoint_cfg = parent["transformer_config"]
    if not isinstance(checkpoint_cfg, Mapping):
        raise ValueError("Stage A parent transformer config is invalid")
    tactile_contract = _audit_checkpoint_tactile_contract(
        checkpoint_cfg,
        train_meta=train_meta,
        released_prior=False,
    )
    tactile_sensor_embedding_shapes = audit_tactile_sensor_embedding_capacity(
        checkpoint / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
        tactile_contract=tactile_contract,
    )
    return {
        "mode": "stage_a_weights_only",
        "source_action_dim": 8,
        "source_action_schema": "qpos8_next_step",
        "target_action_dim": 8,
        "target_action_schema": "qpos8_next_step",
        "parent_training_profile_id": MULTITASK_PRETRAIN_PROFILE,
        "parent_transformer_identity": parent["transformer_identity"],
        "runtime_training_lineage": parent["runtime_training_lineage"],
        "action_migration_report": parent["action_migration_report"],
        "reset_optimizer": True,
        "reset_scheduler": True,
        "reset_rng": True,
        "reset_data_cursor": True,
        "tactile_training_contract": tactile_contract,
        "tactile_sensor_embedding_shapes": tactile_sensor_embedding_shapes,
    }


def _audit_checkpoint(
    *,
    expected_track31_artifacts: Mapping[str, object] | None = None,
) -> dict[str, object]:
    profile_contract = _resolve_profile_contract()
    initialization_mode = str(profile_contract["initialization_mode"])
    checkpoint = Path(str(profile_contract["checkpoint_path"])).resolve(strict=True)
    _require_files(
        checkpoint,
        ("transformer/diffusion_pytorch_model.safetensors",),
    )

    if initialization_mode == "stage_a_weights_only":
        runtime_source_identity = capture_runtime_source_identity()
        checkpoint_invocation_identity = capture_checkpoint_invocation_identity()
        return {
            **_audit_stage_a_parent(
                checkpoint,
                profile_contract=profile_contract,
                expected_track31_artifacts=expected_track31_artifacts,
            ),
            "training_profile_contract": profile_contract,
            "runtime_source_identity": runtime_source_identity,
            "checkpoint_invocation_identity": checkpoint_invocation_identity,
        }

    if initialization_mode == "released_action_migration":
        checkpoint_cfg = capture_stable_json_file(
            checkpoint / "transformer" / "config.json",
            label="released transformer config",
        ).json_object(label="released transformer config")
        if checkpoint_cfg.get("is_mot") is not True:
            raise ValueError("checkpoint is not a MoT checkpoint")
        if (
            strict_integer(
                checkpoint_cfg.get("action_dim"),
                label="released transformer action_dim",
                minimum=1,
            )
            != 20
        ):
            raise ValueError("expected released N0 prior with 20D EE action head")
        recorded_schema = checkpoint_cfg.get("action_schema")
        if recorded_schema not in (None, "ee20_pi05"):
            raise ValueError("released checkpoint action schema is not EE20")
        expected_sha256 = _required_env_sha256("N0_RELEASED_TRANSFORMER_SHA256")
        transformer_identity = audit_transformer_checkpoint(
            checkpoint / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
            expected_action_dim=20,
        )
        if transformer_identity["sha256"] != expected_sha256:
            raise ValueError(
                "released transformer SHA256 does not match "
                "N0_RELEASED_TRANSFORMER_SHA256"
            )
        tactile_contract = _audit_checkpoint_tactile_contract(
            checkpoint_cfg,
            train_meta=None,
            released_prior=True,
        )
        tactile_sensor_embedding_shapes = audit_tactile_sensor_embedding_capacity(
            checkpoint / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
            tactile_contract=tactile_contract,
        )
        runtime_source_identity = capture_runtime_source_identity()
        checkpoint_invocation_identity = capture_checkpoint_invocation_identity()
        return {
            "mode": "migrate_action",
            "source_action_dim": 20,
            "source_action_schema": "ee20_pi05",
            "target_action_dim": 8,
            "target_action_schema": "qpos8_next_step",
            "transformer_identity": transformer_identity,
            "tactile_training_contract": tactile_contract,
            "tactile_sensor_embedding_shapes": tactile_sensor_embedding_shapes,
            "training_profile_contract": profile_contract,
            "runtime_source_identity": runtime_source_identity,
            "checkpoint_invocation_identity": checkpoint_invocation_identity,
        }

    if initialization_mode != "strict_full_state_resume":
        raise ValueError(
            f"unsupported Track3.1 initialization mode: {initialization_mode}"
        )

    strict_snapshot = capture_strict_checkpoint_snapshot(checkpoint)
    completion = strict_snapshot.completion
    sidecar_inventory = strict_snapshot.sidecars.inventory
    training_state = strict_snapshot.training_state
    train_meta = strict_snapshot.train_meta
    checkpoint_cfg = strict_snapshot.transformer_config
    runtime_source_identity = capture_runtime_source_identity()
    checkpoint_invocation_identity = capture_checkpoint_invocation_identity()
    validate_checkpoint_runtime_provenance(
        (
            ("resume train metadata", train_meta),
            ("resume training state", training_state),
            ("resume completion marker", completion),
        ),
        current_runtime_source_identity=runtime_source_identity,
    )
    if train_meta.get("action_schema") != "qpos8_next_step":
        raise ValueError("resume train metadata action schema is not qpos8_next_step")
    _validate_saved_artifact_identity(
        train_meta,
        expected_track31_artifacts=expected_track31_artifacts,
        label="resume train metadata",
    )
    saved_track31_artifacts = train_meta.get("track31_artifacts")
    identity_artifacts = (
        expected_track31_artifacts
        if expected_track31_artifacts is not None
        else saved_track31_artifacts
    )
    if not isinstance(identity_artifacts, Mapping):
        raise ValueError("resume checkpoint has no Track3.1 artifact identity")
    latent_inventory_identity = validate_checkpoint_latent_inventory_binding(
        expected=identity_artifacts,
        payloads=(
            ("resume train metadata", train_meta),
            ("resume training state", training_state),
            ("resume completion marker", completion),
        ),
    )
    expected_profile_identity = _expected_profile_identity(
        profile_contract,
        identity_artifacts,
    )
    if (
        train_meta.get("training_profile_id") != profile_contract["training_profile_id"]
        or train_meta.get("run_role") != profile_contract["run_role"]
        or train_meta.get("training_profile_identity") != expected_profile_identity
        or training_state.get("training_profile_identity") != expected_profile_identity
        or completion.get("training_profile_identity") != expected_profile_identity
    ):
        raise ValueError("resume Track3.1 training profile identity mismatch")
    tactile_contract = _audit_checkpoint_tactile_contract(
        checkpoint_cfg,
        train_meta=train_meta,
        released_prior=False,
    )
    tactile_sensor_embedding_shapes = audit_tactile_sensor_embedding_capacity(
        checkpoint / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
        tactile_contract=tactile_contract,
    )
    if (
        strict_integer(
            checkpoint_cfg.get("action_dim"),
            label="resume transformer action_dim",
            minimum=1,
        )
        != 8
    ):
        raise ValueError("resume checkpoint action head is not qpos8")
    transformer_identity = audit_transformer_checkpoint(
        checkpoint / "transformer" / TRANSFORMER_WEIGHTS_FILENAME,
        expected_action_dim=8,
    )
    validate_transformer_identity_match(
        train_meta.get("transformer_identity"),
        transformer_identity,
        expected_action_dim=8,
        label="train metadata",
    )
    validate_transformer_identity_match(
        training_state.get("transformer_identity"),
        transformer_identity,
        expected_action_dim=8,
        label="training state",
    )
    validate_transformer_identity_match(
        completion.get("transformer_identity"),
        transformer_identity,
        expected_action_dim=8,
        label="completion marker",
    )
    for payload in (checkpoint_cfg, training_state):
        if payload.get("action_schema") != "qpos8_next_step":
            raise ValueError("resume checkpoint action schema is not qpos8_next_step")
    migration = validate_action_migration_report(
        strict_snapshot.action_migration_report
    )
    source_identity = validate_recorded_transformer_identity(
        migration.get("source_transformer_identity"),
        expected_action_dim=20,
    )
    source_sha256 = validate_sha256(
        migration.get("source_checkpoint_sha256"),
        label="migration source checkpoint SHA256",
    )
    if source_identity["sha256"] != source_sha256:
        raise ValueError("migration source transformer identity is inconsistent")
    world_size = strict_snapshot.world_size
    expected_world_size = int(
        os.environ.get("N0_TRACK31_EXPECTED_WORLD_SIZE", str(world_size))
    )
    if world_size != expected_world_size:
        raise ValueError(
            "resume checkpoint world_size does not match requested launch: "
            f"{world_size} vs {expected_world_size}"
        )
    step = strict_snapshot.step
    optimizer_inventory_sha256 = training_state.get("optimizer_inventory_sha256")
    runtime_signature = training_state.get("runtime_signature")
    training_execution_contract = training_state.get("training_execution_contract")
    optimizer_inventory_sha256 = validate_sha256(
        optimizer_inventory_sha256,
        label="resume optimizer inventory SHA256",
    )
    if (
        completion.get("optimizer_state_format") != OPTIMIZER_STATE_FORMAT
        or completion.get("optimizer_inventory_sha256") != optimizer_inventory_sha256
        or completion.get("runtime_signature") != runtime_signature
        or completion.get("training_execution_contract") != training_execution_contract
        or completion.get("training_profile_identity")
        != training_state.get("training_profile_identity")
        or completion.get("transformer_identity")
        != training_state.get("transformer_identity")
        or completion.get("status") != "complete"
        or strict_snapshot.step != step
        or strict_snapshot.world_size != world_size
        or completion.get("action_schema") != "qpos8_next_step"
    ):
        raise ValueError("resume checkpoint completion marker is inconsistent")
    optimizer_inventory = _audit_optimizer_dcp(
        checkpoint,
        expected_inventory_sha256=optimizer_inventory_sha256,
    )
    validate_runtime_signature(runtime_signature)
    if train_meta.get("training_execution_contract") != training_execution_contract:
        raise ValueError("resume train metadata execution contract is inconsistent")
    training_overrides = resolve_track31_training_overrides()
    current_execution_contract = build_training_execution_contract(
        max_latent_frames=resolve_track31_max_latent_frames(),
        gradient_accumulation_steps=training_overrides["gradient_accumulation_steps"],
        batch_size=training_overrides["batch_size"],
        load_worker=resolve_track31_load_worker(),
        num_steps=training_overrides["num_steps"],
        lr_schedule=str(twam_track31_univtac_cfg.lr_schedule),
        warmup_steps=int(twam_track31_univtac_cfg.warmup_steps),
        lr_min_ratio=float(twam_track31_univtac_cfg.lr_min_ratio),
        activation_checkpointing=_activation_checkpointing_enabled(),
        attention_contract=capture_attention_execution_contract(),
    )
    training_execution_contract = validate_training_execution_contract(
        training_execution_contract,
        current_contract=current_execution_contract,
    )
    scheduler_state = validate_scheduler_state(
        strict_snapshot.sidecars.json_object(
            "scheduler_state.json",
            label="resume scheduler state",
        ),
        completed_steps=step,
        learning_rate=float(twam_track31_univtac_cfg.learning_rate),
        execution_contract=training_execution_contract,
    )
    for rank in range(world_size):
        strict_snapshot.sidecars.load_rng_state(
            rank=rank,
            expected_world_size=world_size,
        )
    return {
        "mode": "strict_resume",
        "source_action_dim": 8,
        "source_action_schema": "qpos8_next_step",
        "target_action_dim": 8,
        "target_action_schema": "qpos8_next_step",
        "step": step,
        "world_size": world_size,
        "optimizer_state_format": OPTIMIZER_STATE_FORMAT,
        "optimizer_inventory_sha256": optimizer_inventory_sha256,
        "optimizer_inventory": optimizer_inventory,
        "sidecar_inventory": sidecar_inventory,
        "scheduler_state": scheduler_state,
        "runtime_signature": runtime_signature,
        "training_execution_contract": training_execution_contract,
        "transformer_identity": transformer_identity,
        "migration_source_transformer_identity": source_identity,
        "tactile_training_contract": tactile_contract,
        "tactile_sensor_embedding_shapes": tactile_sensor_embedding_shapes,
        "training_profile_contract": profile_contract,
        "training_profile_identity": expected_profile_identity,
        "runtime_source_identity": runtime_source_identity,
        "checkpoint_invocation_identity": checkpoint_invocation_identity,
        "parent_checkpoint_invocation_identity": (
            strict_snapshot.checkpoint_invocation_identity
        ),
        **latent_inventory_identity,
    }


def main() -> int:
    artifact_root = _required_env_path("N0_TRACK31_ARTIFACT_ROOT")
    dataset_root = _required_env_path("N0_TRACK31_LEROBOT_ROOT")
    base_model = _required_env_path("N0_BASE_MODEL")
    empty_embedding = _required_env_path("N0_EMPTY_EMBEDDING")
    if not empty_embedding.is_file():
        raise FileNotFoundError(empty_embedding)

    profile_contract = _resolve_profile_contract()
    train_view_id = str(profile_contract["train_view_id"])
    validation_view_value = profile_contract["validation_view_id"]
    validation_view_id = (
        None if validation_view_value is None else str(validation_view_value)
    )
    normalizer_id = str(profile_contract["normalizer_id"])
    normalizer_source_view_id = {
        "qpos8_dev719_v1": "stage_a_dev719_v1",
        "qpos8_final759_v1": "stage_a_final759_v1",
    }[normalizer_id]
    manifest_path = _artifact_path(
        "N0_TRACK31_MANIFEST_PATH", artifact_root / "universe_manifest_v4.json"
    )
    normalizer_path = _artifact_path(
        "N0_TRACK31_NORMALIZER_PATH",
        artifact_root / "normalizers" / f"{normalizer_id}.json",
    )
    train_view_path = _artifact_path(
        "N0_TRACK31_TRAIN_VIEW_PATH",
        artifact_root / "views" / f"{train_view_id}.json",
    )
    normalizer_source_view_path = _artifact_path(
        "N0_TRACK31_NORMALIZER_SOURCE_VIEW_PATH",
        artifact_root / "views" / f"{normalizer_source_view_id}.json",
    )
    validation_view_path = (
        None
        if validation_view_id is None
        else _artifact_path(
            "N0_TRACK31_VALIDATION_VIEW_PATH",
            artifact_root / "views" / f"{validation_view_id}.json",
        )
    )
    _require_files(
        base_model,
        (
            "vae/config.json",
            "tokenizer/tokenizer_config.json",
            "text_encoder/config.json",
        ),
    )
    encoder_source_identity = build_encoder_source_identity(base_model)
    artifacts = verify_track31_training_bundle(
        manifest_path=manifest_path,
        normalizer_path=normalizer_path,
        conversion_report_path=artifact_root / "conversion_report.json",
        dataset_root=dataset_root,
        train_view_path=train_view_path,
        validation_view_path=validation_view_path,
        normalizer_source_view_path=normalizer_source_view_path,
        parent_validation_view_path=(
            artifact_root / "views" / "internal_dev40_v1.json"
            if profile_contract["training_profile_id"] == "target_finetune_v1"
            and profile_contract["run_role"] == "development"
            else None
        ),
        encoder_source_identity=encoder_source_identity,
    )
    checkpoint_report = _audit_checkpoint(
        expected_track31_artifacts=artifacts.to_json_dict()
    )
    invocation_contract = _build_invocation_contract(checkpoint_report)
    conversion_report_sha256 = artifacts.conversion_report_sha256
    if conversion_report_sha256 is None:
        raise ValueError("verified training bundle has no conversion report digest")

    report = {
        "schema_version": 1,
        "status": "ready_for_training",
        "training_profile_contract": profile_contract,
        "artifacts": artifacts.to_json_dict(),
        "checkpoint": checkpoint_report,
        "invocation_contract": invocation_contract,
        "splits": {
            "train759": _audit_lerobot_split(
                dataset_root,
                "train759",
                expected_manifest_sha256=artifacts.manifest_sha256,
                expected_conversion_report_sha256=conversion_report_sha256,
                expected_encoder_source_identity=encoder_source_identity,
                prevalidated_latent_inventory=artifacts.latent_inventory_report(),
            )
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
