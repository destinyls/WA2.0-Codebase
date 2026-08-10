# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from n0_twam.checkpointing.identity import audit_transformer_checkpoint
from n0_twam.checkpointing.strict_resume import (
    build_sidecar_inventory,
    capture_rng_state,
    expected_sidecar_paths,
    save_rng_state,
    save_scheduler_state,
)
from n0_twam.configs.twam_track31_univtac_cfg import (
    TRACK31_TRANSFORMER_TACTILE_FIELDS,
    build_track31_tactile_training_contract,
    twam_track31_univtac_cfg,
)
from n0_twam.distributed.optimizer_checkpoint import (
    OPTIMIZER_STATE_FORMAT,
    STRICT_CHECKPOINT_SCHEMA_VERSION,
    build_training_execution_contract,
    capture_runtime_signature,
    write_optimizer_checkpoint_inventory,
)
from n0_twam.models.model import capture_attention_execution_contract
from n0_twam.utils.utils import warmup_cosine_lambda
from script.track3_1.preflight_train import (
    _audit_checkpoint,
    _audit_lerobot_split,
    _build_invocation_contract,
)

_TRACK31_ARTIFACT_IDENTITY = {
    "manifest_sha256": "b" * 64,
    "normalizer_sha256": "c" * 64,
    "conversion_report_sha256": "d" * 64,
    "train_view_id": "stage_a_final759_v1",
    "train_view_sha256": "e" * 64,
    "validation_view_id": None,
    "validation_view_sha256": None,
    "normalizer_source_view_id": "stage_a_final759_v1",
    "normalizer_source_view_sha256": "e" * 64,
    "video_inventory_sha256": "1" * 64,
    "tactile_inventory_sha256": "2" * 64,
}
_TRACK31_TACTILE_CONTRACT = build_track31_tactile_training_contract(
    twam_track31_univtac_cfg
)
_STAGE_A = "multitask_pretrain_v1"
_RUNTIME_SOURCE_IDENTITY = {
    "schema_version": 1,
    "code_manifest_sha256": "3" * 64,
    "image_id": "sha256:" + "4" * 64,
    "overlay_manifest_sha256": "5" * 64,
    "empty_embedding_sha256": "6" * 64,
}
_CHECKPOINT_INVOCATION_IDENTITY = {
    "schema_version": 1,
    "invocation_id": "formal-test-001",
    "launch_manifest_sha256": "7" * 64,
}


def _set_current_runtime_provenance_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("N0_TRACK31_SOURCE_MANIFEST_SHA256", "3" * 64)
    monkeypatch.setenv("N0_TRACK31_IMAGE_ID", "sha256:" + "4" * 64)
    monkeypatch.setenv("N0_TRACK31_OVERLAY_MANIFEST_SHA256", "5" * 64)
    monkeypatch.setenv("N0_EMPTY_EMBEDDING_SHA256", "6" * 64)
    monkeypatch.setenv("N0_TRACK31_INVOCATION_ID", "formal-test-002")
    monkeypatch.setenv("N0_TRACK31_LAUNCH_MANIFEST_SHA256", "8" * 64)


@pytest.fixture(autouse=True)
def _fixed_track31_execution_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("N0_TRACK31_MAX_LATENT_FRAMES", "5")
    monkeypatch.setenv("N0_TRACK31_GRADIENT_ACCUMULATION_STEPS", "4")
    monkeypatch.setenv("N0_TRACK31_TRAIN_PROFILE", _STAGE_A)
    monkeypatch.setenv("N0_TRACK31_RUN_ROLE", "final_refit")
    monkeypatch.delenv("N0_TRACK31_STOP_AFTER_STEP", raising=False)
    monkeypatch.delenv("N0_TRACK31_INIT_FROM", raising=False)
    monkeypatch.delenv("N0_TRACK31_RESUME_FROM", raising=False)
    monkeypatch.delenv("N0_TRACK31_VALIDATION_VIEW_PATH", raising=False)
    _set_current_runtime_provenance_env(monkeypatch)


def _write_transformer_checkpoint(
    root: Path,
    *,
    action_dim: int,
    action_schema: str | None,
    fill_value: float = 0.0,
    sensor_capacity: int = 4,
) -> str:
    transformer = root / "transformer"
    transformer.mkdir(parents=True, exist_ok=True)
    config = {
        "is_mot": True,
        "action_dim": action_dim,
        **{
            field: _TRACK31_TACTILE_CONTRACT[field]
            for field in TRACK31_TRANSFORMER_TACTILE_FIELDS
        },
    }
    config["use_local_tactile"] = action_dim == 8
    if action_dim == 8:
        config["snr_shift"] = _TRACK31_TACTILE_CONTRACT["snr_shift"]
        config["tactile_latent_channels"] = _TRACK31_TACTILE_CONTRACT[
            "tactile_latent_channels"
        ]
    if action_schema is not None:
        config["action_schema"] = action_schema
    (transformer / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "action_embedder.weight": torch.full((3072, action_dim), fill_value),
        "action_embedder.bias": torch.zeros(3072),
        "action_proj_out.weight": torch.zeros(action_dim, 3072),
        "action_proj_out.bias": torch.zeros(action_dim),
        "condition_embedder.text_embedder.linear_1.weight": torch.zeros(1),
        "mot.experts.action.in_proj.weight": torch.zeros(1),
        "mot.experts.tactile.in_proj.weight": torch.zeros(1),
        "sensor_id_embed.weight": torch.zeros(sensor_capacity, 3072),
    }
    if action_dim == 8:
        tensors["local_tactile_sensor_embed.weight"] = torch.zeros(
            sensor_capacity, 3072
        )
    weights_path = transformer / "diffusion_pytorch_model.safetensors"
    save_file(tensors, weights_path)
    return hashlib.sha256(weights_path.read_bytes()).hexdigest()


def test_preflight_rejects_empty_latent_directories(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    (split_root / "meta").mkdir(parents=True)
    required_images = (
        "observation.images.top",
        "observation.images.wrist_l",
        "observation.images.tactile_a",
        "observation.images.tactile_b",
    )
    features = {key: {"dtype": "video", "shape": [8, 8, 3]} for key in required_images}
    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [8]},
        }
    )
    (split_root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "features": features}),
        encoding="utf-8",
    )
    (split_root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 5}) + "\n",
        encoding="utf-8",
    )
    for directory in (
        "latents",
        "latents_tactile/global",
        "latents_tactile/local",
    ):
        (split_root / directory).mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="completion inventory"):
        _audit_lerobot_split(
            tmp_path,
            "train",
            expected_manifest_sha256="1" * 64,
            expected_conversion_report_sha256="2" * 64,
            expected_encoder_source_identity={},
        )


def _write_resume_sidecars(
    root: Path,
    *,
    world_size: int,
    training_profile_id: str = _STAGE_A,
    run_role: str = "final_refit",
    write_optimizer_metadata: bool = True,
    write_optimizer_shard: bool = True,
) -> dict[str, object]:
    transformer_identity = audit_transformer_checkpoint(
        root / "transformer" / "diffusion_pytorch_model.safetensors",
        expected_action_dim=8,
    )
    source_identity = {
        **transformer_identity,
        "sha256": "a" * 64,
        "action_dim": 20,
        "action_shapes": {
            "action_embedder.weight": [3072, 20],
            "action_embedder.bias": [3072],
            "action_proj_out.weight": [20, 3072],
            "action_proj_out.bias": [20],
        },
    }
    (root / "action_migration_report.json").write_text(
        json.dumps(
            {
                "source_action_dim": 20,
                "source_action_schema": "ee20_pi05",
                "target_action_dim": 8,
                "target_action_schema": "qpos8_next_step",
                "source_checkpoint_sha256": source_identity["sha256"],
                "source_transformer_identity": source_identity,
            }
        ),
        encoding="utf-8",
    )
    optimizer_dcp = root / "optimizer_dcp"
    optimizer_dcp.mkdir()
    (optimizer_dcp / ".metadata").write_bytes(b"metadata")
    (optimizer_dcp / "__0_0.distcp").write_bytes(b"optimizer")
    inventory = write_optimizer_checkpoint_inventory(optimizer_dcp)
    if not write_optimizer_metadata:
        (optimizer_dcp / ".metadata").unlink()
    if not write_optimizer_shard:
        (optimizer_dcp / "__0_0.distcp").unlink()
    runtime_signature = capture_runtime_signature()
    training_execution_contract = build_training_execution_contract(
        max_latent_frames=5,
        gradient_accumulation_steps=4,
        batch_size=1,
        load_worker=0,
        num_steps=2000,
        lr_schedule="cosine",
        warmup_steps=20,
        lr_min_ratio=0.1,
        activation_checkpointing=True,
        attention_contract=capture_attention_execution_contract(),
    )
    if training_profile_id == _STAGE_A:
        train_view_id = (
            "stage_a_dev719_v1" if run_role == "development" else "stage_a_final759_v1"
        )
        validation_view_id = "internal_dev40_v1" if run_role == "development" else None
    else:
        train_view_id = (
            "stage_b_dev180_v1" if run_role == "development" else "stage_b_final190_v1"
        )
        validation_view_id = (
            "internal_target_dev10_v1" if run_role == "development" else None
        )
    training_profile_identity = {
        "schema_version": 2,
        "training_profile_id": training_profile_id,
        "run_role": run_role,
        "train_view_id": train_view_id,
        "sampler_coverage_mode": "pad_global",
        "normalizer_source_view_id": (
            "stage_a_dev719_v1" if run_role == "development" else "stage_a_final759_v1"
        ),
        "validation_view_id": validation_view_id,
        "source_manifest_sha256": "b" * 64,
        "normalizer_sha256": "c" * 64,
        "train_view_sha256": "e" * 64,
        "validation_view_sha256": (
            "f" * 64 if validation_view_id is not None else None
        ),
        "normalizer_source_view_sha256": "e" * 64,
        "video_inventory_sha256": "1" * 64,
        "tactile_inventory_sha256": "2" * 64,
    }
    track31_artifacts = {
        **_TRACK31_ARTIFACT_IDENTITY,
        "train_view_id": train_view_id,
        "train_view_sha256": "e" * 64,
        "validation_view_id": validation_view_id,
        "validation_view_sha256": (
            "f" * 64 if validation_view_id is not None else None
        ),
        "normalizer_source_view_id": training_profile_identity[
            "normalizer_source_view_id"
        ],
        "normalizer_source_view_sha256": "e" * 64,
    }
    (root / "train_meta.json").write_text(
        json.dumps(
            {
                "action_schema": "qpos8_next_step",
                "training_profile_id": training_profile_id,
                "run_role": run_role,
                "train_view_id": train_view_id,
                "validation_view_id": validation_view_id,
                "training_profile_identity": training_profile_identity,
                "transformer_identity": transformer_identity,
                "training_execution_contract": training_execution_contract,
                "track31_artifacts": track31_artifacts,
                "runtime_source_identity": _RUNTIME_SOURCE_IDENTITY,
                "checkpoint_invocation_identity": (_CHECKPOINT_INVOCATION_IDENTITY),
                "video_inventory_sha256": "1" * 64,
                "tactile_inventory_sha256": "2" * 64,
                **{
                    field: _TRACK31_TACTILE_CONTRACT[field]
                    for field in (
                        "patch_size",
                        "snr_shift",
                        "use_local_tactile",
                        "max_tactile_streams",
                        "active_tactile_sensor_count",
                        "active_tactile_sensor_ids",
                        "tactile_sensor_id_map",
                        "tactile_in_channels",
                        "tactile_num_tokens",
                        "tactile_encoder_dim",
                        "tactile_latent_channels",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_lambda(
            step,
            warmup_steps=20,
            total_steps=2000,
            min_ratio=0.1,
        ),
    )
    for _ in range(7):
        optimizer.step()
        scheduler.step()
    save_scheduler_state(
        root,
        scheduler.state_dict(),
        completed_steps=7,
        learning_rate=1e-4,
        execution_contract=training_execution_contract,
    )
    for rank in range(world_size):
        save_rng_state(
            root,
            rank=rank,
            world_size=world_size,
            state=capture_rng_state(),
        )
    return {
        "optimizer_inventory_sha256": inventory.inventory_sha256,
        "runtime_signature": runtime_signature,
        "training_execution_contract": training_execution_contract,
        "transformer_identity": transformer_identity,
        "training_profile_identity": training_profile_identity,
        "video_inventory_sha256": "1" * 64,
        "tactile_inventory_sha256": "2" * 64,
        "runtime_source_identity": _RUNTIME_SOURCE_IDENTITY,
        "checkpoint_invocation_identity": _CHECKPOINT_INVOCATION_IDENTITY,
    }


def _write_completion_marker(
    root: Path,
    strict_fields: dict[str, object],
    *,
    world_size: int,
    step: int,
) -> None:
    completion = {
        "schema_version": STRICT_CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "world_size": world_size,
        "action_schema": "qpos8_next_step",
        "optimizer_state_format": OPTIMIZER_STATE_FORMAT,
        **strict_fields,
        "status": "complete",
    }
    completion["sidecar_inventory"] = build_sidecar_inventory(
        root,
        expected_sidecar_paths(
            world_size,
            include_transformer_config=True,
            include_action_migration=True,
        ),
    )
    (root / "checkpoint_complete.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )


def _write_training_state(
    root: Path,
    strict_fields: dict[str, object],
    *,
    world_size: int,
    step: int = 7,
    gradient_accumulation_steps: int = 4,
) -> None:
    (root / "training_state.json").write_text(
        json.dumps(
            {
                "schema_version": STRICT_CHECKPOINT_SCHEMA_VERSION,
                "step": step,
                "data_batches_consumed": step * gradient_accumulation_steps,
                "world_size": world_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "action_schema": "qpos8_next_step",
                "optimizer_state_format": OPTIMIZER_STATE_FORMAT,
                **strict_fields,
            }
        ),
        encoding="utf-8",
    )
    _write_completion_marker(
        root,
        strict_fields,
        world_size=world_size,
        step=step,
    )


def test_preflight_accepts_released_ee20_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "release"
    transformer_sha256 = _write_transformer_checkpoint(
        checkpoint, action_dim=20, action_schema="ee20_pi05"
    )
    monkeypatch.delenv("N0_TRACK31_RESUME_FROM", raising=False)
    monkeypatch.setenv("N0_RELEASED_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("N0_RELEASED_TRANSFORMER_SHA256", transformer_sha256)

    report = _audit_checkpoint()

    assert report["mode"] == "migrate_action"
    assert report["source_action_dim"] == 20
    assert report["target_action_dim"] == 8
    assert report["transformer_identity"]["sha256"] == transformer_sha256
    assert report["tactile_training_contract"]["max_tactile_streams"] == 4
    assert report["tactile_training_contract"]["use_local_tactile"] is False


def test_formal_preflight_rejects_missing_runtime_provenance_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "release"
    transformer_sha256 = _write_transformer_checkpoint(
        checkpoint,
        action_dim=20,
        action_schema="ee20_pi05",
    )
    monkeypatch.setenv("N0_RELEASED_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("N0_RELEASED_TRANSFORMER_SHA256", transformer_sha256)
    monkeypatch.delenv("N0_TRACK31_OVERLAY_MANIFEST_SHA256")

    with pytest.raises(
        ValueError,
        match="N0_TRACK31_OVERLAY_MANIFEST_SHA256",
    ):
        _audit_checkpoint()


def test_preflight_rejects_released_tactile_capacity_narrowing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "release"
    transformer_sha256 = _write_transformer_checkpoint(
        checkpoint, action_dim=20, action_schema="ee20_pi05"
    )
    config_path = checkpoint / "transformer" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_tactile_streams"] = 2
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.delenv("N0_TRACK31_RESUME_FROM", raising=False)
    monkeypatch.setenv("N0_RELEASED_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("N0_RELEASED_TRANSFORMER_SHA256", transformer_sha256)

    with pytest.raises(ValueError, match="max_tactile_streams"):
        _audit_checkpoint()


def test_preflight_rejects_sensor_embedding_shape_below_config_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "release"
    transformer_sha256 = _write_transformer_checkpoint(
        checkpoint,
        action_dim=20,
        action_schema="ee20_pi05",
        sensor_capacity=2,
    )
    monkeypatch.delenv("N0_TRACK31_RESUME_FROM", raising=False)
    monkeypatch.setenv("N0_RELEASED_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("N0_RELEASED_TRANSFORMER_SHA256", transformer_sha256)

    with pytest.raises(ValueError, match="embedding capacity mismatch"):
        _audit_checkpoint()


def test_preflight_accepts_complete_qpos8_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=2)
    _write_training_state(checkpoint, strict_fields, world_size=2)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "2")

    report = _audit_checkpoint()

    assert report["mode"] == "strict_resume"
    assert report["step"] == 7
    assert report["world_size"] == 2
    assert report["optimizer_state_format"] == OPTIMIZER_STATE_FORMAT
    assert (
        report["optimizer_inventory_sha256"]
        == strict_fields["optimizer_inventory_sha256"]
    )
    assert report["tactile_training_contract"]["max_tactile_streams"] == 4
    assert report["tactile_training_contract"]["active_tactile_sensor_count"] == 2


def test_preflight_rejects_resume_tactile_scheduler_metadata_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    meta_path = checkpoint / "train_meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["snr_shift"] = 7.0
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_training_state(checkpoint, strict_fields, world_size=1)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")

    with pytest.raises(ValueError, match="snr_shift"):
        _audit_checkpoint()


def test_preflight_resume_allows_new_invocation_stop_without_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    _write_training_state(checkpoint, strict_fields, world_size=1, step=7)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")
    monkeypatch.setenv("N0_TRACK31_STOP_AFTER_STEP", "9")

    checkpoint_report = _audit_checkpoint()
    invocation_contract = _build_invocation_contract(checkpoint_report)

    assert invocation_contract == {
        "start_step": 7,
        "stop_after_step": 9,
        "num_steps": 2000,
        "optimizer_steps_this_invocation": 2,
        "is_final_invocation": False,
    }
    assert "stop_after_step" not in checkpoint_report["training_execution_contract"]


def test_preflight_rejects_resume_stop_at_completed_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    _write_training_state(checkpoint, strict_fields, world_size=1, step=7)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")
    monkeypatch.setenv("N0_TRACK31_STOP_AFTER_STEP", "7")

    checkpoint_report = _audit_checkpoint()

    with pytest.raises(ValueError, match="start_step < stop_after_step <= num_steps"):
        _build_invocation_contract(checkpoint_report)


def test_preflight_rejects_unloadable_scheduler_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    _write_training_state(checkpoint, strict_fields, world_size=1)
    (checkpoint / "scheduler_state.json").write_text("{}", encoding="utf-8")
    _write_completion_marker(checkpoint, strict_fields, world_size=1, step=7)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")

    with pytest.raises(ValueError, match="scheduler state"):
        _audit_checkpoint()


def test_preflight_rejects_malformed_rng_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    _write_training_state(checkpoint, strict_fields, world_size=1)
    (checkpoint / "rng_state_rank0.json").write_text(
        json.dumps({"rank": 0}), encoding="utf-8"
    )
    _write_completion_marker(checkpoint, strict_fields, world_size=1, step=7)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")

    with pytest.raises(ValueError, match="RNG metadata"):
        _audit_checkpoint()


def test_preflight_rejects_current_artifact_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    _write_training_state(checkpoint, strict_fields, world_size=1)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")
    current_artifacts = dict(_TRACK31_ARTIFACT_IDENTITY)
    current_artifacts["manifest_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="artifact identity mismatch"):
        _audit_checkpoint(expected_track31_artifacts=current_artifacts)


@pytest.mark.parametrize(
    ("write_optimizer_metadata", "write_optimizer_shard", "error_pattern"),
    (
        (False, True, r"\.metadata"),
        (True, False, r"\.distcp"),
    ),
)
def test_preflight_rejects_incomplete_optimizer_dcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_optimizer_metadata: bool,
    write_optimizer_shard: bool,
    error_pattern: str,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(
        checkpoint,
        world_size=1,
        write_optimizer_metadata=write_optimizer_metadata,
        write_optimizer_shard=write_optimizer_shard,
    )
    _write_training_state(checkpoint, strict_fields, world_size=1)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")

    with pytest.raises(ValueError, match=error_pattern):
        _audit_checkpoint()


def test_preflight_rejects_resume_world_size_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    _write_training_state(checkpoint, strict_fields, world_size=1, step=1)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "2")

    with pytest.raises(ValueError, match="world_size"):
        _audit_checkpoint()


def test_preflight_rejects_inventory_not_bound_to_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    strict_fields["optimizer_inventory_sha256"] = "0" * 64
    _write_training_state(checkpoint, strict_fields, world_size=1)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")

    with pytest.raises(ValueError, match="completion marker"):
        _audit_checkpoint()


def test_preflight_rejects_runtime_signature_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    incompatible_runtime = dict(strict_fields["runtime_signature"])
    incompatible_runtime["torch_version"] = "incompatible"
    strict_fields["runtime_signature"] = incompatible_runtime
    _write_training_state(checkpoint, strict_fields, world_size=1)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")

    with pytest.raises(ValueError, match="runtime signature mismatch"):
        _audit_checkpoint()


def test_preflight_rejects_training_execution_contract_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    incompatible_contract = dict(strict_fields["training_execution_contract"])
    incompatible_contract["max_latent_frames"] = 9
    strict_fields["training_execution_contract"] = incompatible_contract
    meta_path = checkpoint / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["training_execution_contract"] = incompatible_contract
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    _write_training_state(checkpoint, strict_fields, world_size=1)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")
    monkeypatch.setenv("N0_TRACK31_MAX_LATENT_FRAMES", "5")

    with pytest.raises(ValueError, match="training execution contract mismatch"):
        _audit_checkpoint()


def test_preflight_rejects_tampered_resume_transformer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    _write_training_state(checkpoint, strict_fields, world_size=1)
    _write_transformer_checkpoint(
        checkpoint,
        action_dim=8,
        action_schema="qpos8_next_step",
        fill_value=1.0,
    )
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")

    with pytest.raises(ValueError, match="does not match actual bytes"):
        _audit_checkpoint()


def test_preflight_rejects_inconsistent_migration_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    _write_training_state(checkpoint, strict_fields, world_size=1)
    migration_path = checkpoint / "action_migration_report.json"
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    migration["source_checkpoint_sha256"] = "b" * 64
    migration_path.write_text(json.dumps(migration), encoding="utf-8")
    _write_completion_marker(checkpoint, strict_fields, world_size=1, step=7)
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")

    with pytest.raises(ValueError, match="source transformer identity"):
        _audit_checkpoint()
