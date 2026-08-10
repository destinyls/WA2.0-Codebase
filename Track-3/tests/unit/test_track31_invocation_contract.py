"""Trainer-loop tests for bounded Track 3.1 invocations."""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from n0_twam.configs.twam_track31_univtac_cfg import (
    build_track31_tactile_training_contract,
    twam_track31_univtac_cfg,
)
from n0_twam.dataset.bucket_sampler import BucketedDistributedBatchSampler


def _load_train_module() -> Any:
    """Import the trainer while stubbing local-test-only dependencies."""
    import n0_twam

    wandb_stub = types.ModuleType("wandb")
    wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    dataset_stub = types.ModuleType("dataset")
    dataset_stub.__spec__ = importlib.machinery.ModuleSpec("dataset", loader=None)
    dataset_stub.MultiLatentLeRobotDataset = object
    dataset_stub.BucketedDistributedBatchSampler = object

    stub_names = ("wandb", "dataset")
    previous_stubs = {name: sys.modules.get(name) for name in stub_names}
    previous_train_module = sys.modules.pop("n0_twam.train", None)
    missing_attribute = object()
    previous_train_attribute = getattr(n0_twam, "train", missing_attribute)
    try:
        sys.modules["wandb"] = wandb_stub
        sys.modules["dataset"] = dataset_stub
        return importlib.import_module("n0_twam.train")
    finally:
        sys.modules.pop("n0_twam.train", None)
        if previous_train_module is not None:
            sys.modules["n0_twam.train"] = previous_train_module
        if previous_train_attribute is missing_attribute:
            if hasattr(n0_twam, "train"):
                delattr(n0_twam, "train")
        else:
            n0_twam.train = previous_train_attribute
        for name, previous_module in previous_stubs.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


@pytest.fixture(scope="module")
def train_module() -> Any:
    return _load_train_module()


class _ProgressBar:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.n = 0

    def set_postfix(self, values: dict[str, object]) -> None:
        del values

    def close(self) -> None:
        return None


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def error(self, message: str) -> None:
        self.messages.append(message)

    def exception(self, message: str, *args: object) -> None:
        self.messages.append(message % args if args else message)


def _make_loop_trainer(
    train_module: Any,
    *,
    start_step: int,
    stop_after_step: int,
    num_steps: int = 10,
    save_interval: int = 2,
    gradient_accumulation_steps: int = 2,
) -> tuple[Any, list[int], list[int]]:
    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.step = start_step
    trainer.config = SimpleNamespace(
        num_steps=num_steps,
        stop_after_step=stop_after_step,
        save_interval=save_interval,
        gc_interval=1000,
        rank=0,
        enable_wandb=False,
    )
    trainer.gradient_accumulation_steps = gradient_accumulation_steps
    trainer.train_loader = [None] * 20
    trainer.val_loader = None
    trainer.transformer = SimpleNamespace(train=lambda: None)
    trainer.optimizer = SimpleNamespace(zero_grad=lambda: None)
    trainer.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [1e-4])
    batch_indices: list[int] = []
    saved_steps: list[int] = []

    trainer._get_next_batch = lambda: None

    def fake_train_step(batch: object, batch_idx: int) -> dict[str, object]:
        del batch
        batch_indices.append(batch_idx)
        value = torch.tensor(1.0)
        return {
            "latent_loss": value,
            "action_loss": value,
            "tactile_loss": value,
            "total_loss": value,
            "total_norm": value,
            "should_log": batch_idx + 1 == gradient_accumulation_steps,
        }

    trainer._train_step = fake_train_step
    trainer.save_checkpoint = lambda: saved_steps.append(trainer.step)
    return trainer, batch_indices, saved_steps


def _patch_loop_runtime(monkeypatch: pytest.MonkeyPatch, train_module: Any) -> _Logger:
    logger = _Logger()
    monkeypatch.setattr(train_module, "tqdm", _ProgressBar)
    monkeypatch.setattr(train_module, "logger", logger)
    monkeypatch.setattr(train_module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(train_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: False)
    return logger


def test_staged_loop_stops_only_after_optimizer_step_and_forces_checkpoint(
    train_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _patch_loop_runtime(monkeypatch, train_module)
    trainer, batch_indices, saved_steps = _make_loop_trainer(
        train_module,
        start_step=0,
        stop_after_step=3,
    )

    trainer.train()

    assert trainer.step == 3
    assert batch_indices == [0, 1, 0, 1, 0, 1]
    assert saved_steps == [2, 3]
    assert any(
        "Training invocation completed" in message for message in logger.messages
    )
    assert not any("Final training completed" in message for message in logger.messages)


def test_continuation_stage_uses_absolute_step_and_deduplicates_periodic_save(
    train_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _patch_loop_runtime(monkeypatch, train_module)
    trainer, batch_indices, saved_steps = _make_loop_trainer(
        train_module,
        start_step=2,
        stop_after_step=4,
        num_steps=4,
        save_interval=2,
    )

    trainer.train()

    assert trainer.step == 4
    assert batch_indices == [0, 1, 0, 1]
    assert saved_steps == [4]
    assert any("Final training completed" in message for message in logger.messages)


def test_checkpoint_persists_completed_epoch_before_stop_iteration_and_resume(
    train_module: Any,
    tmp_path: Path,
) -> None:
    signatures = [(2, 2)] * 719
    tasks = [f"task-{index % 8}" for index in range(719)]
    sample_ids = [f"episode-{index}" for index in range(719)]
    sampler_kwargs = {
        "signatures": signatures,
        "batch_size": 2,
        "num_replicas": 48,
        "rank": 0,
        "shuffle": True,
        "seed": 20260801,
        "drop_last": True,
        "coverage_mode": "pad_global",
        "tasks": tasks,
        "sample_ids": sample_ids,
    }
    sampler = BucketedDistributedBatchSampler(**sampler_kwargs)
    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.config = SimpleNamespace(rank=0, sampler_coverage_mode="pad_global")
    trainer.train_sampler = sampler
    trainer.train_loader = [None] * 8
    trainer.save_dir = tmp_path / "checkpoints"
    trainer.save_dir.mkdir()

    for epoch in range(9):
        sampler.set_epoch(epoch)
        trainer._write_completed_epoch_exposure()
    sampler.set_epoch(9)
    trainer.data_batches_consumed = 80

    trainer._persist_completed_epoch_exposure_at_checkpoint()
    trainer._persist_completed_epoch_exposure_at_checkpoint()

    saved_state = json.loads(json.dumps(sampler.state_dict()))
    assert saved_state["accounted_epochs"] == list(range(10))
    assert sum(saved_state["cumulative_exposure_counts"].values()) == 10 * 768
    report_lines = (
        (tmp_path / "exposure_report.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert [json.loads(line)["epoch"] for line in report_lines] == list(range(10))

    resumed = BucketedDistributedBatchSampler(**sampler_kwargs)
    resumed.load_state_dict(saved_state)
    resumed.set_epoch(10)
    first_summary = resumed.exposure_summary()
    second_summary = resumed.exposure_summary()

    assert 9 in resumed.state_dict()["accounted_epochs"]
    assert first_summary == second_summary
    assert sum(first_summary["cumulative_exposure_counts"].values()) == 11 * 768


def test_checkpoint_does_not_account_a_partially_consumed_epoch(
    train_module: Any,
    tmp_path: Path,
) -> None:
    sampler = BucketedDistributedBatchSampler(
        [(2, 2)] * 719,
        batch_size=2,
        num_replicas=48,
        rank=0,
        shuffle=True,
        seed=20260801,
        drop_last=True,
        coverage_mode="pad_global",
    )
    sampler.set_epoch(9)
    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.config = SimpleNamespace(rank=0, sampler_coverage_mode="pad_global")
    trainer.train_sampler = sampler
    trainer.train_loader = [None] * 8
    trainer.save_dir = tmp_path / "checkpoints"
    trainer.save_dir.mkdir()
    trainer.data_batches_consumed = 79

    trainer._persist_completed_epoch_exposure_at_checkpoint()

    assert sampler.state_dict()["accounted_epochs"] == []
    assert not (tmp_path / "exposure_report.jsonl").exists()


def test_save_checkpoint_settles_exposure_before_building_payload(
    train_module: Any,
    tmp_path: Path,
) -> None:
    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.step = 20
    trainer.save_dir = tmp_path
    calls: list[str] = []

    def fail_after_recording_call() -> None:
        calls.append("persist")
        raise ValueError("exposure sentinel")

    def propagate(error: Exception | None, stage: str) -> None:
        assert stage == "exposure stage"
        assert isinstance(error, ValueError)
        raise RuntimeError("collective exposure failure") from error

    trainer._persist_completed_epoch_exposure_at_checkpoint = fail_after_recording_call
    trainer._raise_if_checkpoint_stage_failed = propagate

    with pytest.raises(RuntimeError, match="collective exposure failure"):
        trainer.save_checkpoint()

    assert calls == ["persist"]
    assert not (tmp_path / ".checkpoint_step_20.incomplete").exists()


def test_training_crash_log_is_written_under_run_root(
    train_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_loop_runtime(monkeypatch, train_module)
    trainer, _, _ = _make_loop_trainer(
        train_module,
        start_step=0,
        stop_after_step=1,
    )
    trainer.config.save_root = str(tmp_path)

    def fail_train_step(batch: object, batch_idx: int) -> dict[str, object]:
        del batch, batch_idx
        raise RuntimeError("track31 crash sentinel")

    trainer._train_step = fail_train_step

    with pytest.raises(RuntimeError, match="track31 crash sentinel"):
        trainer.train()

    crash_log = tmp_path / "logs" / "crash_rank0.log"
    assert crash_log.is_file()
    assert "track31 crash sentinel" in crash_log.read_text(encoding="utf-8")


def test_crash_log_failure_does_not_mask_training_error(
    train_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger = _patch_loop_runtime(monkeypatch, train_module)
    trainer, _, _ = _make_loop_trainer(
        train_module,
        start_step=0,
        stop_after_step=1,
    )
    non_directory = tmp_path / "not-a-directory"
    non_directory.write_text("occupied", encoding="utf-8")
    trainer.config.save_root = str(non_directory)

    def fail_train_step(batch: object, batch_idx: int) -> dict[str, object]:
        del batch, batch_idx
        raise RuntimeError("primary training failure")

    trainer._train_step = fail_train_step

    with pytest.raises(RuntimeError, match="primary training failure"):
        trainer.train()

    assert any(
        "Unable to persist crash diagnostics" in message for message in logger.messages
    )


def test_direct_trainer_launch_rejects_non_advancing_stage(
    train_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loop_runtime(monkeypatch, train_module)
    trainer, _, saved_steps = _make_loop_trainer(
        train_module,
        start_step=4,
        stop_after_step=4,
        num_steps=10,
    )

    with pytest.raises(ValueError, match="start_step < stop_after_step <= num_steps"):
        trainer.train()

    assert saved_steps == []


def test_direct_trainer_rejects_distributed_invocation_contract_mismatch(
    train_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _, _ = _make_loop_trainer(
        train_module,
        start_step=0,
        stop_after_step=3,
        num_steps=10,
    )
    trainer.device = torch.device("cpu")
    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_module.dist, "get_world_size", lambda: 2)

    def fake_all_gather(gathered: list[torch.Tensor], local: torch.Tensor) -> None:
        gathered[0].copy_(local)
        gathered[1].copy_(torch.tensor([1, 1, 1, 0, 4, 10], dtype=torch.int64))

    monkeypatch.setattr(train_module.dist, "all_gather", fake_all_gather)

    with pytest.raises(ValueError, match="distributed invocation contract mismatch"):
        trainer._validate_invocation_contract()


def test_distributed_invalid_boundary_joins_collective_before_failing(
    train_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _, _ = _make_loop_trainer(
        train_module,
        start_step=0,
        stop_after_step=11,
        num_steps=10,
    )
    trainer.device = torch.device("cpu")
    collective_called = False
    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_module.dist, "get_world_size", lambda: 2)

    def fake_all_gather(gathered: list[torch.Tensor], local: torch.Tensor) -> None:
        nonlocal collective_called
        collective_called = True
        gathered[0].copy_(local)
        gathered[1].copy_(torch.tensor([1, 1, 1, 0, 3, 10], dtype=torch.int64))

    monkeypatch.setattr(train_module.dist, "all_gather", fake_all_gather)

    with pytest.raises(
        ValueError, match="distributed invocation contract boundary validation failed"
    ):
        trainer._validate_invocation_contract()

    assert collective_called is True


def test_distributed_int64_overflow_joins_collective_before_failing(
    train_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _, _ = _make_loop_trainer(
        train_module,
        start_step=0,
        stop_after_step=1 << 100,
        num_steps=10,
    )
    trainer.device = torch.device("cpu")
    collective_called = False
    monkeypatch.setattr(train_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_module.dist, "get_world_size", lambda: 2)

    def fake_all_gather(gathered: list[torch.Tensor], local: torch.Tensor) -> None:
        nonlocal collective_called
        collective_called = True
        assert local.tolist() == [1, 0, 1, 0, 0, 10]
        gathered[0].copy_(local)
        gathered[1].copy_(torch.tensor([1, 1, 1, 0, 3, 10]))

    monkeypatch.setattr(train_module.dist, "all_gather", fake_all_gather)

    with pytest.raises(ValueError, match="non-integer or int64-overflow step boundary"):
        trainer._validate_invocation_contract()

    assert collective_called is True


def _track31_transformer_stub() -> SimpleNamespace:
    contract = build_track31_tactile_training_contract(twam_track31_univtac_cfg)
    model_config = {
        "action_schema": "qpos8_next_step",
        "action_dim": 8,
        "patch_size": list(contract["patch_size"]),
        "use_local_tactile": contract["use_local_tactile"],
        "max_tactile_streams": contract["max_tactile_streams"],
        "tactile_in_channels": contract["tactile_in_channels"],
        "tactile_num_tokens": contract["tactile_num_tokens"],
        "tactile_encoder_dim": contract["tactile_encoder_dim"],
        "_name_or_path": "/released/model",
    }
    return SimpleNamespace(
        config=model_config,
        patch_size=tuple(contract["patch_size"]),
        use_local_tactile=contract["use_local_tactile"],
        max_tactile_streams=contract["max_tactile_streams"],
        tactile_latent_channels=contract["tactile_latent_channels"],
        sensor_id_embed=SimpleNamespace(num_embeddings=contract["max_tactile_streams"]),
        local_tactile_sensor_embed=SimpleNamespace(
            num_embeddings=contract["max_tactile_streams"]
        ),
    )


def test_mot_init_overrides_include_complete_tactile_shape_contract(
    train_module: Any,
) -> None:
    loader_kwargs = {
        "patch_size": (1, 2, 2),
        "use_local_tactile": True,
        "max_tactile_streams": 4,
        "tactile_in_channels": 3,
        "tactile_num_tokens": 4,
        "tactile_encoder_dim": 256,
        "use_contact_gate": False,
        "contact_gate_layers": 2,
        "contact_gate_heads": 8,
        "contact_gate_stop_grad": True,
        "torch_dtype": torch.bfloat16,
    }

    overrides = train_module._build_mot_config_overrides(loader_kwargs)

    assert overrides == {
        key: value for key, value in loader_kwargs.items() if key not in {"torch_dtype"}
    }
    assert overrides["max_tactile_streams"] == 4


def test_checkpoint_config_and_metadata_match_runtime_tactile_contract(
    train_module: Any,
) -> None:
    contract = build_track31_tactile_training_contract(twam_track31_univtac_cfg)
    transformer = _track31_transformer_stub()

    checkpoint_config = train_module._build_track31_transformer_checkpoint_config(
        transformer,
        contract,
        action_schema="qpos8_next_step",
        action_dim=8,
    )
    metadata = train_module._build_track31_tactile_checkpoint_metadata(contract)

    assert "_name_or_path" not in checkpoint_config
    assert checkpoint_config["action_schema"] == "qpos8_next_step"
    assert checkpoint_config["snr_shift"] == contract["snr_shift"]
    assert checkpoint_config["max_tactile_streams"] == 4
    assert checkpoint_config["tactile_latent_channels"] == 48
    assert metadata["snr_shift"] == contract["snr_shift"]
    assert metadata["max_tactile_streams"] == 4
    assert metadata["active_tactile_sensor_count"] == 2
    assert metadata["active_tactile_sensor_ids"] == [0, 1]


def test_checkpoint_save_rejects_runtime_capacity_drift(
    train_module: Any,
) -> None:
    contract = build_track31_tactile_training_contract(twam_track31_univtac_cfg)
    transformer = _track31_transformer_stub()
    transformer.config["max_tactile_streams"] = 2

    with pytest.raises(ValueError, match="max_tactile_streams"):
        train_module._build_track31_transformer_checkpoint_config(
            transformer,
            contract,
            action_schema="qpos8_next_step",
            action_dim=8,
        )


def test_checkpoint_save_rejects_sensor_embedding_capacity_drift(
    train_module: Any,
) -> None:
    contract = build_track31_tactile_training_contract(twam_track31_univtac_cfg)
    transformer = _track31_transformer_stub()
    transformer.sensor_id_embed.num_embeddings = 2

    with pytest.raises(ValueError, match="sensor_id_embed capacity"):
        train_module._build_track31_transformer_checkpoint_config(
            transformer,
            contract,
            action_schema="qpos8_next_step",
            action_dim=8,
        )


@pytest.mark.parametrize("runtime_schema", [None, "ee20_pi05"])
def test_checkpoint_save_rejects_action_schema_drift(
    train_module: Any,
    runtime_schema: str | None,
) -> None:
    contract = build_track31_tactile_training_contract(twam_track31_univtac_cfg)
    transformer = _track31_transformer_stub()
    if runtime_schema is None:
        transformer.config.pop("action_schema")
    else:
        transformer.config["action_schema"] = runtime_schema

    with pytest.raises(ValueError, match="action_schema mismatch"):
        train_module._build_track31_transformer_checkpoint_config(
            transformer,
            contract,
            action_schema="qpos8_next_step",
            action_dim=8,
        )


@pytest.mark.parametrize("runtime_dim", [None, True, 20])
def test_checkpoint_save_rejects_action_dimension_drift(
    train_module: Any,
    runtime_dim: int | None,
) -> None:
    contract = build_track31_tactile_training_contract(twam_track31_univtac_cfg)
    transformer = _track31_transformer_stub()
    if runtime_dim is None:
        transformer.config.pop("action_dim")
    else:
        transformer.config["action_dim"] = runtime_dim

    with pytest.raises(ValueError, match="action_dim mismatch"):
        train_module._build_track31_transformer_checkpoint_config(
            transformer,
            contract,
            action_schema="qpos8_next_step",
            action_dim=8,
        )


def _make_direct_resume_tactile_trainer(
    train_module: Any,
    checkpoint_dir: Path,
) -> tuple[Any, dict[str, object], dict[str, object]]:
    contract = build_track31_tactile_training_contract(twam_track31_univtac_cfg)
    transformer = _track31_transformer_stub()
    transformer_config = train_module._build_track31_transformer_checkpoint_config(
        transformer,
        contract,
        action_schema="qpos8_next_step",
        action_dim=8,
    )
    transformer_dir = checkpoint_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(transformer_config), encoding="utf-8"
    )
    metadata = train_module._build_track31_tactile_checkpoint_metadata(contract)
    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.track31_tactile_contract = contract
    trainer.transformer = transformer
    trainer.patch_size = tuple(contract["patch_size"])
    trainer.train_scheduler_latent = SimpleNamespace(shift=contract["snr_shift"])
    trainer.train_scheduler_tactile = SimpleNamespace(shift=contract["snr_shift"])
    return trainer, contract, metadata


def test_direct_strict_resume_accepts_exact_tactile_semantics(
    train_module: Any,
    tmp_path: Path,
) -> None:
    trainer, _, metadata = _make_direct_resume_tactile_trainer(
        train_module,
        tmp_path,
    )

    trainer._validate_strict_resume_tactile_contract(
        tmp_path,
        train_meta=metadata,
    )


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        ("snr_shift", 7.0),
        ("patch_size", [1, 1, 2]),
        ("max_tactile_streams", 3),
        ("active_tactile_sensor_count", 1),
        ("active_tactile_sensor_ids", [0, 3]),
        (
            "tactile_sensor_id_map",
            {
                "observation.images.tactile_a": 1,
                "observation.images.tactile_b": 0,
            },
        ),
        ("tactile_in_channels", 1),
        ("tactile_num_tokens", 8),
        ("tactile_encoder_dim", 128),
    ),
)
def test_direct_strict_resume_rejects_tactile_metadata_drift(
    train_module: Any,
    tmp_path: Path,
    field: str,
    tampered_value: object,
) -> None:
    trainer, _, metadata = _make_direct_resume_tactile_trainer(
        train_module,
        tmp_path,
    )
    metadata[field] = tampered_value

    with pytest.raises(ValueError, match=f"metadata mismatch for {field}"):
        trainer._validate_strict_resume_tactile_contract(
            tmp_path,
            train_meta=metadata,
        )


def test_direct_strict_resume_rejects_saved_transformer_snr_drift(
    train_module: Any,
    tmp_path: Path,
) -> None:
    trainer, _, metadata = _make_direct_resume_tactile_trainer(
        train_module,
        tmp_path,
    )
    config_path = tmp_path / "transformer" / "config.json"
    transformer_config = json.loads(config_path.read_text(encoding="utf-8"))
    transformer_config["snr_shift"] = 7.0
    config_path.write_text(json.dumps(transformer_config), encoding="utf-8")

    with pytest.raises(ValueError, match="config snr_shift"):
        trainer._validate_strict_resume_tactile_contract(
            tmp_path,
            train_meta=metadata,
        )


def test_direct_strict_resume_rejects_runtime_scheduler_drift(
    train_module: Any,
    tmp_path: Path,
) -> None:
    trainer, _, metadata = _make_direct_resume_tactile_trainer(
        train_module,
        tmp_path,
    )
    trainer.train_scheduler_tactile.shift = 7.0

    with pytest.raises(ValueError, match="train_scheduler_tactile shift"):
        trainer._validate_strict_resume_tactile_contract(
            tmp_path,
            train_meta=metadata,
        )


def test_direct_strict_resume_rejects_runtime_model_capacity_drift(
    train_module: Any,
    tmp_path: Path,
) -> None:
    trainer, _, metadata = _make_direct_resume_tactile_trainer(
        train_module,
        tmp_path,
    )
    trainer.transformer.sensor_id_embed.num_embeddings = 2

    with pytest.raises(ValueError, match="sensor_id_embed capacity"):
        trainer._validate_strict_resume_tactile_contract(
            tmp_path,
            train_meta=metadata,
        )


def test_sidecar_loader_invokes_direct_tactile_guard(
    train_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = {"num_steps": 10}
    identity = {"sha256": "a" * 64}
    state = {
        "schema_version": train_module.STRICT_CHECKPOINT_SCHEMA_VERSION,
        "step": 1,
        "data_batches_consumed": 4,
        "world_size": 1,
        "gradient_accumulation_steps": 4,
        "action_schema": "qpos8_next_step",
        "optimizer_state_format": train_module.OPTIMIZER_STATE_FORMAT,
        "optimizer_inventory_sha256": "b" * 64,
        "runtime_signature": {"runtime": "test"},
        "training_execution_contract": contract,
        "transformer_identity": identity,
    }
    completion = {
        **state,
        "status": "complete",
        "sidecar_inventory": {},
    }
    (tmp_path / "optimizer_dcp").mkdir()
    (tmp_path / "optimizer_dcp" / ".metadata").write_bytes(b"metadata")
    (tmp_path / "training_state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "checkpoint_complete.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    (tmp_path / "train_meta.json").write_text("{}", encoding="utf-8")

    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.config = SimpleNamespace(world_size=1, action_dim=8)
    trainer.gradient_accumulation_steps = 4
    trainer.action_codec = SimpleNamespace(spec=SimpleNamespace(name="qpos8_next_step"))
    trainer.training_execution_contract = contract
    guard_called = False

    def fail_from_guard(
        checkpoint_dir: Path,
        *,
        train_meta: dict[str, object],
        transformer_config: dict[str, object] | None = None,
    ) -> None:
        nonlocal guard_called
        assert checkpoint_dir == tmp_path
        assert train_meta == {}
        assert transformer_config == {}
        guard_called = True
        raise ValueError("direct tactile guard sentinel")

    trainer._validate_strict_resume_tactile_contract = fail_from_guard
    monkeypatch.setattr(
        train_module,
        "capture_strict_checkpoint_snapshot",
        lambda checkpoint_dir: SimpleNamespace(
            completion=completion,
            training_state=state,
            train_meta={},
            transformer_config={},
            world_size=1,
            gradient_accumulation_steps=4,
            step=1,
        ),
    )
    monkeypatch.setattr(
        train_module,
        "validate_transformer_identity_match",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="direct tactile guard sentinel"):
        trainer._load_and_validate_resume_sidecars(
            tmp_path,
            actual_transformer_identity=identity,
        )

    assert guard_called is True
