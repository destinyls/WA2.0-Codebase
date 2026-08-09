"""Security and exactness tests for schema-6 strict-resume sidecars."""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from n0_twam.checkpointing.strict_resume import (
    build_sidecar_inventory,
    capture_rng_state,
    expected_sidecar_paths,
    load_rng_state,
    load_scheduler_state,
    restore_rng_state,
    restore_scheduler_state,
    save_rng_state,
    save_scheduler_state,
    validate_scheduler_state,
    validate_sidecar_inventory,
)


def _execution_contract() -> dict[str, object]:
    return {
        "lr_schedule": "cosine",
        "warmup_steps": 2,
        "num_steps": 10,
        "lr_min_ratio": 0.1,
    }


def _cosine(step: int) -> float:
    if step < 2:
        return (step + 1) / 2
    progress = min(max((step - 2) / 8, 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _scheduler() -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine)


def test_scheduler_safe_json_roundtrip_and_template_restore(tmp_path: Path) -> None:
    optimizer, scheduler = _scheduler()
    for _ in range(3):
        optimizer.step()
        scheduler.step()

    saved = save_scheduler_state(
        tmp_path,
        scheduler.state_dict(),
        completed_steps=3,
        learning_rate=1e-4,
        execution_contract=_execution_contract(),
    )
    encoded = json.loads((tmp_path / "scheduler_state.json").read_text())
    assert encoded == saved
    assert not any(isinstance(value, float) for value in saved["base_lrs_hex"])

    fresh_optimizer, fresh_scheduler = _scheduler()
    loaded = load_scheduler_state(
        tmp_path,
        completed_steps=3,
        learning_rate=1e-4,
        execution_contract=_execution_contract(),
    )
    restore_scheduler_state(
        fresh_scheduler,
        loaded,
        completed_steps=3,
        learning_rate=1e-4,
        execution_contract=_execution_contract(),
    )

    assert fresh_scheduler.last_epoch == 3
    assert fresh_scheduler._step_count == 4
    assert fresh_scheduler.get_last_lr() == scheduler.get_last_lr()
    assert fresh_optimizer.param_groups[0]["lr"] == scheduler.get_last_lr()[0]


def test_scheduler_semantic_tamper_is_rejected(tmp_path: Path) -> None:
    optimizer, scheduler = _scheduler()
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    payload = save_scheduler_state(
        tmp_path,
        scheduler.state_dict(),
        completed_steps=2,
        learning_rate=1e-4,
        execution_contract=_execution_contract(),
    )
    payload["last_lrs_hex"] = [(5e-5).hex()]

    with pytest.raises(ValueError, match="schedule semantics"):
        validate_scheduler_state(
            payload,
            completed_steps=2,
            learning_rate=1e-4,
            execution_contract=_execution_contract(),
        )


def test_scheduler_restore_rejects_callable_that_only_matches_current_step(
    tmp_path: Path,
) -> None:
    optimizer, scheduler = _scheduler()
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    payload = save_scheduler_state(
        tmp_path,
        scheduler.state_dict(),
        completed_steps=3,
        learning_rate=1e-4,
        execution_contract=_execution_contract(),
    )
    parameter = torch.nn.Parameter(torch.zeros(()))
    fresh_optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    wrong_scheduler = torch.optim.lr_scheduler.LambdaLR(
        fresh_optimizer,
        lambda step: _cosine(step) if step == 3 else 1.0,
    )

    with pytest.raises(ValueError, match=r"saved schedule: f\("):
        restore_scheduler_state(
            wrong_scheduler,
            payload,
            completed_steps=3,
            learning_rate=1e-4,
            execution_contract=_execution_contract(),
        )


def test_rng_json_safetensors_roundtrip_restores_exact_stream(tmp_path: Path) -> None:
    random.seed(20260801)
    np.random.seed(20260801)
    torch.manual_seed(20260801)
    checkpoint_state = capture_rng_state()
    save_rng_state(tmp_path, rank=0, world_size=1, state=checkpoint_state)
    expected = (random.random(), float(np.random.random()), torch.rand(4))

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    loaded = load_rng_state(tmp_path, rank=0, expected_world_size=1)
    restore_rng_state(loaded)
    actual = (random.random(), float(np.random.random()), torch.rand(4))

    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)


def _write_inventory_files(root: Path, world_size: int) -> tuple[str, ...]:
    paths = expected_sidecar_paths(world_size)
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"payload:{relative}".encode())
    return paths


def test_sidecar_inventory_roundtrip_and_byte_tamper(tmp_path: Path) -> None:
    paths = _write_inventory_files(tmp_path, world_size=2)
    inventory = build_sidecar_inventory(tmp_path, paths)
    assert validate_sidecar_inventory(tmp_path, inventory, paths) == inventory

    (tmp_path / "rng_state_rank1.json").write_bytes(b"tampered-rank-one")
    with pytest.raises(ValueError, match="bytes differ"):
        validate_sidecar_inventory(tmp_path, inventory, paths)


def test_sidecar_inventory_rejects_rank_and_path_tamper(tmp_path: Path) -> None:
    paths = _write_inventory_files(tmp_path, world_size=2)
    inventory = build_sidecar_inventory(tmp_path, paths)

    with pytest.raises(ValueError, match="expected paths"):
        validate_sidecar_inventory(tmp_path, inventory, expected_sidecar_paths(1))

    path_tamper = deepcopy(inventory)
    path_tamper["files"][0]["path"] = "rng_state_rank999.json"
    with pytest.raises(ValueError, match="expected paths"):
        validate_sidecar_inventory(tmp_path, path_tamper, paths)


def test_sidecar_inventory_rejects_symlink_substitution(tmp_path: Path) -> None:
    paths = _write_inventory_files(tmp_path, world_size=1)
    inventory = build_sidecar_inventory(tmp_path, paths)
    sidecar = tmp_path / "train_meta.json"
    target = tmp_path / "other.json"
    target.write_bytes(sidecar.read_bytes())
    sidecar.unlink()
    sidecar.symlink_to(target)

    with pytest.raises(ValueError, match="non-symlink"):
        validate_sidecar_inventory(tmp_path, inventory, paths)


def test_sidecar_inventory_rejects_symlinked_parent_directory(
    tmp_path: Path,
) -> None:
    real_transformer = tmp_path / "real_transformer"
    real_transformer.mkdir()
    (real_transformer / "config.json").write_bytes(b"config")
    (tmp_path / "transformer").symlink_to(real_transformer, target_is_directory=True)
    paths = expected_sidecar_paths(1)
    for relative in paths:
        if relative == "transformer/config.json":
            continue
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload")

    with pytest.raises(ValueError, match="regular directory"):
        build_sidecar_inventory(tmp_path, paths)


def test_load_paths_do_not_use_torch_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    optimizer, scheduler = _scheduler()
    optimizer.step()
    scheduler.step()
    save_scheduler_state(
        tmp_path,
        scheduler.state_dict(),
        completed_steps=1,
        learning_rate=1e-4,
        execution_contract=_execution_contract(),
    )
    save_rng_state(tmp_path, rank=0, world_size=1)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pickle loader must not run")
        ),
    )

    load_scheduler_state(
        tmp_path,
        completed_steps=1,
        learning_rate=1e-4,
        execution_contract=_execution_contract(),
    )
    load_rng_state(tmp_path, rank=0, expected_world_size=1)
