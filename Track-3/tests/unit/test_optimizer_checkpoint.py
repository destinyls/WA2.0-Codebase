import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from n0_twam.distributed.optimizer_checkpoint import (
    _prune_optimizer_state_without_checkpoint_payload,
    _validate_optimizer_param_group_metadata,
    build_training_execution_contract,
    capture_runtime_signature,
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
    validate_optimizer_checkpoint,
    validate_rng_state,
    validate_runtime_signature,
    validate_training_execution_contract,
    write_optimizer_checkpoint_inventory,
)


def test_sparse_optimizer_state_prunes_only_unsaved_parameters() -> None:
    optimizer_state: dict[str, object] = {
        "state": {
            "used.weight": {
                "step": torch.tensor(1),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            },
            "unused.weight": {
                "step": torch.tensor(1),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            },
        },
        "param_groups": [],
    }

    saved = _prune_optimizer_state_without_checkpoint_payload(
        optimizer_state,
        metadata_keys=(
            "optimizer.state.used.weight.step",
            "optimizer.state.used.weight.exp_avg",
            "optimizer.state.used.weight.exp_avg_sq",
            "optimizer.param_groups.0.lr",
        ),
    )

    assert saved == ("used.weight",)
    assert set(optimizer_state["state"]) == {"used.weight", "unused.weight"}
    assert optimizer_state["state"]["unused.weight"] == {}


def test_sparse_optimizer_state_rejects_partial_checkpoint_state() -> None:
    optimizer_state: dict[str, object] = {
        "state": {
            "used.weight": {
                "step": torch.tensor(1),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            }
        },
        "param_groups": [],
    }

    with pytest.raises(ValueError, match="partial state"):
        _prune_optimizer_state_without_checkpoint_payload(
            optimizer_state,
            metadata_keys=("optimizer.state.used.weight.step",),
        )


def test_sparse_optimizer_state_rejects_unknown_checkpoint_parameter() -> None:
    optimizer_state: dict[str, object] = {
        "state": {"used.weight": {"step": torch.tensor(1)}},
        "param_groups": [],
    }

    with pytest.raises(ValueError, match="unknown parameters"):
        _prune_optimizer_state_without_checkpoint_payload(
            optimizer_state,
            metadata_keys=("optimizer.state.removed.weight.step",),
        )


@pytest.mark.parametrize(
    "planner_keys",
    (
        (
            "optimizer.param_groups.0.params",
            "optimizer.param_groups.0.lr",
            "optimizer.param_groups.0.unknown",
        ),
        (
            "optimizer.param_groups.0.params",
            "optimizer.param_groups.0.lr",
            "optimizer.param_groups.1.params",
            "optimizer.param_groups.1.lr",
        ),
    ),
)
def test_optimizer_param_group_metadata_rejects_unknown_structure(
    planner_keys: tuple[str, ...],
) -> None:
    optimizer_state: dict[str, object] = {
        "state": {},
        "param_groups": [{"params": ["used.weight"], "lr": 1e-3}],
    }

    with pytest.raises(ValueError, match="param-group metadata"):
        _validate_optimizer_param_group_metadata(
            optimizer_state,
            planner_keys=planner_keys,
        )


def test_sparse_adam_checkpoint_preserves_lazy_unused_state(tmp_path: Path) -> None:
    class _SparseModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = torch.nn.Linear(3, 2)
            self.unused = torch.nn.Linear(3, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.used(inputs)

    torch.manual_seed(7)
    model = _SparseModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert model.unused.weight not in optimizer.state

    checkpoint_dir = tmp_path / "optimizer_dcp"
    inventory = save_optimizer_checkpoint(model, optimizer, checkpoint_dir)

    resumed_model = _SparseModel()
    resumed_model.load_state_dict(model.state_dict())
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    load_optimizer_checkpoint(
        resumed_model,
        resumed_optimizer,
        checkpoint_dir,
        expected_inventory_sha256=inventory.inventory_sha256,
    )

    assert resumed_model.used.weight in resumed_optimizer.state
    assert resumed_model.unused.weight not in resumed_optimizer.state
    resumed_model.unused.weight.grad = torch.ones_like(resumed_model.unused.weight)
    resumed_optimizer.step()
    assert int(resumed_optimizer.state[resumed_model.unused.weight]["step"]) == 1


def _grouped_execution_contract(
    *,
    max_latent_frames: int = 5,
    gradient_accumulation_steps: int = 4,
    batch_size: int = 1,
    load_worker: int = 0,
    num_steps: int = 2000,
    lr_schedule: str = "cosine",
    token_limit: int = 16384,
    attention_backend: str = "grouped_sdpa",
    mot_cross_attention_backend: str = "sdpa",
    activation_checkpointing: bool = True,
) -> dict[str, object]:
    return build_training_execution_contract(
        max_latent_frames=max_latent_frames,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_size=batch_size,
        load_worker=load_worker,
        num_steps=num_steps,
        lr_schedule=lr_schedule,
        warmup_steps=20,
        lr_min_ratio=0.1,
        activation_checkpointing=activation_checkpointing,
        attention_contract={
            "attention_backend": attention_backend,
            "grouped_sdpa_max_query_tokens": token_limit,
            "mot_cross_attention_backend": mot_cross_attention_backend,
        },
    )


def test_training_execution_contract_accepts_grouped_flash_attention() -> None:
    contract = _grouped_execution_contract(
        attention_backend="grouped_flash_attn",
    )

    assert contract["attention_backend"] == "grouped_flash_attn"
    assert contract["grouped_sdpa_max_query_tokens"] == 16384


def _write_complete_payload(checkpoint_dir: Path) -> str:
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".metadata").write_bytes(b"metadata")
    (checkpoint_dir / "__0_0.distcp").write_bytes(b"shard-zero")
    (checkpoint_dir / "__1_0.distcp").write_bytes(b"shard-one")
    inventory = write_optimizer_checkpoint_inventory(checkpoint_dir)
    return inventory.inventory_sha256


def test_validate_optimizer_checkpoint_accepts_exact_hash_inventory(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "optimizer_dcp"
    inventory_sha256 = _write_complete_payload(checkpoint_dir)

    inventory = validate_optimizer_checkpoint(
        checkpoint_dir,
        expected_inventory_sha256=inventory_sha256,
    )

    assert inventory.inventory_sha256 == inventory_sha256
    assert [entry.path for entry in inventory.files] == [
        ".metadata",
        "__0_0.distcp",
        "__1_0.distcp",
    ]


def test_validate_optimizer_checkpoint_rejects_missing_payload(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "optimizer_dcp"
    _write_complete_payload(checkpoint_dir)
    (checkpoint_dir / "__1_0.distcp").unlink()

    with pytest.raises(ValueError, match="file inventory mismatch"):
        validate_optimizer_checkpoint(checkpoint_dir)


def test_validate_optimizer_checkpoint_rejects_tampered_payload(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "optimizer_dcp"
    _write_complete_payload(checkpoint_dir)
    (checkpoint_dir / "__0_0.distcp").write_bytes(b"shard-Xero")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_optimizer_checkpoint(checkpoint_dir)


def test_validate_optimizer_checkpoint_rejects_uninventoried_shard(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "optimizer_dcp"
    _write_complete_payload(checkpoint_dir)
    (checkpoint_dir / "__2_0.distcp").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="unexpected"):
        validate_optimizer_checkpoint(checkpoint_dir)


def test_validate_optimizer_checkpoint_rejects_marker_digest_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "optimizer_dcp"
    _write_complete_payload(checkpoint_dir)

    with pytest.raises(ValueError, match="completion marker"):
        validate_optimizer_checkpoint(
            checkpoint_dir,
            expected_inventory_sha256="0" * 64,
        )


def test_validate_runtime_signature_accepts_exact_current_runtime() -> None:
    signature = capture_runtime_signature()

    assert validate_runtime_signature(signature) == signature


def test_validate_runtime_signature_rejects_torch_version_change() -> None:
    signature = deepcopy(capture_runtime_signature())
    signature["torch_version"] = "incompatible"

    with pytest.raises(ValueError, match="runtime signature mismatch"):
        validate_runtime_signature(signature)


def test_training_execution_contract_binds_hcu_sequence_and_token_limits() -> None:
    contract = _grouped_execution_contract()

    assert contract == {
        "schema_version": 3,
        "max_latent_frames": 5,
        "gradient_accumulation_steps": 4,
        "batch_size": 1,
        "load_worker": 0,
        "num_steps": 2000,
        "lr_schedule": "cosine",
        "warmup_steps": 20,
        "lr_min_ratio": 0.1,
        "activation_checkpointing": True,
        "attention_backend": "grouped_sdpa",
        "grouped_sdpa_max_query_tokens": 16384,
        "mot_cross_attention_backend": "sdpa",
    }
    assert (
        validate_training_execution_contract(
            contract,
            current_contract=contract,
        )
        == contract
    )


@pytest.mark.parametrize(
    "current_contract",
    (
        _grouped_execution_contract(max_latent_frames=9),
        _grouped_execution_contract(gradient_accumulation_steps=2),
        _grouped_execution_contract(batch_size=2),
        _grouped_execution_contract(load_worker=1),
        _grouped_execution_contract(num_steps=1),
        _grouped_execution_contract(lr_schedule="constant"),
        _grouped_execution_contract(token_limit=32768),
        _grouped_execution_contract(activation_checkpointing=False),
        _grouped_execution_contract(mot_cross_attention_backend="flash_attn"),
        build_training_execution_contract(
            max_latent_frames=5,
            gradient_accumulation_steps=4,
            batch_size=1,
            load_worker=0,
            num_steps=2000,
            lr_schedule="cosine",
            warmup_steps=20,
            lr_min_ratio=0.1,
            activation_checkpointing=True,
            attention_contract={
                "attention_backend": "flex",
                "grouped_sdpa_max_query_tokens": None,
                "mot_cross_attention_backend": "sdpa",
            },
        ),
    ),
)
def test_training_execution_contract_rejects_execution_drift(
    current_contract: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="training execution contract mismatch"):
        validate_training_execution_contract(
            _grouped_execution_contract(),
            current_contract=current_contract,
        )


def test_grouped_execution_contract_requires_positive_token_limit() -> None:
    with pytest.raises(ValueError, match="positive grouped query-token limit"):
        _grouped_execution_contract(token_limit=0)


def test_training_execution_contract_rejects_boolean_schema_version() -> None:
    saved_contract = _grouped_execution_contract()
    saved_contract["schema_version"] = True

    with pytest.raises(ValueError, match="training execution contract mismatch"):
        validate_training_execution_contract(
            saved_contract,
            current_contract=_grouped_execution_contract(),
        )


def _valid_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def test_validate_rng_state_accepts_restorable_states() -> None:
    rng_state = _valid_rng_state()

    assert validate_rng_state(rng_state) is rng_state


def test_validate_rng_state_rejects_malformed_torch_state() -> None:
    rng_state = _valid_rng_state()
    rng_state["torch"] = torch.zeros(4, dtype=torch.float32)

    with pytest.raises((TypeError, RuntimeError)):
        validate_rng_state(rng_state)


def test_validate_rng_state_rejects_cuda_state_count_mismatch() -> None:
    rng_state = _valid_rng_state()
    rng_state["cuda"] = [*rng_state["cuda"], torch.get_rng_state()]

    with pytest.raises(ValueError, match="count mismatch"):
        validate_rng_state(rng_state)
