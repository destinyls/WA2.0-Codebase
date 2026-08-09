"""Schema coverage for the FSDP2 runtime smoke marker."""

import json
from pathlib import Path

from n0_twam.distributed.optimizer_checkpoint import (
    STRICT_CHECKPOINT_SCHEMA_VERSION,
    build_training_execution_contract,
)
from script.track3_1.smoke_fsdp2_runtime import _write_completion_marker


def test_fsdp2_smoke_marker_binds_training_execution_contract(
    tmp_path: Path,
) -> None:
    contract = build_training_execution_contract(
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
    )
    _write_completion_marker(
        tmp_path,
        "a" * 64,
        {"schema_version": 1},
        contract,
    )

    marker = json.loads(
        (tmp_path / "checkpoint_complete.json").read_text(encoding="utf-8")
    )
    assert marker["schema_version"] == STRICT_CHECKPOINT_SCHEMA_VERSION == 6
    assert marker["training_execution_contract"] == contract
