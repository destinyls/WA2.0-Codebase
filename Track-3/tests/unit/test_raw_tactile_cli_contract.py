# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = REPO_ROOT / "script" / "track3_1" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_a_cli_has_no_raw_hdf5_root_surface() -> None:
    module = _load_script("generate_tactile_prediction_artifact.py")
    parser = module._build_parser()
    destinations = {action.dest for action in parser._actions}

    assert "raw_root" not in destinations
    assert "data_root" not in destinations
    assert "hdf5_root" not in destinations
    assert {
        "ckpt",
        "vae",
        "output",
        "n_steps",
        "evaluation_view_manifest",
        "allow_diagnostic_view",
        "protocol",
    } <= destinations
    assert "n_samples" not in destinations
    diagnostic_action = next(
        action for action in parser._actions if action.dest == "allow_diagnostic_view"
    )
    assert diagnostic_action.default is False
    source = (
        REPO_ROOT / "script/track3_1/generate_tactile_prediction_artifact.py"
    ).read_text(encoding="utf-8")
    assert "import h5py" not in source
    assert "hdf5_reader" not in source
    assert "capture_lerobot_h264" not in source
    assert "lerobot_h264_rgb" not in source

    with pytest.raises(ValueError, match="logical cuda:0"):
        module._validate_device_argument("cuda:1")
    assert module._validate_device_argument("cuda:0") == "cuda:0"

    with pytest.raises(ValueError, match="integer dtype"):
        module._tensor_int_vector(
            torch.arange(17, dtype=torch.float32), label="source_row_ids"
        )


def test_stage_a_defaults_to_target10_and_requires_diagnostic_opt_in() -> None:
    module = _load_script("generate_tactile_prediction_artifact.py")
    target10 = types.SimpleNamespace(
        view_id="frozen_target10_v1",
        role="frozen_evaluation",
        physical_split="frozen40",
        entries=tuple(range(10)),
    )
    other30 = types.SimpleNamespace(
        view_id="frozen_other30_v1",
        role="frozen_evaluation",
        physical_split="frozen40",
        entries=tuple(range(30)),
    )

    module._validate_evaluation_view(target10)
    with pytest.raises(ValueError, match="requires --allow-diagnostic-view"):
        module._validate_evaluation_view(other30)
    module._validate_evaluation_view(other30, allow_diagnostic_view=True)


def test_stage_a_legacy_runtime_loader_preserves_top_level_module_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("generate_tactile_prediction_artifact.py")
    models = types.ModuleType("models")
    models.__path__ = []
    model_utils = types.ModuleType("models.utils")
    train = types.ModuleType("train")
    trainer_factory = object()
    mot_loader = object()
    vae_loader = object()
    train.Trainer = trainer_factory
    model_utils.load_mot_checkpoint = mot_loader
    model_utils.load_vae = vae_loader
    monkeypatch.setitem(sys.modules, "models", models)
    monkeypatch.setitem(sys.modules, "models.utils", model_utils)
    monkeypatch.setitem(sys.modules, "train", train)

    assert module._load_legacy_runtime() == (
        trainer_factory,
        mot_loader,
        vae_loader,
    )


def test_stage_a_source_mapping_accepts_standard_frozen40_conversion(
    tmp_path: Path,
) -> None:
    module = _load_script("generate_tactile_prediction_artifact.py")
    manifest_path = tmp_path / "manifest.json"
    conversion_path = tmp_path / "conversion.json"
    validation_entry = {
        "split": "validation",
        "relative_path": "lift_bottle/clean/95.hdf5",
        "sha256": "a" * 64,
    }
    manifest_path.write_text(
        json.dumps({"entries": [validation_entry]}), encoding="utf-8"
    )
    conversion_path.write_text(
        json.dumps(
            {
                "conversions": {
                    "frozen40": {
                        "source_relative_paths": [validation_entry["relative_path"]],
                        "source_sha256": [validation_entry["sha256"]],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = module._load_validation_source_entries(
        manifest_path=manifest_path,
        conversion_report_path=conversion_path,
    )

    assert result[validation_entry["relative_path"]]["_lerobot_episode_index"] == 0


def test_stage_b_cli_requires_raw_root_and_sealed_artifact() -> None:
    module = _load_script("evaluate_raw_tactile_quality.py")
    parser = module._build_parser()
    destinations = {action.dest for action in parser._actions}

    assert {
        "prediction_artifact",
        "evaluation_view_manifest",
        "raw_root",
        "manifest",
        "conversion_report",
        "official_metric_script",
        "official_metric_sha256",
        "output",
    } <= destinations
    official_action = next(
        action for action in parser._actions if action.dest == "official_metric_script"
    )
    assert official_action.required is True
    hash_action = next(
        action for action in parser._actions if action.dest == "official_metric_sha256"
    )
    assert hash_action.required is True
