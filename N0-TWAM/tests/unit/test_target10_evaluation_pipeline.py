# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from n0_twam.evaluation import target10_evaluation_pipeline as pipeline
from n0_twam.evaluation.target10_evaluation_template import (
    target10_request_template,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module() -> object:
    path = (
        REPO_ROOT
        / "script"
        / "track3_1"
        / "run_target10_tactile_evaluation_v18_step1500.py"
    )
    spec = importlib.util.spec_from_file_location("target10_v18_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Artifact:
    def __init__(self, metadata: dict[str, object]) -> None:
        self.metadata = metadata
        self.file_sha256 = "a" * 64
        self.seal_sha256 = "e" * 64
        self.root = Path("/sealed/prediction_artifact")


def _request(root: Path) -> pipeline.Target10EvaluationRequest:
    inputs = root / "inputs"
    for directory in ("checkpoint", "vae", "raw"):
        (inputs / directory).mkdir(parents=True, exist_ok=True)
    for file_name in (
        "view.json",
        "manifest.json",
        "conversion.json",
        "metric.py",
    ):
        (inputs / file_name).write_text("{}", encoding="utf-8")
    metric_sha = pipeline.sha256_file(inputs / "metric.py")
    return pipeline.Target10EvaluationRequest(
        checkpoint=inputs / "checkpoint",
        checkpoint_sha256="b" * 64,
        vae=inputs / "vae",
        evaluation_view_manifest=inputs / "view.json",
        raw_root=inputs / "raw",
        manifest=inputs / "manifest.json",
        conversion_report=inputs / "conversion.json",
        official_metric_script=inputs / "metric.py",
        official_metric_sha256=metric_sha,
        output_root=root / "output",
        hip_visible_devices="0",
    )


def _view() -> SimpleNamespace:
    return SimpleNamespace(
        view_id="frozen_target10_v1",
        view_sha256="c" * 64,
        role="frozen_evaluation",
        physical_split="frozen40",
        entries=tuple(range(10)),
    )


def _input_identity() -> dict[str, object]:
    return {
        "checkpoint_train_meta_sha256": "f" * 64,
        "checkpoint_transformer_identity": {"sha256": "b" * 64},
        "vae_decoder_identity_sha256": "d" * 64,
    }


def _artifact(view: SimpleNamespace) -> _Artifact:
    return _Artifact(
        {
            "checkpoint_sha256": "b" * 64,
            "evaluation_view_id": view.view_id,
            "evaluation_view_sha256": view.view_sha256,
            "decoder_identity_sha256": "d" * 64,
            "conditioning_protocol_id": "causal_future_only_v1",
            "samples": [{} for _ in view.entries],
            "generation_provenance": {
                "config_name": "track31_univtac",
                "n_steps": 8,
                "seed_contract": {"seed": 20260801},
            },
        }
    )


def test_request_rejects_unknown_fields() -> None:
    payload = target10_request_template()
    payload["unexpected"] = "value"

    with pytest.raises(ValueError, match="unexpected"):
        pipeline.Target10EvaluationRequest.from_json_dict(payload)


def test_target10_pipeline_reuses_verified_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    view = _view()
    artifact = _artifact(view)
    input_identity = _input_identity()

    def normalize(
        value: pipeline.Target10EvaluationRequest,
    ) -> tuple[pipeline.Target10EvaluationRequest, SimpleNamespace, dict[str, object]]:
        return value, view, input_identity

    monkeypatch.setattr(pipeline, "_normalize_and_validate_request", normalize)

    def run_stage_a(_: object, output: Path, __: Path) -> None:
        output.mkdir(parents=True)

    monkeypatch.setattr(pipeline, "_run_stage_a", run_stage_a)
    monkeypatch.setattr(
        pipeline, "verify_tactile_prediction_artifact", lambda _: artifact
    )
    calls = {"metrics": 0}

    def metric_evaluator(**_: object) -> dict[str, object]:
        calls["metrics"] += 1
        report = {
            "prediction_artifact": {"file_sha256": artifact.file_sha256},
            "evaluation_view": {
                "view_id": view.view_id,
                "view_sha256": view.view_sha256,
            },
            "official_script": {"sha256": request.official_metric_sha256},
            "official_metrics": {"average_psnr": 1.25, "average_ssim": 0.5},
        }
        output = request.output_root / "raw_tactile_quality"
        output.mkdir(parents=True)
        (output / "raw_tactile_quality.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    first = pipeline.run_target10_evaluation(request, metric_evaluator=metric_evaluator)
    second = pipeline.run_target10_evaluation(
        request, metric_evaluator=metric_evaluator
    )

    assert calls["metrics"] == 1
    assert first["reused_prediction_artifact"] is False
    assert first["reused_metric_report"] is False
    assert second["reused_prediction_artifact"] is True
    assert second["reused_metric_report"] is True
    assert second["average_psnr"] == 1.25
    assert (
        json.loads((request.output_root / ".evaluation.lock").read_text())["status"]
        == "released"
    )

    report_path = (
        request.output_root / "raw_tactile_quality" / "raw_tactile_quality.json"
    )
    report = json.loads(report_path.read_text())
    report["official_metrics"]["average_psnr"] = 999.0
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="existing raw metric report"):
        pipeline.run_target10_evaluation(request, metric_evaluator=metric_evaluator)


def test_target10_pipeline_rejects_missing_metric_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    view = _view()
    artifact = _artifact(view)
    input_identity = _input_identity()

    def normalize(
        value: pipeline.Target10EvaluationRequest,
    ) -> tuple[pipeline.Target10EvaluationRequest, SimpleNamespace, dict[str, object]]:
        return value, view, input_identity

    monkeypatch.setattr(pipeline, "_normalize_and_validate_request", normalize)

    def run_stage_a(_: object, output: Path, __: Path) -> None:
        output.mkdir(parents=True)

    monkeypatch.setattr(pipeline, "_run_stage_a", run_stage_a)
    monkeypatch.setattr(
        pipeline, "verify_tactile_prediction_artifact", lambda _: artifact
    )

    calls = {"metrics": 0}

    def metric_evaluator(**_: object) -> dict[str, object]:
        calls["metrics"] += 1
        output = request.output_root / "raw_tactile_quality"
        output.mkdir(parents=True)
        (output / "raw_tactile_quality.json").write_text(
            json.dumps(
                {
                    "prediction_artifact": {"file_sha256": artifact.file_sha256},
                    "evaluation_view": {
                        "view_id": view.view_id,
                        "view_sha256": view.view_sha256,
                    },
                    "official_script": {"sha256": request.official_metric_sha256},
                    "official_metrics": {"average_psnr": 1.25, "average_ssim": 0.5},
                }
            ),
            encoding="utf-8",
        )
        return json.loads((output / "raw_tactile_quality.json").read_text())

    first = pipeline.run_target10_evaluation(request, metric_evaluator=metric_evaluator)
    metric_receipt = request.output_root / "raw_tactile_quality_receipt.json"
    metric_receipt.unlink()
    with pytest.raises(ValueError, match="no sealed receipt"):
        pipeline.run_target10_evaluation(request, metric_evaluator=metric_evaluator)

    assert first["reused_metric_report"] is False
    assert calls["metrics"] == 1


def test_target10_pipeline_rejects_stale_request_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    view = _view()
    artifact = _artifact(view)
    input_identity = _input_identity()

    def normalize(
        value: pipeline.Target10EvaluationRequest,
    ) -> tuple[pipeline.Target10EvaluationRequest, SimpleNamespace, dict[str, object]]:
        return value, view, input_identity

    monkeypatch.setattr(pipeline, "_normalize_and_validate_request", normalize)
    monkeypatch.setattr(
        pipeline, "verify_tactile_prediction_artifact", lambda _: artifact
    )

    def run_stage_a(_: object, output: Path, __: Path) -> None:
        output.mkdir(parents=True)

    monkeypatch.setattr(pipeline, "_run_stage_a", run_stage_a)
    metric_calls: dict[str, int] = {"count": 0}

    def metric_evaluator(**_: object) -> dict[str, object]:
        metric_calls["count"] += 1
        output = request.output_root / "raw_tactile_quality"
        output.mkdir(parents=True)
        report = {
            "prediction_artifact": {"file_sha256": artifact.file_sha256},
            "evaluation_view": {
                "view_id": view.view_id,
                "view_sha256": view.view_sha256,
            },
            "official_script": {"sha256": request.official_metric_sha256},
            "official_metrics": {"average_psnr": 1.25, "average_ssim": 0.5},
        }
        (output / "raw_tactile_quality.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    pipeline.run_target10_evaluation(request, metric_evaluator=metric_evaluator)
    assert metric_calls["count"] == 1
    tampered_identity = dict(input_identity)
    tampered_identity["checkpoint_train_meta_sha256"] = "0" * 64

    def normalize_bad(
        value: pipeline.Target10EvaluationRequest,
    ) -> tuple[pipeline.Target10EvaluationRequest, SimpleNamespace, dict[str, object]]:
        return value, view, tampered_identity

    monkeypatch.setattr(pipeline, "_normalize_and_validate_request", normalize_bad)
    with pytest.raises(ValueError, match="different evaluation request"):
        pipeline.run_target10_evaluation(request, metric_evaluator=lambda **_: {})


def test_target10_pipeline_rejects_stale_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    view = _view()
    input_identity = _input_identity()
    stale = _artifact(view)
    stale.metadata["checkpoint_sha256"] = "d" * 64
    request.output_root.mkdir()
    (request.output_root / "prediction_artifact").mkdir()

    def normalize(
        value: pipeline.Target10EvaluationRequest,
    ) -> tuple[pipeline.Target10EvaluationRequest, SimpleNamespace, dict[str, object]]:
        return value, view, input_identity

    monkeypatch.setattr(pipeline, "_normalize_and_validate_request", normalize)
    monkeypatch.setattr(pipeline, "verify_tactile_prediction_artifact", lambda _: stale)

    with pytest.raises(ValueError, match="existing prediction artifact"):
        pipeline.run_target10_evaluation(request)

    state = json.loads((request.output_root / "evaluation_state.json").read_text())
    assert state["status"] == "failed"


def test_target10_pipeline_reuses_a_released_os_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / ".evaluation.lock"
    stale_pid = 999_999_999
    lock_path.write_text(
        json.dumps({"pid": stale_pid, "hostname": socket.gethostname()}),
        encoding="utf-8",
    )

    acquired = pipeline._acquire_lock(tmp_path)

    assert acquired.path == lock_path
    assert os.getpid() != stale_pid
    acquired.release()
    assert json.loads(lock_path.read_text())["status"] == "released"


def test_target10_cli_exposes_a_single_request_argument() -> None:
    path = REPO_ROOT / "script" / "track3_1" / "run_target10_tactile_evaluation.py"
    spec = importlib.util.spec_from_file_location("target10_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module._build_parser()
    destinations = {action.dest for action in parser._actions}

    assert {"request", "print_request_template"} <= destinations


def test_v18_target10_cli_exposes_reusable_entry_flags() -> None:
    module = _load_script_module()
    parser = module._parser()
    destinations = {action.dest for action in parser._actions}

    expected = {
        "request",
        "print_template",
        "save_template",
    }
    assert destinations == {"help", *expected}


def test_v18_target10_cli_print_template_can_be_saved(tmp_path: Path) -> None:
    module = _load_script_module()
    out = tmp_path / "target10_v18_step1500_request.json"
    status = module.main(["--print-template", "--save-template", str(out)])
    assert status == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 6
    assert payload["calibration_policy"] == "require_published_golden"
    assert payload["checkpoint_sha256"].startswith("REPLACE_WITH_64_LOWERCASE_HEX")
    assert payload["golden_manifest_sha256"].startswith("REPLACE_WITH_64_LOWERCASE_HEX")
    assert payload["normalizer_sha256"].startswith("REPLACE_WITH_64_LOWERCASE_HEX")
    assert payload["empty_embedding_sha256"].startswith("REPLACE_WITH_64_LOWERCASE_HEX")
    assert {
        "base_model",
        "lerobot_root",
        "normalizer",
        "normalizer_source_view",
        "empty_embedding",
    } <= set(payload)
    assert payload["vae"] == f'{payload["base_model"]}/vae'
    assert "official_metric_sha256" not in payload


def test_v18_target10_cli_runs_with_checkpoint_hash_and_approved_golden_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script_module()
    request_path = tmp_path / "request.json"
    payload = {
        "schema_version": 6,
        "checkpoint": str(tmp_path / "checkpoint_step_1500"),
        "checkpoint_sha256": "c" * 64,
        "vae": str(tmp_path / "vae"),
        "base_model": str(tmp_path / "base_model"),
        "empty_embedding": str(tmp_path / "empty_emb.pt"),
        "empty_embedding_sha256": "e" * 64,
        "lerobot_root": str(tmp_path / "lerobot"),
        "normalizer": str(tmp_path / "normalizer.json"),
        "normalizer_sha256": "b" * 64,
        "normalizer_source_view": str(tmp_path / "train_all759_v1.json"),
        "evaluation_view_manifest": str(tmp_path / "frozen_target10_v1.json"),
        "reference_metadata": str(tmp_path / "metadata_val.json"),
        "raw_root": str(tmp_path / "UniVTAC"),
        "manifest": str(tmp_path / "universe_manifest.json"),
        "conversion_report": str(tmp_path / "conversion_report.json"),
        "official_metric_script": str(tmp_path / "stage1_holdout_metrics.py"),
        "golden_root": str(tmp_path / "golden"),
        "golden_manifest": str(tmp_path / "golden_manifest.json"),
        "golden_manifest_sha256": "d" * 64,
        "calibration_policy": "require_published_golden",
        "output_root": str(tmp_path / "output"),
    }
    request_path.write_text(json.dumps(payload), encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run(request: object) -> dict[str, object]:
        captured["request"] = request
        return {"status": "ok"}

    monkeypatch.setattr(module, "run_target10_reference_evaluation", fake_run)
    status = module.main(["--request", str(request_path)])
    assert status == 0
    request = captured["request"]
    assert request.checkpoint_sha256 == "c" * 64
    assert request.golden_manifest_sha256 == "d" * 64
    assert request.output_root == tmp_path / "output"


def test_v18_target10_cli_rejects_unresolved_placeholders(tmp_path: Path) -> None:
    module = _load_script_module()
    template_path = tmp_path / "request.json"
    status = module.main(["--print-template", "--save-template", str(template_path)])
    assert status == 0
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        module.main(["--request", str(template_path)])


def test_v18_target10_shell_wrapper_executes_help() -> None:
    wrapper = (
        REPO_ROOT
        / "script"
        / "track3_1"
        / "run_target10_tactile_evaluation_v18_step1500.sh"
    )
    completed = subprocess.run(
        ["bash", str(wrapper), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "N0_TWAM_PYTHON": sys.executable},
    )
    assert completed.returncode == 0, completed.stderr
    assert "--request" in completed.stdout
