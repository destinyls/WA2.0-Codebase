# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from n0_twam.checkpointing.runtime_provenance import LOCAL_EXECUTION_TIER
from n0_twam.cli import run_cli
from n0_twam.evaluation.target10_evaluation_template import (
    target10_reference_request_template,
)
from n0_twam.track31.local_provenance import (
    LocalProvenance,
    package_import_root,
    prepare_local_provenance,
    sha256_file,
)
from n0_twam.track31.request import (
    load_track31_train_request,
    track31_train_request_template,
)
from n0_twam.track31.runner import (
    _prepare_isolated_working_directory,
    build_training_command,
    build_training_environment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _request_payload() -> dict[str, object]:
    payload = track31_train_request_template()
    paths = payload["paths"]
    assert isinstance(paths, dict)
    paths["empty_embedding_sha256"] = "a" * 64
    paths["released_transformer_sha256"] = "b" * 64
    return payload


def _write_request(tmp_path: Path) -> Path:
    path = tmp_path / "track31.train.json"
    path.write_text(json.dumps(_request_payload()), encoding="utf-8")
    return path


def _local_provenance(tmp_path: Path) -> LocalProvenance:
    return LocalProvenance(
        invocation_id="public-stage-a",
        code_manifest_path=tmp_path / "code.json",
        environment_manifest_path=tmp_path / "environment.json",
        launch_receipt_path=tmp_path / "launch.json",
        runtime_source_identity={
            "schema_version": 2,
            "execution_tier": LOCAL_EXECUTION_TIER,
            "code_manifest_sha256": "1" * 64,
            "environment_manifest_sha256": "2" * 64,
            "empty_embedding_sha256": "a" * 64,
            "package_version": "0.1.0",
        },
        checkpoint_invocation_identity={
            "schema_version": 2,
            "execution_tier": LOCAL_EXECUTION_TIER,
            "invocation_id": "public-stage-a",
            "launch_receipt_sha256": "3" * 64,
        },
    )


def test_train_request_resolves_relative_paths_beside_json(tmp_path: Path) -> None:
    request = load_track31_train_request(_write_request(tmp_path))

    assert request.paths.artifact_root == tmp_path / "artifacts"
    assert request.paths.output_root == tmp_path / "outputs/track31-stage-a-dev"
    assert request.runtime.devices == (0,)
    assert request.train.run_role == "development"


def test_train_request_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = _request_payload()
    payload["password"] = "must-never-be-accepted"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected=password"):
        load_track31_train_request(path)


def test_train_request_rejects_ambiguous_initialization(tmp_path: Path) -> None:
    payload = _request_payload()
    paths = payload["paths"]
    assert isinstance(paths, dict)
    paths["resume_from"] = "./checkpoint_step_20"
    paths["init_from"] = "./checkpoint_step_25"
    path = tmp_path / "bad-route.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_track31_train_request(path)


def test_train_request_rejects_run_id_that_cannot_fit_invocation(
    tmp_path: Path,
) -> None:
    payload = _request_payload()
    payload["run_id"] = "a" * 52
    path = tmp_path / "long-run-id.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="run_id must match"):
        load_track31_train_request(path)


def test_train_request_rejects_output_overlapping_immutable_input(
    tmp_path: Path,
) -> None:
    payload = _request_payload()
    paths = payload["paths"]
    assert isinstance(paths, dict)
    paths["output_root"] = "./artifacts/runs/unsafe"
    path = tmp_path / "overlapping-output.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="disjoint from paths.artifact_root"):
        load_track31_train_request(path)


def test_public_environment_clears_inherited_scientific_overrides(
    tmp_path: Path,
) -> None:
    request = load_track31_train_request(_write_request(tmp_path))
    environment = build_training_environment(
        request,
        _local_provenance(tmp_path),
        environ={
            "PATH": "/usr/bin",
            "CUDA_VISIBLE_DEVICES": "7",
            "LD_PRELOAD": "/untrusted/inject.so",
            "N0_TRACK31_ARTIFACT_ROOT": "/stale",
            "N0_FSDP_TOPOLOGY": "hsdp",
            "PYTHONHOME": "/untrusted/python",
            "PYTHONPATH": "/untrusted/package",
        },
    )

    assert environment["PATH"] == "/usr/bin"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["HIP_VISIBLE_DEVICES"] == "0"
    assert environment["N0_TRACK31_ARTIFACT_ROOT"] == str(tmp_path / "artifacts")
    assert environment["PYTHONPATH"] == str(package_import_root())
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONHOME" not in environment
    assert "LD_PRELOAD" not in environment
    assert "N0_FSDP_TOPOLOGY" not in environment
    assert environment["N0_TRACK31_EXECUTION_TIER"] == LOCAL_EXECUTION_TIER


def test_public_training_command_uses_packaged_modules(tmp_path: Path) -> None:
    request = load_track31_train_request(_write_request(tmp_path))
    command = build_training_command(request)

    assert "torch.distributed.run" in command
    assert "n0_twam.train" in command
    assert not any("script/" in token for token in command)


def test_public_child_import_is_bound_to_active_package(tmp_path: Path) -> None:
    fake_package = tmp_path / "untrusted" / "n0_twam"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text(
        "raise RuntimeError('shadow package executed')\n", encoding="utf-8"
    )
    request = load_track31_train_request(_write_request(tmp_path))
    provenance = _local_provenance(tmp_path)
    environment = build_training_environment(
        request,
        provenance,
        environ={"PATH": "/usr/bin", "PYTHONPATH": str(fake_package.parent)},
    )
    working_directory = _prepare_isolated_working_directory(provenance)

    process = subprocess.run(
        [sys.executable, "-c", "import n0_twam; print(n0_twam.__file__)"],
        check=True,
        capture_output=True,
        cwd=working_directory,
        env=environment,
        text=True,
    )

    assert Path(process.stdout.strip()).resolve().parent == (
        package_import_root() / "n0_twam"
    )
    assert not any(working_directory.iterdir())
    assert stat.S_IMODE(working_directory.stat().st_mode) == 0o500


def test_cli_dry_run_returns_one_resolved_plan(tmp_path: Path) -> None:
    result = run_cli(
        [
            "track31",
            "train",
            "--config",
            str(_write_request(tmp_path)),
            "--dry-run",
        ]
    )

    assert result["status"] == "dry_run"
    plan = result["plan"]
    assert isinstance(plan, dict)
    assert plan["execution_tier"] == LOCAL_EXECUTION_TIER
    assert plan["formal_track31"] is False
    assert plan["leaderboard_eligible"] is False


def test_public_eval_template_contains_no_internal_site_paths() -> None:
    serialized = json.dumps(target10_reference_request_template(), sort_keys=True)

    assert "/mnt/data/task" not in serialized
    assert "10.232." not in serialized
    assert "49.235." not in serialized
    assert "frozen_target10_v1.json" in serialized


def test_local_provenance_is_read_only_and_secret_free(tmp_path: Path) -> None:
    provenance = prepare_local_provenance(
        output_root=tmp_path,
        run_id="public-stage-a",
        request_sha256="f" * 64,
        launch_plan={
            "empty_embedding_sha256": "a" * 64,
            "command": ["python", "-m", "n0_twam.train"],
        },
    )

    for path in (
        provenance.code_manifest_path,
        provenance.environment_manifest_path,
        provenance.launch_receipt_path,
    ):
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert len(sha256_file(path)) == 64
        contents = path.read_text(encoding="utf-8")
        assert "PASSWORD" not in contents
        assert "TOKEN" not in contents


def test_wheel_declares_console_entry_and_track31_extra() -> None:
    project_config = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'n0-twam = "n0_twam.cli:main"' in project_config
    assert "[project.optional-dependencies]" in project_config
    assert "track31 = [" in project_config
