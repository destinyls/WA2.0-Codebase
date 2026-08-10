# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Formal latent identity, startup broadcast, and resume regressions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from n0_twam.data.track31_training_identity import build_track31_profile_identity
from n0_twam.integrations.univtac.artifact_contracts import VerifiedTrack31Artifacts
from n0_twam.integrations.univtac.training_startup import (
    verify_track31_training_startup,
)
from script.track3_1.preflight_train import _audit_checkpoint
from tests.unit.test_track31_preflight import (
    _TRACK31_ARTIFACT_IDENTITY,
    _set_current_runtime_provenance_env,
    _write_resume_sidecars,
    _write_training_state,
    _write_transformer_checkpoint,
)


def test_training_identity_imports_in_fresh_interpreter() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from n0_twam.data.track31_training_identity import "
            "build_track31_profile_identity",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _profile_config() -> SimpleNamespace:
    return SimpleNamespace(
        training_profile_id="multitask_pretrain_v1",
        run_role="final_refit",
        train_view_id="stage_a_final759_v1",
        sampler_coverage_mode="pad_global",
        normalizer_source_view_id="stage_a_final759_v1",
        validation_view_id=None,
        source_manifest_sha256="a" * 64,
        normalizer_sha256="b" * 64,
        train_view_sha256="c" * 64,
        validation_view_sha256=None,
        normalizer_source_view_sha256="c" * 64,
        video_inventory_sha256="d" * 64,
        tactile_inventory_sha256="e" * 64,
    )


def _verified_artifacts() -> VerifiedTrack31Artifacts:
    return VerifiedTrack31Artifacts(
        manifest_sha256="a" * 64,
        normalizer_sha256="b" * 64,
        train_episode_count=759,
        validation_episode_count=0,
        normalizer_sample_count=1,
        action_q01=(0.0,) * 8,
        action_q99=(1.0,) * 8,
        conversion_report_sha256="c" * 64,
        train_view_id="stage_a_final759_v1",
        train_view_sha256="d" * 64,
        normalizer_source_view_id="stage_a_final759_v1",
        normalizer_source_view_sha256="d" * 64,
        video_inventory_sha256="e" * 64,
        tactile_inventory_sha256="f" * 64,
        latent_segment_count=1,
        video_latent_artifact_count=2,
        tactile_latent_artifact_count=4,
    )


def _set_resume_env(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
) -> None:
    monkeypatch.setenv("N0_TRACK31_TRAIN_PROFILE", "multitask_pretrain_v1")
    monkeypatch.setenv("N0_TRACK31_RUN_ROLE", "final_refit")
    monkeypatch.setenv("N0_TRACK31_MAX_LATENT_FRAMES", "5")
    monkeypatch.setenv("N0_TRACK31_GRADIENT_ACCUMULATION_STEPS", "4")
    monkeypatch.setenv("N0_TRACK31_RESUME_FROM", str(checkpoint))
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "1")
    monkeypatch.delenv("N0_TRACK31_INIT_FROM", raising=False)
    monkeypatch.delenv("N0_TRACK31_VALIDATION_VIEW_PATH", raising=False)
    _set_current_runtime_provenance_env(monkeypatch)


def test_formal_profile_identity_v2_binds_both_latent_inventories() -> None:
    identity = build_track31_profile_identity(_profile_config())

    assert identity is not None
    assert identity["schema_version"] == 2
    assert identity["video_inventory_sha256"] == "d" * 64
    assert identity["tactile_inventory_sha256"] == "e" * 64


@pytest.mark.parametrize(
    "missing_field",
    ("video_inventory_sha256", "tactile_inventory_sha256"),
)
def test_formal_profile_identity_rejects_missing_latent_inventory(
    missing_field: str,
) -> None:
    config = _profile_config()
    delattr(config, missing_field)

    with pytest.raises(ValueError, match=missing_field):
        build_track31_profile_identity(config)


def test_startup_rank_zero_performs_the_only_storage_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _verified_artifacts()
    calls: list[int] = []
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup._verify_on_rank_zero",
        lambda config: calls.append(config.rank) or artifacts,
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.is_initialized",
        lambda: False,
    )

    actual = verify_track31_training_startup(
        SimpleNamespace(rank=0),
        device=None,
    )

    assert actual == artifacts
    assert calls == [0]


def test_startup_nonzero_rank_only_consumes_broadcast_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _verified_artifacts()
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup._verify_on_rank_zero",
        lambda config: pytest.fail(f"rank {config.rank} touched shared storage"),
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.get_rank", lambda: 1
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.get_world_size", lambda: 2
    )

    def fake_gather(statuses: list[object], local_status: object) -> None:
        statuses[:] = [
            {"status": "ok", "rank": 0},
            local_status,
        ]

    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.all_gather_object",
        fake_gather,
    )

    def fake_broadcast(messages: list[object], **kwargs: object) -> None:
        assert kwargs["src"] == 0
        messages[0] = {"status": "ok", "artifacts": artifacts}

    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.broadcast_object_list",
        fake_broadcast,
    )

    assert (
        verify_track31_training_startup(SimpleNamespace(rank=1), device=None)
        == artifacts
    )


def test_startup_rank_zero_broadcasts_failure_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcasts: list[object] = []
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup._verify_on_rank_zero",
        lambda config: (_ for _ in ()).throw(ValueError(f"rank {config.rank} drift")),
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.get_rank", lambda: 0
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.get_world_size", lambda: 1
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.all_gather_object",
        lambda statuses, local_status: statuses.__setitem__(0, local_status),
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.broadcast_object_list",
        lambda messages, **kwargs: broadcasts.append(messages[0]),
    )

    with pytest.raises(ValueError, match="rank 0 drift"):
        verify_track31_training_startup(SimpleNamespace(rank=0), device=None)
    assert broadcasts and broadcasts[0]["status"] == "error"


def test_startup_two_rank_config_mismatch_fails_consistently_before_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authoritative_rank = {"value": 0}
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.get_rank",
        lambda: authoritative_rank["value"],
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.get_world_size", lambda: 2
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup._verify_on_rank_zero",
        lambda config: pytest.fail("storage audit ran before rank agreement"),
    )
    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.broadcast_object_list",
        lambda messages, **kwargs: pytest.fail("broadcast ran after rank mismatch"),
    )

    def fake_gather(statuses: list[object], local_status: object) -> None:
        statuses[:] = [
            {"status": "ok", "rank": 0},
            {
                "status": "error",
                "rank": 1,
                "error": "configured rank 0 disagrees with process group rank 1",
            },
        ]

    monkeypatch.setattr(
        "n0_twam.integrations.univtac.training_startup.dist.all_gather_object",
        fake_gather,
    )

    messages = []
    for rank in (0, 1):
        authoritative_rank["value"] = rank
        with pytest.raises(ValueError) as error:
            verify_track31_training_startup(SimpleNamespace(rank=0), device=None)
        messages.append(str(error.value))

    assert messages[0] == messages[1]
    assert "rank 1" in messages[0]
    assert "configured rank 0" in messages[0]


def test_preflight_rejects_resume_video_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    strict_fields["video_inventory_sha256"] = "9" * 64
    _write_training_state(checkpoint, strict_fields, world_size=1)
    _set_resume_env(monkeypatch, checkpoint)

    with pytest.raises(ValueError, match="latent inventory|profile identity"):
        _audit_checkpoint(expected_track31_artifacts=_TRACK31_ARTIFACT_IDENTITY)


def test_preflight_rejects_legacy_formal_resume_without_inventory_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "resume"
    _write_transformer_checkpoint(
        checkpoint, action_dim=8, action_schema="qpos8_next_step"
    )
    strict_fields = _write_resume_sidecars(checkpoint, world_size=1)
    for key in ("video_inventory_sha256", "tactile_inventory_sha256"):
        strict_fields.pop(key)
    legacy_identity = dict(strict_fields["training_profile_identity"])
    legacy_identity["schema_version"] = 1
    legacy_identity.pop("video_inventory_sha256")
    legacy_identity.pop("tactile_inventory_sha256")
    strict_fields["training_profile_identity"] = legacy_identity
    meta_path = checkpoint / "train_meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["training_profile_identity"] = legacy_identity
    metadata.pop("video_inventory_sha256")
    metadata.pop("tactile_inventory_sha256")
    metadata["track31_artifacts"].pop("video_inventory_sha256")
    metadata["track31_artifacts"].pop("tactile_inventory_sha256")
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    _write_training_state(checkpoint, strict_fields, world_size=1)
    _set_resume_env(monkeypatch, checkpoint)

    with pytest.raises(ValueError, match="latent inventory|profile identity"):
        _audit_checkpoint()
