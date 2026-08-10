"""Exact multi-rank coverage checks used before Stage A launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from n0_twam import dataset as dataset_package
from script.track3_1 import audit_sampler_coverage as coverage_cli
from script.track3_1.audit_sampler_coverage import (
    build_sampler_coverage_core,
    resolve_exact_config_value,
    resolve_exact_world_size,
    seal_report,
)


def test_stage_a_719_episodes_cover_all_samples_across_48_ranks() -> None:
    report = build_sampler_coverage_core(
        [(2, 2)] * 719,
        tasks=[f"task-{index % 8}" for index in range(719)],
        sample_ids=[f"episode-{index}" for index in range(719)],
        batch_size=2,
        world_size=48,
        seed=20260801,
    )

    assert report["unique_sample_count"] == 719
    assert report["total_slot_count"] == 768
    assert report["padding_count"] == 49
    assert report["per_rank_batch_count"] == 8
    assert report["per_rank_slot_count"] == 16
    assert len(str(report["sampler_input_sha256"])) == 64


def test_stage_a_final759_pads_to_768_across_16_ranks() -> None:
    report = build_sampler_coverage_core(
        [(2, 2)] * 759,
        tasks=[f"task-{index % 8}" for index in range(759)],
        sample_ids=[f"episode-{index}" for index in range(759)],
        batch_size=2,
        world_size=16,
        seed=20260801,
    )

    assert report["unique_sample_count"] == 759
    assert report["total_slot_count"] == 768
    assert report["padding_count"] == 9
    assert report["per_rank_batch_count"] == 24
    assert report["per_rank_slot_count"] == 48


def test_coverage_rejects_misaligned_sample_metadata() -> None:
    with pytest.raises(ValueError, match="align"):
        build_sampler_coverage_core(
            [(2, 2)] * 2,
            tasks=["one"],
            sample_ids=["zero", "one"],
            batch_size=1,
            world_size=2,
            seed=0,
        )


def test_sealed_coverage_report_rejects_resealing() -> None:
    report = seal_report({"schema_version": 1, "status": "coverage_ready"})
    assert len(str(report["coverage_report_sha256"])) == 64
    with pytest.raises(ValueError, match="already sealed"):
        seal_report(report)


def test_world_size_must_match_launcher_contract() -> None:
    assert resolve_exact_world_size(48, "48") == 48
    with pytest.raises(ValueError, match="conflicts"):
        resolve_exact_world_size(32, "48")
    with pytest.raises(ValueError, match="required"):
        resolve_exact_world_size(48, None)
    with pytest.raises(ValueError, match="positive"):
        resolve_exact_world_size(0, "0")


def test_diagnostic_values_cannot_drift_from_training_config() -> None:
    assert resolve_exact_config_value(None, 2, label="batch-size") == 2
    assert resolve_exact_config_value(2, 2, label="batch-size") == 2
    with pytest.raises(ValueError, match="conflicts"):
        resolve_exact_config_value(3, 2, label="batch-size")


def _write_self_hashed_json(
    path: Path,
    payload: dict[str, object],
    *,
    digest_field: str,
    ensure_ascii: bool,
) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    path.write_text(
        json.dumps({**payload, digest_field: digest}),
        encoding="utf-8",
    )
    return digest


def test_main_binds_real_view_and_artifact_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_sha256 = "a" * 64
    normalizer_sha256 = "b" * 64
    encoder_sha256 = "c" * 64
    dataset_root = tmp_path / "train759"
    dataset_root.mkdir()
    conversion_path = tmp_path / "conversion_report.json"
    conversion_sha256 = _write_self_hashed_json(
        conversion_path,
        {"schema_version": 2, "conversions": {}},
        digest_field="conversion_report_sha256",
        ensure_ascii=False,
    )
    inventory_digests = {}
    for kind, filename in {
        "video": "latent_video_inventory.json",
        "tactile": "latent_tactile_inventory.json",
    }.items():
        inventory_digests[kind] = _write_self_hashed_json(
            dataset_root / filename,
            {
                "status": "ready",
                "kind": kind,
                "split": "train759",
                "manifest_sha256": manifest_sha256,
                "conversion_report_sha256": conversion_sha256,
                "encoder_source_identity": {"identity_sha256": encoder_sha256},
            },
            digest_field="inventory_sha256",
            ensure_ascii=True,
        )

    config = SimpleNamespace(
        training_profile_id="multitask_pretrain_v1",
        run_role="development",
        train_view_id="stage_a_dev719_v1",
        batch_size=2,
        seed=20260801,
        source_manifest_sha256=manifest_sha256,
        normalizer_sha256=normalizer_sha256,
        conversion_report_path=str(conversion_path),
        dataset_path=str(dataset_root),
    )
    config_module = ModuleType("n0_twam.configs.twam_track31_univtac_cfg")
    config_module.twam_track31_univtac_cfg = config
    monkeypatch.setitem(
        sys.modules,
        "n0_twam.configs.twam_track31_univtac_cfg",
        config_module,
    )

    fake_dataset = SimpleNamespace(
        sample_signatures=[(2, 2)] * 719,
        sample_tasks=[f"task-{index % 8}" for index in range(719)],
        sample_ids=[f"episode-{index}" for index in range(719)],
        dataset_view=SimpleNamespace(view_sha256="d" * 64),
    )
    monkeypatch.setattr(
        dataset_package,
        "MultiLatentLeRobotDataset",
        lambda **_: fake_dataset,
    )
    output = tmp_path / "coverage.json"
    monkeypatch.setattr(
        coverage_cli,
        "_parse_args",
        lambda: argparse.Namespace(
            world_size=48,
            batch_size=2,
            seed=20260801,
            output=output,
        ),
    )
    monkeypatch.setenv("N0_TRACK31_EXPECTED_WORLD_SIZE", "48")

    assert coverage_cli.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["source_manifest_sha256"] == manifest_sha256
    assert report["train_view_sha256"] == "d" * 64
    assert report["conversion_report_sha256"] == conversion_sha256
    assert report["video_inventory_sha256"] == inventory_digests["video"]
    assert report["tactile_inventory_sha256"] == inventory_digests["tactile"]
    assert report["encoder_source_identity_sha256"] == encoder_sha256
    assert report["unique_sample_count"] == 719
    assert len(report["coverage_report_sha256"]) == 64
