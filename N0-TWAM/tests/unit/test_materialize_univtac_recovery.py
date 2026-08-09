# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from script.track3_1 import materialize_transaction, materialize_tree
from script.track3_1 import materialize_univtac as materialize_module
from script.track3_1.materialize_univtac import (
    INCOMPLETE_MARKER_NAME,
    materialize_from_artifacts,
)
from script.track3_1.prepare_univtac import prepare_artifacts
from tests.unit.test_materialize_univtac_cli import (
    _config,
    _fake_materializer,
    _manifest,
    _read_json,
    _staging_roots,
)


def _verified_count(**kwargs: object) -> SimpleNamespace:
    count = 759 if kwargs["physical_split"] == "train759" else 40
    return SimpleNamespace(episode_count=count)


def _patch_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "script.track3_1.materialize_univtac.load_dataset_manifest",
        lambda path, *, verify_sources: _manifest(),
    )
    monkeypatch.setattr(
        "script.track3_1.materialize_univtac._materialize_lerobot_splits",
        _fake_materializer,
    )
    monkeypatch.setattr(
        "script.track3_1.materialize_report.verify_track31_physical_bundle",
        _verified_count,
    )


def _write_crashed_target(
    config: object,
    *,
    status: str,
    publishable_candidate: bool,
    report_already_committed: bool = False,
) -> tuple[Path, dict[str, object]]:
    target_root = Path(config.target_root).resolve(strict=False)
    staging_root = target_root.parent / f".{target_root.name}.incomplete-crash"
    conversions = _fake_materializer(
        manifest=_manifest(),
        materialize_root=target_root,
        repo_id=config.repo_id,
        fps=config.fps,
        image_writer_threads=config.image_writer_threads,
        emit_standard_views=True,
    )
    final_report = materialize_module._conversion_report(
        manifest_sha256="a" * 64,
        conversions=conversions,
        materialize_root=target_root,
    )
    report_path = Path(config.artifact_dir) / "conversion_report.json"
    old_report = b"old-report\n"
    report_path.write_bytes(old_report)
    baseline_sha256 = hashlib.sha256(old_report).hexdigest()
    candidate_path = Path(config.artifact_dir) / (
        f".conversion_report.json.candidate-{staging_root.name}"
    )
    candidate_sha256 = None
    expected_report_sha256 = None
    if publishable_candidate or report_already_committed:
        materialize_transaction._write_json_atomic(candidate_path, final_report)
        candidate_sha256 = materialize_transaction.file_sha256_or_none(candidate_path)
        expected_report_sha256 = str(final_report["conversion_report_sha256"])
    if report_already_committed:
        materialize_transaction.publish_report_cas(
            candidate_path=candidate_path,
            report_path=report_path,
            expected_report_sha256=baseline_sha256,
        )
    materialize_module._write_state_marker(
        target_root,
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256="a" * 64,
        phase=status,
        status=status,
        conversion_report_sha256=expected_report_sha256,
        report_baseline_sha256=baseline_sha256,
        candidate_file_sha256=candidate_sha256,
    )
    return staging_root, final_report


def test_next_invocation_completes_publishable_crash_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _, final_report = _write_crashed_target(
        config,
        status="ready_to_publish",
        publishable_candidate=True,
    )
    _patch_happy_path(monkeypatch)

    result = materialize_from_artifacts(config)

    assert result["status"] == "recovered_materialized"
    assert (
        result["conversion_report_sha256"] == final_report["conversion_report_sha256"]
    )
    assert config.target_root.is_dir()
    marker = _read_json(config.target_root / INCOMPLETE_MARKER_NAME)
    assert marker["status"] == "complete"
    assert marker["phase"] == "complete"


def test_next_invocation_rolls_unpublishable_target_back_then_rebuilds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    crashed_staging, _ = _write_crashed_target(
        config,
        status="incomplete",
        publishable_candidate=False,
    )
    _patch_happy_path(monkeypatch)

    result = materialize_from_artifacts(config)

    assert result["status"] == "materialized"
    assert config.target_root.is_dir()
    assert not crashed_staging.exists()


def test_committed_report_marker_failure_is_degraded_then_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _write_crashed_target(
        config,
        status="ready_to_publish",
        publishable_candidate=True,
        report_already_committed=True,
    )
    _patch_happy_path(monkeypatch)
    real_complete = materialize_module._write_complete_marker
    attempts = 0

    def fail_once(**kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("complete marker storage failure")
        real_complete(**kwargs)

    monkeypatch.setattr(materialize_module, "_write_complete_marker", fail_once)

    with pytest.raises(RuntimeError, match="target retained for a later recovery"):
        materialize_from_artifacts(config)
    assert config.target_root.is_dir()
    assert _read_json(config.target_root / INCOMPLETE_MARKER_NAME)["status"] == (
        "ready_to_publish"
    )

    result = materialize_from_artifacts(config)
    assert result["status"] == "recovered_materialized"
    assert attempts == 2
    marker = _read_json(config.target_root / INCOMPLETE_MARKER_NAME)
    assert marker["status"] == "complete"


def test_verified_report_phase_failure_retains_target_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _write_crashed_target(
        config,
        status="ready_to_publish",
        publishable_candidate=True,
        report_already_committed=True,
    )
    _patch_happy_path(monkeypatch)
    real_write_state = materialize_module._write_state_marker
    phase_attempts = 0

    def fail_report_phase_once(root: Path, **kwargs: object) -> None:
        nonlocal phase_attempts
        if kwargs.get("phase") == "report_committed":
            phase_attempts += 1
            if phase_attempts == 1:
                raise OSError("report phase storage failure")
        real_write_state(root, **kwargs)

    monkeypatch.setattr(
        materialize_module,
        "_write_state_marker",
        fail_report_phase_once,
    )

    with pytest.raises(RuntimeError, match="target retained for recovery"):
        materialize_from_artifacts(config)

    assert config.target_root.is_dir()
    marker = _read_json(config.target_root / INCOMPLETE_MARKER_NAME)
    assert marker["status"] == "ready_to_publish"

    result = materialize_from_artifacts(config)
    assert result["status"] == "recovered_materialized"
    assert _read_json(config.target_root / INCOMPLETE_MARKER_NAME)["status"] == (
        "complete"
    )


def test_markerless_committed_target_is_fully_verified_and_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    verification_calls = 0
    probe_requirements: list[tuple[bool, bool]] = []
    _patch_happy_path(monkeypatch)

    def record_probe_requirements(
        parent: Path,
        *,
        require_exchange: bool = True,
        require_noreplace: bool = True,
    ) -> None:
        assert parent == config.target_root.parent
        probe_requirements.append((require_exchange, require_noreplace))

    monkeypatch.setattr(
        materialize_module,
        "probe_transaction_filesystem",
        record_probe_requirements,
    )

    def count_verification(**kwargs: object) -> SimpleNamespace:
        nonlocal verification_calls
        verification_calls += 1
        return _verified_count(**kwargs)

    monkeypatch.setattr(
        "script.track3_1.materialize_report.verify_track31_physical_bundle",
        count_verification,
    )
    first = materialize_from_artifacts(config)
    assert first["status"] == "materialized"
    assert probe_requirements == [(True, True)]
    marker_path = config.target_root / INCOMPLETE_MARKER_NAME
    marker = _read_json(marker_path)
    assert marker["status"] == "complete"
    marker_path.unlink()
    first_verification_calls = verification_calls

    def forbid_rematerialization(**kwargs: object) -> dict[str, object]:
        del kwargs
        pytest.fail("a fully verified markerless target must not be rebuilt")

    monkeypatch.setattr(
        materialize_module,
        "_materialize_lerobot_splits",
        forbid_rematerialization,
    )
    probe_requirements.clear()
    recovered = materialize_from_artifacts(config)

    assert recovered["status"] == "recovered_materialized"
    assert probe_requirements == [(False, True)]
    assert recovered["conversion_report_sha256"] == first["conversion_report_sha256"]
    assert verification_calls == first_verification_calls + 2
    recovered_marker = _read_json(marker_path)
    assert recovered_marker["status"] == "complete"
    assert recovered_marker["phase"] == "complete"


def test_crash_after_candidate_publish_before_next_marker_is_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    target_root = config.target_root.resolve(strict=False)
    staging_root = target_root.parent / ".lerobot.incomplete-candidate-crash"
    staging_root.mkdir()
    conversions = _fake_materializer(
        manifest=_manifest(),
        materialize_root=staging_root,
        repo_id=config.repo_id,
        fps=config.fps,
        image_writer_threads=config.image_writer_threads,
        emit_standard_views=True,
    )
    final_report = materialize_module._conversion_report(
        manifest_sha256="a" * 64,
        conversions=conversions,
        materialize_root=target_root,
    )
    candidate_path = config.artifact_dir / (
        f"{materialize_module.REPORT_CANDIDATE_PREFIX}{staging_root.name}"
    )
    prepared_path = staging_root / materialize_module.STAGING_CANDIDATE_NAME
    candidate_sha256 = materialize_module.prepare_report_candidate(
        prepared_path=prepared_path,
        candidate_path=candidate_path,
        payload=final_report,
    )
    materialize_module._write_state_marker(
        staging_root,
        target_root=target_root,
        staging_root=staging_root,
        manifest_sha256="a" * 64,
        phase="candidate_prepared",
        conversion_report_sha256=str(final_report["conversion_report_sha256"]),
        report_baseline_sha256=None,
        candidate_file_sha256=candidate_sha256,
    )
    materialize_module.publish_prepared_candidate(
        prepared_path=prepared_path,
        candidate_path=candidate_path,
        expected_sha256=candidate_sha256,
    )
    assert candidate_path.is_file()
    assert not prepared_path.exists()
    _patch_happy_path(monkeypatch)

    result = materialize_from_artifacts(config)

    assert result["status"] == "materialized"
    assert not staging_root.exists()
    assert not candidate_path.exists()


def test_markerless_target_is_not_accepted_when_bundle_verification_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    materialize_from_artifacts(config)
    (config.target_root / INCOMPLETE_MARKER_NAME).unlink()

    def reject_bundle(**kwargs: object) -> SimpleNamespace:
        del kwargs
        raise ValueError("physical bundle drift")

    monkeypatch.setattr(
        "script.track3_1.materialize_report.verify_track31_physical_bundle",
        reject_bundle,
    )

    with pytest.raises(FileExistsError, match="markerless target"):
        materialize_from_artifacts(config)

    assert config.target_root.is_dir()


def test_markerless_target_rejects_noncanonical_report_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    materialize_from_artifacts(config)
    (config.target_root / INCOMPLETE_MARKER_NAME).unlink()
    report_path = config.artifact_dir / "conversion_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="no valid report"):
        materialize_from_artifacts(config)

    assert config.target_root.is_dir()


@pytest.mark.parametrize("marker_kind", ["fifo", "directory"])
def test_existing_nonregular_marker_is_never_treated_as_markerless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_kind: str,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    materialize_from_artifacts(config)
    marker_path = config.target_root / INCOMPLETE_MARKER_NAME
    marker_path.unlink()
    if marker_kind == "fifo":
        os.mkfifo(marker_path)
    else:
        marker_path.mkdir()

    with pytest.raises(FileExistsError, match="target root must not exist"):
        materialize_from_artifacts(config)


def test_markerless_claim_rejects_target_inode_swap_after_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    materialize_from_artifacts(config)
    (config.target_root / INCOMPLETE_MARKER_NAME).unlink()
    displaced_target = tmp_path / "verified-target-displaced"
    calls = 0

    def swap_target(**kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            config.target_root.rename(displaced_target)
            config.target_root.mkdir()
        return _verified_count(**kwargs)

    monkeypatch.setattr(
        "script.track3_1.materialize_report.verify_track31_physical_bundle",
        swap_target,
    )

    with pytest.raises(RuntimeError, match="durable complete claim"):
        materialize_from_artifacts(config)

    assert not (config.target_root / INCOMPLETE_MARKER_NAME).exists()
    assert (displaced_target / "train759").is_dir()


def test_report_committed_recovery_preserves_foreign_displacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    staging_root, _ = _write_crashed_target(
        config,
        status="ready_to_publish",
        publishable_candidate=True,
        report_already_committed=True,
    )
    marker = _read_json(config.target_root / INCOMPLETE_MARKER_NAME)
    materialize_module._write_state_marker(
        config.target_root,
        target_root=config.target_root.resolve(strict=False),
        staging_root=staging_root,
        manifest_sha256="a" * 64,
        phase="report_committed",
        status="ready_to_publish",
        conversion_report_sha256=marker["conversion_report_sha256"],
        report_baseline_sha256=marker["report_baseline_sha256"],
        candidate_file_sha256=marker["candidate_file_sha256"],
    )
    candidate_path = config.artifact_dir / (
        f"{materialize_module.REPORT_CANDIDATE_PREFIX}{staging_root.name}"
    )
    foreign_bytes = b"foreign-displacement\n"
    candidate_path.write_bytes(foreign_bytes)
    _patch_happy_path(monkeypatch)

    result = materialize_from_artifacts(config)

    assert result["status"] == "recovered_materialized"
    assert candidate_path.read_bytes() == foreign_bytes
    assert _read_json(config.target_root / INCOMPLETE_MARKER_NAME)["status"] == (
        "complete"
    )


def test_terminal_claim_no_replace_preserves_foreign_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from script.track3_1 import materialize_state

    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    marker_path = config.target_root / INCOMPLETE_MARKER_NAME
    foreign_bytes = b"foreign-terminal-marker\n"
    real_publish = materialize_state.publish_prepared_file_noreplace

    def inject_foreign_marker(
        prepared_path: Path,
        destination_path: Path,
        *,
        expected_sha256: str,
    ) -> None:
        if destination_path == marker_path:
            destination_path.write_bytes(foreign_bytes)
        real_publish(
            prepared_path,
            destination_path,
            expected_sha256=expected_sha256,
        )

    monkeypatch.setattr(
        materialize_state,
        "publish_prepared_file_noreplace",
        inject_foreign_marker,
    )

    with pytest.raises(RuntimeError, match="foreign (destination|replacement)"):
        materialize_from_artifacts(config)

    assert marker_path.read_bytes() == foreign_bytes
    assert config.target_root.is_dir()


def test_terminal_claim_capture_crash_recovers_original_staging_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from script.track3_1 import materialize_state

    config = _config(tmp_path)
    _patch_happy_path(monkeypatch)
    real_publish = materialize_state.publish_prepared_file_noreplace

    def die_before_terminal_publish(
        prepared_path: Path,
        destination_path: Path,
        *,
        expected_sha256: str,
    ) -> None:
        if prepared_path.name == materialize_state.COMPLETE_PREPARED_NAME:
            raise KeyboardInterrupt("crash after terminal marker capture")
        real_publish(
            prepared_path,
            destination_path,
            expected_sha256=expected_sha256,
        )

    monkeypatch.setattr(
        materialize_state,
        "publish_prepared_file_noreplace",
        die_before_terminal_publish,
    )
    with pytest.raises(KeyboardInterrupt, match="terminal marker capture"):
        materialize_from_artifacts(config)

    marker_path = config.target_root / INCOMPLETE_MARKER_NAME
    assert not marker_path.exists()
    monkeypatch.setattr(
        materialize_state,
        "publish_prepared_file_noreplace",
        real_publish,
    )
    result = materialize_from_artifacts(config)

    assert result["status"] == "recovered_materialized"
    assert _read_json(marker_path)["status"] == "complete"


def test_manifest_identity_change_aborts_before_target_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    old_report = b"old-report\n"
    report_path.write_bytes(old_report)
    _patch_happy_path(monkeypatch)

    def materialize_then_change_manifest(**kwargs: object) -> dict[str, object]:
        conversions = _fake_materializer(**kwargs)
        (config.artifact_dir / "universe_manifest_v4.json").write_text(
            '{"changed": true}\n',
            encoding="utf-8",
        )
        return conversions

    monkeypatch.setattr(
        materialize_module,
        "_materialize_lerobot_splits",
        materialize_then_change_manifest,
    )

    with pytest.raises(RuntimeError, match="manifest changed during materialization"):
        materialize_from_artifacts(config)

    assert not config.target_root.exists()
    assert report_path.read_bytes() == old_report


def test_interrupt_after_directory_rename_rolls_back_owned_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    old_report = b"old-report\n"
    report_path.write_bytes(old_report)
    real_rename = materialize_transaction.rename_directory_noreplace_bound
    _patch_happy_path(monkeypatch)

    def rename_then_interrupt(
        source: Path,
        target: Path,
        *,
        expected_source_identity: tuple[int, int],
    ) -> None:
        real_rename(
            source,
            target,
            expected_source_identity=expected_source_identity,
        )
        if target == config.target_root:
            raise KeyboardInterrupt("interrupt after directory commit")

    monkeypatch.setattr(
        "script.track3_1.materialize_univtac.rename_directory_noreplace_bound",
        rename_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="interrupt after directory commit"):
        materialize_from_artifacts(config)

    assert not config.target_root.exists()
    assert report_path.read_bytes() == old_report
    staging_roots = _staging_roots(config)
    assert len(staging_roots) == 1
    marker = _read_json(staging_roots[0] / INCOMPLETE_MARKER_NAME)
    assert marker["status"] == "incomplete"
    assert marker["phase"] == "recovered_to_staging"
    assert marker["error_type"] == "KeyboardInterrupt"


def test_exchange_then_interrupt_commits_when_previous_matches_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    old_report = b"old-report\n"
    report_path.write_bytes(old_report)
    real_exchange = materialize_transaction.rename_paths_exchange
    exchange_calls = 0
    _patch_happy_path(monkeypatch)

    def exchange_then_interrupt(source: Path, target: Path) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        real_exchange(source, target)
        if exchange_calls == 1:
            raise KeyboardInterrupt("interrupt after report exchange")

    monkeypatch.setattr(
        materialize_transaction,
        "rename_paths_exchange",
        exchange_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="interrupt after report exchange"):
        materialize_from_artifacts(config)

    assert exchange_calls == 1
    assert config.target_root.is_dir()
    assert report_path.read_bytes() != old_report
    marker_path = config.target_root / INCOMPLETE_MARKER_NAME
    assert _read_json(marker_path)["status"] == "complete"


def test_exchange_interrupt_rolls_back_when_previous_changed_after_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    report_path.write_bytes(b"baseline-report\n")
    concurrent_report = b"concurrent-report-after-baseline\n"
    real_exchange = materialize_transaction.rename_paths_exchange
    exchange_calls = 0
    _patch_happy_path(monkeypatch)

    def race_exchange_then_interrupt(source: Path, target: Path) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        if exchange_calls == 1:
            report_path.write_bytes(concurrent_report)
        real_exchange(source, target)
        if exchange_calls == 1:
            raise KeyboardInterrupt("interrupt before previous identity check")

    monkeypatch.setattr(
        materialize_transaction,
        "rename_paths_exchange",
        race_exchange_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="previous identity check"):
        materialize_from_artifacts(config)

    assert exchange_calls == 2
    assert report_path.read_bytes() == concurrent_report
    assert not config.target_root.exists()
    assert len(_staging_roots(config)) == 1


def test_recovery_restores_displaced_concurrent_report_after_hard_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    staging_root, _ = _write_crashed_target(
        config,
        status="ready_to_publish",
        publishable_candidate=True,
    )
    report_path = config.artifact_dir / "conversion_report.json"
    candidate_path = config.artifact_dir / (
        f".conversion_report.json.candidate-{staging_root.name}"
    )
    concurrent_report = b"concurrent-report-after-recorded-baseline\n"
    report_path.write_bytes(concurrent_report)

    # This is the persisted filesystem image of a process dying immediately
    # after RENAME_EXCHANGE and before checking the displaced report baseline.
    materialize_transaction.rename_paths_exchange(candidate_path, report_path)
    _patch_happy_path(monkeypatch)

    with pytest.raises(
        materialize_transaction.ConcurrentArtifactWriteError,
        match="changed during materialization",
    ):
        materialize_from_artifacts(config)

    assert report_path.read_bytes() == concurrent_report
    assert not config.target_root.exists()
    assert staging_root.is_dir()
    assert candidate_path.is_file()


def test_next_invocation_cleans_only_owned_precommit_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    target_root = config.target_root.resolve(strict=False)
    owned_staging = target_root.parent / ".lerobot.incomplete-owned"
    foreign_staging = target_root.parent / ".lerobot.incomplete-foreign"
    owned_staging.mkdir()
    foreign_staging.mkdir()
    owned_candidate = config.artifact_dir / (
        f".conversion_report.json.candidate-{owned_staging.name}"
    )
    materialize_transaction._write_json_atomic(
        owned_candidate,
        {"conversion_report_sha256": "c" * 64},
    )
    owned_candidate_sha256 = hashlib.sha256(owned_candidate.read_bytes()).hexdigest()
    materialize_module._write_state_marker(
        owned_staging,
        target_root=target_root,
        staging_root=owned_staging,
        manifest_sha256="a" * 64,
        phase="committing_target",
        conversion_report_sha256="c" * 64,
        report_baseline_sha256=None,
        candidate_file_sha256=owned_candidate_sha256,
    )
    materialize_module._write_state_marker(
        foreign_staging,
        target_root=target_root,
        staging_root=foreign_staging,
        manifest_sha256="b" * 64,
        phase="committing_target",
        conversion_report_sha256="d" * 64,
        report_baseline_sha256=None,
    )
    _patch_happy_path(monkeypatch)

    result = materialize_from_artifacts(config)

    assert result["status"] == "materialized"
    assert not owned_staging.exists()
    assert not owned_candidate.exists()
    assert foreign_staging.is_dir()


def test_owned_staging_with_mismatched_candidate_is_not_touched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    target_root = config.target_root.resolve(strict=False)
    owned_staging = target_root.parent / ".lerobot.incomplete-mismatch"
    owned_staging.mkdir()
    candidate_path = config.artifact_dir / (
        f".conversion_report.json.candidate-{owned_staging.name}"
    )
    candidate_payload = b"concurrent-candidate\n"
    candidate_path.write_bytes(candidate_payload)
    materialize_module._write_state_marker(
        owned_staging,
        target_root=target_root,
        staging_root=owned_staging,
        manifest_sha256="a" * 64,
        phase="committing_target",
        candidate_file_sha256=hashlib.sha256(b"expected-candidate\n").hexdigest(),
    )
    _patch_happy_path(monkeypatch)

    with pytest.raises(RuntimeError, match="unverified conversion report candidate"):
        materialize_from_artifacts(config)

    assert owned_staging.is_dir()
    assert candidate_path.read_bytes() == candidate_payload
    assert not target_root.exists()


def test_owned_staging_cleanup_unlinks_symlink_and_fifo_without_following(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    target_root = config.target_root.resolve(strict=False)
    owned_staging = target_root.parent / ".lerobot.incomplete-indirect"
    owned_staging.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    external_payload = external_root / "preserve.txt"
    external_payload.write_text("preserve\n", encoding="utf-8")
    (owned_staging / "external-alias").symlink_to(
        external_root,
        target_is_directory=True,
    )
    os.mkfifo(owned_staging / "named-pipe")
    materialize_module._write_state_marker(
        owned_staging,
        target_root=target_root,
        staging_root=owned_staging,
        manifest_sha256="a" * 64,
        phase="materializing",
    )
    _patch_happy_path(monkeypatch)

    result = materialize_from_artifacts(config)

    assert result["status"] == "materialized"
    assert not owned_staging.exists()
    assert external_payload.read_text(encoding="utf-8") == "preserve\n"


def test_owned_staging_partial_cleanup_is_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    target_root = config.target_root.resolve(strict=False)
    owned_staging = target_root.parent / ".lerobot.incomplete-partial"
    owned_staging.mkdir()
    marker_path = owned_staging / INCOMPLETE_MARKER_NAME
    materialize_module._write_state_marker(
        owned_staging,
        target_root=target_root,
        staging_root=owned_staging,
        manifest_sha256="a" * 64,
        phase="materializing",
    )
    _patch_happy_path(monkeypatch)
    real_rmdir = materialize_tree.os.rmdir

    def fail_final_rmdir(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if isinstance(path, str) and path.startswith(".materialize-delete-"):
            raise OSError("injected cleanup failure")
        real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(
        materialize_tree.os,
        "rmdir",
        fail_final_rmdir,
    )
    with pytest.raises(OSError, match="injected cleanup failure"):
        materialize_from_artifacts(config)

    assert marker_path.is_file()
    assert not target_root.exists()
    assert _staging_roots(config) == [owned_staging]

    monkeypatch.setattr(materialize_tree.os, "rmdir", real_rmdir)
    result = materialize_from_artifacts(config)
    assert result["status"] == "materialized"
    assert not owned_staging.exists()


def test_atomic_exchange_preserves_write_injected_after_baseline_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    report_path = config.artifact_dir / "conversion_report.json"
    report_path.write_bytes(b"baseline-report\n")
    concurrent_report = b"write-injected-immediately-before-exchange\n"
    real_exchange = materialize_transaction.rename_paths_exchange
    exchange_calls = 0
    _patch_happy_path(monkeypatch)

    def inject_then_exchange(source: Path, target: Path) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        if exchange_calls == 1:
            report_path.write_bytes(concurrent_report)
        real_exchange(source, target)

    monkeypatch.setattr(
        materialize_transaction,
        "rename_paths_exchange",
        inject_then_exchange,
    )

    with pytest.raises(RuntimeError, match="changed during materialization"):
        materialize_from_artifacts(config)

    assert exchange_calls == 2
    assert report_path.read_bytes() == concurrent_report
    assert not config.target_root.exists()
    assert len(_staging_roots(config)) == 1


def test_prepare_and_materialize_share_the_artifact_lock(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    with materialize_transaction.exclusive_artifact_lock(artifact_dir):
        with pytest.raises(RuntimeError, match="already in progress"):
            prepare_artifacts(SimpleNamespace(artifact_dir=artifact_dir))
