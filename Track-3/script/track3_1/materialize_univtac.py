#!/usr/bin/env python3
# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Materialize formal UniVTAC LeRobot repos from a frozen schema-v4 manifest."""

import logging
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.integrations.univtac.manifest import load_dataset_manifest  # noqa: E402
from script.track3_1.materialize_cleanup import (  # noqa: E402
    UnownedReportCandidateError,
    cleanup_owned_staging_transactions,
    remove_owned_report_displacement,
)

# isort: off
from script.track3_1.materialize_cli import (  # noqa: E402, F401
    MaterializeConfig,
    parse_args as _parse_args,
    run_cli,
)
from script.track3_1.materialize_manifest_contract import (  # noqa: E402
    validate_formal_manifest as _validate_formal_manifest,
)
from script.track3_1.materialize_report import (  # noqa: E402
    conversion_report as _conversion_report,
    prepare_report_candidate,
    publish_prepared_candidate,
    verify_physical_bundles as _verify_physical_bundles,
)
from script.track3_1.materialize_state import (  # noqa: E402
    committed_result as _committed_result,
    directory_identity,
    resolve_terminal_staging_root,
    write_complete_marker as _write_complete_marker,
    write_state_marker as _write_state_marker,
)
from script.track3_1.materialize_publication_journal import (  # noqa: E402
    publication_intent_path,
    publication_intent_present,
    publish_directory_with_intent,
    recover_publication_intents,
)

# isort: on
from script.track3_1.materialize_transaction import (  # noqa: E402, F401; noqa: E402
    ExistingTransaction,
    _canonical_sha256,
    _write_json_atomic,
    directory_identity_or_none,
    exclusive_artifact_lock,
    file_sha256_or_none,
    is_canonical_json_file,
    load_existing_transaction,
    probe_transaction_filesystem,
    publish_report_cas,
    rename_directory_noreplace_bound,
    report_logical_sha256,
    rollback_directory_to_staging,
    root_belongs_to_transaction,
    verify_interrupted_report_commit,
)
from script.track3_1.materialize_writeahead import (  # noqa: E402
    recover_creation_intents,
    release_creation_intent,
    reserve_staging_root,
)
from script.track3_1.prepare_univtac import (  # noqa: E402
    _materialize_lerobot_splits,
)

LOGGER = logging.getLogger("n0_twam.materialize_univtac")
INCOMPLETE_MARKER_NAME = ".materialization_state.json"
STAGING_REPORT_NAME = ".conversion_report.staging.json"
STAGING_CANDIDATE_NAME = ".conversion_report.prepared.json"
REPORT_CANDIDATE_PREFIX = ".conversion_report.json.candidate-"


def _finalize_recovered_commit(
    *,
    manifest_path: Path,
    report_path: Path,
    target_root: Path,
    transaction: ExistingTransaction,
    manifest_sha256: str,
) -> dict[str, object]:
    expected_report_sha256 = transaction.conversion_report_sha256
    if expected_report_sha256 is None:
        raise RuntimeError("committed materialization has no report identity")
    verified_target_identity = directory_identity(target_root)
    if verified_target_identity != transaction.target_identity:
        raise RuntimeError("recovered target identity changed before verification")
    try:
        split_counts = _verify_physical_bundles(
            manifest_path=manifest_path,
            conversion_report_path=report_path,
            materialize_root=target_root,
        )
    except BaseException as exc:
        raise RuntimeError(
            "conversion report is committed but the target cannot be verified; "
            "target retained for manual recovery"
        ) from exc
    if transaction.status != "complete":
        try:
            remove_owned_report_displacement(
                transaction.candidate_path,
                expected_sha256=transaction.report_baseline_sha256,
            )
        except UnownedReportCandidateError:
            LOGGER.warning(
                "Preserving foreign report-candidate bytes after verified commit: %s",
                transaction.candidate_path,
            )
    try:
        _write_complete_marker(
            target_root=target_root,
            staging_root=transaction.staging_root,
            manifest_sha256=manifest_sha256,
            conversion_report_sha256=expected_report_sha256,
            expected_target_identity=verified_target_identity,
            expected_marker_sha256=transaction.marker_file_sha256,
        )
    except BaseException as exc:
        raise RuntimeError(
            "conversion report is committed but the complete marker could not "
            "be finalized; target retained for a later recovery invocation"
        ) from exc
    return _committed_result(  # type: ignore[no-any-return]
        manifest_sha256=manifest_sha256,
        conversion_report_sha256=expected_report_sha256,
        target_root=target_root,
        split_counts=split_counts,
        status="recovered_materialized",
    )


def _write_report_committed_marker(
    *,
    target_root: Path,
    transaction: ExistingTransaction,
    manifest_sha256: str,
) -> ExistingTransaction:
    marker_file_sha256 = _write_state_marker(
        target_root,
        target_root=target_root,
        staging_root=transaction.staging_root,
        manifest_sha256=manifest_sha256,
        phase="report_committed",
        status="ready_to_publish",
        conversion_report_sha256=transaction.conversion_report_sha256,
        report_baseline_sha256=transaction.report_baseline_sha256,
        candidate_file_sha256=transaction.candidate_file_sha256,
        expected_root_identity=transaction.target_identity,
        expected_marker_sha256=transaction.marker_file_sha256,
    )
    return replace(
        transaction,
        phase="report_committed",
        status="ready_to_publish",
        marker_file_sha256=marker_file_sha256,
    )


def _preserve_recovery_failure(
    *,
    target_root: Path,
    transaction: ExistingTransaction,
    manifest_sha256: str,
    error: BaseException,
) -> None:
    if not root_belongs_to_transaction(
        target_root,
        target_root=target_root,
        staging_root=transaction.staging_root,
        manifest_sha256=manifest_sha256,
        marker_name=INCOMPLETE_MARKER_NAME,
        expected_root_identity=transaction.target_identity,
        expected_marker_sha256=transaction.marker_file_sha256,
    ):
        raise RuntimeError(
            "current target ownership was lost; foreign root preserved in place"
        )
    incomplete_root = rollback_directory_to_staging(
        target_root=target_root,
        staging_root=transaction.staging_root,
        expected_target_identity=transaction.target_identity,
    )
    if incomplete_root != transaction.staging_root:
        raise RuntimeError("unable to move owned target back to staging")
    _write_state_marker(
        incomplete_root,
        target_root=target_root,
        staging_root=transaction.staging_root,
        manifest_sha256=manifest_sha256,
        phase="recovered_to_staging",
        conversion_report_sha256=transaction.conversion_report_sha256,
        report_baseline_sha256=transaction.report_baseline_sha256,
        candidate_file_sha256=transaction.candidate_file_sha256,
        error=error,
        expected_root_identity=transaction.target_identity,
        expected_marker_sha256=transaction.marker_file_sha256,
    )


def _recover_existing_target(
    *,
    artifact_dir: Path,
    manifest_path: Path,
    report_path: Path,
    target_root: Path,
    manifest_sha256: str,
) -> dict[str, object] | None:
    transaction = load_existing_transaction(
        artifact_dir=artifact_dir,
        target_root=target_root,
        manifest_sha256=manifest_sha256,
        marker_name=INCOMPLETE_MARKER_NAME,
        candidate_prefix=REPORT_CANDIDATE_PREFIX,
    )
    if transaction.status == "incomplete" and transaction.phase == "rollback_pending":
        _preserve_recovery_failure(
            target_root=target_root,
            transaction=transaction,
            manifest_sha256=manifest_sha256,
            error=RuntimeError("resuming a durable materialization rollback"),
        )
        return None
    expected_report_sha256 = transaction.conversion_report_sha256
    if transaction.status == "complete":
        if (
            transaction.phase != "complete"
            or expected_report_sha256 is None
            or report_logical_sha256(report_path) != expected_report_sha256
        ):
            raise RuntimeError(
                "complete materialization marker does not match the published "
                "report; target retained for manual recovery"
            )
        return _finalize_recovered_commit(
            manifest_path=manifest_path,
            report_path=report_path,
            target_root=target_root,
            transaction=transaction,
            manifest_sha256=manifest_sha256,
        )

    report_is_candidate = (
        expected_report_sha256 is not None
        and transaction.candidate_file_sha256 is not None
        and file_sha256_or_none(report_path) == transaction.candidate_file_sha256
        and report_logical_sha256(report_path) == expected_report_sha256
    )
    if report_is_candidate:
        assert expected_report_sha256 is not None
        assert transaction.candidate_file_sha256 is not None
        report_commit_verified = transaction.phase == "report_committed"
        try:
            if not report_commit_verified:
                if not transaction.baseline_declared:
                    raise RuntimeError("report CAS has no recorded baseline")
                verify_interrupted_report_commit(
                    candidate_path=transaction.candidate_path,
                    report_path=report_path,
                    expected_candidate_sha256=transaction.candidate_file_sha256,
                    expected_report_sha256=expected_report_sha256,
                    expected_baseline_sha256=transaction.report_baseline_sha256,
                )
                report_commit_verified = True
                transaction = _write_report_committed_marker(
                    target_root=target_root,
                    transaction=transaction,
                    manifest_sha256=manifest_sha256,
                )
        except BaseException as exc:
            if report_commit_verified:
                raise RuntimeError(
                    "report commit was verified but its durable phase could not "
                    "be persisted; target retained for recovery"
                ) from exc
            _preserve_recovery_failure(
                target_root=target_root,
                transaction=transaction,
                manifest_sha256=manifest_sha256,
                error=exc,
            )
            raise
        return _finalize_recovered_commit(
            manifest_path=manifest_path,
            report_path=report_path,
            target_root=target_root,
            transaction=transaction,
            manifest_sha256=manifest_sha256,
        )

    candidate_ready = (
        expected_report_sha256 is not None
        and transaction.baseline_declared
        and transaction.candidate_file_sha256 is not None
        and file_sha256_or_none(transaction.candidate_path)
        == transaction.candidate_file_sha256
        and report_logical_sha256(transaction.candidate_path) == expected_report_sha256
    )
    if candidate_ready:
        try:
            _verify_physical_bundles(
                manifest_path=manifest_path,
                conversion_report_path=transaction.candidate_path,
                materialize_root=target_root,
            )
            outcome = publish_report_cas(
                candidate_path=transaction.candidate_path,
                report_path=report_path,
                expected_report_sha256=transaction.report_baseline_sha256,
            )
        except BaseException as exc:
            _preserve_recovery_failure(
                target_root=target_root,
                transaction=transaction,
                manifest_sha256=manifest_sha256,
                error=exc,
            )
            raise
        transaction = _write_report_committed_marker(
            target_root=target_root,
            transaction=transaction,
            manifest_sha256=manifest_sha256,
        )
        result = _finalize_recovered_commit(
            manifest_path=manifest_path,
            report_path=report_path,
            target_root=target_root,
            transaction=transaction,
            manifest_sha256=manifest_sha256,
        )
        if outcome.deferred_error is not None:
            raise outcome.deferred_error
        return result

    recovery_error = RuntimeError(
        "interrupted materialization had no safely publishable report candidate"
    )
    _preserve_recovery_failure(
        target_root=target_root,
        transaction=transaction,
        manifest_sha256=manifest_sha256,
        error=recovery_error,
    )
    return None


def materialize_from_artifacts(config: MaterializeConfig) -> dict[str, object]:
    """Lock, materialize, verify, and publish one formal artifact transaction."""
    artifact_dir = config.artifact_dir.expanduser().resolve(strict=True)
    with exclusive_artifact_lock(artifact_dir):
        return _materialize_locked(config, artifact_dir=artifact_dir)


def _materialize_locked(
    config: MaterializeConfig,
    *,
    artifact_dir: Path,
) -> dict[str, object]:
    requested_target = config.target_root.expanduser()
    report_path = artifact_dir / "conversion_report.json"
    markerless_target = False
    markerless_target_identity: tuple[int, int] | None = None
    if requested_target.is_symlink():
        raise FileExistsError(f"target root must not exist: {requested_target}")
    if requested_target.exists():
        if not requested_target.is_dir():
            raise FileExistsError(f"target root must not exist: {requested_target}")
        marker_path = requested_target / INCOMPLETE_MARKER_NAME
        if marker_path.is_symlink():
            raise FileExistsError(f"target root must not exist: {requested_target}")
        if marker_path.exists() and not marker_path.is_file():
            raise FileExistsError(f"target root must not exist: {requested_target}")
        if not marker_path.exists():
            if report_logical_sha256(report_path) is None:
                raise FileExistsError(f"target root must not exist: {requested_target}")
            markerless_target = True
            markerless_target_identity = directory_identity(requested_target)
    target_root = requested_target.resolve(strict=False)
    if config.fps <= 0:
        raise ValueError("fps must be positive")
    if config.image_writer_threads <= 0:
        raise ValueError("image_writer_threads must be positive")
    if not config.repo_id.strip():
        raise ValueError("repo_id must be non-empty")
    manifest_path = artifact_dir / "universe_manifest_v4.json"
    manifest_file_sha256 = file_sha256_or_none(manifest_path)
    if manifest_file_sha256 is None:
        raise FileNotFoundError(manifest_path)
    manifest = load_dataset_manifest(manifest_path, verify_sources=False)
    if file_sha256_or_none(manifest_path) != manifest_file_sha256:
        raise RuntimeError("dataset manifest changed while it was being loaded")
    _validate_formal_manifest(manifest)
    target_root.parent.mkdir(parents=True, exist_ok=True)
    if artifact_dir.stat().st_dev != target_root.parent.stat().st_dev:
        raise ValueError("artifact and target roots require one atomic filesystem")
    probe_transaction_filesystem(
        target_root.parent,
        require_exchange=not markerless_target,
        require_noreplace=True,
    )
    recover_publication_intents(
        artifact_dir,
        target_root,
        manifest.manifest_sha256,
    )
    if markerless_target:
        markerless_report_sha256 = report_logical_sha256(report_path)
        report_file_sha256 = file_sha256_or_none(report_path)
        if (
            markerless_report_sha256 is None
            or report_file_sha256 is None
            or not is_canonical_json_file(report_path)
        ):
            raise FileExistsError("existing markerless target has no valid report")
        try:
            assert markerless_target_identity is not None
            if directory_identity(target_root) != markerless_target_identity:
                raise RuntimeError("markerless target changed before verification")
            split_counts = _verify_physical_bundles(
                manifest_path=manifest_path,
                conversion_report_path=report_path,
                materialize_root=target_root,
            )
            if (
                file_sha256_or_none(manifest_path) != manifest_file_sha256
                or file_sha256_or_none(report_path) != report_file_sha256
                or report_logical_sha256(report_path) != markerless_report_sha256
            ):
                raise RuntimeError("markerless materialization artifacts changed")
        except BaseException as exc:
            raise FileExistsError(
                "existing markerless target failed full bundle verification"
            ) from exc
        markerless_staging = target_root.parent / (
            f".{target_root.name}.incomplete-markerless-"
            f"{markerless_report_sha256[:16]}"
        )
        markerless_staging = resolve_terminal_staging_root(
            target_root=target_root,
            manifest_sha256=manifest.manifest_sha256,
            fallback=markerless_staging,
        )
        try:
            _write_complete_marker(
                target_root=target_root,
                staging_root=markerless_staging,
                manifest_sha256=manifest.manifest_sha256,
                conversion_report_sha256=markerless_report_sha256,
                expected_target_identity=markerless_target_identity,
            )
        except BaseException as exc:
            raise RuntimeError(
                "verified markerless target could not acquire a durable "
                "complete claim"
            ) from exc
        if (
            file_sha256_or_none(manifest_path) != manifest_file_sha256
            or file_sha256_or_none(report_path) != report_file_sha256
            or report_logical_sha256(report_path) != markerless_report_sha256
        ):
            raise RuntimeError("markerless artifacts changed while claiming target")
        return _committed_result(  # type: ignore[no-any-return]
            manifest_sha256=manifest.manifest_sha256,
            conversion_report_sha256=markerless_report_sha256,
            target_root=target_root,
            split_counts=split_counts,
            status="recovered_materialized",
        )
    if requested_target.exists():
        recovered = _recover_existing_target(
            artifact_dir=artifact_dir,
            manifest_path=manifest_path,
            report_path=report_path,
            target_root=target_root,
            manifest_sha256=manifest.manifest_sha256,
        )
        if recovered is not None:
            return recovered
    recover_creation_intents(
        artifact_dir=artifact_dir,
        target_root=target_root,
        manifest_sha256=manifest.manifest_sha256,
        marker_name=INCOMPLETE_MARKER_NAME,
    )
    cleanup_owned_staging_transactions(
        artifact_dir=artifact_dir,
        target_root=target_root,
        manifest_sha256=manifest.manifest_sha256,
        marker_name=INCOMPLETE_MARKER_NAME,
        candidate_prefix=REPORT_CANDIDATE_PREFIX,
    )
    if target_root.exists() or target_root.is_symlink():
        raise FileExistsError(f"target root must not exist: {target_root}")
    report_baseline = file_sha256_or_none(report_path)
    staging_root, creation_intent_path = reserve_staging_root(
        artifact_dir=artifact_dir,
        target_root=target_root,
        manifest_sha256=manifest.manifest_sha256,
    )
    owned_staging_identity = directory_identity(staging_root)
    candidate_path = artifact_dir / f"{REPORT_CANDIDATE_PREFIX}{staging_root.name}"
    phase = "materializing"
    target_committed = False
    report_committed = False
    candidate_sha256: str | None = None
    expected_report_sha256: str | None = None
    marker_file_sha256: str | None = None
    try:
        marker_file_sha256 = _write_state_marker(
            staging_root,
            target_root=target_root,
            staging_root=staging_root,
            manifest_sha256=manifest.manifest_sha256,
            phase=phase,
            report_baseline_sha256=report_baseline,
            expected_root_identity=owned_staging_identity,
        )
        release_creation_intent(creation_intent_path)
        conversions = _materialize_lerobot_splits(
            manifest=manifest,
            materialize_root=staging_root,
            repo_id=config.repo_id,
            fps=config.fps,
            image_writer_threads=config.image_writer_threads,
            emit_standard_views=True,
        )
        phase = "verifying_staging"
        staging_report_path = staging_root / STAGING_REPORT_NAME
        _write_json_atomic(
            staging_report_path,
            _conversion_report(
                manifest_sha256=manifest.manifest_sha256,
                conversions=conversions,
                materialize_root=staging_root,
            ),
        )
        _verify_physical_bundles(
            manifest_path=manifest_path,
            conversion_report_path=staging_report_path,
            materialize_root=staging_root,
        )
        staging_report_path.unlink()
        final_report = _conversion_report(
            manifest_sha256=manifest.manifest_sha256,
            conversions=conversions,
            materialize_root=target_root,
        )
        prepared_candidate_path = staging_root / STAGING_CANDIDATE_NAME
        candidate_sha256 = prepare_report_candidate(
            prepared_path=prepared_candidate_path,
            candidate_path=candidate_path,
            payload=final_report,
        )
        expected_report_sha256 = str(final_report["conversion_report_sha256"])
        phase = "candidate_prepared"
        marker_file_sha256 = _write_state_marker(
            staging_root,
            target_root=target_root,
            staging_root=staging_root,
            manifest_sha256=manifest.manifest_sha256,
            phase=phase,
            conversion_report_sha256=expected_report_sha256,
            report_baseline_sha256=report_baseline,
            candidate_file_sha256=candidate_sha256,
            expected_root_identity=owned_staging_identity,
            expected_marker_sha256=marker_file_sha256,
        )
        publish_prepared_candidate(
            prepared_path=prepared_candidate_path,
            candidate_path=candidate_path,
            expected_sha256=candidate_sha256,
        )
        phase = "candidate_published"
        marker_file_sha256 = _write_state_marker(
            staging_root,
            target_root=target_root,
            staging_root=staging_root,
            manifest_sha256=manifest.manifest_sha256,
            phase=phase,
            conversion_report_sha256=expected_report_sha256,
            report_baseline_sha256=report_baseline,
            candidate_file_sha256=candidate_sha256,
            expected_root_identity=owned_staging_identity,
            expected_marker_sha256=marker_file_sha256,
        )
        if file_sha256_or_none(manifest_path) != manifest_file_sha256:
            raise RuntimeError("dataset manifest changed during materialization")
        phase = "committing_target"
        marker_file_sha256 = _write_state_marker(
            staging_root,
            target_root=target_root,
            staging_root=staging_root,
            manifest_sha256=manifest.manifest_sha256,
            phase=phase,
            conversion_report_sha256=expected_report_sha256,
            report_baseline_sha256=report_baseline,
            candidate_file_sha256=candidate_sha256,
            expected_root_identity=owned_staging_identity,
            expected_marker_sha256=marker_file_sha256,
        )
        assert candidate_sha256 is not None
        publish_directory_with_intent(
            artifact_dir,
            staging_root,
            target_root,
            candidate_path,
            manifest.manifest_sha256,
            marker_file_sha256,
            candidate_sha256,
            owned_staging_identity,
            rename_directory_noreplace_bound,
        )
        target_committed = True
        phase = "verifying_final"
        verified_target_identity = directory_identity(target_root)
        if verified_target_identity != owned_staging_identity:
            raise RuntimeError("committed target identity changed after publication")
        split_counts = _verify_physical_bundles(
            manifest_path=manifest_path,
            conversion_report_path=candidate_path,
            materialize_root=target_root,
        )
        phase = "publishing_report"
        if file_sha256_or_none(manifest_path) != manifest_file_sha256:
            raise RuntimeError("dataset manifest changed before report publication")
        marker_file_sha256 = _write_state_marker(
            target_root,
            target_root=target_root,
            staging_root=staging_root,
            manifest_sha256=manifest.manifest_sha256,
            phase=phase,
            status="ready_to_publish",
            conversion_report_sha256=expected_report_sha256,
            report_baseline_sha256=report_baseline,
            candidate_file_sha256=candidate_sha256,
            expected_root_identity=verified_target_identity,
            expected_marker_sha256=marker_file_sha256,
        )
        publish_outcome = publish_report_cas(
            candidate_path=candidate_path,
            report_path=report_path,
            expected_report_sha256=report_baseline,
        )
        report_committed = True
        phase = "report_committed"
        marker_file_sha256 = _write_state_marker(
            target_root,
            target_root=target_root,
            staging_root=staging_root,
            manifest_sha256=manifest.manifest_sha256,
            phase=phase,
            status="ready_to_publish",
            conversion_report_sha256=expected_report_sha256,
            report_baseline_sha256=report_baseline,
            candidate_file_sha256=candidate_sha256,
            expected_root_identity=verified_target_identity,
            expected_marker_sha256=marker_file_sha256,
        )
        try:
            remove_owned_report_displacement(
                candidate_path,
                expected_sha256=report_baseline,
            )
        except UnownedReportCandidateError:
            LOGGER.warning(
                "Preserving foreign report-candidate bytes after commit: %s",
                candidate_path,
            )
        _write_complete_marker(
            target_root=target_root,
            staging_root=staging_root,
            manifest_sha256=manifest.manifest_sha256,
            conversion_report_sha256=expected_report_sha256,
            expected_target_identity=verified_target_identity,
            expected_marker_sha256=marker_file_sha256,
        )
        if publish_outcome.deferred_error is not None:
            raise publish_outcome.deferred_error
    except BaseException as exc:
        report_committed = report_committed or (
            candidate_sha256 is not None
            and file_sha256_or_none(report_path) == candidate_sha256
        )
        if report_committed:
            LOGGER.exception(
                "Materialization report committed before interruption; "
                "target was not rolled back"
            )
            raise
        if publication_intent_present(
            publication_intent_path(artifact_dir, staging_root)
        ):
            raise
        trusted_target = marker_file_sha256 is not None and root_belongs_to_transaction(
            target_root,
            target_root=target_root,
            staging_root=staging_root,
            manifest_sha256=manifest.manifest_sha256,
            marker_name=INCOMPLETE_MARKER_NAME,
            expected_root_identity=owned_staging_identity,
            expected_marker_sha256=marker_file_sha256,
        )
        trusted_staging = directory_identity_or_none(staging_root) == (
            owned_staging_identity
        ) and (
            marker_file_sha256 is None
            or root_belongs_to_transaction(
                staging_root,
                target_root=target_root,
                staging_root=staging_root,
                manifest_sha256=manifest.manifest_sha256,
                marker_name=INCOMPLETE_MARKER_NAME,
                expected_root_identity=owned_staging_identity,
                expected_marker_sha256=marker_file_sha256,
            )
        )
        if not trusted_target and (target_committed or not trusted_staging):
            LOGGER.error(
                "Neither the required target nor staging ownership state is safe; "
                "foreign paths and report candidate preserved without rollback"
            )
            raise
        target_committed = target_committed or trusted_target
        try:
            remove_owned_report_displacement(
                candidate_path,
                expected_sha256=candidate_sha256,
            )
        except BaseException:
            LOGGER.exception("Unable to remove the owned conversion report candidate")
        if target_committed:
            assert marker_file_sha256 is not None
            _preserve_recovery_failure(
                target_root=target_root,
                transaction=ExistingTransaction(
                    status="incomplete",
                    phase=phase,
                    staging_root=staging_root,
                    candidate_path=candidate_path,
                    conversion_report_sha256=expected_report_sha256,
                    report_baseline_sha256=report_baseline,
                    baseline_declared=True,
                    candidate_file_sha256=candidate_sha256,
                    marker_file_sha256=marker_file_sha256,
                    target_identity=owned_staging_identity,
                ),
                manifest_sha256=manifest.manifest_sha256,
                error=exc,
            )
        elif trusted_staging:
            try:
                marker_file_sha256 = _write_state_marker(
                    staging_root,
                    target_root=target_root,
                    staging_root=staging_root,
                    manifest_sha256=manifest.manifest_sha256,
                    phase=phase,
                    conversion_report_sha256=expected_report_sha256,
                    report_baseline_sha256=report_baseline,
                    candidate_file_sha256=candidate_sha256,
                    error=exc,
                    expected_root_identity=owned_staging_identity,
                    expected_marker_sha256=marker_file_sha256,
                )
            except BaseException:
                LOGGER.exception(
                    "Unable to update incomplete marker at %s",
                    staging_root,
                )
        LOGGER.exception(
            "Materialization failed; incomplete state preserved at %s",
            staging_root,
        )
        raise
    LOGGER.info("Formal UniVTAC materialization ready at %s", target_root)
    return {
        "status": "materialized",
        "manifest_sha256": manifest.manifest_sha256,
        "conversion_report_sha256": final_report["conversion_report_sha256"],
        "target_root": str(target_root),
        "split_counts": split_counts,
    }


def main(argv: list[str] | None = None) -> int:
    return run_cli(  # type: ignore[no-any-return]
        argv,
        materialize=materialize_from_artifacts,
        logger=LOGGER,
    )


if __name__ == "__main__":
    raise SystemExit(main())
