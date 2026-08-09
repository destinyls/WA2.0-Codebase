from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from script.track3_1 import audit_univtac_sources as audit_module


@contextmanager
def _fake_open(record: object):
    yield f"handle:{record.relative_path}"


def test_source_audit_visits_every_manifest_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_record = SimpleNamespace(
        relative_path="b.hdf5",
        split="validation",
        size_bytes=13,
    )
    records = (
        SimpleNamespace(relative_path="a.hdf5", split="train", size_bytes=11),
        validation_record,
    )
    manifest = SimpleNamespace(
        schema_version=audit_module.MANIFEST_SCHEMA_VERSION,
        manifest_sha256="a" * 64,
        entries=records,
    )
    visited: list[tuple[object, str]] = []
    monkeypatch.setattr(
        audit_module,
        "load_dataset_manifest",
        lambda path, verify_sources: manifest,
    )
    monkeypatch.setattr(audit_module, "open_verified_hdf5", _fake_open)
    monkeypatch.setattr(
        audit_module,
        "_validate_hdf5_storage",
        lambda handle, source_label: visited.append((handle, source_label)),
    )

    report = audit_module.audit_manifest_sources(
        tmp_path / "manifest.json",
        expected_manifest_sha256="a" * 64,
        expected_entries=2,
    )

    assert visited == [
        ("handle:a.hdf5", "a.hdf5"),
        ("handle:b.hdf5", "b.hdf5"),
    ]
    assert report["status"] == "source_storage_audit_passed"
    assert report["split_counts"] == {"train": 1, "validation": 1}
    assert report["bytes_audited"] == 24


@pytest.mark.parametrize(
    ("manifest_sha256", "entry_count", "error"),
    (("b" * 64, 2, "identity"), ("a" * 64, 3, "episode count")),
)
def test_source_audit_rejects_identity_or_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_sha256: str,
    entry_count: int,
    error: str,
) -> None:
    manifest = SimpleNamespace(
        schema_version=audit_module.MANIFEST_SCHEMA_VERSION,
        manifest_sha256=manifest_sha256,
        entries=(SimpleNamespace(),) * entry_count,
    )
    monkeypatch.setattr(
        audit_module,
        "load_dataset_manifest",
        lambda path, verify_sources: manifest,
    )

    with pytest.raises(ValueError, match=error):
        audit_module.audit_manifest_sources(
            tmp_path / "manifest.json",
            expected_manifest_sha256="a" * 64,
            expected_entries=2,
        )
