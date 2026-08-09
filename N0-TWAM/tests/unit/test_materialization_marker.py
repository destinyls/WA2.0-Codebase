# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import json
from pathlib import Path

import pytest

from n0_twam.integrations.univtac.artifact_validation import (
    _table_inventory,
    verify_materialization_marker,
)


def _write_marker(
    root: Path,
    *,
    status: str,
    manifest_sha256: str = "a" * 64,
    report_sha256: str = "b" * 64,
) -> Path:
    marker = root / ".materialization_state.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "phase": "complete" if status == "complete" else status,
                "target_root": str(root.resolve()),
                "staging_root": str(root.parent / ".lerobot.incomplete-unit"),
                "manifest_sha256": manifest_sha256,
                "conversion_report_sha256": report_sha256,
            }
        ),
        encoding="utf-8",
    )
    return marker


def test_formal_verifier_accepts_absent_or_matching_complete_marker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lerobot"
    root.mkdir()

    verify_materialization_marker(
        root,
        manifest_sha256="a" * 64,
        conversion_report_sha256="b" * 64,
    )
    _write_marker(root, status="complete")
    verify_materialization_marker(
        root,
        manifest_sha256="a" * 64,
        conversion_report_sha256="b" * 64,
    )


@pytest.mark.parametrize("status", ["incomplete", "ready_to_publish"])
def test_formal_verifier_rejects_uncommitted_marker(
    tmp_path: Path,
    status: str,
) -> None:
    root = tmp_path / "lerobot"
    root.mkdir()
    _write_marker(root, status=status)

    with pytest.raises(ValueError, match="not complete"):
        verify_materialization_marker(
            root,
            manifest_sha256="a" * 64,
            conversion_report_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("manifest_sha256", "report_sha256", "message"),
    [
        ("c" * 64, "b" * 64, "manifest"),
        ("a" * 64, "c" * 64, "conversion report"),
    ],
)
def test_formal_verifier_rejects_complete_marker_identity_mismatch(
    tmp_path: Path,
    manifest_sha256: str,
    report_sha256: str,
    message: str,
) -> None:
    root = tmp_path / "lerobot"
    root.mkdir()
    _write_marker(
        root,
        status="complete",
        manifest_sha256=manifest_sha256,
        report_sha256=report_sha256,
    )

    with pytest.raises(ValueError, match=message):
        verify_materialization_marker(
            root,
            manifest_sha256="a" * 64,
            conversion_report_sha256="b" * 64,
        )


def test_formal_verifier_rejects_marker_symlink(tmp_path: Path) -> None:
    root = tmp_path / "lerobot"
    root.mkdir()
    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    (root / ".materialization_state.json").symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        verify_materialization_marker(
            root,
            manifest_sha256="a" * 64,
            conversion_report_sha256="b" * 64,
        )


def test_table_inventory_rejects_symlink_entries_and_root(tmp_path: Path) -> None:
    root = tmp_path / "train759"
    for name in ("meta", "data", "videos"):
        (root / name).mkdir(parents=True)
    external = tmp_path / "external.bin"
    external.write_bytes(b"external")
    (root / "videos" / "episode.mp4").symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        _table_inventory(root)

    (root / "videos" / "episode.mp4").unlink()
    (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _table_inventory(alias)
