# Copyright 2025-2026 NeoteAI Team. All rights reserved.
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import n0_twam.integrations.worldarena.agilex_official_download as download
from n0_twam.integrations.worldarena.agilex_manifest import canonical_sha256
from n0_twam.integrations.worldarena.agilex_official_schema import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
)


def _blob_sha(raw: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def test_tree_url_uses_mirror_supported_scoped_pagination() -> None:
    url = download._tree_url(
        "https://hf-mirror.com", prefix="clean_table", recursive=True
    )
    assert "/clean_table?recursive=true" in url
    assert "expand=false" in url
    assert "limit=1000" in url


def test_download_resumes_from_frozen_inventory(tmp_path: Path, monkeypatch) -> None:
    files = {"prompt.json": b"{}", "clean_table/episode_0/meta.json": b"{}"}
    records = [
        {
            "path": path,
            "size": len(raw),
            "identity": {"kind": "git_blob_sha1", "git_blob_sha1": _blob_sha(raw)},
        }
        for path, raw in sorted(files.items())
    ]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "frozen",
                "repo_id": OFFICIAL_REPO_ID,
                "revision": OFFICIAL_REVISION,
                "canonical_records_sha256": canonical_sha256(records),
                "file_count": len(records),
                "total_bytes": sum(len(raw) for raw in files.values()),
                "records": records,
            }
        ),
        encoding="utf-8",
    )

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        url = command[-1]
        relative = url.split(f"/{OFFICIAL_REVISION}/", 1)[1].split("?", 1)[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(files[relative])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    raw_root = tmp_path / "raw"
    receipt = tmp_path / "receipt.json"
    first = download.download_agilex_dataset(
        inventory_path=inventory,
        output_root=raw_root,
        receipt_path=receipt,
        workers=2,
    )
    assert first["downloaded_files"] == 2
    assert first["reused_files"] == 0
    assert (raw_root / "prompt.json").read_bytes() == b"{}"
