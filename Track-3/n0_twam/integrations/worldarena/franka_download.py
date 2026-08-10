# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Resumable manifest-driven download for the official Franka dataset."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .franka_manifest import (
    FrankaDatasetInventory,
    FrankaFileRecord,
    canonical_sha256,
    load_franka_inventory,
    sha256_file,
)

DEFAULT_ENDPOINT = "https://hf-mirror.com"


@dataclass(frozen=True)
class DownloadSummary:
    downloaded_files: int
    reused_files: int
    total_files: int
    total_bytes: int
    receipt_path: Path
    receipt_sha256: str


def _destination(root: Path, relative_path: str) -> Path:
    destination = root.joinpath(*relative_path.split("/"))
    resolved_parent = destination.parent.resolve(strict=False)
    if root != resolved_parent and root not in resolved_parent.parents:
        raise ValueError(f"download path escapes root: {relative_path}")
    return destination


def _record_url(
    inventory: FrankaDatasetInventory,
    record: FrankaFileRecord,
    endpoint: str,
) -> str:
    quoted_repo = "/".join(
        urllib.parse.quote(part, safe="") for part in inventory.repo_id.split("/")
    )
    quoted_path = "/".join(
        urllib.parse.quote(part, safe="") for part in record.relative_path.split("/")
    )
    return (
        f"{endpoint.rstrip('/')}/datasets/{quoted_repo}/resolve/"
        f"{inventory.revision}/{quoted_path}?download=true"
    )


def _download_one(
    *,
    inventory: FrankaDatasetInventory,
    record: FrankaFileRecord,
    root: Path,
    endpoint: str,
    timeout_seconds: float,
    retries: int,
) -> bool:
    destination = _destination(root, record.relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        record.verify(destination)
        return False
    partial = destination.with_name(destination.name + ".part")
    if partial.is_symlink() or destination.is_symlink():
        raise ValueError(f"download target may not be a symlink: {destination}")
    url = _record_url(inventory, record, endpoint)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > record.size_bytes:
            partial.unlink()
            offset = 0
        headers = {"User-Agent": "n0-twam-track32/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                if offset and status != 206:
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("server did not honor the resume range")
                mode = "ab" if offset else "xb"
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            if partial.stat().st_size != record.size_bytes:
                raise RuntimeError(
                    f"incomplete download for {record.relative_path}: "
                    f"{partial.stat().st_size}/{record.size_bytes}"
                )
            record.verify(partial)
            os.replace(partial, destination)
            record.verify(destination)
            return True
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            if attempt >= retries:
                break
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"download failed for {record.relative_path}") from last_error


def _write_receipt(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite download receipt: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def download_franka_dataset(
    *,
    inventory_path: Path,
    output_root: Path,
    receipt_path: Path,
    endpoint: str = DEFAULT_ENDPOINT,
    workers: int = 8,
    timeout_seconds: float = 120.0,
    retries: int = 5,
) -> DownloadSummary:
    """Download, resume, and verify every file before sealing a receipt."""

    if workers <= 0 or retries < 0 or timeout_seconds <= 0.0:
        raise ValueError("download workers/timeout/retries are invalid")
    if not endpoint.startswith("https://"):
        raise ValueError("download endpoint must use HTTPS")
    inventory = load_franka_inventory(inventory_path)
    root = Path(output_root).expanduser().resolve(strict=False)
    receipt = Path(receipt_path).expanduser().resolve(strict=False)
    if receipt == root or root in receipt.parents:
        raise ValueError("download receipt must be outside the immutable data root")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)

    downloaded = 0
    reused = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                inventory=inventory,
                record=record,
                root=root,
                endpoint=endpoint,
                timeout_seconds=timeout_seconds,
                retries=retries,
            ): record.relative_path
            for record in inventory.records
        }
        for future in as_completed(futures):
            if future.result():
                downloaded += 1
            else:
                reused += 1

    for record in inventory.records:
        record.verify(_destination(root, record.relative_path))
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "repo_id": inventory.repo_id,
        "revision": inventory.revision,
        "inventory_path": str(inventory.source_path),
        "inventory_sha256": inventory.manifest_sha256,
        "canonical_records_sha256": inventory.records_sha256,
        "dataset_root": str(root),
        "file_count": len(inventory.records),
        "total_bytes": inventory.total_bytes,
        "dataset_identity_sha256": canonical_sha256(
            {
                "repo_id": inventory.repo_id,
                "revision": inventory.revision,
                "canonical_records_sha256": inventory.records_sha256,
            }
        ),
    }
    receipt_sha256 = _write_receipt(receipt, payload)
    return DownloadSummary(
        downloaded_files=downloaded,
        reused_files=reused,
        total_files=len(inventory.records),
        total_bytes=inventory.total_bytes,
        receipt_path=receipt,
        receipt_sha256=receipt_sha256,
    )


__all__ = ("DEFAULT_ENDPOINT", "DownloadSummary", "download_franka_dataset")
