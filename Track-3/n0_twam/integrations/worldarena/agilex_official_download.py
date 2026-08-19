# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Pinned, resumable download of the official WorldArena AgileX subset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from .agilex_manifest import canonical_sha256, sha256_file
from .agilex_official_schema import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    OFFICIAL_TASKS,
)

DEFAULT_ENDPOINT = "https://hf-mirror.com"
_NEXT_LINK = re.compile(r"<([^>]+)>;\s*rel=\"next\"")


def _git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("official AgileX inventory path must be non-empty")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe official AgileX inventory path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class AgileXDownloadRecord:
    relative_path: str
    size_bytes: int
    identity_kind: str
    identity_digest: str

    def verify(self, path: Path) -> None:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not path.is_file()
            or metadata.st_size != self.size_bytes
        ):
            raise ValueError(f"official AgileX file size/type mismatch: {path}")
        actual = (
            sha256_file(path)
            if self.identity_kind == "content_sha256"
            else _git_blob_sha1(path, self.size_bytes)
        )
        if actual != self.identity_digest:
            raise ValueError(f"official AgileX file identity mismatch: {path}")


@dataclass(frozen=True)
class AgileXDownloadInventory:
    source_path: Path
    records: tuple[AgileXDownloadRecord, ...]
    records_sha256: str
    total_bytes: int


def _tree_url(endpoint: str, *, prefix: str | None, recursive: bool) -> str:
    repo = "/".join(
        urllib.parse.quote(part, safe="") for part in OFFICIAL_REPO_ID.split("/")
    )
    suffix = ""
    if prefix is not None:
        suffix = "/" + "/".join(
            urllib.parse.quote(part, safe="") for part in prefix.split("/")
        )
    return (
        f"{endpoint.rstrip('/')}/api/datasets/{repo}/tree/{OFFICIAL_REVISION}"
        f"{suffix}?recursive={'true' if recursive else 'false'}"
        "&expand=false&limit=1000"
    )


def _mirror_next(url: str, endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    mirror = urllib.parse.urlsplit(endpoint)
    return urllib.parse.urlunsplit(
        (mirror.scheme, mirror.netloc, parsed.path, parsed.query, "")
    )


def _selected(path: str) -> bool:
    return path == "prompt.json" or any(
        path.startswith(f"{task}/") for task in OFFICIAL_TASKS
    )


def _api_record(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping) or value.get("type") != "file":
        return None
    path = _safe_path(value.get("path"))
    if not _selected(path):
        return None
    size = value.get("size")
    oid = value.get("oid")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"invalid official AgileX size for {path}")
    lfs = value.get("lfs")
    if isinstance(lfs, Mapping):
        digest = lfs.get("oid")
        if lfs.get("size") != size or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid official AgileX LFS identity for {path}")
        identity = {"kind": "content_sha256", "sha256": digest}
    else:
        if not isinstance(oid, str) or len(oid) != 40:
            raise ValueError(f"invalid official AgileX git identity for {path}")
        identity = {"kind": "git_blob_sha1", "git_blob_sha1": oid}
    return {"path": path, "size": size, "identity": identity}


def _curl_tree_page(
    url: str, *, timeout_seconds: float
) -> tuple[list[object], str | None]:
    """Read one mirror page without following redirects to another host."""

    with tempfile.TemporaryDirectory(prefix="n0-agilex-tree-") as temporary:
        body = Path(temporary) / "body.json"
        headers = Path(temporary) / "headers.txt"
        command = [
            "curl",
            "--silent",
            "--show-error",
            "--dump-header",
            str(headers),
            "--output",
            str(body),
            "--write-out",
            "%{http_code}",
            "--connect-timeout",
            str(min(timeout_seconds, 30.0)),
            "--max-time",
            str(timeout_seconds),
            url,
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0 or result.stdout.strip() != "200":
            detail = result.stderr.strip() or f"HTTP {result.stdout.strip()}"
            raise RuntimeError(f"official AgileX tree request failed: {detail}")
        page = json.loads(body.read_bytes())
        raw_headers = headers.read_text(encoding="iso-8859-1")
    if not isinstance(page, list):
        raise ValueError("official AgileX tree API did not return a list")
    match = _NEXT_LINK.search(raw_headers)
    return page, None if match is None else match.group(1)


def _atomic_json(path: Path, payload: object) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode()
        + b"\n"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def fetch_agilex_download_inventory(
    *, output: Path, endpoint: str = DEFAULT_ENDPOINT, timeout_seconds: float = 120.0
) -> dict[str, object]:
    """Fetch the full pinned tree without relying on HF client pagination."""

    if not endpoint.startswith("https://") or timeout_seconds <= 0:
        raise ValueError("AgileX endpoint/timeout is invalid")
    records: list[dict[str, object]] = []
    scopes = ((None, False), *((task, True) for task in OFFICIAL_TASKS))
    for prefix, recursive in scopes:
        url: str | None = _tree_url(endpoint, prefix=prefix, recursive=recursive)
        while url is not None:
            page, next_url = _curl_tree_page(url, timeout_seconds=timeout_seconds)
            records.extend(record for value in page if (record := _api_record(value)))
            url = None if next_url is None else _mirror_next(next_url, endpoint)
    records.sort(key=lambda record: str(record["path"]))
    paths = [str(record["path"]) for record in records]
    if len(paths) != len(set(paths)) or "prompt.json" not in paths:
        raise ValueError(
            "official AgileX selected inventory is incomplete or duplicated"
        )
    for task in OFFICIAL_TASKS:
        if not any(path.startswith(f"{task}/") for path in paths):
            raise ValueError(f"official AgileX inventory has no files for {task}")
    core: dict[str, object] = {
        "schema_version": 1,
        "status": "frozen",
        "repo_id": OFFICIAL_REPO_ID,
        "revision": OFFICIAL_REVISION,
        "canonical_records_sha256": canonical_sha256(records),
        "file_count": len(records),
        "total_bytes": sum(int(record["size"]) for record in records),
        "records": records,
    }
    digest = _atomic_json(output, core)
    load_agilex_download_inventory(output)
    return {"status": "complete", "inventory": str(output), "sha256": digest, **core}


def load_agilex_download_inventory(path: Path) -> AgileXDownloadInventory:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("official AgileX download inventory schema mismatch")
    if payload.get("status") != "frozen" or payload.get("repo_id") != OFFICIAL_REPO_ID:
        raise ValueError("official AgileX download inventory source mismatch")
    if payload.get("revision") != OFFICIAL_REVISION:
        raise ValueError("official AgileX download inventory revision mismatch")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or canonical_sha256(
        raw_records
    ) != payload.get("canonical_records_sha256"):
        raise ValueError("official AgileX download record identity mismatch")
    records: list[AgileXDownloadRecord] = []
    for value in raw_records:
        if not isinstance(value, Mapping):
            raise ValueError("official AgileX download record must be an object")
        identity = value.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("official AgileX download identity must be an object")
        kind = identity.get("kind")
        key = "sha256" if kind == "content_sha256" else "git_blob_sha1"
        digest = identity.get(key)
        size = value.get("size")
        if kind not in ("content_sha256", "git_blob_sha1") or not isinstance(
            digest, str
        ):
            raise ValueError("unsupported official AgileX download identity")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("invalid official AgileX download size")
        records.append(
            AgileXDownloadRecord(_safe_path(value.get("path")), size, kind, digest)
        )
    total = sum(record.size_bytes for record in records)
    if payload.get("file_count") != len(records) or payload.get("total_bytes") != total:
        raise ValueError("official AgileX download inventory totals mismatch")
    return AgileXDownloadInventory(
        source, tuple(records), str(payload["canonical_records_sha256"]), total
    )


def _record_url(record: AgileXDownloadRecord, endpoint: str) -> str:
    path = "/".join(
        urllib.parse.quote(part, safe="") for part in record.relative_path.split("/")
    )
    return f"{endpoint.rstrip('/')}/datasets/{OFFICIAL_REPO_ID}/resolve/{OFFICIAL_REVISION}/{path}?download=true"


def _download_one(
    record: AgileXDownloadRecord, root: Path, endpoint: str, retries: int
) -> bool:
    destination = root.joinpath(*record.relative_path.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        record.verify(destination)
        return False
    if destination.is_symlink():
        raise ValueError(f"AgileX download target cannot be a symlink: {destination}")
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(retries + 1):
        command = [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "2",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            _record_url(record, endpoint),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                record.verify(partial)
                os.replace(partial, destination)
                record.verify(destination)
                return True
            except (OSError, ValueError):
                partial.unlink(missing_ok=True)
        if attempt < retries:
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(
        f"download failed for {record.relative_path}: {result.stderr[-500:]}"
    )


def download_agilex_dataset(
    *,
    inventory_path: Path,
    output_root: Path,
    receipt_path: Path,
    endpoint: str = DEFAULT_ENDPOINT,
    workers: int = 16,
    retries: int = 5,
) -> dict[str, object]:
    if workers <= 0 or retries < 0 or not endpoint.startswith("https://"):
        raise ValueError("AgileX download settings are invalid")
    inventory = load_agilex_download_inventory(inventory_path)
    root = Path(output_root).expanduser().resolve(strict=False)
    receipt = Path(receipt_path).expanduser().resolve(strict=False)
    if root == receipt or root in receipt.parents:
        raise ValueError("AgileX download receipt must be outside the raw root")
    root.mkdir(parents=True, exist_ok=True)
    counts = {True: 0, False: 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, record, root, endpoint, retries): record
            for record in inventory.records
        }
        for future in as_completed(futures):
            counts[future.result()] += 1
    for record in inventory.records:
        record.verify(root.joinpath(*record.relative_path.split("/")))
    core: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "repo_id": OFFICIAL_REPO_ID,
        "revision": OFFICIAL_REVISION,
        "inventory_sha256": sha256_file(inventory.source_path),
        "canonical_records_sha256": inventory.records_sha256,
        "raw_root": str(root),
        "file_count": len(inventory.records),
        "total_bytes": inventory.total_bytes,
    }
    receipt_sha = _atomic_json(receipt, core)
    return {
        **core,
        "downloaded_files": counts[True],
        "reused_files": counts[False],
        "receipt": str(receipt),
        "receipt_sha256": receipt_sha,
    }


__all__ = (
    "DEFAULT_ENDPOINT",
    "AgileXDownloadInventory",
    "AgileXDownloadRecord",
    "download_agilex_dataset",
    "fetch_agilex_download_inventory",
    "load_agilex_download_inventory",
)
