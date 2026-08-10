# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fetch and seal the pinned official Franka Hugging Face inventory."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Mapping

from .franka_manifest import (
    OFFICIAL_FILE_COUNT,
    OFFICIAL_RECORDS_SHA256,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    OFFICIAL_TOTAL_BYTES,
    canonical_sha256,
    load_franka_inventory,
    sha256_file,
)

DEFAULT_API_ENDPOINT = "https://huggingface.co"


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _api_url(endpoint: str) -> str:
    if not endpoint.startswith("https://"):
        raise ValueError("Hugging Face API endpoint must use HTTPS")
    repo = "/".join(
        urllib.parse.quote(part, safe="") for part in OFFICIAL_REPO_ID.split("/")
    )
    revision = urllib.parse.quote(OFFICIAL_REVISION, safe="")
    return (
        f"{endpoint.rstrip('/')}/api/datasets/{repo}/revision/" f"{revision}?blobs=true"
    )


def _record(raw: object) -> dict[str, object]:
    sibling = _mapping(raw, label="Hugging Face sibling")
    path = sibling.get("rfilename")
    blob_id = sibling.get("blobId")
    size = sibling.get("size")
    if not isinstance(path, str) or not path:
        raise ValueError("Hugging Face sibling path is invalid")
    if (
        not isinstance(blob_id, str)
        or len(blob_id) != 40
        or any(character not in "0123456789abcdef" for character in blob_id)
    ):
        raise ValueError(f"Hugging Face blob ID is invalid for {path}")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"Hugging Face sibling size is invalid for {path}")
    lfs = sibling.get("lfs")
    if lfs is None:
        identity: dict[str, object] = {
            "kind": "git_blob_sha1",
            "git_blob_sha1": blob_id,
        }
    else:
        lfs_payload = _mapping(lfs, label=f"LFS metadata for {path}")
        digest = lfs_payload.get("sha256")
        lfs_size = lfs_payload.get("size")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or lfs_size != size
        ):
            raise ValueError(f"Hugging Face LFS identity is invalid for {path}")
        identity = {"kind": "content_sha256", "sha256": digest}
    return {
        "blob_id": blob_id,
        "identity": identity,
        "path": path,
        "size": size,
    }


def fetch_official_franka_inventory(
    *,
    output: Path,
    endpoint: str = DEFAULT_API_ENDPOINT,
    timeout_seconds: float = 120.0,
) -> dict[str, object]:
    """Publish a deterministic manifest only if the pinned API roster matches."""

    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    destination = Path(output).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Franka inventory already exists: {destination}")
    url = _api_url(endpoint)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "n0-twam-track32/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = _mapping(
            json.loads(response.read()),
            label="Hugging Face dataset API response",
        )
    if payload.get("sha") != OFFICIAL_REVISION:
        raise ValueError("Hugging Face resolved revision differs from the pin")
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise ValueError("Hugging Face response has no sibling inventory")
    records = sorted(
        (_record(value) for value in siblings), key=lambda item: str(item["path"])
    )
    records_hash = canonical_sha256(records)
    total_bytes = sum(int(record["size"]) for record in records)
    if (
        len(records) != OFFICIAL_FILE_COUNT
        or total_bytes != OFFICIAL_TOTAL_BYTES
        or records_hash != OFFICIAL_RECORDS_SHA256
    ):
        raise ValueError(
            "official Franka API inventory changed; refusing to publish an "
            "unreviewed dataset identity"
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "frozen",
        "repo_id": OFFICIAL_REPO_ID,
        "repo_type": "dataset",
        "requested_revision": OFFICIAL_REVISION,
        "resolved_revision": OFFICIAL_REVISION,
        "canonical_records_sha256": records_hash,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "records": records,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    raw = (
        json.dumps(
            manifest,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        load_franka_inventory(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    load_franka_inventory(destination)
    return {
        "schema_version": 1,
        "status": "complete",
        "inventory": str(destination),
        "inventory_file_sha256": sha256_file(destination),
        "records_sha256": records_hash,
        "file_count": len(records),
        "total_bytes": total_bytes,
    }


__all__ = (
    "DEFAULT_API_ENDPOINT",
    "fetch_official_franka_inventory",
)
