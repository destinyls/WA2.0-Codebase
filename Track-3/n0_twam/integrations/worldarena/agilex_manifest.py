# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Immutable AgileX source inventory and per-repository route contracts."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from n0_twam.actions.qpos14 import (
    CHANNEL_NAMES as QPOS14_CHANNEL_NAMES,
    CHANNEL_UNITS as QPOS14_CHANNEL_UNITS,
    GRIPPER_ENCODING as QPOS14_GRIPPER_ENCODING,
)
from n0_twam.embodiments import AGILEX_RGB_KEYS

from .agilex_actions import (
    MEASURED_NEXT_QPOS_SCHEMA,
    QPOS14_ACTION_SCHEMA,
    AgileXActionLabelContract,
    official_action_contract,
)

AGILEX_EMBODIMENT = "agilex_dual_qpos14_v1"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"manifest path is unsafe: {value!r}")
    return path.as_posix()


def _string_tuple(
    value: object,
    *,
    label: str,
    length: int | None = None,
    unique: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings")
    result = tuple(value)
    if unique and len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    if length is not None and len(result) != length:
        raise ValueError(f"{label} must contain exactly {length} entries")
    return result


@dataclass(frozen=True)
class AgileXRepoRoute:
    repo_id: str
    embodiment: str
    action_contract: AgileXActionLabelContract
    rgb_keys: tuple[str, ...]
    tactile_keys: tuple[str, ...]
    wrench_keys: tuple[str, ...]
    channel_names: tuple[str, ...]
    channel_units: tuple[str, ...]
    gripper_encoding: str
    temporal_alignment_identity: str

    @property
    def formal(self) -> bool:
        return self.action_contract.formal

    @property
    def action_schema(self) -> str:
        return self.action_contract.schema

    @property
    def action_label_source(self) -> str:
        return self.action_contract.label_source

    @property
    def route_identity(self) -> str:
        return canonical_sha256(self.to_json_dict())

    def to_json_dict(self) -> dict[str, object]:
        return {
            "repo_id": self.repo_id,
            "embodiment": self.embodiment,
            "action_schema": self.action_schema,
            "action_label_source": self.action_label_source,
            "formal": self.formal,
            "action_label_offset": self.action_contract.label_offset,
            "rgb_keys": list(self.rgb_keys),
            "tactile_keys": list(self.tactile_keys),
            "wrench_keys": list(self.wrench_keys),
            "channel_names": list(self.channel_names),
            "channel_units": list(self.channel_units),
            "gripper_encoding": self.gripper_encoding,
            "temporal_alignment_identity": self.temporal_alignment_identity,
        }


def build_repo_route(
    repo_id: str,
    payload: Mapping[str, object],
    *,
    allow_engineering: bool = False,
) -> AgileXRepoRoute:
    """Validate one explicit route; dimensional inference is forbidden."""

    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("AgileX route repo_id must be non-empty")
    embodiment = payload.get("embodiment")
    if embodiment != AGILEX_EMBODIMENT:
        raise ValueError("AgileX route embodiment mismatch")
    formal = payload.get("formal")
    if not isinstance(formal, bool):
        raise ValueError("AgileX route formal flag must be boolean")
    schema = payload.get("action_schema")
    source = payload.get("action_label_source")
    if formal:
        contract = official_action_contract(str(source))
        if schema != QPOS14_ACTION_SCHEMA:
            raise ValueError("formal AgileX route action schema mismatch")
    else:
        if not allow_engineering:
            raise ValueError("engineering AgileX route requires explicit opt-in")
        contract = AgileXActionLabelContract(
            schema=str(schema),
            label_source=str(source),
            formal=False,
            label_offset=1,
        )
        if schema != MEASURED_NEXT_QPOS_SCHEMA:
            raise ValueError("engineering AgileX route schema mismatch")
    rgb_keys = _string_tuple(payload.get("rgb_keys"), label="rgb_keys")
    if tuple(rgb_keys) != AGILEX_RGB_KEYS:
        raise ValueError("AgileX route requires the canonical ordered RGB keys")
    tactile_keys = _string_tuple(payload.get("tactile_keys"), label="tactile_keys")
    wrench_keys = _string_tuple(payload.get("wrench_keys"), label="wrench_keys")
    if any(
        not key.startswith("observation.images.") for key in rgb_keys + tactile_keys
    ):
        raise ValueError("AgileX image keys must use observation.images.*")
    if any(not key.startswith("observation.wrench.") for key in wrench_keys):
        raise ValueError("AgileX wrench keys must use observation.wrench.*")
    channel_names = _string_tuple(
        payload.get("channel_names"), label="channel_names", length=14
    )
    channel_units = _string_tuple(
        payload.get("channel_units"),
        label="channel_units",
        length=14,
        unique=False,
    )
    gripper_encoding = payload.get("gripper_encoding")
    if not isinstance(gripper_encoding, str) or not gripper_encoding:
        raise ValueError("AgileX route must seal the gripper encoding")
    if formal and (
        channel_names != QPOS14_CHANNEL_NAMES
        or channel_units != QPOS14_CHANNEL_UNITS
        or gripper_encoding != QPOS14_GRIPPER_ENCODING
    ):
        raise ValueError("formal AgileX route must use the canonical qpos14 layout")
    temporal_identity = _sha256(
        payload.get("temporal_alignment_identity"),
        label="temporal alignment identity",
    )
    return AgileXRepoRoute(
        repo_id=repo_id,
        embodiment=AGILEX_EMBODIMENT,
        action_contract=contract,
        rgb_keys=rgb_keys,
        tactile_keys=tactile_keys,
        wrench_keys=wrench_keys,
        channel_names=channel_names,
        channel_units=channel_units,
        gripper_encoding=gripper_encoding,
        temporal_alignment_identity=temporal_identity,
    )


@dataclass(frozen=True)
class AgileXFileRecord:
    relative_path: str
    size_bytes: int
    sha256: str

    def verify(self, root: Path) -> None:
        source_root = Path(root).expanduser()
        root_metadata = source_root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            raise ValueError("AgileX source root must be a regular directory")
        resolved_root = source_root.resolve(strict=True)
        path = source_root
        parts = PurePosixPath(self.relative_path).parts
        for index, part in enumerate(parts):
            path = path / part
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"AgileX source path contains a symlink: {path}")
            if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"AgileX source parent is not a directory: {path}")
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"AgileX source must be a regular file: {path}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError("AgileX source path escaped source_root") from error
        before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        digest = sha256_file(resolved)
        final = path.lstat()
        after = (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        )
        if before != after or stat.S_ISLNK(final.st_mode):
            raise ValueError(f"AgileX source changed while hashing: {path}")
        if final.st_size != self.size_bytes or digest != self.sha256:
            raise ValueError(f"AgileX source identity mismatch: {self.relative_path}")


@dataclass(frozen=True)
class AgileXDatasetManifest:
    source_path: Path
    dataset_id: str
    revision: str
    records: tuple[AgileXFileRecord, ...]
    routes: tuple[AgileXRepoRoute, ...]
    records_sha256: str

    @property
    def manifest_sha256(self) -> str:
        return sha256_file(self.source_path)

    @property
    def repo_route_manifest_sha256(self) -> str:
        return canonical_sha256(
            {route.repo_id: route.to_json_dict() for route in self.routes}
        )

    def route_for_repo(self, repo_id: str) -> AgileXRepoRoute:
        matches = [route for route in self.routes if route.repo_id == repo_id]
        if len(matches) != 1:
            raise KeyError(f"AgileX repo route is not uniquely defined: {repo_id}")
        return matches[0]


def load_agilex_manifest(
    path: Path,
    *,
    expected_file_sha256: str,
    selected_repo_ids: tuple[str, ...] | None = None,
    allow_engineering: bool = False,
) -> AgileXDatasetManifest:
    """Load a self-consistent frozen inventory bound to an external SHA."""

    candidate = Path(path).expanduser()
    metadata = candidate.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("AgileX manifest must be a regular non-symlink file")
    source = candidate.resolve(strict=True)
    expected = _sha256(expected_file_sha256, label="expected manifest SHA-256")
    if sha256_file(source) != expected:
        raise ValueError("AgileX manifest file identity mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid AgileX manifest: {source}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("AgileX manifest must be schema version 1")
    if payload.get("status") != "frozen":
        raise ValueError("AgileX manifest must be frozen")
    dataset_id, revision = payload.get("dataset_id"), payload.get("revision")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("AgileX manifest dataset_id must be non-empty")
    if not isinstance(revision, str) or not revision:
        raise ValueError("AgileX manifest revision must be non-empty")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("AgileX manifest records must be non-empty")
    declared_records_hash = _sha256(
        payload.get("canonical_records_sha256"), label="records SHA-256"
    )
    if canonical_sha256(raw_records) != declared_records_hash:
        raise ValueError("AgileX manifest canonical record hash mismatch")
    records: list[AgileXFileRecord] = []
    seen: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("AgileX manifest record must be an object")
        relative = _safe_relative_path(raw.get("path"))
        if relative in seen:
            raise ValueError(f"duplicate AgileX manifest path: {relative}")
        seen.add(relative)
        size = raw.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid AgileX record size: {relative}")
        records.append(
            AgileXFileRecord(
                relative_path=relative,
                size_bytes=size,
                sha256=_sha256(raw.get("sha256"), label="record SHA-256"),
            )
        )
    raw_routes = payload.get("repos")
    if not isinstance(raw_routes, dict) or not raw_routes:
        raise ValueError("AgileX manifest repos must be a non-empty object")
    routes = tuple(
        build_repo_route(repo_id, route, allow_engineering=allow_engineering)
        for repo_id, route in sorted(raw_routes.items())
        if isinstance(route, Mapping)
    )
    if len(routes) != len(raw_routes):
        raise ValueError("every AgileX repo route must be an object")
    if selected_repo_ids is not None and set(selected_repo_ids) != {
        route.repo_id for route in routes
    }:
        raise ValueError("AgileX route manifest must exactly cover selected repos")
    return AgileXDatasetManifest(
        source_path=source,
        dataset_id=dataset_id,
        revision=revision,
        records=tuple(records),
        routes=routes,
        records_sha256=declared_records_hash,
    )


__all__ = (
    "AGILEX_EMBODIMENT",
    "AgileXDatasetManifest",
    "AgileXFileRecord",
    "AgileXRepoRoute",
    "build_repo_route",
    "canonical_sha256",
    "load_agilex_manifest",
    "sha256_file",
)
