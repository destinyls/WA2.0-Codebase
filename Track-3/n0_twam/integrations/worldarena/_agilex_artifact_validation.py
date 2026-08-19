"""Private fail-closed validation helpers for AgileX data artifacts."""

from __future__ import annotations

import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from .agilex_manifest import AgileXRepoRoute

CONVERSION_SCHEMA_VERSION = 2
LATENT_INVENTORY_SCHEMA_VERSION = 2

_CONVERSION_KEYS = frozenset(
    "schema_version status kind source_manifest_sha256 "
    "repo_route_manifest_sha256 temporal_alignment_contract_sha256 "
    "dataset_root repo_ids repositories conversion_identity_sha256".split()
)
_REPOSITORY_KEYS = frozenset(
    "repo_id dataset_relative_path formal action_schema action_label_source "
    "action_label_offset repo_route_identity temporal_alignment_identity "
    "table_inventory".split()
)
_LATENT_KEYS = frozenset(
    "schema_version status kind source_manifest_sha256 "
    "repo_route_manifest_sha256 temporal_alignment_contract_sha256 "
    "dataset_root repo_ids conversion_receipt_sha256 conversion_identity_sha256 "
    "record_count records inventory_sha256".split()
)
_FILE_RECORD_KEYS = frozenset("path size_bytes sha256".split())


def require_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def safe_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} is unsafe: {value!r}")
    return path.as_posix()


def safe_repo_id(value: object) -> str:
    repo_id = safe_relative(value, label="AgileX repo_id")
    if len(PurePosixPath(repo_id).parts) != 1:
        raise ValueError("AgileX repo_id must be one dataset-root child")
    return repo_id


def regular_directory(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise ValueError(f"{label} is unavailable: {candidate}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink directory")
    return candidate.resolve(strict=True)


def canonical_routes(
    routes: Sequence[AgileXRepoRoute],
) -> tuple[AgileXRepoRoute, ...]:
    ordered = tuple(sorted(routes, key=lambda route: route.repo_id))
    if not ordered:
        raise ValueError("AgileX artifact routes must be non-empty")
    repo_ids = tuple(safe_repo_id(route.repo_id) for route in ordered)
    if len(repo_ids) != len(set(repo_ids)):
        raise ValueError("AgileX artifact routes contain duplicate repos")
    if any(not route.formal for route in ordered):
        raise ValueError("formal AgileX artifacts require formal action routes")
    return ordered


def resolve_dataset_layout(
    dataset_root: Path,
    routes: Sequence[AgileXRepoRoute],
) -> tuple[Path, tuple[AgileXRepoRoute, ...], dict[str, Path]]:
    """Resolve every route to exactly one direct, canonical LeRobot repo."""

    root = regular_directory(dataset_root, label="AgileX dataset root")
    ordered = canonical_routes(routes)
    repositories: dict[str, Path] = {}
    expected_info: set[Path] = set()
    for route in ordered:
        repo = regular_directory(
            root / route.repo_id, label=f"AgileX repo {route.repo_id}"
        )
        if repo.parent != root or repo.name != route.repo_id:
            raise ValueError(f"AgileX repo escaped dataset root: {route.repo_id}")
        info = repo / "meta" / "info.json"
        try:
            metadata = info.lstat()
        except OSError as error:
            raise ValueError(
                f"AgileX repo is missing canonical info.json: {repo}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"AgileX info.json must be a regular file: {info}")
        repositories[route.repo_id] = repo
        expected_info.add(info.resolve(strict=True))

    actual_info: list[Path] = []
    for candidate in root.rglob("*"):
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"AgileX dataset tree contains a symlink: {candidate}")
        if candidate.name == "info.json":
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    f"AgileX info.json must be a regular file: {candidate}"
                )
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError("AgileX info.json escaped dataset root") from error
            actual_info.append(resolved)
    if len(actual_info) != len(set(actual_info)) or set(actual_info) != expected_info:
        raise ValueError(
            "AgileX dataset must contain exactly one canonical repo root per route"
        )
    return root, ordered, repositories


def _table_record(record: object) -> tuple[str, int, str]:
    expected = {"relative_path", "size_bytes", "sha256"}
    if not isinstance(record, Mapping) or set(record) != expected:
        raise ValueError("AgileX table inventory record has an invalid schema")
    path = safe_relative(record["relative_path"], label="table inventory path")
    if PurePosixPath(path).parts[0] not in ("meta", "data", "videos"):
        raise ValueError("AgileX table inventory path has an invalid root")
    return (
        path,
        require_integer(record["size_bytes"], label="table inventory size", minimum=1),
        require_digest(record["sha256"], label="table inventory SHA-256"),
    )


def validate_conversion_shape(payload: Mapping[str, object]) -> None:
    if set(payload) != _CONVERSION_KEYS:
        raise ValueError("AgileX conversion receipt has an invalid schema")
    if (
        require_integer(payload["schema_version"], label="conversion schema", minimum=1)
        != CONVERSION_SCHEMA_VERSION
        or payload["status"] != "complete"
        or payload["kind"] != "agilex_conversion"
    ):
        raise ValueError("AgileX conversion receipt is not complete schema version 2")
    repositories = payload["repositories"]
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("AgileX conversion repositories must be non-empty")
    seen_repos: set[str] = set()
    for repository in repositories:
        if not isinstance(repository, Mapping) or set(repository) != _REPOSITORY_KEYS:
            raise ValueError("AgileX conversion repository has an invalid schema")
        repo_id = safe_repo_id(repository["repo_id"])
        if repo_id in seen_repos:
            raise ValueError("AgileX conversion receipt contains duplicate repos")
        seen_repos.add(repo_id)
        if repository["dataset_relative_path"] != repo_id:
            raise ValueError("AgileX conversion repository path mismatch")
        if type(repository["formal"]) is not bool or not repository["formal"]:
            raise ValueError("AgileX conversion route must be formal")
        require_integer(repository["action_label_offset"], label="action label offset")
        for field in ("action_schema", "action_label_source"):
            if not isinstance(repository[field], str) or not repository[field]:
                raise ValueError(f"AgileX conversion {field} must be non-empty")
        require_digest(repository["repo_route_identity"], label="repo-route identity")
        require_digest(
            repository["temporal_alignment_identity"], label="temporal identity"
        )
        inventory = repository["table_inventory"]
        if not isinstance(inventory, list) or not inventory:
            raise ValueError("AgileX conversion table inventory must be non-empty")
        paths = [_table_record(record)[0] for record in inventory]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("AgileX table inventory paths must be sorted and unique")
    repo_ids = payload["repo_ids"]
    if (
        not isinstance(repo_ids, list)
        or any(not isinstance(value, str) for value in repo_ids)
        or repo_ids != sorted(seen_repos)
    ):
        raise ValueError("AgileX conversion repo_ids are not canonical")


def validate_latent_shape(payload: Mapping[str, object]) -> None:
    if set(payload) != _LATENT_KEYS:
        raise ValueError("AgileX latent inventory has an invalid schema")
    if (
        require_integer(payload["schema_version"], label="latent schema", minimum=1)
        != LATENT_INVENTORY_SCHEMA_VERSION
        or payload["status"] != "complete"
        or payload["kind"] != "agilex_latent_inventory"
    ):
        raise ValueError("AgileX latent inventory is not complete schema version 2")
    count = require_integer(payload["record_count"], label="record_count", minimum=1)
    records = payload["records"]
    if not isinstance(records, list) or len(records) != count:
        raise ValueError("AgileX latent record_count mismatch")
    paths: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _FILE_RECORD_KEYS:
            raise ValueError("AgileX latent record has an invalid schema")
        path = safe_relative(record["path"], label="latent record path")
        if PurePosixPath(path).suffix != ".pth":
            raise ValueError("AgileX latent record must reference a .pth file")
        require_integer(record["size_bytes"], label="latent payload size", minimum=1)
        require_digest(record["sha256"], label="latent payload SHA-256")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("AgileX latent paths must be sorted and unique")


__all__ = (
    "CONVERSION_SCHEMA_VERSION",
    "LATENT_INVENTORY_SCHEMA_VERSION",
    "canonical_routes",
    "regular_directory",
    "require_digest",
    "resolve_dataset_layout",
    "validate_conversion_shape",
    "validate_latent_shape",
)
