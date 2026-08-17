# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Content-addressed AgileX config bindings for repository and time routes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from n0_twam.embodiments import (
    AGILEX_ACTION_SCHEMA,
    AGILEX_EMBODIMENT_PROFILE_ID,
    RepoRouteManifestContract,
    build_agilex_repo_route_manifest,
    validate_repo_route_manifest_contract,
)

_ARTIFACT_DIGEST_ENV = {
    "source_manifest_file_sha256": "N0_TRACK3_AGILEX_SOURCE_MANIFEST_SHA256",
    "conversion_receipt_file_sha256": ("N0_TRACK3_AGILEX_CONVERSION_RECEIPT_SHA256"),
    "latent_inventory_file_sha256": "N0_TRACK3_AGILEX_LATENT_INVENTORY_SHA256",
    "repo_route_manifest_file_sha256": ("N0_TRACK3_AGILEX_REPO_ROUTE_MANIFEST_SHA256"),
    "temporal_alignment_file_sha256": ("N0_TRACK3_AGILEX_TEMPORAL_ALIGNMENT_SHA256"),
    "normalizer_file_sha256": "N0_TRACK3_AGILEX_NORMALIZER_SHA256",
}


def canonical_sha256(payload: object) -> str:
    """Hash one JSON-compatible contract using the repository convention."""

    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {source}")
    return payload


def load_runner_artifact_identity(profile: str) -> dict[str, object] | None:
    """Load the runner's exact immutable-artifact identity for one profile."""

    selected = os.environ.get("N0_TRACK3_AGILEX_TACTILE_PROFILE")
    raw = os.environ.get("N0_TRACK3_AGILEX_ARTIFACT_IDENTITY_JSON")
    if selected is not None and selected != profile:
        return None
    if raw is None:
        digests = {
            field: os.environ.get(name) for field, name in _ARTIFACT_DIGEST_ENV.items()
        }
        if not any(digests.values()):
            return None
        if any(value is None for value in digests.values()):
            raise ValueError("AgileX artifact identity environment is incomplete")
        payload: dict[str, object] = {
            "schema_version": 1,
            "embodiment_profile_id": AGILEX_EMBODIMENT_PROFILE_ID,
            "action_schema": AGILEX_ACTION_SCHEMA,
            "tactile_profile": profile,
            **digests,
        }
    else:
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("AgileX artifact identity JSON is invalid") from error
        if not isinstance(decoded, dict):
            raise ValueError("AgileX artifact identity must be a JSON object")
        payload = decoded
        if selected is None and payload.get("tactile_profile") != profile:
            return None

    expected = {
        "schema_version",
        "embodiment_profile_id",
        "action_schema",
        "tactile_profile",
        *_ARTIFACT_DIGEST_ENV,
    }
    if set(payload) != expected or payload.get("schema_version") != 1:
        raise ValueError("AgileX artifact identity has an invalid schema")
    fixed = {
        "embodiment_profile_id": AGILEX_EMBODIMENT_PROFILE_ID,
        "action_schema": AGILEX_ACTION_SCHEMA,
        "tactile_profile": profile,
    }
    if any(payload.get(field) != value for field, value in fixed.items()):
        raise ValueError("AgileX artifact identity differs from the selected profile")
    for field, environment_name in _ARTIFACT_DIGEST_ENV.items():
        value = _digest(payload[field], label=field)
        environment_value = os.environ.get(environment_name)
        if environment_value is not None and value != environment_value:
            raise ValueError(
                f"AgileX artifact identity differs from {environment_name}"
            )
    return dict(payload)


@dataclass(frozen=True)
class AgileXRepoRouteBinding:
    """Validated route contract plus the identity of its source file."""

    manifest: RepoRouteManifestContract
    source_path: str | None
    source_file_sha256: str | None
    is_placeholder: bool

    @property
    def routes(self) -> dict[str, dict[str, object]]:
        return {repo: route.to_json_dict() for repo, route in self.manifest.routes}


def build_repo_route_binding(
    routes: Mapping[str, Mapping[str, object]],
    *,
    tactile_sensor_id_map: Mapping[str, int] | None = None,
    wrench_sensor_id_map: Mapping[str, int] | None = None,
    source_path: Path | None = None,
    source_file_sha256: str | None = None,
    is_placeholder: bool,
) -> AgileXRepoRouteBinding:
    manifest = build_agilex_repo_route_manifest(
        routes,
        tactile_sensor_id_map=tactile_sensor_id_map,
        wrench_sensor_id_map=wrench_sensor_id_map,
    )
    return AgileXRepoRouteBinding(
        manifest=manifest,
        source_path=None if source_path is None else str(source_path),
        source_file_sha256=source_file_sha256,
        is_placeholder=is_placeholder,
    )


def load_repo_route_binding(path: Path) -> AgileXRepoRouteBinding:
    """Load a route file without discarding signed sensor IDs or its hash."""

    source = Path(path).expanduser().resolve(strict=True)
    payload = _read_json_object(source, label="AgileX repo route manifest")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if "contract_sha256" not in payload:
        raw_routes = payload.get("routes", payload)
        if not isinstance(raw_routes, Mapping):
            raise ValueError("AgileX repo routes must be a JSON object")
        if any(not isinstance(route, Mapping) for route in raw_routes.values()):
            raise ValueError("every AgileX repo route must be a JSON object")
        return build_repo_route_binding(
            {
                str(repo): dict(cast(Mapping[str, object], route))
                for repo, route in raw_routes.items()
            },
            source_path=source,
            source_file_sha256=source_sha256,
            is_placeholder=False,
        )

    validated = validate_repo_route_manifest_contract(payload)
    routes = cast(Mapping[str, Mapping[str, object]], validated["routes"])
    tactile_ids = cast(Mapping[str, int], validated["tactile_sensor_id_map"])
    wrench_ids = cast(Mapping[str, int], validated["wrench_sensor_id_map"])
    binding = build_repo_route_binding(
        routes,
        tactile_sensor_id_map=tactile_ids,
        wrench_sensor_id_map=wrench_ids,
        source_path=source,
        source_file_sha256=source_sha256,
        is_placeholder=False,
    )
    if binding.manifest.to_json_dict() != validated:
        raise ValueError("signed AgileX repo route manifest is not canonical")
    return binding


@dataclass(frozen=True)
class AgileXTemporalBinding:
    """One self-hashed fixed-shape temporal schedule for selected repos."""

    contract: dict[str, object]
    action_per_frame: int | None
    action_offsets: tuple[tuple[str, tuple[int, ...]], ...]
    route_identities: tuple[tuple[str, str], ...]
    temporal_identities: tuple[tuple[str, str], ...]
    is_placeholder: bool
    source_file_sha256: str | None

    @property
    def contract_sha256(self) -> str:
        return cast(str, self.contract["contract_sha256"])


def _placeholder_temporal_binding(
    *,
    repo_names: tuple[str, ...],
    repo_route_manifest_sha256: str,
) -> AgileXTemporalBinding:
    bindings: dict[str, dict[str, object]] = {}
    for repo in repo_names:
        route_identity = canonical_sha256(
            {
                "kind": "import_safe_route_placeholder",
                "repo": repo,
                "repo_route_manifest_sha256": repo_route_manifest_sha256,
            }
        )
        temporal_identity = canonical_sha256(
            {
                "kind": "import_safe_temporal_placeholder",
                "repo": repo,
                "repo_route_identity": route_identity,
            }
        )
        bindings[repo] = {
            "action_offsets_per_anchor": [],
            "repo_route_identity": route_identity,
            "temporal_alignment_identity": temporal_identity,
        }
    raw: dict[str, object] = {
        "schema_version": 1,
        "status": "import_safe_placeholder",
        "repo_route_manifest_sha256": repo_route_manifest_sha256,
        "per_repo_bindings": bindings,
    }
    contract = {**raw, "contract_sha256": canonical_sha256(raw)}
    return _materialize_temporal_binding(
        contract,
        repo_names=repo_names,
        repo_route_manifest_sha256=repo_route_manifest_sha256,
        source_file_sha256=None,
        allow_placeholder=True,
    )


def _materialize_temporal_binding(
    payload: Mapping[str, object],
    *,
    repo_names: tuple[str, ...],
    repo_route_manifest_sha256: str,
    source_file_sha256: str | None,
    allow_placeholder: bool,
) -> AgileXTemporalBinding:
    expected_fields = {
        "schema_version",
        "status",
        "repo_route_manifest_sha256",
        "per_repo_bindings",
        "contract_sha256",
    }
    if (
        set(payload) != expected_fields
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise ValueError("AgileX temporal binding has an invalid schema")
    status = payload.get("status")
    placeholder = status == "import_safe_placeholder"
    if status != "complete" and not (allow_placeholder and placeholder):
        raise ValueError("formal AgileX temporal binding must be complete")
    if payload.get("repo_route_manifest_sha256") != repo_route_manifest_sha256:
        raise ValueError("AgileX temporal binding route-manifest mismatch")
    raw = dict(payload)
    digest = _digest(raw.pop("contract_sha256"), label="temporal contract SHA-256")
    if canonical_sha256(raw) != digest:
        raise ValueError("AgileX temporal binding SHA256 mismatch")
    bindings = payload.get("per_repo_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(repo_names):
        raise ValueError("AgileX temporal binding must exactly cover selected repos")

    offsets: list[tuple[str, tuple[int, ...]]] = []
    route_ids: list[tuple[str, str]] = []
    temporal_ids: list[tuple[str, str]] = []
    horizons: set[int] = set()
    for repo in sorted(repo_names):
        binding = bindings[repo]
        fields = {
            "action_offsets_per_anchor",
            "repo_route_identity",
            "temporal_alignment_identity",
        }
        if not isinstance(binding, Mapping) or set(binding) != fields:
            raise ValueError(f"AgileX temporal binding is invalid for {repo!r}")
        raw_offsets = binding["action_offsets_per_anchor"]
        if not isinstance(raw_offsets, list):
            raise ValueError("AgileX action offsets must be a list")
        resolved = tuple(raw_offsets)
        if not placeholder and (
            not resolved
            or any(type(value) is not int or value < 0 for value in resolved)
            or resolved[0] != 0
            or any(right <= left for left, right in zip(resolved, resolved[1:]))
        ):
            raise ValueError("AgileX action offsets must increase from zero")
        if placeholder and resolved:
            raise ValueError("import-safe temporal offsets must remain empty")
        horizons.add(len(resolved))
        offsets.append((repo, cast(tuple[int, ...], resolved)))
        route_ids.append(
            (repo, _digest(binding["repo_route_identity"], label="route identity"))
        )
        temporal_ids.append(
            (
                repo,
                _digest(
                    binding["temporal_alignment_identity"],
                    label="temporal alignment identity",
                ),
            )
        )
    if len(horizons) != 1:
        raise ValueError("all AgileX repos must share one fixed action horizon")
    horizon = next(iter(horizons))
    return AgileXTemporalBinding(
        contract=dict(payload),
        action_per_frame=None if placeholder else horizon,
        action_offsets=tuple(offsets),
        route_identities=tuple(route_ids),
        temporal_identities=tuple(temporal_ids),
        is_placeholder=placeholder,
        source_file_sha256=source_file_sha256,
    )


def load_temporal_binding(
    path: Path | None,
    *,
    repo_names: tuple[str, ...],
    repo_route_manifest_sha256: str,
) -> AgileXTemporalBinding:
    if not repo_names or len(repo_names) != len(set(repo_names)):
        raise ValueError("AgileX temporal binding requires unique selected repos")
    _digest(repo_route_manifest_sha256, label="repo route manifest SHA-256")
    if path is None:
        return _placeholder_temporal_binding(
            repo_names=tuple(sorted(repo_names)),
            repo_route_manifest_sha256=repo_route_manifest_sha256,
        )
    source = Path(path).expanduser().resolve(strict=True)
    payload = _read_json_object(source, label="AgileX temporal binding")
    return _materialize_temporal_binding(
        payload,
        repo_names=tuple(sorted(repo_names)),
        repo_route_manifest_sha256=repo_route_manifest_sha256,
        source_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        allow_placeholder=False,
    )


__all__ = (
    "AgileXRepoRouteBinding",
    "AgileXTemporalBinding",
    "build_repo_route_binding",
    "canonical_sha256",
    "load_repo_route_binding",
    "load_runner_artifact_identity",
    "load_temporal_binding",
)
