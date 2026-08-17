# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Canonical tactile-data profiles shared by data, training, and serving."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

from n0_twam.legacy_tactile_checkpoint import (
    validate_legacy_track31_checkpoint_identity,
    validate_legacy_trainability_contract,
)

VISION_TACTILE = "vision_tactile"
MIXED = "mixed"
VISION_ONLY = "vision_only"
TACTILE_PROFILES = (VISION_TACTILE, MIXED, VISION_ONLY)
TACTILE_PROFILE_SCHEMA_VERSION = 1


def _string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of unique non-empty strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result) or len(
        result
    ) != len(set(result)):
        raise ValueError(f"{label} must be a list of unique non-empty strings")
    return result


def _repo_key_map(value: object) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("per_repo_tactile_keys must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for repo, keys in value.items():
        if not isinstance(repo, str) or not repo:
            raise ValueError("per_repo_tactile_keys has an invalid repo name")
        result[repo] = _string_list(keys, label=f"per_repo_tactile_keys[{repo!r}]")
    return dict(sorted(result.items()))


def _resolve_profile(config: object) -> str:
    profile = getattr(config, "tactile_profile", None)
    if isinstance(profile, str) and profile in TACTILE_PROFILES:
        return profile
    if profile is not None:
        raise ValueError(f"tactile_profile must be one of {TACTILE_PROFILES}")
    if bool(getattr(config, "tactile_optional", False)):
        raise ValueError("legacy tactile_optional config has an ambiguous profile")
    mode = getattr(config, "tactile_mode", "enabled")
    global_keys = _string_list(
        getattr(config, "tactile_keys", []), label="tactile_keys"
    )
    repo_map = _repo_key_map(getattr(config, "per_repo_tactile_keys", {}))
    if mode == "disabled" and not global_keys and not any(repo_map.values()):
        return VISION_ONLY
    if (
        mode == "enabled"
        and repo_map
        and any(repo_map.values())
        and any(not keys for keys in repo_map.values())
    ):
        return MIXED
    if (
        mode == "enabled"
        and (global_keys or repo_map)
        and not any(not keys for keys in repo_map.values())
    ):
        return VISION_TACTILE
    raise ValueError("legacy tactile config does not imply one unique profile")


def _probability(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _nonnegative_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class TactileProfileContract:
    """Immutable profile identity persisted in every new strict checkpoint."""

    schema_version: int
    profile: str
    tactile_mode: str
    freeze_tactile_parameters: bool
    tactile_optional: bool
    synthetic_tactile_data: bool
    tactile_cfg_prob: float
    noisy_cond_prob_tactile: float
    tactile_diffusion_loss_weight: float
    global_tactile_keys: tuple[str, ...]
    per_repo_tactile_keys: tuple[tuple[str, tuple[str, ...]], ...]
    tactile_sensor_id_map: tuple[tuple[str, int], ...]
    active_tactile_sensor_count: int
    max_tactile_streams: int
    use_local_tactile: bool
    local_tactile_mode: str
    use_contact_gate: bool
    tactile_global_zero: bool
    contract_sha256: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "tactile_mode": self.tactile_mode,
            "freeze_tactile_parameters": self.freeze_tactile_parameters,
            "tactile_optional": self.tactile_optional,
            "synthetic_tactile_data": self.synthetic_tactile_data,
            "tactile_cfg_prob": self.tactile_cfg_prob,
            "noisy_cond_prob_tactile": self.noisy_cond_prob_tactile,
            "tactile_diffusion_loss_weight": self.tactile_diffusion_loss_weight,
            "global_tactile_keys": list(self.global_tactile_keys),
            "per_repo_tactile_keys": {
                repo: list(keys) for repo, keys in self.per_repo_tactile_keys
            },
            "tactile_sensor_id_map": dict(self.tactile_sensor_id_map),
            "active_tactile_sensor_count": self.active_tactile_sensor_count,
            "max_tactile_streams": self.max_tactile_streams,
            "use_local_tactile": self.use_local_tactile,
            "local_tactile_mode": self.local_tactile_mode,
            "use_contact_gate": self.use_contact_gate,
            "tactile_global_zero": self.tactile_global_zero,
            "contract_sha256": self.contract_sha256,
        }


def _canonical_contract(payload: Mapping[str, object]) -> TactileProfileContract:
    global_keys = cast(Iterable[str], payload["global_tactile_keys"])
    repo_map = cast(Mapping[str, Iterable[str]], payload["per_repo_tactile_keys"])
    sensor_map = cast(Mapping[str, int], payload["tactile_sensor_id_map"])
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return TactileProfileContract(
        schema_version=TACTILE_PROFILE_SCHEMA_VERSION,
        profile=str(payload["profile"]),
        tactile_mode=str(payload["tactile_mode"]),
        freeze_tactile_parameters=bool(payload["freeze_tactile_parameters"]),
        tactile_optional=bool(payload["tactile_optional"]),
        synthetic_tactile_data=bool(payload["synthetic_tactile_data"]),
        tactile_cfg_prob=float(cast(float, payload["tactile_cfg_prob"])),
        noisy_cond_prob_tactile=float(cast(float, payload["noisy_cond_prob_tactile"])),
        tactile_diffusion_loss_weight=float(
            cast(float, payload["tactile_diffusion_loss_weight"])
        ),
        global_tactile_keys=tuple(global_keys),
        per_repo_tactile_keys=tuple(
            (repo, tuple(keys)) for repo, keys in repo_map.items()
        ),
        tactile_sensor_id_map=tuple(sensor_map.items()),
        active_tactile_sensor_count=cast(int, payload["active_tactile_sensor_count"]),
        max_tactile_streams=cast(int, payload["max_tactile_streams"]),
        use_local_tactile=bool(payload["use_local_tactile"]),
        local_tactile_mode=str(payload["local_tactile_mode"]),
        use_contact_gate=bool(payload["use_contact_gate"]),
        tactile_global_zero=bool(payload["tactile_global_zero"]),
        contract_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def validate_tactile_profile_config(
    config: object,
    *,
    repo_names: Iterable[str] | None = None,
) -> TactileProfileContract:
    """Validate a profile and resolve its exact per-repository tactile roster."""

    profile = _resolve_profile(config)
    tactile_mode = getattr(config, "tactile_mode", None)
    freeze = getattr(config, "freeze_tactile_parameters", None)
    optional = getattr(config, "tactile_optional", None)
    synthetic = getattr(config, "synthetic_tactile_data", None)
    if not isinstance(freeze, bool) or not isinstance(optional, bool):
        raise ValueError("tactile freeze/optional switches must be boolean")
    if synthetic is not False:
        raise ValueError("formal tactile profiles reject synthetic tactile data")
    global_keys = _string_list(
        getattr(config, "tactile_keys", None), label="tactile_keys"
    )
    repo_map = _repo_key_map(getattr(config, "per_repo_tactile_keys", {}))
    repositories: tuple[str, ...] = ()
    if repo_names is not None:
        raw_repositories = tuple(repo_names)
        if not raw_repositories or any(
            not isinstance(repo, str) or not repo for repo in raw_repositories
        ):
            raise ValueError("selected repository names must be non-empty strings")
        if len(raw_repositories) != len(set(raw_repositories)):
            raise ValueError("selected repository names must be unique")
        repositories = tuple(sorted(raw_repositories))
    cfg_prob = _probability(
        getattr(config, "tactile_cfg_prob", None), label="tactile_cfg_prob"
    )
    loss_weight = _nonnegative_float(
        getattr(config, "tactile_diffusion_loss_weight", None),
        label="tactile_diffusion_loss_weight",
    )
    noisy_cond_prob = _probability(
        getattr(config, "noisy_cond_prob_tactile", None),
        label="noisy_cond_prob_tactile",
    )
    use_local_tactile = getattr(config, "use_local_tactile", None)
    raw_local_tactile_mode = getattr(config, "local_tactile_mode", None)
    use_contact_gate = getattr(config, "use_contact_gate", None)
    tactile_global_zero = getattr(config, "tactile_global_zero", None)
    if not all(
        isinstance(value, bool)
        for value in (
            use_local_tactile,
            use_contact_gate,
            tactile_global_zero,
        )
    ):
        raise ValueError("tactile conditioning switches must be boolean")
    if use_local_tactile:
        if raw_local_tactile_mode not in {"current", "residual"}:
            raise ValueError("enabled LocalTactile requires current or residual mode")
        local_tactile_mode = str(raw_local_tactile_mode)
    else:
        local_tactile_mode = "disabled"
    sensor_map_raw = getattr(config, "tactile_sensor_id_map", None)
    if not isinstance(sensor_map_raw, Mapping):
        raise ValueError("tactile_sensor_id_map must be a mapping")
    sensor_map = dict(sorted(sensor_map_raw.items()))
    if any(not isinstance(key, str) or not key for key in sensor_map):
        raise ValueError("tactile sensor map keys must be non-empty strings")
    active_sensor_count = getattr(config, "active_tactile_sensor_count", None)
    capacity = getattr(config, "max_tactile_streams", None)
    if (
        isinstance(active_sensor_count, bool)
        or not isinstance(active_sensor_count, int)
        or active_sensor_count < 0
        or isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or capacity <= 0
    ):
        raise ValueError("tactile sensor counts must be valid integers")

    if optional:
        raise ValueError(
            "formal profiles require tactile_optional=False; no-tactile repos "
            "must be declared explicitly"
        )
    if profile == VISION_ONLY:
        if tactile_mode != "disabled" or freeze is not True:
            raise ValueError("vision_only must disable tactile and freeze its modules")
        if global_keys or any(repo_map.values()):
            raise ValueError("vision_only cannot declare active tactile keys")
        if cfg_prob != 0.0 or loss_weight != 0.0:
            raise ValueError("vision_only requires zero tactile drop probability/loss")
        if noisy_cond_prob != 0.0:
            raise ValueError("vision_only requires noisy_cond_prob_tactile=0")
        if sensor_map or active_sensor_count != 0:
            raise ValueError("vision_only cannot activate tactile sensor IDs")
        if use_local_tactile or use_contact_gate:
            raise ValueError("vision_only must disable tactile conditioning branches")
        if tactile_global_zero:
            raise ValueError("vision_only cannot enable a tactile ablation branch")
        if repo_names is not None and set(repo_map) - set(repositories):
            raise ValueError("per_repo_tactile_keys contains an unknown repository")
    else:
        if tactile_mode != "enabled" or freeze is not False:
            raise ValueError(f"{profile} must enable and train tactile modules")
        if loss_weight <= 0.0:
            raise ValueError(f"{profile} requires positive tactile loss weight")
        declared_keys = set(global_keys).union(
            key for keys in repo_map.values() for key in keys
        )
        repo_keys = {key for keys in repo_map.values() for key in keys}
        if not repo_keys.issubset(global_keys):
            raise ValueError(
                "per-repository tactile keys must belong to the global key union"
            )
        if not declared_keys or not declared_keys.issubset(sensor_map):
            raise ValueError("every tactile key must have a sensor ID")
        if set(sensor_map) != declared_keys:
            raise ValueError("tactile sensor map must exactly match declared keys")
        sensor_ids = [sensor_map[key] for key in declared_keys]
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in sensor_ids
            )
            or len(sensor_ids) != len(set(sensor_ids))
            or any(value < 0 or value >= capacity for value in sensor_ids)
        ):
            raise ValueError("tactile sensor IDs must be unique and within capacity")
        if active_sensor_count != len(sensor_map):
            raise ValueError("active tactile sensor count must match the sensor map")
        if tactile_global_zero and not use_local_tactile:
            raise ValueError("enabled profile cannot disable every tactile pathway")

    if profile == VISION_TACTILE:
        if not global_keys and not repo_map:
            raise ValueError("vision_tactile requires tactile keys")
        if any(not keys for keys in repo_map.values()):
            raise ValueError("vision_tactile requires tactile for every repository")
        if cfg_prob != 0.0:
            raise ValueError("vision_tactile requires tactile_cfg_prob=0")
        if repositories:
            missing = [
                repo for repo in repositories if not repo_map.get(repo, global_keys)
            ]
            if missing or set(repo_map) - set(repositories):
                raise ValueError(
                    "vision_tactile repository roster contains missing/unknown entries"
                )
    elif profile == MIXED:
        if not repo_map:
            raise ValueError("mixed requires explicit per-repository tactile keys")
        if repositories and set(repo_map) != set(repositories):
            raise ValueError(
                "mixed per_repo_tactile_keys must exactly cover selected repositories"
            )
        if not any(repo_map.values()) or not any(
            not keys for keys in repo_map.values()
        ):
            raise ValueError("mixed requires both tactile and no-tactile repositories")
        if cfg_prob >= 1.0:
            raise ValueError("mixed requires tactile_cfg_prob < 1")
        routed_keys = {key for keys in repo_map.values() for key in keys}
        if routed_keys != set(global_keys):
            raise ValueError(
                "mixed global tactile keys must exactly equal routed repo keys"
            )

    payload = {
        "schema_version": TACTILE_PROFILE_SCHEMA_VERSION,
        "profile": profile,
        "tactile_mode": tactile_mode,
        "freeze_tactile_parameters": freeze,
        "tactile_optional": optional,
        "synthetic_tactile_data": synthetic,
        "tactile_cfg_prob": cfg_prob,
        "noisy_cond_prob_tactile": noisy_cond_prob,
        "tactile_diffusion_loss_weight": loss_weight,
        "global_tactile_keys": list(global_keys),
        "per_repo_tactile_keys": {repo: list(keys) for repo, keys in repo_map.items()},
        "tactile_sensor_id_map": sensor_map,
        "active_tactile_sensor_count": active_sensor_count,
        "max_tactile_streams": capacity,
        "use_local_tactile": use_local_tactile,
        "local_tactile_mode": local_tactile_mode,
        "use_contact_gate": use_contact_gate,
        "tactile_global_zero": tactile_global_zero,
    }
    return _canonical_contract(payload)


def validate_tactile_profile_contract(payload: object) -> dict[str, object]:
    """Validate the self-hashed JSON form stored in strict checkpoints."""

    if not isinstance(payload, Mapping):
        raise ValueError("tactile profile contract must be an object")
    fields = {
        "schema_version",
        "profile",
        "tactile_mode",
        "freeze_tactile_parameters",
        "tactile_optional",
        "synthetic_tactile_data",
        "tactile_cfg_prob",
        "noisy_cond_prob_tactile",
        "tactile_diffusion_loss_weight",
        "global_tactile_keys",
        "per_repo_tactile_keys",
        "tactile_sensor_id_map",
        "active_tactile_sensor_count",
        "max_tactile_streams",
        "use_local_tactile",
        "local_tactile_mode",
        "use_contact_gate",
        "tactile_global_zero",
        "contract_sha256",
    }
    if set(payload) != fields:
        raise ValueError("tactile profile contract has an invalid field set")
    raw = dict(payload)
    digest = raw.pop("contract_sha256")
    if (
        type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != TACTILE_PROFILE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported tactile profile contract schema")
    expected = hashlib.sha256(
        json.dumps(
            raw, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    if digest != expected:
        raise ValueError("tactile profile contract SHA256 mismatch")
    rebuilt = validate_tactile_profile_config(
        SimpleNamespace(
            tactile_profile=raw["profile"],
            tactile_mode=raw["tactile_mode"],
            freeze_tactile_parameters=raw["freeze_tactile_parameters"],
            tactile_optional=raw["tactile_optional"],
            synthetic_tactile_data=raw["synthetic_tactile_data"],
            tactile_cfg_prob=raw["tactile_cfg_prob"],
            noisy_cond_prob_tactile=raw["noisy_cond_prob_tactile"],
            tactile_diffusion_loss_weight=raw["tactile_diffusion_loss_weight"],
            tactile_keys=raw["global_tactile_keys"],
            per_repo_tactile_keys=raw["per_repo_tactile_keys"],
            tactile_sensor_id_map=raw["tactile_sensor_id_map"],
            active_tactile_sensor_count=raw["active_tactile_sensor_count"],
            max_tactile_streams=raw["max_tactile_streams"],
            use_local_tactile=raw["use_local_tactile"],
            local_tactile_mode=raw["local_tactile_mode"],
            use_contact_gate=raw["use_contact_gate"],
            tactile_global_zero=raw["tactile_global_zero"],
        )
    ).to_json_dict()
    if rebuilt != dict(payload):
        raise ValueError("tactile profile contract is not canonical")
    return rebuilt


def legacy_checkpoint_matches_tactile_profile(
    contract: TactileProfileContract,
    train_meta: Mapping[str, object],
) -> bool:
    """Return whether old sidecars uniquely imply the requested profile.

    Legacy mixed runs never stored their per-repository tactile roster and are
    therefore intentionally not inferable. Track 3.1 tactile checkpoints are
    accepted only when the old artifact/profile/trainability identities are all
    present; generic enabled checkpoints remain weights-only migration inputs.
    """

    trainability = train_meta.get("trainability_contract")
    if contract.profile == VISION_ONLY:
        return (
            train_meta.get("tactile_mode") == "disabled"
            and train_meta.get("tactile_keys") == []
            and validate_legacy_trainability_contract(
                trainability,
                expected_mode="disabled",
                expected_policy="freeze_tactile_only_v1",
            )
        )
    if contract.profile != VISION_TACTILE:
        return False
    if contract.per_repo_tactile_keys:
        return False
    if (
        contract.local_tactile_mode != "current"
        or contract.tactile_global_zero
        or contract.noisy_cond_prob_tactile != 0.0
        or contract.tactile_diffusion_loss_weight != 1.0
        or train_meta.get("local_tactile_mode") != "current"
        or train_meta.get("tactile_global_zero") is not False
    ):
        return False
    return (
        train_meta.get("tactile_mode") == "enabled"
        and validate_legacy_trainability_contract(
            trainability,
            expected_mode="enabled",
            expected_policy="train_all_v1",
        )
        and validate_legacy_track31_checkpoint_identity(
            train_meta,
            expected_tactile_keys=contract.global_tactile_keys,
        )
    )


def validate_profile_segment_inventory(
    *,
    profile: str,
    repo_name: str,
    has_tactile: bool,
    valid_segments: int,
    total_segments: int,
    rejection_counts: Mapping[str, int],
) -> None:
    """Reject filtering that removes a declared repo or tactile segment."""

    if profile not in TACTILE_PROFILES:
        raise ValueError(f"tactile_profile must be one of {TACTILE_PROFILES}")
    if not repo_name:
        raise ValueError("repo_name must be non-empty")
    if total_segments <= 0 or valid_segments <= 0:
        raise ValueError(f"profile-bound repo {repo_name!r} has no valid segments")
    if not has_tactile:
        return
    tactile_rejections = {
        reason: count
        for reason, count in rejection_counts.items()
        if count and "tactile" in reason
    }
    if tactile_rejections:
        raise ValueError(
            f"tactile repo {repo_name!r} has invalid tactile segments: "
            f"{tactile_rejections}"
        )


def validate_serving_tactile_binding(
    payload: object,
    *,
    live_profile: object,
    live_tactile_keys: object,
    live_tactile_mode: object,
    live_max_tactile_streams: object,
    live_use_local_tactile: object,
    live_local_tactile_mode: object,
    live_use_contact_gate: object,
    live_tactile_global_zero: object,
    serve_task: str | None,
    allow_dynamic_signed_routes: bool = False,
) -> tuple[dict[str, object], dict[str, int]]:
    """Validate one server's tactile route against a checkpoint contract."""

    contract = validate_tactile_profile_contract(payload)
    if live_profile != contract["profile"]:
        raise ValueError(
            "serve tactile profile differs from checkpoint: "
            f"train={contract['profile']!r} serve={live_profile!r}"
        )
    if not isinstance(live_use_local_tactile, bool):
        raise ValueError("serve use_local_tactile must be boolean")
    live_semantics = {
        "tactile_mode": live_tactile_mode,
        "max_tactile_streams": live_max_tactile_streams,
        "use_local_tactile": live_use_local_tactile,
        "local_tactile_mode": (
            live_local_tactile_mode if live_use_local_tactile else "disabled"
        ),
        "use_contact_gate": live_use_contact_gate,
        "tactile_global_zero": live_tactile_global_zero,
    }
    for key, live_value in live_semantics.items():
        if live_value != contract[key]:
            raise ValueError(
                f"serve {key} differs from checkpoint: "
                f"train={contract[key]!r} serve={live_value!r}"
            )
    live_keys = _string_list(live_tactile_keys, label="serve tactile_keys")
    global_keys = _string_list(
        contract["global_tactile_keys"], label="checkpoint global_tactile_keys"
    )
    repo_map = _repo_key_map(contract["per_repo_tactile_keys"])
    profile = str(contract["profile"])
    if type(allow_dynamic_signed_routes) is not bool:
        raise ValueError("allow_dynamic_signed_routes must be boolean")
    if serve_task is not None:
        if not isinstance(serve_task, str) or not serve_task:
            raise ValueError("serve_task must be a non-empty string")
        if profile == MIXED and serve_task not in repo_map:
            raise ValueError(
                f"mixed checkpoint has no tactile route for task {serve_task!r}"
            )
        expected_keys = repo_map.get(serve_task, global_keys)
    else:
        if profile == MIXED and not allow_dynamic_signed_routes:
            raise ValueError("mixed checkpoint serving requires an explicit serve_task")
        expected_keys = global_keys
    if live_keys != expected_keys:
        raise ValueError(
            "serve tactile keys differ from checkpoint route: "
            f"train={list(expected_keys)!r} serve={list(live_keys)!r}"
        )
    raw_sensor_map = contract["tactile_sensor_id_map"]
    if not isinstance(raw_sensor_map, Mapping):
        raise ValueError("checkpoint tactile sensor map is invalid")
    sensor_map = {key: int(raw_sensor_map[key]) for key in expected_keys}
    return contract, sensor_map


def resolve_absent_tactile_inference_route(profile: object) -> dict[str, object]:
    """Return the explicit model input marker for a route with no tactile data."""

    if profile == MIXED:
        return {"tactile_cond_drop": True}
    if profile == VISION_ONLY:
        return {"tactile_mode": "disabled"}
    if profile == VISION_TACTILE:
        raise ValueError("vision_tactile serving cannot omit tactile observations")
    raise ValueError(f"tactile_profile must be one of {TACTILE_PROFILES}")


def resolve_repo_tactile_keys(config: object, repo_name: str) -> tuple[str, ...]:
    """Resolve one repo without inferring modality from files or columns."""

    if not isinstance(repo_name, str) or not repo_name:
        raise ValueError("repo_name must be a non-empty string")
    profile = _resolve_profile(config)
    repo_map = _repo_key_map(getattr(config, "per_repo_tactile_keys", {}))
    global_keys = _string_list(
        getattr(config, "tactile_keys", None), label="tactile_keys"
    )
    if profile == MIXED:
        if repo_name not in repo_map:
            raise ValueError(
                f"mixed profile has no tactile declaration for repo {repo_name!r}"
            )
        return repo_map[repo_name]
    if profile == VISION_ONLY:
        return ()
    if profile == VISION_TACTILE:
        keys = repo_map.get(repo_name, global_keys)
        if not keys:
            raise ValueError(
                f"vision_tactile repo {repo_name!r} has no tactile streams"
            )
        return keys
    raise ValueError(f"tactile_profile must be one of {TACTILE_PROFILES}")


__all__ = (
    "MIXED",
    "TACTILE_PROFILES",
    "TactileProfileContract",
    "VISION_ONLY",
    "VISION_TACTILE",
    "legacy_checkpoint_matches_tactile_profile",
    "resolve_absent_tactile_inference_route",
    "resolve_repo_tactile_keys",
    "validate_profile_segment_inventory",
    "validate_serving_tactile_binding",
    "validate_tactile_profile_config",
    "validate_tactile_profile_contract",
)
