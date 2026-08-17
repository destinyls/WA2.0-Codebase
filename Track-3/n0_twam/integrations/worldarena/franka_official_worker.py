# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed WorldArena Hub worker integration for the Franka Policy."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from .franka_policy import (
    FRANKA_CONTROL_ARM,
    Policy,
    load_franka_policy_config,
)

PINNED_WORLD_ARENA_REVISION = "6f5a981b34232fe77812b818a6ad7a4e6b8728ac"
PINNED_ORIGINAL_BRIDGE_SHA256 = (
    "f5d264a1af4ff6b3eb22cd9cfedc9b4ebaff9a1f1d753ff08e60f2428340535e"
)
PINNED_PATCHED_BRIDGE_SHA256 = (
    "4a65011aca4a08093a3024ed1377c76d49c296503aae24a867cc2ba193a4f9d4"
)
BRIDGE_RELATIVE_PATH = Path("real_world_benchmark/worldarena/bridges/legacy_policy.py")
WORKER_RELATIVE_PATH = Path("real_world_benchmark/worldarena/hub_policy_worker.py")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_WORKER_KEY = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_path(path: Path) -> Path:
    lexical = Path(path).expanduser()
    if lexical.is_symlink():
        raise ValueError("WorldArena root must be a non-symlink directory")
    root = lexical.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("WorldArena root must be a directory")
    return root


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must be a non-symlink regular file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _git(root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        },
        text=True,
    )
    return process.stdout.strip()


def _changed_paths(root: Path) -> tuple[str, ...]:
    changed: set[str] = set()
    for arguments in (
        ("diff", "--name-only", "HEAD", "--", "real_world_benchmark"),
        ("diff", "--cached", "--name-only", "--", "real_world_benchmark"),
        (
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "real_world_benchmark",
        ),
    ):
        changed.update(line for line in _git(root, *arguments).splitlines() if line)
    return tuple(sorted(changed))


def _capture_identity(
    worldarena_root: Path,
    *,
    expected_revision: str,
    expected_bridge_sha256: str,
) -> dict[str, object]:
    if not _REVISION.fullmatch(expected_revision):
        raise ValueError("expected WorldArena revision must be 40 lowercase hex")
    if not _SHA256.fullmatch(expected_bridge_sha256):
        raise ValueError("expected bridge SHA256 must be 64 lowercase hex")
    root = _root_path(worldarena_root)
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != root:
        raise ValueError("WorldArena root must be the Git top-level directory")
    revision = _git(root, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise ValueError(
            f"WorldArena revision mismatch: {revision} != {expected_revision}"
        )
    changed = _changed_paths(root)
    allowed = BRIDGE_RELATIVE_PATH.as_posix()
    if any(path != allowed for path in changed):
        raise ValueError(
            "WorldArena checkout has changes outside the audited bridge: "
            + ", ".join(changed)
        )
    bridge = _regular_file(root / BRIDGE_RELATIVE_PATH, label="WorldArena bridge")
    worker = _regular_file(root / WORKER_RELATIVE_PATH, label="WorldArena Hub worker")
    bridge_sha256 = _sha256_file(bridge)
    if bridge_sha256 != expected_bridge_sha256:
        raise ValueError(
            "WorldArena bridge SHA256 mismatch: "
            f"{bridge_sha256} != {expected_bridge_sha256}"
        )
    return {
        "worldarena_root": str(root),
        "worldarena_revision": revision,
        "bridge_path": str(bridge),
        "bridge_sha256": bridge_sha256,
        "hub_worker_sha256": _sha256_file(worker),
        "changed_paths": list(changed),
    }


def _probe_loaded_bridge(bridge: ModuleType, schema: ModuleType) -> dict[str, object]:
    identity_pose = schema.Pose(
        position_m=schema.Vector3(0.1, 0.2, 0.3),
        orientation_xyzw=schema.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        frame="base",
    )
    legacy_pose = np.asarray(
        bridge._arm_end_pose_7d(SimpleNamespace(ee_pose_base=identity_pose)),
        dtype=np.float64,
    )
    expected_legacy = np.asarray((0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    if legacy_pose.shape != (7,) or not np.allclose(legacy_pose, expected_legacy):
        raise ValueError(
            "WorldArena canonical-xyzw to Franka-new_obs-wxyz bridge failed: "
            f"observed={legacy_pose.tolist()}"
        )

    actions = np.asarray(((0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.5),), dtype=np.float32)
    packet = bridge.infer_output_to_action_packet(
        {
            "actions": actions,
            "policy_metadata": {
                "action_format": "end_pose_base",
                "control_arm": FRANKA_CONTROL_ARM,
            },
        },
        context=schema.SessionContext(
            session_id="n0-bridge-audit",
            episode_id="identity",
            task_id="franka",
            task_instruction="hold",
        ),
        observation_timestamp_ns=1,
    )
    arm_action = packet.action_chunk[0].arm_actions[0]
    observed_control_arm = getattr(arm_action, "arm_id", None)
    if observed_control_arm != FRANKA_CONTROL_ARM:
        raise ValueError(
            "WorldArena Franka control-arm routing failed: "
            f"observed={observed_control_arm!r}, expected={FRANKA_CONTROL_ARM!r}"
        )
    quaternion = arm_action.target_pose_base.orientation_xyzw
    canonical_xyzw = np.asarray(
        (quaternion.x, quaternion.y, quaternion.z, quaternion.w), dtype=np.float64
    )
    expected_canonical = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    if not np.allclose(canonical_xyzw, expected_canonical):
        raise ValueError(
            "WorldArena Franka-action-wxyz to canonical-xyzw bridge failed: "
            f"observed={canonical_xyzw.tolist()}"
        )
    return {
        "status": "pass",
        "control_arm": FRANKA_CONTROL_ARM,
        "new_obs_quaternion_order": "wxyz",
        "action_quaternion_order": "wxyz",
        "canonical_quaternion_order": "xyzw",
        "identity_new_obs_pose7": legacy_pose.tolist(),
        "identity_action_packet_xyzw": canonical_xyzw.tolist(),
    }


def _probe_in_isolated_python(root: Path) -> dict[str, object]:
    import_root = Path(__file__).resolve().parents[3]
    source = """
import importlib
import json
import pathlib
import sys

n0_root = pathlib.Path(sys.argv[1]).resolve()
worldarena_root = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str(n0_root))
sys.path.insert(0, str(worldarena_root))
from n0_twam.integrations.worldarena.franka_official_worker import _probe_loaded_bridge
bridge = importlib.import_module(
    'real_world_benchmark.worldarena.bridges.legacy_policy'
)
schema = importlib.import_module('real_world_benchmark.worldarena.schema')
package_root = worldarena_root / 'real_world_benchmark'
for module in (bridge, schema):
    module_path = pathlib.Path(module.__file__).resolve()
    if package_root.resolve() not in module_path.parents:
        raise RuntimeError(
            f'WorldArena module shadowed outside checkout: {module_path}'
        )
print(json.dumps(_probe_loaded_bridge(bridge, schema), sort_keys=True))
"""
    process = subprocess.run(
        (sys.executable, "-I", "-c", source, str(import_root), str(root)),
        check=False,
        capture_output=True,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
        },
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip().splitlines()
        message = detail[-1] if detail else "bridge probe produced no diagnostic"
        raise ValueError(f"WorldArena Franka bridge audit failed: {message}")
    payload = json.loads(process.stdout)
    if not isinstance(payload, dict) or payload.get("status") != "pass":
        raise ValueError("WorldArena Franka bridge probe returned an invalid report")
    return payload


def audit_worldarena_franka_bridge(
    worldarena_root: Path,
    *,
    expected_revision: str,
    expected_bridge_sha256: str,
) -> dict[str, object]:
    """Bind an official checkout and prove both Franka quaternion directions."""
    identity = _capture_identity(
        worldarena_root,
        expected_revision=expected_revision,
        expected_bridge_sha256=expected_bridge_sha256,
    )
    probe = _probe_in_isolated_python(Path(str(identity["worldarena_root"])))
    return {
        "status": "pass",
        "execution_tier": "official_bridge_preflight",
        "organizer_evaluation_completed": False,
        "real_robot_evaluation_completed": False,
        "worldarena_identity": identity,
        "quaternion_probe": probe,
    }


def _validated_hub_url(value: str, *, allow_local_http: bool) -> str:
    parsed = urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Hub URL must not contain credentials, query, or fragment")
    local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not (
        allow_local_http and parsed.scheme == "http" and local
    ):
        raise ValueError("Hub URL must use HTTPS (or explicit localhost HTTP)")
    if not parsed.netloc or not parsed.path.rstrip("/").endswith("/policy"):
        raise ValueError("Hub URL must include an authority and end in /policy")
    return value.rstrip("/")


def _load_official_worker(root: Path) -> Any:
    package_root = (root / "real_world_benchmark").resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name != "real_world_benchmark" and not name.startswith(
            "real_world_benchmark."
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and package_root not in Path(module_file).resolve().parents:
            raise RuntimeError(
                f"shadow WorldArena module already imported: {module_file}"
            )
    sys.path.insert(0, str(root))
    module = importlib.import_module(
        "real_world_benchmark.worldarena.hub_policy_worker"
    )
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError("WorldArena Hub worker has no import source file")
    module_path = Path(module_file).resolve(strict=True)
    if module_path != (root / WORKER_RELATIVE_PATH).resolve(strict=True):
        raise RuntimeError(f"WorldArena Hub worker import was shadowed: {module_path}")
    return module.run_policy_hub_worker


def run_official_franka_worker(
    *,
    worldarena_root: Path,
    expected_revision: str,
    expected_bridge_sha256: str,
    config_path: Path,
    hub_url: str,
    worker_key: str,
    allow_local_http: bool = False,
    dry_run: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Audit inputs, then run the official outbound long-poll worker."""
    audit = audit_worldarena_franka_bridge(
        worldarena_root,
        expected_revision=expected_revision,
        expected_bridge_sha256=expected_bridge_sha256,
    )
    if not _WORKER_KEY.fullmatch(worker_key):
        raise ValueError("worker key contains unsupported characters")
    config = _regular_file(Path(config_path).expanduser(), label="Franka policy config")
    policy_config = load_franka_policy_config(config)
    if policy_config.policy_id != worker_key:
        raise ValueError("policy config policy_id must equal the organizer worker key")
    endpoint = _validated_hub_url(hub_url, allow_local_http=allow_local_http)
    selected_environment = os.environ if environ is None else environ
    token = selected_environment.get("WORLD_ARENA_HUB_TOKEN", "")
    report: dict[str, object] = {
        "status": "ready" if dry_run else "starting",
        "execution_tier": "official_worldarena_hub_worker",
        "organizer_evaluation_completed": False,
        "real_robot_evaluation_completed": False,
        "worldarena_identity": audit["worldarena_identity"],
        "quaternion_probe": audit["quaternion_probe"],
        "policy_id": policy_config.policy_id,
        "config_path": str(config),
        "hub_url": endpoint,
        "hub_token_present": bool(token),
        "connects_to_hub": not dry_run,
    }
    if dry_run:
        return report
    identity = audit.get("worldarena_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("WorldArena audit omitted its source identity")
    root_value = identity.get("worldarena_root")
    if not isinstance(root_value, str):
        raise RuntimeError("WorldArena audit omitted its source root")
    root = Path(root_value)
    worker = _load_official_worker(root)
    policy = Policy(str(config))
    worker(
        policy,
        hub_url=endpoint,
        worker_key=worker_key,
        policy_source="n0_twam.integrations.worldarena.franka_policy.Policy",
        legacy_bridge=True,
        token=token,
    )
    report["status"] = "stopped"
    report["connects_to_hub"] = False
    return report


__all__ = (
    "PINNED_ORIGINAL_BRIDGE_SHA256",
    "PINNED_PATCHED_BRIDGE_SHA256",
    "PINNED_WORLD_ARENA_REVISION",
    "audit_worldarena_franka_bridge",
    "run_official_franka_worker",
)
