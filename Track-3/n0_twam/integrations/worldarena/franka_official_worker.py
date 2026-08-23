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
from collections.abc import Callable, Mapping
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
from .franka_live_observation_adapter import (
    FrankaLiveObservationAdapter,
    LiveAdaptedPolicy,
)

PINNED_WORLD_ARENA_REVISION = "6f5a981b34232fe77812b818a6ad7a4e6b8728ac"
PINNED_ORIGINAL_BRIDGE_SHA256 = (
    "f5d264a1af4ff6b3eb22cd9cfedc9b4ebaff9a1f1d753ff08e60f2428340535e"
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
    if changed:
        raise ValueError(
            "WorldArena checkout must keep the verified XYZW bridge unmodified: "
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
    probe_xyzw = np.asarray((0.1, 0.2, 0.3, np.sqrt(0.86)), dtype=np.float64)
    probe_pose = schema.Pose(
        position_m=schema.Vector3(0.1, 0.2, 0.3),
        orientation_xyzw=schema.Quaternion(
            x=float(probe_xyzw[0]),
            y=float(probe_xyzw[1]),
            z=float(probe_xyzw[2]),
            w=float(probe_xyzw[3]),
        ),
        frame="base",
    )
    legacy_pose = np.asarray(
        bridge._arm_end_pose_7d(SimpleNamespace(ee_pose_base=probe_pose)),
        dtype=np.float64,
    )
    expected_legacy = np.concatenate(
        (np.asarray((0.1, 0.2, 0.3), dtype=np.float64), probe_xyzw)
    )
    if legacy_pose.shape != (7,) or not np.allclose(legacy_pose, expected_legacy):
        raise ValueError(
            "WorldArena canonical-xyzw to Franka-new_obs-xyzw bridge failed: "
            f"observed={legacy_pose.tolist()}"
        )

    actions = np.asarray(
        ((0.1, 0.2, 0.3, *probe_xyzw.tolist(), 0.5),), dtype=np.float32
    )
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
    expected_canonical = probe_xyzw
    if not np.allclose(canonical_xyzw, expected_canonical):
        raise ValueError(
            "WorldArena Franka-action-xyzw to canonical-xyzw bridge failed: "
            f"observed={canonical_xyzw.tolist()}"
        )
    return {
        "status": "pass",
        "control_arm": FRANKA_CONTROL_ARM,
        "new_obs_quaternion_order": "xyzw",
        "action_quaternion_order": "xyzw",
        "canonical_quaternion_order": "xyzw",
        "probe_new_obs_pose7": legacy_pose.tolist(),
        "probe_action_packet_xyzw": canonical_xyzw.tolist(),
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
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
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


def _install_live_observation_adapter(
    root: Path,
    adapter: FrankaLiveObservationAdapter,
    *,
    gripper_max_width_m: float,
) -> Callable[[], None]:
    """Wrap verified observation/action bridges without changing official files."""

    bridge = importlib.import_module(
        "real_world_benchmark.worldarena.bridges.legacy_policy"
    )
    package_root = (root / "real_world_benchmark").resolve(strict=True)
    module_file = getattr(bridge, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError("WorldArena live adapter dependency has no source file")
    if package_root not in Path(module_file).resolve(strict=True).parents:
        raise RuntimeError(
            f"WorldArena live adapter import was shadowed: {module_file}"
        )
    original_observation = getattr(bridge, "observation_packet_to_new_obs")
    original_action = getattr(bridge, "infer_output_to_action_packet")
    decode_camera_frames = getattr(bridge, "stack_camera_frames")
    role_to_model_key = {
        str(getattr(bridge, "CAMERA_ROLE_GLOBAL")): "cam_high",
        str(getattr(bridge, "CAMERA_ROLE_LEFT_WRIST")): "cam_left_wrist",
        str(getattr(bridge, "CAMERA_ROLE_RIGHT_WRIST")): "cam_left_wrist",
    }

    def adapted(
        packet: object, *args: Any, **kwargs: Any
    ) -> dict[str, object]:
        new_obs = original_observation(packet, *args, **kwargs)
        return adapter.augment(
            packet,
            new_obs,
            decode_camera_frames=decode_camera_frames,
            role_to_model_key=role_to_model_key,
        )

    def adapted_action(
        output: Mapping[str, object], *args: Any, **kwargs: Any
    ) -> object:
        adapted_output = _physical_width_output_to_open_ratio(
            output,
            gripper_max_width_m=gripper_max_width_m,
        )
        return original_action(adapted_output, *args, **kwargs)

    bridge.observation_packet_to_new_obs = adapted
    bridge.infer_output_to_action_packet = adapted_action

    def restore() -> None:
        bridge.observation_packet_to_new_obs = original_observation
        bridge.infer_output_to_action_packet = original_action

    return restore


def _physical_width_output_to_open_ratio(
    output: Mapping[str, object],
    *,
    gripper_max_width_m: float,
) -> dict[str, object]:
    """Convert model physical width to the canonical wire open-ratio unit."""

    if not np.isfinite(gripper_max_width_m) or gripper_max_width_m <= 0.0:
        raise ValueError("gripper_max_width_m must be positive and finite")
    actions = np.asarray(output.get("actions"), dtype=np.float32)
    if actions.ndim not in (1, 2) or actions.shape[-1] != 8:
        raise ValueError("Franka live action must have shape (8,) or (N, 8)")
    if not np.isfinite(actions).all():
        raise ValueError("Franka live action must be finite")
    widths = actions[..., 7]
    if np.any(widths < 0.0) or np.any(widths > gripper_max_width_m):
        raise ValueError("Franka live gripper width is outside calibrated limits")
    wire_actions = actions.copy()
    wire_actions[..., 7] = widths / gripper_max_width_m
    adapted_output = dict(output)
    adapted_output["actions"] = wire_actions
    metadata_raw = adapted_output.get("policy_metadata")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
    metadata.update(
        {
            "model_gripper_unit": "width_m",
            "wire_gripper_unit": "open_ratio",
            "gripper_max_width_m": gripper_max_width_m,
        }
    )
    adapted_output["policy_metadata"] = metadata
    return adapted_output


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
        "live_observation_adapter": {
            "enabled": policy_config.live_contract is not None,
            "source": "canonical_observation_packet",
            "camera_history": "dual_camera_10hz_4k_plus_1",
            "state_timestamp": "observation_packet.observation_timestamp_ns",
            "continuous_capture_required_during_action_execution": (
                policy_config.live_contract is not None
            ),
            "execution_mode": (
                "future6_then_fresh_replan"
                if policy_config.live_contract is not None
                else "legacy"
            ),
            "requested_action_hz": (
                policy_config.live_contract.action_hz
                if policy_config.live_contract is not None
                else None
            ),
            "actions_per_replan": (
                policy_config.live_contract.actions_per_replan
                if policy_config.live_contract is not None
                else None
            ),
            "future_prediction_range": (
                [
                    policy_config.live_contract.future_start_index,
                    policy_config.live_contract.future_start_index
                    + policy_config.live_contract.actions_per_replan,
                ]
                if policy_config.live_contract is not None
                else None
            ),
            "requires_fresh_observation_after_chunk": (
                policy_config.live_contract.require_fresh_observation_after_chunk
                if policy_config.live_contract is not None
                else False
            ),
            "private_executor_cadence_verified": False,
        },
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
    policy: object = Policy(str(config))
    restore_bridge = None
    if policy_config.live_contract is not None:
        live_adapter = FrankaLiveObservationAdapter(
            policy_config.live_contract,
            gripper_max_width_m=policy_config.safety.gripper_max,
        )
        restore_bridge = _install_live_observation_adapter(
            root,
            live_adapter,
            gripper_max_width_m=policy_config.safety.gripper_max,
        )
        policy = LiveAdaptedPolicy(policy, live_adapter)
    try:
        worker(
            policy,
            hub_url=endpoint,
            worker_key=worker_key,
            policy_source="n0_twam.integrations.worldarena.franka_policy.Policy",
            legacy_bridge=True,
            token=token,
        )
    finally:
        if restore_bridge is not None:
            restore_bridge()
    report["status"] = "stopped"
    report["connects_to_hub"] = False
    return report


__all__ = (
    "PINNED_ORIGINAL_BRIDGE_SHA256",
    "PINNED_WORLD_ARENA_REVISION",
    "audit_worldarena_franka_bridge",
    "run_official_franka_worker",
)
