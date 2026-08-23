# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed preflight for official Franka Track 3.2 post-training."""

from __future__ import annotations

import ctypes
import json
import os
import socket
import stat
from collections.abc import Mapping
from pathlib import Path

from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
    validate_sha256,
)
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    build_strict_checkpoint_identity,
    capture_strict_checkpoint_snapshot,
)
from n0_twam.actions.loss import build_action_loss_profile
from n0_twam.configs.twam_track32_franka_cfg import twam_track32_franka_cfg
from n0_twam.integrations.worldarena.franka_actions import (
    DERIVED_ACTION_SCHEMA,
    TRACK32_PROFILE_ID,
)
from n0_twam.integrations.worldarena.franka_artifacts import (
    MODEL_ACTION_SCHEMA,
    SOURCE_ACTION_SCHEMA,
    verify_franka_training_artifacts,
)
from n0_twam.integrations.worldarena.franka_manifest import sha256_file

from .completed_init import validate_completed_weights_init


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required environment variable is unset: {name}")
    return value


def _required_path(name: str) -> Path:
    path = Path(_required_env(name)).expanduser().resolve(strict=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{name} may not be a symlink")
    return path


def _required_sha(name: str) -> str:
    return validate_sha256(_required_env(name), label=name)


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _require_hash(path: Path, expected: str, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA256 differs from the request")


def _require_config_contract(config: object) -> None:
    loss_profile = build_action_loss_profile(
        _required_env("N0_TRACK32_ACTION_LOSS_PROFILE"),
        action_dim=20,
        action_horizon=6,
    )
    expected: dict[str, object] = {
        "dataset_adapter": "worldarena_franka_ee10",
        "track32_profile_id": TRACK32_PROFILE_ID,
        "action_schema": MODEL_ACTION_SCHEMA,
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "derived_action_schema": DERIVED_ACTION_SCHEMA,
        "accelerator_profile": _required_env("N0_TRACK32_ACCELERATOR_PROFILE"),
        "fsdp_topology": os.environ.get("N0_FSDP_TOPOLOGY", "global_shard"),
        "fsdp_shard_size": int(
            os.environ.get(
                "N0_FSDP_SHARD_SIZE",
                _required_env("N0_TRACK32_EXPECTED_WORLD_SIZE"),
            )
        ),
        "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
        "nccl_net_gdr_level": os.environ.get("NCCL_NET_GDR_LEVEL"),
        "nccl_dmabuf_enable": os.environ.get("NCCL_DMABUF_ENABLE"),
        "nccl_net_plugin": os.environ.get("NCCL_NET_PLUGIN"),
        "rccl_plugin_sha256": os.environ.get("N0_TRACK32_RCCL_PLUGIN_SHA256"),
        "action_dim": 20,
        "action_per_frame": 6,
        "tactile_profile": "vision_only",
        "tactile_mode": "disabled",
        "use_local_tactile": False,
        "use_contact_gate": False,
        "tactile_diffusion_loss_weight": 0.0,
        "freeze_tactile_parameters": True,
        "strict_training_resume": True,
        "action_loss_profile": loss_profile.name,
        "action_loss_scale": loss_profile.scale,
    }
    mismatches = {
        field: (getattr(config, field, None), wanted)
        for field, wanted in expected.items()
        if getattr(config, field, None) != wanted
    }
    if mismatches:
        raise ValueError(f"Franka config contract mismatch: {mismatches}")
    if list(getattr(config, "used_action_channel_ids", [])) != list(range(10)):
        raise ValueError("Franka active action channels must be exactly 0..9")
    if list(getattr(config, "tactile_keys", [])):
        raise ValueError("Franka vision-only config must declare tactile_keys=[]")
    if tuple(getattr(config, "mot_cross_attn_experts", ())) != (
        "video",
        "action",
    ):
        raise ValueError("Franka cross-attention experts must exclude tactile")
    weighted_fields = {
        "action_channel_loss_weights": list(loss_profile.channel_weights),
        "action_horizon_loss_weights": list(loss_profile.horizon_weights),
    }
    weighted_mismatches = {
        field: (list(getattr(config, field, [])), wanted)
        for field, wanted in weighted_fields.items()
        if list(getattr(config, field, [])) != wanted
    }
    lineage = getattr(config, "training_lineage", {})
    lineage_expected = {
        "action_loss_profile": loss_profile.name,
        "action_loss_scale": loss_profile.scale,
        **weighted_fields,
    }
    lineage_mismatches = {
        field: (lineage.get(field), wanted)
        for field, wanted in lineage_expected.items()
        if not isinstance(lineage, dict) or lineage.get(field) != wanted
    }
    if weighted_mismatches or lineage_mismatches:
        raise ValueError(
            "Franka action loss contract mismatch: "
            f"config={weighted_mismatches}, lineage={lineage_mismatches}"
        )


def _require_recipe(config: object) -> None:
    expected = {
        "run_role": _required_env("N0_TRACK32_RUN_ROLE"),
        "num_steps": int(_required_env("N0_TRACK32_NUM_STEPS")),
        "stop_after_step": int(_required_env("N0_TRACK32_STOP_AFTER_STEP")),
        "save_interval": int(_required_env("N0_TRACK32_SAVE_INTERVAL")),
        "val_interval": int(_required_env("N0_TRACK32_VAL_INTERVAL")),
        "batch_size": int(_required_env("N0_TRACK32_BATCH_SIZE")),
        "gradient_accumulation_steps": int(
            _required_env("N0_TRACK32_GRADIENT_ACCUMULATION_STEPS")
        ),
        "max_latent_frames": int(_required_env("N0_TRACK32_MAX_LATENT_FRAMES")),
        "seed": int(_required_env("N0_TRACK32_SEED")),
    }
    mismatches = {
        field: (getattr(config, field, None), wanted)
        for field, wanted in expected.items()
        if getattr(config, field, None) != wanted
    }
    if mismatches:
        raise ValueError(f"Franka recipe differs from request: {mismatches}")
    expected_world = int(_required_env("N0_TRACK32_EXPECTED_WORLD_SIZE"))
    expected_nodes = int(os.environ.get("N0_TRACK32_EXPECTED_NNODES", "1"))
    expected_local_world = int(
        os.environ.get("N0_TRACK32_EXPECTED_LOCAL_WORLD_SIZE", str(expected_world))
    )
    visible = tuple(
        value for value in _required_env("CUDA_VISIBLE_DEVICES").split(",") if value
    )
    if expected_world <= 0 or expected_nodes <= 0 or expected_local_world <= 0:
        raise ValueError("distributed world dimensions must be positive")
    if expected_nodes * expected_local_world != expected_world:
        raise ValueError("node and local world sizes do not form the expected world")
    if len(visible) != expected_local_world:
        raise ValueError("visible device count differs from expected local world size")


def _require_accelerator_profile() -> None:
    profile = _required_env("N0_TRACK32_ACCELERATOR_PROFILE")
    expected = {
        "hcu_performance": {
            "N0_FLEX_ATTENTION_BACKEND": "grouped_flash_attn",
            "N0_MOT_CROSS_ATTENTION_BACKEND": "flash_attn",
            "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "1",
            "N0_FSDP_REDUCE_DTYPE": "bfloat16",
            "N0_FSDP_EXPERT_RESHARD_POLICY": "after_layer",
            "N0_MOT_ACTIVATION_CHECKPOINTING": "0",
        },
        "portable": {
            "N0_FLEX_ATTENTION_BACKEND": "grouped_sdpa",
            "N0_MOT_CROSS_ATTENTION_BACKEND": "sdpa",
            "N0_FSDP_KEEP_EXPERT_PARAMS_BETWEEN_PRE_POST": "0",
            "N0_FSDP_REDUCE_DTYPE": "float32",
            "N0_FSDP_EXPERT_RESHARD_POLICY": "after_call",
            "N0_MOT_ACTIVATION_CHECKPOINTING": "1",
        },
    }
    if profile not in expected:
        raise ValueError("unsupported Track 3.2 accelerator profile")
    mismatches = {
        name: (os.environ.get(name), value)
        for name, value in expected[profile].items()
        if os.environ.get(name) != value
    }
    if mismatches:
        raise ValueError(f"accelerator profile environment mismatch: {mismatches}")
    network_names = (
        "N0_TRACK32_COLLECTIVE_NETWORK_INTERFACE",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
    )
    if profile == "portable":
        leaked = {
            name: os.environ.get(name) for name in network_names if name in os.environ
        }
        if leaked:
            raise ValueError(
                f"portable profile inherited HCU network bindings: {leaked}"
            )
        return
    interface = _required_env("N0_TRACK32_COLLECTIVE_NETWORK_INTERFACE")
    socket_bindings = {
        "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
        "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME"),
    }
    if any(value != interface for value in socket_bindings.values()):
        raise ValueError(f"collective socket interface mismatch: {socket_bindings}")
    try:
        available = frozenset(name for _, name in socket.if_nameindex())
    except OSError as error:
        raise RuntimeError("unable to enumerate collective interfaces") from error
    if interface not in available:
        raise ValueError(
            f"collective interface is unavailable in this container: {interface}"
        )
    expected_nodes = int(os.environ.get("N0_TRACK32_EXPECTED_NNODES", "1"))
    if expected_nodes > 1:
        _require_multinode_hsdp_and_ib()


def _require_character_device(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"required RDMA device is unavailable: {path}") from error
    if not stat.S_ISCHR(metadata.st_mode):
        raise ValueError(f"required RDMA path is not a character device: {path}")
    if not os.access(path, os.R_OK | os.W_OK):
        raise ValueError(f"required RDMA device is not readable/writable: {path}")


def _require_hca_binding(entry: str, *, uverbs_index: int) -> None:
    hca_name, _, port_text = entry.partition(":")
    port = port_text or "1"
    root = Path("/sys/class/infiniband") / hca_name
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"NCCL HCA sysfs entry is unavailable: {entry}") from error
    uverbs = resolved_root / "device" / "infiniband_verbs" / f"uverbs{uverbs_index}"
    if not uverbs.exists():
        raise ValueError(f"NCCL HCA does not map to uverbs{uverbs_index}: {entry}")
    port_root = resolved_root / "ports" / port
    try:
        state = (port_root / "state").read_text(encoding="utf-8").strip()
        physical_state = (port_root / "phys_state").read_text(
            encoding="utf-8"
        ).strip()
        rate = (port_root / "rate").read_text(encoding="utf-8").strip()
        gid = (port_root / "gids" / "3").read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"unable to audit NCCL HCA port: {entry}") from error
    if "ACTIVE" not in state or "LinkUp" not in physical_state:
        raise ValueError(f"NCCL HCA port is not active/link-up: {entry}")
    if not rate.startswith("400"):
        raise ValueError(f"NCCL HCA port is not operating at 400 Gb/sec: {entry}")
    if not gid or set(gid.replace(":", "")) == {"0"}:
        raise ValueError(f"NCCL HCA GID index 3 is empty: {entry}")


def _require_rccl_network_plugin() -> None:
    plugin_root = Path(_required_env("N0_TRACK32_RCCL_PLUGIN_DIR"))
    if plugin_root.is_symlink() or not plugin_root.is_dir():
        raise ValueError("RCCL network plugin directory is unavailable")
    plugin_name = _required_env("N0_TRACK32_RCCL_PLUGIN_FILENAME")
    if Path(plugin_name).name != plugin_name:
        raise ValueError("RCCL network plugin filename is invalid")
    plugin_path = plugin_root / plugin_name
    _require_hash(
        plugin_path,
        _required_sha("N0_TRACK32_RCCL_PLUGIN_SHA256"),
        label="RCCL network plugin",
    )
    library_roster = _required_env("LD_LIBRARY_PATH").split(":")
    if not library_roster or library_roster[0] != str(plugin_root):
        raise ValueError("RCCL network plugin is not first in LD_LIBRARY_PATH")
    try:
        plugin = ctypes.CDLL(
            str(plugin_path),
            mode=ctypes.RTLD_LOCAL | os.RTLD_NOW,
        )
    except OSError as error:
        raise ValueError(
            "RCCL network plugin cannot be loaded with RTLD_NOW"
        ) from error
    if not hasattr(plugin, "ncclNetPlugin_v8"):
        raise ValueError("RCCL network plugin does not export ncclNetPlugin_v8")


def _require_multinode_hsdp_and_ib() -> None:
    local_world_size = int(_required_env("N0_TRACK32_EXPECTED_LOCAL_WORLD_SIZE"))
    expected = {
        "N0_FSDP_TOPOLOGY": "hsdp",
        "N0_FSDP_SHARD_SIZE": str(local_world_size),
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_IB_QPS_PER_CONNECTION": "4",
        "NCCL_IB_TC": "160",
        "NCCL_IB_TIMEOUT": "22",
        "NCCL_NET_PLUGIN": "shca",
        "N0_TRACK32_RCCL_PLUGIN_HOST_DIR": (
            "/opt/hpc/software/app/rccl/shca_rdma_plugins/v8/lib"
        ),
        "NCCL_NET_GDR_LEVEL": "PHB",
    }
    mismatches = {
        name: (os.environ.get(name), value)
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    if mismatches:
        raise ValueError(f"multinode HSDP/IB environment mismatch: {mismatches}")
    if "NCCL_DMABUF_ENABLE" in os.environ:
        raise ValueError("NCCL_DMABUF_ENABLE must remain unset for PHB peer-memory GDR")
    hca_roster = _required_env("NCCL_IB_HCA").split(",")
    if not hca_roster or len(hca_roster) != len(set(hca_roster)):
        raise ValueError("NCCL_IB_HCA must contain unique HCA ports")
    expected_uverbs = int(_required_env("N0_TRACK32_EXPECTED_IB_UVERBS"))
    if expected_uverbs != len(hca_roster):
        raise ValueError("NCCL HCA roster differs from required uverbs count")
    _require_rccl_network_plugin()
    _require_character_device(Path("/dev/infiniband/rdma_cm"))
    for index in range(expected_uverbs):
        _require_character_device(Path(f"/dev/infiniband/uverbs{index}"))
    for index, entry in enumerate(hca_roster):
        _require_hca_binding(entry, uverbs_index=index)


def _require_initial_checkpoint(
    path: Path,
    expected_sha256: str,
    expected_completion_sha256: str | None,
) -> dict[str, object]:
    if expected_completion_sha256 is not None:
        completed = validate_completed_weights_init(
            path,
            expected_completion_sha256=expected_completion_sha256,
        )
        if completed.transformer_identity["sha256"] != expected_sha256:
            raise ValueError("initial transformer SHA256 differs from the request")
        return {
            "mode": "completed_weights_init",
            "checkpoint_complete_sha256": completed.completion_sha256,
            "transformer_identity": completed.transformer_identity,
            "source_step": completed.step,
            "source_world_size": completed.world_size,
        }
    transformer = path / "transformer"
    weights = transformer / TRANSFORMER_WEIGHTS_FILENAME
    identity = audit_transformer_checkpoint(weights, expected_action_dim=20)
    if identity["sha256"] != expected_sha256:
        raise ValueError("initial transformer SHA256 differs from the request")
    config = _json_object(
        transformer / "config.json", label="initial transformer config"
    )
    if config.get("action_dim") != 20:
        raise ValueError("initial transformer does not retain the released 20D head")
    action_schema = config.get("action_schema")
    if action_schema not in (None, MODEL_ACTION_SCHEMA):
        raise ValueError("initial transformer declares an incompatible action schema")
    return {"mode": "init", "transformer_identity": identity}


def _require_resume_checkpoint(path: Path, expected_identity: str) -> dict[str, object]:
    snapshot = capture_strict_checkpoint_snapshot(path)
    identity = build_strict_checkpoint_identity(snapshot)
    if identity.get("identity_sha256") != expected_identity:
        raise ValueError("resume checkpoint identity differs from the request")
    if snapshot.transformer_config.get("action_schema") != MODEL_ACTION_SCHEMA:
        raise ValueError("resume checkpoint action schema is not ee20_absee")
    meta = snapshot.train_meta
    expected_meta: dict[str, object] = {
        "track32_profile_id": TRACK32_PROFILE_ID,
        "action_schema": MODEL_ACTION_SCHEMA,
        "action_dim": 20,
        "action_per_frame": 6,
        "used_action_channel_ids": list(range(10)),
        "tactile_mode": "disabled",
        "tactile_keys": [],
        "use_local_tactile": False,
        "source_action_schema": SOURCE_ACTION_SCHEMA,
        "derived_action_schema": DERIVED_ACTION_SCHEMA,
        "accelerator_profile": _required_env("N0_TRACK32_ACCELERATOR_PROFILE"),
    }
    mismatches = {
        field: (meta.get(field), wanted)
        for field, wanted in expected_meta.items()
        if meta.get(field) != wanted
    }
    trainability = meta.get("trainability_contract")
    if not isinstance(trainability, Mapping):
        raise ValueError("resume checkpoint lacks a trainability contract")
    if (
        trainability.get("policy") != "freeze_tactile_only_v1"
        or trainability.get("tactile_mode") != "disabled"
        or trainability.get("tactile_profile", "vision_only") != "vision_only"
        or int(trainability.get("frozen_parameter_count", 0)) <= 0
    ):
        raise ValueError("resume checkpoint did not freeze tactile-only parameters")
    if mismatches:
        raise ValueError(f"resume Franka metadata mismatch: {mismatches}")
    return {
        "mode": "resume",
        "checkpoint_identity": identity,
        "start_step": snapshot.step,
    }


def run_preflight() -> dict[str, object]:
    """Audit all request-bound inputs before distributed model loading."""

    artifact_root = _required_path("N0_TRACK32_ARTIFACT_ROOT")
    lerobot_root = _required_path("N0_TRACK32_LEROBOT_ROOT")
    base_model = _required_path("N0_BASE_MODEL")
    empty_embedding = _required_path("N0_EMPTY_EMBEDDING")
    if not empty_embedding.is_file():
        raise ValueError("N0_EMPTY_EMBEDDING must name a regular file")
    _require_hash(
        empty_embedding,
        _required_sha("N0_EMPTY_EMBEDDING_SHA256"),
        label="empty embedding",
    )
    _require_config_contract(twam_track32_franka_cfg)
    _require_recipe(twam_track32_franka_cfg)
    _require_accelerator_profile()

    artifacts = verify_franka_training_artifacts(
        artifact_root=artifact_root,
        lerobot_root=lerobot_root,
        base_model=base_model,
        run_role=_required_env("N0_TRACK32_RUN_ROLE"),
        train_view_id=_required_env("N0_TRACK32_TRAIN_VIEW_ID"),
        normalizer_source_view_id=_required_env(
            "N0_TRACK32_NORMALIZER_SOURCE_VIEW_ID"
        ),
    )
    expected_artifacts = {
        "prepare receipt file": (
            artifacts.prepare_receipt_sha256,
            _required_sha("N0_TRACK32_PREPARE_RECEIPT_SHA256"),
        ),
        "conversion report file": (
            artifacts.conversion_report_sha256,
            _required_sha("N0_TRACK32_CONVERSION_REPORT_SHA256"),
        ),
        "latent inventory file": (
            artifacts.latent_inventory_sha256,
            _required_sha("N0_TRACK32_LATENT_INVENTORY_FILE_SHA256"),
        ),
        "training view": (
            artifacts.train_view.view_sha256,
            _required_sha("N0_TRACK32_TRAIN_VIEW_SHA256"),
        ),
        "normalizer": (
            artifacts.normalizer.get("normalizer_sha256"),
            _required_sha("N0_TRACK32_NORMALIZER_SHA256"),
        ),
        "normalizer source view": (
            artifacts.normalizer_source_view.view_sha256,
            _required_sha("N0_TRACK32_NORMALIZER_SOURCE_VIEW_SHA256"),
        ),
    }
    full_verification_expected = os.environ.get(
        "N0_TRACK32_FULL_VERIFICATION_RECEIPT_SHA256"
    )
    if artifacts.full_verification_receipt_sha256 is None:
        if full_verification_expected:
            raise ValueError("request declares an unexpected full verification receipt")
    elif artifacts.full_verification_receipt_sha256 != validate_sha256(
        full_verification_expected,
        label="N0_TRACK32_FULL_VERIFICATION_RECEIPT_SHA256",
    ):
        raise ValueError("full verification receipt differs from the request")
    validation_expected = os.environ.get("N0_TRACK32_VALIDATION_VIEW_SHA256")
    if artifacts.validation_view is None:
        if validation_expected:
            raise ValueError("final_refit may not declare a validation view")
    else:
        if artifacts.validation_view.view_sha256 != validate_sha256(
            validation_expected,
            label="N0_TRACK32_VALIDATION_VIEW_SHA256",
        ):
            raise ValueError("validation view differs from the request")
    mismatches = [
        label
        for label, (actual, expected) in expected_artifacts.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("Franka artifact identity mismatch: " + ", ".join(mismatches))

    init_raw = os.environ.get("N0_TRACK32_INIT_FROM")
    resume_raw = os.environ.get("N0_TRACK32_RESUME_FROM")
    if bool(init_raw) == bool(resume_raw):
        raise ValueError("exactly one Franka init/resume checkpoint is required")
    if init_raw:
        checkpoint = _require_initial_checkpoint(
            Path(init_raw).expanduser().resolve(strict=True),
            _required_sha("N0_TRACK32_INIT_TRANSFORMER_SHA256"),
            os.environ.get("N0_TRACK32_INIT_CHECKPOINT_COMPLETE_SHA256"),
        )
    else:
        checkpoint = _require_resume_checkpoint(
            Path(str(resume_raw)).expanduser().resolve(strict=True),
            _required_sha("N0_TRACK32_RESUME_CHECKPOINT_IDENTITY_SHA256"),
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "profile": TRACK32_PROFILE_ID,
        "run_role": _required_env("N0_TRACK32_RUN_ROLE"),
        "world_size": int(_required_env("N0_TRACK32_EXPECTED_WORLD_SIZE")),
        "accelerator_profile": _required_env("N0_TRACK32_ACCELERATOR_PROFILE"),
        "fsdp_topology": os.environ.get("N0_FSDP_TOPOLOGY", "global_shard"),
        "fsdp_shard_size": int(
            os.environ.get(
                "N0_FSDP_SHARD_SIZE",
                _required_env("N0_TRACK32_EXPECTED_WORLD_SIZE"),
            )
        ),
        "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
        "nccl_net_gdr_level": os.environ.get("NCCL_NET_GDR_LEVEL"),
        "nccl_dmabuf_enable": os.environ.get("NCCL_DMABUF_ENABLE"),
        "nccl_net_plugin": os.environ.get("NCCL_NET_PLUGIN"),
        "rccl_plugin_sha256": os.environ.get("N0_TRACK32_RCCL_PLUGIN_SHA256"),
        "tactile_profile": "vision_only",
        "tactile_mode": "disabled",
        "action_route": "end_pose_base8_xyzw_to_ee10_to_ee20_mask_0_9",
        "action_loss_profile": twam_track32_franka_cfg.action_loss_profile,
        "action_loss_scale": twam_track32_franka_cfg.action_loss_scale,
        "action_channel_loss_weights": (
            twam_track32_franka_cfg.action_channel_loss_weights
        ),
        "action_horizon_loss_weights": (
            twam_track32_franka_cfg.action_horizon_loss_weights
        ),
        "train_view_sha256": artifacts.train_view.view_sha256,
        "train_view_id": artifacts.train_view.view_id,
        "normalizer_source_view_id": artifacts.normalizer_source_view.view_id,
        "normalizer_source_view_sha256": (
            artifacts.normalizer_source_view.view_sha256
        ),
        "validation_view_sha256": (
            None
            if artifacts.validation_view is None
            else artifacts.validation_view.view_sha256
        ),
        "normalizer_sha256": artifacts.normalizer["normalizer_sha256"],
        "latent_inventory_sha256": artifacts.latent_inventory["inventory_sha256"],
        "full_verification_receipt_sha256": (
            artifacts.full_verification_receipt_sha256
        ),
        "checkpoint": checkpoint,
    }


def main() -> int:
    print(json.dumps(run_preflight(), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
