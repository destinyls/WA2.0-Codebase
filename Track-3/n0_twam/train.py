# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import argparse
import hashlib
import os
import random
import sys
from collections.abc import Mapping
from pathlib import Path
import wandb

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import TWAM_CONFIGS
from distributed.fsdp import (
    _activation_checkpointing_enabled,
    apply_ac,
    shard_model,
)
from distributed.util import (
    _configure_model,
    dist_mean,
    dist_mean_and_max,
    init_distributed,
)
from einops import rearrange
from models.utils import (
    load_transformer,
    load_mot_transformer,
    load_mot_checkpoint,
)
from n0_twam.models.trainability import configure_parameter_trainability
from n0_twam.tactile_profiles import (
    legacy_checkpoint_matches_tactile_profile,
    validate_tactile_profile_config,
    validate_tactile_profile_contract,
)
from models.model import capture_attention_execution_contract
from utils import (
    init_logger,
    logger,
    get_mesh_id,
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset, BucketedDistributedBatchSampler
from n0_twam.data.exact_eval_sampler import (
    ExactDistributedEvalSampler,
    deterministic_validation_seed,
)
from n0_twam.data.track31_training_identity import (
    build_track31_profile_identity as _build_track31_profile_identity,
    validate_checkpoint_latent_inventory_binding,
    validate_latent_inventory_identity,
)
from n0_twam.actions import (
    build_action_codec_from_config,
    resolve_action_codec_name,
)
from n0_twam.checkpointing.identity import (
    TRANSFORMER_WEIGHTS_FILENAME,
    audit_transformer_checkpoint,
    validate_sha256,
    validate_transformer_identity_match,
)
from n0_twam.checkpointing.runtime_checkpoint_snapshot import (
    RuntimeTransformerSnapshot,
    create_runtime_transformer_snapshot,
    validate_loaded_transformer_architecture,
)
from n0_twam.checkpointing.runtime_provenance import (
    capture_formal_checkpoint_provenance,
    validate_checkpoint_invocation_identity,
    validate_checkpoint_runtime_provenance,
    validate_runtime_source_identity,
)
from n0_twam.checkpointing.strict_resume import (
    build_sidecar_inventory,
    capture_rng_state,
    expected_sidecar_paths,
    restore_scheduler_state,
    save_rng_state,
    save_scheduler_state,
    validate_scheduler_state,
    write_json_atomic,
)
from n0_twam.checkpointing.strict_checkpoint_snapshot import (
    capture_strict_checkpoint_snapshot,
)
from n0_twam.checkpointing.training_lineage import (
    load_validated_action_migration_report,
    validate_stage_a_parent_checkpoint,
)
from n0_twam.configs.twam_track31_univtac_cfg import (
    TRACK31_TRANSFORMER_TACTILE_FIELDS,
    build_track31_invocation_contract,
    build_track31_tactile_training_contract,
    validate_track31_checkpoint_tactile_contract,
)
from n0_twam.integrations.univtac.artifact_contracts import (
    verify_track31_evaluation_bundle,
)
from n0_twam.integrations.univtac.training_startup import (
    verify_track31_training_startup,
)
from n0_twam.integrations.univtac.validation_config import (
    build_validation_dataset_config,
)
from n0_twam.distributed.optimizer_checkpoint import (
    OPTIMIZER_DCP_DIRNAME,
    OPTIMIZER_STATE_FORMAT,
    STRICT_CHECKPOINT_SCHEMA_VERSION,
    build_training_execution_contract,
    capture_runtime_signature,
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
    validate_runtime_signature,
    validate_training_execution_contract,
)
from n0_twam.distributed.attention_schedule import (
    sample_attention_mask_schedule,
)
import gc


_MOT_CONFIG_OVERRIDE_FIELDS = (
    *TRACK31_TRANSFORMER_TACTILE_FIELDS,
    "use_contact_gate",
    "contact_gate_layers",
    "contact_gate_heads",
    "contact_gate_stop_grad",
)
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _read_config_field(config: object, field: str) -> object:
    if isinstance(config, Mapping):
        return config.get(field)
    return getattr(config, field, None)


def _canonical_tactile_field(field: str, value: object) -> object:
    if field == "patch_size" and isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return value


def _build_mot_config_overrides(
    loader_kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Select model-shape overrides used only for init-from migration."""
    return {
        field: loader_kwargs[field]
        for field in _MOT_CONFIG_OVERRIDE_FIELDS
        if field in loader_kwargs
    }


def _validate_track31_transformer_contract(
    transformer: object,
    contract: Mapping[str, object],
    *,
    label: str,
) -> None:
    """Fail closed when the loaded model differs from Track 3.1 config."""
    model_config = getattr(transformer, "config", None)
    for field in TRACK31_TRANSFORMER_TACTILE_FIELDS:
        saved_value = _canonical_tactile_field(
            field, _read_config_field(model_config, field)
        )
        desired_value = _canonical_tactile_field(field, contract[field])
        if saved_value != desired_value:
            raise ValueError(
                f"{label} config mismatch for {field}: "
                f"checkpoint={saved_value!r} config={desired_value!r}"
            )
    attribute_fields = (
        "patch_size",
        "use_local_tactile",
        "max_tactile_streams",
        "tactile_latent_channels",
    )
    for field in attribute_fields:
        runtime_value = _canonical_tactile_field(
            field, getattr(transformer, field, None)
        )
        desired_value = _canonical_tactile_field(field, contract[field])
        if runtime_value != desired_value:
            raise ValueError(
                f"{label} runtime mismatch for {field}: "
                f"model={runtime_value!r} config={desired_value!r}"
            )
    embedding_names = ["sensor_id_embed"]
    if bool(contract["use_local_tactile"]):
        embedding_names.append("local_tactile_sensor_embed")
    expected_capacity = int(contract["max_tactile_streams"])
    for module_name in embedding_names:
        embedding = getattr(transformer, module_name, None)
        actual_capacity = getattr(embedding, "num_embeddings", None)
        if actual_capacity != expected_capacity:
            raise ValueError(
                f"{label} runtime mismatch for {module_name} capacity: "
                f"model={actual_capacity!r} config={expected_capacity!r}"
            )


def _build_track31_transformer_checkpoint_config(
    transformer: object,
    contract: Mapping[str, object],
    *,
    action_schema: str,
    action_dim: int,
) -> dict[str, object]:
    """Create a saved config bound to the validated training contract."""
    if not isinstance(action_schema, str) or not action_schema:
        raise ValueError("Track 3.1 checkpoint action_schema must be non-empty")
    if (
        isinstance(action_dim, bool)
        or not isinstance(action_dim, int)
        or action_dim <= 0
    ):
        raise ValueError("Track 3.1 checkpoint action_dim must be a positive integer")
    _validate_track31_transformer_contract(
        transformer,
        contract,
        label="checkpoint save",
    )
    config_dict = dict(getattr(transformer, "config"))
    config_dict.pop("_name_or_path", None)
    runtime_action_schema = config_dict.get("action_schema")
    if runtime_action_schema != action_schema:
        raise ValueError(
            "checkpoint save action_schema mismatch: "
            f"model={runtime_action_schema!r} codec={action_schema!r}"
        )
    runtime_action_dim = config_dict.get("action_dim")
    if (
        isinstance(runtime_action_dim, bool)
        or not isinstance(runtime_action_dim, int)
        or runtime_action_dim != action_dim
    ):
        raise ValueError(
            "checkpoint save action_dim mismatch: "
            f"model={runtime_action_dim!r} codec={action_dim!r}"
        )
    config_dict["action_schema"] = action_schema
    config_dict["action_dim"] = action_dim
    config_dict["snr_shift"] = float(contract["snr_shift"])
    config_dict["tactile_latent_channels"] = int(
        contract["tactile_latent_channels"]
    )
    return config_dict


def _build_track31_tactile_checkpoint_metadata(
    contract: Mapping[str, object],
) -> dict[str, object]:
    """Persist model capacity separately from the active UniVTAC streams."""
    return {
        "patch_size": list(contract["patch_size"]),
        "snr_shift": float(contract["snr_shift"]),
        "use_local_tactile": bool(contract["use_local_tactile"]),
        "max_tactile_streams": int(contract["max_tactile_streams"]),
        "active_tactile_sensor_count": int(
            contract["active_tactile_sensor_count"]
        ),
        "active_tactile_sensor_ids": list(
            contract["active_tactile_sensor_ids"]
        ),
        "tactile_sensor_id_map": dict(contract["tactile_sensor_id_map"]),
        "tactile_in_channels": int(contract["tactile_in_channels"]),
        "tactile_num_tokens": int(contract["tactile_num_tokens"]),
        "tactile_encoder_dim": int(contract["tactile_encoder_dim"]),
        "tactile_latent_channels": int(contract["tactile_latent_channels"]),
    }


def _set_reproducibility(seed, rank):
    """Seed the Track 3.1 process without changing legacy configs by default."""
    process_seed = int(seed) + int(rank)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return process_seed


class Trainer:
    def __init__(self, config, inference_only=False):
        if config.enable_wandb and config.rank == 0 and not inference_only:
            # self-hosted wandb via WANDB_BASE_URL/WANDB_API_KEY; else standard wandb.ai
            if os.getenv('WANDB_BASE_URL') and os.getenv('WANDB_API_KEY'):
                wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.getenv("WANDB_TEAM_NAME") or None,
                project=os.getenv("WANDB_PROJECT", "twam-pretrain"),
                config=config,
                mode="online",
                name=os.getenv("WANDB_RUN_NAME", "twam-train")
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.tactile_profile_contract = validate_tactile_profile_config(config)
        self.training_lineage = getattr(config, 'training_lineage', None)
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # FlowMatch schedulers — needed by _add_noise / _prepare_input_dict and by
        # inference sampling, so set them up before the (training-only) heavy setup.
        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)
        # symdiff: GlobalTactile becomes a diffusion target. Re-use video snr_shift.
        self.train_scheduler_tactile = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_tactile.set_timesteps(1000, training=True)
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.track31_artifacts = None
        self.track31_tactile_contract = None
        self.runtime_source_identity = None
        self.checkpoint_invocation_identity = None
        if resolve_action_codec_name(config) == 'qpos8_next_step':
            if inference_only and bool(getattr(
                    config, 'raw_tactile_evaluation', False)):
                self.track31_artifacts = verify_track31_evaluation_bundle(
                    manifest_path=Path(config.dataset_manifest_path),
                    normalizer_path=Path(config.norm_stat_path),
                    conversion_report_path=Path(config.conversion_report_path),
                    dataset_root=Path(config.lerobot_root),
                    evaluation_view_path=Path(config.dataset_view_path),
                    normalizer_source_view_path=Path(
                        config.normalizer_source_view_path),
                )
            else:
                self.track31_artifacts = verify_track31_training_startup(
                    config,
                    device=self.device,
                )
                config.source_manifest_sha256 = (
                    self.track31_artifacts.manifest_sha256
                )
                config.normalizer_sha256 = (
                    self.track31_artifacts.normalizer_sha256
                )
                config.train_view_sha256 = (
                    self.track31_artifacts.train_view_sha256
                )
                config.validation_view_sha256 = (
                    self.track31_artifacts.validation_view_sha256
                )
                config.normalizer_source_view_sha256 = (
                    self.track31_artifacts.normalizer_source_view_sha256
                )
                config.video_inventory_sha256 = (
                    self.track31_artifacts.video_inventory_sha256
                )
                config.tactile_inventory_sha256 = (
                    self.track31_artifacts.tactile_inventory_sha256
                )
            config.norm_stat = {
                'q01': list(self.track31_artifacts.action_q01),
                'q99': list(self.track31_artifacts.action_q99),
            }
            self.track31_tactile_contract = (
                build_track31_tactile_training_contract(config)
            )
        (
            self.runtime_source_identity,
            self.checkpoint_invocation_identity,
        ) = capture_formal_checkpoint_provenance(
            formal_track31=(
                (
                    self.track31_tactile_contract is not None
                    or bool(getattr(config, 'capture_runtime_provenance', False))
                )
                and not inference_only
            )
        )
        self.action_codec = build_action_codec_from_config(config)

        if inference_only:
            # Lean setup for rendering / denoising-consistency checks: the caller
            # assigns self.transformer (e.g. via load_mot_checkpoint) and builds
            # any dataset it needs. No FSDP / optimizer / dataloaders / wandb here,
            # so all the _prepare_input_dict / _add_noise / compute_loss machinery
            # is reused EXACTLY (the inference must match training, not re-derive it).
            return

        # Refuse placeholder norm stats (mirror of the server-side guard).
        _ns = getattr(config, 'norm_stat', None) or {}
        _q01 = [float(v) for v in _ns.get('q01', [])]
        _q99 = [float(v) for v in _ns.get('q99', [])]
        if _q01 and _q01 == [-1.0] * len(_q01) and _q99 == [1.0] * len(_q99):
            raise RuntimeError(
                'norm_stat is the [-1, 1] placeholder — the stats file was '
                f'missing when the config was imported '
                f'(norm_stat_path={getattr(config, "norm_stat_path", None)!r}). '
                'Compute the norm stats (script/build_task_pool.py) before training.')
        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        resume_from = getattr(config, 'resume_from', None)
        init_from = getattr(config, 'init_from', None)
        if resume_from and init_from:
            raise ValueError("resume_from and init_from are mutually exclusive")
        strict_training_resume_requested = bool(getattr(
            config, 'strict_training_resume', False))
        if (
            strict_training_resume_requested
            and int(getattr(config, 'load_worker', 0)) != 0
        ):
            raise ValueError(
                "strict_training_resume requires load_worker=0 until DataLoader "
                "worker RNG state is checkpointed")
        if resume_from and not strict_training_resume_requested:
            raise ValueError(
                "resume_from requires strict_training_resume=True; use "
                "init_from for weights-only initialization")
        checkpoint_source = resume_from or init_from
        load_checkpoint_source = checkpoint_source
        checkpoint_compatibility = str(getattr(
            config, 'checkpoint_compatibility', 'strict'))
        if resume_from and checkpoint_compatibility != 'strict':
            raise ValueError("resume_from requires checkpoint_compatibility='strict'")
        released_transformer_sha256 = None
        released_transformer_identity = None
        if checkpoint_source and checkpoint_compatibility == 'migrate_action':
            released_transformer_sha256 = validate_sha256(
                os.environ.get('N0_RELEASED_TRANSFORMER_SHA256'),
                label='N0_RELEASED_TRANSFORMER_SHA256',
            )
            stage_error = None
            if config.rank == 0:
                try:
                    released_transformer_identity = audit_transformer_checkpoint(
                        Path(checkpoint_source)
                        / 'transformer'
                        / TRANSFORMER_WEIGHTS_FILENAME,
                        expected_action_dim=int(getattr(
                            config, 'checkpoint_source_action_dim', 0)),
                    )
                    if (
                        released_transformer_identity['sha256']
                        != released_transformer_sha256
                    ):
                        raise ValueError(
                            "released transformer does not match "
                            "N0_RELEASED_TRANSFORMER_SHA256")
                except Exception as error:
                    stage_error = error
                    logger.exception("Failed to audit released transformer")
            self._raise_if_checkpoint_stage_failed(
                stage_error, "released transformer identity stage")
            if dist.is_initialized():
                identity_payload = [released_transformer_identity]
                dist.broadcast_object_list(
                    identity_payload,
                    src=0,
                    device=self.device,
                )
                released_transformer_identity = identity_payload[0]
            if released_transformer_identity is None:
                raise RuntimeError(
                    "released transformer audit returned no identity")
        stage_a_parent_contract = None
        runtime_parent_snapshot = None
        if (
            init_from
            and checkpoint_compatibility == 'strict'
            and bool(getattr(config, 'inherit_action_migration_report', False))
        ):
            stage_error = None
            if config.rank == 0:
                try:
                    run_role = getattr(config, 'run_role', None)
                    if not isinstance(run_role, str):
                        raise ValueError(
                            "Stage B requires a canonical run role"
                        )
                    if self.track31_artifacts is None:
                        raise ValueError(
                            "Stage B has no verified Track 3.1 artifacts"
                        )
                    stage_a_parent_contract = validate_stage_a_parent_checkpoint(
                        Path(init_from),
                        expected_run_role=run_role,
                        expected_track31_artifacts=(
                            self.track31_artifacts.to_json_dict()
                        ),
                    )
                except Exception as error:
                    stage_error = error
                    logger.exception("Failed to validate Stage A parent checkpoint")
            self._raise_if_checkpoint_stage_failed(
                stage_error, "Stage A parent lineage stage"
            )
            if dist.is_initialized():
                parent_payload = [stage_a_parent_contract]
                dist.broadcast_object_list(parent_payload, src=0, device=self.device)
                stage_a_parent_contract = parent_payload[0]
            if stage_a_parent_contract is None:
                raise RuntimeError("Stage A parent lineage audit returned no contract")
            runtime_lineage = stage_a_parent_contract.get(
                'runtime_training_lineage'
            )
            if not isinstance(runtime_lineage, Mapping):
                raise RuntimeError(
                    "Stage A parent audit returned no runtime lineage"
                )
            self.training_lineage = dict(runtime_lineage)
            config.training_lineage = self.training_lineage
            stage_error = None
            runtime_snapshot_path = None
            if config.rank == 0:
                try:
                    parent_sidecars = stage_a_parent_contract.get(
                        'sidecar_snapshot'
                    )
                    if parent_sidecars is None:
                        raise RuntimeError(
                            "Stage A parent has no stable sidecar snapshot"
                        )
                    runtime_parent_snapshot = create_runtime_transformer_snapshot(
                        checkpoint_root=Path(init_from),
                        run_root=Path(config.save_root),
                        sidecars=parent_sidecars,
                        transformer_identity=stage_a_parent_contract.get(
                            'transformer_identity'
                        ),
                    )
                    runtime_snapshot_path = str(
                        runtime_parent_snapshot.checkpoint_root
                    )
                except Exception as error:
                    stage_error = error
                    logger.exception(
                        "Failed to create private Stage A runtime snapshot"
                    )
            self._raise_if_checkpoint_stage_failed(
                stage_error, "Stage A runtime snapshot stage"
            )
            if dist.is_initialized():
                runtime_snapshot_payload = [runtime_snapshot_path]
                dist.broadcast_object_list(
                    runtime_snapshot_payload,
                    src=0,
                    device=self.device,
                )
                runtime_snapshot_path = runtime_snapshot_payload[0]
            if not isinstance(runtime_snapshot_path, str):
                raise RuntimeError("Stage A runtime snapshot path is invalid")
            load_checkpoint_source = runtime_snapshot_path
        if load_checkpoint_source:
            transformer_path = os.path.join(load_checkpoint_source, 'transformer')
            if config.rank == 0:
                load_kind = "resume" if resume_from else "initialization"
                logger.info(
                    f"Loading checkpoint for {load_kind}: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        _loader_kwargs = dict(
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode="flex",
            target_action_dim=int(getattr(config, 'action_dim', 30)),
            patch_size=tuple(config.patch_size),
            max_tactile_streams=int(config.max_tactile_streams),
            tactile_in_channels=int(config.tactile_in_channels),
            tactile_num_tokens=int(config.tactile_num_tokens),
            tactile_encoder_dim=int(config.tactile_encoder_dim),
            # LocalTactile cross-attn branch. The released pretrain checkpoint has
            # it OFF; the post-train config flips it True (built zero-init on
            # resume). Default False = match the released checkpoint.
            use_local_tactile=bool(getattr(config, 'use_local_tactile', False)),
            # opt-in predictive-contact gate (defaults OFF -> base ckpts unchanged;
            # passed as from_pretrained kwarg so it overrides the ckpt config when
            # resuming a checkpoint whose config predates the gate).
            use_contact_gate=bool(getattr(config, 'use_contact_gate', False)),
            contact_gate_layers=int(getattr(config, 'contact_gate_layers', 2)),
            contact_gate_heads=int(getattr(config, 'contact_gate_heads', 8)),
            contact_gate_stop_grad=bool(getattr(config, 'contact_gate_stop_grad', True)),
        )
        _mot_overrides = {}
        if bool(getattr(config, 'use_mot', False)):
            # Mixture-of-Transformers: 3 per-modality experts warm-started from the
            # shared backbone. Cross-attn / warm-start experts are config-overridable.
            # Built in bf16: an fp32 3-expert build on every rank OOMs the container
            # memory cgroup. (Proper fp32-master via meta-init is a later optimization.)
            _mot_kwargs = dict(_loader_kwargs)
            _mot_kwargs['torch_dtype'] = torch.bfloat16
            if checkpoint_source:
                # Resuming: transformer_path is a SAVED MoT checkpoint (is_mot in its
                # config.json). Load it directly (rebuild saved expert structure +
                # strict weights) — do NOT warm-start from a legacy backbone.
                if config.rank == 0:
                    logger.info(f"Resuming MoT via load_mot_checkpoint: {transformer_path}")
                # Post-training may flip on modules the pretrain ckpt config lacks
                # (use_local_tactile / use_contact_gate): pass them as overrides so
                # the branch is built (zero-init) and its absent-in-ckpt weights are
                # tolerated, instead of rebuilding purely from the local-off ckpt
                # config. Matches the load_mot_transformer (non-resume) path, which
                # already receives these via **_mot_kwargs.
                _mot_overrides = _build_mot_config_overrides(_mot_kwargs)
                stage_error = None
                try:
                    self.transformer = load_mot_checkpoint(
                        transformer_path, torch_dtype=torch.bfloat16,
                        torch_device='cpu', attn_mode='flex',
                        config_overrides=None if resume_from else _mot_overrides,
                        compatibility=checkpoint_compatibility,
                        target_action_dim=int(getattr(config, 'action_dim', 30)),
                        action_init_seed=int(getattr(config, 'action_init_seed', 0)),
                        expected_source_action_dim=getattr(
                            config, 'checkpoint_source_action_dim', None),
                        expected_source_action_schema=getattr(
                            config, 'checkpoint_source_action_schema', None),
                        target_action_schema=self.action_codec.spec.name,
                        adopt_missing_action_schema=bool(getattr(
                            config, 'adopt_missing_action_schema', False)),
                        expected_checkpoint_sha256=getattr(
                            config, 'expected_init_transformer_sha256', None))
                    if resume_from:
                        for field in ('action_dim', 'use_contact_gate'):
                            saved_value = getattr(
                                self.transformer.config, field, None)
                            desired_value = getattr(config, field, None)
                            if saved_value != desired_value:
                                raise ValueError(
                                    f"resume config mismatch for {field}: "
                                    f"checkpoint={saved_value!r} "
                                    f"config={desired_value!r}")
                except Exception as error:
                    stage_error = error
                    logger.exception("Failed to load MoT checkpoint")
                self._raise_if_checkpoint_stage_failed(
                    stage_error, "transformer load stage")
            else:
                if config.rank == 0:
                    logger.info("Building Mixture-of-Transformers (MoT) model (bf16).")
                self.transformer = load_mot_transformer(
                    transformer_path,
                    mot_expert_ffn_dim=getattr(config, 'mot_expert_ffn_dim', None),
                    mot_expert_hidden_dim=getattr(config, 'mot_expert_hidden_dim', None),
                    mot_cross_attn_experts=tuple(getattr(
                        config, 'mot_cross_attn_experts', ("video", "action"))),
                    mot_warmstart_experts=tuple(getattr(
                        config, 'mot_warmstart_experts', ("video", "action", "tactile"))),
                    **_mot_kwargs,
                )
        else:
            if str(getattr(config, 'checkpoint_compatibility', 'strict')) != 'strict':
                raise ValueError(
                    "checkpoint action migration currently requires use_mot=True")
            self.transformer = load_transformer(transformer_path, **_loader_kwargs)

        if stage_a_parent_contract is not None:
            stage_error = None
            try:
                verified_parent_config = stage_a_parent_contract.get(
                    'transformer_config'
                )
                if not isinstance(verified_parent_config, Mapping):
                    raise RuntimeError(
                        "Stage A parent has no verified transformer config"
                    )
                validate_loaded_transformer_architecture(
                    self.transformer.config,
                    verified_config=verified_parent_config,
                    overrides={**_mot_overrides, 'attn_mode': 'flex'},
                )
                if config.rank == 0:
                    post_load_parent_identity = audit_transformer_checkpoint(
                        Path(load_checkpoint_source)
                        / 'transformer'
                        / TRANSFORMER_WEIGHTS_FILENAME,
                        expected_action_dim=8,
                    )
                    validate_transformer_identity_match(
                        stage_a_parent_contract.get('transformer_identity'),
                        post_load_parent_identity,
                        expected_action_dim=8,
                        label='post-load Stage A parent',
                    )
                    revalidated_parent = validate_stage_a_parent_checkpoint(
                        Path(init_from),
                        expected_run_role=str(config.run_role),
                        expected_track31_artifacts=(
                            self.track31_artifacts.to_json_dict()
                        ),
                    )
                    if (
                        revalidated_parent.get('runtime_training_lineage')
                        != stage_a_parent_contract.get(
                            'runtime_training_lineage'
                        )
                    ):
                        raise ValueError(
                            "Stage A parent changed across transformer load"
                        )
            except Exception as error:
                stage_error = error
                logger.exception(
                    "Stage A parent changed while loading transformer"
                )
            self._raise_if_checkpoint_stage_failed(
                stage_error, "post-load Stage A parent identity stage"
            )
            if config.rank == 0:
                if not isinstance(
                    runtime_parent_snapshot,
                    RuntimeTransformerSnapshot,
                ):
                    raise RuntimeError("Stage A runtime snapshot owner was lost")
                runtime_parent_snapshot.cleanup()

        if self.track31_tactile_contract is not None:
            stage_error = None
            try:
                _validate_track31_transformer_contract(
                    self.transformer,
                    self.track31_tactile_contract,
                    label="loaded Track 3.1 transformer",
                )
            except Exception as error:
                stage_error = error
                logger.exception(
                    "Loaded transformer violates Track 3.1 tactile contract"
                )
            self._raise_if_checkpoint_stage_failed(
                stage_error, "Track 3.1 tactile contract stage"
            )

        self.training_execution_contract = build_training_execution_contract(
            max_latent_frames=int(getattr(config, 'max_latent_frames', 0)),
            gradient_accumulation_steps=int(
                self.gradient_accumulation_steps),
            batch_size=int(getattr(config, 'batch_size', 1)),
            load_worker=int(getattr(config, 'load_worker', 0)),
            num_steps=int(getattr(config, 'num_steps', 0)),
            lr_schedule=str(getattr(config, 'lr_schedule', 'constant')),
            warmup_steps=int(getattr(config, 'warmup_steps', 0)),
            lr_min_ratio=float(getattr(config, 'lr_min_ratio', 0.0)),
            activation_checkpointing=_activation_checkpointing_enabled(),
            attention_contract=capture_attention_execution_contract(),
        )
        self.action_migration_report = getattr(
            self.transformer, 'action_migration_report', None)
        if init_from and checkpoint_compatibility == 'migrate_action':
            stage_error = None
            try:
                if self.action_migration_report is None:
                    raise ValueError(
                        "initial action migration did not produce provenance")
                report_sha256 = validate_sha256(
                    self.action_migration_report.get('source_checkpoint_sha256'),
                    label='migration source checkpoint SHA256',
                )
                if report_sha256 != released_transformer_sha256:
                    raise ValueError(
                        "loaded released transformer does not match "
                        "N0_RELEASED_TRANSFORMER_SHA256")
                if released_transformer_identity['sha256'] != report_sha256:
                    raise ValueError(
                        "migration report does not match pre-load "
                        "transformer identity")
                self.action_migration_report = dict(
                    self.action_migration_report)
                self.action_migration_report[
                    'source_transformer_identity'
                ] = released_transformer_identity
            except Exception as error:
                stage_error = error
                logger.exception("Failed to validate action migration provenance")
            self._raise_if_checkpoint_stage_failed(
                stage_error, "action migration provenance stage")
        if stage_a_parent_contract is not None:
            inherited_report = stage_a_parent_contract.get(
                'action_migration_report'
            )
            if not isinstance(inherited_report, dict):
                raise ValueError(
                    "Stage A parent lineage has no action migration report"
                )
            self.action_migration_report = dict(inherited_report)
        self.strict_training_resume = strict_training_resume_requested
        if resume_from and self.strict_training_resume:
            stage_error = None
            try:
                if self.action_codec.spec.name not in {
                    "qpos8_next_step", "ee20_absee"
                }:
                    raise ValueError(
                        "strict_training_resume currently requires "
                        "qpos8_next_step or ee20_absee")
                if self.action_codec.spec.name == "qpos8_next_step":
                    self.action_migration_report = (
                        load_validated_action_migration_report(
                            Path(resume_from),
                            target_action_schema=self.action_codec.spec.name,
                        )
                    )
                elif str(getattr(config, 'dataset_adapter', '')) != (
                    'worldarena_franka_ee10'
                ):
                    raise ValueError(
                        "ee20_absee strict resume is restricted to the Franka adapter"
                    )
            except Exception as error:
                stage_error = error
                logger.exception(
                    "Failed to validate resume action migration provenance")
            self._raise_if_checkpoint_stage_failed(
                stage_error, "resume migration provenance stage")

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        self.trainability_contract = configure_parameter_trainability(
            self.transformer,
            tactile_mode=str(getattr(config, 'tactile_mode', 'enabled')),
            freeze_tactile_parameters=bool(getattr(
                config, 'freeze_tactile_parameters', False)),
            tactile_profile=self.tactile_profile_contract.profile,
        )

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        if str(getattr(config, 'lr_schedule', 'constant')) == 'cosine':
            from utils import warmup_cosine_lambda
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: warmup_cosine_lambda(
                    step, warmup_steps=config.warmup_steps,
                    total_steps=config.num_steps,
                    min_ratio=float(getattr(config, 'lr_min_ratio', 0.1))))
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        self.data_batches_consumed = 0
        self._pending_rng_state = None
        self._pending_sampler_state = None
        if resume_from and self.strict_training_resume:
            self._restore_training_state(Path(resume_from))
        self.invocation_contract = self._validate_invocation_contract()

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        dataset_profile_contract = validate_tactile_profile_config(
            config,
            repo_names=(dataset.repo_name for dataset in train_dataset._datasets),
        )
        if (
            dataset_profile_contract.profile
            != self.tactile_profile_contract.profile
            or dataset_profile_contract.per_repo_tactile_keys
            != self.tactile_profile_contract.per_repo_tactile_keys
        ):
            raise ValueError("dataset tactile profile differs from launch contract")
        _bs = int(getattr(config, 'batch_size', 1))
        sampler_coverage_mode = str(
            getattr(config, 'sampler_coverage_mode', 'legacy')
        )
        if _bs > 1 or sampler_coverage_mode == 'pad_global':
            # Per-GPU batch>1: the pool is shape-heterogeneous (cameras 1/3,
            # tactile streams 0/2/4) so default_collate cannot stack arbitrary
            # samples. Bucket same-signature samples into batches and shard them
            # equally across ranks (each rank gets the same #batches => no FSDP
            # step desync). Requires the dataset to expose per-sample signatures.
            if not hasattr(train_dataset, 'sample_signatures'):
                raise RuntimeError(
                    "batch_size>1 needs a dataset exposing .sample_signatures "
                    "for shape bucketing (only the pi05-delta dataset does)."
                )
            train_batch_sampler = BucketedDistributedBatchSampler(
                train_dataset.sample_signatures,
                batch_size=_bs,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=int(getattr(config, 'seed', 42)),
                drop_last=True,
                coverage_mode=sampler_coverage_mode,
                rank_alignment_mode=str(
                    getattr(config, 'sampler_rank_alignment', 'contiguous')
                ),
                tasks=getattr(train_dataset, 'sample_tasks', None),
                sample_ids=getattr(train_dataset, 'sample_ids', None),
            )
            self.train_sampler = train_batch_sampler
            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                num_workers=config.load_worker,
            )
            if config.rank == 0:
                logger.info(
                    f"Bucketed batch_size={_bs}: {len(train_batch_sampler)} "
                    f"batches/rank across {config.world_size} ranks."
                )
        else:
            # Use an epoch-addressable sampler even on one GPU. RandomSampler's
            # permutation cannot be reconstructed from an epoch plus batch
            # offset, which would make strict checkpoint resume non-deterministic.
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=int(getattr(config, 'seed', 42)),
            )
            self.train_sampler = train_sampler
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.load_worker,
                sampler=train_sampler,
            )

        # Optional validation set: episode-level held-out repos. Loss-only —
        # never touches the optimizer. Sharded across ranks like training.
        self.val_loader = None
        val_path = getattr(config, 'val_dataset_path', None)
        if val_path:
            val_config = build_validation_dataset_config(config)
            val_dataset = MultiLatentLeRobotDataset(config=val_config)
            val_sampler = ExactDistributedEvalSampler(
                val_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
            )
            self.val_sample_validity = val_sampler.validity_mask
            raw_sample_ids = getattr(val_dataset, 'sample_ids', None)
            if raw_sample_ids is not None and len(raw_sample_ids) != len(val_dataset):
                raise ValueError("validation sample identity inventory is incomplete")
            self.val_sample_seeds = tuple(
                deterministic_validation_seed(
                    int(getattr(config, 'seed', 42)),
                    (
                        str(raw_sample_ids[index])
                        if raw_sample_ids is not None
                        else f"dataset-index:{index}"
                    ),
                )
                for index in val_sampler.sample_indices
            )
            self.val_loader = DataLoader(
                val_dataset,
                # Validity is per sample. Batch size one keeps every FSDP rank
                # shape-aligned without mixing a real loss with a padding loss.
                batch_size=1,
                shuffle=False,
                num_workers=2,
                sampler=val_sampler,
            )
            self.val_interval = int(getattr(config, 'val_interval', 100))
            if config.rank == 0:
                logger.info(f"Validation set: {len(val_dataset)} segments, "
                            f"every {self.val_interval} steps")

        if self._pending_sampler_state is not None:
            if not hasattr(self.train_sampler, 'load_state_dict'):
                raise ValueError(
                    "resume checkpoint contains sampler state but the runtime "
                    "sampler cannot restore it"
                )
            self.train_sampler.load_state_dict(self._pending_sampler_state)
            self._pending_sampler_state = None

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader_iter = None

    def _validate_invocation_contract(self) -> dict[str, int | bool]:
        """Validate this process's bounded step range after any strict resume."""
        raw_num_steps = getattr(self.config, 'num_steps', 0)
        raw_steps = (
            self.step,
            getattr(self.config, 'stop_after_step', raw_num_steps),
            raw_num_steps,
        )
        type_flags = tuple(
            int(
                isinstance(value, int)
                and not isinstance(value, bool)
                and _INT64_MIN <= value <= _INT64_MAX
            )
            for value in raw_steps
        )
        normalized_steps = tuple(
            int(value) if is_valid else 0
            for value, is_valid in zip(raw_steps, type_flags, strict=True)
        )

        if not dist.is_initialized():
            if not all(type_flags):
                raise ValueError(
                    "Track 3.1 invocation requires integer optimizer-step "
                    "boundaries"
                )
            return build_track31_invocation_contract(
                start_step=normalized_steps[0],
                stop_after_step=normalized_steps[1],
                num_steps=normalized_steps[2],
            )

        local_payload = torch.tensor(
            [*type_flags, *normalized_steps],
            dtype=torch.int64,
            device=self.device,
        )
        gathered_payloads = [
            torch.empty_like(local_payload)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_payloads, local_payload)
        rank_payloads = [
            tuple(int(value) for value in payload.cpu().tolist())
            for payload in gathered_payloads
        ]

        boundary_errors: list[str] = []
        rank_contracts: list[tuple[int, ...]] = []
        for rank, payload in enumerate(rank_payloads):
            rank_type_flags = payload[:3]
            rank_steps = payload[3:]
            rank_contracts.append(rank_steps)
            if not all(rank_type_flags):
                boundary_errors.append(
                    f"rank {rank}: non-integer or int64-overflow step "
                    "boundary flags="
                    f"{rank_type_flags}"
                )
                continue
            try:
                build_track31_invocation_contract(
                    start_step=rank_steps[0],
                    stop_after_step=rank_steps[1],
                    num_steps=rank_steps[2],
                )
            except ValueError as error:
                boundary_errors.append(f"rank {rank}: {error}")
        if boundary_errors:
            raise ValueError(
                "distributed invocation contract boundary validation failed: "
                + "; ".join(boundary_errors)
            )
        if any(
            rank_contract != rank_contracts[0]
            for rank_contract in rank_contracts[1:]
        ):
            raise ValueError(
                "distributed invocation contract mismatch across ranks: "
                f"{rank_contracts}"
            )
        return build_track31_invocation_contract(
            start_step=normalized_steps[0],
            stop_after_step=normalized_steps[1],
            num_steps=normalized_steps[2],
        )

    def _restore_training_state(self, checkpoint_dir):
        """Restore the optimizer/scheduler/step/RNG state for a strict resume."""
        stage_error = None
        actual_transformer_identity = None
        if self.config.rank == 0:
            try:
                actual_transformer_identity = audit_transformer_checkpoint(
                    checkpoint_dir
                    / 'transformer'
                    / TRANSFORMER_WEIGHTS_FILENAME,
                    expected_action_dim=int(getattr(self.config, 'action_dim', 0)),
                )
            except Exception as error:
                stage_error = error
                logger.exception("Failed to audit strict-resume transformer")
        self._raise_if_checkpoint_stage_failed(
            stage_error, "resume transformer identity stage")
        if dist.is_initialized():
            identity_payload = [actual_transformer_identity]
            dist.broadcast_object_list(
                identity_payload,
                src=0,
                device=self.device,
            )
            actual_transformer_identity = identity_payload[0]
        if actual_transformer_identity is None:
            raise RuntimeError("strict-resume transformer audit returned no identity")

        stage_error = None
        resume_payload = None
        try:
            resume_payload = self._load_and_validate_resume_sidecars(
                checkpoint_dir,
                actual_transformer_identity=actual_transformer_identity,
            )
        except Exception as error:
            stage_error = error
            logger.exception("Failed to validate strict-resume sidecars")
        self._raise_if_checkpoint_stage_failed(
            stage_error, "resume validation stage"
        )
        if resume_payload is None:
            raise RuntimeError("strict-resume validation returned no payload")
        (
            state,
            saved_runtime_signature,
            saved_execution_contract,
            optimizer_inventory_sha256,
            scheduler_state,
            rng_state,
        ) = resume_payload

        validate_runtime_signature(saved_runtime_signature)
        validate_training_execution_contract(
            saved_execution_contract,
            current_contract=self.training_execution_contract,
        )
        load_optimizer_checkpoint(
            self.transformer,
            self.optimizer,
            checkpoint_dir / OPTIMIZER_DCP_DIRNAME,
            expected_inventory_sha256=optimizer_inventory_sha256,
        )
        self._validate_optimizer_scheduler_alignment(scheduler_state)
        gc.collect()
        self.step = int(state["step"])
        self.data_batches_consumed = int(state["data_batches_consumed"])
        self._pending_rng_state = rng_state
        logger.info(
            f"Strict resume restored step={self.step}, "
            f"data_batches_consumed={self.data_batches_consumed}")

    def _load_and_validate_resume_sidecars(
            self, checkpoint_dir, *, actual_transformer_identity):
        """Read every rank-local sidecar before entering a restore collective."""
        optimizer_path = checkpoint_dir / OPTIMIZER_DCP_DIRNAME
        optimizer_metadata_path = optimizer_path / ".metadata"
        if not optimizer_metadata_path.is_file():
            raise FileNotFoundError(
                "resume checkpoint is incomplete; missing "
                f"{optimizer_metadata_path}"
            )
        strict_snapshot = capture_strict_checkpoint_snapshot(checkpoint_dir)
        completion = strict_snapshot.completion
        state = strict_snapshot.training_state
        meta = strict_snapshot.train_meta
        saved_runtime_signature = state.get("runtime_signature")
        saved_execution_contract = state.get("training_execution_contract")
        saved_profile_identity = state.get("training_profile_identity")
        current_profile_identity = _build_track31_profile_identity(self.config)
        saved_track32_artifacts = state.get("track32_artifact_identity")
        current_track32_artifacts = getattr(
            self.config, "track32_artifact_identity", None
        )
        optimizer_inventory_sha256 = state.get("optimizer_inventory_sha256")
        current_tactile_contract = getattr(
            self, "tactile_profile_contract", None
        )
        current_tactile_profile = (
            None
            if current_tactile_contract is None
            else current_tactile_contract.to_json_dict()
        )
        saved_tactile_profile = state.get("tactile_profile_contract")
        sidecar_tactile_profiles = (
            meta.get("tactile_profile_contract"),
            saved_tactile_profile,
            completion.get("tactile_profile_contract"),
        )
        if current_tactile_contract is None:
            if any(value is not None for value in sidecar_tactile_profiles):
                raise ValueError("runtime has no tactile profile contract")
        elif all(value is None for value in sidecar_tactile_profiles):
            if not legacy_checkpoint_matches_tactile_profile(
                current_tactile_contract,
                meta,
            ):
                raise ValueError(
                    "legacy checkpoint has no uniquely inferable tactile profile"
                )
        else:
            canonical_profiles = [
                validate_tactile_profile_contract(value)
                for value in sidecar_tactile_profiles
            ]
            if any(value != current_tactile_profile for value in canonical_profiles):
                raise ValueError("strict resume tactile profile mismatch")
            if (
                strict_snapshot.transformer_config.get("tactile_profile")
                != current_tactile_contract.profile
                or strict_snapshot.transformer_config.get(
                    "tactile_profile_contract_sha256"
                )
                != current_tactile_contract.contract_sha256
            ):
                raise ValueError("strict resume transformer tactile profile mismatch")
        if current_profile_identity is not None or getattr(
            self, "runtime_source_identity", None
        ) is not None:
            validate_checkpoint_runtime_provenance(
                (
                    ("resume train metadata", meta),
                    ("resume training state", state),
                    ("resume completion marker", completion),
                ),
                current_runtime_source_identity=self.runtime_source_identity,
            )
        validate_transformer_identity_match(
            meta.get('transformer_identity'),
            actual_transformer_identity,
            expected_action_dim=int(getattr(self.config, 'action_dim', 0)),
            label='train metadata',
        )
        validate_transformer_identity_match(
            state.get('transformer_identity'),
            actual_transformer_identity,
            expected_action_dim=int(getattr(self.config, 'action_dim', 0)),
            label='training state',
        )
        validate_transformer_identity_match(
            completion.get('transformer_identity'),
            actual_transformer_identity,
            expected_action_dim=int(getattr(self.config, 'action_dim', 0)),
            label='completion marker',
        )
        if (
            completion.get("action_schema") != self.action_codec.spec.name
            or completion.get("optimizer_state_format")
            != OPTIMIZER_STATE_FORMAT
            or completion.get("optimizer_inventory_sha256")
            != optimizer_inventory_sha256
            or completion.get("runtime_signature")
            != saved_runtime_signature
            or completion.get("training_execution_contract")
            != saved_execution_contract
            or completion.get("training_profile_identity")
            != saved_profile_identity
            or completion.get("track32_artifact_identity")
            != saved_track32_artifacts
            or completion.get('transformer_identity')
            != state.get('transformer_identity')
        ):
            raise ValueError("resume checkpoint completion marker is inconsistent")
        if (
            not isinstance(optimizer_inventory_sha256, str)
            or len(optimizer_inventory_sha256) != 64
        ):
            raise ValueError("resume optimizer inventory digest is invalid")
        if strict_snapshot.world_size != int(self.config.world_size):
            raise ValueError(
                "resume world_size mismatch: "
                f"{strict_snapshot.world_size} vs {self.config.world_size}")
        if strict_snapshot.gradient_accumulation_steps != int(
                self.gradient_accumulation_steps):
            raise ValueError("resume gradient_accumulation_steps mismatch")
        if state.get("action_schema") != self.action_codec.spec.name:
            raise ValueError("resume training-state action schema mismatch")
        self._validate_strict_resume_tactile_contract(
            checkpoint_dir,
            train_meta=meta,
            transformer_config=strict_snapshot.transformer_config,
        )
        if meta.get("action_schema") != self.action_codec.spec.name:
            raise ValueError("resume train metadata action schema mismatch")
        if meta.get("training_execution_contract") != saved_execution_contract:
            raise ValueError(
                "resume train metadata execution contract is inconsistent")
        if saved_execution_contract != self.training_execution_contract:
            raise ValueError("resume training execution contract mismatch")
        if current_profile_identity is not None:
            if (
                saved_profile_identity != current_profile_identity
                or meta.get("training_profile_identity")
                != current_profile_identity
            ):
                raise ValueError("resume Track 3.1 training profile identity mismatch")
        if current_track32_artifacts is not None:
            if (
                saved_track32_artifacts != current_track32_artifacts
                or meta.get("track32_artifact_identity")
                != current_track32_artifacts
            ):
                raise ValueError("resume Track 3.2 artifact identity mismatch")
        saved_sampler_state = meta.get("sampler_state")
        if getattr(self.config, "sampler_coverage_mode", None) == "pad_global":
            if not isinstance(saved_sampler_state, dict):
                raise ValueError("resume checkpoint is missing pad_global sampler state")
            self._pending_sampler_state = saved_sampler_state
        saved_artifacts = meta.get("track31_artifacts")
        current_artifacts = (
            None
            if self.track31_artifacts is None
            else self.track31_artifacts.to_json_dict()
        )
        identity_keys = (
            "manifest_sha256",
            "normalizer_sha256",
            "conversion_report_sha256",
            "video_inventory_sha256",
            "tactile_inventory_sha256",
        )
        if current_profile_identity is not None:
            if (
                not isinstance(saved_artifacts, dict)
                or not isinstance(current_artifacts, dict)
                or any(
                    saved_artifacts.get(key) != current_artifacts.get(key)
                    for key in identity_keys
                )
            ):
                raise ValueError("resume UniVTAC artifact identity mismatch")
        if current_profile_identity is not None:
            validate_checkpoint_latent_inventory_binding(
                expected=current_artifacts,
                payloads=(
                    ("resume train metadata", meta),
                    ("resume training state", state),
                    ("resume completion marker", completion),
                ),
            )
        scheduler_state = validate_scheduler_state(
            strict_snapshot.sidecars.json_object(
                "scheduler_state.json",
                label="resume scheduler state",
            ),
            completed_steps=strict_snapshot.step,
            learning_rate=float(self.config.learning_rate),
            execution_contract=saved_execution_contract,
        )
        restore_scheduler_state(
            self.lr_scheduler,
            scheduler_state,
            completed_steps=strict_snapshot.step,
            learning_rate=float(self.config.learning_rate),
            execution_contract=saved_execution_contract,
        )
        rng_state = strict_snapshot.sidecars.load_rng_state(
            rank=int(self.config.rank),
            expected_world_size=int(self.config.world_size),
        )
        return (
            state,
            saved_runtime_signature,
            saved_execution_contract,
            optimizer_inventory_sha256,
            scheduler_state,
            rng_state,
        )

    def _validate_strict_resume_tactile_contract(
        self,
        checkpoint_dir: Path,
        *,
        train_meta: Mapping[str, object],
        transformer_config: Mapping[str, object] | None = None,
    ) -> None:
        """Bind direct strict resume to saved and runtime tactile semantics."""
        contract = getattr(self, "track31_tactile_contract", None)
        if contract is None:
            return
        if transformer_config is None:
            transformer_config_path = checkpoint_dir / "transformer" / "config.json"
            transformer_config = json.loads(
                transformer_config_path.read_text(encoding="utf-8")
            )
        if not isinstance(transformer_config, dict):
            raise ValueError("resume transformer config must be a JSON object")
        validate_track31_checkpoint_tactile_contract(
            transformer_config=transformer_config,
            train_meta=train_meta,
            expected_contract=contract,
            released_prior=False,
        )
        _validate_track31_transformer_contract(
            self.transformer,
            contract,
            label="strict-resume Track 3.1 transformer",
        )
        trainer_patch_size = _canonical_tactile_field(
            "patch_size", getattr(self, "patch_size", None)
        )
        if trainer_patch_size != contract["patch_size"]:
            raise ValueError(
                "strict-resume trainer patch_size differs from tactile contract"
            )
        for scheduler_name in (
            "train_scheduler_latent",
            "train_scheduler_tactile",
        ):
            scheduler = getattr(self, scheduler_name, None)
            scheduler_shift = getattr(scheduler, "shift", None)
            if scheduler_shift != contract["snr_shift"]:
                raise ValueError(
                    f"strict-resume {scheduler_name} shift differs from "
                    "tactile contract"
                )

    def _validate_optimizer_scheduler_alignment(self, scheduler_state):
        """Reject a DCP optimizer whose LR fields disagree with the scheduler."""
        base_lrs = [
            float.fromhex(value) for value in scheduler_state["base_lrs_hex"]
        ]
        last_lrs = [
            float.fromhex(value) for value in scheduler_state["last_lrs_hex"]
        ]
        groups = self.optimizer.param_groups
        if len(groups) != len(base_lrs):
            raise ValueError("restored optimizer LR group count mismatch")
        for group, base_lr, last_lr in zip(
            groups, base_lrs, last_lrs, strict=True
        ):
            if (
                float(group.get("lr", float("nan"))).hex() != last_lr.hex()
                or float(group.get("initial_lr", float("nan"))).hex()
                != base_lr.hex()
            ):
                raise ValueError(
                    "restored optimizer LR fields disagree with scheduler state"
                )

    def _restore_pending_rng_state(self):
        if self._pending_rng_state is None:
            return
        state = self._pending_rng_state
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["cuda"])
        self._pending_rng_state = None

    def _new_train_loader_iter(self):
        batches_per_epoch = len(self.train_loader)
        if batches_per_epoch <= 0:
            raise ValueError("training dataloader is empty")
        epoch, offset = divmod(self.data_batches_consumed, batches_per_epoch)
        for sampler in (getattr(self.train_loader, 'batch_sampler', None),
                        self.train_loader.sampler):
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
                break
        if hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch)
        # DataLoader iterator construction consumes the global Torch RNG even
        # with num_workers=0. At an epoch boundary the uninterrupted path
        # creates the next iterator *after* the checkpoint RNG was captured, so
        # restore that state before iter() to preserve the same consumption.
        # Mid-epoch resumes must instead replay skipped batches first, then
        # restore the checkpoint state so their stochastic crops have no effect.
        if offset == 0:
            self._restore_pending_rng_state()
        iterator = iter(self.train_loader)
        for _ in range(offset):
            next(iterator)
        if offset > 0:
            self._restore_pending_rng_state()
        return iterator

    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = self._new_train_loader_iter()

        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            self._write_completed_epoch_exposure()
            # Reset sampler/batch_sampler and iterator when epoch finishes.
            # batch_size>1 uses a bucketed batch_sampler; batch_size==1 uses a
            # plain (Distributed)Sampler. Both expose set_epoch for reshuffle.
            for _samp in (getattr(self.train_loader, 'batch_sampler', None),
                          self.train_loader.sampler):
                if _samp is not None and hasattr(_samp, 'set_epoch'):
                    next_epoch = getattr(_samp, 'epoch', 0) + 1
                    _samp.set_epoch(next_epoch)
                    if hasattr(self.train_loader.dataset, 'set_epoch'):
                        self.train_loader.dataset.set_epoch(next_epoch)
                    break
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        self.data_batches_consumed += 1
        return batch

    def _write_completed_epoch_exposure(self) -> None:
        """Append one verified global sampler plan after an epoch completes."""
        sampler = getattr(self, 'train_sampler', None)
        config = getattr(self, 'config', None)
        if (
            config is None
            or getattr(config, 'rank', 0) != 0
            or sampler is None
            or not hasattr(sampler, 'exposure_summary')
        ):
            return
        payload = dict(sampler.exposure_summary())
        crop_policy = {
            'policy_id': getattr(
                self.config,
                'crop_window_policy_id',
                'checkpointed_rng_stream_crop_v1',
            ),
            'max_latent_frames': int(
                getattr(self.config, 'max_latent_frames', 0)
            ),
        }
        payload['crop_window_policy'] = crop_policy
        payload['crop_window_policy_sha256'] = hashlib.sha256(
            json.dumps(
                crop_policy,
                allow_nan=False,
                separators=(',', ':'),
                sort_keys=True,
            ).encode('utf-8')
        ).hexdigest()
        report_path = self.save_dir.parent / 'exposure_report.jsonl'
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(',', ':'),
            sort_keys=True,
        )
        if report_path.is_file():
            lines = [
                line for line in report_path.read_text(encoding='utf-8').splitlines()
                if line
            ]
            if lines:
                previous = json.loads(lines[-1])
                previous_epoch = previous.get('epoch')
                if previous_epoch == payload['epoch']:
                    if previous != payload:
                        raise ValueError(
                            'existing sampler exposure epoch differs from runtime'
                        )
                    return
                if isinstance(previous_epoch, int) and previous_epoch > payload['epoch']:
                    raise ValueError('sampler exposure report epoch moved backwards')
        with report_path.open('a', encoding='utf-8') as handle:
            handle.write(encoded + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def _persist_completed_epoch_exposure_at_checkpoint(self) -> None:
        """Account a fully consumed epoch before its iterator raises StopIteration."""
        config = getattr(self, "config", None)
        if (
            config is None
            or getattr(config, "sampler_coverage_mode", None) != "pad_global"
        ):
            return
        sampler = getattr(self, "train_sampler", None)
        if sampler is None or not hasattr(sampler, "exposure_summary"):
            raise ValueError("pad_global checkpoint requires an exposure-aware sampler")
        batches_per_epoch = len(self.train_loader)
        if batches_per_epoch <= 0:
            raise ValueError("training dataloader is empty")
        batches_consumed = getattr(self, "data_batches_consumed", None)
        if (
            isinstance(batches_consumed, bool)
            or not isinstance(batches_consumed, int)
            or batches_consumed < 0
        ):
            raise ValueError("data_batches_consumed must be a non-negative integer")
        completed_epoch_count, offset = divmod(
            batches_consumed,
            batches_per_epoch,
        )
        if completed_epoch_count == 0 or offset != 0:
            return
        expected_epoch = completed_epoch_count - 1
        sampler_epoch = getattr(sampler, "epoch", None)
        if sampler_epoch != expected_epoch:
            raise ValueError(
                "completed sampler epoch does not match consumed batches: "
                f"{sampler_epoch!r} vs {expected_epoch}"
            )
        self._write_completed_epoch_exposure()

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5,
                    max_timestep_bd=1.0,
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _sample_attention_mask_schedule(self) -> tuple[int, int]:
        """Sample one chunk/window pair, broadcasting rank zero when enabled."""
        return sample_attention_mask_schedule(self.device)

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from n0_twam_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'],
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=0.5)

        action_dict = self._add_noise(
            latent=batch_dict['actions'],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch_dict['actions_mask'],
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']
        action_dict['tactile_mode'] = str(
            getattr(self.config, 'tactile_mode', 'enabled')
        )
        # Tactile inputs (shared CFG-drop mechanism):
        #   - tactile_cond_drop is explicit CFG dropout; when set, no tactile
        #     tensors are passed to the model so tactile modules get no grad
        #     (the model's tactile_zero_anchor keeps their params in the graph).
        #   - LocalTactile + sensor_ids are a clean cross-attention condition.
        # The CFG-drop decision is made ONCE in convert_input_format (which runs
        # immediately before this in both _train_step and _validate) and stashed
        # on self, so device movement there and the target/forward path here use
        # the SAME flag — no independent re-draw that could disagree (the old
        # double-decision built a target from a CPU latent -> device crash).
        tactile_cond_drop = getattr(self, '_tactile_cond_drop', False)
        action_dict['tactile_cond_drop'] = torch.tensor(
            tactile_cond_drop, dtype=torch.bool, device=self.device)
        if not tactile_cond_drop:
            for key in ('tactile_local_latent', 'tactile_sensor_ids'):
                if key in batch_dict:
                    action_dict[key] = batch_dict[key]
            # symdiff: noise the GlobalTactile latent so the dedicated
            # tactile_proj_out head can predict the velocity. Treat each sensor as
            # an independent diffusion sample by folding S into the batch dim.
            # GlobalTactile is a diffusion target here, not a clean condition, so
            # the raw tactile_global_latent is consumed via the noised tensors below.
            if 'tactile_global_latent' in batch_dict:
                g = batch_dict['tactile_global_latent']                # (B, S, C, F, H, W)
                if getattr(self.config, 'tactile_global_zero', False):
                    # global-ablation: zero the GlobalTactile latent so the noisy
                    # segment (= pure noise) and clean condition carry no contact
                    # info; sequence/loss structure stays identical to full model.
                    if not getattr(self, '_tgz_logged', False):
                        self._tgz_logged = True
                        logger.info(
                            "tactile_global_zero ACTIVE: zeroing global latent "
                            "(incoming max=%.4f)", g.abs().max().item())
                    g = torch.zeros_like(g)
                B, S, C, F, H, W = g.shape
                g_flat = g.reshape(B * S, C, F, H, W).contiguous()
                tdict = self._add_noise(
                    latent=g_flat,
                    train_scheduler=self.train_scheduler_tactile,
                    action_mask=None,
                    action_mode=False,
                    noisy_cond_prob=getattr(self.config, 'noisy_cond_prob_tactile', 0.5),
                )
                action_dict['tactile_global_noisy_latent'] = (
                    tdict['noisy_latents'].reshape(B, S, C, F, H, W))
                action_dict['tactile_global_clean_latent'] = (
                    tdict['latent'].reshape(B, S, C, F, H, W))
                action_dict['tactile_global_targets'] = (
                    tdict['targets'].reshape(B, S, C, F, H, W))
                # timesteps from _add_noise are (B*S, F); collapse the S dim by
                # taking the first sensor's schedule (per-frame timestep is shared).
                action_dict['tactile_global_timesteps'] = (
                    tdict['timesteps'].reshape(B, S, F)[:, 0])         # (B, F)
                action_dict['tactile_global_cond_timesteps'] = (
                    tdict['cond_timesteps'].reshape(B, S, F)[:, 0])    # (B, F)

        chunk_size, window_size = self._sample_attention_mask_schedule()
        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': chunk_size,
            'window_size': window_size,
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        # Decide the tactile CFG-drop ONCE here, BEFORE moving tensors, and stash
        # it on self so _prepare_input_dict reuses the SAME flag. Otherwise the
        # drop used HERE (for device movement) and one re-drawn in _prepare can
        # disagree: a "drop" here skips moving tactile_global_latent to GPU, then
        # a "keep" in _prepare builds the tactile target from that CPU tensor ->
        # mse_loss device-mismatch crash. Batch-level (one flag/batch); shape
        # bucketing keeps tactile presence uniform within a batch, so one draw is
        # correct and avoids per-sample OR-ing that over-drops at batch>1.
        _has_tactile = (('tactile_global_latent' in input_dict)
                        or ('tactile_local_latent' in input_dict))
        if not _has_tactile:
            tactile_cond_drop = True
        else:
            _cfg_p = float(getattr(self.config, 'tactile_cfg_prob', 0.1))
            tactile_cond_drop = bool(torch.rand(1).item() < _cfg_p)
        self._tactile_cond_drop = tactile_cond_drop
        for key, value in input_dict.items():
            if tactile_cond_drop and key in (
                'tactile_global_latent',
                'tactile_local_latent',
            ):
                continue
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def _compute_latent_loss(self, input_dict, latent_pred):
        latent_target = input_dict['latent_dict']['targets']
        # Transformer video output is a patch sequence:
        #   latent_pred_seq: [B, N_video_tokens, C_patch]
        # Convert it back to dense FlowMatch target layout:
        #   latent_pred/latent_target: [B, C_latent, F_video, H_latent, W_latent]
        latent_pred = data_seq_to_patch(
            self.patch_size, latent_pred,
            latent_target.shape[-3],
            latent_target.shape[-2],
            latent_target.shape[-1],
            batch_size=latent_pred.shape[0])
        # timesteps: [B, F_video], one sampled diffusion/flow timestep per frame.
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        # latent_loss_weight: [B, F_video], broadcast over C/H/W below.
        latent_loss_weight = self.train_scheduler_latent.training_weight(
            input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Per-element MSE:
        #   latent_loss: [B, C_latent, F_video, H_latent, W_latent]
        latent_loss = F.mse_loss(
            latent_pred.float(),
            latent_target.float().detach(),
            reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Move frame next to batch and flatten all non-frame dimensions:
        #   [B, C, F, H, W] -> [B, F, H, W, C] -> [B*F, H*W*C]
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)
        # Normalize each frame by its element count, then average over B*F.
        latent_loss_per_frame = latent_loss.sum(dim=1)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)
        return (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

    def _compute_action_loss(self, input_dict, action_pred):
        action_target = input_dict['action_dict']['targets']
        action_mask = input_dict['action_dict']['actions_mask'].float()
        # Transformer action output is a token sequence:
        #   action_pred_seq: [B, F_action * N_action, C_action]
        # Dense action target/mask layout is:
        #   action_target/action_mask: [B, C_action, F_action, N_action, 1]
        action_pred = rearrange(
            action_pred,
            'b (f n) c -> b c f n 1',
            f=action_target.shape[-3])
        # timesteps: [B, F_action], one sampled diffusion/flow timestep per frame.
        Bn, Fn = input_dict['action_dict']['timesteps'].shape
        # action_loss_weight: [B, F_action], broadcast over C/N below.
        action_loss_weight = self.train_scheduler_action.training_weight(
            input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Per-element MSE:
        #   action_loss: [B, C_action, F_action, N_action, 1]
        action_loss = F.mse_loss(
            action_pred.float(),
            action_target.float().detach(),
            reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * action_mask
        # Move frame next to batch and flatten action channels/horizon:
        #   [B, C, F, N, 1] -> [B, F, N, 1, C] -> [B*F, N*C]
        action_loss = action_loss.permute(0, 2, 3, 4, 1)
        action_mask = action_mask.permute(0, 2, 3, 4, 1)
        action_loss = action_loss.flatten(0, 1).flatten(1)
        action_mask = action_mask.flatten(0, 1).flatten(1)
        # Normalize by valid action elements per frame. Frames with no valid
        # action tokens, such as pi05's first condition frame, do not contribute.
        action_loss_per_frame = action_loss.sum(dim=1)
        action_mask_per_frame = action_mask.sum(dim=1)
        valid_frame = action_mask_per_frame > 0
        if not valid_frame.any():
            return action_loss_per_frame.sum() * 0.0
        return (action_loss_per_frame[valid_frame] / action_mask_per_frame[valid_frame]).mean()

    def compute_loss(self,
        input_dict,
        pred
    ):
        # Symdiff: pred is 3-tuple (video_pred, action_pred, tactile_pred).
        # Legacy (video+action): pred is 2-tuple (video_pred, action_pred). Handle both.
        #   latent_pred: [B, N_video_tokens, C_patch]
        #   action_pred: [B, F_action * N_action, C_action]
        if isinstance(pred, tuple) and len(pred) == 3:
            latent_pred, action_pred, tactile_pred = pred
        else:
            latent_pred, action_pred = pred
            tactile_pred = None

        latent_loss = self._compute_latent_loss(input_dict, latent_pred)
        action_loss = self._compute_action_loss(input_dict, action_pred)

        # Symdiff tactile diffusion MSE: target = ε - x_0 (flow matching),
        # stored under action_dict['tactile_global_targets']. Zero when target
        # absent (CFG drop) — but we MUST still touch tactile_pred so the
        # tactile_proj_out parameters get a grad and FSDP doesn't see mixed
        # dtypes (None grad treated as fp32 vs bf16 grad on the rest).
        device = latent_pred.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        tactile_loss = zero
        if tactile_pred is not None:
            if 'tactile_global_targets' in input_dict['action_dict']:
                target = input_dict['action_dict']['tactile_global_targets']
                # target: dense (B, S, C, F, H, W). tactile_pred is a patch
                # sequence (B, S*F*H*W, C) in (sensor, [f hp wp], [pf ph pw]) patch
                # order — the SAME convention as the video latent prediction. So,
                # exactly like _compute_latent_loss, un-patchify the prediction
                # back to dense (folding the sensor axis into the batch) and
                # compare dense-to-dense. The previous code instead row-major-
                # flattened the *target*, which silently mismatched the patch-block
                # ordering of the prediction and supervised the wrong spatial cells.
                B, S, C, F_lat, H_lat, W_lat = target.shape
                pred_dense = data_seq_to_patch(
                    self.patch_size,
                    tactile_pred.reshape(B * S, F_lat * H_lat * W_lat, C),
                    F_lat, H_lat, W_lat,
                    batch_size=B * S,
                ).reshape(B, S, C, F_lat, H_lat, W_lat)
                # Keep MSE in pred.dtype (bf16) so tactile_proj_out's grad dtype
                # stays uniform under FSDP, then promote the scalar to fp32.
                tactile_loss = torch.nn.functional.mse_loss(
                    pred_dense, target.to(pred_dense.dtype)).float()
            else:
                # CFG-drop / no-target step: keep tactile_proj_out in autograd
                # graph with zero contribution so FSDP grad dtype stays uniform.
                tactile_loss = (tactile_pred.sum() * 0.0).float()

        weight = float(getattr(self.config, 'tactile_diffusion_loss_weight', 1.0))
        total_loss = latent_loss + action_loss + weight * tactile_loss
        # Divide before backward so accumulated micro-batches produce the same
        # gradient scale as a single large batch.
        scale = self.gradient_accumulation_steps
        return {
            'latent_loss': latent_loss / scale,
            'action_loss': action_loss / scale,
            'tactile_loss': tactile_loss / scale,
            'total_loss': total_loss / scale,
        }

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        # batch_idx is the micro-step index inside the accumulation window.
        # FSDP gradient sync is skipped until the last micro-step.
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0

        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        output = self.transformer(input_dict, train_mode=True)
        loss_dict = self.compute_loss(input_dict, output)
        # total_loss is already divided by gradient_accumulation_steps.
        loss_dict['total_loss'].backward()

        losses = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in loss_dict.items()
        }

        # Only update weights after accumulating gradients
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    @torch.no_grad()
    def _validate(self):
        """Loss-only pass over the held-out set.

        Randomness (timesteps, noisy-cond, chunk/window sizes, CFG drops) is
        re-seeded identically every call via fork_rng, so successive val losses
        are comparable rather than dominated by diffusion-timestep noise.
        """
        metrics = {'latent_loss': [], 'action_loss': [], 'tactile_loss': [], 'total_loss': []}
        with torch.random.fork_rng(devices=[self.device]):
            for batch_index, batch in enumerate(self.val_loader):
                sample_seed = self.val_sample_seeds[batch_index]
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                batch = self.convert_input_format(batch)
                input_dict = self._prepare_input_dict(batch)
                output = self.transformer(input_dict, train_mode=True)
                loss_dict = self.compute_loss(input_dict, output)
                if not self.val_sample_validity[batch_index]:
                    continue
                for name in metrics:
                    if name in loss_dict:
                        # undo the grad-accum division for honest per-sample loss
                        metrics[name].append(
                            loss_dict[name].detach() * self.gradient_accumulation_steps)
        out = {}
        for name, vals in metrics.items():
            local_sum = (
                torch.stack(vals).sum()
                if vals
                else torch.zeros((), device=self.device)
            )
            local_count = torch.tensor(
                float(len(vals)), device=self.device, dtype=torch.float32
            )
            if dist.is_initialized():
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            if float(local_count.item()) <= 0:
                out[name] = 0.0
            else:
                out[name] = (local_sum / local_count.to(local_sum.dtype)).item()
        return out

    def _raise_if_checkpoint_stage_failed(self, error, stage):
        """Propagate a rank-local checkpoint error to every distributed rank."""
        failed = torch.tensor(
            [0 if error is None else 1],
            device=self.device if torch.cuda.is_available() else 'cpu')
        if dist.is_initialized():
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if int(failed.item()):
            raise RuntimeError(
                f"checkpoint {stage} failed at step {self.step}; all ranks stop"
            ) from error

    def save_checkpoint(self):
        """Save model and complete state required for a strict resume."""
        final_checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
        checkpoint_dir = self.save_dir / f".checkpoint_step_{self.step}.incomplete"
        stage_error = None
        try:
            self._persist_completed_epoch_exposure_at_checkpoint()
        except Exception as error:
            stage_error = error
            logger.exception(
                "Failed to persist completed sampler exposure before checkpoint"
            )
        self._raise_if_checkpoint_stage_failed(stage_error, "exposure stage")
        runtime_signature = capture_runtime_signature()
        validate_runtime_signature(runtime_signature)
        current_execution_contract = build_training_execution_contract(
            max_latent_frames=int(getattr(self.config, 'max_latent_frames', 0)),
            gradient_accumulation_steps=int(
                self.gradient_accumulation_steps),
            batch_size=int(getattr(self.config, 'batch_size', 1)),
            load_worker=int(getattr(self.config, 'load_worker', 0)),
            num_steps=int(getattr(self.config, 'num_steps', 0)),
            lr_schedule=str(getattr(
                self.config, 'lr_schedule', 'constant')),
            warmup_steps=int(getattr(self.config, 'warmup_steps', 0)),
            lr_min_ratio=float(getattr(self.config, 'lr_min_ratio', 0.0)),
            activation_checkpointing=_activation_checkpointing_enabled(),
            attention_contract=capture_attention_execution_contract(),
        )
        training_execution_contract = validate_training_execution_contract(
            self.training_execution_contract,
            current_contract=current_execution_contract,
        )
        training_profile_identity = _build_track31_profile_identity(self.config)
        latent_inventory_identity: dict[str, str] = {}
        checkpoint_provenance: dict[str, object] = {}
        track32_artifact_identity = getattr(
            self.config, 'track32_artifact_identity', None
        )
        if getattr(self.config, 'track32_profile_id', None) is not None:
            if not isinstance(track32_artifact_identity, dict):
                raise ValueError(
                    "Track 3.2 checkpoint requires a bound artifact identity"
                )
        if training_profile_identity is not None:
            if self.track31_artifacts is None:
                raise ValueError(
                    "formal Track 3.1 checkpoint requires verified artifacts"
                )
            latent_inventory_identity = validate_latent_inventory_identity(
                self.track31_artifacts.to_json_dict(),
                label="checkpoint Track 3.1 artifacts",
            )
            if any(
                training_profile_identity.get(key) != value
                for key, value in latent_inventory_identity.items()
            ):
                raise ValueError(
                    "training profile latent inventory identity changed before save"
                )
        if (
            self.runtime_source_identity is None
        ) != (
            self.checkpoint_invocation_identity is None
        ):
            raise ValueError("checkpoint runtime provenance is incomplete")
        if self.runtime_source_identity is not None:
            checkpoint_provenance = {
                "runtime_source_identity": validate_runtime_source_identity(
                    self.runtime_source_identity
                ),
                "checkpoint_invocation_identity": (
                    validate_checkpoint_invocation_identity(
                        self.checkpoint_invocation_identity
                    )
                ),
            }
        rng_state = capture_rng_state()
        transformer_identity = None
        stage_error = None
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            if self.config.rank == 0:
                if final_checkpoint_dir.exists() or checkpoint_dir.exists():
                    raise FileExistsError(
                        "refusing to overwrite an existing final or staging "
                        f"checkpoint: {final_checkpoint_dir}, {checkpoint_dir}")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving transformer to {transformer_dir}")
                state_dict_bf16 = {
                    key: value.to(torch.bfloat16)
                    for key, value in state_dict.items()
                }
                save_file(
                    state_dict_bf16,
                    transformer_dir / "diffusion_pytorch_model.safetensors",
                )
                if self.track31_tactile_contract is None:
                    config_dict = dict(self.transformer.config)
                    config_dict.pop('_name_or_path', None)
                else:
                    config_dict = _build_track31_transformer_checkpoint_config(
                        self.transformer,
                        self.track31_tactile_contract,
                        action_schema=self.action_codec.spec.name,
                        action_dim=self.action_codec.spec.dim,
                    )
                config_dict["tactile_profile"] = (
                    self.tactile_profile_contract.profile
                )
                config_dict["tactile_profile_contract_sha256"] = (
                    self.tactile_profile_contract.contract_sha256
                )
                write_json_atomic(transformer_dir / "config.json", config_dict)
                transformer_identity = audit_transformer_checkpoint(
                    transformer_dir / TRANSFORMER_WEIGHTS_FILENAME,
                    expected_action_dim=int(getattr(self.config, 'action_dim', 0)),
                )

                config = self.config
                meta = {
                    'step': self.step,
                    'norm_stat': {
                        key: list(value)
                        for key, value in dict(config.norm_stat).items()
                        if isinstance(value, (list, tuple))
                    },
                    'norm_stat_path': getattr(config, 'norm_stat_path', None),
                    'action_norm_method': getattr(
                        config, 'action_norm_method', None),
                    'action_delta_mode': getattr(
                        config, 'action_delta_mode', None),
                    'action_schema': getattr(config, 'action_schema', None),
                    'action_spec': self.action_codec.spec.to_json_dict(),
                    'action_dim': int(getattr(config, 'action_dim', 0)),
                    'pi05_action_horizon': int(getattr(
                        config, 'pi05_action_horizon', 0)),
                    'action_per_frame': int(getattr(
                        config, 'action_per_frame', 0)),
                    'used_action_channel_ids': [
                        int(value) for value in getattr(
                            config, 'used_action_channel_ids', [])
                    ],
                    'use_local_tactile': bool(getattr(
                        config, 'use_local_tactile', False)),
                    'local_tactile_mode': getattr(
                        config, 'local_tactile_mode', None),
                    'tactile_global_zero': bool(getattr(
                        config, 'tactile_global_zero', False)),
                    'obs_cam_keys': list(getattr(config, 'obs_cam_keys', [])),
                    'tactile_keys': list(getattr(config, 'tactile_keys', [])),
                    'eval_prompt': getattr(config, 'eval_prompt', None),
                    'training_execution_contract': training_execution_contract,
                    'training_profile_id': getattr(
                        config, 'training_profile_id', None),
                    'track32_profile_id': getattr(
                        config, 'track32_profile_id', None),
                    'track32_artifact_identity': track32_artifact_identity,
                    'run_role': getattr(config, 'run_role', None),
                    'accelerator_profile': getattr(
                        config, 'accelerator_profile', None),
                    'train_view_id': getattr(config, 'train_view_id', None),
                    'validation_view_id': getattr(
                        config, 'validation_view_id', None),
                    'training_profile_identity': training_profile_identity,
                    'training_lineage': self.training_lineage,
                    'tactile_mode': getattr(config, 'tactile_mode', 'enabled'),
                    'tactile_profile': self.tactile_profile_contract.profile,
                    'tactile_profile_contract': (
                        self.tactile_profile_contract.to_json_dict()
                    ),
                    'source_action_schema': getattr(
                        config, 'source_action_schema', None),
                    'derived_action_schema': getattr(
                        config, 'derived_action_schema', None),
                    'trainability_contract': (
                        self.trainability_contract.to_json_dict()
                    ),
                    'initialization_mode': getattr(
                        config, 'initialization_mode', None),
                    'transformer_identity': transformer_identity,
                    'sampler_state': (
                        self.train_sampler.state_dict()
                        if hasattr(self.train_sampler, 'state_dict')
                        else None
                    ),
                    'track31_artifacts': (
                        None
                        if self.track31_artifacts is None
                        else self.track31_artifacts.to_json_dict()
                    ),
                    **latent_inventory_identity,
                    **checkpoint_provenance,
                }
                if self.track31_tactile_contract is not None:
                    meta.update(
                        _build_track31_tactile_checkpoint_metadata(
                            self.track31_tactile_contract
                        )
                    )
                write_json_atomic(checkpoint_dir / "train_meta.json", meta)
                if self.action_migration_report is not None:
                    write_json_atomic(
                        checkpoint_dir / "action_migration_report.json",
                        self.action_migration_report,
                    )
        except Exception as error:
            stage_error = error
            logger.exception("Failed to save model checkpoint stage")
        self._raise_if_checkpoint_stage_failed(stage_error, "model stage")

        del state_dict
        if self.config.rank == 0:
            del state_dict_bf16
        gc.collect()

        stage_error = None
        try:
            optimizer_inventory = save_optimizer_checkpoint(
                self.transformer,
                self.optimizer,
                checkpoint_dir / OPTIMIZER_DCP_DIRNAME,
            )
            if self.config.rank == 0:
                if transformer_identity is None:
                    raise RuntimeError(
                        "saved transformer identity was not produced")
                save_scheduler_state(
                    checkpoint_dir,
                    self.lr_scheduler.state_dict(),
                    completed_steps=int(self.step),
                    learning_rate=float(self.config.learning_rate),
                    execution_contract=training_execution_contract,
                )
                training_state = {
                    'schema_version': STRICT_CHECKPOINT_SCHEMA_VERSION,
                    'step': self.step,
                    'data_batches_consumed': self.data_batches_consumed,
                    'world_size': int(self.config.world_size),
                    'gradient_accumulation_steps': int(
                        self.gradient_accumulation_steps),
                    'action_schema': self.action_codec.spec.name,
                    'optimizer_state_format': OPTIMIZER_STATE_FORMAT,
                    'optimizer_inventory_sha256': (
                        optimizer_inventory.inventory_sha256
                    ),
                    'runtime_signature': runtime_signature,
                    'training_execution_contract': training_execution_contract,
                    'training_profile_identity': training_profile_identity,
                    'training_lineage': self.training_lineage,
                    'transformer_identity': transformer_identity,
                    'track32_artifact_identity': track32_artifact_identity,
                    'tactile_profile_contract': (
                        self.tactile_profile_contract.to_json_dict()
                    ),
                    **latent_inventory_identity,
                    **checkpoint_provenance,
                }
                write_json_atomic(
                    checkpoint_dir / "training_state.json", training_state
                )
            save_rng_state(
                checkpoint_dir,
                rank=int(self.config.rank),
                world_size=int(self.config.world_size),
                state=rng_state,
            )
        except Exception as error:
            stage_error = error
            logger.exception("Failed to save optimizer/resume checkpoint stage")
        self._raise_if_checkpoint_stage_failed(stage_error, "resume-state stage")
        gc.collect()

        if dist.is_initialized():
            dist.barrier()
        stage_error = None
        try:
            if self.config.rank == 0:
                if transformer_identity is None:
                    raise RuntimeError(
                        "saved transformer identity was not produced")
                sidecar_paths = expected_sidecar_paths(
                    int(self.config.world_size),
                    include_transformer_config=True,
                    include_action_migration=(
                        self.action_migration_report is not None
                    ),
                )
                sidecar_inventory = build_sidecar_inventory(
                    checkpoint_dir, sidecar_paths
                )
                completion = {
                    'schema_version': STRICT_CHECKPOINT_SCHEMA_VERSION,
                    'step': self.step,
                    'world_size': int(self.config.world_size),
                    'action_schema': self.action_codec.spec.name,
                    'optimizer_state_format': OPTIMIZER_STATE_FORMAT,
                    'optimizer_inventory_sha256': (
                        optimizer_inventory.inventory_sha256
                    ),
                    'runtime_signature': runtime_signature,
                    'training_execution_contract': training_execution_contract,
                    'training_profile_identity': training_profile_identity,
                    'training_lineage': self.training_lineage,
                    'transformer_identity': transformer_identity,
                    'track32_artifact_identity': track32_artifact_identity,
                    'tactile_profile_contract': (
                        self.tactile_profile_contract.to_json_dict()
                    ),
                    'sidecar_inventory': sidecar_inventory,
                    'status': 'complete',
                    **latent_inventory_identity,
                    **checkpoint_provenance,
                }
                write_json_atomic(
                    checkpoint_dir / "checkpoint_complete.json", completion
                )
        except Exception as error:
            stage_error = error
            logger.exception("Failed to finalize checkpoint completion marker")
        self._raise_if_checkpoint_stage_failed(stage_error, "completion stage")
        if dist.is_initialized():
            dist.barrier()

        stage_error = None
        try:
            if self.config.rank == 0:
                os.replace(checkpoint_dir, final_checkpoint_dir)
                parent_descriptor = os.open(self.save_dir, os.O_RDONLY)
                try:
                    os.fsync(parent_descriptor)
                finally:
                    os.close(parent_descriptor)
        except Exception as error:
            stage_error = error
            logger.exception("Failed to publish completed checkpoint directory")
        self._raise_if_checkpoint_stage_failed(stage_error, "publish stage")
        if dist.is_initialized():
            dist.barrier()
        if self.config.rank == 0:
            logger.info(
                f"Checkpoint saved successfully at {final_checkpoint_dir}"
            )

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        invocation_contract = self._validate_invocation_contract()
        self.invocation_contract = invocation_contract
        stop_after_step = int(invocation_contract['stop_after_step'])
        # optimizer-steps per epoch (len() covers both the bucketed batch_sampler
        # and the plain DistributedSampler paths) -> show epoch progress alongside
        # the step-based loop.
        try:
            self.steps_per_epoch = max(
                1, len(self.train_loader) // max(1, self.gradient_accumulation_steps))
        except Exception:
            self.steps_per_epoch = 0
        _ep = (f"{self.config.num_steps / self.steps_per_epoch:.2f}"
               if self.steps_per_epoch else "?")
        logger.info(
            "Starting training invocation at global optimizer step "
            f"{self.step}; stop_after_step={stop_after_step}, "
            f"final scheduler horizon={self.config.num_steps} "
            f"(~{self.steps_per_epoch} optim-steps/epoch -> ~{_ep} epochs)..."
        )
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        metric_names = [
            'latent_loss',
            'action_loss',
            'total_loss',
            'tactile_loss',     # symdiff; always tracked (0 if unused)
        ]
        accumulated_metrics = {name: [] for name in metric_names}
        step_in_accumulation = 0

        while self.step < stop_after_step:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()

            try:
                losses = self._train_step(batch, step_in_accumulation)
            except Exception as e:
                import traceback

                rank = self.config.rank
                err_msg = f"\n{'='*60}\n[RANK {rank}] CRASH at step {self.step}, iter {step_in_accumulation}\n{traceback.format_exc()}{'='*60}\n"
                logger.error(err_msg)
                # Keep immutable source snapshots read-only in formal runs. Crash
                # diagnostics belong to the writable run root, and a secondary
                # logging failure must never mask the original training error.
                try:
                    crash_dir = Path(self.config.save_root) / "logs"
                    crash_dir.mkdir(parents=True, exist_ok=True)
                    (crash_dir / f"crash_rank{rank}.log").write_text(
                        err_msg,
                        encoding="utf-8",
                    )
                except OSError:
                    logger.exception(
                        "Unable to persist crash diagnostics for rank %s",
                        rank,
                    )
                raise

            # Accumulate losses for logging
            for name in metric_names:
                if name in losses:
                    accumulated_metrics[name].append(losses[name])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                present_metrics = [
                    name for name in metric_names if accumulated_metrics[name]
                ]
                local_metrics = torch.stack(
                    [
                        torch.stack(accumulated_metrics[name]).sum()
                        for name in present_metrics
                    ]
                )
                mean_metrics, max_metrics = dist_mean_and_max(local_metrics)
                metric_shows = dict(
                    zip(
                        present_metrics,
                        mean_metrics.detach().cpu().tolist(),
                        strict=True,
                    )
                )
                max_metric_shows = dict(
                    zip(
                        present_metrics,
                        max_metrics.detach().cpu().tolist(),
                        strict=True,
                    )
                )

                accumulated_metrics = {name: [] for name in metric_names}
                step_in_accumulation = 0

                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_postfix = {
                        'latent_loss':  f'{metric_shows.get("latent_loss", 0.0):.4f}',
                        'action_loss':  f'{metric_shows.get("action_loss", 0.0):.4f}',
                        'tactile_loss': f'{metric_shows.get("tactile_loss", 0.0):.4f}',
                        'step': self.step,
                        'epoch': (f'{self.step / self.steps_per_epoch:.2f}'
                                  if self.steps_per_epoch else '?'),
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    }
                    progress_bar.set_postfix(progress_postfix)
                    if self.config.enable_wandb:
                        wandb_log = {
                            'loss_metrics/global_avg_video_loss':   metric_shows.get('latent_loss', 0.0),
                            'loss_metrics/global_avg_action_loss':  metric_shows.get('action_loss', 0.0),
                            'loss_metrics/global_avg_tactile_loss': metric_shows.get('tactile_loss', 0.0),
                            'loss_metrics/global_max_video_loss':   max_metric_shows.get('latent_loss', 0.0),
                            'loss_metrics/global_max_action_loss':  max_metric_shows.get('action_loss', 0.0),
                            'loss_metrics/global_max_tactile_loss': max_metric_shows.get('tactile_loss', 0.0),
                            'loss_metrics/global_avg_total_loss':   metric_shows.get('total_loss', 0.0),
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                            'epoch': (self.step / self.steps_per_epoch
                                      if self.steps_per_epoch else 0),
                        }
                        self.wandb.log(wandb_log, step=self.step)

                self.step += 1

                if self.val_loader is not None and self.step % self.val_interval == 0:
                    val_metrics = self._validate()
                    if self.config.rank == 0:
                        logger.info(
                            f"[val @ step {self.step}] " + " ".join(
                                f"{k}={v:.4f}" for k, v in val_metrics.items()))
                        if self.config.enable_wandb:
                            self.wandb.log({f'val_metrics/{k}': v
                                            for k, v in val_metrics.items()},
                                           step=self.step)

                checkpoint_due = (
                    self.step % self.config.save_interval == 0
                    or self.step == stop_after_step
                )
                if checkpoint_due:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

        progress_bar.close()
        if self.step == self.config.num_steps:
            logger.info(
                f"Final training completed at optimizer step {self.step}."
            )
        else:
            logger.info(
                "Training invocation completed at optimizer step "
                f"{self.step} of final horizon {self.config.num_steps}; "
                "resume from this checkpoint for the next stage."
            )


def run(args):
    """Main entry point."""
    config = TWAM_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if hasattr(config, 'seed'):
        process_seed = _set_reproducibility(config.seed, rank)
    else:
        process_seed = None

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        if process_seed is not None:
            logger.info(f"Reproducibility seed: {process_seed}")

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='posttrain',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
