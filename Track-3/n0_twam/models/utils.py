# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import hashlib
import os
import types

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel

_WAN_CONV3D_FALLBACK_ENV = "N0_WAN_VAE_CONV3D_FALLBACK"


def _wan_causal_conv3d_as_2d(module, x, cache_x=None):
    """Execute a causal Conv3d exactly as temporal Conv2d slices.

    Some HCU PyTorch builds expose the CUDA compatibility device but omit the
    eager ``slow_conv3d_forward`` kernel used by Diffusers' Wan VAE.  Spatial
    Conv2d is available and the temporal kernel is small, so decomposing the
    convolution preserves the weights and causal-cache semantics without
    moving either the VAE or observations to CPU.
    """

    padding = list(module._padding)
    if cache_x is not None and module._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)

    kernel_t, _, _ = module.kernel_size
    stride_t, stride_h, stride_w = module.stride
    dilation_t, dilation_h, dilation_w = module.dilation
    effective_kernel_t = dilation_t * (kernel_t - 1) + 1
    output_t = (x.shape[2] - effective_kernel_t) // stride_t + 1
    if output_t <= 0:
        raise ValueError("Wan Conv3d fallback received an invalid temporal shape")

    frames = []
    for output_index in range(output_t):
        start = output_index * stride_t
        frame = None
        for kernel_index in range(kernel_t):
            temporal_index = start + kernel_index * dilation_t
            contribution = F.conv2d(
                x[:, :, temporal_index],
                module.weight[:, :, kernel_index],
                bias=None,
                stride=(stride_h, stride_w),
                padding=(0, 0),
                dilation=(dilation_h, dilation_w),
                groups=module.groups,
            )
            frame = contribution if frame is None else frame + contribution
        if module.bias is not None:
            frame = frame + module.bias.view(1, -1, 1, 1)
        frames.append(frame)
    return torch.stack(frames, dim=2)


def _install_wan_conv3d_fallback(vae):
    patched = 0
    for module in vae.modules():
        if module.__class__.__name__ != "WanCausalConv3d":
            continue
        if getattr(module, "_n0_twam_conv3d_fallback", False):
            continue
        module.forward = types.MethodType(_wan_causal_conv3d_as_2d, module)
        module._n0_twam_conv3d_fallback = True
        patched += 1
    if patched == 0:
        raise RuntimeError("Wan VAE Conv3d fallback did not find causal convolutions")


def _wan_conv3d_fallback_requested():
    value = os.environ.get(_WAN_CONV3D_FALLBACK_ENV, "").strip().lower()
    if value in {"", "0", "false", "no"}:
        return False
    if value in {"1", "true", "yes"}:
        return True
    raise ValueError(f"{_WAN_CONV3D_FALLBACK_ENV} must be one of 0/1/false/true/no/yes")


def _sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reset_module_parameters(module):
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def _materialize_tactile_modules(model, device):
    tactile_module_names = [
        # GlobalTactile latent modules (predicted in the self-attention sequence)
        "tactile_patch_embed",
        "sensor_id_embed",
        "tactile_norm",
        # LocalTactile cross-attn (action stream)
        "local_tactile_patch_embed",
        "local_tactile_sensor_embed",
        "local_tactile_frame_embed",
        "local_tactile_h_embed",
        "local_tactile_w_embed",
        "local_tactile_norm",
        "local_tactile_pre_norm",
        "local_tactile_cross_attn",
        # symdiff tactile: tactile diffusion output head
        "tactile_proj_out",
        # predictive-contact gate (opt-in predictive-contact-gate add-on; not in base ckpts)
        "contact_gate",
    ]
    target_device = torch.device(device)
    for module_name in tactile_module_names:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        has_meta_params = any(
            getattr(param, "is_meta", False)
            for param in module.parameters(recurse=True)
        )
        has_meta_buffers = any(
            getattr(buf, "is_meta", False) for buf in module.buffers(recurse=True)
        )
        if not has_meta_params and not has_meta_buffers:
            continue
        module.to_empty(device=target_device)
        _reset_module_parameters(module)
        # contact_gate was just materialised fresh (absent from the checkpoint):
        # re-zero its FiLM so it starts as an exact identity / no-op. A checkpoint
        # that already contains a trained contact_gate is NOT meta -> skipped above
        # -> keeps its trained weights.
        if module_name == "contact_gate" and hasattr(module, "reset_film"):
            module.reset_film()


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    if _wan_conv3d_fallback_requested():
        _install_wan_conv3d_fallback(vae)
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
    direct_device_load=False,
):
    if direct_device_load:
        return UMT5EncoderModel.from_pretrained(
            text_encoder_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map={"": str(torch.device(torch_device))},
        )
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(
    tokenizer_path,
):
    tokenizer = T5TokenizerFast.from_pretrained(
        tokenizer_path,
    )
    return tokenizer


def _resize_action_head(model, target_action_dim, device):
    """Replace action_embedder (Linear in→inner) and action_proj_out
    (Linear inner→out) when the new schema size differs from the pretrained
    one. Both layers are re-initialised from scratch — the rest of the
    transformer keeps the loaded weights.

    Used when a downstream task needs a different action super-schema (e.g.
    20-dim left/right canonical, or any other reshuffle) without retraining
    the entire VLA.
    """
    import torch.nn as nn

    old_in_dim = model.action_embedder.in_features
    inner_dim = model.action_embedder.out_features
    if target_action_dim == old_in_dim:
        return
    model.action_embedder = nn.Linear(target_action_dim, inner_dim).to(
        device=device, dtype=model.action_embedder.weight.dtype
    )
    model.action_proj_out = nn.Linear(inner_dim, target_action_dim).to(
        device=device, dtype=model.action_proj_out.weight.dtype
    )
    # Keep config in sync — downstream code (loss, eval) AND from_pretrained on
    # resume read model.config.action_dim. Assigning model.config.action_dim
    # directly is a NO-OP (diffusers FrozenDict ignores it), which makes the saved
    # config.json disagree with the resized weights and re-randomises the action
    # head on every resume. register_to_config actually persists the new value so
    # the saved config matches the 20-dim head and resume skips a second resize.
    if hasattr(model, "register_to_config"):
        model.register_to_config(action_dim=target_action_dim)
    elif hasattr(model, "config"):
        model.config.action_dim = target_action_dim


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    target_action_dim=None,
    **kwargs,
):
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs,
    )
    _materialize_tactile_modules(model, torch_device)
    if target_action_dim is not None:
        _resize_action_head(model, target_action_dim, torch_device)
    return model.to(torch_device)


@torch.no_grad()
def _warmstart_mot_from_legacy(mot, legacy, warmstart_experts):
    """In-place copy a legacy (shared-backbone) model's weights into a MoT model:
    every `blocks.{i}.*` tensor is copied into each chosen FULL-width expert
    `mot.experts.{e}.blocks.{i}.*`; all other tensors (token embeddings, text/time
    conditioning, tactile modules, heads, output norm) are copied verbatim. In-place
    `copy_` avoids building a 3x-cloned state_dict.

    NARROW experts (hidden_dim != shared dim) cannot receive the wide WAN blocks, so
    they are skipped and keep their random init (+ their thin in/out/time/text
    projections). Returns (n_copied, full_warmstarted, skipped_narrow)."""
    legacy_sd = legacy.state_dict()
    tgt = dict(mot.named_parameters())
    tgt.update(dict(mot.named_buffers()))
    # only FULL-width experts can take the wide WAN blocks
    full_ws = [e for e in warmstart_experts if not mot.mot.experts[e].narrow]
    skipped = [e for e in warmstart_experts if mot.mot.experts[e].narrow]
    copied, missed = 0, []

    def _copy(dst_name, src):
        nonlocal copied
        dst = tgt.get(dst_name)
        if dst is None:
            missed.append(dst_name)
        elif tuple(dst.shape) != tuple(src.shape):
            missed.append(f"{dst_name}[{tuple(dst.shape)} vs {tuple(src.shape)}]")
        else:
            dst.copy_(src)
            copied += 1

    for name, src in legacy_sd.items():
        if name.startswith("blocks."):
            rest = name[len("blocks.") :]
            for e in full_ws:
                _copy(f"mot.experts.{e}.blocks.{rest}", src)
        else:
            _copy(name, src)  # model-level embeddings/heads/conditioning, verbatim

    if missed:
        raise RuntimeError(
            f"MoT warm-start: {len(missed)} legacy keys could not be mapped, "
            f"e.g. {missed[:10]}"
        )
    return copied, full_ws, skipped


def load_mot_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    target_action_dim=None,
    mot_expert_ffn_dim=None,
    mot_expert_hidden_dim=None,
    mot_cross_attn_experts=("video", "action"),
    mot_warmstart_experts=("video", "action", "tactile"),
    **kwargs,
):
    """Build a Mixture-of-Transformers TWAM model, warm-started from a shared-backbone
    WAN checkpoint. The legacy model is loaded normally (WAN transformer weights +
    materialised tactile modules + resized action head); its weights are then copied
    into the MoT model — `blocks.{i}.*` into each `mot_warmstart_experts` (default
    all three), everything else verbatim. Experts not in `mot_warmstart_experts`
    keep their random init."""
    from .mot import WanMoTTransformer3DModel

    legacy = load_transformer(
        transformer_path,
        torch_dtype=torch_dtype,
        torch_device="cpu",
        target_action_dim=target_action_dim,
        **kwargs,
    )
    cfg = {k: v for k, v in dict(legacy.config).items() if not k.startswith("_")}
    # Build the MoT skeleton in the SAME (low-precision) dtype: 3 experts built in
    # fp32 on every rank would OOM the container memory cgroup (a single rank peaks
    # at ~3x the base model). set_default_dtype makes nn.Linear/LayerNorm/Parameter
    # allocate directly in torch_dtype, avoiding a transient fp32 copy. FSDP master
    # params therefore start in torch_dtype (pass bf16 for the memory-lean path).
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        mot = WanMoTTransformer3DModel(
            **cfg,
            mot_expert_ffn_dim=mot_expert_ffn_dim,
            mot_expert_hidden_dim=mot_expert_hidden_dim,
            mot_cross_attn_experts=tuple(mot_cross_attn_experts),
        )
    finally:
        torch.set_default_dtype(old_dtype)
    n, full_ws, skipped = _warmstart_mot_from_legacy(
        mot, legacy, tuple(mot_warmstart_experts)
    )
    print(
        f"[load_mot_transformer] warm-started {n} tensors; full experts from WAN="
        f"{full_ws}; narrow experts left random={skipped}; "
        f"cross-attn experts={tuple(mot_cross_attn_experts)}; "
        f"hidden_dim={mot_expert_hidden_dim}"
    )
    del legacy
    import gc

    gc.collect()
    return mot.to(torch_device)


def load_mot_checkpoint(
    checkpoint_dir,
    torch_dtype=torch.bfloat16,
    torch_device="cpu",
    attn_mode="torch",
    config_overrides=None,
    compatibility="strict",
    target_action_dim=None,
    action_init_seed=0,
    expected_source_action_dim=None,
    expected_source_action_schema=None,
    target_action_schema=None,
    adopt_missing_action_schema=False,
    expected_checkpoint_sha256=None,
):
    """Load a trained MoT checkpoint (a `.../transformer` dir with config.json +
    diffusion_pytorch_model.safetensors), rebuilding the WanMoTTransformer3DModel
    with the saved expert structure (hidden_dim / ffn_dim / cross-attn experts
    recorded in config.json by register_to_config).

    config_overrides (dict|None): optional model-init overrides applied on top of
    the checkpoint's config.json BEFORE building. Used for post-training "flips"
    (e.g. use_local_tactile False->True) where the local-off pretrain ckpt lacks a
    module the post-train wants: the module is built (zero-init) and its
    absent-in-ckpt weights are tolerated as missing. None (default, e.g. inference/
    serve) => exact rebuild from the ckpt config, strict load as before.

    compatibility="migrate_action" copies only exact-shape non-action tensors and
    deterministically resets the complete action projection allowlist. This keeps
    20D EE and 8D qpos semantics from being silently mixed or sliced.
    """
    import json
    import os
    from safetensors.torch import load_file
    from n0_twam.checkpointing.compatibility import build_action_migration_plan
    from .mot import WanMoTTransformer3DModel

    tdir = os.path.join(checkpoint_dir, "transformer")
    if not os.path.isdir(tdir):
        tdir = checkpoint_dir
    with open(os.path.join(tdir, "config.json")) as f:
        cfg = json.load(f)
    if not cfg.get("is_mot", False):
        raise ValueError(
            f"{tdir}/config.json is not a MoT checkpoint (is_mot missing)."
        )
    if compatibility not in {"strict", "migrate_action"}:
        raise ValueError(
            "compatibility must be either 'strict' or 'migrate_action', "
            f"got {compatibility!r}"
        )
    if "action_dim" not in cfg:
        raise ValueError("checkpoint config is missing required action_dim")
    source_action_dim = int(cfg["action_dim"])
    recorded_action_schema = cfg.get("action_schema")
    if recorded_action_schema is not None and (
        not isinstance(recorded_action_schema, str) or not recorded_action_schema
    ):
        raise ValueError("checkpoint action_schema must be a non-empty string")
    if config_overrides and "action_dim" in config_overrides:
        raise ValueError(
            "action_dim must be provided through target_action_dim, not "
            "config_overrides"
        )
    if config_overrides and "action_schema" in config_overrides:
        raise ValueError(
            "action_schema must be provided through target_action_schema, not "
            "config_overrides"
        )
    if compatibility == "strict":
        if (
            target_action_dim is not None
            and int(target_action_dim) != source_action_dim
        ):
            raise ValueError(
                "strict checkpoint load cannot change action_dim: "
                f"{source_action_dim} -> {target_action_dim}"
            )
        if target_action_schema is not None:
            if recorded_action_schema is None and adopt_missing_action_schema:
                if source_action_dim != 20 or target_action_schema != "ee20_absee":
                    raise ValueError(
                        "missing action_schema adoption is restricted to the "
                        "released 20D ee20_absee checkpoint"
                    )
                if not isinstance(expected_checkpoint_sha256, str) or (
                    len(expected_checkpoint_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in expected_checkpoint_sha256
                    )
                ):
                    raise ValueError(
                        "missing action_schema adoption requires a lowercase "
                        "expected_checkpoint_sha256"
                    )
                actual_checkpoint_sha256 = _sha256_path(
                    os.path.join(tdir, "diffusion_pytorch_model.safetensors")
                )
                if actual_checkpoint_sha256 != expected_checkpoint_sha256:
                    raise ValueError(
                        "checkpoint SHA256 does not match the schema adoption contract"
                    )
            elif recorded_action_schema != target_action_schema:
                raise ValueError(
                    "strict checkpoint action schema mismatch: "
                    f"{recorded_action_schema!r} vs {target_action_schema!r}"
                )
        resolved_action_schema = target_action_schema or recorded_action_schema
    else:
        if target_action_dim is None or int(target_action_dim) <= 0:
            raise ValueError("migrate_action requires a positive target_action_dim")
        if (
            expected_source_action_dim is None
            or expected_source_action_schema is None
            or target_action_schema is None
        ):
            raise ValueError(
                "migrate_action requires source/target action schemas and source dimension"
            )
        if source_action_dim != int(expected_source_action_dim):
            raise ValueError(
                "checkpoint source action_dim does not match migration contract: "
                f"{source_action_dim} vs {expected_source_action_dim}"
            )
        if (
            recorded_action_schema is not None
            and recorded_action_schema != expected_source_action_schema
        ):
            raise ValueError(
                "checkpoint source action schema does not match migration contract: "
                f"{recorded_action_schema!r} vs "
                f"{expected_source_action_schema!r}"
            )
        cfg["action_dim"] = int(target_action_dim)
        resolved_action_schema = str(target_action_schema)
    mot_ffn = cfg.get("mot_expert_ffn_dim")
    mot_hdim = cfg.get("mot_expert_hidden_dim")
    mot_cross = cfg.get("mot_cross_attn_experts", ("video", "action"))

    # Capture the ckpt's ORIGINAL modality flags before applying overrides, so we
    # can tell which modules a config_overrides "flip" newly enables (their weights
    # are legitimately absent from the ckpt -> keep zero-init, tolerate as missing).
    ckpt_use_local = bool(cfg.get("use_local_tactile", False))
    ckpt_use_gate = bool(cfg.get("use_contact_gate", False))
    if config_overrides:
        for k, v in config_overrides.items():
            cfg[k] = v

    # keep only the kwargs the parent WanTransformer3DModel actually accepts: drop
    # diffusers private keys, the MoT-specific ones we pass explicitly, and any
    # config keys this code version's model no longer accepts -- e.g. a newer-branch
    # pretrain ckpt whose config.json carries use_proprio_state that this tree lacks
    # (forwarding it would raise TypeError in WanTransformer3DModel.__init__).
    import inspect
    from .model import WanTransformer3DModel

    _sig = inspect.signature(WanTransformer3DModel.__init__).parameters
    _accepted = (
        None
        if any(p.kind == p.VAR_KEYWORD for p in _sig.values())
        else (set(_sig) - {"self"})
    )
    drop = {
        "is_mot",
        "mot_expert_ffn_dim",
        "mot_expert_hidden_dim",
        "mot_cross_attn_experts",
    }
    init_cfg = {
        k: v
        for k, v in cfg.items()
        if not k.startswith("_")
        and k not in drop
        and (_accepted is None or k in _accepted)
    }
    init_cfg["attn_mode"] = attn_mode  # inference: SDPA, no flex compile

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        model = WanMoTTransformer3DModel(
            **init_cfg,
            mot_expert_ffn_dim=mot_ffn,
            mot_expert_hidden_dim=mot_hdim,
            mot_cross_attn_experts=tuple(mot_cross),
        )
    finally:
        torch.set_default_dtype(old)

    sd = load_file(os.path.join(tdir, "diffusion_pytorch_model.safetensors"))
    # Tolerated missing keys: the shared-attn stub is always allowed; and a module
    # NEWLY enabled via config_overrides (built here but off in the ckpt) is
    # legitimately absent -> keep its init. LocalTactile's out-proj is zero-init
    # (identity at step 0), so warm-starting a local-off ckpt is a no-op until
    # training moves it. Normal loads (no override) keep the strict check unchanged.
    tolerated = ["mot.shared_attn"]
    if bool(cfg.get("use_local_tactile", False)) and not ckpt_use_local:
        tolerated.append("local_tactile")
    if bool(cfg.get("use_contact_gate", False)) and not ckpt_use_gate:
        tolerated.append("contact_gate")
    if compatibility == "migrate_action":
        copied_state, migration_plan = build_action_migration_plan(
            sd,
            model.state_dict(),
            source_action_dim=source_action_dim,
            target_action_dim=int(target_action_dim),
        )
        migration_plan.raise_for_incompatible(
            tolerated_missing_prefixes=tuple(tolerated)
        )
        missing, unexpected = model.load_state_dict(copied_state, strict=False)
        expected_missing = set(migration_plan.reset_keys) | set(
            migration_plan.missing_target_keys
        )
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                "action migration load mismatch: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(action_init_seed))
            model.action_embedder.reset_parameters()
            model.action_proj_out.reset_parameters()
        model.action_migration_report = {
            "schema_version": 1,
            "compatibility": compatibility,
            "source_action_dim": source_action_dim,
            "source_action_schema": str(expected_source_action_schema),
            "target_action_dim": int(target_action_dim),
            "target_action_schema": str(target_action_schema),
            "action_init_seed": int(action_init_seed),
            "source_checkpoint": os.path.abspath(tdir),
            "source_checkpoint_sha256": _sha256_path(
                os.path.join(tdir, "diffusion_pytorch_model.safetensors")
            ),
            "plan": migration_plan.to_json_dict(),
        }
    else:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not any(k.startswith(p) for p in tolerated)]
        if missing or unexpected:
            raise RuntimeError(
                f"load_mot_checkpoint mismatch: missing={missing[:8]} "
                f"unexpected={unexpected[:8]}"
            )
    if resolved_action_schema is not None:
        if not hasattr(model, "register_to_config"):
            raise RuntimeError(
                "MoT model cannot preserve checkpoint action_schema metadata"
            )
        model.register_to_config(action_schema=resolved_action_schema)
    model.eval()
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(
        batch_size,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(
        batch_size,
        channels * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if (
            hasattr(self.vae.config, "patch_size")
            and self.vae.config.patch_size is not None
        ):
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
