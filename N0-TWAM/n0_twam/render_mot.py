"""MoT inference — denoising-consistency check (correctness first).

This does NOT re-derive an inference path (the legacy one drifted from the model).
It reuses the EXACT training machinery — Trainer(inference_only=True) + the same
_prepare_input_dict / _add_noise / forward_train / compute_loss — and verifies that
a trained (overfit) MoT checkpoint, run in eval, reconstructs the training data:

  * the three flow-matching losses should be ~the trained-step losses (model+forward
    are self-consistent: inference computes the same function as training);
  * the predicted velocity v=ε−x0 should recover x0 from the noised latent
    (x0_hat = x_noisy − σ·v), measured as a per-frame latent reconstruction MSE.

Run on a GPU (flex) box:
  cd <code> && python n0_twam/render_mot.py --ckpt <.../checkpoint_step_400/transformer> --n-batches 5
"""

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

# The package mixes import styles: train.py uses `from models...` (needs code/n0_twam
# on path) while model.py uses `from n0_twam.utils...` (needs code/ on path). Add BOTH.
_HERE = os.path.dirname(os.path.abspath(__file__))  # .../code/n0_twam
sys.path.insert(0, _HERE)  # configs / models / train / utils / dataset
sys.path.insert(0, os.path.dirname(_HERE))  # code/  -> `import n0_twam`

import torch

from configs import TWAM_CONFIGS
from dataset import MultiLatentLeRobotDataset
from models.utils import load_mot_checkpoint
from n0_twam.evaluation.tactile_checkpoint import audit_evaluation_checkpoint
from n0_twam.evaluation.tactile_model_contract import (
    audit_loaded_tactile_runtime,
    build_expected_tactile_model_contract,
)
from n0_twam.evaluation.tactile_artifacts import save_tactile_metric_frames
from n0_twam.evaluation.tactile_provenance import (
    audit_evaluation_dataset,
    audit_vae_decoder,
    build_deterministic_eval_loader,
    build_provenance_bound_report,
    capture_source_sample_metadata,
    mask_future_local_tactile,
    resolve_dataset_index_metadata,
    set_evaluation_seed,
    set_sample_seed,
    write_json_atomic,
)
from n0_twam.evaluation.render_helpers import (
    decode_video_latent,
    save_compare_video,
    save_frames,
    sigma_for_timesteps,
    timestep_for_sigma,
)
from n0_twam.evaluation.tactile_sampling import sample_tactile_ar
from train import Trainer


@torch.no_grad()
def sample_video(trainer, batch, n_steps=30, teacher_force=False):
    """i2v rectified-flow sampling that DRIVES forward_train (the exact trained
    computation) — frame 0 is the conditioning image, frames 1..F-1 are generated
    from noise via Euler steps x <- x + v*(dsigma), v=ε−x0 predicted by the model.

    teacher_force=True  : clean-stream = true x0 (plumbing test; must reconstruct).
    teacher_force=False : free i2v — clean-stream = self-conditioned x0_hat estimate
                          (frame 0 = true), so the model only ever sees frame 0 as
                          ground truth and generates the rest.
    Returns (x_final, x0_true) latents [B,48,F,H,W]."""
    model = trainer.transformer
    sched = trainer.train_scheduler_latent
    input_dict = trainer._prepare_input_dict(batch)
    # Fixed chunk/window for sampling: chunk_size=1 -> each frame its own chunk, so
    # frame f attends clean frames < f (clean autoregressive i2v causality).
    input_dict["chunk_size"] = 1
    input_dict["window_size"] = 64

    ld = input_dict["latent_dict"]
    x0_true = ld["latent"].clone()  # [B,48,F,H,W]
    B, C, F, H, W = x0_true.shape
    dev = x0_true.device

    # condition the action/tactile streams as CLEAN (sigma~0): for video i2v we feed
    # them as context, not as generation targets (their noisy halves = their clean).
    t0 = timestep_for_sigma(sched, 0.0)
    ad = input_dict["action_dict"]
    ad["noisy_latents"] = ad["latent"].clone()
    ad["timesteps"] = torch.full_like(ad["timesteps"], t0)
    if (
        "tactile_global_noisy_latent" in ad
        and ad.get("tactile_global_clean_latent") is not None
    ):
        ad["tactile_global_noisy_latent"] = ad["tactile_global_clean_latent"].clone()
        if "tactile_global_timesteps" in ad:
            ad["tactile_global_timesteps"] = torch.full_like(
                ad["tactile_global_timesteps"], t0
            )

    # sigma schedule (subset of the training sigmas, high->low), n_steps+1 points
    full_sig = sched.sigmas
    idx = torch.linspace(0, len(full_sig) - 1, n_steps + 1).round().long()
    sigmas = full_sig[idx].tolist()  # [n_steps+1], ~1 -> ~0

    # init: frame0 = true (condition), rest = noise at sigma[0]
    x = torch.randn_like(x0_true) * 1.0
    x[:, :, 0:1] = x0_true[:, :, 0:1]
    x0_hat = x0_true[:, :, 0:1].repeat(1, 1, F, 1, 1).clone()  # clean-stream init

    cond_ts = torch.full((B, F), t0, device=dev, dtype=ld["timesteps"].dtype)
    for i in range(n_steps):
        sig, sig_next = sigmas[i], sigmas[i + 1]
        t_sig = timestep_for_sigma(sched, sig)
        ts = torch.full((B, F), t_sig, device=dev, dtype=ld["timesteps"].dtype)
        ts[:, 0] = t0  # frame 0 clean
        ld["noisy_latents"] = x
        ld["timesteps"] = ts
        ld["cond_timesteps"] = cond_ts
        ld["latent"] = x0_true if teacher_force else x0_hat
        pred = model(input_dict, train_mode=True)
        v = _v_dense(trainer, pred[0], x0_true)  # dense velocity [B,48,F,H,W]
        x = x + v * (sig_next - sig)
        x[:, :, 0:1] = x0_true[:, :, 0:1]  # keep frame 0 fixed
        x0_hat = x - sig_next * v
        x0_hat[:, :, 0:1] = x0_true[:, :, 0:1]
    return x, x0_true


@torch.no_grad()
def sample_video_ar(trainer, batch, n_steps=8):
    """Faithful autoregressive diffusion-forcing i2v (chunk_size=1, frame-by-frame).
    Frame 0 is the condition; frame k (k=1..F-1) is denoised from noise over n_steps
    Euler steps while frames 0..k-1 are held CLEAN (the truly-generated past) and
    future frames are noise. The chunk-causal mask makes frame k attend only clean
    frames < k, so each frame conditions on the real generated past (no self-cond
    approximation). Returns (x_clean[B,48,F,H,W], x0_true)."""
    model = trainer.transformer
    sched = trainer.train_scheduler_latent
    input_dict = trainer._prepare_input_dict(batch)
    input_dict["chunk_size"] = 1  # each frame its own chunk
    input_dict["window_size"] = 256
    ld = input_dict["latent_dict"]
    x0_true = ld["latent"].clone()  # [B,48,F,H,W]
    B, C, F, H, W = x0_true.shape
    dev = x0_true.device
    tdtype = ld["timesteps"].dtype

    t0 = timestep_for_sigma(sched, 0.0)
    tmax = timestep_for_sigma(sched, sched.sigmas[0].item())
    # action/tactile as clean conditioning context
    ad = input_dict["action_dict"]
    ad["noisy_latents"] = ad["latent"].clone()
    ad["timesteps"] = torch.full_like(ad["timesteps"], t0)
    if (
        "tactile_global_noisy_latent" in ad
        and ad.get("tactile_global_clean_latent") is not None
    ):
        ad["tactile_global_noisy_latent"] = ad["tactile_global_clean_latent"].clone()
        if "tactile_global_timesteps" in ad:
            ad["tactile_global_timesteps"] = torch.full_like(
                ad["tactile_global_timesteps"], t0
            )

    full_sig = sched.sigmas
    idx = torch.linspace(0, len(full_sig) - 1, n_steps + 1).round().long()
    sigmas = full_sig[idx].tolist()  # ~1 -> ~0

    clean = x0_true.clone()  # holds generated clean frames
    # only frame 0 is truly known; init others to frame0 (overwritten as generated)
    clean[:, :, 1:] = x0_true[:, :, 0:1]
    cond_ts = torch.full((B, F), t0, device=dev, dtype=tdtype)

    for k in range(1, F):
        xk = torch.randn(B, C, 1, H, W, device=dev, dtype=x0_true.dtype)
        for i in range(n_steps):
            sig, sig_next = sigmas[i], sigmas[i + 1]
            # full sequence: past(<k)=clean, current(k)=xk@sig, future(>k)=noise
            noisy = clean.clone()
            noisy[:, :, k : k + 1] = xk
            if k + 1 < F:
                noisy[:, :, k + 1 :] = torch.randn_like(noisy[:, :, k + 1 :])
            ts = torch.full((B, F), t0, device=dev, dtype=tdtype)
            ts[:, k] = timestep_for_sigma(sched, sig)
            if k + 1 < F:
                ts[:, k + 1 :] = tmax
            ld["noisy_latents"] = noisy
            ld["timesteps"] = ts
            ld["cond_timesteps"] = cond_ts
            ld["latent"] = clean  # clean stream = generated past (true)
            pred = model(input_dict, train_mode=True)
            v = _v_dense(trainer, pred[0], x0_true)[:, :, k : k + 1]
            xk = xk + v * (sig_next - sig)
        clean[:, :, k : k + 1] = xk  # fix frame k as clean
    return clean, x0_true


def _v_dense(trainer, video_pred, ref_bcfhw):
    from utils import data_seq_to_patch

    return data_seq_to_patch(
        trainer.patch_size,
        video_pred,
        ref_bcfhw.shape[-3],
        ref_bcfhw.shape[-2],
        ref_bcfhw.shape[-1],
        batch_size=ref_bcfhw.shape[0],
    ).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="base")
    ap.add_argument(
        "--ckpt", required=True, help="path to .../checkpoint_step_N/transformer"
    )
    ap.add_argument("--n-batches", type=int, default=5)
    ap.add_argument(
        "--decode-check",
        action="store_true",
        help="decode the TRUE training video latent -> RGB (validate VAE decode in isolation)",
    )
    ap.add_argument(
        "--tactile-decode",
        action="store_true",
        help="try decoding the GlobalTactile latent -> RGB",
    )
    ap.add_argument(
        "--generate", action="store_true", help="i2v sampling -> decode -> frames"
    )
    ap.add_argument(
        "--teacher-force",
        action="store_true",
        help="clean-stream = true x0 (plumbing test); else free i2v (self-cond)",
    )
    ap.add_argument("--n-steps", type=int, default=30, help="Euler sampling steps")
    ap.add_argument(
        "--ar",
        action="store_true",
        help="faithful autoregressive frame-by-frame diffusion-forcing i2v",
    )
    ap.add_argument(
        "--gen-tactile",
        action="store_true",
        help="autoregressively GENERATE GlobalTactile (co-gen with video), "
        "decode, and save side-by-side [generated | true] compare video",
    )
    ap.add_argument(
        "--fps",
        type=int,
        default=15,
        help="mp4 playback fps (129 frames / fps = duration)",
    )
    ap.add_argument("--vae", default="/path/to/base-model/vae")
    ap.add_argument("--out-dir", default="/path/to/workspace/pretrained/render_out")
    ap.add_argument(
        "--val",
        action="store_true",
        help="use the VAL split (config.val_dataset_path) instead of train",
    )
    ap.add_argument(
        "--n-gen",
        type=int,
        default=1,
        help="number of samples to generate per run (--generate); one model load",
    )
    ap.add_argument("--metric-split", choices=("validation",), default="validation")
    ap.add_argument(
        "--seed", type=int, default=20260801, help="deterministic evaluation RNG seed"
    )
    args = ap.parse_args()
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive")
    if args.n_gen <= 0:
        raise ValueError("--n-gen must be positive")
    if args.gen_tactile and not args.val:
        raise ValueError("--gen-tactile Track 3.1 evaluation requires --val")

    config = TWAM_CONFIGS[args.config_name]
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.enable_wandb = False
    config.load_worker = 0
    dev = "cuda:0"
    seed_contract = None
    tactile_model_contract = None
    active_tactile_sensor_count = None
    loaded_tactile_runtime = None

    checkpoint_provenance = None
    if args.gen_tactile:
        seed_contract = set_evaluation_seed(args.seed)
        print(f"[render] deterministic RNG contract: {seed_contract}")
        config.tactile_cfg_prob = 0.0
        config.noisy_cond_prob_tactile = 0.0
        action_schema = getattr(config, "action_schema", None)
        if not isinstance(action_schema, str) or not action_schema:
            raise ValueError(
                "--gen-tactile provenance requires a config with action_schema"
            )
        tactile_model_contract = build_expected_tactile_model_contract(config)
        tactile_keys = getattr(config, "tactile_keys", None)
        if not isinstance(tactile_keys, (list, tuple)) or not tactile_keys:
            raise ValueError("--gen-tactile requires configured tactile sensor keys")
        active_tactile_sensor_count = len(tactile_keys)
        checkpoint_provenance = audit_evaluation_checkpoint(
            Path(args.ckpt),
            expected_action_dim=int(config.action_dim),
            expected_action_schema=action_schema,
            expected_model_contract=tactile_model_contract,
        )
        print(
            "[render] audited transformer SHA256: "
            f"{checkpoint_provenance['transformer_sha256']}"
        )

    if args.val:
        config.dataset_path = config.val_dataset_path
        print(f"[render] using VAL data: {config.dataset_path}")

    dataset_provenance = None
    decoder_provenance = None
    if args.gen_tactile:
        dataset_provenance = audit_evaluation_dataset(
            dataset_path=Path(config.dataset_path),
            manifest_path=Path(config.dataset_manifest_path),
            conversion_report_path=Path(config.conversion_report_path),
            normalizer_path=Path(config.norm_stat_path),
            expected_manifest_sha256=str(config.source_manifest_sha256),
            expected_normalizer_sha256=str(config.normalizer_sha256),
            base_model_path=Path(config.wan22_pretrained_model_name_or_path),
        )
        decoder_provenance = audit_vae_decoder(Path(args.vae))

    # put every render run in its own timestamped subfolder for easy browsing:
    # <out-dir>/<ckptstep>_<mode>_<YYYYmmdd_HHMMSS>/
    _stamp = time.strftime("%Y%m%d_%H%M%S")
    _run_nonce = f"{time.time_ns()}_{uuid.uuid4().hex[:12]}"
    _ckstep = "unknown"
    for _p in str(args.ckpt).split("/"):
        if _p.startswith("checkpoint_step_"):
            _ckstep = _p.replace("checkpoint_", "")
    _mode = (
        "gentac"
        if args.gen_tactile
        else (
            "video"
            if args.generate
            else (
                "tacdec"
                if args.tactile_decode
                else "deccheck" if args.decode_check else "consist"
            )
        )
    )
    args.out_dir = os.path.join(
        args.out_dir, f"{_ckstep}_{_mode}_{_stamp}_{_run_nonce}"
    )
    os.makedirs(args.out_dir, exist_ok=False)
    print(f"[render] output dir: {args.out_dir}")

    trainer = Trainer(config, inference_only=True)
    if not (args.decode_check or args.tactile_decode):
        model_checkpoint = (
            checkpoint_provenance["transformer_directory"]
            if checkpoint_provenance is not None
            else args.ckpt
        )
        print(f"[render] loading MoT checkpoint: {model_checkpoint}")
        trainer.transformer = load_mot_checkpoint(
            model_checkpoint, torch_dtype=torch.bfloat16, torch_device=dev
        )
        if checkpoint_provenance is not None:
            loaded_checkpoint_provenance = audit_evaluation_checkpoint(
                Path(model_checkpoint),
                expected_action_dim=int(config.action_dim),
                expected_action_schema=action_schema,
                expected_model_contract=tactile_model_contract,
            )
            if loaded_checkpoint_provenance != checkpoint_provenance:
                raise RuntimeError(
                    "checkpoint bytes changed while the evaluation model loaded"
                )
        trainer.transformer.eval()
        if tactile_model_contract is not None:
            loaded_tactile_runtime = audit_loaded_tactile_runtime(
                trainer, tactile_model_contract
            )
        print(
            "[render] expert structure:",
            {
                n: (
                    trainer.transformer.mot.experts[n].hidden_dim,
                    trainer.transformer.mot.experts[n].narrow,
                )
                for n in trainer.transformer.mot.expert_names
            },
        )

    ds = MultiLatentLeRobotDataset(config=config)
    dl = build_deterministic_eval_loader(ds)
    it = iter(dl)

    if args.tactile_decode:
        # Try to decode the GlobalTactile latent (per-sensor, WAN-VAE encoded 48ch
        # residual) back to RGB — user wasn't sure it's decodable.
        from models.utils import load_vae

        vae = load_vae(args.vae, torch_dtype=torch.bfloat16, torch_device=dev)
        batch = next(it)
        batch = trainer.convert_input_format(batch)
        gt = batch.get("tactile_global_latent")
        if gt is None:
            print("[tactile] no tactile_global_latent in batch")
            return
        if gt.dim() == 5:
            gt = gt.unsqueeze(0)
        B, S, C, F, H, W = gt.shape
        print(f"[tactile] global tactile latent shape: {tuple(gt.shape)} (B,S,C,F,H,W)")
        for s in range(S):
            vid = decode_video_latent(vae, gt[:, s].to(dev))  # [B,48,F,H,W] -> RGB
            save_frames(vid, args.out_dir, f"tactile_s{s}")
        print("TACTILE DECODE DONE")
        return

    if args.decode_check:
        # Stage 0 (lowest risk): decode the TRUE training video latent -> RGB, with
        # NO model / NO sampling. Validates the VAE decode + view layout in isolation.
        from models.utils import load_vae

        vae = load_vae(args.vae, torch_dtype=torch.bfloat16, torch_device=dev)
        batch = next(it)
        batch = trainer.convert_input_format(batch)
        input_dict = trainer._prepare_input_dict(batch)
        x0 = input_dict["latent_dict"]["latent"]  # [B, z, F, H, W] clean
        print(f"[decode-check] true video latent shape: {tuple(x0.shape)}")
        vid = decode_video_latent(vae, x0)
        save_frames(vid, args.out_dir, "true")
        print("DECODE CHECK DONE")
        return

    if args.generate:
        from models.utils import load_vae

        vae = load_vae(args.vae, torch_dtype=torch.bfloat16, torch_device=dev)
        for g in range(args.n_gen):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            batch = trainer.convert_input_format(batch)
            if args.ar:
                x_gen, x0_true = sample_video_ar(trainer, batch, n_steps=args.n_steps)
                tag = f"video_gen{g:02d}_ar"
            else:
                x_gen, x0_true = sample_video(
                    trainer,
                    batch,
                    n_steps=args.n_steps,
                    teacher_force=args.teacher_force,
                )
                tag = (
                    f"video_gen{g:02d}_tf"
                    if args.teacher_force
                    else f"video_gen{g:02d}_free"
                )
            mse = torch.nn.functional.mse_loss(x_gen.float(), x0_true.float()).item()
            # left = generated video, right = ground-truth video, caption on top
            save_compare_video(
                decode_video_latent(vae, x_gen),
                decode_video_latent(vae, x0_true),
                args.out_dir,
                tag,
                fps=args.fps,
                left_label="generated",
                right_label="ground truth",
                title=f"VIDEO  latentMSE={mse:.3f}",
            )
            print(f"[generate {g}] {tag} vs true latent MSE = {mse:.4f}")
        print("GENERATE DONE")
        return

    if args.gen_tactile:
        import av
        from models.utils import load_vae

        vae = load_vae(args.vae, torch_dtype=torch.bfloat16, torch_device=dev)
        loaded_decoder_provenance = audit_vae_decoder(Path(args.vae))
        if loaded_decoder_provenance != decoder_provenance:
            raise RuntimeError("VAE decoder bytes changed while the model loaded")
        sample_selection = []
        for g in range(args.n_gen):
            dataset_index = g % len(ds)
            loader_cycle = g // len(ds)
            sample_seed = set_sample_seed(args.seed, g)
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            batch = mask_future_local_tactile(batch)
            dataset_metadata = resolve_dataset_index_metadata(ds, dataset_index)
            selection = capture_source_sample_metadata(
                batch,
                output_sample_id=f"sample_{g:06d}",
                dataset_index=dataset_index,
                loader_cycle=loader_cycle,
                sample_seed=sample_seed,
                dataset_metadata=dataset_metadata,
            )
            tasks = selection["source_metadata"]["tasks"]
            if (
                not isinstance(tasks, list)
                or not tasks
                or not isinstance(tasks[0], str)
            ):
                raise ValueError("validation sample has no real task metadata")
            task_label = tasks[0]
            if Path(task_label).name != task_label or task_label in {"", ".", ".."}:
                raise ValueError("validation task is not one safe path component")
            trainer._tactile_cond_drop = False
            batch = trainer.convert_input_format(batch)
            trainer._tactile_cond_drop = False
            gt_gen, gt_true = sample_tactile_ar(trainer, batch, n_steps=args.n_steps)
            # gt_*: (B,S,C,F,H,W). Decode each sensor; stack sensors vertically.
            B, S, C, F, H, W = gt_true.shape
            import torch as _t

            mse = _t.nn.functional.mse_loss(gt_gen.float(), gt_true.float()).item()
            sensor_ids = batch["tactile_sensor_ids"].reshape(-1).tolist()
            if len(sensor_ids) != S:
                raise ValueError("tactile sensor IDs do not match generated streams")
            decoded_tactile_streams = []
            for s, sensor_id in enumerate(sensor_ids):
                vid_gen = decode_video_latent(vae, gt_gen[:, s].to(dev))  # gen residual
                vid_true = decode_video_latent(
                    vae, gt_true[:, s].to(dev)
                )  # true residual
                stream_contract = save_tactile_metric_frames(
                    vid_gen,
                    vid_true,
                    Path(args.out_dir),
                    task=task_label,
                    sample_id=f"sample_{g:06d}",
                    sensor=f"sensor_{int(sensor_id)}",
                    fps=args.fps,
                )
                stream_contract["sensor_id"] = int(sensor_id)
                decoded_tactile_streams.append(stream_contract)
                save_compare_video(
                    vid_gen,
                    vid_true,
                    args.out_dir,
                    f"tactile_gen{g:02d}_s{s}",
                    fps=args.fps,
                    left_label="generated",
                    right_label="true global-residual",
                    title=f"TACTILE global s{s}  latentMSE={mse:.3f}",
                )
            selection["decoded_tactile_streams"] = decoded_tactile_streams
            sample_selection.append(selection)
            print(
                f"[gen-tactile {g}] generated vs true GlobalTactile latent MSE = {mse:.4f}"
            )
        from n0_twam.evaluation.tactile_quality import (
            evaluate_tactile_prediction_quality,
        )

        report = evaluate_tactile_prediction_quality(
            prediction_root=Path(args.out_dir) / "prediction",
            ground_truth_root=Path(args.out_dir) / "ground_truth",
            skip_first_frames=1,
        )
        if checkpoint_provenance is None:
            raise RuntimeError("missing audited checkpoint provenance")
        if seed_contract is None:
            raise RuntimeError("missing deterministic RNG contract")
        if active_tactile_sensor_count is None or loaded_tactile_runtime is None:
            raise RuntimeError("missing tactile model/runtime contract")
        report = build_provenance_bound_report(
            report,
            checkpoint=checkpoint_provenance,
            dataset=dataset_provenance,
            decoder=decoder_provenance,
            runtime={
                "config_name": args.config_name,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device": dev,
                "mp4_writer": "pyav_h264_yuv420p_rgb24_input_verified_frame_count",
                "pyav_version": av.__version__,
                "fps": args.fps,
                "tactile_cfg_prob": float(config.tactile_cfg_prob),
                "noisy_cond_prob_tactile": float(config.noisy_cond_prob_tactile),
                "tactile_runtime_contract": loaded_tactile_runtime,
            },
            rng_contract=seed_contract,
            seed=args.seed,
            n_steps=args.n_steps,
            requested_n_gen=args.n_gen,
            max_latent_frames=int(config.max_latent_frames),
            expected_active_tactile_sensor_count=active_tactile_sensor_count,
            metric_split=args.metric_split,
            sample_selection=sample_selection,
        )
        write_json_atomic(Path(args.out_dir) / "tactile_quality.json", report)
        print(f"[gen-tactile] quality={report['overall']}")
        print("GEN TACTILE DONE")
        return

    agg = {"latent": [], "action": [], "tactile": [], "vrec": []}
    for i in range(args.n_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = trainer.convert_input_format(batch)
        input_dict = trainer._prepare_input_dict(batch)
        with torch.no_grad():
            pred = trainer.transformer(input_dict, train_mode=True)
            loss_dict = trainer.compute_loss(input_dict, pred)

            # video velocity -> reconstruct x0_hat = x_noisy - sigma * v, compare to clean x0
            video_pred = pred[0]  # [B, N_tok, C_patch]
            from utils import data_seq_to_patch

            ld = input_dict["latent_dict"]
            x0 = ld["latent"].float()  # [B,C,F,H,W] clean
            xN = ld["noisy_latents"].float()  # [B,C,F,H,W] noised
            v = data_seq_to_patch(
                trainer.patch_size,
                video_pred,
                x0.shape[-3],
                x0.shape[-2],
                x0.shape[-1],
                batch_size=x0.shape[0],
            ).float()
            sig = sigma_for_timesteps(
                trainer.train_scheduler_latent, ld["timesteps"].flatten()
            ).to(
                x0.device
            )  # [B*F]
            sig = sig.view(x0.shape[0], x0.shape[2], 1, 1).permute(0, 2, 1, 3)[
                ..., None
            ]  # [B,1,F,1,1]
            x0_hat = xN - sig * v
            vrec = torch.nn.functional.mse_loss(x0_hat, x0).item()

        la = loss_dict["latent_loss"].item()
        ac = loss_dict.get("action_loss", torch.zeros(())).item()
        ta = loss_dict.get("tactile_loss", torch.zeros(())).item()
        agg["latent"].append(la)
        agg["action"].append(ac)
        agg["tactile"].append(ta)
        agg["vrec"].append(vrec)
        print(
            f"[batch {i}] latent_loss={la:.4f} action_loss={ac:.4f} "
            f"tactile_loss={ta:.4f} | video x0-reconstruction MSE={vrec:.4f}"
        )

    n = len(agg["latent"])
    print("\n==== MEAN over %d batches ====" % n)
    for k in ("latent", "action", "tactile", "vrec"):
        print(f"  {k}: {sum(agg[k])/n:.4f}")
    print(
        "CONSISTENCY CHECK DONE (compare to the ~step-400 training losses; "
        "low = inference computes the same function as training)"
    )


if __name__ == "__main__":
    main()
