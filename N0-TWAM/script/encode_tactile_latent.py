#!/usr/bin/env python3
"""Encode LeRobot tactile videos into global and local Wan VAE latents."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.data.latent_inventory import (
    TRACK31_TACTILE_ENCODING_CONTRACT,
    TRACK31_TACTILE_KEYS,
    build_encoder_source_identity,
    build_track31_payload_provenance,
    build_track31_source_video_identity,
    finalize_latent_inventory,
    invalidate_latent_inventory,
    load_encoder_source_identity,
    load_expected_segments,
    validate_existing_track31_payload,
)
from n0_twam.data.video_decode import decode_sampled_video_frames
from n0_twam.integrations.univtac.artifact_contracts import (
    VerifiedTrack31Artifacts,
    VerifiedTrack31PhysicalArtifacts,
    verify_track31_physical_bundle,
    verify_track31_training_bundle,
)
from n0_twam.models.utils import load_vae
from script.track3_1.latent_sharding import (
    resolve_execution_device,
    select_episode_shard,
    validate_shard_contract,
)


@dataclass
class EpisodeRequest:
    episode_index: int
    length: int
    episode_task: str
    source_video: Path
    output_paths: dict  # {"global": Path, "local": Path}
    start_timestamp: float
    end_timestamp: float


FORMAL_PHYSICAL_SPLITS = ("train759", "frozen40")
LEGACY_LOGICAL_SPLITS = ("train", "validation")
TRACK31_ENCODER_SPLITS = LEGACY_LOGICAL_SPLITS + FORMAL_PHYSICAL_SPLITS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path containing 'vae/' subdir (use base-model)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="default: <dataset-root>/latents_tactile",
    )
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="must be multiple of 32 (VAE constraint)",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--tactile-keys",
        nargs="+",
        required=True,
        help="e.g. observation.images.tactile_a observation.images.tactile_b",
    )
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Formal Track 3.1 worker count; episodes are assigned by modulo.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based formal Track 3.1 worker index.",
    )
    parser.add_argument(
        "--defer-inventory",
        action="store_true",
        help="Worker mode: write payloads but leave ready inventory to one finalizer.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--encoder-source-identity-path",
        type=Path,
        default=None,
        help="Optional invocation-scoped encoder identity cache.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["global", "local", "both"],
        default="both",
        help="which residual streams to compute",
    )
    parser.add_argument(
        "--local-mode",
        choices=["current", "residual"],
        default="current",
        help="local stream: 'current'=the observed frame itself (default, "
        "absolute, streaming-robust; input observation for the "
        "use_local_tactile post-train branch); 'residual'="
        "frame[t]-frame[t-1] (legacy, fragile under streaming)",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Track 3.1 artifact directory; requires --split and enables a "
        "content-addressed completion inventory.",
    )
    parser.add_argument(
        "--split",
        choices=TRACK31_ENCODER_SPLITS,
        default=None,
        help=(
            "Physical Track 3.1 repo (train759/frozen40). The legacy "
            "train/validation values remain available for old artifacts."
        ),
    )
    return parser.parse_args(argv)


def atomic_torch_save(
    payload: object,
    output_path: Path,
    *,
    shard_index: int | None,
) -> None:
    """Atomically publish one payload using a cross-container-unique temp name."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_label = "single" if shard_index is None else str(shard_index)
    temporary_path = output_path.parent / (
        f".{output_path.name}.shard-{shard_label}.pid-{os.getpid()}."
        f"{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_track31_encoder_contract(
    *,
    artifact_root: Path | None,
    split: str | None,
    dataset_root: Path,
    output_root: Path,
    mode: str,
    local_mode: str,
    target_fps: int,
    height: int,
    width: int,
    dtype: str,
    tactile_keys: list[str],
    execution_device: str,
) -> VerifiedTrack31Artifacts | VerifiedTrack31PhysicalArtifacts | None:
    """Validate the tactile encoder boundary before any output write."""

    if (artifact_root is None) != (split is None):
        raise ValueError("--artifact-root and --split must be provided together")
    if artifact_root is None:
        return None
    if split not in TRACK31_ENCODER_SPLITS:
        raise ValueError(f"unsupported Track 3.1 encoder split: {split!r}")
    if dataset_root.name != split:
        raise ValueError(f"dataset root/split mismatch: {dataset_root} vs {split!r}")
    canonical_output_root = dataset_root / "latents_tactile"
    if output_root != canonical_output_root:
        raise ValueError(
            "Track 3.1 inventory requires the standard "
            "<dataset>/latents_tactile root"
        )
    if canonical_output_root.is_symlink():
        raise ValueError("Track 3.1 output root cannot be a symlink")
    try:
        canonical_output_root.resolve(strict=False).relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("Track 3.1 output root escaped the selected dataset") from exc
    if mode != "both" or local_mode != "current":
        raise ValueError(
            "Track 3.1 inventory requires --mode both --local-mode current"
        )

    resolved_artifact_root = artifact_root.resolve(strict=True)
    if split in FORMAL_PHYSICAL_SPLITS:
        try:
            vae_device = torch.device(execution_device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError("formal tactile encoding requires HCU cuda:0") from exc
        if vae_device != torch.device("cuda:0"):
            raise ValueError("formal tactile encoding requires HCU cuda:0")
        actual_contract = {
            "height": height,
            "width": width,
            "target_fps": target_fps,
            "compute_dtype": dtype,
            "tactile_keys": list(tactile_keys),
            "mode": mode,
            "local_mode": local_mode,
            "execution_device_policy": "explicit_no_fallback_v1",
            "vae_device_policy": "logical_cuda0_only_v1",
        }
        expected_contract = {
            key: value
            for key, value in TRACK31_TACTILE_ENCODING_CONTRACT.items()
            if key != "schema_version"
        }
        if actual_contract != expected_contract:
            raise ValueError(
                "formal Track 3.1 tactile encoding requires 128x128, 10 FPS, "
                "bf16, exact tactile_a/tactile_b streams, and both/current mode"
            )
        return verify_track31_physical_bundle(
            manifest_path=resolved_artifact_root / "universe_manifest_v4.json",
            conversion_report_path=resolved_artifact_root / "conversion_report.json",
            dataset_root=dataset_root,
            physical_split=split,
        )

    # Explicit compatibility branch for pre-schema-v4 train/validation repos.
    return verify_track31_training_bundle(
        manifest_path=resolved_artifact_root / "dataset_manifest.json",
        normalizer_path=resolved_artifact_root / "qpos8_normalizer.json",
        conversion_report_path=resolved_artifact_root / "conversion_report.json",
        dataset_root=dataset_root.parent,
    )


def require_contained_output_path(
    *,
    path: Path,
    output_root: Path,
    dataset_root: Path,
) -> None:
    """Reject symlink traversal before creating or replacing one latent file."""

    try:
        relative_path = path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            "latent output path escaped the canonical output root"
        ) from exc
    current = output_root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"latent output path contains a symlink: {current}")
        if not current.exists():
            break
    try:
        path.resolve(strict=False).relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("latent output path escaped the selected dataset") from exc


def require_symlink_free_output_tree(
    *,
    output_root: Path,
    dataset_root: Path,
) -> None:
    """Reject a pre-existing symlink anywhere below the canonical output root."""

    if not output_root.exists():
        return
    for candidate in output_root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"latent output tree contains a symlink: {candidate}")
        try:
            candidate.resolve(strict=True).relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError("latent output tree escaped the selected dataset") from exc


def get_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_info(dataset_root: Path) -> dict:
    with (dataset_root / "meta" / "info.json").open() as f:
        return json.load(f)


def load_episode_table(dataset_root: Path) -> pd.DataFrame:
    episode_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if episode_files:
        return pd.concat([pd.read_parquet(p) for p in episode_files], ignore_index=True)
    jsonl_path = dataset_root / "meta" / "episodes.jsonl"
    if jsonl_path.exists():
        records = [
            json.loads(line) for line in jsonl_path.read_text().strip().split("\n")
        ]
        return pd.DataFrame(records)
    raise FileNotFoundError(f"No episode metadata under {dataset_root / 'meta'}")


def build_source_video(
    dataset_root: Path,
    video_key: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    return (
        dataset_root
        / "videos"
        / video_key
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def normalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device).view(
        1, -1, 1, 1, 1
    )
    latents_std = torch.tensor(vae.config.latents_std, device=latents.device).view(
        1, -1, 1, 1, 1
    )
    return ((latents.float() - latents_mean) * (1.0 / latents_std)).to(latents.dtype)


def extract_frames(
    request: EpisodeRequest,
    width: int,
    height: int,
    target_fps: int,
    ori_fps: int,
) -> tuple[np.ndarray, list[int]]:
    """Decode and resize; return uint8 ``(N,H,W,3)`` frames and frame IDs."""
    return decode_sampled_video_frames(
        source_video=request.source_video,
        start_timestamp=request.start_timestamp,
        end_timestamp=request.end_timestamp,
        length=request.length,
        width=width,
        height=height,
        target_fps=target_fps,
        ori_fps=ori_fps,
    )


def compute_residuals(
    frames_u8: np.ndarray, mode: str, local_mode: str = "current"
) -> dict[str, np.ndarray]:
    """Compute global and/or local tactile streams in float32, normalized to [-1, 1].

    frames_u8: (N, H, W, 3) uint8 in [0, 255]
    Returns dict { "global": (N, H, W, 3) float32 in [-1, 1], "local": ... }

    local_mode:
      - "residual" (legacy): local[t] = frame[t] - frame[t-1] (high-freq inter-frame
        delta). Fragile under streaming inference (needs the contiguous prev frame).
      - "current": local[t] = the current frame itself, mapped [0,255]->[-1,1]. The
        CURRENT tactile latent (absolute), not a delta — robust to streaming (server
        just encodes the current frame, no prev_frames bookkeeping). Pair with the
        use_local_tactile post-train switch.
    """
    # int16 to avoid uint8 wrap-around in subtraction
    frames_i16 = frames_u8.astype(np.int16)
    out: dict[str, np.ndarray] = {}
    if mode in ("global", "both"):
        # global[t] = frame[t] - frame[0]   ∈ [-255, 255]
        # normalize by /255 → [-1, 1]
        residual = (frames_i16 - frames_i16[0:1]).astype(np.float32) / 255.0
        out["global"] = residual
    if mode in ("local", "both"):
        if local_mode == "current":
            # local[t] = current frame -> [-1, 1] (VAE input range), no delta.
            out["local"] = frames_u8.astype(np.float32) / 127.5 - 1.0
        else:
            # local[t] = frame[t] - frame[t-1]; first frame residual = 0
            residual = np.zeros_like(frames_i16, dtype=np.float32)
            if frames_i16.shape[0] > 1:
                residual[1:] = (frames_i16[1:] - frames_i16[:-1]).astype(
                    np.float32
                ) / 255.0
            out["local"] = residual
    return out


def encode_residual_to_latent(
    residual_frames: np.ndarray,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int, int]:
    """Encode a (N, H, W, 3) residual array to latent.

    residual_frames values are already in [-1, 1] (same range Wan VAE expects).
    Returns (flat_latent (F*H*W, C), latent_F, latent_H, latent_W).
    """
    # Convert to (C, N, H, W) → unsqueeze batch → (1, C, N, H, W)
    video = torch.from_numpy(residual_frames).permute(3, 0, 1, 2).unsqueeze(0)
    # already in [-1, 1] range from compute_residuals, no further normalize needed
    with torch.no_grad():
        posterior = vae.encode(video.to(device=device, dtype=dtype)).latent_dist
        latents = normalize_latents(posterior.mean, vae)[0]  # (C, F, H, W)
    lat_F, lat_H, lat_W = latents.shape[1:]
    flat = rearrange(latents, "c f h w -> (f h w) c").contiguous().cpu()
    return flat, lat_F, lat_H, lat_W


def build_requests(
    dataset_root: Path,
    output_root: Path,
    episodes_df: pd.DataFrame,
    tactile_key: str,
    info: dict,
    is_v21: bool,
    modes: list[str],
) -> list[EpisodeRequest]:
    requests = []
    chunks_size = info.get("chunks_size", 1000)
    total_chunks = int(info.get("total_chunks", 1))
    ori_fps = int(round(info["fps"]))
    for row in episodes_df.to_dict(orient="records"):
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        tasks = row.get("tasks")
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        if isinstance(tasks, (list, tuple)) and len(tasks) == 1:
            episode_task = str(tasks[0])
        elif isinstance(tasks, str) and tasks:
            episode_task = tasks
        else:
            raise ValueError(
                f"episode {episode_index} must expose exactly one frozen task"
            )
        if is_v21:
            # single-chunk repos (converter -> chunk-000, total_chunks=1) must
            # use chunk-000; naive idx//size sends ep>=1000 to a missing
            # chunk-001 and crashes (the giant-repo "wall"). See video encoder.
            chunk_index = 0 if total_chunks <= 1 else episode_index // chunks_size
            source_video = (
                dataset_root
                / "videos"
                / f"chunk-{chunk_index:03d}"
                / tactile_key
                / f"episode_{episode_index:06d}.mp4"
            )
            start_ts = 0.0
            end_ts = length / ori_fps
        else:
            chunk_index = int(row[f"videos/{tactile_key}/chunk_index"])
            file_index = int(row[f"videos/{tactile_key}/file_index"])
            source_video = build_source_video(
                dataset_root, tactile_key, chunk_index, file_index
            )
            start_ts = float(row[f"videos/{tactile_key}/from_timestamp"])
            end_ts = float(row[f"videos/{tactile_key}/to_timestamp"])

        output_paths = {
            mode: (
                output_root
                / mode
                / f"chunk-{chunk_index:03d}"
                / tactile_key
                / f"episode_{episode_index:06d}_0_{length}.pth"
            )
            for mode in modes
        }
        requests.append(
            EpisodeRequest(
                episode_index=episode_index,
                length=length,
                episode_task=episode_task,
                source_video=source_video,
                output_paths=output_paths,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
            )
        )
    return requests


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    model_path = args.model_path.resolve()
    output_root = Path(
        os.path.abspath(args.output_root or (dataset_root / "latents_tactile"))
    )
    validate_shard_contract(
        split=args.split,
        artifact_root=args.artifact_root,
        episodes=args.episodes,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        defer_inventory=args.defer_inventory,
    )
    verified_artifacts = verify_track31_encoder_contract(
        artifact_root=args.artifact_root,
        split=args.split,
        dataset_root=dataset_root,
        output_root=output_root,
        mode=args.mode,
        local_mode=args.local_mode,
        target_fps=args.target_fps,
        height=args.height,
        width=args.width,
        dtype=args.dtype,
        tactile_keys=args.tactile_keys,
        execution_device=args.device,
    )
    encoder_source_identity = None
    if verified_artifacts is not None:
        load_expected_segments(dataset_root)
    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError(
            "Height and width must both be multiples of 32 (Wan VAE constraint)"
        )

    info = load_info(dataset_root)
    ori_fps = int(round(info["fps"]))
    if args.split in FORMAL_PHYSICAL_SPLITS:
        available_tactile_keys = tuple(
            key
            for key, value in info.get("features", {}).items()
            if isinstance(value, dict)
            and value.get("dtype") in ("video", "image")
            and "tactile" in key
        )
        if available_tactile_keys != TRACK31_TACTILE_KEYS:
            raise ValueError(
                "formal Track 3.1 dataset must expose exact tactile_a/tactile_b keys"
            )
    if verified_artifacts is not None:
        require_symlink_free_output_tree(
            output_root=output_root,
            dataset_root=dataset_root,
        )

    # Filter episodes
    episodes_df = load_episode_table(dataset_root)
    if args.num_shards is not None:
        episodes_df = select_episode_shard(
            episodes_df,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
    elif args.episodes is not None:
        episodes_df = episodes_df[
            episodes_df["episode_index"].isin(args.episodes)
        ].copy()
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)
    if episodes_df.empty:
        if args.num_shards is not None:
            print(
                f"tactile shard {args.shard_index}/{args.num_shards} has no episodes; "
                "nothing to encode"
            )
            return
        raise ValueError("No episodes selected")

    if verified_artifacts is not None:
        if args.encoder_source_identity_path is None:
            encoder_source_identity = build_encoder_source_identity(model_path)
        else:
            encoder_source_identity = load_encoder_source_identity(
                args.encoder_source_identity_path
            )
    output_root.mkdir(parents=True, exist_ok=True)

    # Detect dataset format
    first_key = args.tactile_keys[0]
    is_v21 = f"videos/{first_key}/chunk_index" not in episodes_df.columns

    # Set up VAE
    device_name = resolve_execution_device(
        args.device,
        accelerator_available=torch.cuda.is_available(),
    )
    device = torch.device(device_name)
    if args.split in FORMAL_PHYSICAL_SPLITS and (
        device != torch.device("cuda:0")
        or torch.cuda.device_count() != 1
        or torch.cuda.current_device() != 0
    ):
        raise RuntimeError(
            "formal tactile workers require exactly one visible HCU as cuda:0"
        )
    dtype = get_dtype(args.dtype)
    vae = load_vae(model_path / "vae", torch_dtype=dtype, torch_device=device)
    loaded_vae_device = next(vae.parameters()).device
    if args.split in FORMAL_PHYSICAL_SPLITS and loaded_vae_device != device:
        raise RuntimeError("formal tactile VAE parameters were not loaded on HCU")
    print(f"TACTILE_HCU_READY vae={loaded_vae_device}", flush=True)
    # All read-only validation/model loading succeeded. Invalidate the old
    # ready marker immediately before latent payloads may be changed.
    try:
        invalidate_latent_inventory(dataset_root, "tactile")
    except FileNotFoundError:
        # Multiple formal workers may race while deleting one old marker.
        pass

    modes = ["global", "local"] if args.mode == "both" else [args.mode]

    # Process each tactile key
    for tactile_key in args.tactile_keys:
        print(f"\n=== Processing tactile_key={tactile_key} ===")
        requests = build_requests(
            dataset_root, output_root, episodes_df, tactile_key, info, is_v21, modes
        )
        for req in requests:
            if verified_artifacts is not None:
                for output_path in req.output_paths.values():
                    require_contained_output_path(
                        path=output_path,
                        output_root=output_root,
                        dataset_root=dataset_root,
                    )
            provenances: dict[str, dict[str, object]] = {}
            if args.split in FORMAL_PHYSICAL_SPLITS:
                if encoder_source_identity is None or verified_artifacts is None:
                    raise RuntimeError(
                        "formal Track 3.1 provenance sources were not initialized"
                    )
                conversion_sha256 = verified_artifacts.conversion_report_sha256
                if conversion_sha256 is None:
                    raise ValueError(
                        "verified Track 3.1 bundle has no conversion digest"
                    )
                source_video_identity = build_track31_source_video_identity(
                    dataset_root,
                    req.source_video,
                )
                for mode in modes:
                    provenances[mode] = build_track31_payload_provenance(
                        split=args.split,
                        manifest_sha256=verified_artifacts.manifest_sha256,
                        conversion_report_sha256=conversion_sha256,
                        encoder_source_identity_sha256=str(
                            encoder_source_identity["identity_sha256"]
                        ),
                        kind="tactile",
                        episode_index=req.episode_index,
                        start_frame=0,
                        end_frame=req.length,
                        episode_task=req.episode_task,
                        stream_key=tactile_key,
                        tactile_mode=mode,
                        prompt=None,
                        source_video_identity=source_video_identity,
                        execution_device=str(device),
                        text_encoder_device=None,
                    )
                if not args.overwrite:
                    for mode, output_path in req.output_paths.items():
                        if not output_path.exists():
                            continue
                        try:
                            validate_existing_track31_payload(
                                dataset_root,
                                output_path,
                                kind="tactile",
                                expected_encoder_source_identity_sha256=str(
                                    encoder_source_identity["identity_sha256"]
                                ),
                                expected_provenance=provenances[mode],
                            )
                        except Exception as exc:
                            raise ValueError(
                                "existing formal tactile latent is not reusable; "
                                "re-run this shard with --overwrite to re-encode: "
                                f"{output_path}"
                            ) from exc
            # Skip if all outputs exist and not overwriting
            if not args.overwrite and all(
                p.exists() for p in req.output_paths.values()
            ):
                print(f"  [skip] episode {req.episode_index} (all modes present)")
                continue

            if not req.source_video.exists():
                print(f"  [warn] missing source video {req.source_video}")
                continue

            # 1. extract frames
            frames_u8, local_frame_ids = extract_frames(
                req,
                args.width,
                args.height,
                args.target_fps,
                ori_fps,
            )

            # 2. compute residuals
            residuals = compute_residuals(
                frames_u8, args.mode, local_mode=args.local_mode
            )

            # 3. encode each residual stream
            for mode, residual_arr in residuals.items():
                out_path = req.output_paths[mode]
                if not args.overwrite and out_path.exists():
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                flat_latent, lat_F, lat_H, lat_W = encode_residual_to_latent(
                    residual_arr, vae, device, dtype
                )
                payload = {
                    "latent": flat_latent.to(torch.bfloat16),
                    "latent_num_frames": int(lat_F),
                    "latent_height": int(lat_H),
                    "latent_width": int(lat_W),
                    "frame_ids": local_frame_ids,
                    "start_frame": 0,
                    "end_frame": int(req.length),
                    "fps": int(args.target_fps),
                    "ori_fps": int(ori_fps),
                    "tactile_residual_mode": mode,
                    "tactile_local_mode": (
                        args.local_mode if mode == "local" else None
                    ),
                    "encoder_source_identity_sha256": (
                        None
                        if encoder_source_identity is None
                        else encoder_source_identity["identity_sha256"]
                    ),
                    "tactile_resize_h": int(args.height),
                    "tactile_resize_w": int(args.width),
                    "track31_encoding_contract": (
                        TRACK31_TACTILE_ENCODING_CONTRACT
                        if args.split in FORMAL_PHYSICAL_SPLITS
                        else None
                    ),
                    "track31_provenance": provenances.get(mode),
                }
                atomic_torch_save(
                    payload,
                    out_path,
                    shard_index=args.shard_index,
                )
                print(
                    f"  [ok] episode {req.episode_index} {mode} -> {out_path.name} "
                    f"latent shape=({lat_F},{lat_H},{lat_W})"
                )

    if verified_artifacts is not None and not args.defer_inventory:
        if encoder_source_identity is None:
            raise RuntimeError("Track 3.1 encoder source identity was not initialized")
        conversion_sha256 = verified_artifacts.conversion_report_sha256
        if conversion_sha256 is None:
            raise ValueError("verified Track 3.1 bundle has no conversion digest")
        inventory = finalize_latent_inventory(
            dataset_root,
            kind="tactile",
            split=args.split,
            manifest_sha256=verified_artifacts.manifest_sha256,
            conversion_report_sha256=conversion_sha256,
            encoder_source_identity=encoder_source_identity,
        )
        print(f"tactile latent inventory ready: {inventory['inventory_sha256']}")
    elif args.defer_inventory:
        print(
            f"tactile shard {args.shard_index}/{args.num_shards} complete; "
            "ready inventory intentionally deferred"
        )
    print(f"\nDone. Outputs under: {output_root}")


if __name__ == "__main__":
    main()
