#!/usr/bin/env python3
"""Encode LeRobot video episodes into Wan2.2 VAE latents.

It reads the LeRobot v2.1 episode parquet metadata, samples/resizes frames for
each episode, encodes the sampled clip with the Wan2.2 VAE, and saves one
`.pth` file per episode/view under `latents/`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.data.latent_inventory import (
    TRACK31_VIDEO_ENCODING_CONTRACT,
    TRACK31_VIDEO_KEYS,
    build_encoder_source_identity,
    build_track31_payload_provenance,
    build_track31_source_video_identity,
    canonical_bytes,
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
from n0_twam.models.utils import load_text_encoder, load_tokenizer, load_vae
from script.track3_1.latent_sharding import (
    resolve_execution_device,
    select_episode_shard,
    validate_shard_contract,
)
from script.track3_1.hcu_prompt_encoding import (
    build_text_embedding_cache,
    clean_episode_task as clean_prompt,
    load_prompt_embedding_cache,
    normalize_prompt,
    resolve_episode_prompt,
)
from script.track3_1.video_encoder_cli import (
    FORMAL_PHYSICAL_SPLITS,
    TRACK31_ENCODER_SPLITS,
    parse_video_encoder_args,
)


@dataclass
class EpisodeRequest:
    episode_index: int
    length: int
    prompt: str
    source_video: Path
    output_path: Path
    start_timestamp: float
    end_timestamp: float
    provenance: dict[str, object] | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return parse_video_encoder_args(argv)


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
    write_episodes_jsonl: bool,
    target_fps: int,
    height: int,
    width: int,
    max_sequence_length: int,
    dtype: str,
    video_keys: list[str] | None,
    prompt: str | None,
    execution_device: str,
    text_encoder_device: str,
) -> VerifiedTrack31Artifacts | VerifiedTrack31PhysicalArtifacts | None:
    """Validate the encoder's data/artifact boundary before any output write.

    Formal physical encoding deliberately reads only the schema-v4 universe
    manifest, the conversion report, and the selected physical repository. It
    never falls back to the legacy manifest/normalizer pair and never verifies
    the sibling physical repository.
    """

    if (artifact_root is None) != (split is None):
        raise ValueError("--artifact-root and --split must be provided together")
    if artifact_root is None:
        return None
    if split not in TRACK31_ENCODER_SPLITS:
        raise ValueError(f"unsupported Track 3.1 encoder split: {split!r}")
    if dataset_root.name != split:
        raise ValueError(f"dataset root/split mismatch: {dataset_root} vs {split!r}")
    if write_episodes_jsonl:
        raise ValueError(
            "Track 3.1 encoding cannot rewrite conversion-frozen episodes.jsonl"
        )
    canonical_output_root = dataset_root / "latents"
    if output_root != canonical_output_root:
        raise ValueError(
            "Track 3.1 inventory requires the standard <dataset>/latents root"
        )
    if canonical_output_root.is_symlink():
        raise ValueError("Track 3.1 output root cannot be a symlink")
    try:
        canonical_output_root.resolve(strict=False).relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("Track 3.1 output root escaped the selected dataset") from exc

    resolved_artifact_root = artifact_root.resolve(strict=True)
    if split in FORMAL_PHYSICAL_SPLITS:
        try:
            vae_device = torch.device(execution_device)
            umt5_device = torch.device(text_encoder_device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError("formal Track 3.1 requires explicit CUDA devices") from exc
        if (
            vae_device.type != "cuda"
            or umt5_device.type != "cuda"
            or vae_device.index is None
            or umt5_device.index is None
            or vae_device != umt5_device
            or vae_device != torch.device("cuda:0")
        ):
            raise ValueError(
                "formal Track 3.1 requires umT5 and VAE on the same HCU "
                "using an explicit CUDA index"
            )
        actual_contract = {
            "height": height,
            "width": width,
            "target_fps": target_fps,
            "max_sequence_length": max_sequence_length,
            "compute_dtype": dtype,
            "video_keys": list(
                TRACK31_VIDEO_KEYS if video_keys is None else video_keys
            ),
            "prompt_source": (
                "frozen_episode_task" if prompt is None else "cli_override"
            ),
            "execution_device_policy": "explicit_no_fallback_v1",
            "text_encoder_device_policy": "dedicated_hcu_prompt_precompute_v1",
            "text_encoder_model_class": "UMT5EncoderModel",
            "text_encoder_load_policy": "low_cpu_mem_device_map_v1",
            "text_tokenizer_class": "T5TokenizerFast",
            "prompt_clean_policy": "diffusers_wan_prompt_clean_v1",
            "text_encoder_forward_policy": "all_unique_max8_eval_inference_v1",
            "text_encoder_lifecycle": ("shared_read_only_cache_before_vae_workers_v1"),
            "prompt_embedding_cache_schema": 1,
        }
        expected_contract = {
            key: value
            for key, value in TRACK31_VIDEO_ENCODING_CONTRACT.items()
            if key != "schema_version"
        }
        if actual_contract != expected_contract:
            raise ValueError(
                "formal Track 3.1 video encoding requires 256x256, 10 FPS, "
                "max sequence length 512, bf16, exact top/wrist streams, "
                "HCU-only cached umT5/VAE execution, and frozen episode prompts"
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
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def load_info(dataset_root: Path) -> dict:
    with (dataset_root / "meta" / "info.json").open() as f:
        return json.load(f)


def get_video_keys(info: dict) -> list[str]:
    # RGB cameras only — tactile streams are encoded by encode_tactile_latent.py
    # with their own global/local semantics, not the RGB pipeline.
    return [
        key
        for key, value in info["features"].items()
        if isinstance(value, dict)
        and value.get("dtype") in ("video", "image")
        and "image" in key
        and "tactile" not in key
    ]


def resize_frame(frame: torch.Tensor | None, width: int, height: int):
    if frame is None:
        raise ValueError("Missing frame")
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized


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


def load_episode_table(dataset_root: Path) -> pd.DataFrame:
    # Try v3.0 parquet format first
    episode_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if episode_files:
        df = pd.concat(
            [pd.read_parquet(path) for path in episode_files], ignore_index=True
        )
        # Some converters write the v3 episodes parquet without a `tasks`
        # column; recover it from the v2.1 episodes.jsonl when present so the
        # per-episode prompt lookup below keeps working (else use --prompt).
        jsonl_path = dataset_root / "meta" / "episodes.jsonl"
        if "tasks" not in df.columns and jsonl_path.exists():
            records = [
                json.loads(line) for line in jsonl_path.read_text().strip().split("\n")
            ]
            tasks_by_ep = {r["episode_index"]: r.get("tasks") for r in records}
            df["tasks"] = df["episode_index"].map(tasks_by_ep)
        return df
    # Fallback to v2.1 jsonl format
    jsonl_path = dataset_root / "meta" / "episodes.jsonl"
    if jsonl_path.exists():
        records = [
            json.loads(line) for line in jsonl_path.read_text().strip().split("\n")
        ]
        return pd.DataFrame(records)
    raise FileNotFoundError(f"No episode metadata found under {dataset_root / 'meta'}")


def normalize_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device).view(
        1, -1, 1, 1, 1
    )
    latents_std = torch.tensor(vae.config.latents_std, device=latents.device).view(
        1, -1, 1, 1, 1
    )
    return ((latents.float() - latents_mean) * (1.0 / latents_std)).to(latents.dtype)


def ensure_complete(frames: Iterable) -> None:
    if any(frame is None for frame in frames):
        raise RuntimeError(
            "Missing decoded frame while collecting sampled video frames"
        )


def extract_episode_frames(
    request: EpisodeRequest,
    width: int,
    height: int,
    target_fps: int,
    ori_fps: int,
) -> tuple[list[np.ndarray], list[int]]:
    frames, local_frame_ids = decode_sampled_video_frames(
        source_video=request.source_video,
        start_timestamp=request.start_timestamp,
        end_timestamp=request.end_timestamp,
        length=request.length,
        width=width,
        height=height,
        target_fps=target_fps,
        ori_fps=ori_fps,
    )
    return list(frames), local_frame_ids


def encode_video(
    frames: list,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int, int]:
    ensure_complete(frames)
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    video = video.to(torch.float32) / 127.5 - 1.0
    with torch.no_grad():
        posterior = vae.encode(video.to(device=device, dtype=dtype)).latent_dist
        latents = normalize_latents(posterior.mean, vae)[0]
    latent_num_frames, latent_height, latent_width = latents.shape[1:]
    flat_latent = rearrange(latents, "c f h w -> (f h w) c").contiguous().cpu()
    return flat_latent, latent_num_frames, latent_height, latent_width


def maybe_write_episodes_jsonl(dataset_root: Path, episodes_df: pd.DataFrame) -> None:
    output_path = dataset_root / "meta" / "episodes.jsonl"
    lines = []
    for row in episodes_df.to_dict(orient="records"):
        prompt = clean_prompt(row["tasks"])
        item = {
            "episode_index": int(row["episode_index"]),
            "tasks": list(row["tasks"]),
            "length": int(row["length"]),
            "action_config": [
                {
                    "start_frame": 0,
                    "end_frame": int(row["length"]),
                    "action_text": prompt,
                }
            ],
        }
        lines.append(json.dumps(item, ensure_ascii=True))
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    model_path = args.model_path.resolve()
    output_root = Path(os.path.abspath(args.output_root or (dataset_root / "latents")))
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
        write_episodes_jsonl=args.write_episodes_jsonl,
        target_fps=args.target_fps,
        height=args.height,
        width=args.width,
        max_sequence_length=args.max_sequence_length,
        dtype=args.dtype,
        video_keys=args.video_keys,
        prompt=args.prompt,
        execution_device=args.device,
        text_encoder_device=args.text_encoder_device,
    )
    encoder_source_identity = None
    formal_episode_prompts: list[str] | None = None
    if verified_artifacts is not None:
        expected_segments = load_expected_segments(dataset_root)
        if args.split in FORMAL_PHYSICAL_SPLITS:
            formal_episode_prompts = [
                segment.episode_task for segment in expected_segments
            ]
    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError("Height and width must both be multiples of 32")

    info = load_info(dataset_root)
    ori_fps = int(round(info["fps"]))
    all_video_keys = get_video_keys(info)
    video_keys = args.video_keys or all_video_keys
    if args.split in FORMAL_PHYSICAL_SPLITS and tuple(video_keys) != TRACK31_VIDEO_KEYS:
        raise ValueError(
            "formal Track 3.1 dataset must expose exact top/wrist RGB keys"
        )

    missing_keys = sorted(set(video_keys) - set(all_video_keys))
    if missing_keys:
        raise ValueError(f"Unknown video keys: {missing_keys}")
    if verified_artifacts is not None:
        require_symlink_free_output_tree(
            output_root=output_root,
            dataset_root=dataset_root,
        )

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
                f"video shard {args.shard_index}/{args.num_shards} has no episodes; "
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

    if args.write_episodes_jsonl and args.episodes is not None:
        raise ValueError(
            "--write-episodes-jsonl cannot be combined with --episodes "
            "(it would rewrite meta/episodes.jsonl with only the selected subset)"
        )
    if args.write_episodes_jsonl:
        invalidate_latent_inventory(dataset_root, "video")
        maybe_write_episodes_jsonl(dataset_root, episodes_df)

    device_name = resolve_execution_device(
        args.device,
        accelerator_available=torch.cuda.is_available(),
    )
    device = torch.device(device_name)
    if args.text_encoder_device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("A non-CPU text encoder device requires an accelerator")
    text_encoder_device = torch.device(args.text_encoder_device)
    if args.split in FORMAL_PHYSICAL_SPLITS and (
        device.type != "cuda"
        or text_encoder_device.type != "cuda"
        or device.index is None
        or text_encoder_device.index is None
        or device != text_encoder_device
    ):
        raise RuntimeError("formal Track 3.1 umT5 and VAE must execute on the same HCU")
    if args.split in FORMAL_PHYSICAL_SPLITS and (
        torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0
    ):
        raise RuntimeError(
            "formal Track 3.1 workers require exactly one visible HCU as cuda:0"
        )
    dtype = get_dtype(args.dtype)
    episode_rows = episodes_df.to_dict(orient="records")
    episode_prompts = [resolve_episode_prompt(row, args.prompt) for row in episode_rows]
    if args.split in FORMAL_PHYSICAL_SPLITS:
        if args.prompt_embedding_cache_path is None:
            raise ValueError(
                "formal Track 3.1 workers require --prompt-embedding-cache-path"
            )
        if (
            encoder_source_identity is None
            or verified_artifacts is None
            or formal_episode_prompts is None
        ):
            raise RuntimeError("formal prompt cache provenance was not initialized")
        conversion_sha256 = verified_artifacts.conversion_report_sha256
        if conversion_sha256 is None:
            raise ValueError("formal prompt cache requires a conversion digest")
        text_cache = load_prompt_embedding_cache(
            args.prompt_embedding_cache_path,
            expected_prompts=formal_episode_prompts,
            expected_split=str(args.split),
            expected_manifest_sha256=verified_artifacts.manifest_sha256,
            expected_conversion_report_sha256=conversion_sha256,
            expected_encoder_source_identity_sha256=str(
                encoder_source_identity["identity_sha256"]
            ),
            expected_encoding_contract_sha256=hashlib.sha256(
                canonical_bytes(TRACK31_VIDEO_ENCODING_CONTRACT)
            ).hexdigest(),
            max_sequence_length=args.max_sequence_length,
        )
        loaded_text_device = text_encoder_device
    else:
        if args.prompt_embedding_cache_path is not None:
            raise ValueError("prompt embedding cache is restricted to formal splits")
        tokenizer = load_tokenizer(model_path / "tokenizer")
        text_encoder = load_text_encoder(
            model_path / "text_encoder",
            torch_dtype=dtype,
            torch_device=text_encoder_device,
        )
        loaded_text_device = next(text_encoder.parameters()).device
        text_cache = build_text_embedding_cache(
            prompts=episode_prompts,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            dtype=dtype,
            max_sequence_length=args.max_sequence_length,
        )
        del text_encoder
        del tokenizer
        if text_encoder_device.type == "cuda":
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    vae = load_vae(model_path / "vae", torch_dtype=dtype, torch_device=device)
    loaded_vae_device = next(vae.parameters()).device
    if args.split in FORMAL_PHYSICAL_SPLITS and loaded_vae_device != device:
        raise RuntimeError(
            "formal Track 3.1 VAE parameters were not materialized on the HCU"
        )
    print(
        "HCU_PIPELINE_READY "
        f"umt5_cache=verified-hcu:{loaded_text_device} vae={loaded_vae_device} "
        f"unique_prompts={len(text_cache)}",
        flush=True,
    )
    if not args.write_episodes_jsonl:
        # All read-only validation/model loading succeeded. Invalidate the old
        # ready marker immediately before latent payloads may be changed.
        try:
            invalidate_latent_inventory(dataset_root, "video")
        except FileNotFoundError:
            # Multiple formal workers may race while deleting one old marker.
            pass

    # Detect dataset format: v3.0 has per-video columns, v2.1 uses per-episode videos
    is_v21 = f"videos/{video_keys[0]}/chunk_index" not in episodes_df.columns

    for video_key in video_keys:
        requests: list[EpisodeRequest] = []
        for row in episode_rows:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            prompt = resolve_episode_prompt(row, args.prompt)

            if is_v21:
                # v2.1: videos/chunk-{chunk}/{video_key}/episode_{idx}.mp4
                # Single-chunk repos (our converter writes ALL segments to
                # chunk-000 with total_chunks=1) must use chunk-000 — the
                # naive idx//chunks_size sends episode_index>=1000 to a
                # non-existent chunk-001 and the encode crashes (the giant-repo
                # "wall"). Honor total_chunks; fall back to idx//size otherwise.
                if int(info.get("total_chunks", 1)) <= 1:
                    chunk_index = 0
                else:
                    chunk_index = episode_index // int(info.get("chunks_size", 1000))
                source_video = (
                    dataset_root
                    / "videos"
                    / f"chunk-{chunk_index:03d}"
                    / video_key
                    / f"episode_{episode_index:06d}.mp4"
                )
                start_timestamp = 0.0
                end_timestamp = length / ori_fps
            else:
                # v3.0: videos/{video_key}/chunk-{chunk}/file-{file}.mp4
                chunk_index = int(row[f"videos/{video_key}/chunk_index"])
                file_index = int(row[f"videos/{video_key}/file_index"])
                source_video = build_source_video(
                    dataset_root, video_key, chunk_index, file_index
                )
                start_timestamp = float(row[f"videos/{video_key}/from_timestamp"])
                end_timestamp = float(row[f"videos/{video_key}/to_timestamp"])

            output_path = (
                output_root
                / f"chunk-{chunk_index:03d}"
                / video_key
                / f"episode_{episode_index:06d}_0_{length}.pth"
            )
            if verified_artifacts is not None:
                require_contained_output_path(
                    path=output_path,
                    output_root=output_root,
                    dataset_root=dataset_root,
                )
            provenance = None
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
                    source_video,
                )
                provenance = build_track31_payload_provenance(
                    split=args.split,
                    manifest_sha256=verified_artifacts.manifest_sha256,
                    conversion_report_sha256=conversion_sha256,
                    encoder_source_identity_sha256=str(
                        encoder_source_identity["identity_sha256"]
                    ),
                    kind="video",
                    episode_index=episode_index,
                    start_frame=0,
                    end_frame=length,
                    episode_task=prompt,
                    stream_key=video_key,
                    tactile_mode=None,
                    prompt=prompt,
                    source_video_identity=source_video_identity,
                    execution_device=str(device),
                    text_encoder_device=str(text_encoder_device),
                )
            if output_path.exists() and not args.overwrite:
                if provenance is not None and encoder_source_identity is not None:
                    try:
                        validate_existing_track31_payload(
                            dataset_root,
                            output_path,
                            kind="video",
                            expected_encoder_source_identity_sha256=str(
                                encoder_source_identity["identity_sha256"]
                            ),
                            expected_provenance=provenance,
                        )
                    except Exception as exc:
                        raise ValueError(
                            "existing formal video latent is not reusable; "
                            "re-run this shard with --overwrite to re-encode: "
                            f"{output_path}"
                        ) from exc
                continue

            requests.append(
                EpisodeRequest(
                    episode_index=episode_index,
                    length=length,
                    prompt=prompt,
                    source_video=source_video,
                    output_path=output_path,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    provenance=provenance,
                )
            )

        for request in requests:
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            frames, local_frame_ids = extract_episode_frames(
                request=request,
                width=args.width,
                height=args.height,
                target_fps=args.target_fps,
                ori_fps=ori_fps,
            )
            flat_latent, latent_num_frames, latent_height, latent_width = encode_video(
                frames=frames,
                vae=vae,
                device=device,
                dtype=dtype,
            )
            text_emb = text_cache[normalize_prompt(request.prompt)]
            payload = {
                "latent": flat_latent.to(torch.bfloat16),
                "latent_num_frames": int(latent_num_frames),
                "latent_height": int(latent_height),
                "latent_width": int(latent_width),
                "video_num_frames": int(len(local_frame_ids)),
                "video_height": int(args.height),
                "video_width": int(args.width),
                "text_emb": text_emb.to(torch.bfloat16),
                "text": request.prompt,
                "frame_ids": list(local_frame_ids),
                "start_frame": 0,
                "end_frame": int(request.length),
                "fps": int(args.target_fps),
                "ori_fps": int(ori_fps),
                "encoder_source_identity_sha256": (
                    None
                    if encoder_source_identity is None
                    else encoder_source_identity["identity_sha256"]
                ),
                "track31_encoding_contract": (
                    TRACK31_VIDEO_ENCODING_CONTRACT
                    if args.split in FORMAL_PHYSICAL_SPLITS
                    else None
                ),
                "track31_provenance": request.provenance,
            }
            atomic_torch_save(
                payload,
                request.output_path,
                shard_index=args.shard_index,
            )
            print(f"saved {request.output_path}", flush=True)

    if verified_artifacts is not None and not args.defer_inventory:
        if encoder_source_identity is None:
            raise RuntimeError("Track 3.1 encoder source identity was not initialized")
        conversion_sha256 = verified_artifacts.conversion_report_sha256
        if conversion_sha256 is None:
            raise ValueError("verified Track 3.1 bundle has no conversion digest")
        inventory = finalize_latent_inventory(
            dataset_root,
            kind="video",
            split=args.split,
            manifest_sha256=verified_artifacts.manifest_sha256,
            conversion_report_sha256=conversion_sha256,
            encoder_source_identity=encoder_source_identity,
        )
        print(f"video latent inventory ready: {inventory['inventory_sha256']}")
    elif args.defer_inventory:
        print(
            f"video shard {args.shard_index}/{args.num_shards} complete; "
            "ready inventory intentionally deferred"
        )


if __name__ == "__main__":
    main()
