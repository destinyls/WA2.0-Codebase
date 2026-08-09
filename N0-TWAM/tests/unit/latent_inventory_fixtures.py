"""Small, valid Track 3.1 latent fixtures shared by inventory tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from n0_twam.data.latent_inventory import (
    TRACK31_TACTILE_ENCODING_CONTRACT,
    TRACK31_TACTILE_KEYS,
    TRACK31_TACTILE_MODES,
    TRACK31_VIDEO_ENCODING_CONTRACT,
    TRACK31_VIDEO_KEYS,
    build_track31_payload_provenance,
    build_track31_source_video_identity,
    finalize_latent_inventory,
)

MANIFEST_SHA256 = "1" * 64
CONVERSION_SHA256 = "2" * 64
_ENCODER_SOURCE_CORE = {
    "schema_version": 1,
    "files": [
        {
            "relative_path": "vae/config.json",
            "size_bytes": 1,
            "sha256": "3" * 64,
        }
    ],
}
ENCODER_SOURCE_IDENTITY = {
    **_ENCODER_SOURCE_CORE,
    "identity_sha256": hashlib.sha256(
        json.dumps(
            _ENCODER_SOURCE_CORE,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest(),
}


def write_dataset(
    root: Path,
    *,
    segments: tuple[tuple[int, int], ...] = ((0, 5),),
) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    features = {
        key: {"dtype": "video", "shape": [8, 8, 3]}
        for key in TRACK31_VIDEO_KEYS + TRACK31_TACTILE_KEYS
    }
    features.update(
        {
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [8]},
        }
    )
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "features": features,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": 10,
                "total_episodes": 1,
                "total_frames": max(end for _, end in segments),
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "length": max(end for _, end in segments),
                "tasks": ["insert_HDMI"],
                "action_config": [
                    {"start_frame": start, "end_frame": end} for start, end in segments
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for key in TRACK31_VIDEO_KEYS + TRACK31_TACTILE_KEYS:
        video_path = root / "videos" / "chunk-000" / key / "episode_000000.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"fixture-video:{key}".encode("utf-8"))


def latent_payload(
    start_frame: int,
    end_frame: int,
    *,
    tactile_mode: str | None = None,
    formal: bool = False,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    frame_count = end_frame - start_frame
    retained_count = frame_count - ((frame_count - 1) % 4)
    frame_ids = list(range(start_frame, start_frame + retained_count))
    latent_num_frames = (len(frame_ids) - 1) // 4 + 1
    is_video = tactile_mode is None
    height = 16 if is_video else 8
    width = 16 if is_video else 8
    payload = {
        "latent": torch.zeros(
            (latent_num_frames * height * width, 48), dtype=torch.bfloat16
        ),
        "latent_num_frames": latent_num_frames,
        "latent_height": height,
        "latent_width": width,
        "frame_ids": frame_ids,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": 10,
        "ori_fps": 10,
        "encoder_source_identity_sha256": ENCODER_SOURCE_IDENTITY["identity_sha256"],
        "tactile_residual_mode": tactile_mode,
        "tactile_local_mode": "current" if tactile_mode == "local" else None,
        "track31_encoding_contract": (
            (
                TRACK31_VIDEO_ENCODING_CONTRACT
                if is_video
                else TRACK31_TACTILE_ENCODING_CONTRACT
            )
            if formal
            else None
        ),
        "track31_provenance": provenance,
    }
    if is_video:
        payload.update(
            {
                "video_num_frames": len(frame_ids),
                "video_height": 256,
                "video_width": 256,
                "text_emb": torch.zeros((512, 4096), dtype=torch.bfloat16),
                "text": "insert_HDMI",
            }
        )
    else:
        payload.update({"tactile_resize_h": 128, "tactile_resize_w": 128})
    return payload


def write_segment_artifacts(
    root: Path,
    start_frame: int,
    end_frame: int,
    *,
    video: bool = True,
    tactile: bool = True,
) -> None:
    filename = f"episode_000000_{start_frame}_{end_frame}.pth"
    formal = root.name in {"train759", "frozen40"}
    if video:
        for key in TRACK31_VIDEO_KEYS:
            path = root / "latents" / "chunk-000" / key / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            provenance = None
            if formal:
                source_identity = build_track31_source_video_identity(
                    root,
                    root / "videos" / "chunk-000" / key / "episode_000000.mp4",
                )
                provenance = build_track31_payload_provenance(
                    split=root.name,
                    manifest_sha256=MANIFEST_SHA256,
                    conversion_report_sha256=CONVERSION_SHA256,
                    encoder_source_identity_sha256=str(
                        ENCODER_SOURCE_IDENTITY["identity_sha256"]
                    ),
                    kind="video",
                    episode_index=0,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    episode_task="insert_HDMI",
                    stream_key=key,
                    tactile_mode=None,
                    prompt="insert_HDMI",
                    source_video_identity=source_identity,
                    execution_device="cuda:0",
                    text_encoder_device="cuda:0",
                )
            torch.save(
                latent_payload(
                    start_frame,
                    end_frame,
                    formal=formal,
                    provenance=provenance,
                ),
                path,
            )
    if tactile:
        for mode in TRACK31_TACTILE_MODES:
            for key in TRACK31_TACTILE_KEYS:
                path = root / "latents_tactile" / mode / "chunk-000" / key / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                provenance = None
                if formal:
                    source_identity = build_track31_source_video_identity(
                        root,
                        root / "videos" / "chunk-000" / key / "episode_000000.mp4",
                    )
                    provenance = build_track31_payload_provenance(
                        split=root.name,
                        manifest_sha256=MANIFEST_SHA256,
                        conversion_report_sha256=CONVERSION_SHA256,
                        encoder_source_identity_sha256=str(
                            ENCODER_SOURCE_IDENTITY["identity_sha256"]
                        ),
                        kind="tactile",
                        episode_index=0,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        episode_task="insert_HDMI",
                        stream_key=key,
                        tactile_mode=mode,
                        prompt=None,
                        source_video_identity=source_identity,
                        execution_device="cuda:0",
                        text_encoder_device=None,
                    )
                torch.save(
                    latent_payload(
                        start_frame,
                        end_frame,
                        tactile_mode=mode,
                        formal=formal,
                        provenance=provenance,
                    ),
                    path,
                )


def finalize_pair(root: Path) -> None:
    for kind in ("video", "tactile"):
        finalize_latent_inventory(
            root,
            kind=kind,
            split="train",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )
