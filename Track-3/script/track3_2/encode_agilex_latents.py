#!/usr/bin/env python3
"""Encode one explicit HCU shard of both official AgileX mixed repositories."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Mapping

import torch

from n0_twam.data.encoder_source_identity import load_encoder_source_identity
from n0_twam.integrations.worldarena.agilex_official_schema import (
    TACTILE_FILES,
    VISION_ONLY_REPO_ID,
    VISION_TACTILE_REPO_ID,
)
from script.encode_lerobot_n0_latents import main as encode_video
from script.encode_tactile_latent import main as encode_tactile


def _episode_rows(repo: Path) -> tuple[tuple[int, int], ...]:
    jsonl = repo / "meta" / "episodes.jsonl"
    if jsonl.is_file():
        values = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                values.append((int(payload["episode_index"]), int(payload["length"])))
        return tuple(values)
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - production dependency
        raise ImportError(
            "pandas is required to read AgileX episode metadata"
        ) from error
    files = sorted((repo / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise ValueError(f"AgileX repo has no episode metadata: {repo}")
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    return tuple(
        (int(row.episode_index), int(row.length))
        for row in frame.itertuples(index=False)
    )


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _expected_frame_ids(length: int) -> list[int]:
    count = (2 * length * 10 + 30) // 60
    while count > 1 and count % 4 != 1:
        count -= 1
    return [index * 3 for index in range(count)]


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("AgileX latent dimension must be a positive integer")
    return value


def _validate_existing_payload(
    path: Path,
    *,
    length: int,
    encoder_identity_sha256: str,
    kind: str,
    tactile_mode: str | None,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("AgileX latent payload must be a regular file")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("AgileX latent payload must be a mapping")
    frame_ids = payload.get("frame_ids")
    if frame_ids != _expected_frame_ids(length):
        raise ValueError(
            "AgileX latent frame IDs do not match the 30-to-10 FPS contract"
        )
    if (
        payload.get("start_frame") != 0
        or payload.get("end_frame") != length
        or payload.get("fps") != 10
        or payload.get("ori_fps") != 30
        or payload.get("encoder_source_identity_sha256") != encoder_identity_sha256
    ):
        raise ValueError("AgileX latent source or temporal identity mismatch")
    latent = payload.get("latent")
    latent_frames = _positive_int(payload.get("latent_num_frames"))
    latent_height = _positive_int(payload.get("latent_height"))
    latent_width = _positive_int(payload.get("latent_width"))
    if (
        not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.bfloat16
        or latent.ndim != 2
        or latent.shape[0] != latent_frames * latent_height * latent_width
        or latent.shape[1] <= 0
        or not torch.isfinite(latent).all().item()
    ):
        raise ValueError("AgileX latent tensor is invalid")
    if kind == "video":
        text_emb = payload.get("text_emb")
        if (
            payload.get("video_num_frames") != len(frame_ids)
            or payload.get("video_height") != 256
            or payload.get("video_width") != 256
            or not isinstance(payload.get("text"), str)
            or not isinstance(text_emb, torch.Tensor)
            or text_emb.dtype != torch.bfloat16
            or not torch.isfinite(text_emb).all().item()
        ):
            raise ValueError("AgileX video latent payload is invalid")
        return
    if (
        tactile_mode not in ("global", "local")
        or payload.get("tactile_residual_mode") != tactile_mode
        or payload.get("tactile_resize_h") != 128
        or payload.get("tactile_resize_w") != 128
        or (tactile_mode == "local" and payload.get("tactile_local_mode") != "current")
    ):
        raise ValueError("AgileX tactile latent payload is invalid")


def _prepare_existing_payloads(
    repo: Path,
    *,
    rows: tuple[tuple[int, int], ...],
    encoder_identity_sha256: str,
    include_tactile: bool,
) -> None:
    for episode_index, length in rows:
        name = f"episode_{episode_index:06d}_0_{length}.pth"
        candidates = [
            (repo / "latents" / "chunk-000" / key / name, "video", None)
            for key in (
                "observation.images.top",
                "observation.images.wrist_l",
                "observation.images.wrist_r",
            )
        ]
        if include_tactile:
            candidates.extend(
                (
                    repo / "latents_tactile" / mode / "chunk-000" / key / name,
                    "tactile",
                    mode,
                )
                for mode in ("global", "local")
                for key in TACTILE_FILES
            )
        for path, kind, tactile_mode in candidates:
            if not os.path.lexists(path):
                continue
            try:
                _validate_existing_payload(
                    path,
                    length=length,
                    encoder_identity_sha256=encoder_identity_sha256,
                    kind=kind,
                    tactile_mode=tactile_mode,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                if path.is_dir() and not path.is_symlink():
                    raise ValueError(
                        f"AgileX latent output is a directory: {path}"
                    ) from error
                path.unlink(missing_ok=True)
                print(f"removed invalid AgileX latent for re-encoding: {path}")


def _video_args(
    *, repo: Path, model: Path, identity: Path, episodes: tuple[int, ...], device: str
) -> list[str]:
    return [
        "encode_lerobot_n0_latents.py",
        "--dataset-root",
        str(repo),
        "--model-path",
        str(model),
        "--target-fps",
        "10",
        "--height",
        "256",
        "--width",
        "256",
        "--video-keys",
        "observation.images.top",
        "observation.images.wrist_l",
        "observation.images.wrist_r",
        "--device",
        device,
        "--text-encoder-device",
        device,
        "--dtype",
        "bf16",
        "--encoder-source-identity-path",
        str(identity),
        "--episodes",
        *(str(value) for value in episodes),
    ]


def _tactile_args(
    *,
    repo: Path,
    model: Path,
    identity: Path,
    episodes: tuple[int, ...],
    device: str,
) -> list[str]:
    return [
        "encode_tactile_latent.py",
        "--dataset-root",
        str(repo),
        "--model-path",
        str(model),
        "--target-fps",
        "10",
        "--height",
        "128",
        "--width",
        "128",
        "--tactile-keys",
        *TACTILE_FILES.keys(),
        "--device",
        device,
        "--dtype",
        "bf16",
        "--encoder-source-identity-path",
        str(identity),
        "--mode",
        "both",
        "--local-mode",
        "current",
        "--episodes",
        *(str(value) for value in episodes),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--encoder-source-identity", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid AgileX latent shard index/count")
    identity = load_encoder_source_identity(args.encoder_source_identity)
    identity_sha256 = str(identity["identity_sha256"])
    for repo_id in (VISION_ONLY_REPO_ID, VISION_TACTILE_REPO_ID):
        repo = (args.dataset_root / repo_id).resolve(strict=True)
        selected_rows = tuple(
            row
            for row in _episode_rows(repo)
            if row[0] % args.num_shards == args.shard_index
        )
        if not selected_rows:
            continue
        selected = tuple(row[0] for row in selected_rows)
        _prepare_existing_payloads(
            repo,
            rows=selected_rows,
            encoder_identity_sha256=identity_sha256,
            include_tactile=repo_id == VISION_TACTILE_REPO_ID,
        )
        sys.argv = _video_args(
            repo=repo,
            model=args.model_path,
            identity=args.encoder_source_identity,
            episodes=selected,
            device=args.device,
        )
        encode_video()
        _release()
        if repo_id == VISION_TACTILE_REPO_ID:
            sys.argv = _tactile_args(
                repo=repo,
                model=args.model_path,
                identity=args.encoder_source_identity,
                episodes=selected,
                device=args.device,
            )
            encode_tactile()
            _release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
