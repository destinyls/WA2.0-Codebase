#!/usr/bin/env python3
"""Encode one deterministic modulo shard of the official Franka all600 repo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from script.encode_lerobot_n0_latents import main as encode_main


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
        raise ValueError("invalid Franka latent shard index/count")
    episode_ids = [
        str(index)
        for index in range(600)
        if index % args.num_shards == args.shard_index
    ]
    sys.argv = [
        "encode_lerobot_n0_latents.py",
        "--dataset-root",
        str(args.dataset_root),
        "--model-path",
        str(args.model_path),
        "--target-fps",
        "10",
        "--height",
        "256",
        "--width",
        "256",
        "--video-keys",
        "observation.images.top",
        "observation.images.wrist_l",
        "--device",
        args.device,
        "--text-encoder-device",
        args.device,
        "--encoder-source-identity-path",
        str(args.encoder_source_identity),
        "--episodes",
        *episode_ids,
    ]
    encode_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
