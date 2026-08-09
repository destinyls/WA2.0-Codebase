"""CLI contract for the Track 3.1 video latent encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

FORMAL_PHYSICAL_SPLITS = ("train759", "frozen40")
LEGACY_LOGICAL_SPLITS = ("train", "validation")
TRACK31_ENCODER_SPLITS = LEGACY_LOGICAL_SPLITS + FORMAL_PHYSICAL_SPLITS


def parse_video_encoder_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse the stable video-encoder command-line contract."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=512)
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
    parser.add_argument("--video-keys", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--text-encoder-device",
        default="cuda:0",
        help=(
            "Logical umT5 producer device recorded by the formal prompt cache. "
            "Formal Track 3.1 requires HCU cuda:0 and never falls back to CPU."
        ),
    )
    parser.add_argument(
        "--prompt-embedding-cache-path",
        type=Path,
        default=None,
        help=(
            "Read-only embedding cache produced by the formal HCU umT5 "
            "precompute stage. Required for train759/frozen40 workers."
        ),
    )
    parser.add_argument(
        "--encoder-source-identity-path",
        type=Path,
        default=None,
        help=(
            "Optional precomputed, validated encoder-source identity JSON. "
            "Formal multi-worker launchers use one run-local cache per node."
        ),
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-episodes-jsonl", action="store_true")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Track 3.1 artifact directory; requires --split and enables a "
            "content-addressed completion inventory."
        ),
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
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Override the text prompt for all episodes instead of using the "
            "task string from the dataset."
        ),
    )
    return parser.parse_args(argv)


__all__ = (
    "FORMAL_PHYSICAL_SPLITS",
    "LEGACY_LOGICAL_SPLITS",
    "TRACK31_ENCODER_SPLITS",
    "parse_video_encoder_args",
)
