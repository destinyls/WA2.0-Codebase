#!/usr/bin/env python3
"""Precompute all formal Track 3.1 umT5 embeddings on one HCU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.data.latent_inventory import (
    TRACK31_VIDEO_ENCODING_CONTRACT,
    canonical_bytes,
    load_encoder_source_identity,
    load_expected_segments,
)
from n0_twam.integrations.univtac.artifact_contracts import (
    verify_track31_physical_bundle,
)
from n0_twam.models.utils import load_text_encoder, load_tokenizer
from script.track3_1.hcu_prompt_encoding import (
    build_text_embedding_cache,
    write_prompt_embedding_cache,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train759", "frozen40"), required=True)
    parser.add_argument("--encoder-source-identity-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve(strict=True)
    model_path = args.model_path.resolve(strict=True)
    artifact_root = args.artifact_root.resolve(strict=True)
    if dataset_root.name != args.split:
        raise ValueError("dataset root does not match the physical split")
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("formal umT5 prompt precompute requires HCU cuda:0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("prompt precompute requires exactly one visible HCU")
    if args.max_sequence_length != 512:
        raise ValueError("formal prompt precompute requires sequence length 512")

    verified = verify_track31_physical_bundle(
        manifest_path=artifact_root / "universe_manifest_v4.json",
        conversion_report_path=artifact_root / "conversion_report.json",
        dataset_root=dataset_root,
        physical_split=args.split,
    )
    encoder_identity = load_encoder_source_identity(args.encoder_source_identity_path)
    prompts = [segment.episode_task for segment in load_expected_segments(dataset_root)]
    device = torch.device(args.device)
    tokenizer = load_tokenizer(model_path / "tokenizer")
    text_encoder = load_text_encoder(
        model_path / "text_encoder",
        torch_dtype=torch.bfloat16,
        torch_device=device,
        direct_device_load=True,
    )
    parameter_devices = {
        str(parameter.device) for parameter in text_encoder.parameters()
    }
    if parameter_devices != {"cuda:0"}:
        raise RuntimeError("umT5 parameters were not materialized exclusively on HCU")
    embeddings = build_text_embedding_cache(
        prompts=prompts,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        dtype=torch.bfloat16,
        max_sequence_length=args.max_sequence_length,
        max_unique_prompts=8,
    )
    del text_encoder
    del tokenizer
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    write_prompt_embedding_cache(
        args.output_path,
        embeddings=embeddings,
        split=args.split,
        manifest_sha256=verified.manifest_sha256,
        conversion_report_sha256=verified.conversion_report_sha256,
        encoder_source_identity_sha256=str(encoder_identity["identity_sha256"]),
        encoding_contract_sha256=hashlib.sha256(
            canonical_bytes(TRACK31_VIDEO_ENCODING_CONTRACT)
        ).hexdigest(),
        max_sequence_length=args.max_sequence_length,
    )
    print(
        "HCU_PROMPT_CACHE_READY "
        f"umt5=cuda:0 prompts={len(embeddings)} output={args.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
