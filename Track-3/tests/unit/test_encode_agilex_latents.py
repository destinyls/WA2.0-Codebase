# Copyright 2025-2026 NeoteAI Team. All rights reserved.
from __future__ import annotations

from pathlib import Path

import torch

from script.track3_2.encode_agilex_latents import (
    _expected_frame_ids,
    _prepare_existing_payloads,
    _tactile_args,
)
from script.encode_tactile_latent import parse_args


def _video_payload(*, length: int, identity: str) -> dict[str, object]:
    frame_ids = _expected_frame_ids(length)
    return {
        "latent": torch.zeros((12, 16), dtype=torch.bfloat16),
        "latent_num_frames": 3,
        "latent_height": 2,
        "latent_width": 2,
        "video_num_frames": len(frame_ids),
        "video_height": 256,
        "video_width": 256,
        "text_emb": torch.zeros((4, 8), dtype=torch.bfloat16),
        "text": "clean the table",
        "frame_ids": frame_ids,
        "start_frame": 0,
        "end_frame": length,
        "fps": 10,
        "ori_fps": 30,
        "encoder_source_identity_sha256": identity,
    }


def test_existing_latent_is_kept_only_when_contract_matches(tmp_path: Path) -> None:
    identity = "a" * 64
    path = (
        tmp_path
        / "latents"
        / "chunk-000"
        / "observation.images.top"
        / "episode_000000_0_30.pth"
    )
    path.parent.mkdir(parents=True)
    torch.save(_video_payload(length=30, identity=identity), path)
    _prepare_existing_payloads(
        tmp_path,
        rows=((0, 30),),
        encoder_identity_sha256=identity,
        include_tactile=False,
    )
    assert path.is_file()

    torch.save(_video_payload(length=30, identity="b" * 64), path)
    _prepare_existing_payloads(
        tmp_path,
        rows=((0, 30),),
        encoder_identity_sha256=identity,
        include_tactile=False,
    )
    assert not path.exists()


def test_tactile_encoder_receives_same_source_identity(tmp_path: Path) -> None:
    identity = tmp_path / "identity.json"
    args = _tactile_args(
        repo=tmp_path,
        model=tmp_path / "model",
        identity=identity,
        episodes=(1, 3),
        device="cuda:0",
    )
    index = args.index("--encoder-source-identity-path")
    assert args[index + 1] == str(identity)


def test_tactile_encoder_accepts_identity_outside_track31(
    tmp_path: Path, monkeypatch
) -> None:
    identity = tmp_path / "identity.json"
    monkeypatch.setattr(
        "sys.argv",
        _tactile_args(
            repo=tmp_path,
            model=tmp_path / "model",
            identity=identity,
            episodes=(1,),
            device="cuda:0",
        ),
    )
    args = parse_args()
    assert args.split is None
    assert args.encoder_source_identity_path == identity
