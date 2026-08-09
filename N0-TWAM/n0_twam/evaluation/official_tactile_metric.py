# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict subprocess adapter for the sibling official tactile metric script."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from n0_twam.checkpointing.identity import validate_sha256
from n0_twam.evaluation.sealed_artifact_io import sha256_file

TACTILE_FRAME_COUNT = 17
TACTILE_FRAME_SHAPE = (128, 256, 3)


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"official JSON contains non-finite constant {value}")


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid official metric JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("official metric JSON must contain an object")
    return payload


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def load_strict_official_metrics(
    path: Path,
    *,
    expected_video_names: Sequence[str],
) -> dict[str, object]:
    """Accept only the exact finite JSON contract emitted by the official script."""

    payload = _read_json_object(path)
    if set(payload) != {"num_videos", "average_psnr", "average_ssim", "per_video"}:
        raise ValueError("official metric JSON fields are invalid")
    expected_names = tuple(sorted(expected_video_names))
    if payload["num_videos"] != len(expected_names):
        raise ValueError("official metric video count is invalid")
    _finite_number(payload["average_psnr"], label="average_psnr")
    _finite_number(payload["average_ssim"], label="average_ssim")
    per_video = payload["per_video"]
    if not isinstance(per_video, list) or len(per_video) != len(expected_names):
        raise ValueError("official per-video metric count is invalid")
    actual_names = []
    for item in per_video:
        if not isinstance(item, dict) or set(item) != {
            "video_name",
            "num_frames",
            "psnr",
            "ssim",
        }:
            raise ValueError("official per-video metric fields are invalid")
        actual_names.append(item["video_name"])
        if item["num_frames"] != TACTILE_FRAME_COUNT:
            raise ValueError("official per-video frame count must be 17")
        _finite_number(item["psnr"], label="per-video PSNR")
        _finite_number(item["ssim"], label="per-video SSIM")
    if tuple(sorted(actual_names)) != expected_names:
        raise ValueError("official metric filenames differ from generated inputs")
    return payload


def _verify_video_inputs(
    staging_root: Path, *, expected_video_names: Sequence[str]
) -> None:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise ImportError("PyAV is required to verify official metric inputs") from exc
    expected = set(expected_video_names)
    for directory_name in ("generate_videos", "gt_videos"):
        directory = staging_root / directory_name
        members = tuple(directory.iterdir())
        actual = {path.name for path in members}
        if actual != expected:
            raise ValueError(f"official {directory_name} filenames differ")
        if any(path.is_symlink() or not path.is_file() for path in members):
            raise ValueError(f"official {directory_name} inputs must be regular files")
        for video_name in expected_video_names:
            frames = []
            with av.open(str(directory / video_name)) as container:
                for frame in container.decode(video=0):
                    frames.append(np.asarray(frame.to_ndarray(format="rgb24")))
            if (
                len(frames) != TACTILE_FRAME_COUNT
                or any(frame.shape != TACTILE_FRAME_SHAPE for frame in frames)
                or any(frame.dtype != np.uint8 for frame in frames)
            ):
                raise ValueError(
                    f"official {directory_name}/{video_name} has invalid count/shape"
                )


def invoke_official_script(
    *,
    staging_root: Path,
    expected_video_names: Sequence[str],
    script_path: Path,
    expected_script_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify inputs and invoke the actual sibling script as a subprocess."""

    script = script_path.resolve(strict=True)
    _verify_video_inputs(staging_root, expected_video_names=expected_video_names)
    script_sha = sha256_file(script)
    approved_sha = validate_sha256(
        expected_script_sha256, label="approved metric script SHA256"
    )
    if script_sha != approved_sha:
        raise ValueError("official metric script SHA256 is not the approved identity")
    output_json = staging_root / "official_metrics.json"
    command = [
        sys.executable,
        str(script),
        "--dataroot",
        str(staging_root),
        "--output_json",
        str(output_json),
        "--device",
        "cpu",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if sha256_file(script) != approved_sha:
        raise RuntimeError("official metric script changed during evaluation")
    metrics = load_strict_official_metrics(
        output_json, expected_video_names=tuple(expected_video_names)
    )
    return metrics, {
        "path": str(script),
        "sha256": script_sha,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
