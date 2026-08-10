# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed PyAV H.264 writer for evaluation artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import numpy.typing as npt


def _validate_frames(
    frames: Iterable[npt.NDArray[np.uint8]],
) -> tuple[npt.NDArray[np.uint8], ...]:
    selected = tuple(frames)
    if not selected:
        raise ValueError("MP4 output requires at least one frame")
    expected_shape = selected[0].shape
    if (
        len(expected_shape) != 3
        or expected_shape[2] != 3
        or min(expected_shape[:2]) <= 0
    ):
        raise ValueError("MP4 frames must have shape [height, width, 3]")
    if expected_shape[0] % 2 or expected_shape[1] % 2:
        raise ValueError("yuv420p H.264 output requires even frame dimensions")
    for frame in selected:
        if frame.dtype != np.uint8 or frame.shape != expected_shape:
            raise ValueError("MP4 frames must share one uint8 RGB24 shape")
    return selected


def _encode_pyav(
    path: Path,
    frames: tuple[npt.NDArray[np.uint8], ...],
    *,
    fps: int,
    codec_name: str,
) -> None:
    import av  # type: ignore[import-not-found]

    height, width, _ = frames[0].shape
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream(codec_name, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for array in frames:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _verify_pyav(
    path: Path,
    *,
    expected_frames: int,
    expected_shape: tuple[int, int, int] | None = None,
) -> dict[str, object]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = str(stream.codec_context.name)
        pixel_format = str(stream.codec_context.format.name)
        frames = tuple(
            np.asarray(frame.to_ndarray(format="rgb24"))
            for frame in container.decode(video=0)
        )
    decoded_frames = len(frames)
    if decoded_frames != expected_frames:
        raise RuntimeError(
            f"encoded MP4 frame count mismatch: {decoded_frames} vs {expected_frames}"
        )
    if expected_shape is not None and any(
        frame.dtype != np.uint8 or frame.shape != expected_shape for frame in frames
    ):
        raise RuntimeError("encoded MP4 decoded frame contract is invalid")
    return {
        "codec": codec,
        "pixel_format": pixel_format,
        "decoded_frames": decoded_frames,
        "decoded_shape": list(frames[0].shape),
    }


def write_rgb_mp4_pyav(
    path: Path,
    frames: Iterable[npt.NDArray[np.uint8]],
    *,
    fps: int,
) -> str:
    """Atomically publish fixed-size RGB24 frames as an H.264 MP4."""

    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("MP4 fps must be a positive integer")
    selected = _validate_frames(frames)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for codec_name in ("libx264", "h264"):
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".mp4",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            _encode_pyav(
                temporary,
                selected,
                fps=fps,
                codec_name=codec_name,
            )
            _verify_pyav(temporary, expected_frames=len(selected))
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, output)
            directory = os.open(
                output.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return codec_name
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
    raise RuntimeError("PyAV could not encode H.264 MP4") from last_error


def write_rgb_mp4_reference(
    path: Path,
    frames: Iterable[npt.NDArray[np.uint8]],
    *,
    fps: int,
) -> dict[str, object]:
    """Publish the frozen reference MP4 profile without codec fallbacks."""

    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("MP4 fps must be a positive integer")
    selected = _validate_frames(frames)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".mp4",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        import imageio.v3 as imageio_v3

        imageio_v3.imwrite(
            temporary,
            np.stack(selected),
            plugin="pyav",
            codec="libx264",
            fps=fps,
        )
        evidence = _verify_pyav(
            temporary,
            expected_frames=len(selected),
            expected_shape=selected[0].shape,
        )
        if evidence["codec"] not in {"h264", "libx264"}:
            raise RuntimeError("reference MP4 did not decode as H.264")
        if evidence["pixel_format"] != "yuv420p":
            raise RuntimeError("reference MP4 pixel format is not yuv420p")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return {"encoder": "libx264", **evidence}
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = ("write_rgb_mp4_pyav", "write_rgb_mp4_reference")
