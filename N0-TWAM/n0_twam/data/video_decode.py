# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Strict RGB video decoding shared by the latent encoders."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any, TypeAlias

import cv2
import numpy as np
import numpy.typing as npt

UInt8Frame: TypeAlias = npt.NDArray[np.uint8]
UInt8Video: TypeAlias = npt.NDArray[np.uint8]


class VideoDecoderUnavailableError(RuntimeError):
    """Raised only when a decoder backend cannot be executed."""


def _validate_request(
    source_video: Path,
    start_timestamp: float,
    end_timestamp: float,
    length: int,
    width: int,
    height: int,
    target_fps: int,
    ori_fps: int,
) -> None:
    if not source_video.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source_video}")
    if not math.isfinite(start_timestamp) or not math.isfinite(end_timestamp):
        raise ValueError("Video timestamps must be finite")
    if start_timestamp < 0.0 or end_timestamp <= start_timestamp:
        raise ValueError(f"Invalid video interval [{start_timestamp}, {end_timestamp})")
    if length <= 0:
        raise ValueError(f"Episode length must be positive, got {length}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Decode dimensions must be positive, got {width}x{height}")
    if target_fps <= 0 or ori_fps <= 0:
        raise ValueError(
            f"Frame rates must be positive, got target={target_fps}, source={ori_fps}"
        )


def _decode_with_imageio_ffmpeg(
    ffmpeg: ModuleType,
    source_video: Path,
    start_timestamp: float,
    end_timestamp: float,
    width: int,
    height: int,
    target_fps: int,
) -> UInt8Video:
    try:
        ffmpeg_executable = ffmpeg.get_ffmpeg_exe()
    except (OSError, RuntimeError) as exc:
        raise VideoDecoderUnavailableError(
            "imageio_ffmpeg could not provide an executable"
        ) from exc
    command = [
        ffmpeg_executable,
        "-v",
        "error",
        "-ss",
        str(start_timestamp),
        "-to",
        str(end_timestamp),
        "-i",
        str(source_video),
        "-vf",
        f"fps={target_fps},scale={width}:{height}:flags=area",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        raw = subprocess.check_output(command, stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise VideoDecoderUnavailableError(
            f"FFmpeg executable is unavailable for {source_video}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg failed to decode {source_video}") from exc

    frame_bytes = width * height * 3
    if len(raw) % frame_bytes != 0:
        raise RuntimeError(
            f"Unexpected FFmpeg output size for {source_video}: {len(raw)} bytes"
        )
    frame_count = len(raw) // frame_bytes
    if frame_count <= 0:
        raise RuntimeError(f"No decoded frames from {source_video}")
    return (
        np.frombuffer(raw, dtype=np.uint8).reshape(frame_count, height, width, 3).copy()
    )


def _frame_timestamp(frame: Any, time_base: Any, source_video: Path) -> float:
    if frame.pts is None or time_base is None:
        raise RuntimeError(f"PyAV frame has no usable timestamp in {source_video}")
    timestamp = float(frame.pts * time_base)
    if not math.isfinite(timestamp):
        raise RuntimeError(f"PyAV frame has a non-finite timestamp in {source_video}")
    return timestamp


def _frame_to_rgb(
    frame: Any,
    source_video: Path,
    width: int,
    height: int,
) -> UInt8Frame:
    array = np.asarray(frame.to_ndarray(format="rgb24"))
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(
            f"PyAV returned a non-RGB24 frame for {source_video}: "
            f"shape={array.shape}, dtype={array.dtype}"
        )
    if array.shape[:2] != (height, width):
        array = cv2.resize(
            array,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(array, dtype=np.uint8)


def _collect_pyav_source_frames(
    av: ModuleType,
    source_video: Path,
    start_timestamp: float,
    end_timestamp: float,
    length: int,
    width: int,
    height: int,
    ori_fps: int,
) -> list[UInt8Frame]:
    try:
        container_context = av.open(str(source_video))
    except Exception as exc:
        raise RuntimeError(f"PyAV failed to open {source_video}") from exc

    try:
        with container_context as container:
            video_streams = list(container.streams.video)
            if not video_streams:
                raise RuntimeError(f"PyAV found no video stream in {source_video}")
            stream = video_streams[0]
            if stream.time_base is None:
                raise RuntimeError(
                    f"PyAV video stream has no time base in {source_video}"
                )
            seek_offset = max(0, int(start_timestamp / float(stream.time_base)))
            container.seek(
                seek_offset,
                stream=stream,
                backward=True,
                any_frame=False,
            )

            indexed_frames: dict[int, UInt8Frame] = {}
            for frame in container.decode(stream):
                timestamp = _frame_timestamp(frame, stream.time_base, source_video)
                relative_index = (timestamp - start_timestamp) * ori_fps
                local_index = int(round(relative_index))
                if local_index < 0:
                    continue
                if local_index >= length:
                    if timestamp >= end_timestamp:
                        break
                    continue
                if abs(relative_index - local_index) > 0.25:
                    raise RuntimeError(
                        f"PyAV found a non-CFR timestamp in {source_video}: "
                        f"timestamp={timestamp}, expected_fps={ori_fps}"
                    )
                if local_index in indexed_frames:
                    raise RuntimeError(
                        f"PyAV found duplicate source frame {local_index} "
                        f"in {source_video}"
                    )
                indexed_frames[local_index] = _frame_to_rgb(
                    frame,
                    source_video,
                    width,
                    height,
                )
                if len(indexed_frames) == length:
                    break
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"PyAV failed to decode {source_video}") from exc

    missing = sorted(set(range(length)) - set(indexed_frames))
    if missing:
        preview = missing[:8]
        raise RuntimeError(
            "PyAV decoded frames do not cover the requested episode in "
            f"{source_video}; "
            f"missing local frame ids {preview} (missing_count={len(missing)})"
        )
    return [indexed_frames[index] for index in range(length)]


def _round_half_up_ratio(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)


def _ffmpeg_sample_source_index(
    output_index: int,
    ori_fps: int,
    target_fps: int,
    length: int,
) -> int:
    numerator = (2 * output_index + 1) * ori_fps
    denominator = 2 * target_fps
    source_index = (numerator + denominator - 1) // denominator - 1
    return min(length - 1, max(0, source_index))


def _decode_with_pyav(
    av: ModuleType,
    source_video: Path,
    start_timestamp: float,
    end_timestamp: float,
    length: int,
    width: int,
    height: int,
    target_fps: int,
    ori_fps: int,
) -> UInt8Video:
    source_frames = _collect_pyav_source_frames(
        av=av,
        source_video=source_video,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        length=length,
        width=width,
        height=height,
        ori_fps=ori_fps,
    )
    sample_count = _round_half_up_ratio(length * target_fps, ori_fps)
    if sample_count <= 0:
        raise RuntimeError(f"No decoded frames from {source_video}")

    sampled_frames = []
    for output_index in range(sample_count):
        source_index = _ffmpeg_sample_source_index(
            output_index=output_index,
            ori_fps=ori_fps,
            target_fps=target_fps,
            length=length,
        )
        sampled_frames.append(source_frames[source_index])
    return np.stack(sampled_frames)


def _trim_for_wan(
    frames: UInt8Video,
    length: int,
    ori_fps: int,
    target_fps: int,
    source_video: Path,
) -> tuple[UInt8Video, list[int]]:
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(
            f"Decoder returned invalid RGB frames for {source_video}: "
            f"shape={frames.shape}, dtype={frames.dtype}"
        )
    frame_ids = [
        min(length - 1, int(round(index * ori_fps / target_fps)))
        for index in range(len(frames))
    ]
    while len(frames) > 1 and len(frames) % 4 != 1:
        frames = frames[:-1]
        frame_ids.pop()
    if len(frames) <= 0:
        raise RuntimeError(f"No valid Wan frame sequence remained for {source_video}")
    return np.ascontiguousarray(frames), frame_ids


def decode_sampled_video_frames(
    *,
    source_video: Path,
    start_timestamp: float,
    end_timestamp: float,
    length: int,
    width: int,
    height: int,
    target_fps: int,
    ori_fps: int,
) -> tuple[UInt8Video, list[int]]:
    """Decode an episode as RGB24, preferring FFmpeg and falling back to PyAV."""
    _validate_request(
        source_video=source_video,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        length=length,
        width=width,
        height=height,
        target_fps=target_fps,
        ori_fps=ori_fps,
    )
    try:
        import imageio_ffmpeg as ffmpeg  # type: ignore[import-untyped]
    except ImportError:
        ffmpeg = None

    if ffmpeg is None:
        try:
            import av
        except ImportError as av_import_error:
            raise RuntimeError(
                "Neither imageio_ffmpeg nor PyAV is available for video decoding"
            ) from av_import_error
        frames = _decode_with_pyav(
            av=av,
            source_video=source_video,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            length=length,
            width=width,
            height=height,
            target_fps=target_fps,
            ori_fps=ori_fps,
        )
    else:
        try:
            frames = _decode_with_imageio_ffmpeg(
                ffmpeg=ffmpeg,
                source_video=source_video,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                width=width,
                height=height,
                target_fps=target_fps,
            )
        except VideoDecoderUnavailableError:
            try:
                import av
            except ImportError as av_import_error:
                raise RuntimeError(
                    "imageio_ffmpeg is unusable and PyAV is unavailable"
                ) from av_import_error
            frames = _decode_with_pyav(
                av=av,
                source_video=source_video,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                length=length,
                width=width,
                height=height,
                target_fps=target_fps,
                ori_fps=ori_fps,
            )
    return _trim_for_wan(
        frames=frames,
        length=length,
        ori_fps=ori_fps,
        target_fps=target_fps,
        source_video=source_video,
    )
