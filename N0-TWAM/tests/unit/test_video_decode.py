# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import pytest

from n0_twam.data import video_decode
from n0_twam.data.video_decode import decode_sampled_video_frames


class _FakeFrame:
    def __init__(self, index: int, rgb: np.ndarray) -> None:
        self.pts = index
        self._rgb = rgb

    def to_ndarray(self, *, format: str) -> np.ndarray:
        assert format == "rgb24"
        return self._rgb.copy()


class _FakeStream:
    def __init__(self, fps: int) -> None:
        self.time_base = Fraction(1, fps)


class _FakeContainer:
    def __init__(self, frames: list[_FakeFrame], fps: int) -> None:
        self._frames = frames
        self.stream = _FakeStream(fps)
        self.streams = SimpleNamespace(video=[self.stream])
        self.seek_calls: list[tuple[int, object, bool, bool]] = []

    def __enter__(self) -> _FakeContainer:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def seek(
        self,
        offset: int,
        *,
        stream: object,
        backward: bool,
        any_frame: bool,
    ) -> None:
        self.seek_calls.append((offset, stream, backward, any_frame))

    def decode(self, stream: object) -> Iterator[_FakeFrame]:
        assert stream is self.stream
        yield from self._frames


def _rgb_frame(index: int, height: int = 2, width: int = 3) -> np.ndarray:
    color = np.asarray((10 + index, 70 + index, 190 - index), dtype=np.uint8)
    return np.broadcast_to(color, (height, width, 3)).copy()


def _install_pyav_only(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[_FakeFrame],
    fps: int,
) -> _FakeContainer:
    container = _FakeContainer(frames, fps)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(open=lambda _: container),
    )
    return container


def test_decoder_prefers_imageio_ffmpeg_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "episode_000000.mp4"
    source.touch()
    expected_frames = np.stack([_rgb_frame(index) for index in range(5)])
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: "unused"),
    )

    def fake_ffmpeg_decode(**_: object) -> np.ndarray:
        calls.append("imageio_ffmpeg")
        return expected_frames.copy()

    def reject_pyav(**_: object) -> np.ndarray:
        raise AssertionError("PyAV fallback must not run when imageio_ffmpeg imports")

    monkeypatch.setattr(video_decode, "_decode_with_imageio_ffmpeg", fake_ffmpeg_decode)
    monkeypatch.setattr(video_decode, "_decode_with_pyav", reject_pyav)

    frames, frame_ids = decode_sampled_video_frames(
        source_video=source,
        start_timestamp=0.0,
        end_timestamp=0.5,
        length=5,
        width=3,
        height=2,
        target_fps=10,
        ori_fps=10,
    )

    assert calls == ["imageio_ffmpeg"]
    assert frame_ids == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(frames, expected_frames)


def test_pyav_fallback_preserves_v21_rgb_frame_ids_and_wan_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "episode_000000.mp4"
    source.touch()
    source_frames = [_FakeFrame(index, _rgb_frame(index)) for index in range(10)]
    container = _install_pyav_only(monkeypatch, source_frames, fps=10)

    frames, frame_ids = decode_sampled_video_frames(
        source_video=source,
        start_timestamp=0.0,
        end_timestamp=1.0,
        length=10,
        width=2,
        height=1,
        target_fps=10,
        ori_fps=10,
    )

    assert frames.dtype == np.uint8
    assert frames.shape == (9, 1, 2, 3)
    assert frame_ids == list(range(9))
    assert frames.shape[0] % 4 == 1
    np.testing.assert_array_equal(
        frames[:, 0, 0], np.stack([_rgb_frame(index)[0, 0] for index in range(9)])
    )
    assert container.seek_calls == [(0, container.stream, True, False)]


def test_pyav_fallback_when_ffmpeg_executable_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "episode_000000.mp4"
    source.touch()
    source_frames = [_FakeFrame(index, _rgb_frame(index)) for index in range(5)]
    container = _FakeContainer(source_frames, fps=10)
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: "/missing/ffmpeg"),
    )
    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(open=lambda _: container),
    )

    def unavailable_binary(*_: object, **__: object) -> bytes:
        raise FileNotFoundError("missing FFmpeg binary")

    monkeypatch.setattr(video_decode.subprocess, "check_output", unavailable_binary)

    frames, frame_ids = decode_sampled_video_frames(
        source_video=source,
        start_timestamp=0.0,
        end_timestamp=0.5,
        length=5,
        width=3,
        height=2,
        target_fps=10,
        ori_fps=10,
    )

    assert frame_ids == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(
        frames, np.stack([frame._rgb for frame in source_frames])
    )


def test_ffmpeg_content_error_does_not_fall_back_to_pyav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "broken.mp4"
    source.touch()
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: "/available/ffmpeg"),
    )
    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(
            open=lambda _: (_ for _ in ()).throw(
                AssertionError("PyAV must not hide an FFmpeg content error")
            )
        ),
    )

    def corrupt_video(*_: object, **__: object) -> bytes:
        raise subprocess.CalledProcessError(returncode=1, cmd="ffmpeg")

    monkeypatch.setattr(video_decode.subprocess, "check_output", corrupt_video)

    with pytest.raises(RuntimeError, match="FFmpeg failed to decode"):
        decode_sampled_video_frames(
            source_video=source,
            start_timestamp=0.0,
            end_timestamp=0.5,
            length=5,
            width=3,
            height=2,
            target_fps=10,
            ori_fps=10,
        )


def test_pyav_fallback_uses_existing_frame_id_semantics_when_downsampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "episode_000001.mp4"
    source.touch()
    source_frames = [_FakeFrame(index, _rgb_frame(index)) for index in range(15)]
    _install_pyav_only(monkeypatch, source_frames, fps=30)

    frames, frame_ids = decode_sampled_video_frames(
        source_video=source,
        start_timestamp=0.0,
        end_timestamp=0.5,
        length=15,
        width=3,
        height=2,
        target_fps=10,
        ori_fps=30,
    )

    assert frame_ids == [0, 3, 6, 9, 12]
    # FFmpeg's fps filter keeps the final input frame before each half-open
    # output interval boundary: source indices 1, 4, 7, 10, 13 here.
    np.testing.assert_array_equal(
        frames[:, 0, 0],
        np.stack([_rgb_frame(index)[0, 0] for index in (1, 4, 7, 10, 13)]),
    )


def test_pyav_fallback_respects_nonzero_v3_timestamp_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "file-000.mp4"
    source.touch()
    source_frames = [_FakeFrame(index, _rgb_frame(index)) for index in range(20)]
    container = _install_pyav_only(monkeypatch, source_frames, fps=10)

    frames, frame_ids = decode_sampled_video_frames(
        source_video=source,
        start_timestamp=1.0,
        end_timestamp=1.5,
        length=5,
        width=3,
        height=2,
        target_fps=10,
        ori_fps=10,
    )

    assert frame_ids == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(
        frames[:, 0, 0],
        np.stack([_rgb_frame(index)[0, 0] for index in range(10, 15)]),
    )
    assert container.seek_calls == [(10, container.stream, True, False)]


@pytest.mark.parametrize("failure", ("no_stream", "no_frames"))
def test_pyav_fallback_fails_closed_for_unknown_or_empty_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    source = tmp_path / "broken.mp4"
    source.touch()
    container = _install_pyav_only(monkeypatch, [], fps=10)
    if failure == "no_stream":
        container.streams.video = []

    with pytest.raises(RuntimeError, match="video stream|decoded frames"):
        decode_sampled_video_frames(
            source_video=source,
            start_timestamp=0.0,
            end_timestamp=0.5,
            length=5,
            width=3,
            height=2,
            target_fps=10,
            ori_fps=10,
        )


def test_decoder_rejects_missing_source_before_backend_import(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Source video does not exist"):
        decode_sampled_video_frames(
            source_video=tmp_path / "missing.mp4",
            start_timestamp=0.0,
            end_timestamp=0.5,
            length=5,
            width=3,
            height=2,
            target_fps=10,
            ori_fps=10,
        )
