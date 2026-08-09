# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from pathlib import Path

import numpy as np
import pytest

from n0_twam.evaluation.video_writer import write_rgb_mp4_pyav


def test_pyav_writer_roundtrips_frame_count(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    frames = tuple(
        np.full((16, 16, 3), value, dtype=np.uint8) for value in (0, 64, 128)
    )
    output = tmp_path / "generate_videos" / "sample.mp4"

    codec = write_rgb_mp4_pyav(output, frames, fps=10)

    assert codec in {"libx264", "h264"}
    with av.open(str(output)) as container:
        decoded = list(container.decode(video=0))
    assert len(decoded) == len(frames)
    assert output.stat().st_size > 0
