# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Official Franka HDF5/MP4 to one no-tactile LeRobot v2.1 repository."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from n0_twam.integrations.univtac.convert_lerobot import (
    build_lerobot_table_inventory,
    freeze_episode_action_config,
)

from .franka_actions import (
    DERIVED_ACTION_SCHEMA,
    FRANKA_ACTION_SCHEMA,
    end_pose8_to_ee10,
)
from .franka_source import SOURCE_FPS, TASK_PROMPTS, FrankaEpisode, read_end_pose8

EE10_CHANNEL_NAMES = (
    "x",
    "y",
    "z",
    "rot6d_col0_x",
    "rot6d_col0_y",
    "rot6d_col0_z",
    "rot6d_col1_x",
    "rot6d_col1_y",
    "rot6d_col1_z",
    "gripper",
)


class LeRobotSink(Protocol):
    def add_frame(self, frame: dict[str, Any], task: str, timestamp: float) -> None: ...

    def save_episode(self) -> None: ...


@dataclass(frozen=True)
class FrankaConvertedFrame:
    third_person: npt.NDArray[np.uint8]
    wrist: npt.NDArray[np.uint8]
    state: npt.NDArray[np.float32]
    action: npt.NDArray[np.float32]
    source_relative_path: str
    source_episode_id: int
    source_row_index: int

    def to_lerobot_dict(self) -> dict[str, Any]:
        return {
            "observation.images.top": self.third_person.copy(),
            "observation.images.wrist_l": self.wrist.copy(),
            "observation.state": self.state.copy(),
            "action": self.action.copy(),
            "source.relative_path": self.source_relative_path,
            "source.episode_id": np.asarray(
                ((self.source_episode_id,),), dtype=np.int64
            ),
            "source.row_index": np.asarray(((self.source_row_index,),), dtype=np.int64),
        }


def _video_frames(path: Path) -> Iterator[npt.NDArray[np.uint8]]:
    try:
        import av  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - production dependency
        raise ImportError("PyAV is required for Franka conversion") from error
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(f"Franka MP4 must contain one video stream: {path}")
        for frame in container.decode(streams[0]):
            image = np.asarray(frame.to_ndarray(format="rgb24"))
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"Franka decoded frame contract mismatch: {path}")
            yield np.ascontiguousarray(image)


def iter_episode_frames(episode: FrankaEpisode) -> Iterator[FrankaConvertedFrame]:
    pose8 = read_end_pose8(episode)
    ee10 = end_pose8_to_ee10(pose8)
    third = _video_frames(episode.root / "third_person.mp4")
    wrist = _video_frames(episode.root / "wrist.mp4")
    missing = object()
    count = 0
    for row_index, pair in enumerate(zip_longest(third, wrist, fillvalue=missing)):
        third_frame, wrist_frame = pair
        if third_frame is missing or wrist_frame is missing:
            raise ValueError(
                f"Franka camera frame counts differ: {episode.relative_root}"
            )
        if row_index >= episode.length:
            raise ValueError(f"Franka MP4 has excess frames: {episode.relative_root}")
        if row_index == episode.length - 1:
            count += 1
            continue
        assert isinstance(third_frame, np.ndarray)
        assert isinstance(wrist_frame, np.ndarray)
        yield FrankaConvertedFrame(
            third_person=third_frame,
            wrist=wrist_frame,
            state=np.ascontiguousarray(ee10[row_index]),
            action=np.ascontiguousarray(ee10[row_index + 1]),
            source_relative_path=f"{episode.relative_root}/episode.hdf5",
            source_episode_id=episode.source_episode_id,
            source_row_index=row_index,
        )
        count += 1
    if count != episode.length:
        raise ValueError(
            f"Franka MP4/HDF5 length mismatch: {count} vs {episode.length}"
        )


def build_lerobot_features(
    first_frame: FrankaConvertedFrame,
) -> dict[str, dict[str, object]]:
    features: dict[str, dict[str, object]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),
            "names": list(EE10_CHANNEL_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (10,),
            "names": list(EE10_CHANNEL_NAMES),
        },
        "source.relative_path": {"dtype": "string", "shape": (1,), "names": None},
        "source.episode_id": {"dtype": "int64", "shape": (1, 1), "names": None},
        "source.row_index": {"dtype": "int64", "shape": (1, 1), "names": None},
    }
    for key, image in (
        ("observation.images.top", first_frame.third_person),
        ("observation.images.wrist_l", first_frame.wrist),
    ):
        features[key] = {
            "dtype": "video",
            "shape": tuple(int(value) for value in image.shape),
            "names": ("height", "width", "channels"),
        }
    return features


def convert_episode_to_sink(episode: FrankaEpisode, sink: LeRobotSink) -> int:
    count = 0
    for frame in iter_episode_frames(episode):
        sink.add_frame(
            frame.to_lerobot_dict(),
            task=TASK_PROMPTS[episode.task],
            timestamp=count / float(SOURCE_FPS),
        )
        count += 1
    if count != episode.converted_length:
        raise RuntimeError("Franka converted episode length mismatch")
    sink.save_episode()
    return count


def write_franka_lerobot_dataset(
    episodes: Sequence[FrankaEpisode],
    *,
    output_root: Path,
    repo_id: str,
    image_writer_threads: int = 8,
) -> dict[str, object]:
    """Write all 600 episodes once; immutable views select train/validation."""

    if len(episodes) != 600:
        raise ValueError(
            "official Franka materialization requires exactly 600 episodes"
        )
    output = Path(output_root).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Franka LeRobot output must not exist: {output}")
    first_iterator = iter_episode_frames(episodes[0])
    try:
        first_frame = next(first_iterator)
    except StopIteration as error:
        raise ValueError("first Franka episode has no convertible frames") from error
    features = build_lerobot_features(first_frame)
    try:
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset,  # type: ignore[import-not-found]
        )
    except ImportError as error:  # pragma: no cover
        raise ImportError("LeRobot 0.3.3 is required for Franka conversion") from error
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(output),
        fps=SOURCE_FPS,
        robot_type="franka_fr3",
        features=features,
        use_videos=True,
        image_writer_threads=image_writer_threads,
    )
    converted_frames = 0
    try:
        for episode in episodes:
            converted_frames += convert_episode_to_sink(episode, dataset)
    finally:
        dataset.stop_image_writer()
    freeze_episode_action_config(output)
    return {
        "schema_version": 1,
        "source_action_schema": FRANKA_ACTION_SCHEMA,
        "derived_action_schema": DERIVED_ACTION_SCHEMA,
        "label_offset": "next_recorded_end_pose_v1",
        "tactile_profile": "vision_only",
        "tactile_mode": "disabled",
        "repo_id": repo_id,
        "output_root": str(output),
        "fps": SOURCE_FPS,
        "episode_count": len(episodes),
        "frame_count": converted_frames,
        "table_inventory": build_lerobot_table_inventory(output),
    }


__all__ = (
    "EE10_CHANNEL_NAMES",
    "FrankaConvertedFrame",
    "build_lerobot_features",
    "convert_episode_to_sink",
    "iter_episode_frames",
    "write_franka_lerobot_dataset",
)
