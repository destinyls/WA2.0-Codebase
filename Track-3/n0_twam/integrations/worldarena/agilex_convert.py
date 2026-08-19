# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Canonical AgileX episode conversion with explicit action provenance."""

from __future__ import annotations

import itertools
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, cast

import numpy as np
import numpy.typing as npt

from n0_twam.integrations.univtac.convert_lerobot import (
    build_lerobot_table_inventory,
    freeze_episode_action_config,
)

from .agilex_manifest import AgileXRepoRoute
from .agilex_source import AgileXEpisode


class LeRobotSink(Protocol):
    def add_frame(self, frame: dict[str, Any], task: str, timestamp: float) -> None:
        pass

    def save_episode(self) -> None:
        pass


class AgileXFrameEpisode(Protocol):
    """Streaming episode contract used by official storage readers."""

    route: AgileXRepoRoute
    length: int
    task: str

    def iter_converted_frames(self) -> Iterator["AgileXConvertedFrame"]:
        pass


class ExternalVideoFrameEpisode(AgileXFrameEpisode, Protocol):
    source_video_paths: Mapping[str, Path]

    def feature_frame(self) -> "AgileXConvertedFrame":
        pass

    def iter_tabular_frames(self) -> Iterator["AgileXConvertedFrame"]:
        pass


@dataclass(frozen=True)
class AgileXConvertedFrame:
    rgb: Mapping[str, npt.NDArray[np.uint8]]
    tactile: Mapping[str, npt.NDArray[np.uint8]]
    wrench: Mapping[str, npt.NDArray[np.float32]]
    joint_qpos: npt.NDArray[np.float32]
    action: npt.NDArray[np.float32]
    action_valid: bool
    action_schema: str
    action_label_source: str
    source_relative_path: str
    source_episode_id: int
    source_row_index: int
    timestamp: float

    def to_lerobot_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {key: value.copy() for key, value in self.rgb.items()}
        output.update({key: value.copy() for key, value in self.tactile.items()})
        output.update({key: value.copy() for key, value in self.wrench.items()})
        output.update(
            {
                "observation.joint_qpos": self.joint_qpos.copy(),
                "action": self.action.copy(),
                "action.valid": np.asarray((self.action_valid,), dtype=np.bool_),
                "action.schema": self.action_schema,
                "action.label_source": self.action_label_source,
                "source.relative_path": self.source_relative_path,
                "source.episode_id": np.asarray(
                    ((self.source_episode_id,),), dtype=np.int64
                ),
                "source.row_index": np.asarray(
                    ((self.source_row_index,),), dtype=np.int64
                ),
            }
        )
        return output


def iter_episode_frames(episode: AgileXEpisode) -> Iterator[AgileXConvertedFrame]:
    """Yield all rows exactly once; formal actions remain on their source row."""

    for row_index in range(episode.length):
        yield AgileXConvertedFrame(
            rgb={key: value[row_index] for key, value in episode.rgb.items()},
            tactile={key: value[row_index] for key, value in episode.tactile.items()},
            wrench={key: value[row_index] for key, value in episode.wrench.items()},
            joint_qpos=episode.joint_qpos[row_index],
            action=episode.actions[row_index],
            action_valid=bool(episode.action_valid_mask[row_index]),
            action_schema=episode.route.action_schema,
            action_label_source=episode.route.action_label_source,
            source_relative_path=episode.source_relative_path,
            source_episode_id=episode.episode_id,
            source_row_index=row_index,
            timestamp=float(episode.timestamps[row_index]),
        )


def _image_feature(image: npt.NDArray[np.uint8]) -> dict[str, object]:
    return {
        "dtype": "video",
        "shape": tuple(int(value) for value in image.shape),
        "names": ("height", "width", "channels"),
    }


def build_lerobot_features(
    first_frame: AgileXConvertedFrame,
    *,
    route: AgileXRepoRoute,
) -> dict[str, dict[str, object]]:
    """Build an exact feature roster from an already audited first frame."""

    if first_frame.action_schema != route.action_schema:
        raise ValueError("AgileX frame action schema differs from route")
    if set(first_frame.rgb) != set(route.rgb_keys):
        raise ValueError("AgileX frame RGB roster differs from route")
    if set(first_frame.tactile) != set(route.tactile_keys):
        raise ValueError("AgileX frame tactile roster differs from route")
    if set(first_frame.wrench) != set(route.wrench_keys):
        raise ValueError("AgileX frame wrench roster differs from route")
    features: dict[str, dict[str, object]] = {
        "observation.joint_qpos": {
            "dtype": "float32",
            "shape": (14,),
            "names": list(route.channel_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": list(route.channel_names),
        },
        "action.valid": {"dtype": "bool", "shape": (1,), "names": None},
        "action.schema": {"dtype": "string", "shape": (1,), "names": None},
        "action.label_source": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "source.relative_path": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "source.episode_id": {"dtype": "int64", "shape": (1, 1), "names": None},
        "source.row_index": {"dtype": "int64", "shape": (1, 1), "names": None},
    }
    for key, image in (*first_frame.rgb.items(), *first_frame.tactile.items()):
        features[key] = _image_feature(image)
    for key in first_frame.wrench:
        features[key] = {
            "dtype": "float32",
            "shape": (6,),
            "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
        }
    return features


def convert_episode_to_sink(episode: AgileXEpisode, sink: LeRobotSink) -> int:
    count = 0
    for frame in iter_episode_frames(episode):
        sink.add_frame(
            frame.to_lerobot_dict(),
            task=episode.task,
            timestamp=frame.timestamp,
        )
        count += 1
    if count != episode.length:
        raise RuntimeError("AgileX converted episode length mismatch")
    sink.save_episode()
    return count


def convert_frame_episode_to_sink(
    episode: AgileXFrameEpisode,
    sink: LeRobotSink,
) -> int:
    """Write one storage-backed episode while holding only one decoded row."""

    count = 0
    for frame in episode.iter_converted_frames():
        sink.add_frame(
            frame.to_lerobot_dict(),
            task=episode.task,
            timestamp=frame.timestamp,
        )
        count += 1
    if count != episode.length:
        raise RuntimeError("AgileX streaming conversion length mismatch")
    sink.save_episode()
    return count


_DEFAULT_FRAME_KEYS = frozenset(
    ("index", "episode_index", "frame_index", "timestamp", "task_index")
)


def _append_external_video_frame(
    dataset: Any,
    frame: AgileXConvertedFrame,
    *,
    task: str,
    stats_image_paths: Mapping[str, Path],
) -> None:
    """Append one row while its already-audited videos remain external."""

    payload = frame.to_lerobot_dict()
    video_keys = set(dataset.meta.video_keys)
    expected = set(dataset.features).difference(_DEFAULT_FRAME_KEYS, video_keys)
    if set(payload) != expected:
        raise ValueError("AgileX external-video tabular feature roster mismatch")
    if dataset.episode_buffer is None:
        dataset.episode_buffer = dataset.create_episode_buffer()
    buffer = dataset.episode_buffer
    frame_index = int(buffer["size"])
    buffer["frame_index"].append(frame_index)
    buffer["timestamp"].append(frame.timestamp)
    buffer["task"].append(task)
    for key, value in payload.items():
        buffer[key].append(value)
    for key in video_keys:
        buffer[key].append(str(stats_image_paths[key]))
    buffer["size"] += 1


def _link_external_episode_videos(
    dataset: Any,
    episode: ExternalVideoFrameEpisode,
) -> None:
    sources = dict(episode.source_video_paths)
    if set(sources) != set(dataset.meta.video_keys):
        raise ValueError("AgileX external-video source roster mismatch")
    episode_index = int(dataset.meta.total_episodes)
    for key, source in sources.items():
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"AgileX source video must be a regular file: {source}")
        destination = dataset.root / dataset.meta.get_video_file_path(
            ep_index=episode_index, vid_key=key
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"AgileX derived video already exists: {destination}")
        os.link(source, destination)


def _convert_external_video_episode(
    episode: ExternalVideoFrameEpisode,
    dataset: Any,
) -> int:
    _link_external_episode_videos(dataset, episode)
    feature_frame = episode.feature_frame()
    images = {**feature_frame.rgb, **feature_frame.tactile}
    if set(images) != set(dataset.meta.video_keys):
        raise ValueError("AgileX external-video stats image roster mismatch")
    episode_index = int(dataset.meta.total_episodes)
    stats_image_paths: dict[str, Path] = {}
    for key, image in images.items():
        factor = max(1, max(image.shape[:2]) // 150)
        sampled = np.ascontiguousarray(image[::factor, ::factor])
        path = dataset.root / ".video_stats" / key / f"episode_{episode_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset._save_image(sampled, path)
        stats_image_paths[key] = path
    count = 0
    try:
        for frame in episode.iter_tabular_frames():
            _append_external_video_frame(
                dataset,
                frame,
                task=episode.task,
                stats_image_paths=stats_image_paths,
            )
            count += 1
        if count != episode.length:
            raise RuntimeError("AgileX external-video episode length mismatch")
        dataset.save_episode()
    finally:
        for path in stats_image_paths.values():
            path.unlink(missing_ok=True)
        for parent in sorted(
            {path.parent for path in stats_image_paths.values()}, reverse=True
        ):
            try:
                parent.rmdir()
            except OSError:
                pass
        try:
            (dataset.root / ".video_stats").rmdir()
        except OSError:
            pass
    return count


def _external_video_episode(
    episode: AgileXFrameEpisode,
) -> ExternalVideoFrameEpisode | None:
    sources = getattr(episode, "source_video_paths", None)
    feature_frame = getattr(episode, "feature_frame", None)
    rows = getattr(episode, "iter_tabular_frames", None)
    if isinstance(sources, Mapping) and callable(feature_frame) and callable(rows):
        return cast(ExternalVideoFrameEpisode, episode)
    return None


def write_agilex_lerobot_dataset(
    episodes: Sequence[AgileXEpisode],
    *,
    output_root: Path,
    repo_id: str,
    fps: int,
    image_writer_threads: int = 8,
) -> dict[str, object]:
    """Materialize one route-homogeneous repository and return its receipt."""

    if not episodes:
        raise ValueError("AgileX conversion requires at least one episode")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("AgileX LeRobot fps must be a positive integer")
    route = episodes[0].route
    if any(
        episode.route.route_identity != route.route_identity for episode in episodes
    ):
        raise ValueError("one AgileX LeRobot repo cannot mix route contracts")
    output = Path(output_root).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"AgileX LeRobot output must not exist: {output}")
    first = next(iter_episode_frames(episodes[0]), None)
    if first is None:
        raise ValueError("first AgileX episode has no convertible frames")
    features = build_lerobot_features(first, route=route)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - production dependency
        raise ImportError("LeRobot 0.3.3 is required for AgileX conversion") from error
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(output),
        fps=fps,
        robot_type="agilex_dual_arm",
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
        "status": "complete",
        "repo_id": repo_id,
        "output_root": str(output),
        "episode_count": len(episodes),
        "frame_count": converted_frames,
        "fps": fps,
        "formal": route.formal,
        "action_schema": route.action_schema,
        "action_label_source": route.action_label_source,
        "action_label_offset": route.action_contract.label_offset,
        "repo_route_identity": route.route_identity,
        "temporal_alignment_identity": route.temporal_alignment_identity,
        "table_inventory": build_lerobot_table_inventory(output),
    }


def write_agilex_streaming_lerobot_dataset(
    episodes: Iterable[AgileXFrameEpisode],
    *,
    output_root: Path,
    repo_id: str,
    fps: int,
    image_writer_threads: int = 8,
) -> dict[str, object]:
    """Materialize a route-homogeneous iterable without retaining video arrays."""

    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("AgileX LeRobot fps must be a positive integer")
    output = Path(output_root).expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"AgileX LeRobot output must not exist: {output}")
    iterator = iter(episodes)
    try:
        first_episode = next(iterator)
    except StopIteration as error:
        raise ValueError("AgileX conversion requires at least one episode") from error
    external_first = _external_video_episode(first_episode)
    first_frame = (
        external_first.feature_frame()
        if external_first is not None
        else next(first_episode.iter_converted_frames(), None)
    )
    if first_frame is None:
        raise ValueError("first AgileX episode has no convertible frames")
    route = first_episode.route
    features = build_lerobot_features(first_frame, route=route)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - production dependency
        raise ImportError("LeRobot 0.3.3 is required for AgileX conversion") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.conversion-{uuid.uuid4().hex}.partial")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"AgileX conversion staging path exists: {staging}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(staging),
        fps=fps,
        robot_type="agilex_dual_arm",
        features=features,
        use_videos=True,
        image_writer_threads=(
            0 if external_first is not None else image_writer_threads
        ),
    )
    converted_frames = 0
    episode_count = 0
    try:
        for episode in itertools.chain((first_episode,), iterator):
            if episode.route.route_identity != route.route_identity:
                raise ValueError("one AgileX LeRobot repo cannot mix route contracts")
            external_episode = _external_video_episode(episode)
            if (external_episode is None) != (external_first is None):
                raise ValueError("one AgileX repo cannot mix video storage strategies")
            if external_episode is None:
                converted_frames += convert_frame_episode_to_sink(episode, dataset)
            else:
                converted_frames += _convert_external_video_episode(
                    external_episode, dataset
                )
            episode_count += 1
    finally:
        dataset.stop_image_writer()
    freeze_episode_action_config(staging)
    os.replace(staging, output)
    return {
        "schema_version": 1,
        "status": "complete",
        "repo_id": repo_id,
        "output_root": str(output),
        "episode_count": episode_count,
        "frame_count": converted_frames,
        "fps": fps,
        "formal": route.formal,
        "action_schema": route.action_schema,
        "action_label_source": route.action_label_source,
        "action_label_offset": route.action_contract.label_offset,
        "repo_route_identity": route.route_identity,
        "temporal_alignment_identity": route.temporal_alignment_identity,
        "table_inventory": build_lerobot_table_inventory(output),
    }


__all__ = (
    "AgileXFrameEpisode",
    "AgileXConvertedFrame",
    "LeRobotSink",
    "build_lerobot_features",
    "convert_episode_to_sink",
    "convert_frame_episode_to_sink",
    "iter_episode_frames",
    "write_agilex_lerobot_dataset",
    "write_agilex_streaming_lerobot_dataset",
)
