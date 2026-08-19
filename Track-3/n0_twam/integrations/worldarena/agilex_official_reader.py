# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Streaming reader for the pinned WorldArena AgileX episode layout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping

import numpy as np
import numpy.typing as npt

from .agilex_actions import QPOS14_ACTION_SCHEMA, validate_qpos14
from .agilex_convert import AgileXConvertedFrame
from .agilex_manifest import AgileXRepoRoute
from .agilex_official_schema import (
    ALLOWED_DEVICE_NAMES,
    BASE_FILES,
    EXPECTED_TASK_EPISODES,
    OFFICIAL_EPISODE_COUNT,
    OFFICIAL_SOURCE_FPS,
    OFFICIAL_TASKS,
    PROMPT_KEYS,
    RAW_ACTION_DATASET,
    RAW_QPOS_DATASET,
    RGB_FILES,
    TACTILE_FILES,
    TACTILE_FILES_REQUIRED,
    VISION_ONLY_REPO_ID,
    VISION_TACTILE_REPO_ID,
    WRENCH_DATASETS,
    task_has_contact,
)


@dataclass(frozen=True)
class VideoHeader:
    path: Path
    frame_count: int
    height: int
    width: int
    fps: float


@dataclass(frozen=True)
class OfficialAgileXEpisode:
    route: AgileXRepoRoute
    root: Path
    source_relative_path: str
    source_episode_id: int
    task_id: str
    task_prompt: str
    qpos: npt.NDArray[np.float32]
    actions: npt.NDArray[np.float32]
    action_valid_mask: npt.NDArray[np.bool_]
    rgb_headers: Mapping[str, VideoHeader]
    tactile_headers: Mapping[str, VideoHeader]
    wrench: Mapping[str, npt.NDArray[np.float32]]
    terminal_video_frame_trimmed: bool

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def task(self) -> str:
        return self.task_prompt

    @property
    def consumed_files(self) -> tuple[Path, ...]:
        files = [self.root / name for name in BASE_FILES]
        if self.tactile_headers:
            files.extend(self.root / name for name in TACTILE_FILES_REQUIRED)
        return tuple(files)

    @property
    def source_video_paths(self) -> Mapping[str, Path]:
        return MappingProxyType(
            {
                **{key: header.path for key, header in self.rgb_headers.items()},
                **{key: header.path for key, header in self.tactile_headers.items()},
            }
        )

    def feature_frame(self) -> AgileXConvertedFrame:
        """Return one shape-bearing row without decoding any source video."""

        rgb = {
            key: next(_video_frames(header)) for key, header in self.rgb_headers.items()
        }
        tactile = {
            key: next(_video_frames(header))
            for key, header in self.tactile_headers.items()
        }
        return self._tabular_frame(0, rgb=rgb, tactile=tactile)

    def iter_tabular_frames(self) -> Iterator[AgileXConvertedFrame]:
        """Yield audited state/action/contact rows without decoding RGB bytes."""

        for row_index in range(self.length):
            yield self._tabular_frame(row_index, rgb={}, tactile={})

    def _tabular_frame(
        self,
        row_index: int,
        *,
        rgb: Mapping[str, npt.NDArray[np.uint8]],
        tactile: Mapping[str, npt.NDArray[np.uint8]],
    ) -> AgileXConvertedFrame:
        return AgileXConvertedFrame(
            rgb=rgb,
            tactile=tactile,
            wrench={key: value[row_index] for key, value in self.wrench.items()},
            joint_qpos=self.qpos[row_index],
            action=self.actions[row_index],
            action_valid=bool(self.action_valid_mask[row_index]),
            action_schema=QPOS14_ACTION_SCHEMA,
            action_label_source="commanded",
            source_relative_path=self.source_relative_path,
            source_episode_id=self.source_episode_id,
            source_row_index=row_index,
            timestamp=row_index / float(OFFICIAL_SOURCE_FPS),
        )

    def iter_converted_frames(self) -> Iterator[AgileXConvertedFrame]:
        streams = {
            **{key: _video_frames(header) for key, header in self.rgb_headers.items()},
            **{
                key: _video_frames(header)
                for key, header in self.tactile_headers.items()
            },
        }
        keys = tuple(streams)
        missing = object()
        decoded = 0
        for row_index, frames in enumerate(
            zip_longest(*(streams[key] for key in keys), fillvalue=missing)
        ):
            if any(frame is missing for frame in frames):
                raise ValueError(
                    f"AgileX video streams have different lengths: {self.root}"
                )
            decoded += 1
            if row_index == self.length and self.terminal_video_frame_trimmed:
                continue
            if row_index >= self.length:
                raise ValueError(f"AgileX MP4 has excess frames: {self.root}")
            images = dict(zip(keys, frames, strict=True))
            yield self._tabular_frame(
                row_index,
                rgb={key: images[key] for key in self.rgb_headers},
                tactile={key: images[key] for key in self.tactile_headers},
            )
        expected = self.length + int(self.terminal_video_frame_trimmed)
        if decoded != expected:
            raise ValueError(
                f"AgileX MP4/HDF5 length mismatch: {decoded} vs {expected}"
            )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"AgileX JSON must contain an object: {path}")
    return payload


def load_official_task_prompts(raw_root: Path) -> dict[str, str]:
    payload = _read_json(Path(raw_root) / "prompt.json")
    prompts: dict[str, str] = {}
    for task in OFFICIAL_TASKS:
        value = payload.get(PROMPT_KEYS[task])
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"official AgileX prompt is missing for {task}")
        prompts[task] = value.strip()
    return prompts


def _video_header(path: Path) -> VideoHeader:
    try:
        import av  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - production dependency
        raise ImportError("PyAV is required for official AgileX conversion") from error
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(f"AgileX MP4 must contain one video stream: {path}")
        stream = streams[0]
        frame_count = int(stream.frames)
        if frame_count <= 0:
            frame_count = sum(1 for _ in container.decode(stream))
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        if frame_count <= 0 or stream.height <= 0 or stream.width <= 0:
            raise ValueError(f"AgileX MP4 header is invalid: {path}")
        if abs(fps - OFFICIAL_SOURCE_FPS) > 0.05:
            raise ValueError(f"AgileX MP4 must be 30 FPS: {path} ({fps})")
        return VideoHeader(
            path, frame_count, int(stream.height), int(stream.width), fps
        )


def _video_frames(header: VideoHeader) -> Iterator[npt.NDArray[np.uint8]]:
    import av  # type: ignore[import-untyped]

    with av.open(str(header.path), mode="r") as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            image = np.asarray(frame.to_ndarray(format="rgb24"))
            if image.dtype != np.uint8 or image.shape != (
                header.height,
                header.width,
                3,
            ):
                raise ValueError(
                    f"AgileX decoded frame contract mismatch: {header.path}"
                )
            yield np.ascontiguousarray(image)


def _hdf_arrays(root: Path, *, contact: bool) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    Mapping[str, npt.NDArray[np.float32]],
]:
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - production dependency
        raise ImportError("h5py is required for official AgileX data") from error
    with h5py.File(root / "episode.hdf5", "r") as handle:
        if int(handle.attrs.get("fps", -1)) != OFFICIAL_SOURCE_FPS:
            raise ValueError(f"AgileX episode HDF5 must be 30 FPS: {root}")
        if bool(handle.attrs.get("sim", True)):
            raise ValueError(f"AgileX formal data must be real-robot: {root}")
        qpos = validate_qpos14(handle[RAW_QPOS_DATASET][...], label="joint_qpos")
        actions = validate_qpos14(handle[RAW_ACTION_DATASET][...], label="action")
    if qpos.ndim != 2 or actions.shape != qpos.shape or len(actions) < 2:
        raise ValueError(f"AgileX qpos/action arrays must be aligned [T>=2,14]: {root}")
    wrench: dict[str, npt.NDArray[np.float32]] = {}
    if contact:
        with h5py.File(root / "tactile_information.hdf5", "r") as handle:
            for key, dataset in WRENCH_DATASETS.items():
                values = np.asarray(handle[dataset][...], dtype=np.float32)
                if values.shape != (len(actions), 6) or not np.isfinite(values).all():
                    raise ValueError(
                        f"AgileX wrench must be finite [T,6]: {root}/{dataset}"
                    )
                wrench[key] = np.ascontiguousarray(values)
    return qpos, actions, MappingProxyType(wrench)


def audit_official_agilex_episode(
    *,
    root: Path,
    raw_root: Path,
    task: str,
    source_episode_id: int,
    route: AgileXRepoRoute,
    task_prompt: str,
) -> OfficialAgileXEpisode:
    episode_root = Path(root).resolve(strict=True)
    contact = task_has_contact(task)
    meta = _read_json(episode_root / "meta.json")
    if str(meta.get("device_name", "")).lower() not in ALLOWED_DEVICE_NAMES:
        raise ValueError(f"AgileX device metadata mismatch: {episode_root}")
    for name in BASE_FILES + (TACTILE_FILES_REQUIRED if contact else ()):
        path = episode_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"AgileX episode file is missing/unsafe: {path}")
    if not contact and any(
        (episode_root / name).exists() for name in TACTILE_FILES_REQUIRED
    ):
        raise ValueError(
            f"vision-only AgileX episode unexpectedly contains tactile data: {episode_root}"
        )
    qpos, actions, wrench = _hdf_arrays(episode_root, contact=contact)
    expected_rgb = tuple(RGB_FILES)
    expected_tactile = tuple(TACTILE_FILES) if contact else ()
    if route.rgb_keys != expected_rgb or route.tactile_keys != expected_tactile:
        raise ValueError(f"AgileX episode route/image contract mismatch: {task}")
    if route.wrench_keys != tuple(wrench) or route.action_label_source != "commanded":
        raise ValueError(f"AgileX episode route/action contract mismatch: {task}")
    rgb = {key: _video_header(episode_root / name) for key, name in RGB_FILES.items()}
    tactile = (
        {key: _video_header(episode_root / name) for key, name in TACTILE_FILES.items()}
        if contact
        else {}
    )
    counts = {header.frame_count for header in (*rgb.values(), *tactile.values())}
    if len(counts) != 1 or next(iter(counts)) not in (len(actions), len(actions) + 1):
        raise ValueError(
            f"AgileX videos must consistently contain T or T+1 frames: {episode_root}"
        )
    frame_count = next(iter(counts))
    relative = (episode_root / "episode.hdf5").relative_to(
        raw_root.resolve(strict=True)
    )
    valid = np.ones(len(actions), dtype=np.bool_)
    return OfficialAgileXEpisode(
        route=route,
        root=episode_root,
        source_relative_path=relative.as_posix(),
        source_episode_id=source_episode_id,
        task_id=task,
        task_prompt=task_prompt,
        qpos=qpos,
        actions=actions,
        action_valid_mask=valid,
        rgb_headers=MappingProxyType(rgb),
        tactile_headers=MappingProxyType(tactile),
        wrench=wrench,
        terminal_video_frame_trimmed=frame_count == len(actions) + 1,
    )


def discover_official_agilex_episodes(
    *, raw_root: Path, routes: Mapping[str, AgileXRepoRoute]
) -> tuple[OfficialAgileXEpisode, ...]:
    root = Path(raw_root).expanduser().resolve(strict=True)
    if set(routes) != {VISION_ONLY_REPO_ID, VISION_TACTILE_REPO_ID}:
        raise ValueError("AgileX official mixed routes must cover VO and VT repos")
    prompts = load_official_task_prompts(root)
    episodes: list[OfficialAgileXEpisode] = []
    for task in OFFICIAL_TASKS:
        candidates = sorted((root / task).rglob("episode.hdf5"))
        agilex_roots = [
            path.parent for path in candidates if (path.parent / "meta.json").is_file()
        ]
        if len(agilex_roots) != EXPECTED_TASK_EPISODES[task]:
            raise ValueError(
                f"official AgileX episode count mismatch for {task}: {len(agilex_roots)}"
            )
        for episode_id, episode_root in enumerate(agilex_roots):
            repo_id = (
                VISION_TACTILE_REPO_ID
                if task_has_contact(task)
                else VISION_ONLY_REPO_ID
            )
            episodes.append(
                audit_official_agilex_episode(
                    root=episode_root,
                    raw_root=root,
                    task=task,
                    source_episode_id=episode_id,
                    route=routes[repo_id],
                    task_prompt=prompts[task],
                )
            )
    if len(episodes) != OFFICIAL_EPISODE_COUNT:
        raise RuntimeError("official AgileX discovery did not produce 983 episodes")
    return tuple(episodes)


__all__ = (
    "OfficialAgileXEpisode",
    "VideoHeader",
    "audit_official_agilex_episode",
    "discover_official_agilex_episodes",
    "load_official_task_prompts",
)
