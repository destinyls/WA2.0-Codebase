# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Read-only audit of official WorldArena Franka episode artifacts."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from .franka_actions import end_pose8_to_ee10
from .franka_manifest import (
    OFFICIAL_TASKS,
    FrankaDatasetInventory,
    FrankaFileRecord,
)

SOURCE_FPS = 15
END_POSE_PATH = "observations/end_pose"
QPOS_PATH = "observations/qpos"
EMBEDDED_CAMERA_PATHS = (
    "observation/camera/head",
    "observation/camera/left",
)
EPISODE_FILENAMES = (
    "camera_timestamps.json",
    "episode.hdf5",
    "metadata.json",
    "robot_state.json",
    "third_person.mp4",
    "wrist.mp4",
)
TASK_PROMPTS = {
    "clear_up": "Clear up the objects on the table.",
    "pour": "Pour the contents into the target container.",
    "wipe": "Wipe the marked area clean.",
}


@dataclass(frozen=True)
class FrankaEpisode:
    task: str
    source_episode_id: int
    root: Path
    length: int
    files: tuple[FrankaFileRecord, ...]

    @property
    def relative_root(self) -> str:
        return f"{self.task}/episode_{self.source_episode_id:03d}"

    @property
    def hdf5_path(self) -> Path:
        return self.root / "episode.hdf5"

    @property
    def converted_length(self) -> int:
        return self.length - 1


def _json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Franka JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Franka JSON artifact must be an object: {path}")
    return payload


def _validate_regular_tree(data_root: Path, episode_root: Path) -> None:
    root = data_root.resolve(strict=True)
    current = root
    for part in episode_root.relative_to(root).parts:
        current /= part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Franka source path may not contain symlinks: {current}")
    if not episode_root.is_dir():
        raise NotADirectoryError(episode_root)


def _record_map(inventory: FrankaDatasetInventory) -> dict[str, FrankaFileRecord]:
    return {record.relative_path: record for record in inventory.records}


def _validate_timestamps(path: Path, *, length: int) -> None:
    payload = _json_object(path)
    if payload.get("reference_camera") != "third_person":
        raise ValueError("Franka timestamps must use third_person as reference")
    raw = payload.get("raw_timestamps")
    frame_maps = payload.get("frame_maps")
    expected_cameras = {"third_person", "wrist"}
    if not isinstance(raw, dict) or set(raw) != expected_cameras:
        raise ValueError("Franka timestamp raw_timestamps camera set mismatch")
    if not isinstance(frame_maps, dict) or set(frame_maps) != expected_cameras:
        raise ValueError("Franka timestamp frame_maps camera set mismatch")

    raw_arrays: dict[str, npt.NDArray[np.float64]] = {}
    for camera, values in raw.items():
        if (
            not isinstance(values, list)
            or len(values) < 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in values
            )
        ):
            raise ValueError(f"Franka raw_timestamps.{camera} is invalid")
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all() or not np.all(np.diff(array) > 0.0):
            raise ValueError(
                f"Franka raw_timestamps.{camera} must be finite and increasing"
            )
        raw_arrays[camera] = array

    if len(raw_arrays["third_person"]) != length:
        raise ValueError("Franka reference timestamp length mismatch")
    # The pinned official release contains 111 episodes where the unsynchronised
    # wrist capture has one frame fewer or one frame more than the reference.
    # ``frame_maps.wrist`` is the authoritative nearest-frame synchronisation
    # roster used to materialise the aligned wrist video.
    if abs(len(raw_arrays["wrist"]) - length) > 1:
        raise ValueError("Franka wrist timestamp length differs by more than one")

    mapped: dict[str, npt.NDArray[np.int64]] = {}
    for camera, values in frame_maps.items():
        if (
            not isinstance(values, list)
            or len(values) != length
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            )
        ):
            raise ValueError(f"Franka frame_maps.{camera} is invalid")
        indices = np.asarray(values, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(raw_arrays[camera])):
            raise ValueError(f"Franka frame_maps.{camera} index is out of range")
        mapped[camera] = indices

    if not np.array_equal(mapped["third_person"], np.arange(length, dtype=np.int64)):
        raise ValueError("Franka reference frame map is not identity")
    wrist_steps = np.diff(mapped["wrist"])
    if np.any(wrist_steps < 0) or np.any(wrist_steps > 1):
        raise ValueError("Franka wrist frame map is not monotonic nearest-frame data")
    aligned_wrist = raw_arrays["wrist"][mapped["wrist"]]
    insertion = np.searchsorted(
        raw_arrays["wrist"], raw_arrays["third_person"], side="left"
    )
    lower = np.clip(insertion - 1, 0, len(raw_arrays["wrist"]) - 1)
    upper = np.clip(insertion, 0, len(raw_arrays["wrist"]) - 1)
    nearest_error = np.minimum(
        np.abs(raw_arrays["third_person"] - raw_arrays["wrist"][lower]),
        np.abs(raw_arrays["third_person"] - raw_arrays["wrist"][upper]),
    )
    selected_error = np.abs(raw_arrays["third_person"] - aligned_wrist)
    # The frozen release has up to about 2.14 ms of timestamp quantisation
    # excess over the mathematical nearest neighbour.  Keep a narrow bound
    # that accepts that release while rejecting an entire-frame shift.
    maximum_nearest_excess = 1.0 / float(SOURCE_FPS * 30)
    if np.any(selected_error > nearest_error + maximum_nearest_excess):
        raise ValueError("Franka wrist frame map does not select nearest timestamps")
    maximum_alignment_error = 1.5 / float(SOURCE_FPS)
    if np.any(selected_error > maximum_alignment_error):
        raise ValueError("Franka wrist/reference timestamp alignment is too distant")


def _validate_hdf5(path: Path, *, length: int) -> None:
    with h5py.File(path, "r") as handle:
        if int(handle.attrs.get("fps", -1)) != SOURCE_FPS:
            raise ValueError("Franka HDF5 FPS is not 15")
        if bool(handle.attrs.get("sim", True)):
            raise ValueError(
                "Franka official real dataset unexpectedly declares sim=True"
            )
        end_pose = handle.get(END_POSE_PATH)
        qpos = handle.get(QPOS_PATH)
        if end_pose is None or end_pose.shape != (length, 16):
            raise ValueError(
                f"Franka end_pose shape mismatch: {getattr(end_pose, 'shape', None)}"
            )
        if qpos is None or qpos.shape != (length, 14):
            raise ValueError(
                f"Franka qpos shape mismatch: {getattr(qpos, 'shape', None)}"
            )
        for camera_path in EMBEDDED_CAMERA_PATHS:
            camera = handle.get(camera_path)
            if camera is None or camera.shape != (length,):
                raise ValueError(
                    f"Franka embedded camera shape mismatch: {camera_path}"
                )
        sample = np.asarray(end_pose[:, :8], dtype=np.float32)
        if not np.isfinite(sample).all():
            raise ValueError("Franka end_pose contains non-finite values")
        end_pose8_to_ee10(sample)


def audit_episode(
    *,
    data_root: Path,
    inventory: FrankaDatasetInventory,
    task: str,
    source_episode_id: int,
) -> FrankaEpisode:
    """Verify all six official files and the action/camera timeline contract."""

    if task not in OFFICIAL_TASKS or not 0 <= source_episode_id < 200:
        raise ValueError("Franka episode identity is outside the official dataset")
    data_root = Path(data_root).expanduser().resolve(strict=True)
    episode_root = data_root / task / f"episode_{source_episode_id:03d}"
    _validate_regular_tree(data_root, episode_root)
    records_by_path = _record_map(inventory)
    records: list[FrankaFileRecord] = []
    for filename in EPISODE_FILENAMES:
        relative_path = f"{task}/episode_{source_episode_id:03d}/{filename}"
        record = records_by_path.get(relative_path)
        if record is None:
            raise ValueError(f"official inventory is missing {relative_path}")
        record.verify(episode_root / filename)
        records.append(record)

    metadata = _json_object(episode_root / "metadata.json")
    length = metadata.get("frames")
    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise ValueError("Franka metadata frames must be an integer >=2")
    if (
        metadata.get("task_name") != task
        or metadata.get("episode_id") != source_episode_id
        or metadata.get("camera_fps") != SOURCE_FPS
    ):
        raise ValueError("Franka metadata identity/FPS mismatch")
    _validate_timestamps(episode_root / "camera_timestamps.json", length=length)
    _validate_hdf5(episode_root / "episode.hdf5", length=length)
    return FrankaEpisode(
        task=task,
        source_episode_id=source_episode_id,
        root=episode_root,
        length=length,
        files=tuple(records),
    )


def read_end_pose8(episode: FrankaEpisode) -> npt.NDArray[np.float32]:
    """Read base-frame labels as ``[xyz, qx, qy, qz, qw, gripper]``."""

    with h5py.File(episode.hdf5_path, "r") as handle:
        pose = np.asarray(handle[END_POSE_PATH][:, :8], dtype=np.float32)
    if pose.shape != (episode.length, 8) or not np.isfinite(pose).all():
        raise ValueError("Franka end-pose trajectory changed after audit")
    return np.ascontiguousarray(pose)


__all__ = (
    "END_POSE_PATH",
    "FrankaEpisode",
    "SOURCE_FPS",
    "TASK_PROMPTS",
    "audit_episode",
    "read_end_pose8",
)
