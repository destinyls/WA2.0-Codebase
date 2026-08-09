# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""UniVTAC HDF5 to LeRobot v2.1 conversion with explicit provenance."""

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence, TypeAlias

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from n0_twam.actions.qpos8 import CHANNEL_NAMES

from .hdf5_reader import (
    open_verified_hdf5,
    read_image_at,
    read_validated_steps,
    sha256_file,
)
from .schema import (
    IMAGE_PATHS,
    JOINT_PATH,
    OUTPUT_COLOR_SPACE,
    SOURCE_IMAGE_ENCODING_CONTRACT,
    UniVTACEpisodeRecord,
)

FloatArray: TypeAlias = npt.NDArray[np.float32]
UInt8Image: TypeAlias = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class UniVTACFrame:
    """One recorded row paired with the next recorded-row qpos target.

    ``source_frame_index`` retains UniVTAC's raw simulator ``step`` counter.
    Timestamps use the recorded-row clock because the simulator step frequency
    is not encoded in the dataset and can differ from the exported video FPS.
    """

    images: tuple[tuple[str, UInt8Image], ...]
    state: FloatArray
    action: FloatArray
    relative_path: str
    source_row_index: int
    source_frame_index: int
    source_timestamp: float
    timeline_timestamp: float

    def to_lerobot_dict(self) -> dict[str, Any]:
        return {
            **dict(self.images),
            "observation.state": self.state.copy(),
            "action": self.action.copy(),
            "source.relative_path": self.relative_path,
            "source.row_index": np.asarray(((self.source_row_index,),), dtype=np.int64),
            "source.frame_index": np.asarray(
                ((self.source_frame_index,),), dtype=np.int64
            ),
            "source.timestamp": np.asarray(
                ((self.source_timestamp,),), dtype=np.float32
            ),
        }


class LeRobotSink(Protocol):
    """Narrow writer surface shared by LeRobot 0.3.3 and test sinks."""

    def add_frame(self, frame: dict[str, Any], task: str, timestamp: float) -> None: ...

    def save_episode(self) -> None: ...


def _validate_hdf5_storage(handle: h5py.File, *, source_label: str) -> None:
    """Reject source bytes that are not physically contained in this HDF5 file."""

    seen_groups: set[int] = set()

    def visit(group: h5py.Group, prefix: str) -> None:
        group_address = int(h5py.h5o.get_info(group.id).addr)
        if group_address in seen_groups:
            return
        seen_groups.add(group_address)
        for name in group.keys():
            path = f"{prefix}/{name}" if prefix else str(name)
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                raise ValueError(
                    f"HDF5 ExternalLink is forbidden for {source_label}: {path}"
                )
            item = group.get(name)
            if item is None:
                raise ValueError(f"HDF5 link cannot be resolved: {source_label}:{path}")
            if isinstance(item, h5py.Dataset):
                if item.external is not None:
                    raise ValueError(
                        f"HDF5 external storage is forbidden for "
                        f"{source_label}: {path}"
                    )
                if item.is_virtual:
                    raise ValueError(
                        f"HDF5 virtual dataset is forbidden for "
                        f"{source_label}: {path}"
                    )
            elif isinstance(item, h5py.Group):
                visit(item, path)

    visit(handle, "")


def _resolve_symlink_free_directory(path: Path) -> tuple[Path, tuple[int, int]]:
    lexical_path = Path(os.path.abspath(path))
    current = Path(lexical_path.anchor)
    for component in lexical_path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ValueError(f"LeRobot output path is unavailable: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"LeRobot output path contains a symlink: {current}")
    try:
        root_metadata = lexical_path.lstat()
    except OSError as exc:
        raise ValueError(f"LeRobot output path is unavailable: {lexical_path}") from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"LeRobot output root is not a directory: {lexical_path}")
    return lexical_path.resolve(strict=True), (
        root_metadata.st_dev,
        root_metadata.st_ino,
    )


def build_lerobot_table_inventory(output_root: Path) -> list[dict[str, object]]:
    """Hash metadata, Parquet tables, and every source video consumed later."""

    root, root_identity = _resolve_symlink_free_directory(output_root)

    inventory: list[dict[str, object]] = []
    for directory in (root / "meta", root / "data", root / "videos"):
        if directory.is_symlink():
            raise ValueError(f"LeRobot table path contains a symlink: {directory}")
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError(f"LeRobot table path is not a directory: {directory}")
        for path in directory.rglob("*"):
            initial = path.lstat()
            if stat.S_ISLNK(initial.st_mode):
                raise ValueError(f"LeRobot table path contains a symlink: {path}")
            resolved = path.resolve(strict=True)
            try:
                relative_path = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"LeRobot table path resolved outside output root: {path}"
                ) from exc
            if stat.S_ISDIR(initial.st_mode):
                continue
            if not stat.S_ISREG(initial.st_mode):
                raise ValueError(f"LeRobot table path is not a regular file: {path}")
            digest = sha256_file(resolved)
            final = path.lstat()
            initial_identity = (
                initial.st_dev,
                initial.st_ino,
                initial.st_size,
                initial.st_mtime_ns,
            )
            final_identity = (
                final.st_dev,
                final.st_ino,
                final.st_size,
                final.st_mtime_ns,
            )
            if initial_identity != final_identity or stat.S_ISLNK(final.st_mode):
                raise ValueError(f"LeRobot table path changed while hashing: {path}")
            inventory.append(
                {
                    "relative_path": relative_path,
                    "size_bytes": final.st_size,
                    "sha256": digest,
                }
            )
    final_root, final_root_identity = _resolve_symlink_free_directory(output_root)
    if final_root != root or final_root_identity != root_identity:
        raise ValueError("LeRobot output root changed while building inventory")
    if not inventory:
        raise ValueError(f"LeRobot table inventory is empty: {root}")
    return sorted(inventory, key=lambda item: str(item["relative_path"]))


def freeze_episode_action_config(output_root: Path) -> None:
    """Add the exact action segments consumed by ``LatentLeRobotDataset``."""

    episodes_path = output_root / "meta" / "episodes.jsonl"
    try:
        raw_lines = episodes_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"unable to read LeRobot episodes: {episodes_path}") from exc
    if not raw_lines:
        raise ValueError(f"LeRobot episodes metadata is empty: {episodes_path}")

    output_lines: list[str] = []
    seen_episode_ids: set[int] = set()
    for line_number, raw_line in enumerate(raw_lines, start=1):
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid LeRobot episode metadata at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"LeRobot episode line {line_number} must be an object")
        raw_episode_index = record.get("episode_index")
        raw_length = record.get("length")
        if (
            isinstance(raw_episode_index, bool)
            or not isinstance(raw_episode_index, Integral)
            or isinstance(raw_length, bool)
            or not isinstance(raw_length, Integral)
        ):
            raise ValueError(f"invalid LeRobot episode metadata at line {line_number}")
        episode_index = int(raw_episode_index)
        length = int(raw_length)
        tasks = record.get("tasks")
        if (
            episode_index < 0
            or episode_index in seen_episode_ids
            or length <= 0
            or not isinstance(tasks, list)
            or not tasks
            or not isinstance(tasks[0], str)
            or not tasks[0].strip()
        ):
            raise ValueError(f"invalid LeRobot episode metadata at line {line_number}")
        seen_episode_ids.add(episode_index)
        action_config = [
            {
                "start_frame": 0,
                "end_frame": length,
                "action_text": tasks[0],
            }
        ]
        existing = record.get("action_config")
        if existing is not None and existing != action_config:
            raise ValueError(
                f"episode {episode_index} has an incompatible action_config"
            )
        record["action_config"] = action_config
        output_lines.append(
            json.dumps(record, ensure_ascii=True, separators=(",", ":"))
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=episodes_path.parent,
            prefix=f".{episodes_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write("\n".join(output_lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, episodes_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _immutable_float32(values: npt.ArrayLike) -> FloatArray:
    output = np.asarray(values, dtype=np.float32).copy()
    output.setflags(write=False)
    return output


def iter_episode_frames(
    record: UniVTACEpisodeRecord,
    *,
    fps: float,
) -> Iterator[UniVTACFrame]:
    """Yield ``joint[t,:8] -> joint[t+1,:8]`` conversion rows."""

    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    with open_verified_hdf5(record) as handle:
        _validate_hdf5_storage(handle, source_label=record.relative_path)
        joint = np.asarray(handle[JOINT_PATH][:, :8], dtype=np.float32)
        if joint.shape != (record.length, 8):
            raise ValueError(
                f"audited joint shape drift for {record.relative_path}: {joint.shape}"
            )
        source_steps = read_validated_steps(
            handle,
            expected_length=record.length,
            source_label=record.relative_path,
            temporal_selection=record.temporal_selection,
        )
        assert record.usable_end is not None
        for frame_index in range(record.usable_start, record.usable_end - 1):
            images = tuple(
                (
                    feature_name,
                    read_image_at(
                        handle,
                        hdf5_path=source_path,
                        frame_index=frame_index,
                    ),
                )
                for feature_name, source_path in IMAGE_PATHS
            )
            source_step = int(source_steps[frame_index])
            timeline_index = frame_index - record.usable_start
            yield UniVTACFrame(
                images=images,
                state=_immutable_float32(joint[frame_index]),
                action=_immutable_float32(joint[frame_index + 1]),
                relative_path=record.relative_path,
                source_row_index=frame_index,
                source_frame_index=source_step,
                source_timestamp=frame_index / fps,
                timeline_timestamp=timeline_index / fps,
            )


def build_lerobot_features(
    record: UniVTACEpisodeRecord,
) -> dict[str, dict[str, object]]:
    """Build the exact LeRobot v2.1 feature schema consumed by N0."""

    features: dict[str, dict[str, object]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(CHANNEL_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(CHANNEL_NAMES),
        },
        "source.relative_path": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
        "source.frame_index": {
            "dtype": "int64",
            "shape": (1, 1),
            "names": None,
        },
        "source.row_index": {
            "dtype": "int64",
            "shape": (1, 1),
            "names": None,
        },
        "source.timestamp": {
            "dtype": "float32",
            "shape": (1, 1),
            "names": None,
        },
    }
    for feature_name, shape in record.image_shapes:
        features[feature_name] = {
            "dtype": "video",
            "shape": shape,
            "names": ("height", "width", "channels"),
        }
    return features


def convert_episode_to_sink(
    record: UniVTACEpisodeRecord,
    *,
    sink: LeRobotSink,
    fps: float,
    task_prompt: str | None = None,
) -> int:
    """Write one audited episode without exposing the source's terminal frame."""

    prompt = task_prompt or record.task
    converted_count = 0
    for converted_frame in iter_episode_frames(record, fps=fps):
        sink.add_frame(
            converted_frame.to_lerobot_dict(),
            task=prompt,
            timestamp=converted_frame.timeline_timestamp,
        )
        converted_count += 1
    if converted_count != record.converted_length:
        raise RuntimeError("converted frame count does not match next-step contract")
    sink.save_episode()
    return converted_count


def write_lerobot_dataset(
    records: Sequence[UniVTACEpisodeRecord],
    *,
    output_root: str,
    repo_id: str,
    fps: int = 10,
    image_writer_threads: int = 8,
) -> dict[str, object]:
    """Materialize audited records with LeRobot 0.3.3's v2.1 writer."""

    if not records:
        raise ValueError("records must be non-empty")
    if fps <= 0:
        raise ValueError("fps must be positive")
    feature_schema = build_lerobot_features(records[0])
    for record in records[1:]:
        if build_lerobot_features(record) != feature_schema:
            raise ValueError("all converted episodes must share one feature schema")
    try:
        from lerobot.datasets.lerobot_dataset import (  # type: ignore[import-not-found]
            LeRobotDataset,
        )
    except ImportError as exc:  # pragma: no cover - optional conversion dependency
        raise ImportError(
            "LeRobot 0.3.3 is required for materialization; install it as "
            "documented in docs/INSTALL.md"
        ) from exc

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        robot_type="franka_panda",
        features=feature_schema,
        use_videos=True,
        image_writer_threads=image_writer_threads,
    )
    converted_frames = 0
    try:
        for record in records:
            converted_frames += convert_episode_to_sink(
                record,
                sink=dataset,
                fps=float(fps),
            )
    finally:
        dataset.stop_image_writer()
    freeze_episode_action_config(Path(output_root))
    return {
        "schema_version": 2,
        "action_schema": "qpos8_next_step",
        "source_image_encoding_contract": SOURCE_IMAGE_ENCODING_CONTRACT,
        "output_color_space": OUTPUT_COLOR_SPACE,
        "repo_id": repo_id,
        "output_root": output_root,
        "fps": fps,
        "episode_count": len(records),
        "frame_count": converted_frames,
        "source_relative_paths": [record.relative_path for record in records],
        "source_sha256": [record.sha256 for record in records],
        "temporal_contract": "content_addressed_source_range_v1",
        "source_temporal_selections": [
            {
                "relative_path": record.relative_path,
                "raw_length": record.length,
                "usable_source_range": [record.usable_start, record.usable_end],
                "converted_length": record.converted_length,
                "dropped_prefix_rows": record.usable_start,
                "dropped_suffix_rows": (
                    record.length - record.temporal_selection.usable_end
                ),
                "step_discontinuities_after_rows": list(
                    record.step_discontinuities_after_rows
                ),
                "temporal_policy": record.temporal_policy,
            }
            for record in records
        ],
        "table_inventory": build_lerobot_table_inventory(Path(output_root)),
    }
