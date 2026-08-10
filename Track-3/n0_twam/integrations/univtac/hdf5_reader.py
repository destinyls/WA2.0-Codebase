# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Read-only, fail-closed UniVTAC HDF5 access."""

import hashlib
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Sequence, TypeAlias

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from .schema import (
    IMAGE_PATHS,
    JOINT_PATH,
    LEGACY_UNIVTAC_JPEG_CONTRACT,
    REQUIRED_HDF5_PATHS,
    SOURCE_IMAGE_ENCODING_CONTRACT,
    STANDARD_JPEG_RGB_CONTRACT,
    STEP_PATH,
    TRACK31_TASKS,
    UniVTACEpisodeRecord,
    canonical_task_scope,
)
from .temporal_contract import (
    TemporalSelection,
    resolve_temporal_selection,
    validate_declared_temporal_selection,
    validate_temporal_selection,
)

UInt8Image: TypeAlias = npt.NDArray[np.uint8]


def sha256_file(path: Path) -> str:
    """Hash a source artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
    handle.seek(0)
    return digest.hexdigest()


def _source_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _verify_open_source_identity(
    record: UniVTACEpisodeRecord,
    source: BinaryIO,
    *,
    expected_identity: tuple[int, int, int, int],
    phase: str,
) -> None:
    descriptor_metadata = os.fstat(source.fileno())
    if not stat.S_ISREG(descriptor_metadata.st_mode):
        raise ValueError(
            f"source artifact must be a regular file: {record.relative_path}"
        )
    if _source_identity(descriptor_metadata) != expected_identity:
        raise ValueError(
            f"source artifact descriptor changed {phase}: {record.relative_path}"
        )
    try:
        path_metadata = os.stat(record.absolute_path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(
            f"source artifact path changed {phase}: {record.relative_path}"
        ) from exc
    if stat.S_ISLNK(path_metadata.st_mode):
        raise ValueError(
            f"source artifact became a symlink {phase}: {record.relative_path}"
        )
    if not stat.S_ISREG(path_metadata.st_mode):
        raise ValueError(
            f"source artifact must remain a regular file: {record.relative_path}"
        )
    if _source_identity(path_metadata) != expected_identity:
        raise ValueError(
            f"source artifact path identity changed {phase}: {record.relative_path}"
        )


def _verify_temporal_record(
    record: UniVTACEpisodeRecord,
    handle: h5py.File,
) -> None:
    steps = _read_step_counters(
        handle,
        expected_length=record.length,
        source_label=record.relative_path,
    )
    resolved_selection = resolve_temporal_selection(
        steps,
        relative_path=record.relative_path,
        source_sha256=record.sha256,
    )
    if resolved_selection != record.temporal_selection:
        raise ValueError(
            f"source temporal selection changed after audit: {record.relative_path}"
        )
    validate_declared_temporal_selection(
        relative_path=record.relative_path,
        source_sha256=record.sha256,
        raw_length=record.length,
        split=record.split,
        selection=record.temporal_selection,
    )


@contextmanager
def open_verified_hdf5(
    record: UniVTACEpisodeRecord,
) -> Iterator[h5py.File]:
    """Bind hashing, HDF5 reads, and final verification to one source fd."""

    if record.sha256 is None:
        message = "source record has no sha256 and cannot be verified: "
        raise ValueError(message + record.relative_path)
    try:
        descriptor = os.open(
            record.absolute_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        message = "source artifact is unavailable or is a symlink: "
        raise ValueError(message + record.relative_path) from exc
    try:
        source = os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise

    with source:
        opened_metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise ValueError(
                f"source artifact must be a regular file: {record.relative_path}"
            )
        if opened_metadata.st_size != record.size_bytes:
            raise ValueError(
                "source artifact size changed after audit: "
                f"{record.relative_path} "
                f"({record.size_bytes} -> {opened_metadata.st_size})"
            )
        opened_identity = _source_identity(opened_metadata)
        actual_sha256 = _sha256_stream(source)
        _verify_open_source_identity(
            record,
            source,
            expected_identity=opened_identity,
            phase="while verifying",
        )
        if actual_sha256 != record.sha256:
            raise ValueError(
                "source artifact sha256 changed after audit: "
                f"{record.relative_path} ({record.sha256} -> {actual_sha256})"
            )

        try:
            with h5py.File(source, "r") as handle:
                _verify_open_source_identity(
                    record,
                    source,
                    expected_identity=opened_identity,
                    phase="while opening HDF5",
                )
                _verify_temporal_record(record, handle)
                yield handle
        finally:
            final_sha256 = _sha256_stream(source)
            _verify_open_source_identity(
                record,
                source,
                expected_identity=opened_identity,
                phase="during consumption",
            )
            if final_sha256 != record.sha256:
                raise ValueError(
                    "source artifact sha256 changed during consumption: "
                    f"{record.relative_path} "
                    f"({record.sha256} -> {final_sha256})"
                )


def verify_source_record(record: UniVTACEpisodeRecord) -> None:
    """Revalidate an audited source through one stable HDF5 descriptor."""

    with open_verified_hdf5(record):
        pass


def _read_step_counters(
    handle: h5py.File,
    *,
    expected_length: int,
    source_label: str,
) -> npt.NDArray[np.int64]:
    """Read structurally valid, non-negative simulator step counters."""

    if STEP_PATH not in handle:
        return np.arange(expected_length, dtype=np.int64)
    raw_steps = np.asarray(handle[STEP_PATH][:])
    if raw_steps.ndim != 1 or raw_steps.shape[0] != expected_length:
        raise ValueError(
            f"step must have shape [{expected_length}] for {source_label}, "
            f"got {raw_steps.shape}"
        )
    if not np.issubdtype(raw_steps.dtype, np.integer):
        raise ValueError(f"step values must be integers for {source_label}")
    steps = raw_steps.astype(np.int64, copy=False)
    if np.any(steps < 0):
        raise ValueError(f"step values must be non-negative for {source_label}")
    output = np.ascontiguousarray(steps)
    output.setflags(write=False)
    return output


def read_validated_steps(
    handle: h5py.File,
    *,
    expected_length: int,
    source_label: str,
    temporal_selection: TemporalSelection | None = None,
) -> npt.NDArray[np.int64]:
    """Return raw step counters under a strict or pinned temporal selection."""

    output = _read_step_counters(
        handle,
        expected_length=expected_length,
        source_label=source_label,
    )
    if temporal_selection is None:
        if output.size > 1 and np.any(np.diff(output) <= 0):
            raise ValueError(
                f"step values must be strictly increasing for {source_label}"
            )
    else:
        validate_temporal_selection(
            output,
            selection=temporal_selection,
            source_label=source_label,
        )
    return output


def resolve_source_path(data_root: Path, relative_path: str) -> tuple[Path, str]:
    """Resolve one relative HDF5 path and reject root escapes."""

    root = data_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    raw_path = Path(relative_path)
    if raw_path.is_absolute():
        raise ValueError(f"source path must be relative: {relative_path}")
    if ".." in raw_path.parts:
        raise ValueError(f"source path may not contain '..': {relative_path}")
    candidate = root
    for component in raw_path.parts:
        candidate /= component
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise ValueError(
                f"source artifact is unavailable: {relative_path}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"source path must not contain symlinks: {relative_path}")
    absolute_path = candidate.resolve(strict=True)
    if root not in absolute_path.parents:
        raise ValueError(f"source path escapes data root: {relative_path}")
    if absolute_path.suffix not in {".hdf5", ".h5"}:
        raise ValueError(f"source path is not HDF5: {relative_path}")
    normalized = absolute_path.relative_to(root).as_posix()
    return absolute_path, normalized


def _compressed_bytes(payload: Any) -> npt.NDArray[np.uint8]:
    if isinstance(payload, np.ndarray) and payload.ndim == 1:
        return payload.astype(np.uint8, copy=False)
    if isinstance(payload, (bytes, bytearray, np.bytes_)):
        return np.frombuffer(bytes(payload), dtype=np.uint8)
    raise TypeError(f"Unsupported compressed image payload: {type(payload)!r}")


def decode_image_payload(
    payload: Any,
    *,
    encoding_contract: str = SOURCE_IMAGE_ENCODING_CONTRACT,
) -> UInt8Image:
    """Decode one source frame to RGB under an explicit writer contract."""

    supported_contracts = {
        LEGACY_UNIVTAC_JPEG_CONTRACT,
        STANDARD_JPEG_RGB_CONTRACT,
    }
    if encoding_contract not in supported_contracts:
        raise ValueError(f"Unsupported image encoding contract: {encoding_contract!r}")

    if isinstance(payload, np.ndarray) and payload.ndim == 3:
        # Raw HWC arrays in UniVTAC are already simulator RGB.
        image = payload
    else:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - production dependency guard
            raise ImportError(
                "opencv-python is required to decode compressed UniVTAC frames"
            ) from exc
        image = cv2.imdecode(_compressed_bytes(payload), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV failed to decode a UniVTAC image payload")
        if encoding_contract == STANDARD_JPEG_RGB_CONTRACT:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # The legacy UniVTAC writer passed simulator RGB directly to
        # cv2.imencode. OpenCV decode passthrough therefore restores the source
        # numeric RGB ordering; applying BGR2RGB would swap red and blue.
    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[-1] != 3:
        raise ValueError(f"Expected image [H,W,3], got {image_array.shape}")
    if image_array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image_array.dtype}")
    output = np.ascontiguousarray(image_array)
    output.setflags(write=False)
    return output


def read_image_at(
    handle: h5py.File,
    *,
    hdf5_path: str,
    frame_index: int,
    encoding_contract: str = SOURCE_IMAGE_ENCODING_CONTRACT,
) -> UInt8Image:
    return decode_image_payload(
        handle[hdf5_path][frame_index],
        encoding_contract=encoding_contract,
    )


def read_qpos8(path: Path) -> npt.NDArray[np.float32]:
    """Load the executable 7-joint + gripper trajectory from one episode."""

    with h5py.File(path, "r") as handle:
        values = np.asarray(handle[JOINT_PATH][:, :8], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"Expected qpos8 [T,8], got {values.shape}: {path}")
    if values.shape[0] < 2:
        raise ValueError(f"Episode must contain at least two qpos frames: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Episode contains non-finite qpos8 values: {path}")
    return values


def audit_episode(
    data_root: Path,
    relative_path: str,
    *,
    split: str,
    hash_file: bool = True,
    allowed_tasks: Sequence[str] = TRACK31_TASKS,
) -> UniVTACEpisodeRecord:
    """Validate one episode under an explicit task scope without mutating it."""

    if split not in {"train", "validation", "quarantine"}:
        raise ValueError(f"Unsupported split: {split!r}")
    absolute_path, normalized_path = resolve_source_path(data_root, relative_path)
    path = Path(normalized_path)
    if len(path.parts) != 3 or path.parts[1] != "clean":
        raise ValueError(
            "source path must match <task>/clean/<episode_id>.hdf5: "
            f"{normalized_path}"
        )
    if not path.stem.isdecimal():
        raise ValueError(f"source episode filename must be numeric: {normalized_path}")
    task = path.parts[0]
    task_scope = canonical_task_scope(allowed_tasks)
    if task not in task_scope:
        raise ValueError(
            f"UniVTAC task {task!r} is outside allowed scope {task_scope!r}"
        )

    source_sha256 = sha256_file(absolute_path) if hash_file else None
    with h5py.File(absolute_path, "r") as handle:
        missing = [path for path in REQUIRED_HDF5_PATHS if path not in handle]
        if missing:
            raise KeyError(f"Missing required HDF5 datasets {missing}: {absolute_path}")
        joint = handle[JOINT_PATH]
        if joint.ndim != 2 or joint.shape[1] < 8:
            raise ValueError(
                f"Expected joint [T,>=8], got {joint.shape}: {absolute_path}"
            )
        length = int(joint.shape[0])
        joint_shape = tuple(int(value) for value in joint.shape)
        if length < 2:
            raise ValueError(
                f"Episode must contain at least two frames: {absolute_path}"
            )

        lengths = {JOINT_PATH: length}
        image_shapes: list[tuple[str, tuple[int, int, int]]] = []
        for feature_name, source_path in IMAGE_PATHS:
            dataset = handle[source_path]
            lengths[source_path] = int(dataset.shape[0])
            first_image = read_image_at(
                handle,
                hdf5_path=source_path,
                frame_index=0,
            )
            height, width, channels = first_image.shape
            image_shapes.append(
                (feature_name, (int(height), int(width), int(channels)))
            )
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Unaligned episode lengths {lengths}: {absolute_path}")

        qpos8 = np.asarray(joint[:, :8], dtype=np.float32)
        if not np.isfinite(qpos8).all():
            raise ValueError(
                f"Episode contains non-finite qpos8 values: {absolute_path}"
            )
        raw_steps = _read_step_counters(
            handle,
            expected_length=length,
            source_label=normalized_path,
        )

    # A hash is mandatory to accept a declared exception. For normal episodes,
    # ``hash_file=False`` retains the lightweight audit behavior used by tests.
    has_discontinuity = raw_steps.size > 1 and np.any(np.diff(raw_steps) <= 0)
    temporal_sha256 = source_sha256
    if has_discontinuity and temporal_sha256 is None:
        temporal_sha256 = sha256_file(absolute_path)
    temporal_selection = resolve_temporal_selection(
        raw_steps,
        relative_path=normalized_path,
        source_sha256=temporal_sha256,
    )
    validate_declared_temporal_selection(
        relative_path=normalized_path,
        source_sha256=temporal_sha256,
        raw_length=length,
        split=split,
        selection=temporal_selection,
    )

    return UniVTACEpisodeRecord(
        relative_path=normalized_path,
        absolute_path=absolute_path,
        task=task,
        split=split,
        length=length,
        length_source=f"{JOINT_PATH}.shape[0]",
        joint_shape=joint_shape,
        image_shapes=tuple(image_shapes),
        size_bytes=absolute_path.stat().st_size,
        sha256=source_sha256,
        usable_start=temporal_selection.usable_start,
        usable_end=temporal_selection.usable_end,
        step_discontinuities_after_rows=(temporal_selection.discontinuities_after_rows),
        temporal_policy=temporal_selection.policy,
    )
