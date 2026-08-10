# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import os
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

from n0_twam.integrations.univtac.convert_lerobot import convert_episode_to_sink
from n0_twam.integrations.univtac.hdf5_reader import audit_episode, sha256_file


def _write_episode(path: Path, *, episode_id: int, offset: float) -> str:
    relative_path = f"insert_HDMI/clean/{episode_id}.hdf5"
    absolute_path = path / relative_path
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    images = np.zeros((6, 8, 10, 3), dtype=np.uint8)
    joint = np.arange(54, dtype=np.float32).reshape(6, 9) + offset
    with h5py.File(absolute_path, "w") as handle:
        handle.create_dataset("step", data=np.arange(6, dtype=np.int64))
        handle.create_dataset("embodiment/joint", data=joint)
        handle.create_dataset("observation/head/rgb", data=images)
        handle.create_dataset("observation/wrist/rgb", data=images + np.uint8(1))
        handle.create_dataset(
            "tactile/left_gsmini/rgb_marker",
            data=images + np.uint8(2),
        )
        handle.create_dataset(
            "tactile/right_gsmini/rgb_marker",
            data=images + np.uint8(3),
        )
    return relative_path


class _RecordingSink:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.saved_episodes = 0

    def add_frame(self, frame: dict[str, Any], task: str, timestamp: float) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved_episodes += 1


def test_converter_rejects_replacement_between_hash_and_hdf5_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_path = _write_episode(tmp_path, episode_id=3, offset=0.0)
    replacement_relative_path = _write_episode(
        tmp_path,
        episode_id=4,
        offset=1_000.0,
    )
    source_path = tmp_path / relative_path
    replacement_path = tmp_path / replacement_relative_path
    record = audit_episode(tmp_path, relative_path, split="train", hash_file=True)
    original_h5py_file = h5py.File
    replaced = False

    def replace_then_open(name: object, *args: object, **kwargs: object) -> Any:
        nonlocal replaced
        if not replaced:
            os.replace(replacement_path, source_path)
            replaced = True
        return original_h5py_file(name, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", replace_then_open)
    sink = _RecordingSink()

    with pytest.raises(ValueError, match="source artifact .* changed"):
        convert_episode_to_sink(record, sink=sink, fps=10.0)

    assert replaced
    assert sha256_file(source_path) != record.sha256
    assert sink.frames == []
    assert sink.saved_episodes == 0


def test_converter_reverifies_path_identity_after_consumption(
    tmp_path: Path,
) -> None:
    relative_path = _write_episode(tmp_path, episode_id=3, offset=0.0)
    replacement_relative_path = _write_episode(
        tmp_path,
        episode_id=4,
        offset=1_000.0,
    )
    source_path = tmp_path / relative_path
    replacement_path = tmp_path / replacement_relative_path
    record = audit_episode(tmp_path, relative_path, split="train", hash_file=True)

    class ReplacingSink(_RecordingSink):
        def add_frame(
            self,
            frame: dict[str, Any],
            task: str,
            timestamp: float,
        ) -> None:
            super().add_frame(frame, task, timestamp)
            if replacement_path.exists():
                os.replace(replacement_path, source_path)

    sink = ReplacingSink()

    with pytest.raises(ValueError, match="source artifact .* changed"):
        convert_episode_to_sink(record, sink=sink, fps=10.0)

    assert sink.frames
    assert sink.saved_episodes == 0
    assert sha256_file(source_path) != record.sha256


def test_converter_reverifies_sha256_after_consumption(tmp_path: Path) -> None:
    relative_path = _write_episode(tmp_path, episode_id=3, offset=0.0)
    source_path = tmp_path / relative_path
    record = audit_episode(tmp_path, relative_path, split="train", hash_file=True)
    audited_metadata = source_path.stat()

    class MutatingSink(_RecordingSink):
        def add_frame(
            self,
            frame: dict[str, Any],
            task: str,
            timestamp: float,
        ) -> None:
            super().add_frame(frame, task, timestamp)
            if len(self.frames) != record.converted_length:
                return
            with source_path.open("r+b") as source:
                source.seek(-1, os.SEEK_END)
                original_byte = source.read(1)
                source.seek(-1, os.SEEK_END)
                source.write(bytes((original_byte[0] ^ 1,)))
                source.flush()
                os.fsync(source.fileno())
            os.utime(
                source_path,
                ns=(audited_metadata.st_atime_ns, audited_metadata.st_mtime_ns),
            )

    sink = MutatingSink()

    with pytest.raises(ValueError, match="sha256 changed during consumption"):
        convert_episode_to_sink(record, sink=sink, fps=10.0)

    assert sink.frames
    assert sink.saved_episodes == 0
    assert source_path.stat().st_size == record.size_bytes
    assert source_path.stat().st_mtime_ns == audited_metadata.st_mtime_ns


def test_audit_rejects_symlink_source(tmp_path: Path) -> None:
    target_relative_path = _write_episode(tmp_path, episode_id=4, offset=0.0)
    source_relative_path = "insert_HDMI/clean/3.hdf5"
    source_path = tmp_path / source_relative_path
    source_path.symlink_to(tmp_path / target_relative_path)

    with pytest.raises(ValueError, match="symlink"):
        audit_episode(
            tmp_path,
            source_relative_path,
            split="train",
            hash_file=True,
        )


def test_converter_rejects_symlink_substitution_after_audit(tmp_path: Path) -> None:
    relative_path = _write_episode(tmp_path, episode_id=3, offset=0.0)
    target_relative_path = _write_episode(tmp_path, episode_id=4, offset=0.0)
    source_path = tmp_path / relative_path
    target_path = tmp_path / target_relative_path
    record = audit_episode(tmp_path, relative_path, split="train", hash_file=True)
    source_path.unlink()
    source_path.symlink_to(target_path)

    with pytest.raises(ValueError, match="symlink"):
        convert_episode_to_sink(record, sink=_RecordingSink(), fps=10.0)
