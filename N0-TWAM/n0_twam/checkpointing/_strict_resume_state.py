"""Safe atomic I/O and RNG serialization for strict-resume sidecars."""

from __future__ import annotations

import json
import math
import os
import random
import stat
import tempfile
from pathlib import Path
from typing import Mapping, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load_file, save_file

RNG_STATE_SCHEMA_VERSION = 1

PythonRngState: TypeAlias = tuple[int, tuple[int, ...], float | None]
NumpyRngState: TypeAlias = tuple[str, NDArray[np.uint32], int, int, float]


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _regular_file(path: Path) -> os.stat_result:
    try:
        result = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"missing strict-resume sidecar: {path}") from None
    if (
        stat.S_ISLNK(result.st_mode)
        or not stat.S_ISREG(result.st_mode)
        or result.st_size <= 0
    ):
        raise ValueError(f"sidecar must be a non-empty regular non-symlink: {path}")
    return result


def _strict_json(path: Path) -> object:
    _regular_file(path)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read strict JSON sidecar: {path}") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _refuse_symlink_target(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing to replace sidecar symlink: {path}")


def write_json_atomic(path: Path, payload: object) -> None:
    """Write JSON via fsync and same-directory atomic replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _refuse_symlink_target(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_safetensors_atomic(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    """Write tensors without pickle, then atomically publish the complete file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _refuse_symlink_target(path)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(name)
    temporary: Path | None = temporary_path
    try:
        safe_tensors = {
            key: value.detach().cpu().contiguous() for key, value in tensors.items()
        }
        save_file(safe_tensors, temporary_path)
        descriptor = os.open(temporary_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary_path, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _float_hex(value: object, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite float")
    return result.hex()


def _parse_float_hex(value: object, label: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a canonical float hex string")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a canonical float hex string") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise ValueError(f"{label} must be a canonical finite float hex string")
    return result


def capture_rng_state() -> dict[str, object]:
    numpy_state = cast(NumpyRngState, np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
        "torch": torch.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
    }


def _validated_rng_state_components(
    state: object,
    *,
    expected_cuda_device_count: int | None = None,
) -> tuple[PythonRngState, NumpyRngState, torch.Tensor, tuple[torch.Tensor, ...]]:
    if not isinstance(state, dict) or set(state) != {
        "python",
        "numpy",
        "torch",
        "cuda",
    }:
        raise ValueError("RNG state has an invalid field set")
    python_state = cast(PythonRngState, state["python"])
    python_rng = random.Random()
    python_rng.setstate(python_state)
    numpy_state = cast(NumpyRngState, state["numpy"])
    numpy_rng = np.random.RandomState()
    numpy_rng.set_state(numpy_state)
    cpu = state["torch"]
    cuda = state["cuda"]
    if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise ValueError("CPU torch RNG state must be a 1D uint8 tensor")
    torch.Generator(device="cpu").set_state(cpu)
    if not isinstance(cuda, (list, tuple)):
        raise ValueError("CUDA RNG state must be a sequence")
    expected = (
        torch.cuda.device_count()
        if expected_cuda_device_count is None
        else expected_cuda_device_count
    )
    if len(cuda) != expected:
        raise ValueError(f"CUDA RNG state count mismatch: {len(cuda)} vs {expected}")
    for index, value in enumerate(cuda):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
        ):
            raise ValueError(f"CUDA RNG state {index} must be a 1D uint8 tensor")
        if torch.cuda.is_available():
            torch.Generator(device=torch.device("cuda", index)).set_state(value)
    return python_state, numpy_state, cpu, tuple(cuda)


def validate_rng_state(
    state: object,
    *,
    expected_cuda_device_count: int | None = None,
) -> dict[str, object]:
    _validated_rng_state_components(
        state,
        expected_cuda_device_count=expected_cuda_device_count,
    )
    return cast(dict[str, object], state)


def _rank_values(rank: int, world_size: int) -> tuple[int, int]:
    checked_world = _integer(world_size, "world_size", minimum=1)
    checked_rank = _integer(rank, "rank")
    if checked_rank >= checked_world:
        raise ValueError("rank must be smaller than world_size")
    return checked_rank, checked_world


def _rng_paths(checkpoint_dir: Path, rank: int) -> tuple[Path, Path]:
    root = Path(checkpoint_dir)
    return (
        root / f"rng_state_rank{rank}.json",
        root / f"rng_state_rank{rank}.safetensors",
    )


def save_rng_state(
    checkpoint_dir: Path,
    *,
    rank: int,
    world_size: int,
    state: object | None = None,
) -> dict[str, object]:
    checked_rank, checked_world = _rank_values(rank, world_size)
    safe = validate_rng_state(capture_rng_state() if state is None else state)
    python_state, numpy_state, cpu, cuda = _validated_rng_state_components(safe)
    json_path, tensor_path = _rng_paths(checkpoint_dir, checked_rank)
    tensors: dict[str, torch.Tensor] = {
        "numpy_mt19937": torch.from_numpy(numpy_state[1].astype(np.int64)),
        "torch_cpu": cpu,
        **{f"torch_cuda_{index}": value for index, value in enumerate(cuda)},
    }
    metadata = {
        "schema_version": RNG_STATE_SCHEMA_VERSION,
        "state_format": "python_numpy_torch_safetensors_v1",
        "rank": checked_rank,
        "world_size": checked_world,
        "tensor_file": tensor_path.name,
        "python_version": python_state[0],
        "python_internal_state": list(python_state[1]),
        "python_gauss_hex": (
            None
            if python_state[2] is None
            else _float_hex(python_state[2], "python gaussian cache")
        ),
        "numpy_bit_generator": numpy_state[0],
        "numpy_position": numpy_state[2],
        "numpy_has_gauss": numpy_state[3],
        "numpy_cached_gaussian_hex": _float_hex(numpy_state[4], "NumPy gaussian cache"),
        "cuda_device_count": len(cuda),
        "tensor_keys": sorted(tensors),
    }
    write_safetensors_atomic(tensor_path, tensors)
    write_json_atomic(json_path, metadata)
    return metadata


def load_rng_state_payloads(
    payload: object,
    tensors: Mapping[str, torch.Tensor],
    *,
    rank: int,
    expected_world_size: int,
    tensor_file_name: str,
) -> dict[str, object]:
    checked_rank, checked_world = _rank_values(rank, expected_world_size)
    expected_fields = {
        "schema_version",
        "state_format",
        "rank",
        "world_size",
        "tensor_file",
        "python_version",
        "python_internal_state",
        "python_gauss_hex",
        "numpy_bit_generator",
        "numpy_position",
        "numpy_has_gauss",
        "numpy_cached_gaussian_hex",
        "cuda_device_count",
        "tensor_keys",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("RNG metadata has an invalid field set")
    saved_schema = _integer(payload["schema_version"], "RNG schema_version", minimum=1)
    saved_rank = _integer(payload["rank"], "saved RNG rank")
    saved_world = _integer(payload["world_size"], "saved RNG world_size", minimum=1)
    if (
        saved_schema != RNG_STATE_SCHEMA_VERSION
        or payload["state_format"] != "python_numpy_torch_safetensors_v1"
        or saved_rank != checked_rank
        or saved_world != checked_world
        or payload["tensor_file"] != tensor_file_name
    ):
        raise ValueError("RNG metadata identity is inconsistent")
    cuda_count = _integer(payload["cuda_device_count"], "cuda_device_count")
    if cuda_count != torch.cuda.device_count():
        raise ValueError("RNG CUDA device count differs from current runtime")
    expected_keys = [
        "numpy_mt19937",
        "torch_cpu",
        *[f"torch_cuda_{index}" for index in range(cuda_count)],
    ]
    if payload["tensor_keys"] != sorted(expected_keys) or set(tensors) != set(
        expected_keys
    ):
        raise ValueError("RNG tensor inventory is inconsistent")
    internal = payload["python_internal_state"]
    if not isinstance(internal, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in internal
    ):
        raise ValueError("Python RNG internal state is invalid")
    gauss = payload["python_gauss_hex"]
    python_gauss = (
        None if gauss is None else _parse_float_hex(gauss, "python gaussian cache")
    )
    numpy_keys = tensors["numpy_mt19937"]
    if numpy_keys.dtype != torch.int64 or numpy_keys.ndim != 1:
        raise ValueError("NumPy RNG tensor is invalid")
    if bool(torch.any(numpy_keys < 0)) or bool(torch.any(numpy_keys > 0xFFFFFFFF)):
        raise ValueError("NumPy RNG tensor contains values outside uint32")
    numpy_has_gauss = _integer(payload["numpy_has_gauss"], "numpy_has_gauss")
    if numpy_has_gauss not in (0, 1):
        raise ValueError("numpy_has_gauss must be 0 or 1")
    numpy_state = (
        payload["numpy_bit_generator"],
        numpy_keys.numpy().astype(np.uint32),
        _integer(payload["numpy_position"], "numpy_position"),
        numpy_has_gauss,
        _parse_float_hex(payload["numpy_cached_gaussian_hex"], "NumPy gaussian cache"),
    )
    state = {
        "python": (
            _integer(payload["python_version"], "python_version"),
            tuple(internal),
            python_gauss,
        ),
        "numpy": numpy_state,
        "torch": tensors["torch_cpu"].clone(),
        "cuda": [tensors[f"torch_cuda_{index}"].clone() for index in range(cuda_count)],
    }
    return validate_rng_state(state, expected_cuda_device_count=cuda_count)


def load_rng_state(
    checkpoint_dir: Path,
    *,
    rank: int,
    expected_world_size: int,
) -> dict[str, object]:
    checked_rank, _ = _rank_values(rank, expected_world_size)
    json_path, tensor_path = _rng_paths(checkpoint_dir, checked_rank)
    payload = _strict_json(json_path)
    _regular_file(tensor_path)
    tensors = load_file(tensor_path, device="cpu")
    return load_rng_state_payloads(
        payload,
        tensors,
        rank=rank,
        expected_world_size=expected_world_size,
        tensor_file_name=tensor_path.name,
    )


def restore_rng_state(state: object) -> None:
    python_state, numpy_state, cpu, cuda = _validated_rng_state_components(state)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(list(cuda))
