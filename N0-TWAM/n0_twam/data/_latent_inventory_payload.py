"""Internal construction and validation of latent inventory payloads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import torch

from ._latent_payload_provenance import (
    LATENT_INVENTORY_SCHEMA_VERSION as LATENT_INVENTORY_SCHEMA_VERSION,
)
from ._latent_payload_provenance import (
    TRACK31_DATASET_SPLITS,
)
from ._latent_payload_provenance import TRACK31_FORMAL_SPLITS as TRACK31_FORMAL_SPLITS
from ._latent_payload_provenance import (
    TRACK31_FPS,
    TRACK31_LATENT_CHANNELS,
)
from ._latent_payload_provenance import (
    TRACK31_TACTILE_PAYLOAD_PROVENANCE_SCHEMA_VERSION,
    TRACK31_VIDEO_PAYLOAD_PROVENANCE_SCHEMA_VERSION,
)
from ._latent_payload_provenance import (
    TRACK31_TACTILE_ENCODING_CONTRACT as TRACK31_TACTILE_ENCODING_CONTRACT,
)
from ._latent_payload_provenance import TRACK31_TACTILE_KEYS as TRACK31_TACTILE_KEYS
from ._latent_payload_provenance import (
    TRACK31_TACTILE_LATENT_SIZE,
)
from ._latent_payload_provenance import TRACK31_TACTILE_MODES as TRACK31_TACTILE_MODES
from ._latent_payload_provenance import (
    TRACK31_TEXT_EMBEDDING_SHAPE,
)
from ._latent_payload_provenance import (
    TRACK31_VIDEO_ENCODING_CONTRACT as TRACK31_VIDEO_ENCODING_CONTRACT,
)
from ._latent_payload_provenance import TRACK31_VIDEO_KEYS as TRACK31_VIDEO_KEYS
from ._latent_payload_provenance import (
    TRACK31_VIDEO_LATENT_SIZE,
)
from ._latent_payload_provenance import InventoryKind as InventoryKind
from ._latent_payload_provenance import (
    _require_contained_regular_file,
    _require_sha256,
    _sha256_file,
)
from ._latent_payload_provenance import (
    build_track31_payload_provenance as build_track31_payload_provenance,
)
from ._latent_payload_provenance import (
    build_track31_source_video_identity as build_track31_source_video_identity,
)
from ._latent_payload_provenance import canonical_bytes as canonical_bytes
from ._latent_payload_provenance import (
    expected_track31_sampled_frame_ids,
)
from .encoder_source_identity import validate_encoder_source_identity

VIDEO_INVENTORY_FILENAME = "latent_video_inventory.json"
TACTILE_INVENTORY_FILENAME = "latent_tactile_inventory.json"


@dataclass(frozen=True)
class ExpectedLatentSegment:
    episode_index: int
    start_frame: int
    end_frame: int
    episode_task: str


@dataclass(frozen=True)
class LatentArtifact:
    relative_path: str
    size_bytes: int
    sha256: str
    latent_num_frames: int
    source_frame_count: int
    frame_ids_sha256: str
    provenance_sha256: str | None


def _require_integer(value: object, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def inventory_path(dataset_root: Path, kind: InventoryKind) -> Path:
    """Return the only completion-marker path accepted by preflight."""

    filename = (
        VIDEO_INVENTORY_FILENAME if kind == "video" else TACTILE_INVENTORY_FILENAME
    )
    return Path(dataset_root) / filename


def invalidate_latent_inventory(dataset_root: Path, kind: InventoryKind) -> None:
    """Remove a stale ready marker before an encoder may mutate its payloads."""

    path = inventory_path(dataset_root, kind)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"latent inventory marker is not a regular file: {path}")
    if path.exists():
        path.unlink()


def load_expected_segments(dataset_root: Path) -> tuple[ExpectedLatentSegment, ...]:
    """Load the exact episode/segment keys used by the latent dataset."""

    root = Path(dataset_root).resolve(strict=True)
    episodes_path = root / "meta" / "episodes.jsonl"
    try:
        lines = episodes_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"unable to read LeRobot episodes: {episodes_path}") from exc
    if not lines:
        raise ValueError(f"LeRobot episodes metadata is empty: {episodes_path}")

    segments: list[ExpectedLatentSegment] = []
    episode_ids: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid episodes.jsonl record at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"episode record {line_number} must be an object")
        episode_index = int(record.get("episode_index", -1))
        length = int(record.get("length", 0))
        if episode_index < 0 or episode_index in episode_ids or length <= 0:
            raise ValueError(f"invalid episode metadata at line {line_number}")
        episode_ids.add(episode_index)
        tasks = record.get("tasks")
        if (
            not isinstance(tasks, list)
            or len(tasks) != 1
            or not isinstance(tasks[0], str)
            or not tasks[0].strip()
        ):
            raise ValueError(f"episode {episode_index} has an invalid frozen task")
        episode_task = tasks[0]
        action_config = record.get("action_config")
        if action_config is None:
            raise ValueError(
                f"episode {episode_index} is missing the action_config consumed "
                "by LatentLeRobotDataset"
            )
        if not isinstance(action_config, list) or not action_config:
            raise ValueError(f"episode {episode_index} has no action segments")
        for raw_segment in action_config:
            if not isinstance(raw_segment, dict):
                raise ValueError(f"episode {episode_index} has an invalid segment")
            start_frame = int(raw_segment.get("start_frame", -1))
            end_frame = int(raw_segment.get("end_frame", -1))
            if not 0 <= start_frame < end_frame <= length:
                raise ValueError(
                    f"episode {episode_index} has out-of-range segment "
                    f"[{start_frame}, {end_frame})"
                )
            action_text = raw_segment.get("action_text")
            if action_text is not None and action_text != episode_task:
                raise ValueError(
                    f"episode {episode_index} segment task does not match metadata"
                )
            segments.append(
                ExpectedLatentSegment(
                    episode_index,
                    start_frame,
                    end_frame,
                    episode_task,
                )
            )
    ordered = tuple(
        sorted(
            segments,
            key=lambda item: (
                item.episode_index,
                item.start_frame,
                item.end_frame,
            ),
        )
    )
    segment_keys = {
        (item.episode_index, item.start_frame, item.end_frame) for item in ordered
    }
    if len(ordered) != len(segment_keys):
        raise ValueError("LeRobot episode metadata contains duplicate latent segments")
    return ordered


def _chunk_index(info: dict[str, Any], episode_index: int) -> int:
    total_chunks = int(info.get("total_chunks", 1))
    chunks_size = int(info.get("chunks_size", 1000))
    if total_chunks <= 0 or chunks_size <= 0:
        raise ValueError("invalid LeRobot chunk metadata")
    return 0 if total_chunks == 1 else episode_index // chunks_size


def _artifact_relative_paths(
    info: dict[str, Any],
    segment: ExpectedLatentSegment,
    kind: InventoryKind,
) -> tuple[str, ...]:
    chunk = _chunk_index(info, segment.episode_index)
    filename = (
        f"episode_{segment.episode_index:06d}_"
        f"{segment.start_frame}_{segment.end_frame}.pth"
    )
    if kind == "video":
        return tuple(
            (Path("latents") / f"chunk-{chunk:03d}" / key / filename).as_posix()
            for key in TRACK31_VIDEO_KEYS
        )
    return tuple(
        (
            Path("latents_tactile") / mode / f"chunk-{chunk:03d}" / key / filename
        ).as_posix()
        for mode in TRACK31_TACTILE_MODES
        for key in TRACK31_TACTILE_KEYS
    )


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:  # vendor torch builds may reject mmap with custom errors
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"latent artifact payload must be a dictionary: {path}")
    return payload


def _load_artifact(
    dataset_root: Path,
    path: Path,
    relative_path: str,
    expected_encoder_identity_sha256: str,
    expected_encoding_contract: dict[str, object] | None,
    expected_provenance: dict[str, object] | None,
    expected_segment: ExpectedLatentSegment,
) -> tuple[LatentArtifact, dict[str, Any]]:
    path = _require_contained_regular_file(
        dataset_root,
        path,
        label="latent artifact",
    )
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"latent artifact is empty: {path}")
    payload = _load_payload(path)
    if (
        payload.get("encoder_source_identity_sha256")
        != expected_encoder_identity_sha256
    ):
        raise ValueError(f"latent encoder source identity mismatch: {path}")
    if (
        expected_encoding_contract is not None
        and payload.get("track31_encoding_contract") != expected_encoding_contract
    ):
        raise ValueError(f"latent encoding contract mismatch: {path}")
    is_video = Path(relative_path).parts[0] == "latents"
    if expected_provenance is not None:
        actual_provenance = payload.get("track31_provenance")
        if not isinstance(actual_provenance, dict):
            raise ValueError(f"latent payload provenance mismatch: {path}")
        normalized_expected_provenance = dict(expected_provenance)
        if actual_provenance != normalized_expected_provenance:
            raise ValueError(f"latent payload provenance mismatch: {path}")
        expected_provenance = normalized_expected_provenance
    latent_num_frames = int(payload.get("latent_num_frames", 0))
    height = int(payload.get("latent_height", 0))
    width = int(payload.get("latent_width", 0))
    latent = payload.get("latent")
    frame_ids = payload.get("frame_ids")
    expected_size = (
        TRACK31_VIDEO_LATENT_SIZE if is_video else TRACK31_TACTILE_LATENT_SIZE
    )
    if (
        latent_num_frames <= 0
        or (height, width) != expected_size
        or not isinstance(latent, torch.Tensor)
        or latent.ndim != 2
        or int(latent.shape[0]) != latent_num_frames * height * width
        or int(latent.shape[1]) != TRACK31_LATENT_CHANNELS
        or latent.dtype != torch.bfloat16
        or not isinstance(frame_ids, (list, tuple))
        or not frame_ids
    ):
        raise ValueError(f"invalid latent tensor metadata: {path}")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool)
        for value in frame_ids
    ):
        raise ValueError(f"latent frame_ids must contain integers: {path}")
    normalized_ids = [int(value) for value in frame_ids]
    if any(
        right - left != 1 for left, right in zip(normalized_ids, normalized_ids[1:])
    ):
        raise ValueError(f"Track 3.1 latent frame_ids must have unit stride: {path}")
    expected_source_frames = (latent_num_frames - 1) * 4 + 1
    if len(normalized_ids) != expected_source_frames:
        raise ValueError(f"latent frame count violates the Wan 4n+1 contract: {path}")
    if expected_provenance is not None:
        complete_frame_ids = expected_track31_sampled_frame_ids(
            expected_segment.start_frame,
            expected_segment.end_frame,
        )
        if normalized_ids != complete_frame_ids:
            raise ValueError(
                "latent frame_ids do not cover the complete formal segment: " f"{path}"
            )
    video_num_frames = payload.get("video_num_frames")
    if video_num_frames is not None and int(video_num_frames) != len(normalized_ids):
        raise ValueError(f"video_num_frames disagrees with frame_ids: {path}")
    if (
        int(payload.get("fps", -1)) != TRACK31_FPS
        or int(payload.get("ori_fps", -1)) != TRACK31_FPS
    ):
        raise ValueError(f"Track 3.1 latent fps must be {TRACK31_FPS}: {path}")
    if is_video:
        text_embedding = payload.get("text_emb")
        if (
            not isinstance(text_embedding, torch.Tensor)
            or tuple(text_embedding.shape) != TRACK31_TEXT_EMBEDDING_SHAPE
            or text_embedding.dtype != torch.bfloat16
            or not isinstance(payload.get("text"), str)
            or not payload["text"].strip()
            or int(payload.get("video_height", -1)) != 256
            or int(payload.get("video_width", -1)) != 256
            or (
                expected_provenance is not None
                and payload.get("text") != expected_provenance.get("prompt")
            )
        ):
            raise ValueError(f"invalid Track 3.1 video latent payload: {path}")
    elif (
        int(payload.get("tactile_resize_h", -1)) != 128
        or int(payload.get("tactile_resize_w", -1)) != 128
    ):
        raise ValueError(f"invalid Track 3.1 tactile latent resize contract: {path}")
    artifact = LatentArtifact(
        relative_path=relative_path,
        size_bytes=size_bytes,
        sha256=_sha256_file(path),
        latent_num_frames=latent_num_frames,
        source_frame_count=len(normalized_ids),
        frame_ids_sha256=hashlib.sha256(canonical_bytes(normalized_ids)).hexdigest(),
        provenance_sha256=(
            None
            if expected_provenance is None
            else hashlib.sha256(canonical_bytes(expected_provenance)).hexdigest()
        ),
    )
    return artifact, payload


def _require_symlink_free_payload_tree(
    dataset_root: Path,
    payload_root: Path,
) -> None:
    if payload_root.is_symlink():
        raise ValueError(f"latent payload root cannot be a symlink: {payload_root}")
    if not payload_root.exists():
        return
    root = dataset_root.resolve(strict=True)
    try:
        payload_root.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError("latent payload root escaped the dataset root") from exc
    for candidate in payload_root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(f"latent payload tree contains a symlink: {candidate}")
        try:
            candidate.resolve(strict=True).relative_to(root)
        except ValueError as exc:
            raise ValueError("latent payload tree escaped the dataset root") from exc


def _artifact_stream_dimensions(
    relative_path: str,
    kind: InventoryKind,
) -> tuple[str, str | None]:
    parts = Path(relative_path).parts
    if kind == "video":
        if len(parts) != 4 or parts[0] != "latents":
            raise ValueError(f"invalid video latent path: {relative_path}")
        stream_key = parts[2]
        tactile_mode = None
    else:
        if len(parts) != 5 or parts[0] != "latents_tactile":
            raise ValueError(f"invalid tactile latent path: {relative_path}")
        tactile_mode = parts[1]
        stream_key = parts[3]
    return stream_key, tactile_mode


def _expected_source_video_relative_path(
    info: dict[str, Any],
    segment: ExpectedLatentSegment,
    stream_key: str,
) -> str:
    chunk = _chunk_index(info, segment.episode_index)
    return (
        Path("videos")
        / f"chunk-{chunk:03d}"
        / stream_key
        / f"episode_{segment.episode_index:06d}.mp4"
    ).as_posix()


def _expected_payload_provenance(
    *,
    root: Path,
    info: dict[str, Any],
    split: str,
    kind: InventoryKind,
    segment: ExpectedLatentSegment,
    relative_path: str,
    manifest_sha256: str,
    conversion_report_sha256: str,
    encoder_source_sha256: str,
    source_identity_cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    stream_key, tactile_mode = _artifact_stream_dimensions(relative_path, kind)
    source_relative_path = _expected_source_video_relative_path(
        info,
        segment,
        stream_key,
    )
    source_identity = source_identity_cache.get(source_relative_path)
    if source_identity is None:
        source_identity = build_track31_source_video_identity(
            root,
            root / source_relative_path,
        )
        source_identity_cache[source_relative_path] = source_identity
    return build_track31_payload_provenance(
        split=split,
        manifest_sha256=manifest_sha256,
        conversion_report_sha256=conversion_report_sha256,
        encoder_source_identity_sha256=encoder_source_sha256,
        kind=kind,
        episode_index=segment.episode_index,
        start_frame=segment.start_frame,
        end_frame=segment.end_frame,
        episode_task=segment.episode_task,
        stream_key=stream_key,
        tactile_mode=tactile_mode,
        prompt=segment.episode_task if kind == "video" else None,
        source_video_identity=source_identity,
        execution_device="cuda:0",
        text_encoder_device="cuda:0" if kind == "video" else None,
    )


def build_inventory_core(
    dataset_root: Path,
    *,
    kind: InventoryKind,
    split: str,
    manifest_sha256: str,
    conversion_report_sha256: str,
    encoder_source_identity: object,
) -> dict[str, object]:
    """Reconstruct the complete inventory directly from on-disk payloads."""

    root = Path(dataset_root).resolve(strict=True)
    if split not in TRACK31_DATASET_SPLITS or root.name != split:
        raise ValueError(f"dataset root/split mismatch: {root} vs {split!r}")
    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read LeRobot info: {info_path}") from exc
    features = info.get("features", {})
    required_features = set(TRACK31_VIDEO_KEYS + TRACK31_TACTILE_KEYS)
    if not isinstance(features, dict) or not required_features.issubset(features):
        raise ValueError("LeRobot feature contract is missing Track 3.1 streams")
    if float(info.get("fps", 0.0)) != float(TRACK31_FPS):
        raise ValueError(f"Track 3.1 LeRobot fps must be {TRACK31_FPS}")
    segments = load_expected_segments(root)
    episode_count = len({segment.episode_index for segment in segments})
    if not segments or int(info.get("total_episodes", 0)) != episode_count:
        raise ValueError("latent inventory episode set does not match info.json")

    normalized_encoder_source = validate_encoder_source_identity(
        encoder_source_identity
    )
    encoder_source_sha256 = str(normalized_encoder_source["identity_sha256"])
    expected_encoding_contract = (
        (
            TRACK31_VIDEO_ENCODING_CONTRACT
            if kind == "video"
            else TRACK31_TACTILE_ENCODING_CONTRACT
        )
        if split in TRACK31_FORMAL_SPLITS
        else None
    )
    payload_root = root / ("latents" if kind == "video" else "latents_tactile")
    _require_symlink_free_payload_tree(root, payload_root)
    segment_payloads: list[dict[str, object]] = []
    expected_paths: set[str] = set()
    source_identity_cache: dict[str, dict[str, object]] = {}
    for segment in segments:
        relative_paths = _artifact_relative_paths(info, segment, kind)
        loaded: list[tuple[LatentArtifact, dict[str, Any]]] = []
        for relative_path in relative_paths:
            expected_provenance = (
                _expected_payload_provenance(
                    root=root,
                    info=info,
                    split=split,
                    kind=kind,
                    segment=segment,
                    relative_path=relative_path,
                    manifest_sha256=manifest_sha256,
                    conversion_report_sha256=conversion_report_sha256,
                    encoder_source_sha256=encoder_source_sha256,
                    source_identity_cache=source_identity_cache,
                )
                if split in TRACK31_FORMAL_SPLITS
                else None
            )
            loaded.append(
                _load_artifact(
                    root,
                    root / relative_path,
                    relative_path,
                    encoder_source_sha256,
                    expected_encoding_contract,
                    expected_provenance,
                    segment,
                )
            )
        artifacts = tuple(item[0] for item in loaded)
        expected_paths.update(relative_paths)
        temporal_shapes = {
            (
                artifact.latent_num_frames,
                artifact.source_frame_count,
                artifact.frame_ids_sha256,
            )
            for artifact in artifacts
        }
        if len(temporal_shapes) != 1:
            raise ValueError(f"latent streams disagree for segment {asdict(segment)}")
        latent_frames, source_frames, frame_ids_sha256 = temporal_shapes.pop()
        for artifact, payload in loaded:
            frame_ids = [int(value) for value in payload["frame_ids"]]
            if (
                int(payload.get("start_frame", -1)) != segment.start_frame
                or int(payload.get("end_frame", -1)) != segment.end_frame
                or frame_ids[0] != segment.start_frame
                or frame_ids[-1] >= segment.end_frame
            ):
                raise ValueError(
                    f"latent payload segment mismatch: {artifact.relative_path}"
                )
            if kind == "tactile":
                mode = Path(artifact.relative_path).parts[1]
                if payload.get("tactile_residual_mode") != mode:
                    raise ValueError(
                        f"tactile mode metadata mismatch: {artifact.relative_path}"
                    )
                if mode == "local" and payload.get("tactile_local_mode") != "current":
                    raise ValueError(
                        "Track 3.1 requires current-frame local tactile: "
                        f"{artifact.relative_path}"
                    )
        segment_payloads.append(
            {
                **asdict(segment),
                "latent_num_frames": latent_frames,
                "source_frame_count": source_frames,
                "frame_ids_sha256": frame_ids_sha256,
                "artifacts": [asdict(artifact) for artifact in artifacts],
            }
        )

    actual_paths = (
        {
            path.relative_to(root).as_posix()
            for path in payload_root.rglob("*.pth")
            if path.is_file()
        }
        if payload_root.is_dir()
        else set()
    )
    if actual_paths != expected_paths:
        raise ValueError(
            "latent payload file set mismatch: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"unexpected={sorted(actual_paths - expected_paths)}"
        )
    return {
        "schema_version": LATENT_INVENTORY_SCHEMA_VERSION,
        "status": "ready",
        "kind": kind,
        "split": split,
        "dataset_root": str(root),
        "action_schema": "qpos8_next_step",
        "manifest_sha256": _require_sha256(manifest_sha256, "manifest_sha256"),
        "conversion_report_sha256": _require_sha256(
            conversion_report_sha256, "conversion_report_sha256"
        ),
        "encoder_source_identity": normalized_encoder_source,
        "track31_encoding_contract": expected_encoding_contract,
        "payload_provenance_schema_version": (
            (
                TRACK31_VIDEO_PAYLOAD_PROVENANCE_SCHEMA_VERSION
                if kind == "video"
                else TRACK31_TACTILE_PAYLOAD_PROVENANCE_SCHEMA_VERSION
            )
            if split in TRACK31_FORMAL_SPLITS
            else None
        ),
        "required_video_keys": list(TRACK31_VIDEO_KEYS),
        "required_tactile_keys": list(TRACK31_TACTILE_KEYS),
        "required_tactile_modes": list(TRACK31_TACTILE_MODES),
        "segments": segment_payloads,
    }


def validate_existing_track31_payload(
    dataset_root: Path,
    artifact_path: Path,
    *,
    kind: InventoryKind,
    expected_encoder_source_identity_sha256: str,
    expected_provenance: dict[str, object],
) -> None:
    """Fail closed before a formal encoder skips an existing payload."""

    root = Path(dataset_root).resolve(strict=True)
    path = Path(artifact_path)
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("existing latent payload escaped the dataset root") from exc
    if expected_provenance.get("kind") != kind:
        raise ValueError("existing latent provenance kind mismatch")
    segment = ExpectedLatentSegment(
        episode_index=_require_integer(
            expected_provenance.get("episode_index"),
            "existing latent provenance episode_index",
        ),
        start_frame=_require_integer(
            expected_provenance.get("start_frame"),
            "existing latent provenance start_frame",
        ),
        end_frame=_require_integer(
            expected_provenance.get("end_frame"),
            "existing latent provenance end_frame",
        ),
        episode_task=str(expected_provenance.get("episode_task", "")),
    )
    try:
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read LeRobot info for existing payload") from exc
    if relative_path not in _artifact_relative_paths(info, segment, kind):
        raise ValueError("existing latent path does not match its provenance")
    stream_key, tactile_mode = _artifact_stream_dimensions(relative_path, kind)
    if (
        expected_provenance.get("stream_key") != stream_key
        or expected_provenance.get("tactile_mode") != tactile_mode
    ):
        raise ValueError("existing latent stream does not match its provenance")
    expected_contract = (
        TRACK31_VIDEO_ENCODING_CONTRACT
        if kind == "video"
        else TRACK31_TACTILE_ENCODING_CONTRACT
    )
    _load_artifact(
        root,
        path,
        relative_path,
        _require_sha256(
            expected_encoder_source_identity_sha256,
            "expected_encoder_source_identity_sha256",
        ),
        expected_contract,
        expected_provenance,
        segment,
    )
