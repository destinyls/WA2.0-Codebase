#!/usr/bin/env python3
# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Audit UniVTAC, freeze qpos8 artifacts, and optionally write LeRobot repos."""

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.integrations.univtac.convert_lerobot import (  # noqa: E402
    write_lerobot_dataset,
)
from n0_twam.integrations.univtac.dataset_view import (  # noqa: E402
    STAGE_A_DEV_VIEW_ID,
    STAGE_A_FINAL_VIEW_ID,
    DatasetView,
    build_standard_dataset_views,
)
from n0_twam.integrations.univtac.manifest import (  # noqa: E402
    UniVTACDatasetManifest,
    build_dataset_manifest,
    derive_split_paths,
)
from n0_twam.integrations.univtac.normalizer import (  # noqa: E402
    Qpos8Normalizer,
    compute_qpos8_normalizer,
)
from n0_twam.integrations.univtac.schema import (  # noqa: E402
    CANONICAL_TASK_PROMPT_MAP,
    DEFAULT_SOURCE_FPS,
    OUTPUT_COLOR_SPACE,
    SOURCE_IMAGE_ENCODING_CONTRACT,
    TASK_PROMPT_MAP_SHA256,
    TRACK31_TASKS,
    UNIVTAC_ALL_TASKS,
    canonical_task_scope,
)
from script.track3_1.materialize_transaction import (  # noqa: E402
    _canonical_sha256,
    _write_json_atomic,
    exclusive_artifact_lock,
)

LOGGER = logging.getLogger("n0_twam.prepare_univtac")
DEFAULT_TRAIN_COUNTS = {
    task: 94 if task == "grasp_classify" else 95 for task in UNIVTAC_ALL_TASKS
}
DEFAULT_VALIDATION_COUNTS = {task: 5 for task in UNIVTAC_ALL_TASKS}
DEFAULT_QUARANTINE_COUNTS = {
    task: 1 if task == "grasp_classify" else 0 for task in UNIVTAC_ALL_TASKS
}
STANDARD_PROFILE_NORMALIZER_VIEWS = {
    "qpos8_dev719_v1": STAGE_A_DEV_VIEW_ID,
    "qpos8_final759_v1": STAGE_A_FINAL_VIEW_ID,
}


def _legacy_expected_counts(
    train_per_task: int,
    validation_per_task: int,
) -> dict[str, dict[str, int]]:
    if train_per_task <= 0 or validation_per_task <= 0:
        raise ValueError("expected per-task counts must be positive")
    return {
        "train": {task: train_per_task for task in TRACK31_TASKS},
        "validation": {task: validation_per_task for task in TRACK31_TASKS},
        "quarantine": {task: 0 for task in TRACK31_TASKS},
    }


def _parse_task_count_map(value: str) -> dict[str, int]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("task counts must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("task counts must be a JSON object")
    counts: dict[str, int] = {}
    for task, count in payload.items():
        if (
            not isinstance(task, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
        ):
            raise argparse.ArgumentTypeError("task counts must map strings to integers")
        counts[task] = count
    return counts


def _explicit_count_map(
    counts: Mapping[str, int],
    *,
    tasks: Sequence[str],
    label: str,
) -> dict[str, int]:
    if set(counts) != set(tasks):
        raise ValueError(f"{label} counts must explicitly name every selected task")
    result = {task: counts[task] for task in tasks}
    if any(isinstance(count, bool) or count < 0 for count in result.values()):
        raise ValueError(f"{label} counts must be non-negative integers")
    return result


def _prepare_contract(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], dict[str, dict[str, int]], bool]:
    """Resolve an explicit standard/custom scope or the legacy two-task default."""

    if not hasattr(args, "expected_train_counts"):
        return (
            TRACK31_TASKS,
            _legacy_expected_counts(
                args.expected_train_per_task,
                args.expected_validation_per_task,
            ),
            False,
        )
    emit_standard_views = bool(getattr(args, "emit_standard_views", False))
    legacy_two_task = bool(getattr(args, "legacy_two_task", False))
    raw_tasks = getattr(args, "tasks", None)
    if legacy_two_task and raw_tasks is not None:
        raise ValueError("legacy two-task mode cannot be combined with --tasks")
    if legacy_two_task and emit_standard_views:
        raise ValueError("legacy two-task mode cannot emit standard views")
    tasks = (
        UNIVTAC_ALL_TASKS
        if raw_tasks is None and emit_standard_views
        else TRACK31_TASKS if raw_tasks is None else canonical_task_scope(raw_tasks)
    )
    if emit_standard_views and tasks != UNIVTAC_ALL_TASKS:
        raise ValueError("standard views require the complete eight-task scope")

    raw_count_maps = (
        getattr(args, "expected_train_counts", None),
        getattr(args, "expected_validation_counts", None),
        getattr(args, "expected_quarantine_counts", None),
    )
    if any(counts is not None for counts in raw_count_maps):
        if any(counts is None for counts in raw_count_maps):
            raise ValueError("explicit task counts must be supplied for every split")
        train_counts, validation_counts, quarantine_counts = raw_count_maps
        assert train_counts is not None
        assert validation_counts is not None
        assert quarantine_counts is not None
        expected = {
            "train": _explicit_count_map(train_counts, tasks=tasks, label="train"),
            "validation": _explicit_count_map(
                validation_counts, tasks=tasks, label="validation"
            ),
            "quarantine": _explicit_count_map(
                quarantine_counts, tasks=tasks, label="quarantine"
            ),
        }
    elif emit_standard_views:
        expected = {
            "train": dict(DEFAULT_TRAIN_COUNTS),
            "validation": dict(DEFAULT_VALIDATION_COUNTS),
            "quarantine": dict(DEFAULT_QUARANTINE_COUNTS),
        }
    elif tasks == TRACK31_TASKS:
        expected = _legacy_expected_counts(
            int(getattr(args, "expected_train_per_task", 95)),
            int(getattr(args, "expected_validation_per_task", 5)),
        )
    else:
        raise ValueError("custom --tasks requires explicit counts for every split")
    return tasks, expected, emit_standard_views


def _conversion_split_specs(
    emit_standard_views: bool,
) -> tuple[tuple[str, str], ...]:
    """Map source split names to their on-disk LeRobot repository names."""

    if emit_standard_views:
        return (("train", "train759"), ("validation", "frozen40"))
    return (("train", "train"), ("validation", "validation"))


def _write_standard_profile_normalizers(
    *,
    artifact_dir: Path,
    manifest: UniVTACDatasetManifest,
    views: Mapping[str, DatasetView],
    legacy_train759_normalizer: Qpos8Normalizer,
) -> dict[str, str]:
    """Write Stage A normalizers; Stage B deliberately inherits these files."""

    dev_view = views[STANDARD_PROFILE_NORMALIZER_VIEWS["qpos8_dev719_v1"]]
    final_view = views[STANDARD_PROFILE_NORMALIZER_VIEWS["qpos8_final759_v1"]]
    profile_normalizers = {
        "qpos8_dev719_v1": compute_qpos8_normalizer(
            manifest,
            view=dev_view,
        ),
        # The final view is exactly the physical train759 split, so its numeric
        # statistics equal the retained legacy train-only normalizer. Bind the
        # same values to the immutable view identity without rereading 759 HDF5s.
        "qpos8_final759_v1": replace(
            legacy_train759_normalizer,
            source_view_id=final_view.view_id,
            source_view_sha256=final_view.view_sha256,
        ),
    }
    hashes: dict[str, str] = {}
    for normalizer_id, profile_normalizer in profile_normalizers.items():
        _write_json_atomic(
            artifact_dir / "normalizers" / f"{normalizer_id}.json",
            profile_normalizer.to_json_dict(),
        )
        hashes[normalizer_id] = profile_normalizer.normalizer_sha256
    return hashes


def _materialize_lerobot_splits(
    *,
    manifest: UniVTACDatasetManifest,
    materialize_root: Path,
    repo_id: str,
    fps: int,
    image_writer_threads: int,
    emit_standard_views: bool,
) -> dict[str, object]:
    """Materialize only train/evaluation physical repos with stable names."""

    conversions: dict[str, object] = {}
    for source_split, physical_split in _conversion_split_specs(emit_standard_views):
        records = tuple(
            entry for entry in manifest.entries if entry.split == source_split
        )
        conversions[physical_split] = write_lerobot_dataset(
            records,
            output_root=str(materialize_root / physical_split),
            repo_id=f"{repo_id}_{physical_split}",
            fps=fps,
            image_writer_threads=image_writer_threads,
        )
    return conversions


def prepare_artifacts(args: argparse.Namespace) -> dict[str, object]:
    artifact_dir = Path(args.artifact_dir).expanduser()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_artifact_lock(artifact_dir.resolve(strict=True)):
        return _prepare_artifacts_locked(args)


def _prepare_artifacts_locked(args: argparse.Namespace) -> dict[str, object]:
    tasks, expected_counts, emit_standard_views = _prepare_contract(args)
    split_paths = derive_split_paths(
        data_root=args.data_root,
        validation_manifest_path=args.validation_manifest,
        quarantine_manifest_path=args.quarantine_manifest,
        tasks=tasks,
    )
    manifest = build_dataset_manifest(
        data_root=args.data_root,
        train_paths=split_paths["train"],
        validation_paths=split_paths["validation"],
        quarantine_paths=split_paths["quarantine"],
        expected_task_counts=expected_counts,
        tasks=tasks,
    )
    normalizer = compute_qpos8_normalizer(manifest)
    manifest_payload = manifest.to_json_dict()
    _write_json_atomic(args.artifact_dir / "dataset_manifest.json", manifest_payload)
    _write_json_atomic(
        args.artifact_dir / "qpos8_normalizer.json",
        normalizer.to_json_dict(),
    )

    view_hashes: dict[str, str] = {}
    profile_normalizer_hashes: dict[str, str] = {}
    if emit_standard_views:
        views = build_standard_dataset_views(manifest)
        _write_json_atomic(
            args.artifact_dir / "universe_manifest_v4.json",
            manifest_payload,
        )
        _write_json_atomic(
            args.artifact_dir / "prompt_map_v1.json",
            {
                "schema_version": 1,
                "tasks": list(UNIVTAC_ALL_TASKS),
                "prompts": dict(CANONICAL_TASK_PROMPT_MAP),
                "prompt_map_sha256": TASK_PROMPT_MAP_SHA256,
            },
        )
        for view_id, view in views.items():
            _write_json_atomic(
                args.artifact_dir / "views" / f"{view_id}.json",
                view.to_json_dict(),
            )
            view_hashes[view_id] = view.view_sha256
        _write_json_atomic(
            args.artifact_dir / "quarantine_v1.json",
            views["quarantine1_v1"].to_json_dict(),
        )
        profile_normalizer_hashes = _write_standard_profile_normalizers(
            artifact_dir=args.artifact_dir,
            manifest=manifest,
            views=views,
            legacy_train759_normalizer=normalizer,
        )

    conversions: dict[str, object] = {}
    if args.materialize_root is not None:
        if args.materialize_root.exists():
            raise FileExistsError(
                "materialize root must not exist so conversion cannot mix with "
                f"stale files: {args.materialize_root}"
            )
        args.materialize_root.mkdir(parents=True)
        conversions = _materialize_lerobot_splits(
            manifest=manifest,
            materialize_root=args.materialize_root,
            repo_id=args.repo_id,
            fps=args.fps,
            image_writer_threads=args.image_writer_threads,
            emit_standard_views=emit_standard_views,
        )
        conversion_payload = {
            "schema_version": 2,
            "source_manifest_sha256": manifest.manifest_sha256,
            "source_image_encoding_contract": SOURCE_IMAGE_ENCODING_CONTRACT,
            "output_color_space": OUTPUT_COLOR_SPACE,
            "conversions": conversions,
        }
        conversion_payload["conversion_report_sha256"] = _canonical_sha256(
            conversion_payload
        )
        _write_json_atomic(
            args.artifact_dir / "conversion_report.json",
            conversion_payload,
        )

    return {
        "manifest_sha256": manifest.manifest_sha256,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "split_counts": manifest_payload["split_counts"],
        "task_counts": manifest.task_counts,
        "quarantine": [
            {
                "relative_path": entry.relative_path,
                "sha256": entry.sha256,
                "episode_id": entry.episode_id,
            }
            for entry in manifest.entries
            if entry.split == "quarantine"
        ],
        "view_sha256": view_hashes,
        "profile_normalizer_sha256": profile_normalizer_hashes,
        "conversions": conversions,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--quarantine-manifest", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--materialize-root", type=Path)
    parser.add_argument("--repo-id", default="univtac_track31")
    parser.add_argument("--fps", type=int, default=int(DEFAULT_SOURCE_FPS))
    task_scope = parser.add_mutually_exclusive_group()
    task_scope.add_argument(
        "--tasks",
        nargs="+",
        choices=UNIVTAC_ALL_TASKS,
        help="explicit non-standard UniVTAC task scope",
    )
    task_scope.add_argument(
        "--legacy-two-task",
        action="store_true",
        help="explicitly select the original Insert HDMI/Lift Bottle scope",
    )
    parser.add_argument(
        "--expected-train-counts",
        type=_parse_task_count_map,
    )
    parser.add_argument(
        "--expected-validation-counts",
        type=_parse_task_count_map,
    )
    parser.add_argument(
        "--expected-quarantine-counts",
        type=_parse_task_count_map,
    )
    parser.add_argument("--expected-train-per-task", type=int, default=95)
    parser.add_argument("--expected-validation-per-task", type=int, default=5)
    view_mode = parser.add_mutually_exclusive_group()
    view_mode.add_argument(
        "--emit-standard-views",
        action="store_true",
        help="select the eight-task 759/40/1 protocol and emit its views",
    )
    view_mode.add_argument(
        "--skip-standard-views",
        action="store_false",
        dest="emit_standard_views",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(emit_standard_views=False)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = prepare_artifacts(args)
    LOGGER.info("UniVTAC artifacts ready: %s", json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
