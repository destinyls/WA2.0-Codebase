# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from n0_twam.evaluation.target10_view_contract import load_canonical_target10_view
from n0_twam.integrations.univtac.dataset_view import (
    ALLOWED_OVERLAP_MATRIX,
    DEFAULT_UNIFIED_EVALUATION_EPISODE_COUNT,
    DEFAULT_UNIFIED_EVALUATION_EPISODE_IDS,
    DEFAULT_UNIFIED_EVALUATION_TASKS,
    DEFAULT_UNIFIED_EVALUATION_VIEW_ID,
    DIAGNOSTIC_EVALUATION_VIEW_IDS,
    DatasetView,
    build_standard_dataset_views,
    content_addressed_sample_seed,
    load_dataset_view,
    select_content_addressed_crop_start,
    verify_standard_view_set,
)
from n0_twam.integrations.univtac.manifest import UniVTACDatasetManifest
from n0_twam.integrations.univtac.schema import (
    TRACK31_TARGET_TASKS,
    UNIVTAC_ALL_TASKS,
    UniVTACEpisodeRecord,
)
from n0_twam.integrations.univtac.temporal_contract import (
    LIFT_CAN_62_TEMPORAL_CONTRACT,
    PINNED_MAXIMAL_PREFIX_POLICY,
)

VALIDATION_EPISODE_IDS = frozenset((0, 1, 2, 3, 5))
STANDARD_VIEW_IDS = frozenset(
    (
        "stage_a_dev719_v1",
        "internal_dev40_v1",
        "stage_a_final759_v1",
        "stage_b_dev180_v1",
        "internal_target_dev10_v1",
        "stage_b_final190_v1",
        "frozen_target10_v1",
        "frozen_other30_v1",
        "quarantine1_v1",
    )
)


def _source_record(
    root: Path,
    *,
    task: str,
    episode_id: int,
    split: str,
) -> UniVTACEpisodeRecord:
    relative_path = f"{task}/clean/{episode_id}.hdf5"
    payload = relative_path.encode("utf-8")
    is_pinned = relative_path == LIFT_CAN_62_TEMPORAL_CONTRACT.relative_path
    length = (
        LIFT_CAN_62_TEMPORAL_CONTRACT.raw_length
        if is_pinned
        else 43 if split == "quarantine" else 45
    )
    return UniVTACEpisodeRecord(
        relative_path=relative_path,
        absolute_path=root / relative_path,
        task=task,
        split=split,
        length=length,
        length_source="embodiment/joint.shape[0]",
        joint_shape=(length, 9),
        image_shapes=(("observation.images.top", (192, 256, 3)),),
        size_bytes=len(payload),
        sha256=(
            LIFT_CAN_62_TEMPORAL_CONTRACT.sha256
            if is_pinned
            else hashlib.sha256(payload).hexdigest()
        ),
        usable_start=0,
        usable_end=(LIFT_CAN_62_TEMPORAL_CONTRACT.usable_end if is_pinned else length),
        step_discontinuities_after_rows=(
            (LIFT_CAN_62_TEMPORAL_CONTRACT.discontinuity_after_row,)
            if is_pinned
            else ()
        ),
        temporal_policy=(
            PINNED_MAXIMAL_PREFIX_POLICY if is_pinned else "strict_monotonic_v1"
        ),
    )


def _universe_manifest(tmp_path: Path) -> UniVTACDatasetManifest:
    entries: list[UniVTACEpisodeRecord] = []
    for split in ("train", "validation", "quarantine"):
        for task in UNIVTAC_ALL_TASKS:
            for episode_id in range(100):
                if task == "grasp_classify" and episode_id == 90:
                    expected_split = "quarantine"
                elif episode_id in VALIDATION_EPISODE_IDS:
                    expected_split = "validation"
                else:
                    expected_split = "train"
                if split == expected_split:
                    entries.append(
                        _source_record(
                            tmp_path,
                            task=task,
                            episode_id=episode_id,
                            split=split,
                        )
                    )
    return UniVTACDatasetManifest(
        data_root=str(tmp_path),
        entries=tuple(entries),
        tasks=UNIVTAC_ALL_TASKS,
    )


def _entry_paths(view: DatasetView) -> set[str]:
    return {entry.relative_path for entry in view.entries}


def test_standard_views_have_exact_counts_and_relations(tmp_path: Path) -> None:
    manifest = _universe_manifest(tmp_path)

    views = build_standard_dataset_views(
        manifest,
        selection_seed="unit-test-v1",
        verify_sources=False,
    )

    assert set(views) == STANDARD_VIEW_IDS
    assert len(views["stage_a_dev719_v1"].entries) == 719
    assert len(views["internal_dev40_v1"].entries) == 40
    assert len(views["stage_a_final759_v1"].entries) == 759
    assert len(views["stage_b_dev180_v1"].entries) == 180
    assert len(views["internal_target_dev10_v1"].entries) == 10
    assert len(views["stage_b_final190_v1"].entries) == 190
    assert len(views["frozen_target10_v1"].entries) == 10
    assert len(views["frozen_other30_v1"].entries) == 30
    assert len(views["quarantine1_v1"].entries) == 1
    assert views["stage_a_dev719_v1"].per_task_counts == {
        "grasp_classify": 89,
        "insert_HDMI": 90,
        "insert_hole": 90,
        "insert_tube": 90,
        "lift_bottle": 90,
        "lift_can": 90,
        "pull_out_key": 90,
        "put_bottle_in_shelf": 90,
    }
    assert views["frozen_target10_v1"].per_task_counts == {
        task: 5 for task in TRACK31_TARGET_TASKS
    }
    assert views["quarantine1_v1"].entries[0].relative_path == (
        "grasp_classify/clean/90.hdf5"
    )
    assert verify_standard_view_set(views) is None
    assert len(ALLOWED_OVERLAP_MATRIX) == 36


def test_target10_is_the_default_unified_evaluation_view(tmp_path: Path) -> None:
    views = build_standard_dataset_views(
        _universe_manifest(tmp_path),
        verify_sources=False,
    )
    default_view = views[DEFAULT_UNIFIED_EVALUATION_VIEW_ID]

    assert DEFAULT_UNIFIED_EVALUATION_VIEW_ID == "frozen_target10_v1"
    assert DEFAULT_UNIFIED_EVALUATION_TASKS == TRACK31_TARGET_TASKS
    assert DEFAULT_UNIFIED_EVALUATION_EPISODE_IDS == (0, 1, 2, 3, 5)
    assert len(default_view.entries) == DEFAULT_UNIFIED_EVALUATION_EPISODE_COUNT
    assert DIAGNOSTIC_EVALUATION_VIEW_IDS == ("frozen_other30_v1",)
    assert {
        task: tuple(
            entry.source_episode_id
            for entry in default_view.entries
            if entry.task == task
        )
        for task in default_view.tasks
    } == {
        task: DEFAULT_UNIFIED_EVALUATION_EPISODE_IDS
        for task in DEFAULT_UNIFIED_EVALUATION_TASKS
    }


def test_default_unified_evaluation_rejects_episode_id_drift(tmp_path: Path) -> None:
    manifest = _universe_manifest(tmp_path)
    drifted_entries = tuple(
        replace(
            entry,
            split=(
                "train"
                if entry.task == "insert_HDMI" and entry.episode_id == 5
                else (
                    "validation"
                    if entry.task == "insert_HDMI" and entry.episode_id == 6
                    else entry.split
                )
            ),
        )
        for entry in manifest.entries
    )

    with pytest.raises(ValueError, match="canonical Target-10"):
        build_standard_dataset_views(
            replace(manifest, entries=drifted_entries),
            verify_sources=False,
        )


def test_target10_contract_rejects_a_self_signed_noncanonical_roster(
    tmp_path: Path,
) -> None:
    manifest = _universe_manifest(tmp_path)
    views = build_standard_dataset_views(manifest, verify_sources=False)
    canonical = views["frozen_target10_v1"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_json_dict()), encoding="utf-8")
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(json.dumps(canonical.to_json_dict()), encoding="utf-8")

    loaded = load_canonical_target10_view(
        view_path=canonical_path,
        manifest_path=manifest_path,
    )
    assert loaded == canonical

    other_entry = views["frozen_other30_v1"].entries[0]
    wrong_view = DatasetView(
        view_id=canonical.view_id,
        role=canonical.role,
        physical_split=canonical.physical_split,
        source_manifest_sha256=canonical.source_manifest_sha256,
        tasks=(*canonical.tasks, other_entry.task),
        entries=(*canonical.entries[:-1], other_entry),
        selection_method=canonical.selection_method,
    )
    wrong_path = tmp_path / "wrong.json"
    wrong_path.write_text(json.dumps(wrong_view.to_json_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="not the canonical frozen_target10_v1"):
        load_canonical_target10_view(
            view_path=wrong_path,
            manifest_path=manifest_path,
        )


def test_internal_selection_is_content_addressed_and_stable(tmp_path: Path) -> None:
    manifest = _universe_manifest(tmp_path)

    first = build_standard_dataset_views(
        manifest,
        selection_seed="fixed-v1",
        verify_sources=False,
    )
    second = build_standard_dataset_views(
        replace(manifest, entries=tuple(reversed(manifest.entries))),
        selection_seed="fixed-v1",
        verify_sources=False,
    )

    first_internal = first["internal_dev40_v1"]
    second_internal = second["internal_dev40_v1"]
    assert _entry_paths(first_internal) == _entry_paths(second_internal)
    first_seed = content_addressed_sample_seed(
        first_internal.entries[0], seed=2026, epoch=3
    )
    reordered_entry = next(
        entry
        for entry in second_internal.entries
        if entry.relative_path == first_internal.entries[0].relative_path
    )
    assert first_seed == content_addressed_sample_seed(
        reordered_entry, seed=2026, epoch=3
    )
    crop = select_content_addressed_crop_start(
        first_internal.entries[0],
        seed=2026,
        epoch=3,
        num_latent_frames=9,
        max_latent_frames=5,
    )
    assert crop == select_content_addressed_crop_start(
        reordered_entry,
        seed=2026,
        epoch=3,
        num_latent_frames=9,
        max_latent_frames=5,
    )
    assert 0 <= crop <= 4


def test_dataset_view_round_trip_is_immutable_and_content_addressed(
    tmp_path: Path,
) -> None:
    view = build_standard_dataset_views(
        _universe_manifest(tmp_path),
        verify_sources=False,
    )["stage_b_final190_v1"]
    path = tmp_path / "stage_b_final190_v1.json"
    path.write_text(json.dumps(view.to_json_dict()), encoding="utf-8")

    loaded = load_dataset_view(path)

    assert loaded == view
    assert loaded.view_sha256 == view.view_sha256
    with pytest.raises((AttributeError, TypeError)):
        loaded.entries[0].task = "changed"  # type: ignore[misc]


def test_overlap_matrix_rejects_path_sha_realpath_and_episode_id_drift(
    tmp_path: Path,
) -> None:
    views = build_standard_dataset_views(
        _universe_manifest(tmp_path),
        verify_sources=False,
    )
    target = views["frozen_target10_v1"]
    other = views["frozen_other30_v1"]
    mutated = dict(views)
    mutated["frozen_other30_v1"] = replace(
        other,
        entries=(target.entries[0],) + other.entries[1:],
        tasks=UNIVTAC_ALL_TASKS,
    )

    with pytest.raises(ValueError, match="overlap.*relative_path"):
        verify_standard_view_set(mutated)


def test_overlap_matrix_rejects_recombined_composite_episode_identity(
    tmp_path: Path,
) -> None:
    """Marginal identity sets must not hide a child entry absent from its parent."""

    views = build_standard_dataset_views(
        _universe_manifest(tmp_path),
        verify_sources=False,
    )
    child = views["stage_b_dev180_v1"]
    first, second = child.entries[:2]
    recombined = (
        replace(first, source_sha256=second.source_sha256),
        replace(second, source_sha256=first.source_sha256),
        *child.entries[2:],
    )
    mutated = dict(views)
    mutated[child.view_id] = replace(child, entries=recombined)

    with pytest.raises(ValueError, match="composite episode identity"):
        verify_standard_view_set(mutated)


def test_standard_verifier_rejects_per_task_drift_between_dev_partitions(
    tmp_path: Path,
) -> None:
    views = build_standard_dataset_views(
        _universe_manifest(tmp_path),
        verify_sources=False,
    )
    internal = views["internal_dev40_v1"]
    dev = views["stage_a_dev719_v1"]
    internal_grasp = next(
        entry for entry in internal.entries if entry.task == "grasp_classify"
    )
    dev_hole = next(entry for entry in dev.entries if entry.task == "insert_hole")
    mutated = dict(views)
    mutated[internal.view_id] = replace(
        internal,
        entries=tuple(
            dev_hole if entry == internal_grasp else entry for entry in internal.entries
        ),
    )
    mutated[dev.view_id] = replace(
        dev,
        entries=tuple(
            internal_grasp if entry == dev_hole else entry for entry in dev.entries
        ),
    )

    with pytest.raises(ValueError, match="per-task counts"):
        verify_standard_view_set(mutated)


def test_standard_verifier_recomputes_content_addressed_internal_selection(
    tmp_path: Path,
) -> None:
    views = build_standard_dataset_views(
        _universe_manifest(tmp_path),
        verify_sources=False,
    )
    internal = views["internal_dev40_v1"]
    dev = views["stage_a_dev719_v1"]
    internal_hole = next(
        entry for entry in internal.entries if entry.task == "insert_hole"
    )
    dev_hole = next(entry for entry in dev.entries if entry.task == "insert_hole")
    mutated = dict(views)
    mutated[internal.view_id] = replace(
        internal,
        entries=tuple(
            dev_hole if entry == internal_hole else entry for entry in internal.entries
        ),
    )
    mutated[dev.view_id] = replace(
        dev,
        entries=tuple(
            internal_hole if entry == dev_hole else entry for entry in dev.entries
        ),
    )

    with pytest.raises(ValueError, match="deterministic per-task selection"):
        verify_standard_view_set(mutated)


def test_view_constructor_rejects_duplicate_identity(tmp_path: Path) -> None:
    view = build_standard_dataset_views(
        _universe_manifest(tmp_path),
        verify_sources=False,
    )["internal_target_dev10_v1"]
    original = view.entries[0]
    duplicate_sha = replace(
        original,
        relative_path=f"{original.task}/clean/999.hdf5",
        realpath=str(tmp_path / original.task / "clean/999.hdf5"),
        source_episode_id=999,
        lerobot_episode_id=999,
    )

    with pytest.raises(ValueError, match="duplicate.*source_sha256"):
        replace(
            view,
            entries=(original, duplicate_sha),
        )
