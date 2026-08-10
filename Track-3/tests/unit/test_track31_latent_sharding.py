"""Formal sharding and single-finalizer contracts for Track 3.1 latents."""

from __future__ import annotations

import shutil
import signal
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from n0_twam.data.latent_inventory import inventory_path
from script import encode_lerobot_n0_latents as video_encoder
from script import encode_tactile_latent as tactile_encoder
from script.track3_1 import finalize_track31_latents as finalizer
from tests.unit.latent_inventory_fixtures import (
    CONVERSION_SHA256,
    ENCODER_SOURCE_IDENTITY,
    MANIFEST_SHA256,
    write_dataset,
    write_segment_artifacts,
)

ENCODER_MODULES = (video_encoder, tactile_encoder)


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_modulo_shards_are_disjoint_and_cover_train759(module: ModuleType) -> None:
    episodes = pd.DataFrame({"episode_index": range(759)})
    shard_sets = [
        set(
            module.select_episode_shard(
                episodes,
                num_shards=48,
                shard_index=shard_index,
            )["episode_index"]
        )
        for shard_index in range(48)
    ]

    assert set.union(*shard_sets) == set(range(759))
    assert sum(len(shard) for shard in shard_sets) == 759
    assert {len(shard) for shard in shard_sets} == {15, 16}


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_frozen40_allows_empty_high_rank_shards(module: ModuleType) -> None:
    episodes = pd.DataFrame({"episode_index": range(40)})

    selected = module.select_episode_shard(
        episodes,
        num_shards=48,
        shard_index=47,
    )

    assert selected.empty


@pytest.mark.parametrize("module", ENCODER_MODULES)
@pytest.mark.parametrize(
    ("kwargs", "error_pattern"),
    (
        (
            {"num_shards": 48, "shard_index": None, "defer_inventory": True},
            "provided together",
        ),
        (
            {"num_shards": 48, "shard_index": 48, "defer_inventory": True},
            "shard index",
        ),
        (
            {"num_shards": 48, "shard_index": 0, "defer_inventory": False},
            "--defer-inventory",
        ),
        (
            {"num_shards": None, "shard_index": None, "defer_inventory": True},
            "requires --num-shards",
        ),
    ),
)
def test_invalid_shard_combinations_fail_closed(
    module: ModuleType,
    kwargs: dict[str, object],
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        module.validate_shard_contract(
            split="train759",
            artifact_root=Path("/artifacts"),
            episodes=None,
            **kwargs,
        )


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_sharding_is_formal_only_and_rejects_manual_episode_mix(
    module: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="formal physical"):
        module.validate_shard_contract(
            split="train",
            artifact_root=Path("/artifacts"),
            episodes=None,
            num_shards=4,
            shard_index=0,
            defer_inventory=True,
        )

    with pytest.raises(ValueError, match="--episodes"):
        module.validate_shard_contract(
            split="train759",
            artifact_root=Path("/artifacts"),
            episodes=[0],
            num_shards=4,
            shard_index=0,
            defer_inventory=True,
        )


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_atomic_payload_temp_names_are_unique_for_same_pid_and_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    target = tmp_path / "episode_000000_0_5.pth"
    temporary_paths: list[Path] = []

    def fake_save(payload: object, path: Path) -> None:
        temporary_path = Path(path)
        temporary_paths.append(temporary_path)
        temporary_path.write_text(repr(payload), encoding="utf-8")

    monkeypatch.setattr(module.torch, "save", fake_save)
    monkeypatch.setattr(module.os, "getpid", lambda: 7)

    module.atomic_torch_save({"attempt": 1}, target, shard_index=3)
    module.atomic_torch_save({"attempt": 2}, target, shard_index=3)

    assert len(set(temporary_paths)) == 2
    assert all(".shard-3." in path.name for path in temporary_paths)
    assert target.read_text(encoding="utf-8") == "{'attempt': 2}"
    assert not any(path.exists() for path in temporary_paths)


def _patch_finalizer_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        finalizer,
        "verify_track31_physical_bundle",
        lambda **_: SimpleNamespace(
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "build_encoder_source_identity",
        lambda _: ENCODER_SOURCE_IDENTITY,
    )


def test_single_finalizer_writes_both_ready_markers_and_pair_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "train759"
    write_dataset(root)
    write_segment_artifacts(root, 0, 5)
    model_root = tmp_path / "model"
    artifact_root = tmp_path / "artifacts"
    model_root.mkdir()
    artifact_root.mkdir()
    _patch_finalizer_sources(monkeypatch)

    report = finalizer.finalize_requested_inventories(
        dataset_root=root,
        model_path=model_root,
        artifact_root=artifact_root,
        split="train759",
        kind="both",
        validate_pair=True,
    )

    assert report["status"] == "ready"
    assert report["pair_validation"]["segment_count"] == 1
    assert inventory_path(root, "video").is_file()
    assert inventory_path(root, "tactile").is_file()


def test_failed_both_finalization_clears_all_requested_ready_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "frozen40"
    write_dataset(root)
    write_segment_artifacts(root, 0, 5, tactile=False)
    model_root = tmp_path / "model"
    artifact_root = tmp_path / "artifacts"
    model_root.mkdir()
    artifact_root.mkdir()
    for kind in ("video", "tactile"):
        inventory_path(root, kind).write_text("stale", encoding="utf-8")
    _patch_finalizer_sources(monkeypatch)

    with pytest.raises(FileNotFoundError, match="tactile"):
        finalizer.finalize_requested_inventories(
            dataset_root=root,
            model_path=model_root,
            artifact_root=artifact_root,
            split="frozen40",
            kind="both",
            validate_pair=True,
        )

    assert not inventory_path(root, "video").exists()
    assert not inventory_path(root, "tactile").exists()


def test_keyboard_interrupt_during_second_kind_clears_first_ready_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "train759"
    write_dataset(root)
    write_segment_artifacts(root, 0, 5)
    model_root = tmp_path / "model"
    artifact_root = tmp_path / "artifacts"
    model_root.mkdir()
    artifact_root.mkdir()
    _patch_finalizer_sources(monkeypatch)
    real_finalize = finalizer.finalize_latent_inventory

    def interrupt_tactile(*args: object, **kwargs: object) -> dict[str, object]:
        if kwargs.get("kind") == "tactile":
            raise KeyboardInterrupt("operator interrupt")
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(
        finalizer,
        "finalize_latent_inventory",
        interrupt_tactile,
    )

    with pytest.raises(KeyboardInterrupt, match="operator interrupt"):
        finalizer.finalize_requested_inventories(
            dataset_root=root,
            model_path=model_root,
            artifact_root=artifact_root,
            split="train759",
            kind="both",
            validate_pair=True,
        )

    assert not inventory_path(root, "video").exists()
    assert not inventory_path(root, "tactile").exists()


def test_sigterm_during_second_kind_clears_both_ready_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "train759"
    write_dataset(root)
    write_segment_artifacts(root, 0, 5)
    model_root = tmp_path / "model"
    artifact_root = tmp_path / "artifacts"
    model_root.mkdir()
    artifact_root.mkdir()
    _patch_finalizer_sources(monkeypatch)
    real_finalize = finalizer.finalize_latent_inventory
    handlers: dict[int, object] = {signal.SIGTERM: signal.SIG_DFL}

    def fake_signal(signum: int, handler: object) -> object:
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    def terminate_tactile(*args: object, **kwargs: object) -> dict[str, object]:
        if kwargs.get("kind") == "tactile":
            handler = handlers[signal.SIGTERM]
            assert callable(handler)
            handler(signal.SIGTERM, None)
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(finalizer.signal, "signal", fake_signal)
    monkeypatch.setattr(finalizer, "finalize_latent_inventory", terminate_tactile)
    monkeypatch.setattr(
        finalizer,
        "parse_args",
        lambda: SimpleNamespace(
            dataset_root=root,
            model_path=model_root,
            artifact_root=artifact_root,
            split="train759",
            kind="both",
            validate_pair=True,
        ),
    )

    with pytest.raises(finalizer.FinalizerTerminated, match="SIGTERM"):
        finalizer.main()

    assert handlers[signal.SIGTERM] is signal.SIG_DFL
    assert not inventory_path(root, "video").exists()
    assert not inventory_path(root, "tactile").exists()


def test_repeated_sigterm_is_ignored_during_failure_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "train759"
    write_dataset(root)
    write_segment_artifacts(root, 0, 5)
    model_root = tmp_path / "model"
    artifact_root = tmp_path / "artifacts"
    model_root.mkdir()
    artifact_root.mkdir()
    _patch_finalizer_sources(monkeypatch)
    handlers: dict[int, object] = {signal.SIGTERM: signal.SIG_DFL}
    real_invalidate = finalizer._invalidate_requested
    invalidate_calls = 0

    def fake_signal(signum: int, handler: object) -> object:
        previous = handlers.get(signum, signal.SIG_DFL)
        handlers[signum] = handler
        return previous

    def interrupting_cleanup(
        dataset_root: Path,
        kinds: tuple[finalizer.InventoryKind, ...],
    ) -> tuple[str, ...]:
        nonlocal invalidate_calls
        invalidate_calls += 1
        if invalidate_calls == 2:
            inventory_path(dataset_root, "video").unlink()
            assert handlers[signal.SIGTERM] is signal.SIG_IGN
            inventory_path(dataset_root, "tactile").unlink()
            return ()
        return real_invalidate(dataset_root, kinds)

    monkeypatch.setattr(finalizer.signal, "signal", fake_signal)
    monkeypatch.setattr(finalizer, "_invalidate_requested", interrupting_cleanup)
    monkeypatch.setattr(
        finalizer,
        "validate_latent_inventory_pair",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("pair failed")),
    )
    monkeypatch.setattr(
        finalizer,
        "parse_args",
        lambda: SimpleNamespace(
            dataset_root=root,
            model_path=model_root,
            artifact_root=artifact_root,
            split="train759",
            kind="both",
            validate_pair=True,
        ),
    )

    with pytest.raises(RuntimeError, match="pair failed"):
        finalizer.main()

    assert handlers[signal.SIGTERM] is signal.SIG_DFL
    assert not inventory_path(root, "video").exists()
    assert not inventory_path(root, "tactile").exists()


def test_pair_validation_requires_both_kind() -> None:
    with pytest.raises(ValueError, match="requires --kind both"):
        finalizer.validate_options(kind="video", validate_pair=True)


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_cuda_request_never_silently_falls_back_to_cpu(module: ModuleType) -> None:
    with pytest.raises(RuntimeError, match="accelerator|cuda"):
        module.resolve_execution_device(
            "cuda:0",
            accelerator_available=False,
        )

    assert module.resolve_execution_device("cpu", accelerator_available=False) == "cpu"


def test_finalizer_clears_requested_markers_before_physical_preverify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "train759"
    write_dataset(root)
    model_root = tmp_path / "model"
    artifact_root = tmp_path / "artifacts"
    model_root.mkdir()
    artifact_root.mkdir()
    for kind in ("video", "tactile"):
        inventory_path(root, kind).write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        finalizer,
        "verify_track31_physical_bundle",
        lambda **_: (_ for _ in ()).throw(ValueError("physical preverify failed")),
    )

    with pytest.raises(ValueError, match="physical preverify failed"):
        finalizer.finalize_requested_inventories(
            dataset_root=root,
            model_path=model_root,
            artifact_root=artifact_root,
            split="train759",
            kind="both",
            validate_pair=True,
        )

    assert not inventory_path(root, "video").exists()
    assert not inventory_path(root, "tactile").exists()


def test_finalizer_unlinks_marker_symlink_without_following_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "train759"
    write_dataset(root)
    model_root = tmp_path / "model"
    artifact_root = tmp_path / "artifacts"
    model_root.mkdir()
    artifact_root.mkdir()
    outside_marker = tmp_path / "outside-marker.json"
    outside_marker.write_text("keep", encoding="utf-8")
    inventory_path(root, "video").symlink_to(outside_marker)
    inventory_path(root, "tactile").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        finalizer,
        "verify_track31_physical_bundle",
        lambda **_: (_ for _ in ()).throw(ValueError("physical preverify failed")),
    )

    with pytest.raises(ValueError, match="physical preverify failed"):
        finalizer.finalize_requested_inventories(
            dataset_root=root,
            model_path=model_root,
            artifact_root=artifact_root,
            split="train759",
            kind="both",
            validate_pair=True,
        )

    assert not inventory_path(root, "video").is_symlink()
    assert not inventory_path(root, "tactile").exists()
    assert outside_marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("symlink_level", ("root", "ancestor"))
def test_finalizer_rejects_payload_tree_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_level: str,
) -> None:
    root = tmp_path / "train759"
    write_dataset(root)
    write_segment_artifacts(root, 0, 5)
    model_root = tmp_path / "model"
    artifact_root = tmp_path / "artifacts"
    model_root.mkdir()
    artifact_root.mkdir()
    _patch_finalizer_sources(monkeypatch)

    if symlink_level == "root":
        source = root / "latents"
        outside = tmp_path / "outside-latents"
    else:
        source = root / "latents" / "chunk-000"
        outside = tmp_path / "outside-chunk"
    shutil.move(str(source), outside)
    source.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|escaped"):
        finalizer.finalize_requested_inventories(
            dataset_root=root,
            model_path=model_root,
            artifact_root=artifact_root,
            split="train759",
            kind="video",
            validate_pair=False,
        )

    assert not inventory_path(root, "video").exists()
