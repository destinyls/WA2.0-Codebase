# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from n0_twam.data.latent_inventory import (
    TRACK31_VIDEO_KEYS,
    canonical_bytes,
    finalize_latent_inventory,
    inventory_path,
    validate_existing_track31_payload,
    validate_latent_inventory,
    validate_latent_inventory_pair,
    validate_track31_training_dataset_isolation,
)
from tests.unit.latent_inventory_fixtures import (
    CONVERSION_SHA256,
    ENCODER_SOURCE_IDENTITY,
    MANIFEST_SHA256,
)
from tests.unit.latent_inventory_fixtures import finalize_pair as _finalize_pair
from tests.unit.latent_inventory_fixtures import latent_payload as _payload
from tests.unit.latent_inventory_fixtures import write_dataset as _write_dataset
from tests.unit.latent_inventory_fixtures import (
    write_segment_artifacts as _write_segment_artifacts,
)


def test_complete_pair_binds_every_segment_and_artifact(tmp_path: Path) -> None:
    root = tmp_path / "train"
    _write_dataset(root, segments=((0, 5), (5, 10)))
    _write_segment_artifacts(root, 0, 5)
    _write_segment_artifacts(root, 5, 10)
    _finalize_pair(root)

    report = validate_latent_inventory_pair(
        root,
        expected_split="train",
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_conversion_report_sha256=CONVERSION_SHA256,
        expected_encoder_source_identity=ENCODER_SOURCE_IDENTITY,
    )

    assert report["segment_count"] == 2
    assert report["video_artifact_count"] == 4
    assert report["tactile_artifact_count"] == 8


def test_training_dataset_isolation_rejects_frozen40_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "train759").mkdir()
    (tmp_path / "frozen40").mkdir()

    with pytest.raises(ValueError, match="frozen40"):
        validate_track31_training_dataset_isolation(tmp_path)


def test_training_dataset_isolation_rejects_broken_frozen40_symlink(
    tmp_path: Path,
) -> None:
    (tmp_path / "train759").mkdir()
    (tmp_path / "frozen40").symlink_to(tmp_path / "not-mounted")

    with pytest.raises(ValueError, match="frozen40"):
        validate_track31_training_dataset_isolation(tmp_path)


def test_training_dataset_isolation_accepts_train759_only(tmp_path: Path) -> None:
    (tmp_path / "train759").mkdir()

    assert (
        validate_track31_training_dataset_isolation(tmp_path)
        == (tmp_path / "train759").resolve()
    )


def test_physical_train759_inventory_records_physical_split(tmp_path: Path) -> None:
    root = tmp_path / "train759"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5)

    inventory = finalize_latent_inventory(
        root,
        kind="video",
        split="train759",
        manifest_sha256=MANIFEST_SHA256,
        conversion_report_sha256=CONVERSION_SHA256,
        encoder_source_identity=ENCODER_SOURCE_IDENTITY,
    )

    assert inventory["split"] == "train759"
    validated = validate_latent_inventory(
        root,
        kind="video",
        expected_split="train759",
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_conversion_report_sha256=CONVERSION_SHA256,
        expected_encoder_source_identity=ENCODER_SOURCE_IDENTITY,
    )
    assert validated["split"] == "train759"
    assert validated["schema_version"] == 3
    assert validated["track31_encoding_contract"] is not None


def test_formal_inventory_rejects_missing_encoding_contract(tmp_path: Path) -> None:
    root = tmp_path / "train759"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5)
    target = (
        root
        / "latents"
        / "chunk-000"
        / TRACK31_VIDEO_KEYS[0]
        / "episode_000000_0_5.pth"
    )
    payload = torch.load(target, map_location="cpu", weights_only=True)
    payload.pop("track31_encoding_contract")
    torch.save(payload, target)

    with pytest.raises(ValueError, match="encoding contract mismatch"):
        finalize_latent_inventory(
            root,
            kind="video",
            split="train759",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )


def test_formal_inventory_rejects_missing_payload_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "train759"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5)
    target = (
        root
        / "latents"
        / "chunk-000"
        / TRACK31_VIDEO_KEYS[0]
        / "episode_000000_0_5.pth"
    )
    payload = torch.load(target, map_location="cpu", weights_only=True)
    payload.pop("track31_provenance")
    torch.save(payload, target)

    with pytest.raises(ValueError, match="provenance"):
        finalize_latent_inventory(
            root,
            kind="video",
            split="train759",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )


def test_matching_short_video_and_tactile_pair_cannot_finalize(
    tmp_path: Path,
) -> None:
    root = tmp_path / "train759"
    _write_dataset(root, segments=((0, 9),))
    _write_segment_artifacts(root, 0, 9)
    for path in tuple((root / "latents").rglob("*.pth")) + tuple(
        (root / "latents_tactile").rglob("*.pth")
    ):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["frame_ids"] = list(range(5))
        payload["latent_num_frames"] = 2
        payload["latent"] = torch.zeros(
            (2 * int(payload["latent_height"]) * int(payload["latent_width"]), 48),
            dtype=torch.bfloat16,
        )
        if "video_num_frames" in payload:
            payload["video_num_frames"] = 5
        torch.save(payload, path)

    with pytest.raises(ValueError, match="complete formal segment"):
        for kind in ("video", "tactile"):
            finalize_latent_inventory(
                root,
                kind=kind,
                split="train759",
                manifest_sha256=MANIFEST_SHA256,
                conversion_report_sha256=CONVERSION_SHA256,
                encoder_source_identity=ENCODER_SOURCE_IDENTITY,
            )


def test_self_consistent_schema2_formal_ready_inventory_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "train759"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5)
    inventory = finalize_latent_inventory(
        root,
        kind="video",
        split="train759",
        manifest_sha256=MANIFEST_SHA256,
        conversion_report_sha256=CONVERSION_SHA256,
        encoder_source_identity=ENCODER_SOURCE_IDENTITY,
    )
    inventory["schema_version"] = 2
    core = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    inventory["inventory_sha256"] = hashlib.sha256(canonical_bytes(core)).hexdigest()
    inventory_path(root, "video").write_text(
        json.dumps(inventory),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema 3|re-encode"):
        validate_latent_inventory(
            root,
            kind="video",
            expected_split="train759",
            expected_manifest_sha256=MANIFEST_SHA256,
            expected_conversion_report_sha256=CONVERSION_SHA256,
            expected_encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest",
        "conversion",
        "encoder",
        "contract",
        "source_video",
        "prompt",
        "stream_key",
        "device",
        "code_schema",
    ),
)
def test_formal_payload_provenance_tampering_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "train759"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5)
    target = (
        root
        / "latents"
        / "chunk-000"
        / TRACK31_VIDEO_KEYS[0]
        / "episode_000000_0_5.pth"
    )
    payload = torch.load(target, map_location="cpu", weights_only=True)
    provenance = payload["track31_provenance"]
    if mutation == "manifest":
        provenance["manifest_sha256"] = "f" * 64
    elif mutation == "conversion":
        provenance["conversion_report_sha256"] = "f" * 64
    elif mutation == "encoder":
        provenance["encoder_source_identity_sha256"] = "f" * 64
    elif mutation == "contract":
        provenance["encoding_contract_sha256"] = "f" * 64
    elif mutation == "source_video":
        provenance["source_video"]["sha256"] = "f" * 64
    elif mutation == "prompt":
        provenance["prompt"] = "lift_bottle"
    elif mutation == "stream_key":
        provenance["stream_key"] = TRACK31_VIDEO_KEYS[1]
    elif mutation == "device":
        provenance["execution_device"] = "mps"
    else:
        provenance["code_schema_version"] = 2
    torch.save(payload, target)

    with pytest.raises(ValueError, match="provenance"):
        finalize_latent_inventory(
            root,
            kind="video",
            split="train759",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )


def test_source_video_byte_change_invalidates_formal_payload(tmp_path: Path) -> None:
    root = tmp_path / "train759"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5)
    source_video = (
        root / "videos" / "chunk-000" / TRACK31_VIDEO_KEYS[0] / "episode_000000.mp4"
    )
    source_video.write_bytes(source_video.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="provenance"):
        finalize_latent_inventory(
            root,
            kind="video",
            split="train759",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )


def test_existing_formal_payload_without_provenance_requires_reencode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "train759"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5)
    target = (
        root
        / "latents"
        / "chunk-000"
        / TRACK31_VIDEO_KEYS[0]
        / "episode_000000_0_5.pth"
    )
    payload = torch.load(target, map_location="cpu", weights_only=True)
    expected_provenance = payload.pop("track31_provenance")
    torch.save(payload, target)

    with pytest.raises(ValueError, match="provenance"):
        validate_existing_track31_payload(
            root,
            target,
            kind="video",
            expected_encoder_source_identity_sha256=str(
                ENCODER_SOURCE_IDENTITY["identity_sha256"]
            ),
            expected_provenance=expected_provenance,
        )


def test_empty_latent_directory_cannot_finalize(tmp_path: Path) -> None:
    root = tmp_path / "train"
    _write_dataset(root)
    (root / "latents").mkdir()

    with pytest.raises(FileNotFoundError, match="missing regular latent artifact"):
        finalize_latent_inventory(
            root,
            kind="video",
            split="train",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )
    assert not inventory_path(root, "video").exists()


def test_missing_consumed_action_config_cannot_finalize(tmp_path: Path) -> None:
    root = tmp_path / "train"
    _write_dataset(root)
    episodes_path = root / "meta" / "episodes.jsonl"
    record = json.loads(episodes_path.read_text(encoding="utf-8"))
    record.pop("action_config")
    episodes_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing the action_config consumed"):
        finalize_latent_inventory(
            root,
            kind="video",
            split="train",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )


def test_missing_single_segment_clears_ready_marker(tmp_path: Path) -> None:
    root = tmp_path / "train"
    _write_dataset(root, segments=((0, 5), (5, 10)))
    _write_segment_artifacts(root, 0, 5, tactile=False)
    inventory_path(root, "video").write_text('{"status":"ready"}', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="episode_000000_5_10"):
        finalize_latent_inventory(
            root,
            kind="video",
            split="train",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )
    assert not inventory_path(root, "video").exists()


def test_payload_hash_tampering_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "train"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5)
    finalize_latent_inventory(
        root,
        kind="video",
        split="train",
        manifest_sha256=MANIFEST_SHA256,
        conversion_report_sha256=CONVERSION_SHA256,
        encoder_source_identity=ENCODER_SOURCE_IDENTITY,
    )
    target = (
        root
        / "latents"
        / "chunk-000"
        / TRACK31_VIDEO_KEYS[0]
        / "episode_000000_0_5.pth"
    )
    target.write_bytes(target.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="does not match current"):
        validate_latent_inventory(
            root,
            kind="video",
            expected_split="train",
            expected_manifest_sha256=MANIFEST_SHA256,
            expected_conversion_report_sha256=CONVERSION_SHA256,
            expected_encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )


def test_non_unit_track31_frame_stride_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "train"
    _write_dataset(root, segments=((0, 6),))
    for key in TRACK31_VIDEO_KEYS:
        path = root / "latents" / "chunk-000" / key / "episode_000000_0_6.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _payload(0, 5)
        payload["end_frame"] = 6
        payload["frame_ids"] = [0, 1, 2, 4, 5]
        torch.save(payload, path)

    with pytest.raises(ValueError, match="unit stride"):
        finalize_latent_inventory(
            root,
            kind="video",
            split="train",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )


@pytest.mark.parametrize("mutation", ("missing_text", "wrong_channels", "wrong_size"))
def test_video_consumer_payload_contract_is_required(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "train"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5, tactile=False)
    target = (
        root
        / "latents"
        / "chunk-000"
        / TRACK31_VIDEO_KEYS[0]
        / "episode_000000_0_5.pth"
    )
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if mutation == "missing_text":
        payload.pop("text_emb")
    elif mutation == "wrong_channels":
        payload["latent"] = torch.zeros((2 * 16 * 16, 47), dtype=torch.bfloat16)
    else:
        payload["latent_height"] = 8
        payload["latent"] = torch.zeros((2 * 8 * 16, 48), dtype=torch.bfloat16)
    torch.save(payload, target)

    with pytest.raises(ValueError, match="invalid latent|video latent payload"):
        finalize_latent_inventory(
            root,
            kind="video",
            split="train",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )


@pytest.mark.parametrize(
    ("split", "manifest_sha256", "error_pattern"),
    (
        ("validation", MANIFEST_SHA256, "source/split contract mismatch"),
        ("train", "3" * 64, "source/split contract mismatch"),
    ),
)
def test_split_or_source_contract_mismatch_fails_closed(
    tmp_path: Path,
    split: str,
    manifest_sha256: str,
    error_pattern: str,
) -> None:
    root = tmp_path / "train"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5)
    finalize_latent_inventory(
        root,
        kind="video",
        split="train",
        manifest_sha256=MANIFEST_SHA256,
        conversion_report_sha256=CONVERSION_SHA256,
        encoder_source_identity=ENCODER_SOURCE_IDENTITY,
    )

    with pytest.raises(ValueError, match=error_pattern):
        validate_latent_inventory(
            root,
            kind="video",
            expected_split=split,
            expected_manifest_sha256=manifest_sha256,
            expected_conversion_report_sha256=CONVERSION_SHA256,
            expected_encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )


def test_encoder_source_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "train"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5)
    finalize_latent_inventory(
        root,
        kind="video",
        split="train",
        manifest_sha256=MANIFEST_SHA256,
        conversion_report_sha256=CONVERSION_SHA256,
        encoder_source_identity=ENCODER_SOURCE_IDENTITY,
    )
    different_identity = dict(ENCODER_SOURCE_IDENTITY)
    different_identity["identity_sha256"] = "4" * 64

    with pytest.raises(ValueError, match="source/split contract mismatch"):
        validate_latent_inventory(
            root,
            kind="video",
            expected_split="train",
            expected_manifest_sha256=MANIFEST_SHA256,
            expected_conversion_report_sha256=CONVERSION_SHA256,
            expected_encoder_source_identity=different_identity,
        )


def test_skipped_artifact_from_other_encoder_source_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "train"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5, tactile=False)
    target = (
        root
        / "latents"
        / "chunk-000"
        / TRACK31_VIDEO_KEYS[0]
        / "episode_000000_0_5.pth"
    )
    payload = torch.load(target, map_location="cpu", weights_only=True)
    payload["encoder_source_identity_sha256"] = "4" * 64
    torch.save(payload, target)

    with pytest.raises(ValueError, match="encoder source identity mismatch"):
        finalize_latent_inventory(
            root,
            kind="video",
            split="train",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )


def test_unexpected_stale_segment_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "train"
    _write_dataset(root)
    _write_segment_artifacts(root, 0, 5)
    stale = root / "latents" / "chunk-000" / TRACK31_VIDEO_KEYS[0] / "stale.pth"
    torch.save(_payload(0, 5), stale)

    with pytest.raises(ValueError, match="unexpected=.*stale.pth"):
        finalize_latent_inventory(
            root,
            kind="video",
            split="train",
            manifest_sha256=MANIFEST_SHA256,
            conversion_report_sha256=CONVERSION_SHA256,
            encoder_source_identity=ENCODER_SOURCE_IDENTITY,
        )
