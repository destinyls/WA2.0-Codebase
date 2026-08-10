# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Deterministic conversion-report construction and candidate publication."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from n0_twam.integrations.univtac.artifact_contracts import (
    verify_track31_physical_bundle,
)
from n0_twam.integrations.univtac.schema import (
    OUTPUT_COLOR_SPACE,
    SOURCE_IMAGE_ENCODING_CONTRACT,
)
from script.track3_1.materialize_transaction import _canonical_sha256
from script.track3_1.materialize_writeahead import (
    prepare_json_file,
    publish_prepared_file_noreplace,
)

PHYSICAL_SPLITS = ("train759", "frozen40")


def verify_physical_bundles(
    *,
    manifest_path: Path,
    conversion_report_path: Path,
    materialize_root: Path,
) -> dict[str, int]:
    """Verify the exact formal physical-split cardinalities."""

    counts: dict[str, int] = {}
    for physical_split in PHYSICAL_SPLITS:
        verified = verify_track31_physical_bundle(
            manifest_path=manifest_path,
            conversion_report_path=conversion_report_path,
            dataset_root=materialize_root / physical_split,
            physical_split=physical_split,
            allow_transaction_state=True,
        )
        counts[physical_split] = verified.episode_count
    expected = {"train759": 759, "frozen40": 40}
    if counts != expected:
        raise ValueError(f"verified physical split counts changed: {counts}")
    return counts


def prepare_report_candidate(
    *,
    prepared_path: Path,
    candidate_path: Path,
    payload: object,
) -> str:
    """Prepare candidate bytes inside owned staging before external publish."""

    if prepared_path.parent == candidate_path.parent:
        raise ValueError("report candidate must be prepared inside staging")
    return cast(
        str,
        prepare_json_file(
            prepared_path,
            payload,
            preserve_dir=candidate_path.parent,
            preserve_foreign=True,
        ),
    )


def publish_prepared_candidate(
    *,
    prepared_path: Path,
    candidate_path: Path,
    expected_sha256: str,
) -> None:
    """Publish an identity-bound report candidate without replacement."""

    publish_prepared_file_noreplace(
        prepared_path,
        candidate_path,
        expected_sha256=expected_sha256,
    )


def conversion_report(
    *,
    manifest_sha256: str,
    conversions: Mapping[str, object],
    materialize_root: Path,
) -> dict[str, object]:
    """Rebind split roots and attach a deterministic logical report digest."""

    if set(conversions) != set(PHYSICAL_SPLITS):
        raise ValueError("materialization must produce train759 and frozen40")
    rebound: dict[str, object] = {}
    for physical_split in PHYSICAL_SPLITS:
        conversion = conversions[physical_split]
        if not isinstance(conversion, dict):
            raise ValueError(f"invalid conversion payload: {physical_split}")
        rebound[physical_split] = {
            **conversion,
            "output_root": str(materialize_root / physical_split),
        }
    payload: dict[str, object] = {
        "schema_version": 2,
        "source_manifest_sha256": manifest_sha256,
        "source_image_encoding_contract": SOURCE_IMAGE_ENCODING_CONTRACT,
        "output_color_space": OUTPUT_COLOR_SPACE,
        "conversions": rebound,
    }
    payload["conversion_report_sha256"] = _canonical_sha256(payload)
    return payload
