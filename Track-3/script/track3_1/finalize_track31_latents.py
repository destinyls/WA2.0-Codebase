#!/usr/bin/env python3
"""Finalize formal Track 3.1 latent inventories after all shards succeed."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Literal, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.data.latent_inventory import (
    build_encoder_source_identity,
    finalize_latent_inventory,
    inventory_path,
    validate_latent_inventory_pair,
)
from n0_twam.integrations.univtac.artifact_contracts import (
    verify_track31_physical_bundle,
)

FORMAL_PHYSICAL_SPLITS = ("train759", "frozen40")
INVENTORY_KINDS = ("video", "tactile", "both")
InventoryKind = Literal["video", "tactile"]


class FinalizerTerminated(BaseException):
    """Turn SIGTERM into fail-closed Python control flow."""


def _raise_on_termination(signum: int, _frame: FrameType | None) -> None:
    signal_name = signal.Signals(signum).name
    raise FinalizerTerminated(f"finalizer received {signal_name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--split", choices=FORMAL_PHYSICAL_SPLITS, required=True)
    parser.add_argument("--kind", choices=INVENTORY_KINDS, default="both")
    parser.add_argument(
        "--validate-pair",
        action="store_true",
        help="After finalizing both kinds, rebuild and compare their temporal grids.",
    )
    return parser.parse_args(argv)


def validate_options(*, kind: str, validate_pair: bool) -> None:
    if kind not in INVENTORY_KINDS:
        raise ValueError(f"unsupported inventory kind: {kind!r}")
    if validate_pair and kind != "both":
        raise ValueError("--validate-pair requires --kind both")


def _invalidate_requested(
    dataset_root: Path,
    kinds: tuple[InventoryKind, ...],
) -> tuple[str, ...]:
    """Unlink safe marker types for every kind and aggregate unsafe failures."""

    errors: list[str] = []
    for inventory_kind in kinds:
        path = inventory_path(dataset_root, inventory_kind)
        try:
            # unlink() removes the link itself and never follows its target.
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                raise ValueError(f"ready marker is not safely unlinkable: {path}")
        except FileNotFoundError:
            # A concurrent stale-marker deletion remains safely marker-free.
            pass
        except Exception as exc:
            errors.append(f"{inventory_kind}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _invalidate_after_failure(
    dataset_root: Path,
    kinds: tuple[InventoryKind, ...],
) -> tuple[str, ...]:
    """Make marker cleanup non-reentrant under a repeated SIGTERM."""

    try:
        previous_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except ValueError:
        return _invalidate_requested(dataset_root, kinds)
    try:
        return _invalidate_requested(dataset_root, kinds)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def finalize_requested_inventories(
    *,
    dataset_root: Path,
    model_path: Path,
    artifact_root: Path,
    split: str,
    kind: str,
    validate_pair: bool,
) -> dict[str, object]:
    """Verify immutable sources, finalize requested kinds, and fail closed."""

    validate_options(kind=kind, validate_pair=validate_pair)
    requested_kinds: tuple[InventoryKind, ...] = (
        ("video", "tactile") if kind == "both" else (cast(InventoryKind, kind),)
    )
    resolved_dataset_root = dataset_root.resolve(strict=True)
    # A requested ready marker is a completion claim. Remove it before any
    # source/model preverification so every subsequent failure remains closed.
    try:
        initial_cleanup_errors = _invalidate_requested(
            resolved_dataset_root,
            requested_kinds,
        )
        if initial_cleanup_errors:
            raise ValueError(
                "unable to clear all requested latent ready markers: "
                + "; ".join(initial_cleanup_errors)
            )
        if split not in FORMAL_PHYSICAL_SPLITS:
            raise ValueError("formal finalization requires train759 or frozen40")
        resolved_model_path = model_path.resolve(strict=True)
        resolved_artifact_root = artifact_root.resolve(strict=True)
        verified = verify_track31_physical_bundle(
            manifest_path=resolved_artifact_root / "universe_manifest_v4.json",
            conversion_report_path=resolved_artifact_root / "conversion_report.json",
            dataset_root=resolved_dataset_root,
            physical_split=split,
        )
        encoder_source_identity = build_encoder_source_identity(resolved_model_path)
        conversion_sha256 = verified.conversion_report_sha256
        if conversion_sha256 is None:
            raise ValueError("verified Track 3.1 bundle has no conversion digest")

        digests: dict[str, object] = {}
        for inventory_kind in requested_kinds:
            inventory = finalize_latent_inventory(
                resolved_dataset_root,
                kind=inventory_kind,
                split=split,
                manifest_sha256=verified.manifest_sha256,
                conversion_report_sha256=conversion_sha256,
                encoder_source_identity=encoder_source_identity,
            )
            digests[inventory_kind] = inventory["inventory_sha256"]

        pair_report = None
        if validate_pair:
            pair_report = validate_latent_inventory_pair(
                resolved_dataset_root,
                expected_split=split,
                expected_manifest_sha256=verified.manifest_sha256,
                expected_conversion_report_sha256=conversion_sha256,
                expected_encoder_source_identity=encoder_source_identity,
            )
    except BaseException as exc:
        cleanup_errors = _invalidate_after_failure(
            resolved_dataset_root,
            requested_kinds,
        )
        for cleanup_error in cleanup_errors:
            exc.add_note(f"latent marker cleanup failure: {cleanup_error}")
        raise

    return {
        "status": "ready",
        "split": split,
        "dataset_root": str(resolved_dataset_root),
        "kind": kind,
        "inventory_sha256": digests,
        "encoder_source_identity_sha256": encoder_source_identity["identity_sha256"],
        "pair_validation": pair_report,
    }


def main() -> None:
    args = parse_args()
    previous_handler = signal.signal(signal.SIGTERM, _raise_on_termination)
    try:
        report = finalize_requested_inventories(
            dataset_root=args.dataset_root,
            model_path=args.model_path,
            artifact_root=args.artifact_root,
            split=args.split,
            kind=args.kind,
            validate_pair=args.validate_pair,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
