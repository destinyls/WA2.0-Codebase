# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""CLI boundary for formal UniVTAC materialization."""

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from n0_twam.integrations.univtac.schema import DEFAULT_SOURCE_FPS


@dataclass(frozen=True)
class MaterializeConfig:
    """Immutable inputs for one formal materialization transaction."""

    artifact_dir: Path
    target_root: Path
    repo_id: str
    fps: int
    image_writer_threads: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse materializer CLI arguments without starting the transaction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="univtac_track31")
    parser.add_argument("--fps", type=int, default=int(DEFAULT_SOURCE_FPS))
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def run_cli(
    argv: Sequence[str] | None,
    *,
    materialize: Callable[[MaterializeConfig], dict[str, object]],
    logger: logging.Logger,
) -> int:
    """Run the thin CLI adapter around the identity-bound materializer."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = materialize(
        MaterializeConfig(
            artifact_dir=args.artifact_dir,
            target_root=args.target_root,
            repo_id=args.repo_id,
            fps=args.fps,
            image_writer_threads=args.image_writer_threads,
        )
    )
    logger.info("Materialization report: %s", json.dumps(result, sort_keys=True))
    return 0
