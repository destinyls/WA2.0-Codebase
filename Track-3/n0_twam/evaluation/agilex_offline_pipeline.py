# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""One-command AgileX training-distribution offline evaluation pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

from n0_twam.evaluation.franka_atomic_io import publish_atomic_file
from n0_twam.evaluation.franka_prediction_io import capture_prediction_input

from .agilex_offline_generation import (
    generate_agilex_offline_predictions,
    prepare_agilex_evaluation_environment,
)
from .agilex_offline_metrics import evaluate_agilex_offline_predictions
from .agilex_prediction_artifact import verify_agilex_prediction_file


def _build_proxy_view(**kwargs: object) -> dict[str, object]:
    """Import config-dependent view code only after request env installation."""

    from .agilex_proxy_view import build_agilex_proxy_evaluation_view

    return build_agilex_proxy_evaluation_view(**kwargs)  # type: ignore[arg-type]


def _new_root(path: Path) -> Path:
    lexical = Path(path).expanduser()
    if lexical.exists() or lexical.is_symlink():
        raise FileExistsError(f"AgileX offline evaluation output exists: {lexical}")
    destination = lexical.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(destination, 0o750)
    return destination


def _publish_closeout(path: Path, payload: dict[str, object]) -> str:
    raw = (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )

    def writer(handle: BinaryIO) -> None:
        handle.write(raw)

    def validator(candidate: bytes) -> None:
        loaded = json.loads(candidate)
        if not isinstance(loaded, dict) or loaded.get("status") != "complete":
            raise ValueError("AgileX offline closeout is not complete")

    return publish_atomic_file(
        output=path,
        writer=writer,
        validator=validator,
        label="AgileX offline closeout",
    ).sha256


def recover_agilex_offline_closeout(
    *,
    train_request: Path,
    policy_config: Path,
    evaluation_view: Path,
    predictions: Path,
    metrics: Path,
    replay_observation: Path,
    output_root: Path,
    replay_steps: int = 7,
    control_hz: float = 10.0,
    seed: int = 20260813,
    device: str = "0",
) -> dict[str, object]:
    """Replay and close out already-generated, hash-bound AgileX artifacts."""

    from ._agilex_offline_recovery import recover_agilex_offline_closeout as recover

    return recover(
        train_request=train_request,
        policy_config=policy_config,
        evaluation_view=evaluation_view,
        predictions=predictions,
        metrics=metrics,
        replay_observation=replay_observation,
        output_root=output_root,
        new_root=_new_root,
        build_closeout=_closeout_core,
        publish_closeout=_publish_closeout,
        prepare_environment=prepare_agilex_evaluation_environment,
        replay_steps=replay_steps,
        control_hz=control_hz,
        seed=seed,
        device=device,
    )


def _closeout_core(
    *,
    metadata: Mapping[str, object],
    view_path: Path,
    prediction_path: Path,
    metrics_path: Path,
    observation_path: Path,
    replay_path: Path,
    prediction_sha256: str,
    metrics_sha256: str,
    replay: Mapping[str, object],
    sample_count: int,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "n0_twam_track32_agilex_offline_closeout",
        "status": "complete",
        "execution_tier": "training_distribution_offline_proxy",
        "safety_contract_tier": "offline_non_robot_envelope",
        "independent_holdout": False,
        "organizer_evaluation_completed": False,
        "real_robot_evaluation_completed": False,
        "checkpoint_identity_sha256": metadata["checkpoint_identity_sha256"],
        "dataset_view_id": metadata["dataset_view_id"],
        "dataset_view_sha256": metadata["dataset_view_sha256"],
        "view_file_sha256": capture_prediction_input(view_path).sha256,
        "prediction_file_sha256": prediction_sha256,
        "metrics_file_sha256": metrics_sha256,
        "replay_observation_file_sha256": capture_prediction_input(
            observation_path
        ).sha256,
        "replay_receipt_file_sha256": replay["receipt_file_sha256"],
        "sample_count": sample_count,
        "metrics": dict(metrics),
        "realtime_assessment": replay["realtime_assessment"],
        "paths": {
            "view": str(view_path),
            "predictions": str(prediction_path),
            "metrics": str(metrics_path),
            "replay_observation": str(observation_path),
            "policy_replay": str(replay_path),
        },
    }


def run_agilex_offline_evaluation(
    *,
    train_request: Path,
    policy_config: Path,
    dataset_root: Path,
    output_root: Path,
    seed: int = 20260813,
    device: str = "0",
    decode_batch_size: int = 4,
    samples_per_task: int = 1,
    replay_steps: int = 7,
    control_hz: float = 10.0,
) -> dict[str, object]:
    """Build view, predict, score, replay, and seal one offline closeout."""

    root = _new_root(output_root)
    prepare_agilex_evaluation_environment(
        train_request=train_request,
        device=device,
        seed=seed,
    )
    view_path = root / "evaluation_view.json"
    prediction_path = root / "predictions.npz"
    observation_path = root / "replay_observation.npz"
    metrics_path = root / "metrics.json"
    replay_path = root / "policy_replay.json"
    closeout_path = root / "closeout.json"
    view = _build_proxy_view(
        config_path=policy_config,
        dataset_root=dataset_root,
        output=view_path,
        view_id="agilex-all-task-training-proxy-v1",
        samples_per_task=samples_per_task,
    )
    generation = generate_agilex_offline_predictions(
        train_request=train_request,
        policy_config=policy_config,
        dataset_view=view_path,
        output=prediction_path,
        seed=seed,
        device=device,
        decode_batch_size=decode_batch_size,
        replay_observation_output=observation_path,
    )
    artifact = verify_agilex_prediction_file(
        prediction_path,
        expected_sha256=str(generation["sha256"]),
    )
    metadata = artifact.metadata
    metrics = evaluate_agilex_offline_predictions(
        predictions=prediction_path,
        output=metrics_path,
        checkpoint_identity_sha256=str(metadata["checkpoint_identity_sha256"]),
        dataset_view_id=str(metadata["dataset_view_id"]),
        dataset_view_sha256=str(metadata["dataset_view_sha256"]),
        decoder_sha256=str(metadata["decoder_sha256"]),
    )

    from n0_twam.integrations.worldarena.agilex_policy_replay import (
        run_agilex_policy_replay,
    )

    replay = run_agilex_policy_replay(
        config_path=policy_config,
        observation_path=observation_path,
        steps=replay_steps,
        output=replay_path,
        control_hz=control_hz,
        require_realtime=False,
    )
    core = _closeout_core(
        metadata=metadata,
        view_path=view_path,
        prediction_path=prediction_path,
        metrics_path=metrics_path,
        observation_path=observation_path,
        replay_path=replay_path,
        prediction_sha256=str(generation["sha256"]),
        metrics_sha256=str(metrics["report_file_sha256"]),
        replay=replay,
        sample_count=int(generation["sample_count"]),
        metrics=metrics["overall"],  # type: ignore[arg-type]
    )
    closeout_sha256 = _publish_closeout(closeout_path, core)
    return {
        **core,
        "view": view,
        "closeout": str(closeout_path),
        "closeout_file_sha256": closeout_sha256,
    }


__all__ = (
    "recover_agilex_offline_closeout",
    "run_agilex_offline_evaluation",
)
