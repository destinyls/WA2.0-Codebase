from __future__ import annotations

from pathlib import Path

import numpy as np

from n0_twam.cli import run_cli
from n0_twam.evaluation.franka_policy_conditioning_parity import _summary


def test_policy_conditioning_summary_accepts_identical_pose_metrics() -> None:
    target = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.5],
            [0.2, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )
    prediction = target.copy()
    prediction[:, 0] += 0.004

    result = _summary(
        raw_actions=prediction,
        latent_actions=prediction.copy(),
        target_actions=target,
    )

    assert result["raw_vs_cached"]["position_error_cm_max"] == 0.0
    assert result["training_corpus_regression"][
        "raw_rgb_position_error_cm_mean"
    ] == result["training_corpus_regression"][
        "cached_latent_position_error_cm_mean"
    ]
    assert result["acceptance"]["passed"] is True


def test_policy_conditioning_parity_cli_routes_all_inputs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"acceptance": {"passed": True}}

    monkeypatch.setattr(
        "n0_twam.evaluation.franka_policy_conditioning_parity."
        "evaluate_franka_policy_conditioning_parity",
        fake_evaluate,
    )
    result = run_cli(
        [
            "track32",
            "policy-conditioning-parity",
            "--policy-config",
            "policy.json",
            "--artifact-root",
            "artifacts",
            "--lerobot-root",
            "lerobot",
            "--base-model",
            "base-model",
            "--normalizer",
            "normalizer.json",
            "--dataset-view",
            "wipe.json",
            "--output",
            "metrics.json",
            "--samples-output",
            "samples.npz",
            "--max-samples",
            "12",
            "--action-selection",
            "last_future_action",
        ]
    )

    assert result == {"acceptance": {"passed": True}}
    assert captured == {
        "policy_config": Path("policy.json"),
        "artifact_root": Path("artifacts"),
        "lerobot_root": Path("lerobot"),
        "base_model": Path("base-model"),
        "normalizer": Path("normalizer.json"),
        "dataset_view": Path("wipe.json"),
        "output": Path("metrics.json"),
        "samples_output": Path("samples.npz"),
        "max_samples": 12,
        "action_selection": "last_future_action",
    }
