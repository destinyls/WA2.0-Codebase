# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Deterministic all-task proxy view builder for converted AgileX data."""

from __future__ import annotations

from pathlib import Path

from n0_twam.configs.twam_track3_agilex_contracts import load_repo_route_binding
from n0_twam.integrations.worldarena.agilex_official_schema import OFFICIAL_TASKS
from n0_twam.integrations.worldarena.agilex_policy_io import load_agilex_policy_config

from .agilex_evaluation_view import (
    AgileXEvaluationEntry,
    publish_agilex_evaluation_view,
)


def _task(value: object, *, prompts: dict[str, str]) -> str:
    raw = value if isinstance(value, (list, tuple)) else [value]
    if len(raw) != 1 or not isinstance(raw[0], str):
        raise ValueError(
            "AgileX converted episode must contain exactly one task prompt"
        )
    matches = [task for task, prompt in prompts.items() if prompt == raw[0]]
    if len(matches) != 1:
        raise ValueError("AgileX converted task prompt is absent or ambiguous")
    return matches[0]


def build_agilex_proxy_evaluation_view(
    *,
    config_path: Path,
    dataset_root: Path,
    output: Path,
    view_id: str,
    samples_per_task: int = 1,
) -> dict[str, object]:
    """Select the first converted episodes for all ten official tasks.

    The resulting roster is a training-distribution engineering proxy, never an
    independent holdout or an organizer score.
    """

    if type(samples_per_task) is not int or not 1 <= samples_per_task <= 100:
        raise ValueError("samples_per_task must be an integer in [1,100]")
    config = load_agilex_policy_config(config_path)
    root = Path(dataset_root).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("AgileX dataset root must be a regular directory")
    prompts = {
        task_id: route.prompt for task_id, route in config.policy.task_routes.items()
    }
    if set(prompts) != set(OFFICIAL_TASKS):
        raise ValueError("AgileX proxy view requires all ten official task routes")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ImportError as error:  # pragma: no cover - production dependency
        raise ImportError(
            "LeRobot is required to build the AgileX proxy view"
        ) from error

    selected: dict[str, list[AgileXEvaluationEntry]] = {
        task: [] for task in OFFICIAL_TASKS
    }
    binding = load_repo_route_binding(config.serve_bundle / "repo_route_manifest.json")
    expected_repos = set(binding.routes)
    repo_roots = tuple(
        path.parent.parent for path in sorted(root.glob("*/meta/info.json"))
    )
    if not repo_roots or {path.name for path in repo_roots} != expected_repos:
        raise ValueError("AgileX proxy view repository roster mismatch")
    for repo_root in repo_roots:
        metadata = LeRobotDatasetMetadata(
            repo_root.name,
            repo_root,
            None,
            force_cache_sync=False,
        )
        for episode in metadata.episodes.values():
            task = _task(episode["tasks"], prompts=prompts)
            if len(selected[task]) < samples_per_task:
                selected[task].append(
                    AgileXEvaluationEntry(
                        repo_id=repo_root.name,
                        episode_id=int(episode["episode_index"]),
                        task_id=task,
                    )
                )
    if any(len(selected[task]) != samples_per_task for task in OFFICIAL_TASKS):
        raise ValueError("AgileX proxy view could not cover every official task")
    entries = tuple(entry for task in OFFICIAL_TASKS for entry in selected[task])
    result = publish_agilex_evaluation_view(
        output=output,
        view_id=view_id,
        entries=entries,
    )
    return {
        **result,
        "execution_tier": "training_distribution_proxy_view",
        "independent_holdout": False,
        "organizer_evaluation_completed": False,
    }


__all__ = ("build_agilex_proxy_evaluation_view",)
