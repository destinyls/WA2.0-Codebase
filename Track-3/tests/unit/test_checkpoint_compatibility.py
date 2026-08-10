# Copyright 2025-2026 NeoteAI Team. All rights reserved.

import numpy as np
import pytest

from n0_twam.checkpointing.compatibility import (
    ACTION_PROJECTION_KEYS,
    build_action_migration_plan,
)


def _states(source_dim: int, target_dim: int) -> tuple[dict, dict]:
    source = {
        "backbone.weight": np.zeros((4, 4), dtype=np.float32),
        "action_embedder.weight": np.zeros((4, source_dim), dtype=np.float32),
        "action_embedder.bias": np.zeros((4,), dtype=np.float32),
        "action_proj_out.weight": np.zeros((source_dim, 4), dtype=np.float32),
        "action_proj_out.bias": np.zeros((source_dim,), dtype=np.float32),
    }
    target = {
        "backbone.weight": np.zeros((4, 4), dtype=np.float32),
        "action_embedder.weight": np.zeros((4, target_dim), dtype=np.float32),
        "action_embedder.bias": np.zeros((4,), dtype=np.float32),
        "action_proj_out.weight": np.zeros((target_dim, 4), dtype=np.float32),
        "action_proj_out.bias": np.zeros((target_dim,), dtype=np.float32),
    }
    return source, target


def test_action_migration_resets_only_projection_allowlist() -> None:
    source, target = _states(source_dim=20, target_dim=8)

    copied, plan = build_action_migration_plan(
        source,
        target,
        source_action_dim=20,
        target_action_dim=8,
    )

    assert set(copied) == {"backbone.weight"}
    assert plan.copied_keys == ("backbone.weight",)
    assert plan.reset_keys == tuple(sorted(ACTION_PROJECTION_KEYS))
    assert not plan.missing_target_keys
    assert not plan.unexpected_source_keys
    assert not plan.shape_mismatches
    plan.raise_for_incompatible()


def test_action_migration_resets_semantic_head_even_when_dims_match() -> None:
    source, target = _states(source_dim=8, target_dim=8)

    copied, plan = build_action_migration_plan(source, target)

    assert set(copied) == {"backbone.weight"}
    assert set(plan.reset_keys) == ACTION_PROJECTION_KEYS


def test_action_migration_rejects_non_action_shape_mismatch() -> None:
    source, target = _states(source_dim=20, target_dim=8)
    target["backbone.weight"] = np.zeros((5, 4), dtype=np.float32)

    _, plan = build_action_migration_plan(source, target)

    with pytest.raises(ValueError, match="shape mismatches"):
        plan.raise_for_incompatible()


def test_action_migration_rejects_unexpected_or_missing_non_action_keys() -> None:
    source, target = _states(source_dim=20, target_dim=8)
    source["obsolete.weight"] = np.zeros((1,), dtype=np.float32)
    target["new_module.weight"] = np.zeros((1,), dtype=np.float32)

    _, plan = build_action_migration_plan(source, target)

    with pytest.raises(ValueError, match="unexpected source keys"):
        plan.raise_for_incompatible()

    del source["obsolete.weight"]
    _, plan = build_action_migration_plan(source, target)
    with pytest.raises(ValueError, match="missing target keys"):
        plan.raise_for_incompatible(tolerated_missing_prefixes=("obsolete",))


def test_action_migration_allows_explicit_module_flip_prefix() -> None:
    source, target = _states(source_dim=20, target_dim=8)
    target["local_tactile_cross_attn.weight"] = np.zeros((2, 2), dtype=np.float32)

    _, plan = build_action_migration_plan(source, target)

    plan.raise_for_incompatible(
        tolerated_missing_prefixes=("local_tactile_cross_attn",)
    )


def test_action_migration_requires_all_projection_keys_on_both_sides() -> None:
    source, target = _states(source_dim=20, target_dim=8)
    del source["action_proj_out.bias"]

    with pytest.raises(ValueError, match="source is missing required action keys"):
        build_action_migration_plan(source, target)


def test_action_migration_validates_declared_projection_dimensions() -> None:
    source, target = _states(source_dim=20, target_dim=8)

    with pytest.raises(ValueError, match="source action projection shapes"):
        build_action_migration_plan(
            source,
            target,
            source_action_dim=30,
            target_action_dim=8,
        )
    with pytest.raises(ValueError, match="target action projection shapes"):
        build_action_migration_plan(
            source,
            target,
            source_action_dim=20,
            target_action_dim=7,
        )
