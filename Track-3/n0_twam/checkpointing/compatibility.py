# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Fail-closed planning for EE20-to-qpos8 action-head migration."""

from dataclasses import dataclass
from typing import Any, Mapping

ACTION_PROJECTION_KEYS = frozenset(
    {
        "action_embedder.weight",
        "action_embedder.bias",
        "action_proj_out.weight",
        "action_proj_out.bias",
    }
)


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(size) for size in value.shape)
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            f"state-dict value has no valid shape: {type(value)!r}"
        ) from exc


def _validate_projection_shapes(
    state: Mapping[str, Any],
    *,
    action_dim: int,
    label: str,
) -> None:
    embedder_weight = _shape(state["action_embedder.weight"])
    embedder_bias = _shape(state["action_embedder.bias"])
    output_weight = _shape(state["action_proj_out.weight"])
    output_bias = _shape(state["action_proj_out.bias"])
    if len(embedder_weight) != 2:
        raise ValueError(f"{label} action_embedder.weight must be 2D")
    inner_dim = embedder_weight[0]
    expected = {
        "action_embedder.weight": (inner_dim, action_dim),
        "action_embedder.bias": (inner_dim,),
        "action_proj_out.weight": (action_dim, inner_dim),
        "action_proj_out.bias": (action_dim,),
    }
    actual = {
        "action_embedder.weight": embedder_weight,
        "action_embedder.bias": embedder_bias,
        "action_proj_out.weight": output_weight,
        "action_proj_out.bias": output_bias,
    }
    mismatched = [
        f"{key}:{actual[key]}!={shape}"
        for key, shape in expected.items()
        if actual[key] != shape
    ]
    if mismatched:
        raise ValueError(
            f"{label} action projection shapes violate action_dim={action_dim}: "
            + ", ".join(mismatched)
        )


@dataclass(frozen=True)
class TensorShapeMismatch:
    """One non-action tensor that cannot be copied safely."""

    key: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "source_shape": list(self.source_shape),
            "target_shape": list(self.target_shape),
        }


@dataclass(frozen=True)
class ActionMigrationPlan:
    """Exact copied/reset/rejected key partition for one migration."""

    copied_keys: tuple[str, ...]
    reset_keys: tuple[str, ...]
    missing_target_keys: tuple[str, ...]
    unexpected_source_keys: tuple[str, ...]
    shape_mismatches: tuple[TensorShapeMismatch, ...]

    def raise_for_incompatible(
        self,
        *,
        tolerated_missing_prefixes: tuple[str, ...] = (),
    ) -> None:
        """Reject every incompatibility outside explicit new-module prefixes."""

        if self.unexpected_source_keys:
            raise ValueError(
                "action migration has unexpected source keys: "
                + ", ".join(self.unexpected_source_keys[:8])
            )
        if self.shape_mismatches:
            examples = ", ".join(
                f"{item.key}:{item.source_shape}->{item.target_shape}"
                for item in self.shape_mismatches[:8]
            )
            raise ValueError(
                "action migration has non-action shape mismatches: " + examples
            )
        missing = tuple(
            key
            for key in self.missing_target_keys
            if not any(key.startswith(prefix) for prefix in tolerated_missing_prefixes)
        )
        if missing:
            raise ValueError(
                "action migration has missing target keys: " + ", ".join(missing[:8])
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "copied_keys": list(self.copied_keys),
            "reset_keys": list(self.reset_keys),
            "missing_target_keys": list(self.missing_target_keys),
            "unexpected_source_keys": list(self.unexpected_source_keys),
            "shape_mismatches": [
                mismatch.to_json_dict() for mismatch in self.shape_mismatches
            ],
        }


def build_action_migration_plan(
    source_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    *,
    source_action_dim: int | None = None,
    target_action_dim: int | None = None,
) -> tuple[dict[str, Any], ActionMigrationPlan]:
    """Copy exact non-action tensors and reset the full semantic action head."""

    missing_source_action = sorted(ACTION_PROJECTION_KEYS - set(source_state))
    if missing_source_action:
        raise ValueError(
            "source is missing required action keys: "
            + ", ".join(missing_source_action)
        )
    missing_target_action = sorted(ACTION_PROJECTION_KEYS - set(target_state))
    if missing_target_action:
        raise ValueError(
            "target is missing required action keys: "
            + ", ".join(missing_target_action)
        )
    if source_action_dim is not None:
        _validate_projection_shapes(
            source_state,
            action_dim=int(source_action_dim),
            label="source",
        )
    if target_action_dim is not None:
        _validate_projection_shapes(
            target_state,
            action_dim=int(target_action_dim),
            label="target",
        )

    copied: dict[str, Any] = {}
    mismatches: list[TensorShapeMismatch] = []
    unexpected: list[str] = []
    for key, source_value in source_state.items():
        if key in ACTION_PROJECTION_KEYS:
            continue
        target_value = target_state.get(key)
        if target_value is None:
            unexpected.append(key)
            continue
        source_shape = _shape(source_value)
        target_shape = _shape(target_value)
        if source_shape != target_shape:
            mismatches.append(
                TensorShapeMismatch(
                    key=key,
                    source_shape=source_shape,
                    target_shape=target_shape,
                )
            )
            continue
        copied[key] = source_value

    missing_target = sorted(set(target_state) - set(copied) - ACTION_PROJECTION_KEYS)
    plan = ActionMigrationPlan(
        copied_keys=tuple(sorted(copied)),
        reset_keys=tuple(sorted(ACTION_PROJECTION_KEYS)),
        missing_target_keys=tuple(missing_target),
        unexpected_source_keys=tuple(sorted(unexpected)),
        shape_mismatches=tuple(sorted(mismatches, key=lambda item: item.key)),
    )
    return copied, plan
