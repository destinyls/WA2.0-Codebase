"""Deterministic worker sharding for formal Track 3.1 latent encoding."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

FORMAL_PHYSICAL_SPLITS = ("train759", "frozen40")


def resolve_execution_device(
    requested_device: str,
    accelerator_available: bool,
) -> str:
    """Require an explicit CPU request; never downgrade a CUDA request."""

    if requested_device == "cpu":
        return "cpu"
    if not requested_device.startswith("cuda"):
        raise ValueError("latent encoding device must be cpu or cuda[:index]")
    if not accelerator_available:
        raise RuntimeError(
            f"requested {requested_device!r}, but no CUDA accelerator is available; "
            "use --device cpu only for an explicit CPU run"
        )
    return requested_device


def validate_shard_contract(
    *,
    split: str | None,
    artifact_root: Path | None,
    episodes: list[int] | None,
    num_shards: int | None,
    shard_index: int | None,
    defer_inventory: bool,
) -> None:
    """Require deterministic, disjoint sharding for formal physical repos."""

    has_num_shards = num_shards is not None
    has_shard_index = shard_index is not None
    if has_num_shards != has_shard_index:
        raise ValueError("--num-shards and --shard-index must be provided together")
    if not has_num_shards:
        if defer_inventory:
            raise ValueError(
                "--defer-inventory requires --num-shards and --shard-index"
            )
        if split in FORMAL_PHYSICAL_SPLITS and episodes is not None:
            raise ValueError(
                "formal Track 3.1 does not accept manual --episodes; use the "
                "deterministic shard options"
            )
        return
    if split not in FORMAL_PHYSICAL_SPLITS or artifact_root is None:
        raise ValueError("sharding is available only for a formal physical split")
    if episodes is not None:
        raise ValueError("--episodes cannot be combined with formal sharding")
    if num_shards is None or num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index is None or not 0 <= shard_index < num_shards:
        raise ValueError("shard index (--shard-index) must be in [0, num_shards)")
    if not defer_inventory:
        raise ValueError("formal sharding requires --defer-inventory")


def select_episode_shard(
    episodes_df: pd.DataFrame,
    *,
    num_shards: int,
    shard_index: int,
) -> pd.DataFrame:
    """Select one disjoint modulo shard without assuming contiguous IDs."""

    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard index/count")
    if "episode_index" not in episodes_df.columns:
        raise ValueError("episode metadata has no episode_index")
    episode_ids = episodes_df["episode_index"].astype(int)
    if episode_ids.duplicated().any():
        raise ValueError("episode metadata contains duplicate episode_index values")
    selected = episodes_df[(episode_ids % num_shards) == shard_index].copy()
    return selected.sort_values("episode_index").reset_index(drop=True)
