# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from pathlib import Path

import pytest

from n0_twam.integrations.univtac.convert_lerobot import (
    build_lerobot_table_inventory,
)


def _write_inventory_file(split_root: Path) -> None:
    metadata_root = split_root / "meta"
    metadata_root.mkdir(parents=True)
    (metadata_root / "info.json").write_text("{}", encoding="utf-8")


@pytest.mark.parametrize(
    "symlink_level",
    ("repo", "split", "subtree", "file"),
)
def test_table_inventory_rejects_symlink_at_any_level(
    tmp_path: Path,
    symlink_level: str,
) -> None:
    if symlink_level == "repo":
        real_repo = tmp_path / "real-repo"
        _write_inventory_file(real_repo / "train")
        linked_repo = tmp_path / "linked-repo"
        linked_repo.symlink_to(real_repo, target_is_directory=True)
        split_root = linked_repo / "train"
    elif symlink_level == "split":
        real_split = tmp_path / "real-split"
        _write_inventory_file(real_split)
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        split_root = repo_root / "train"
        split_root.symlink_to(real_split, target_is_directory=True)
    elif symlink_level == "subtree":
        split_root = tmp_path / "train"
        split_root.mkdir()
        outside_meta = tmp_path / "outside-meta"
        outside_meta.mkdir()
        (outside_meta / "info.json").write_text("{}", encoding="utf-8")
        (split_root / "meta").symlink_to(outside_meta, target_is_directory=True)
    else:
        split_root = tmp_path / "train"
        metadata_root = split_root / "meta"
        metadata_root.mkdir(parents=True)
        outside_file = tmp_path / "outside.json"
        outside_file.write_text("{}", encoding="utf-8")
        (metadata_root / "info.json").symlink_to(outside_file)

    with pytest.raises(ValueError, match="symlink|outside"):
        build_lerobot_table_inventory(split_root)
