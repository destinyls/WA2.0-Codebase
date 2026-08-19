# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Pinned WorldArena 2.0 AgileX storage and modality contract."""

from __future__ import annotations

OFFICIAL_REPO_ID = "WorldArena/WorldArena2.0"
OFFICIAL_REVISION = "af1ac34d3881f84096345542c631fbb1b9540d50"
OFFICIAL_SOURCE_FPS = 30
OFFICIAL_EPISODE_COUNT = 983
VISION_ONLY_REPO_ID = "agilex_vision_only"
VISION_TACTILE_REPO_ID = "agilex_vision_tactile"

VISION_ONLY_TASKS = (
    "clean_table",
    "clean_table_instruction_follow",
    "fold_box",
    "fold_shirt",
    "pour_over_coffee",
    "pour_water",
    "wipe_table",
)
VISION_TACTILE_TASKS = (
    "insert",
    "peel_cucumber",
    "pick_potato_chip",
)
OFFICIAL_TASKS = VISION_ONLY_TASKS + VISION_TACTILE_TASKS
EXPECTED_TASK_EPISODES = {
    task: (83 if task == "pour_over_coffee" else 100) for task in OFFICIAL_TASKS
}

PROMPT_KEYS = {
    **{task: task for task in OFFICIAL_TASKS},
    "clean_table_instruction_follow": "clean_table_instructions_follow",
}
ALLOWED_DEVICE_NAMES = frozenset(("cobot-magic-max", "cobot_magic_max", "recap"))

RGB_FILES = {
    "observation.images.top": "cam_high.mp4",
    "observation.images.wrist_l": "cam_left_wrist.mp4",
    "observation.images.wrist_r": "cam_right_wrist.mp4",
}
TACTILE_FILES = {
    "observation.images.tactile_l": "tactile_gripper_left_rectify.mp4",
    "observation.images.tactile_r": "tactile_gripper_right_rectify.mp4",
}
WRENCH_DATASETS = {
    "observation.wrench.left": "gripper_left/ForceResultant",
    "observation.wrench.right": "gripper_right/ForceResultant",
}
BASE_FILES = ("episode.hdf5", "meta.json", *RGB_FILES.values())
TACTILE_FILES_REQUIRED = (*TACTILE_FILES.values(), "tactile_information.hdf5")

RAW_ACTION_DATASET = "action"
RAW_QPOS_DATASET = "observations/qpos"
ACTION_OFFSETS_PER_LATENT_ANCHOR = tuple(range(12))


def task_has_contact(task: str) -> bool:
    if task not in OFFICIAL_TASKS:
        raise ValueError(f"unsupported official AgileX task: {task!r}")
    return task in VISION_TACTILE_TASKS


__all__ = (
    "ACTION_OFFSETS_PER_LATENT_ANCHOR",
    "ALLOWED_DEVICE_NAMES",
    "BASE_FILES",
    "EXPECTED_TASK_EPISODES",
    "OFFICIAL_EPISODE_COUNT",
    "OFFICIAL_REPO_ID",
    "OFFICIAL_REVISION",
    "OFFICIAL_SOURCE_FPS",
    "OFFICIAL_TASKS",
    "PROMPT_KEYS",
    "RAW_ACTION_DATASET",
    "RAW_QPOS_DATASET",
    "RGB_FILES",
    "TACTILE_FILES",
    "TACTILE_FILES_REQUIRED",
    "VISION_ONLY_TASKS",
    "VISION_ONLY_REPO_ID",
    "VISION_TACTILE_TASKS",
    "VISION_TACTILE_REPO_ID",
    "WRENCH_DATASETS",
    "task_has_contact",
)
