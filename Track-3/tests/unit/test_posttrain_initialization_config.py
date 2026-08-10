"""Regression tests for released-checkpoint post-training semantics."""

from n0_twam.configs.twam_mixed_cfg import twam_mixed_cfg
from n0_twam.configs.twam_posttrain_cfg import twam_posttrain_cfg
from n0_twam.configs.twam_posttrain_server_cfg import twam_posttrain_server_cfg


def test_posttrain_uses_released_checkpoint_as_weights_only_initialization() -> None:
    assert twam_posttrain_cfg.tactile_profile == "vision_tactile"
    assert twam_posttrain_cfg.resume_from is None
    assert isinstance(twam_posttrain_cfg.init_from, str)
    assert twam_posttrain_cfg.init_from
    assert twam_posttrain_cfg.checkpoint_compatibility == "strict"
    assert twam_posttrain_cfg.strict_training_resume is False


def test_posttrain_server_does_not_inherit_training_checkpoint_source() -> None:
    assert twam_posttrain_server_cfg.resume_from is None
    assert twam_posttrain_server_cfg.init_from is None


def test_mixed_posttrain_has_explicit_tactile_and_rgb_only_repos() -> None:
    assert twam_mixed_cfg.tactile_profile == "mixed"
    assert any(twam_mixed_cfg.per_repo_tactile_keys.values())
    assert any(not keys for keys in twam_mixed_cfg.per_repo_tactile_keys.values())
    assert twam_mixed_cfg.tactile_optional is False
    assert twam_mixed_cfg.freeze_tactile_parameters is False
