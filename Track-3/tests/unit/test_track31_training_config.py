"""Track 3.1 bounded-sequence configuration tests."""

from copy import deepcopy

import pytest

from n0_twam.configs.twam_track31_univtac_cfg import (
    _resolve_track31_stop_after_step_for_trainer,
    build_track31_invocation_contract,
    build_track31_tactile_training_contract,
    resolve_track31_load_worker,
    resolve_track31_max_latent_frames,
    resolve_track31_stop_after_step,
    resolve_track31_training_overrides,
    twam_track31_univtac_cfg,
)

_TRAINING_OVERRIDE_ENVS = (
    "N0_TRACK31_NUM_STEPS",
    "N0_TRACK31_SAVE_INTERVAL",
    "N0_TRACK31_VAL_INTERVAL",
    "N0_TRACK31_GRADIENT_ACCUMULATION_STEPS",
    "N0_TRACK31_BATCH_SIZE",
)


@pytest.fixture(autouse=True)
def _clear_staged_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("N0_TRACK31_STOP_AFTER_STEP", raising=False)


def test_track31_defaults_to_five_latent_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("N0_TRACK31_MAX_LATENT_FRAMES", raising=False)

    assert resolve_track31_max_latent_frames() == 5
    assert twam_track31_univtac_cfg.max_latent_frames > 0
    assert twam_track31_univtac_cfg.raw_tactile_evaluation is False
    assert twam_track31_univtac_cfg.deterministic_evaluation_crop_zero is False


def test_track31_accepts_explicit_positive_latent_frame_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("N0_TRACK31_MAX_LATENT_FRAMES", "17")

    assert resolve_track31_max_latent_frames() == 17


def test_track31_strict_resume_uses_main_process_data_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("N0_TRACK31_LOAD_WORKER", raising=False)

    assert resolve_track31_load_worker() == 0
    assert twam_track31_univtac_cfg.load_worker == 0


def test_track31_preserves_released_capacity_with_two_active_sensors() -> None:
    assert twam_track31_univtac_cfg.tactile_profile == "vision_tactile"
    contract = build_track31_tactile_training_contract(twam_track31_univtac_cfg)

    assert contract["max_tactile_streams"] == 4
    assert contract["active_tactile_sensor_count"] == 2
    assert contract["active_tactile_sensor_ids"] == [0, 1]
    assert all(
        sensor_id < contract["max_tactile_streams"]
        for sensor_id in contract["active_tactile_sensor_ids"]
    )


def test_track31_rejects_active_sensor_id_outside_model_capacity() -> None:
    config = deepcopy(twam_track31_univtac_cfg)
    config.tactile_sensor_id_map = {
        config.tactile_keys[0]: 0,
        config.tactile_keys[1]: config.max_tactile_streams,
    }

    with pytest.raises(ValueError, match="below model capacity"):
        build_track31_tactile_training_contract(config)


@pytest.mark.parametrize("invalid_value", ("1", "4", "-1", "worker"))
def test_track31_rejects_nonzero_or_invalid_load_worker(
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: str,
) -> None:
    monkeypatch.setenv("N0_TRACK31_LOAD_WORKER", invalid_value)

    with pytest.raises(ValueError, match="N0_TRACK31_LOAD_WORKER"):
        resolve_track31_load_worker()


def test_track31_training_overrides_default_to_immediate_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _TRAINING_OVERRIDE_ENVS:
        monkeypatch.delenv(name, raising=False)

    assert resolve_track31_training_overrides() == {
        "num_steps": 2000,
        "save_interval": 500,
        "val_interval": 100,
        "gradient_accumulation_steps": 1,
        "batch_size": 1,
    }


def test_track31_training_overrides_support_one_step_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = ("1", "1", "1", "1", "1")
    for name, value in zip(_TRAINING_OVERRIDE_ENVS, values, strict=True):
        monkeypatch.setenv(name, value)

    assert resolve_track31_training_overrides() == {
        "num_steps": 1,
        "save_interval": 1,
        "val_interval": 1,
        "gradient_accumulation_steps": 1,
        "batch_size": 1,
    }


def test_track31_stop_defaults_to_final_scheduler_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("N0_TRACK31_STOP_AFTER_STEP", raising=False)

    assert resolve_track31_stop_after_step(num_steps=2000) == 2000


def test_track31_invocation_contract_accepts_fresh_and_resume_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("N0_TRACK31_STOP_AFTER_STEP", "500")

    stop_after_step = resolve_track31_stop_after_step(num_steps=2000)
    fresh = build_track31_invocation_contract(
        start_step=0,
        stop_after_step=stop_after_step,
        num_steps=2000,
    )
    resumed = build_track31_invocation_contract(
        start_step=250,
        stop_after_step=stop_after_step,
        num_steps=2000,
    )

    assert fresh == {
        "start_step": 0,
        "stop_after_step": 500,
        "num_steps": 2000,
        "optimizer_steps_this_invocation": 500,
        "is_final_invocation": False,
    }
    assert resumed["optimizer_steps_this_invocation"] == 250


@pytest.mark.parametrize(
    ("start_step", "stop_after_step", "num_steps"),
    (
        (0, 0, 2000),
        (0, 2001, 2000),
        (500, 500, 2000),
        (501, 500, 2000),
    ),
)
def test_track31_invocation_contract_rejects_invalid_stage_boundaries(
    start_step: int,
    stop_after_step: int,
    num_steps: int,
) -> None:
    with pytest.raises(ValueError, match="start_step < stop_after_step <= num_steps"):
        build_track31_invocation_contract(
            start_step=start_step,
            stop_after_step=stop_after_step,
            num_steps=num_steps,
        )


@pytest.mark.parametrize("invalid_value", ("", "0", "-1", "1.5", "2001", "one"))
def test_track31_rejects_invalid_stop_after_step(
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: str,
) -> None:
    monkeypatch.setenv("N0_TRACK31_STOP_AFTER_STEP", invalid_value)

    with pytest.raises(ValueError, match="N0_TRACK31_STOP_AFTER_STEP"):
        resolve_track31_stop_after_step(num_steps=2000)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    (("2001", 2001), ("invalid", "invalid"), ("", "")),
)
def test_trainer_stop_resolution_defers_errors_until_rank_collective(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    expected: object,
) -> None:
    monkeypatch.setenv("N0_TRACK31_STOP_AFTER_STEP", raw_value)

    assert _resolve_track31_stop_after_step_for_trainer(num_steps=2000) == expected


@pytest.mark.parametrize("invalid_value", ("", "0", "-1", "+5", "5.0", "five"))
def test_track31_rejects_non_positive_or_non_decimal_latent_frame_limit(
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: str,
) -> None:
    monkeypatch.setenv("N0_TRACK31_MAX_LATENT_FRAMES", invalid_value)

    with pytest.raises(ValueError, match="N0_TRACK31_MAX_LATENT_FRAMES"):
        resolve_track31_max_latent_frames()


@pytest.mark.parametrize("name", _TRAINING_OVERRIDE_ENVS)
@pytest.mark.parametrize("invalid_value", ("0", "-1", "1.5", "one"))
def test_track31_rejects_invalid_positive_training_override(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    invalid_value: str,
) -> None:
    monkeypatch.setenv(name, invalid_value)

    with pytest.raises(ValueError, match=name):
        resolve_track31_training_overrides()
