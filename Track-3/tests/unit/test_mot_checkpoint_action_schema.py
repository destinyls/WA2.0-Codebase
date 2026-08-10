"""Regression tests for MoT action-schema config round trips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import safetensors.torch
import torch
from diffusers.configuration_utils import ConfigMixin

from n0_twam.models import mot, utils


class _FakeMoTModel(ConfigMixin):
    config_name = "config.json"

    def __init__(self, **kwargs: object) -> None:
        self.register_to_config(**kwargs)
        self._state = {"weight": torch.zeros(1)}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._state)

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        strict: bool,
    ) -> tuple[list[str], list[str]]:
        del strict
        self._state = dict(state_dict)
        return [], []

    def register_to_config(self, **kwargs: object) -> None:
        super().register_to_config(**kwargs)

    def eval(self) -> _FakeMoTModel:
        return self

    def to(self, device: object) -> _FakeMoTModel:
        del device
        return self


def test_strict_mot_load_preserves_validated_action_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "is_mot": True,
                "action_dim": 8,
                "action_schema": "qpos8_next_step",
                "mot_cross_attn_experts": ["video", "action"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mot, "WanMoTTransformer3DModel", _FakeMoTModel)
    monkeypatch.setattr(
        safetensors.torch,
        "load_file",
        lambda path: {"weight": torch.zeros(1)},
    )

    model = utils.load_mot_checkpoint(
        tmp_path,
        torch_dtype=torch.float32,
        torch_device="cpu",
        compatibility="strict",
        target_action_dim=8,
        target_action_schema="qpos8_next_step",
    )

    assert model.config["action_dim"] == 8
    assert model.config["action_schema"] == "qpos8_next_step"
    saved_config_dir = tmp_path / "saved_config"
    model.save_config(saved_config_dir)
    saved_config = json.loads(
        (saved_config_dir / "config.json").read_text(encoding="utf-8")
    )
    assert saved_config["action_dim"] == 8
    assert saved_config["action_schema"] == "qpos8_next_step"


@pytest.mark.parametrize("recorded_schema", [None, "ee20_pi05"])
def test_strict_mot_load_rejects_missing_or_wrong_action_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_schema: str | None,
) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    config: dict[str, object] = {
        "is_mot": True,
        "action_dim": 8,
        "mot_cross_attn_experts": ["video", "action"],
    }
    if recorded_schema is not None:
        config["action_schema"] = recorded_schema
    (transformer_dir / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    monkeypatch.setattr(mot, "WanMoTTransformer3DModel", _FakeMoTModel)
    monkeypatch.setattr(
        safetensors.torch,
        "load_file",
        lambda path: {"weight": torch.zeros(1)},
    )

    with pytest.raises(ValueError, match="strict checkpoint action schema mismatch"):
        utils.load_mot_checkpoint(
            tmp_path,
            torch_dtype=torch.float32,
            torch_device="cpu",
            compatibility="strict",
            target_action_dim=8,
            target_action_schema="qpos8_next_step",
        )


def test_strict_mot_load_requires_explicit_non_track31_schema(
    tmp_path: Path,
) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps({"is_mot": True, "action_dim": 20}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict checkpoint action schema mismatch"):
        utils.load_mot_checkpoint(
            tmp_path,
            torch_dtype=torch.float32,
            torch_device="cpu",
            compatibility="strict",
            target_action_dim=20,
            target_action_schema="ee20_pi05",
        )
