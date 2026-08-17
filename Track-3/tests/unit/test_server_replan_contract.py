# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Synchronous re-grounding contracts shared by AgileX and Franka serving."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from easydict import EasyDict

from n0_twam.configs.twam_track3_agilex_mixed_cfg import (
    build_track3_agilex_mixed_config,
)
from n0_twam.configs.twam_track3_agilex_server_cfg import (
    build_track3_agilex_server_config,
)
from n0_twam.configs.twam_track3_agilex_vision_only_cfg import (
    build_track3_agilex_vision_only_config,
)
from n0_twam.configs.twam_track3_agilex_vision_tactile_cfg import (
    build_track3_agilex_vision_tactile_config,
)
from n0_twam.n0_twam_server import TWAM_Server

RGB_KEYS = [
    "observation.images.top",
    "observation.images.wrist_l",
    "observation.images.wrist_r",
]
TACTILE_KEYS = [
    "observation.images.tactile_l",
    "observation.images.tactile_r",
]
WRENCH_KEYS = [
    "observation.wrench.left",
    "observation.wrench.right",
]


def _route(*, tactile: bool) -> dict[str, object]:
    return {
        "embodiment": "agilex_dual_qpos14_v1",
        "action_schema": "qpos14_joint_absolute_v1",
        "rgb_keys": RGB_KEYS,
        "tactile_keys": TACTILE_KEYS if tactile else [],
        "wrench_keys": WRENCH_KEYS if tactile else [],
    }


def _routes(profile: str) -> dict[str, dict[str, object]]:
    if profile == "vision_tactile":
        return {"touch_a": _route(tactile=True), "touch_b": _route(tactile=True)}
    if profile == "mixed":
        return {"touch": _route(tactile=True), "rgb": _route(tactile=False)}
    return {"rgb": _route(tactile=False)}


class _CacheTransformer:
    """Small semantic cache double: committed entries survive pred eviction."""

    def __init__(self) -> None:
        self.entries = ["committed-real", "old-predicted"]
        self.clear_pred_calls = 0

    def clear_pred_cache(self, cache_name: str) -> None:
        assert cache_name == "pos"
        self.clear_pred_calls += 1
        self.entries = [entry for entry in self.entries if entry != "old-predicted"]


class _StreamingVAE:
    def clear_cache(self) -> None:
        return None


def _plain_infer_server(*, frame_st_id: int) -> tuple[TWAM_Server, _CacheTransformer]:
    server = object.__new__(TWAM_Server)
    transformer = _CacheTransformer()
    server.cache_name = "pos"
    server.frame_st_id = frame_st_id
    server.transformer = transformer
    server.streaming_vae = _StreamingVAE()
    server.job_config = SimpleNamespace()
    server._require_active_contact_route = lambda observation: None
    server._reset_tactile_state = lambda: None

    def infer_chunk(observation: object, frame_st_id: int) -> tuple[list[str], None]:
        before_generation = list(transformer.entries)
        transformer.entries.append("new-predicted")
        return before_generation, None

    server._infer = infer_chunk
    return server, transformer


@pytest.mark.parametrize("frame_st_id", [0, 12])
def test_full_replan_discards_only_old_predictions_before_plain_infer(
    frame_st_id: int,
) -> None:
    server, transformer = _plain_infer_server(frame_st_id=frame_st_id)

    result = server.infer({"full_replan": True})

    assert result["action"] == ["committed-real"]
    assert transformer.entries == ["committed-real", "new-predicted"]
    assert transformer.clear_pred_calls == 1


def test_plain_franka_request_without_full_replan_preserves_existing_cache() -> None:
    server, transformer = _plain_infer_server(frame_st_id=0)

    result = server.infer({})

    assert result["action"] == ["committed-real", "old-predicted"]
    assert transformer.clear_pred_calls == 0


def test_full_replan_rejects_ambiguous_non_boolean_flag() -> None:
    server, _ = _plain_infer_server(frame_st_id=0)

    with pytest.raises(ValueError, match="full_replan must be a boolean"):
        server.infer({"full_replan": 1})


@pytest.mark.parametrize(
    ("builder", "profile", "expected_denoise"),
    [
        (build_track3_agilex_vision_tactile_config, "vision_tactile", True),
        (build_track3_agilex_mixed_config, "mixed", True),
        (build_track3_agilex_vision_only_config, "vision_only", False),
    ],
)
def test_agilex_server_tactile_denoise_is_profile_derived_and_signed(
    builder: Callable[..., EasyDict],
    profile: str,
    expected_denoise: bool,
) -> None:
    training = builder(repo_routes=_routes(profile))
    training.server_tactile_denoise = not expected_denoise

    server = build_track3_agilex_server_config(training_config=training)

    assert server.server_tactile_denoise is expected_denoise
    assert server.server_runtime_contract["tactile_profile"] == profile
    assert server.server_runtime_contract["server_tactile_denoise"] is expected_denoise
    assert server.server_runtime_contract["full_replan_cache_policy"] == (
        "clear_predicted_preserve_committed_v1"
    )
    assert server.server_runtime_contract["contract_sha256"] == (
        server.server_runtime_contract_sha256
    )


def test_agilex_server_runtime_contract_ignores_deployment_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = build_track3_agilex_mixed_config(repo_routes=_routes("mixed"))
    monkeypatch.setenv("N0_TRACK3_AGILEX_SERVER_PORT", "30001")
    monkeypatch.setenv("N0_TRACK3_AGILEX_SERVE_OUTPUT", "/tmp/serve-a")
    first = build_track3_agilex_server_config(training_config=training)
    monkeypatch.setenv("N0_TRACK3_AGILEX_SERVER_PORT", "30002")
    monkeypatch.setenv("N0_TRACK3_AGILEX_SERVE_OUTPUT", "/tmp/serve-b")
    second = build_track3_agilex_server_config(training_config=training)

    assert first.port != second.port
    assert first.save_root != second.save_root
    assert first.server_runtime_contract == second.server_runtime_contract
