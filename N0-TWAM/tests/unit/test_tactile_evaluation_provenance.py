# Copyright 2025-2026 NeoteAI Team. All rights reserved.

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler, TensorDataset

from n0_twam.checkpointing.identity import (
    TRANSFORMER_SENTINEL_KEYS,
    TRANSFORMER_WEIGHTS_FILENAME,
)
from n0_twam.evaluation.tactile_checkpoint import (
    audit_evaluation_checkpoint,
    validate_checkpoint_dataset_binding,
)
from n0_twam.evaluation.tactile_provenance import (
    OFFLINE_TACTILE_PROTOCOL,
    TACTILE_METRIC_DOMAIN,
    audit_evaluation_dataset,
    build_deterministic_eval_loader,
    build_provenance_bound_report,
    capture_source_sample_metadata,
    mask_future_local_tactile,
    resolve_dataset_index_metadata,
    restore_raw_video_context,
    set_evaluation_seed,
    write_json_atomic,
)


def _transformer_identity() -> dict[str, object]:
    return {
        "schema_version": 1,
        "file_name": TRANSFORMER_WEIGHTS_FILENAME,
        "size_bytes": 123,
        "sha256": "a" * 64,
        "tensor_count": 7,
        "action_dim": 8,
        "action_shapes": {
            "action_embedder.weight": [3072, 8],
            "action_embedder.bias": [3072],
            "action_proj_out.weight": [8, 3072],
            "action_proj_out.bias": [8],
        },
        "required_sentinel_keys": list(TRANSFORMER_SENTINEL_KEYS),
    }


def _metric_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "metric": "tactile_prediction_quality",
        "protocol": OFFLINE_TACTILE_PROTOCOL,
        "pixel_domain": TACTILE_METRIC_DOMAIN,
        "leaderboard_compatible": False,
        "aggregation": "macro_over_videos",
        "input_pairs_sha256": "b" * 64,
        "skip_first_frames_per_video": 1,
        "overall": {
            "video_count": 2,
            "frame_count": 32,
            "psnr": 20.0,
            "ssim": 0.5,
        },
        "per_video": [
            {
                "video_id": "lift_bottle/sample_000000/sensor_0",
                "task": "lift_bottle",
                "episode": "sample_000000",
                "sensor": "sensor_0",
                "frame_count": 16,
            },
            {
                "video_id": "lift_bottle/sample_000000/sensor_1",
                "task": "lift_bottle",
                "episode": "sample_000000",
                "sensor": "sensor_1",
                "frame_count": 16,
            },
        ],
    }


def _model_contract() -> dict[str, object]:
    return {
        "patch_size": [1, 2, 2],
        "snr_shift": 5.0,
        "use_local_tactile": True,
        "max_tactile_streams": 4,
        "tactile_in_channels": 3,
        "tactile_num_tokens": 4,
        "tactile_encoder_dim": 256,
        "tactile_latent_channels": 48,
    }


def _tactile_head_contract() -> dict[str, object]:
    inner_dim = 3072
    patch_channels = 192
    rope_max_seq_len = 1024
    shapes = {
        "tactile_patch_embed.weight": [inner_dim, patch_channels],
        "tactile_patch_embed.bias": [inner_dim],
        "sensor_id_embed.weight": [4, inner_dim],
        "tactile_norm.weight": [inner_dim],
        "tactile_norm.bias": [inner_dim],
        "tactile_proj_out.weight": [patch_channels, inner_dim],
        "tactile_proj_out.bias": [patch_channels],
        "local_tactile_patch_embed.weight": [inner_dim, patch_channels],
        "local_tactile_patch_embed.bias": [inner_dim],
        "local_tactile_sensor_embed.weight": [4, inner_dim],
        "local_tactile_frame_embed.weight": [rope_max_seq_len, inner_dim],
        "local_tactile_h_embed.weight": [rope_max_seq_len, inner_dim],
        "local_tactile_w_embed.weight": [rope_max_seq_len, inner_dim],
        "local_tactile_norm.weight": [inner_dim],
        "local_tactile_norm.bias": [inner_dim],
        "local_tactile_cross_attn.to_q.weight": [inner_dim, inner_dim],
        "local_tactile_cross_attn.to_q.bias": [inner_dim],
        "local_tactile_cross_attn.to_k.weight": [inner_dim, inner_dim],
        "local_tactile_cross_attn.to_k.bias": [inner_dim],
        "local_tactile_cross_attn.to_v.weight": [inner_dim, inner_dim],
        "local_tactile_cross_attn.to_v.bias": [inner_dim],
        "local_tactile_cross_attn.to_out.0.weight": [inner_dim, inner_dim],
        "local_tactile_cross_attn.to_out.0.bias": [inner_dim],
        "local_tactile_cross_attn.norm_q.weight": [inner_dim],
        "local_tactile_cross_attn.norm_k.weight": [inner_dim],
    }
    return {
        "schema_version": 1,
        "inner_dim": inner_dim,
        "patch_volume": 4,
        "rope_max_seq_len": rope_max_seq_len,
        "required_tensor_shapes": shapes,
    }


def _checkpoint_provenance() -> dict[str, object]:
    return {
        "transformer_directory": "/checkpoint/transformer",
        "config_path": "/checkpoint/transformer/config.json",
        "config_sha256": "c" * 64,
        "training_metadata": {
            "path": "/checkpoint/train_meta.json",
            "sha256": "d" * 64,
            "snr_shift": 5.0,
            "active_tactile_sensor_count": 2,
            "active_tactile_sensor_ids": [0, 1],
            "tactile_sensor_id_map": {
                "observation.images.tactile_a": 0,
                "observation.images.tactile_b": 1,
            },
            "track31_artifacts": _track31_artifacts(),
        },
        "transformer_sha256": "a" * 64,
        "transformer_identity": _transformer_identity(),
        "model_contract": _model_contract(),
        "tactile_head_contract": _tactile_head_contract(),
    }


def _track31_artifacts() -> dict[str, object]:
    return {
        "manifest_sha256": "1" * 64,
        "normalizer_sha256": "2" * 64,
        "conversion_report_sha256": "3" * 64,
        "train_view_id": "stage_a_final759_v1",
        "train_view_sha256": "4" * 64,
        "normalizer_source_view_id": "stage_a_final759_v1",
        "normalizer_source_view_sha256": "4" * 64,
    }


def _sample_selection() -> dict[str, object]:
    return {
        "output_sample_id": "sample_000000",
        "dataset_index": 0,
        "loader_cycle": 0,
        "source_metadata": {
            "tasks": ["lift_bottle"],
            "tactile_sensor_ids": [[0, 1]],
        },
        "unavailable_source_metadata_keys": [],
        "available_batch_keys": ["tactile_global_latent"],
        "batch_tensor_contract": {
            "tactile_global_latent": {"shape": [1, 2, 48, 5, 8, 8]}
        },
        "decoded_tactile_streams": [
            {
                "sensor_id": sensor_id,
                "decoded_frame_count": 17,
                "written_frame_indices": list(range(17)),
                "metric_scored_frame_indices": list(range(1, 17)),
                "reference_layout_mp4_frame_count": 16,
                "verified_h264_codecs": {
                    "generate_videos": "libx264",
                    "gt_videos": "libx264",
                },
            }
            for sensor_id in (0, 1)
        ],
    }


def test_set_evaluation_seed_replays_python_numpy_torch_and_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_seeds: list[int] = []
    monkeypatch.setenv("PYTHONHASHSEED", "20260801")
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seeds.append)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled: None)

    first_contract = set_evaluation_seed(20260801)
    first = (random.random(), float(np.random.rand()), torch.rand(3))
    second_contract = set_evaluation_seed(20260801)
    second = (random.random(), float(np.random.rand()), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0.0, atol=0.0)
    assert cuda_seeds and cuda_seeds[-1] == 20260801
    assert first_contract == second_contract
    assert first_contract["seed"] == 20260801


def test_deterministic_eval_loader_uses_sequential_order() -> None:
    dataset = TensorDataset(torch.tensor([4, 2, 9]))
    loader = build_deterministic_eval_loader(dataset)

    assert isinstance(loader.sampler, SequentialSampler)
    assert [int(batch[0].item()) for batch in loader] == [4, 2, 9]


def test_qpos8_dataset_index_metadata_propagates_real_task() -> None:
    source = type(
        "SourceDataset",
        (),
        {
            "repo_id": "univtac_validation",
            "new_metas": [
                {
                    "episode_index": 5,
                    "start_frame": 0,
                    "end_frame": 20,
                    "tasks": ["lift_bottle"],
                }
            ],
        },
    )()
    dataset = type(
        "Qpos8Dataset",
        (),
        {"_datasets": [source], "_offsets": [0]},
    )()

    metadata = resolve_dataset_index_metadata(dataset, 0)

    assert metadata["episode_index"] == 5
    assert metadata["tasks"] == ["lift_bottle"]


def test_dataset_audit_fails_closed_without_latent_inventories(tmp_path: Path) -> None:
    dataset = tmp_path / "validation"
    dataset.mkdir()
    manifest_sha = "a" * 64
    normalizer_sha = "b" * 64
    manifest = tmp_path / "manifest.json"
    conversion = tmp_path / "conversion.json"
    normalizer = tmp_path / "normalizer.json"
    manifest.write_text(json.dumps({"manifest_sha256": manifest_sha}))
    conversion.write_text(json.dumps({"source_manifest_sha256": manifest_sha}))
    normalizer.write_text(json.dumps({"normalizer_sha256": normalizer_sha}))

    with pytest.raises(FileNotFoundError, match="latent_video_inventory"):
        audit_evaluation_dataset(
            dataset_path=dataset,
            manifest_path=manifest,
            conversion_report_path=conversion,
            normalizer_path=normalizer,
            evaluation_view_path=manifest,
            normalizer_source_view_path=manifest,
            expected_manifest_sha256=manifest_sha,
            expected_normalizer_sha256=normalizer_sha,
            base_model_path=tmp_path,
        )


def test_dataset_audit_revalidates_current_latents_and_records_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    frozen40 = dataset_root / "frozen40"
    frozen40.mkdir(parents=True)
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    manifest_sha = "a" * 64
    normalizer_sha = "b" * 64
    conversion_sha = "c" * 64
    manifest = tmp_path / "manifest.json"
    conversion = tmp_path / "conversion.json"
    normalizer = tmp_path / "normalizer.json"
    evaluation_view = tmp_path / "evaluation_view.json"
    normalizer_source_view = tmp_path / "normalizer_source_view.json"
    manifest.write_text(json.dumps({"manifest_sha256": manifest_sha}))
    conversion.write_text(
        json.dumps(
            {
                "source_manifest_sha256": manifest_sha,
                "conversion_report_sha256": conversion_sha,
            }
        )
    )
    normalizer.write_text(json.dumps({"normalizer_sha256": normalizer_sha}))
    evaluation_view.write_text("{}", encoding="utf-8")
    normalizer_source_view.write_text("{}", encoding="utf-8")
    inventory = {
        "split": "frozen40",
        "manifest_sha256": manifest_sha,
        "segments": [],
        "inventory_sha256": "e" * 64,
    }
    for name in ("latent_video_inventory.json", "latent_tactile_inventory.json"):
        (frozen40 / name).write_text(json.dumps(inventory))
    verified = SimpleNamespace(
        manifest_sha256=manifest_sha,
        normalizer_sha256=normalizer_sha,
        conversion_report_sha256=conversion_sha,
        evaluation_view_id="frozen_target10_v1",
        evaluation_view_sha256="3" * 64,
        evaluation_episode_count=10,
        normalizer_source_view_id="stage_a_final759_v1",
        normalizer_source_view_sha256="4" * 64,
    )
    encoder_identity = {
        "schema_version": 1,
        "files": [{"relative_path": "vae/config.json"}],
        "identity_sha256": "f" * 64,
    }
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "n0_twam.evaluation.tactile_provenance.verify_track31_evaluation_bundle",
        lambda **kwargs: calls.setdefault("verify", kwargs) and verified,
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.tactile_provenance.build_encoder_source_identity",
        lambda path: calls.setdefault("encoder_path", path) and encoder_identity,
    )

    def _validate_pair(path: Path, **kwargs: object) -> dict[str, object]:
        calls["inventory_path"] = path
        calls["inventory_kwargs"] = kwargs
        return {
            "video_inventory_sha256": "1" * 64,
            "tactile_inventory_sha256": "2" * 64,
            "segment_count": 0,
            "video_artifact_count": 0,
            "tactile_artifact_count": 0,
        }

    monkeypatch.setattr(
        "n0_twam.evaluation.tactile_provenance.validate_latent_inventory_pair",
        _validate_pair,
    )

    result = audit_evaluation_dataset(
        dataset_path=frozen40,
        manifest_path=manifest,
        conversion_report_path=conversion,
        normalizer_path=normalizer,
        evaluation_view_path=evaluation_view,
        normalizer_source_view_path=normalizer_source_view,
        expected_manifest_sha256=manifest_sha,
        expected_normalizer_sha256=normalizer_sha,
        base_model_path=base_model,
    )

    assert calls["verify"]["dataset_root"] == dataset_root.resolve()
    assert calls["inventory_path"] == frozen40.resolve()
    assert calls["inventory_kwargs"] == {
        "expected_split": "frozen40",
        "expected_manifest_sha256": manifest_sha,
        "expected_conversion_report_sha256": conversion_sha,
        "expected_encoder_source_identity": encoder_identity,
    }
    assert result["conversion_report_sha256"] == conversion_sha
    assert result["evaluation_view_id"] == "frozen_target10_v1"
    assert result["encoder_source_identity"] == encoder_identity
    assert result["latent_inventory_validation"]["segment_count"] == 0
    assert result["physical_split"] == "frozen40"


def test_capture_source_metadata_records_values_and_explicit_nulls() -> None:
    batch = {
        "episode_index": torch.tensor([17]),
        "tasks": ["lift_bottle"],
        "tactile_sensor_ids": torch.tensor([[0, 1]]),
        "latents": torch.zeros(1, 48, 5, 16, 32),
    }

    selection = capture_source_sample_metadata(
        batch,
        output_sample_id="sample_000000",
        dataset_index=0,
        loader_cycle=0,
    )

    assert selection["source_metadata"]["episode_index"] == 17
    assert selection["source_metadata"]["tasks"] == ["lift_bottle"]
    assert selection["source_metadata"]["tactile_sensor_ids"] == [[0, 1]]
    assert selection["source_metadata"]["repo_id"] is None
    assert "repo_id" in selection["unavailable_source_metadata_keys"]
    assert selection["available_batch_keys"] == sorted(batch)
    assert selection["batch_tensor_contract"]["latents"] == {
        "shape": [1, 48, 5, 16, 32],
        "dtype": "torch.float32",
        "content_sha256": selection["batch_tensor_contract"]["latents"][
            "content_sha256"
        ],
    }
    assert len(selection["batch_tensor_contract"]["latents"]["content_sha256"]) == 64


def test_future_local_tactile_is_hidden_before_prepare_input() -> None:
    common = {
        "latents": torch.zeros(1, 48, 3, 2, 2),
        "tactile_global_latent": torch.zeros(1, 2, 48, 3, 2, 2),
        "tactile_sensor_ids": torch.tensor([[0, 1]]),
    }
    local_a = torch.arange(1 * 2 * 48 * 3 * 2 * 2).reshape(1, 2, 48, 3, 2, 2)
    local_b = local_a.clone()
    local_b[:, :, :, 1:] += 10000

    conditioned_a = mask_future_local_tactile(
        {**common, "tactile_local_latent": local_a}
    )
    conditioned_b = mask_future_local_tactile(
        {**common, "tactile_local_latent": local_b}
    )

    def _prepare_input(value: dict[str, object]) -> torch.Tensor:
        local = value["tactile_local_latent"]
        assert isinstance(local, torch.Tensor)
        return local.clone()

    prepared_a = _prepare_input(conditioned_a)
    prepared_b = _prepare_input(conditioned_b)

    torch.testing.assert_close(prepared_a, prepared_b)
    expected = local_a[:, :, :, 0:1].expand_as(local_a)
    torch.testing.assert_close(conditioned_a["tactile_local_latent"], expected)
    assert conditioned_a["tactile_local_latent"] is not local_a


def test_future_local_tactile_mask_supports_unbatched_5d_layout() -> None:
    global_tactile = torch.zeros(2, 48, 3, 2, 2)
    local_tactile = torch.arange(2 * 48 * 3 * 2 * 2).reshape(2, 48, 3, 2, 2)

    conditioned = mask_future_local_tactile(
        {
            "tactile_global_latent": global_tactile,
            "tactile_local_latent": local_tactile,
            "tactile_sensor_ids": torch.tensor([0, 1]),
        }
    )

    expected = local_tactile[:, :, 0:1].expand_as(local_tactile)
    torch.testing.assert_close(conditioned["tactile_local_latent"], expected)


@pytest.mark.parametrize(
    "missing_key",
    [
        "tactile_global_latent",
        "tactile_local_latent",
        "tactile_sensor_ids",
    ],
)
def test_offline_tactile_mask_rejects_missing_required_input(
    missing_key: str,
) -> None:
    batch = {
        "tactile_global_latent": torch.zeros(1, 2, 48, 3, 2, 2),
        "tactile_local_latent": torch.zeros(1, 2, 48, 3, 2, 2),
        "tactile_sensor_ids": torch.tensor([[0, 1]]),
    }
    batch.pop(missing_key)

    with pytest.raises(ValueError, match=missing_key):
        mask_future_local_tactile(batch)


def test_raw_video_context_replaces_prepare_input_noise_and_timesteps() -> None:
    raw = torch.arange(1 * 48 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 48, 3, 2, 2)
    prepared = {
        "latent_dict": {
            "latent": torch.randn_like(raw),
            "noisy_latents": torch.randn_like(raw),
            "timesteps": torch.ones(1, 3, dtype=torch.int64),
            "cond_timesteps": torch.ones(1, 3, dtype=torch.int64),
        },
        "action_dict": {},
    }

    restore_raw_video_context({"latents": raw}, prepared, clean_timestep=7)

    torch.testing.assert_close(prepared["latent_dict"]["latent"], raw)
    torch.testing.assert_close(prepared["latent_dict"]["noisy_latents"], raw)
    assert prepared["latent_dict"]["timesteps"].tolist() == [[7, 7, 7]]
    assert prepared["latent_dict"]["cond_timesteps"].tolist() == [[7, 7, 7]]


def test_checkpoint_audit_binds_config_and_transformer_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoint_step_10"
    transformer_dir = checkpoint_root / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "is_mot": True,
                "action_dim": 8,
                "action_schema": "qpos8_next_step",
                **_model_contract(),
                "num_attention_heads": 24,
                "attention_head_dim": 128,
                "rope_max_seq_len": 1024,
            }
        ),
        encoding="utf-8",
    )
    train_meta = {
        **_model_contract(),
        "active_tactile_sensor_count": 2,
        "active_tactile_sensor_ids": [0, 1],
        "tactile_sensor_id_map": {
            "observation.images.tactile_a": 0,
            "observation.images.tactile_b": 1,
        },
        "track31_artifacts": _track31_artifacts(),
    }
    (checkpoint_root / "train_meta.json").write_text(
        json.dumps(train_meta), encoding="utf-8"
    )
    weights_path = transformer_dir / TRANSFORMER_WEIGHTS_FILENAME
    weights_path.write_bytes(b"placeholder")
    expected_identity = _transformer_identity()

    monkeypatch.setattr(
        "n0_twam.evaluation.tactile_checkpoint.audit_transformer_checkpoint",
        lambda path, expected_action_dim: expected_identity,
    )
    monkeypatch.setattr(
        "n0_twam.evaluation.tactile_checkpoint.audit_tactile_head_contract",
        lambda path, model_contract, config_payload: _tactile_head_contract(),
    )
    result = audit_evaluation_checkpoint(
        checkpoint_root,
        expected_action_dim=8,
        expected_action_schema="qpos8_next_step",
        expected_model_contract=_model_contract(),
    )

    assert result["transformer_directory"] == str(transformer_dir.resolve())
    assert result["transformer_identity"] == expected_identity
    assert result["transformer_sha256"] == "a" * 64
    assert result["model_contract"] == _model_contract()
    assert len(result["training_metadata"]["sha256"]) == 64
    assert result["training_metadata"]["track31_artifacts"] == _track31_artifacts()
    assert len(result["config_sha256"]) == 64

    train_meta.pop("snr_shift")
    (checkpoint_root / "train_meta.json").write_text(
        json.dumps(train_meta), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metadata mismatch for snr_shift"):
        audit_evaluation_checkpoint(
            checkpoint_root,
            expected_action_dim=8,
            expected_action_schema="qpos8_next_step",
            expected_model_contract=_model_contract(),
        )


def test_checkpoint_dataset_binding_requires_exact_training_artifacts() -> None:
    checkpoint = _checkpoint_provenance()
    dataset = {
        "source_manifest_sha256": "1" * 64,
        "normalizer_sha256": "2" * 64,
        "conversion_report_sha256": "3" * 64,
        "normalizer_source_view_id": "stage_a_final759_v1",
        "normalizer_source_view_sha256": "4" * 64,
    }

    result = validate_checkpoint_dataset_binding(checkpoint, dataset)

    assert result["train_view_id"] == "stage_a_final759_v1"
    assert result["manifest_sha256"] == "1" * 64

    dataset["normalizer_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="normalizer_sha256"):
        validate_checkpoint_dataset_binding(checkpoint, dataset)


def test_checkpoint_dataset_binding_rejects_legacy_checkpoint() -> None:
    checkpoint = _checkpoint_provenance()
    checkpoint["training_metadata"]["track31_artifacts"] = None

    with pytest.raises(ValueError, match="no Track 3.1 artifact identity"):
        validate_checkpoint_dataset_binding(
            checkpoint,
            {
                "source_manifest_sha256": "1" * 64,
                "normalizer_sha256": "2" * 64,
                "conversion_report_sha256": "3" * 64,
                "normalizer_source_view_id": "stage_a_final759_v1",
                "normalizer_source_view_sha256": "4" * 64,
            },
        )


def test_provenance_report_binds_checkpoint_sampling_context_and_samples() -> None:
    checkpoint = _checkpoint_provenance()
    samples = [_sample_selection()]

    report = build_provenance_bound_report(
        _metric_report(),
        checkpoint=checkpoint,
        dataset={"split": "validation"},
        decoder={"sha256": "d" * 64},
        runtime={"device": "cuda:0"},
        rng_contract={"seed": 20260801},
        seed=20260801,
        n_steps=8,
        requested_n_gen=1,
        max_latent_frames=5,
        expected_active_tactile_sensor_count=2,
        metric_split="validation",
        sample_selection=samples,
    )

    assert report["schema_version"] == 2
    provenance = report["provenance"]
    assert provenance["checkpoint"]["transformer_sha256"] == "a" * 64
    assert provenance["sampling"]["seed"] == 20260801
    assert provenance["sampling"]["generated_sample_count"] == 1
    assert provenance["metric_output_contract"] == {
        "active_tactile_sensor_count": 2,
        "active_tactile_sensor_ids": [0, 1],
        "model_tactile_stream_capacity": 4,
        "expected_video_count": 2,
        "expected_scored_frame_count": 32,
    }
    assert provenance["context_contract"] == {
        "name": "offline_conditional_tactile_prediction",
        "protocol": "internal_offline",
        "closed_loop": False,
        "ground_truth_video_context": True,
        "ground_truth_action_context": True,
        "global_tactile_observed_frame_count": 1,
        "local_tactile_observed_frame_count": 1,
        "future_global_tactile_ground_truth_visible_to_predictor": False,
        "future_local_tactile_ground_truth_visible_to_predictor": False,
    }
    assert provenance["metric_domain"] == TACTILE_METRIC_DOMAIN
    assert len(provenance["provenance_sha256"]) == 64
    assert len(report["metric_payload_sha256"]) == 64
    assert len(report["evaluation_identity_sha256"]) == 64


def test_provenance_report_rejects_closed_loop_or_wrong_domain() -> None:
    report = _metric_report()
    report["protocol"] = "closed_loop_observed"
    with pytest.raises(ValueError, match="internal_offline"):
        build_provenance_bound_report(
            report,
            checkpoint={},
            dataset={},
            decoder={},
            runtime={},
            rng_contract={},
            seed=1,
            n_steps=1,
            requested_n_gen=1,
            max_latent_frames=1,
            expected_active_tactile_sensor_count=2,
            metric_split="validation",
            sample_selection=[],
        )

    report = _metric_report()
    report["pixel_domain"] = "raw_rgb"
    with pytest.raises(ValueError, match="metric domain"):
        build_provenance_bound_report(
            report,
            checkpoint={},
            dataset={},
            decoder={},
            runtime={},
            rng_contract={},
            seed=1,
            n_steps=1,
            requested_n_gen=1,
            max_latent_frames=1,
            expected_active_tactile_sensor_count=2,
            metric_split="validation",
            sample_selection=[],
        )


@pytest.mark.parametrize("requested", [1, 3])
def test_provenance_report_requires_exact_requested_sample_count(
    requested: int,
) -> None:
    checkpoint = _checkpoint_provenance()
    samples = [_sample_selection(), _sample_selection()]

    with pytest.raises(ValueError, match="must equal requested_n_gen"):
        build_provenance_bound_report(
            _metric_report(),
            checkpoint=checkpoint,
            dataset={},
            decoder={},
            runtime={},
            rng_contract={},
            seed=1,
            n_steps=1,
            requested_n_gen=requested,
            max_latent_frames=1,
            expected_active_tactile_sensor_count=2,
            metric_split="validation",
            sample_selection=samples,
        )


def test_provenance_report_cross_checks_metric_videos_and_sensors() -> None:
    metric = _metric_report()
    metric["per_video"][0]["sensor"] = "sensor_2"
    metric["per_video"][0]["video_id"] = "lift_bottle/sample_000000/sensor_2"

    with pytest.raises(ValueError, match="metric video set"):
        build_provenance_bound_report(
            metric,
            checkpoint=_checkpoint_provenance(),
            dataset={},
            decoder={},
            runtime={},
            rng_contract={},
            seed=1,
            n_steps=1,
            requested_n_gen=1,
            max_latent_frames=5,
            expected_active_tactile_sensor_count=2,
            metric_split="validation",
            sample_selection=[_sample_selection()],
        )


def test_provenance_rejects_sensor_ids_outside_checkpoint_active_set() -> None:
    sample = _sample_selection()
    sample["source_metadata"]["tactile_sensor_ids"] = [[0, 4]]

    with pytest.raises(ValueError, match="differ from checkpoint metadata"):
        build_provenance_bound_report(
            _metric_report(),
            checkpoint=_checkpoint_provenance(),
            dataset={},
            decoder={},
            runtime={},
            rng_contract={},
            seed=1,
            n_steps=1,
            requested_n_gen=1,
            max_latent_frames=5,
            expected_active_tactile_sensor_count=2,
            metric_split="validation",
            sample_selection=[sample],
        )


def test_provenance_rejects_checkpoint_sensor_id_above_capacity() -> None:
    checkpoint = _checkpoint_provenance()
    metadata = checkpoint["training_metadata"]
    metadata["active_tactile_sensor_ids"] = [0, 4]
    metadata["tactile_sensor_id_map"] = {"tactile_a": 0, "tactile_b": 4}

    with pytest.raises(ValueError, match="sensor provenance is invalid"):
        build_provenance_bound_report(
            _metric_report(),
            checkpoint=checkpoint,
            dataset={},
            decoder={},
            runtime={},
            rng_contract={},
            seed=1,
            n_steps=1,
            requested_n_gen=1,
            max_latent_frames=5,
            expected_active_tactile_sensor_count=2,
            metric_split="validation",
            sample_selection=[_sample_selection()],
        )


def test_provenance_rejects_non_integer_sensor_map_value() -> None:
    checkpoint = _checkpoint_provenance()
    checkpoint["training_metadata"]["tactile_sensor_id_map"] = {
        "tactile_a": 0,
        "tactile_b": "1",
    }

    with pytest.raises(ValueError, match="sensor ID"):
        build_provenance_bound_report(
            _metric_report(),
            checkpoint=checkpoint,
            dataset={},
            decoder={},
            runtime={},
            rng_contract={},
            seed=1,
            n_steps=1,
            requested_n_gen=1,
            max_latent_frames=5,
            expected_active_tactile_sensor_count=2,
            metric_split="validation",
            sample_selection=[_sample_selection()],
        )


def test_atomic_json_write_publishes_complete_payload(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "tactile_quality.json"
    payload = {"schema_version": 2, "evaluation_identity_sha256": "d" * 64}

    write_json_atomic(output, payload)

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []
