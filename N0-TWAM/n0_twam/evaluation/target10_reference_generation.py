# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Generate a sealed nine-slot Target-10 prediction from causal inputs only."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch

from n0_twam.evaluation.tactile_generation_runtime import (
    canonical_sha256,
    load_legacy_runtime,
    validate_device_argument,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="track31_univtac")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--causal-input-bundle", type=Path, required=True)
    parser.add_argument("--empty-embedding", type=Path, required=True)
    parser.add_argument("--empty-embedding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--conversion-report-sha256", required=True)
    parser.add_argument("--strict-checkpoint-identity-sha256", required=True)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _sample_ordinal(sample_id: str) -> int:
    """Map a stable sample identity onto the existing seed-contract ordinal."""

    return int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big")


def select_target10_reference_grid(
    continuous: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Select raw-time-equivalent rows 0,5,...,40 from a 41-frame decode."""

    from n0_twam.evaluation.target10_reference_contract import REFERENCE_CONTRACT

    expected = (
        2,
        REFERENCE_CONTRACT.model_decoded_frame_count,
        128,
        128,
        3,
    )
    value = np.asarray(continuous)
    if value.dtype != np.float32 or value.shape != expected:
        raise ValueError(f"continuous tactile residual must be float32 {expected}")
    if not np.isfinite(value).all():
        raise ValueError("continuous tactile residual contains non-finite values")
    selected = np.ascontiguousarray(
        value[:, REFERENCE_CONTRACT.selected_output_indices],
        dtype=np.float32,
    )
    return selected


def _metadata_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"condition metadata {label} must be non-negative integer")
    return value


def _evaluation_runtime_paths(
    dataset_provenance: dict[str, object],
) -> tuple[Path, Path]:
    """Recover sealed evaluation paths without consulting training defaults."""

    artifact_files = dataset_provenance.get("artifact_files")
    if not isinstance(artifact_files, dict):
        raise ValueError("causal dataset provenance lacks artifact_files")
    evaluation_view = artifact_files.get("evaluation_view")
    if not isinstance(evaluation_view, dict):
        raise ValueError("causal dataset provenance lacks evaluation_view artifact")
    evaluation_view_path = evaluation_view.get("path")
    dataset_path = dataset_provenance.get("dataset_path")
    if not isinstance(evaluation_view_path, str) or not evaluation_view_path:
        raise ValueError("causal evaluation_view path is invalid")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ValueError("causal evaluation dataset path is invalid")
    return (
        Path(evaluation_view_path).resolve(strict=True),
        Path(dataset_path).resolve(strict=True),
    )


def generate_target10_reference_prediction(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Run HCU inference without raw HDF5 or future-GT tensors mounted."""

    from n0_twam.configs import TWAM_CONFIGS
    from n0_twam.checkpointing.strict_checkpoint_snapshot import (
        build_strict_checkpoint_identity,
        capture_strict_checkpoint_snapshot,
    )
    from n0_twam.evaluation.fair_protocol import CAUSAL_FUTURE_ONLY_PROTOCOL
    from n0_twam.evaluation.render_helpers import decode_signed_video_latent
    from n0_twam.evaluation.sealed_artifact_io import validate_sha256
    from n0_twam.evaluation.tactile_checkpoint import audit_evaluation_checkpoint
    from n0_twam.evaluation.tactile_checkpoint import (
        validate_checkpoint_dataset_binding,
    )
    from n0_twam.evaluation.tactile_model_contract import (
        audit_loaded_tactile_runtime,
        build_expected_tactile_model_contract,
    )
    from n0_twam.evaluation.tactile_provenance import (
        audit_vae_decoder,
        set_evaluation_seed,
        set_sample_seed,
    )
    from n0_twam.evaluation.tactile_sampling import (
        sample_tactile_causal_future_only,
    )
    from n0_twam.evaluation.target10_causal_input import (
        _empty_embedding_identity,
        verify_target10_causal_input_bundle,
    )
    from n0_twam.evaluation.target10_prediction_artifact_v3 import (
        RESIDUAL_SHAPE,
        Target10PredictionSample,
        write_target10_prediction_artifact,
    )
    from n0_twam.evaluation.target10_reference_contract import REFERENCE_CONTRACT

    if args.config_name != "track31_univtac":
        raise ValueError("reference9 generation requires track31_univtac")
    if (
        args.n_steps != REFERENCE_CONTRACT.sampling_steps
        or args.seed != REFERENCE_CONTRACT.evaluation_seed
    ):
        raise ValueError("reference9 generation fixes n-steps=50 and seed=2026")
    device = validate_device_argument(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("reference9 generation requires an available HCU/GPU")
    manifest_sha = validate_sha256(
        args.source_manifest_sha256, label="source manifest SHA256"
    )
    conversion_sha = validate_sha256(
        args.conversion_report_sha256, label="conversion report SHA256"
    )
    expected_strict_checkpoint_sha = validate_sha256(
        args.strict_checkpoint_identity_sha256,
        label="strict checkpoint identity SHA256",
    )
    strict_checkpoint_identity = build_strict_checkpoint_identity(
        capture_strict_checkpoint_snapshot(args.ckpt)
    )
    if strict_checkpoint_identity["identity_sha256"] != expected_strict_checkpoint_sha:
        raise ValueError("strict checkpoint identity differs from evaluation preflight")
    causal = verify_target10_causal_input_bundle(args.causal_input_bundle)
    dataset_provenance = causal.metadata.get("dataset_provenance")
    if not isinstance(dataset_provenance, dict):
        raise ValueError("causal input bundle lacks audited dataset provenance")
    if (
        dataset_provenance.get("source_manifest_sha256") != manifest_sha
        or dataset_provenance.get("conversion_report_sha256") != conversion_sha
    ):
        raise ValueError("requested dataset identities differ from causal input bundle")
    empty_embedding = _empty_embedding_identity(
        args.empty_embedding,
        expected_sha256=args.empty_embedding_sha256,
    )
    if dataset_provenance.get("empty_embedding") != empty_embedding:
        raise ValueError("requested empty embedding differs from causal input bundle")
    evaluation_view_path, evaluation_dataset_path = _evaluation_runtime_paths(
        dataset_provenance
    )

    config = deepcopy(TWAM_CONFIGS[args.config_name])
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.enable_wandb = False
    config.load_worker = 0
    config.max_latent_frames = REFERENCE_CONTRACT.model_latent_frames
    config.tactile_cfg_prob = 0.0
    config.noisy_cond_prob_tactile = 0.0
    config.empty_emb_path = str(empty_embedding["path"])
    config.raw_tactile_evaluation = True
    config.dataset_path = str(evaluation_dataset_path)
    config.dataset_view_path = str(evaluation_view_path)
    config.val_dataset_path = None
    config.val_dataset_view_path = None
    seed_contract = set_evaluation_seed(args.seed)

    model_contract = build_expected_tactile_model_contract(config)
    checkpoint = audit_evaluation_checkpoint(
        args.ckpt,
        expected_action_dim=int(config.action_dim),
        expected_action_schema=str(config.action_schema),
        expected_model_contract=model_contract,
    )
    checkpoint_dataset_binding = validate_checkpoint_dataset_binding(
        checkpoint,
        dataset_provenance,
    )
    decoder = audit_vae_decoder(args.vae)
    Trainer, load_mot_checkpoint, load_vae = load_legacy_runtime()
    trainer = Trainer(config, inference_only=True)
    trainer.transformer = load_mot_checkpoint(
        checkpoint["transformer_directory"],
        torch_dtype=torch.bfloat16,
        torch_device=device,
    )
    trainer.transformer.eval()
    runtime_contract = audit_loaded_tactile_runtime(trainer, model_contract)
    vae = load_vae(args.vae, torch_dtype=torch.bfloat16, torch_device=device)
    if audit_vae_decoder(args.vae) != decoder:
        raise RuntimeError("VAE bytes changed while reference9 generation loaded")

    samples = []
    sample_seeds = []
    for condition in causal.samples:
        sample_id = str(condition.metadata["sample_id"])
        seed = set_sample_seed(args.seed, _sample_ordinal(sample_id))
        converted = trainer.convert_input_format(
            {key: value.clone() for key, value in condition.tensors.items()}
        )
        generated, _ = sample_tactile_causal_future_only(
            trainer, converted, n_steps=args.n_steps
        )
        residuals = []
        for sensor_index in range(2):
            decoded = decode_signed_video_latent(
                vae, generated[:, sensor_index].to(device)
            )
            residual = decoded[0].permute(1, 2, 3, 0).numpy()
            residuals.append(np.ascontiguousarray(residual, dtype=np.float32))
        continuous = np.stack(residuals)
        expected_continuous_shape = (
            2,
            REFERENCE_CONTRACT.model_decoded_frame_count,
            128,
            128,
            3,
        )
        if continuous.shape != expected_continuous_shape:
            raise RuntimeError(
                "reference9 continuous decode produced "
                f"{continuous.shape}, expected {expected_continuous_shape}"
            )
        stacked = select_target10_reference_grid(continuous)
        if stacked.shape != RESIDUAL_SHAPE:
            raise RuntimeError("reference9 stride-5 selection returned invalid shape")
        sample_seeds.append({"sample_id": sample_id, "seed": seed})
        samples.append(
            Target10PredictionSample(
                sample_id=sample_id,
                dataset_index=_metadata_integer(
                    condition.metadata["dataset_index"], label="dataset_index"
                ),
                episode_index=_metadata_integer(
                    condition.metadata["episode_index"], label="episode_index"
                ),
                task=str(condition.metadata["task"]),
                source_relative_path=str(condition.metadata["source_relative_path"]),
                signed_residual=stacked,
            )
        )
    final_strict_checkpoint_identity = build_strict_checkpoint_identity(
        capture_strict_checkpoint_snapshot(args.ckpt)
    )
    if (
        audit_evaluation_checkpoint(
            args.ckpt,
            expected_action_dim=int(config.action_dim),
            expected_action_schema=str(config.action_schema),
            expected_model_contract=model_contract,
        )
        != checkpoint
        or final_strict_checkpoint_identity != strict_checkpoint_identity
        or audit_vae_decoder(args.vae) != decoder
    ):
        raise RuntimeError("checkpoint or VAE bytes changed during generation")
    decoder_identity = canonical_sha256(decoder)
    output = write_target10_prediction_artifact(
        args.output,
        samples=samples,
        source_manifest_sha256=manifest_sha,
        conversion_report_sha256=conversion_sha,
        checkpoint_sha256=str(checkpoint["transformer_sha256"]),
        decoder_identity_sha256=decoder_identity,
        evaluation_view_id=str(causal.metadata["evaluation_view_id"]),
        evaluation_view_sha256=str(causal.metadata["evaluation_view_sha256"]),
        generation_provenance={
            "config_name": args.config_name,
            "seed_contract": seed_contract,
            "sample_seeds": sample_seeds,
            "n_steps": args.n_steps,
            "temporal_binding": REFERENCE_CONTRACT.temporal_binding,
            "model_latent_frames": REFERENCE_CONTRACT.model_latent_frames,
            "model_decoded_frame_count": (REFERENCE_CONTRACT.model_decoded_frame_count),
            "selected_output_indices": list(REFERENCE_CONTRACT.selected_output_indices),
            "decoded_frame_count": REFERENCE_CONTRACT.decoded_frame_count,
            "sequence_length_status": REFERENCE_CONTRACT.sequence_length_status,
            "conditioning_protocol_id": CAUSAL_FUTURE_ONLY_PROTOCOL,
            "conditioning_bundle": {
                "seal_sha256": causal.seal_sha256,
                "file_sha256": causal.file_sha256,
            },
            "future_gt_visible_to_model": False,
            "future_gt_embedded": False,
            "runtime_contract": runtime_contract,
            "checkpoint_provenance": checkpoint,
            "strict_checkpoint_identity": strict_checkpoint_identity,
            "checkpoint_dataset_binding": checkpoint_dataset_binding,
            "dataset_provenance": dataset_provenance,
            "empty_embedding": empty_embedding,
            "decoder_provenance": decoder,
        },
    )
    if (
        build_strict_checkpoint_identity(capture_strict_checkpoint_snapshot(args.ckpt))
        != strict_checkpoint_identity
        or audit_vae_decoder(args.vae) != decoder
    ):
        raise RuntimeError("checkpoint or VAE bytes changed while publishing artifact")
    return {
        "output": str(output.resolve(strict=True)),
        "sample_count": len(samples),
        "checkpoint_sha256": checkpoint["transformer_sha256"],
        "decoder_identity_sha256": decoder_identity,
        "contract_id": REFERENCE_CONTRACT.contract_id,
        "contract_sha256": REFERENCE_CONTRACT.sha256,
        "causal_input_seal_sha256": causal.seal_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = generate_target10_reference_prediction(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
