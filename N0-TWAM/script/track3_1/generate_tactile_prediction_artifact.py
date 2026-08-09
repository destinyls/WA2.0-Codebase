#!/usr/bin/env python3
# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Stage A: generate a sealed raw-tactile prediction artifact from LeRobot only."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence, cast

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    from n0_twam.integrations.univtac.dataset_view import DatasetView

REPO_ROOT = Path(__file__).resolve().parents[2]
N0_PACKAGE_ROOT = REPO_ROOT / "n0_twam"
for import_root in (REPO_ROOT, N0_PACKAGE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


class _TransformerRuntime(Protocol):
    """Transformer surface required by the sealed artifact generator."""

    def eval(self) -> object: ...


class _TrainerRuntime(Protocol):
    """Legacy trainer surface used by Stage-A inference."""

    transformer: _TransformerRuntime

    def convert_input_format(
        self, input_dict: dict[str, object]
    ) -> dict[str, object]: ...


class _TrainerFactory(Protocol):
    def __call__(
        self, config: object, *, inference_only: bool = False
    ) -> _TrainerRuntime: ...


class _LoadMotCheckpoint(Protocol):
    def __call__(
        self,
        checkpoint_dir: object,
        *,
        torch_dtype: torch.dtype,
        torch_device: str,
    ) -> _TransformerRuntime: ...


class _LoadVae(Protocol):
    def __call__(
        self,
        vae_path: object,
        *,
        torch_dtype: torch.dtype,
        torch_device: str,
    ) -> object: ...


def _load_legacy_runtime() -> tuple[
    _TrainerFactory,
    _LoadMotCheckpoint,
    _LoadVae,
]:
    """Type the legacy top-level imports without changing their runtime identity."""

    model_utils = importlib.import_module("models.utils")
    train_module = importlib.import_module("train")
    return (
        cast(_TrainerFactory, getattr(train_module, "Trainer")),
        cast(
            _LoadMotCheckpoint,
            getattr(model_utils, "load_mot_checkpoint"),
        ),
        cast(_LoadVae, getattr(model_utils, "load_vae")),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="track31_univtac")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation-view-manifest",
        type=Path,
        required=True,
        help=(
            "Sealed default frozen_target10_v1 DatasetView. Diagnostic views "
            "require --allow-diagnostic-view."
        ),
    )
    parser.add_argument(
        "--allow-diagnostic-view",
        action="store_true",
        help=(
            "Explicitly opt in to frozen_other30_v1 diagnostics; never use "
            "this flag for the unified leaderboard-oriented report."
        ),
    )
    parser.add_argument(
        "--protocol",
        default="causal_future_only_v1",
        choices=("causal_future_only_v1",),
    )
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help=(
            "Logical device; only cuda:0 is supported. Select a physical HCU "
            "via HIP_VISIBLE_DEVICES."
        ),
    )
    return parser


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Stage-A source JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Stage-A source JSON must contain an object: {path}")
    return payload


def _load_validation_source_entries(
    *, manifest_path: Path, conversion_report_path: Path
) -> dict[str, dict[str, object]]:
    manifest = _read_json(manifest_path)
    conversion = _read_json(conversion_report_path)
    entries = manifest.get("entries")
    conversions = conversion.get("conversions")
    if not isinstance(entries, list) or not isinstance(conversions, dict):
        raise ValueError("Stage-A manifest/conversion mapping is incomplete")
    validation_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("split") == "validation"
    ]
    expected_paths = [str(entry.get("relative_path")) for entry in validation_entries]
    expected_hashes = [str(entry.get("sha256")) for entry in validation_entries]
    candidates = [
        conversion_payload
        for conversion_payload in conversions.values()
        if isinstance(conversion_payload, dict)
        and conversion_payload.get("source_relative_paths") == expected_paths
        and conversion_payload.get("source_sha256") == expected_hashes
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Stage-A validation episodes must map to exactly one conversion repo"
        )
    converted_paths = candidates[0]["source_relative_paths"]
    by_path = {str(entry["relative_path"]): entry for entry in validation_entries}
    return {
        str(path): {
            **by_path[str(path)],
            "_lerobot_episode_index": episode_index,
        }
        for episode_index, path in enumerate(converted_paths)
    }


def _tensor_int_vector(
    value: object, *, label: str, expected_length: int = 17
) -> npt.NDArray[np.int64]:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Stage-A batch is missing {label}")
    from n0_twam.evaluation.tactile_prediction_schema import strict_int64_vector

    raw_array = value.detach().cpu().numpy().reshape(-1)
    result: npt.NDArray[np.int64] = strict_int64_vector(
        raw_array,
        label=f"Stage-A {label}",
        expected_length=expected_length,
    )
    return result


def _scalar_string(value: object, *, label: str) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Stage-A batch has an invalid {label}")
    return value


def _scalar_integer(value: object, *, label: str) -> int:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy().reshape(-1)
    else:
        array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Stage-A batch has an invalid {label}")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"Stage-A {label} must have an integer dtype")
    result = int(array[0])
    if result < 0:
        raise ValueError(f"Stage-A {label} must be non-negative")
    return result


def _validate_device_argument(device: object) -> str:
    resolved = str(device)
    if resolved != "cuda:0":
        raise ValueError(
            "Stage-A supports only logical cuda:0; select the physical HCU "
            "with HIP_VISIBLE_DEVICES"
        )
    return resolved


def _validate_sample_mapping(
    *,
    source_entries: Mapping[str, Mapping[str, object]],
    source_relative_path: str,
    task: str,
    lerobot_episode_index: int,
) -> None:
    entry = source_entries.get(source_relative_path)
    if entry is None:
        raise ValueError(
            "Stage-A LeRobot source path is absent from validation mapping"
        )
    if entry.get("task") != task or Path(source_relative_path).parts[0] != task:
        raise ValueError("Stage-A LeRobot task/source mapping differs from manifest")
    if entry.get("_lerobot_episode_index") != lerobot_episode_index:
        raise ValueError(
            "Stage-A LeRobot episode/source mapping differs from conversion"
        )


def _validate_evaluation_view(
    view: DatasetView,
    *,
    allow_diagnostic_view: bool = False,
) -> None:
    from n0_twam.integrations.univtac.dataset_view import (
        DEFAULT_UNIFIED_EVALUATION_EPISODE_COUNT,
        DEFAULT_UNIFIED_EVALUATION_VIEW_ID,
        DIAGNOSTIC_EVALUATION_VIEW_IDS,
    )

    view_id = getattr(view, "view_id", None)
    allowed_view_ids = {DEFAULT_UNIFIED_EVALUATION_VIEW_ID}
    if allow_diagnostic_view:
        allowed_view_ids.update(DIAGNOSTIC_EVALUATION_VIEW_IDS)
    if view_id not in allowed_view_ids:
        if view_id in DIAGNOSTIC_EVALUATION_VIEW_IDS:
            raise ValueError(
                "diagnostic evaluation view requires --allow-diagnostic-view"
            )
        raise ValueError(
            "prediction generation requires the default unified evaluation view"
        )
    if (
        getattr(view, "role", None) != "frozen_evaluation"
        or getattr(view, "physical_split", None) != "frozen40"
    ):
        raise ValueError("evaluation view must be a frozen40 evaluation cohort")
    expected_size = (
        DEFAULT_UNIFIED_EVALUATION_EPISODE_COUNT
        if view_id == DEFAULT_UNIFIED_EVALUATION_VIEW_ID
        else 30
    )
    if len(getattr(view, "entries", ())) != expected_size:
        raise ValueError("evaluation view has an invalid frozen cohort size")


def _sample_seed_ordinal(*, view_sha256: str, source_sha256: str) -> int:
    payload = f"{view_sha256}\0{source_sha256}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _content_addressed_sample_id(*, view_sha256: str, source_sha256: str) -> str:
    payload = f"{view_sha256}\0{source_sha256}".encode("ascii")
    return "sample_" + hashlib.sha256(payload).hexdigest()


def generate_tactile_prediction_artifact(args: argparse.Namespace) -> dict[str, object]:
    from n0_twam.configs import TWAM_CONFIGS
    from n0_twam.dataset import MultiLatentLeRobotDataset
    from n0_twam.evaluation.fair_protocol import CAUSAL_FUTURE_ONLY_PROTOCOL
    from n0_twam.evaluation.render_helpers import decode_signed_video_latent
    from n0_twam.evaluation.tactile_checkpoint import (
        audit_evaluation_checkpoint,
        validate_checkpoint_dataset_binding,
    )
    from n0_twam.evaluation.tactile_model_contract import (
        audit_loaded_tactile_runtime,
        build_expected_tactile_model_contract,
    )
    from n0_twam.evaluation.tactile_prediction_artifact import (
        TactilePredictionSample,
        write_tactile_prediction_artifact,
    )
    from n0_twam.evaluation.tactile_provenance import (
        audit_evaluation_dataset,
        audit_vae_decoder,
        build_deterministic_eval_loader,
        resolve_dataset_index_metadata,
        set_evaluation_seed,
        set_sample_seed,
    )
    from n0_twam.evaluation.tactile_sampling import (
        sample_tactile_causal_future_only,
    )
    from n0_twam.integrations.univtac.dataset_view import load_dataset_view

    if args.config_name != "track31_univtac":
        raise ValueError("raw tactile artifact generation requires track31_univtac")
    if args.n_steps <= 0 or args.seed < 0:
        raise ValueError("n-steps must be positive and seed non-negative")
    if args.protocol != CAUSAL_FUTURE_ONLY_PROTOCOL:
        raise ValueError("unsupported causal prediction protocol")
    device = _validate_device_argument(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-A model generation requires an available CUDA device")

    config = deepcopy(TWAM_CONFIGS[args.config_name])
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1
    config.enable_wandb = False
    config.load_worker = 0
    evaluation_view_path = Path(args.evaluation_view_manifest).resolve(strict=True)
    evaluation_view = load_dataset_view(evaluation_view_path)
    _validate_evaluation_view(
        evaluation_view,
        allow_diagnostic_view=bool(args.allow_diagnostic_view),
    )
    config.dataset_path = str(
        (Path(config.lerobot_root) / evaluation_view.physical_split).resolve(
            strict=True
        )
    )
    config.dataset_view_path = str(evaluation_view_path)
    config.train_view_id = evaluation_view.view_id
    config.validation_view_id = None
    config.val_dataset_path = None
    config.val_dataset_view_path = None
    config.raw_tactile_evaluation = True
    config.deterministic_evaluation_crop_zero = True
    config.max_latent_frames = 5
    config.tactile_cfg_prob = 0.0
    config.noisy_cond_prob_tactile = 0.0
    seed_contract = set_evaluation_seed(args.seed)

    model_contract = build_expected_tactile_model_contract(config)
    checkpoint = audit_evaluation_checkpoint(
        args.ckpt,
        expected_action_dim=int(config.action_dim),
        expected_action_schema=str(config.action_schema),
        expected_model_contract=model_contract,
    )
    dataset_provenance = audit_evaluation_dataset(
        dataset_path=Path(config.dataset_path),
        manifest_path=Path(config.dataset_manifest_path),
        conversion_report_path=Path(config.conversion_report_path),
        normalizer_path=Path(config.norm_stat_path),
        evaluation_view_path=evaluation_view_path,
        normalizer_source_view_path=Path(config.normalizer_source_view_path),
        expected_manifest_sha256=str(config.source_manifest_sha256),
        expected_normalizer_sha256=str(config.normalizer_sha256),
        base_model_path=Path(config.wan22_pretrained_model_name_or_path),
    )
    checkpoint_dataset_binding = validate_checkpoint_dataset_binding(
        checkpoint,
        dataset_provenance,
    )
    decoder = audit_vae_decoder(args.vae)
    source_entries = _load_validation_source_entries(
        manifest_path=Path(config.dataset_manifest_path),
        conversion_report_path=Path(config.conversion_report_path),
    )

    Trainer, load_mot_checkpoint, load_vae = _load_legacy_runtime()

    trainer = Trainer(config, inference_only=True)
    trainer.transformer = load_mot_checkpoint(
        checkpoint["transformer_directory"],
        torch_dtype=torch.bfloat16,
        torch_device=device,
    )
    trainer.transformer.eval()
    runtime_contract = audit_loaded_tactile_runtime(trainer, model_contract)
    vae = load_vae(
        args.vae,
        torch_dtype=torch.bfloat16,
        torch_device=device,
    )
    if audit_vae_decoder(args.vae) != decoder:
        raise RuntimeError("VAE decoder bytes changed while Stage A loaded")

    dataset = MultiLatentLeRobotDataset(config=config)
    if len(dataset) != len(evaluation_view.entries):
        raise ValueError("evaluation dataset does not exactly cover its frozen view")
    batches = iter(build_deterministic_eval_loader(dataset))
    samples = []
    sample_seeds = []
    view_entries_by_path = {
        entry.relative_path: entry for entry in evaluation_view.entries
    }
    for dataset_index in range(len(evaluation_view.entries)):
        batch = next(batches)
        source_row_ids = _tensor_int_vector(
            batch.get("source_row_ids"), label="source_row_ids"
        )
        source_step_ids = _tensor_int_vector(
            batch.get("source_step_ids"), label="source_step_ids"
        )
        source_relative_path = _scalar_string(
            batch.get("source_relative_path"), label="source_relative_path"
        )
        episode_index = _scalar_integer(
            batch.get("lerobot_episode_index"), label="lerobot_episode_index"
        )
        dataset_metadata = resolve_dataset_index_metadata(dataset, dataset_index)
        if (
            _scalar_integer(
                dataset_metadata.get("episode_index"), label="dataset episode_index"
            )
            != episode_index
        ):
            raise ValueError(
                "Stage-A batch episode differs from dataset-index metadata"
            )
        raw_tasks = dataset_metadata.get("tasks")
        if not isinstance(raw_tasks, list) or len(raw_tasks) != 1:
            raise ValueError("Stage-A dataset metadata must contain one task")
        task = str(raw_tasks[0])
        _validate_sample_mapping(
            source_entries=source_entries,
            source_relative_path=source_relative_path,
            task=task,
            lerobot_episode_index=episode_index,
        )
        view_entry = view_entries_by_path.get(source_relative_path)
        if view_entry is None:
            raise ValueError("Stage-A sample is outside the sealed evaluation view")
        sample_seed = set_sample_seed(
            args.seed,
            _sample_seed_ordinal(
                view_sha256=evaluation_view.view_sha256,
                source_sha256=view_entry.source_sha256,
            ),
        )
        sample_seeds.append(
            {
                "sample_id": _content_addressed_sample_id(
                    view_sha256=evaluation_view.view_sha256,
                    source_sha256=view_entry.source_sha256,
                ),
                "seed": sample_seed,
            }
        )
        active_sensor_ids = _tensor_int_vector(
            torch.as_tensor(batch["tactile_sensor_ids"]).reshape(-1),
            label="active_sensor_ids",
            expected_length=2,
        )
        if active_sensor_ids.shape != (2,) or active_sensor_ids.tolist() != [0, 1]:
            raise ValueError("Stage-A active tactile sensor IDs must be [0, 1]")
        model_batch = dict(batch)
        for metadata_key in (
            "source_relative_path",
            "source_row_ids",
            "source_step_ids",
            "lerobot_episode_index",
        ):
            model_batch.pop(metadata_key, None)
        converted_batch = trainer.convert_input_format(model_batch)
        generated, _ = sample_tactile_causal_future_only(
            trainer, converted_batch, n_steps=args.n_steps
        )
        residuals = []
        for sensor_index in range(2):
            decoded = decode_signed_video_latent(
                vae, generated[:, sensor_index].to(device)
            )
            residual = decoded[0].permute(1, 2, 3, 0).numpy()
            residuals.append(np.ascontiguousarray(residual, dtype=np.float32))
        samples.append(
            TactilePredictionSample(
                sample_id=str(sample_seeds[-1]["sample_id"]),
                dataset_index=dataset_index,
                lerobot_episode_index=episode_index,
                task=task,
                source_relative_path=source_relative_path,
                source_row_ids=source_row_ids,
                source_step_ids=source_step_ids,
                active_sensor_ids=active_sensor_ids,
                signed_residual=np.stack(residuals),
            )
        )

    if (
        audit_evaluation_checkpoint(
            args.ckpt,
            expected_action_dim=int(config.action_dim),
            expected_action_schema=str(config.action_schema),
            expected_model_contract=model_contract,
        )
        != checkpoint
        or audit_vae_decoder(args.vae) != decoder
        or audit_evaluation_dataset(
            dataset_path=Path(config.dataset_path),
            manifest_path=Path(config.dataset_manifest_path),
            conversion_report_path=Path(config.conversion_report_path),
            normalizer_path=Path(config.norm_stat_path),
            evaluation_view_path=evaluation_view_path,
            normalizer_source_view_path=Path(config.normalizer_source_view_path),
            expected_manifest_sha256=str(config.source_manifest_sha256),
            expected_normalizer_sha256=str(config.normalizer_sha256),
            base_model_path=Path(config.wan22_pretrained_model_name_or_path),
        )
        != dataset_provenance
    ):
        raise RuntimeError(
            "Stage-A model/decoder/dataset bytes changed during generation"
        )
    decoder_identity = _canonical_sha256(decoder)
    output = write_tactile_prediction_artifact(
        args.output,
        samples=samples,
        source_manifest_sha256=str(dataset_provenance["source_manifest_sha256"]),
        conversion_report_sha256=str(dataset_provenance["conversion_report_sha256"]),
        checkpoint_sha256=str(checkpoint["transformer_sha256"]),
        decoder_identity_sha256=decoder_identity,
        conditioning_protocol_id=CAUSAL_FUTURE_ONLY_PROTOCOL,
        evaluation_view_id=evaluation_view.view_id,
        evaluation_view_sha256=evaluation_view.view_sha256,
        generation_provenance={
            "config_name": args.config_name,
            "seed_contract": seed_contract,
            "sample_seeds": sample_seeds,
            "n_steps": args.n_steps,
            "n_samples": len(evaluation_view.entries),
            "crop_start": 0,
            "max_latent_frames": 5,
            "conditioning_protocol_id": CAUSAL_FUTURE_ONLY_PROTOCOL,
            "conditioning_visibility": {
                "task_prompt": True,
                "video_frame0_only": True,
                "tactile_frame0_only": True,
                "future_video": False,
                "future_tactile": False,
                "ground_truth_action": False,
                "action_initialization": "all_zero_with_zero_mask",
            },
            "evaluation_view": evaluation_view.to_json_dict(),
            "ground_truth_embedded": False,
            "runtime_contract": runtime_contract,
            "dataset_provenance": dataset_provenance,
            "checkpoint_provenance": checkpoint,
            "checkpoint_dataset_binding": checkpoint_dataset_binding,
            "decoder_provenance": decoder,
        },
    )
    return {
        "output": str(output.resolve(strict=True)),
        "sample_count": len(samples),
        "source_manifest_sha256": dataset_provenance["source_manifest_sha256"],
        "conversion_report_sha256": dataset_provenance["conversion_report_sha256"],
        "checkpoint_sha256": checkpoint["transformer_sha256"],
        "decoder_identity_sha256": decoder_identity,
        "conditioning_protocol_id": CAUSAL_FUTURE_ONLY_PROTOCOL,
        "evaluation_view_id": evaluation_view.view_id,
        "evaluation_view_sha256": evaluation_view.view_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = generate_tactile_prediction_artifact(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
