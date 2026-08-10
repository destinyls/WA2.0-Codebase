"""Contracts for physical-split Track 3.1 latent encoding."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from script import encode_lerobot_n0_latents as video_encoder
from script import encode_tactile_latent as tactile_encoder

ENCODER_MODULES = (video_encoder, tactile_encoder)


def _parse_args(module: ModuleType, split: str) -> Any:
    argv = [
        "--dataset-root",
        f"/datasets/{split}",
        "--model-path",
        "/models/wan",
        "--artifact-root",
        "/artifacts",
        "--split",
        split,
    ]
    if module is tactile_encoder:
        argv.extend(["--tactile-keys", "observation.images.tactile_left"])
    return module.parse_args(argv)


def _verify(
    module: ModuleType,
    *,
    artifact_root: Path | None,
    split: str | None,
    dataset_root: Path,
    output_root: Path,
) -> object | None:
    common = {
        "artifact_root": artifact_root,
        "split": split,
        "dataset_root": dataset_root,
        "output_root": output_root,
    }
    if module is video_encoder:
        return module.verify_track31_encoder_contract(
            **common,
            write_episodes_jsonl=False,
            target_fps=10,
            height=256,
            width=256,
            max_sequence_length=512,
            dtype="bf16",
            video_keys=list(video_encoder.TRACK31_VIDEO_KEYS),
            prompt=None,
            execution_device="cuda:0",
            text_encoder_device="cuda:0",
        )
    return module.verify_track31_encoder_contract(
        **common,
        mode="both",
        local_mode="current",
        target_fps=10,
        height=128,
        width=128,
        dtype="bf16",
        tactile_keys=list(tactile_encoder.TRACK31_TACTILE_KEYS),
        execution_device="cuda:0",
    )


@pytest.mark.parametrize("module", ENCODER_MODULES)
@pytest.mark.parametrize("split", ("train759", "frozen40", "train", "validation"))
def test_parser_accepts_physical_and_explicit_legacy_splits(
    module: ModuleType,
    split: str,
) -> None:
    assert _parse_args(module, split).split == split


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_parser_rejects_unknown_split(module: ModuleType) -> None:
    with pytest.raises(SystemExit):
        _parse_args(module, "dev")


def test_video_parser_defaults_to_one_explicit_hcu() -> None:
    args = _parse_args(video_encoder, "train759")
    assert args.device == "cuda:0"
    assert args.text_encoder_device == "cuda:0"


def test_tactile_parser_defaults_to_one_explicit_hcu() -> None:
    assert _parse_args(tactile_encoder, "train759").device == "cuda:0"


@pytest.mark.parametrize("module", ENCODER_MODULES)
@pytest.mark.parametrize("split", ("train759", "frozen40"))
def test_formal_contract_uses_only_universe_manifest_and_selected_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    split: str,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    dataset_root = tmp_path / "lerobot" / split
    dataset_root.mkdir(parents=True)
    output_name = "latents" if module is video_encoder else "latents_tactile"
    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_physical_verifier(**kwargs: object) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        module,
        "verify_track31_physical_bundle",
        fake_physical_verifier,
    )
    monkeypatch.setattr(
        module,
        "verify_track31_training_bundle",
        lambda **_: pytest.fail("formal encoding used the legacy verifier"),
    )

    result = _verify(
        module,
        artifact_root=artifact_root,
        split=split,
        dataset_root=dataset_root,
        output_root=(dataset_root / output_name).resolve(),
    )

    assert result is sentinel
    assert calls == [
        {
            "manifest_path": artifact_root / "universe_manifest_v4.json",
            "conversion_report_path": artifact_root / "conversion_report.json",
            "dataset_root": dataset_root,
            "physical_split": split,
        }
    ]
    assert "normalizer_path" not in calls[0]


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_formal_contract_rejects_dataset_repo_name_mismatch(
    tmp_path: Path,
    module: ModuleType,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    dataset_root = tmp_path / "lerobot" / "train759"
    dataset_root.mkdir(parents=True)
    output_name = "latents" if module is video_encoder else "latents_tactile"

    with pytest.raises(ValueError, match="dataset root/split mismatch"):
        _verify(
            module,
            artifact_root=artifact_root,
            split="frozen40",
            dataset_root=dataset_root,
            output_root=(dataset_root / output_name).resolve(),
        )


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_legacy_contract_remains_explicitly_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    dataset_root = tmp_path / "lerobot" / "train"
    dataset_root.mkdir(parents=True)
    output_name = "latents" if module is video_encoder else "latents_tactile"
    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_legacy_verifier(**kwargs: object) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        module,
        "verify_track31_training_bundle",
        fake_legacy_verifier,
    )
    monkeypatch.setattr(
        module,
        "verify_track31_physical_bundle",
        lambda **_: pytest.fail("legacy encoding used the physical verifier"),
    )

    result = _verify(
        module,
        artifact_root=artifact_root,
        split="train",
        dataset_root=dataset_root,
        output_root=(dataset_root / output_name).resolve(),
    )

    assert result is sentinel
    assert calls == [
        {
            "manifest_path": artifact_root / "dataset_manifest.json",
            "normalizer_path": artifact_root / "qpos8_normalizer.json",
            "conversion_report_path": artifact_root / "conversion_report.json",
            "dataset_root": dataset_root.parent,
        }
    ]


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_artifact_root_and_split_are_atomic_options(
    tmp_path: Path,
    module: ModuleType,
) -> None:
    dataset_root = tmp_path / "train759"
    output_name = "latents" if module is video_encoder else "latents_tactile"
    with pytest.raises(ValueError, match="must be provided together"):
        _verify(
            module,
            artifact_root=tmp_path,
            split=None,
            dataset_root=dataset_root,
            output_root=(dataset_root / output_name).resolve(),
        )


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_formal_contract_rejects_symlink_output_root(
    tmp_path: Path,
    module: ModuleType,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    dataset_root = tmp_path / "lerobot" / "train759"
    dataset_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    output_name = "latents" if module is video_encoder else "latents_tactile"
    output_root = dataset_root / output_name
    output_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="standard|symlink"):
        _verify(
            module,
            artifact_root=artifact_root,
            split="train759",
            dataset_root=dataset_root,
            output_root=output_root,
        )


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_formal_contract_rejects_resolved_symlink_target_before_marker_change(
    tmp_path: Path,
    module: ModuleType,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    dataset_root = tmp_path / "lerobot" / "train759"
    dataset_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    output_name = "latents" if module is video_encoder else "latents_tactile"
    (dataset_root / output_name).symlink_to(outside, target_is_directory=True)
    marker_name = (
        "latent_video_inventory.json"
        if module is video_encoder
        else "latent_tactile_inventory.json"
    )
    marker = dataset_root / marker_name
    marker.write_text('{"status":"ready"}', encoding="utf-8")

    with pytest.raises(ValueError, match="standard|symlink|escaped"):
        _verify(
            module,
            artifact_root=artifact_root,
            split="train759",
            dataset_root=dataset_root,
            output_root=outside.resolve(),
        )

    assert marker.read_text(encoding="utf-8") == '{"status":"ready"}'


def test_formal_video_rejects_prompt_override_before_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    dataset_root = tmp_path / "train759"
    dataset_root.mkdir()
    monkeypatch.setattr(
        video_encoder,
        "verify_track31_physical_bundle",
        lambda **_: pytest.fail("invalid formal parameters reached the verifier"),
    )

    with pytest.raises(ValueError, match="frozen episode prompts"):
        video_encoder.verify_track31_encoder_contract(
            artifact_root=artifact_root,
            split="train759",
            dataset_root=dataset_root,
            output_root=dataset_root / "latents",
            write_episodes_jsonl=False,
            target_fps=10,
            height=256,
            width=256,
            max_sequence_length=512,
            dtype="bf16",
            video_keys=list(video_encoder.TRACK31_VIDEO_KEYS),
            prompt="override",
            execution_device="cuda:0",
            text_encoder_device="cuda:0",
        )


@pytest.mark.parametrize(
    ("execution_device", "text_encoder_device"),
    (
        ("cuda:0", "cpu"),
        ("cpu", "cuda:0"),
        ("cuda:0", "cuda:1"),
        ("cuda", "cuda"),
        ("cuda:1", "cuda:1"),
    ),
)
def test_formal_video_requires_same_hcu_for_vae_and_text_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_device: str,
    text_encoder_device: str,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    dataset_root = tmp_path / "train759"
    dataset_root.mkdir()
    monkeypatch.setattr(
        video_encoder,
        "verify_track31_physical_bundle",
        lambda **_: pytest.fail("invalid HCU policy reached the verifier"),
    )

    with pytest.raises(ValueError, match="same HCU|CUDA"):
        video_encoder.verify_track31_encoder_contract(
            artifact_root=artifact_root,
            split="train759",
            dataset_root=dataset_root,
            output_root=dataset_root / "latents",
            write_episodes_jsonl=False,
            target_fps=10,
            height=256,
            width=256,
            max_sequence_length=512,
            dtype="bf16",
            video_keys=list(video_encoder.TRACK31_VIDEO_KEYS),
            prompt=None,
            execution_device=execution_device,
            text_encoder_device=text_encoder_device,
        )


def test_formal_tactile_rejects_noncanonical_shape_before_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    dataset_root = tmp_path / "frozen40"
    dataset_root.mkdir()
    monkeypatch.setattr(
        tactile_encoder,
        "verify_track31_physical_bundle",
        lambda **_: pytest.fail("invalid formal parameters reached the verifier"),
    )

    with pytest.raises(ValueError, match="128x128"):
        tactile_encoder.verify_track31_encoder_contract(
            artifact_root=artifact_root,
            split="frozen40",
            dataset_root=dataset_root,
            output_root=dataset_root / "latents_tactile",
            mode="both",
            local_mode="current",
            target_fps=10,
            height=96,
            width=128,
            dtype="bf16",
            tactile_keys=list(tactile_encoder.TRACK31_TACTILE_KEYS),
            execution_device="cuda:0",
        )


def test_formal_tactile_rejects_cpu_before_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    dataset_root = tmp_path / "train759"
    dataset_root.mkdir()
    monkeypatch.setattr(
        tactile_encoder,
        "verify_track31_physical_bundle",
        lambda **_: pytest.fail("CPU policy reached the physical verifier"),
    )

    with pytest.raises(ValueError, match="HCU cuda:0"):
        tactile_encoder.verify_track31_encoder_contract(
            artifact_root=artifact_root,
            split="train759",
            dataset_root=dataset_root,
            output_root=dataset_root / "latents_tactile",
            mode="both",
            local_mode="current",
            target_fps=10,
            height=128,
            width=128,
            dtype="bf16",
            tactile_keys=list(tactile_encoder.TRACK31_TACTILE_KEYS),
            execution_device="cpu",
        )


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_nested_output_symlink_is_rejected(
    tmp_path: Path,
    module: ModuleType,
) -> None:
    dataset_root = tmp_path / "train759"
    dataset_root.mkdir()
    output_name = "latents" if module is video_encoder else "latents_tactile"
    output_root = dataset_root / output_name
    output_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output_root / "chunk-000").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        module.require_symlink_free_output_tree(
            output_root=output_root,
            dataset_root=dataset_root,
        )


@pytest.mark.parametrize("module", ENCODER_MODULES)
def test_invalid_formal_parameters_do_not_mutate_ready_marker_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    dataset_root = tmp_path / "train759"
    dataset_root.mkdir()
    marker_name = (
        "latent_video_inventory.json"
        if module is video_encoder
        else "latent_tactile_inventory.json"
    )
    marker = dataset_root / marker_name
    marker.write_bytes(b"existing-ready-marker")
    argv = [
        "--dataset-root",
        str(dataset_root),
        "--model-path",
        str(tmp_path / "model"),
        "--artifact-root",
        str(artifact_root),
        "--split",
        "train759",
    ]
    output_name = "latents" if module is video_encoder else "latents_tactile"
    if module is video_encoder:
        argv.extend(["--prompt", "invalid override"])
    else:
        argv.extend(
            [
                "--tactile-keys",
                *tactile_encoder.TRACK31_TACTILE_KEYS,
                "--height",
                "96",
            ]
        )
    args = module.parse_args(argv)
    monkeypatch.setattr(module, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="formal Track 3.1"):
        module.main()

    assert marker.read_bytes() == b"existing-ready-marker"
    assert not (dataset_root / output_name).exists()
