"""Unit coverage for batched umT5 prompt encoding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from n0_twam.data.latent_inventory import build_track31_payload_provenance
from n0_twam.models import utils as model_utils
from script import encode_lerobot_n0_latents as video_encoder
from script.track3_1 import hcu_prompt_encoding


class _Tokenizer:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompts: list[str], **_: object) -> SimpleNamespace:
        self.calls += 1
        self.prompts = list(prompts)
        batch_size = len(prompts)
        input_ids = torch.arange(4).repeat(batch_size, 1)
        attention_mask = torch.tensor(
            [[1, 1, 0, 0], [1, 1, 1, 0]],
            dtype=torch.long,
        )
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


class _TextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.forward_calls = 0
        self.grad_enabled_during_forward: bool | None = None
        self.inference_mode_during_forward: bool | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        self.forward_calls += 1
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        self.inference_mode_during_forward = torch.is_inference_mode_enabled()
        values = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 3)
        values = values + self.anchor
        return SimpleNamespace(last_hidden_state=values)


def test_direct_hcu_loader_uses_low_memory_device_map(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []
    sentinel = object()

    def fake_from_pretrained(path: Path, **kwargs: object) -> object:
        calls.append((path, kwargs))
        return sentinel

    monkeypatch.setattr(
        model_utils,
        "UMT5EncoderModel",
        SimpleNamespace(from_pretrained=fake_from_pretrained),
    )

    loaded = model_utils.load_text_encoder(
        tmp_path / "text_encoder",
        torch_dtype=torch.bfloat16,
        torch_device=torch.device("cuda:0"),
        direct_device_load=True,
    )

    assert loaded is sentinel
    assert calls == [
        (
            tmp_path / "text_encoder",
            {
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
                "device_map": {"": "cuda:0"},
            },
        )
    ]


def test_unique_prompts_use_one_inference_only_batch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(hcu_prompt_encoding, "normalize_prompt", lambda value: value)
    model = _TextEncoder()
    tokenizer = _Tokenizer()

    cache = video_encoder.build_text_embedding_cache(
        prompts=["insert_HDMI", "insert_HDMI", "lift_bottle"],
        tokenizer=tokenizer,
        text_encoder=model,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_sequence_length=4,
    )

    assert model.forward_calls == 1
    assert model.grad_enabled_during_forward is False
    assert model.inference_mode_during_forward is True
    assert model.training is False
    assert tokenizer.calls == 1
    assert tokenizer.prompts == ["insert_HDMI", "lift_bottle"]
    assert set(cache) == {"insert_HDMI", "lift_bottle"}
    assert all(value.device.type == "cpu" for value in cache.values())
    assert all(not value.requires_grad for value in cache.values())
    assert all(value.grad_fn is None for value in cache.values())
    assert tuple(cache["insert_HDMI"].shape) == (4, 3)
    assert torch.count_nonzero(cache["insert_HDMI"][2:]) == 0
    assert torch.count_nonzero(cache["lift_bottle"][3:]) == 0


def test_formal_prompt_batch_rejects_more_than_eight_unique_prompts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(hcu_prompt_encoding, "normalize_prompt", lambda value: value)
    with pytest.raises(ValueError, match="exceeds"):
        video_encoder.build_text_embedding_cache(
            prompts=[f"task-{index}" for index in range(9)],
            tokenizer=_Tokenizer(),
            text_encoder=_TextEncoder(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_sequence_length=4,
            max_unique_prompts=8,
        )


def test_formal_prompt_cache_round_trip_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "prompt-cache.pt"
    prompts = {"insert hdmi", "lift bottle"}
    embeddings = {
        prompt: torch.full(
            (4, hcu_prompt_encoding.PROMPT_CACHE_EMBEDDING_WIDTH),
            index + 1,
            dtype=torch.bfloat16,
        )
        for index, prompt in enumerate(sorted(prompts))
    }
    hcu_prompt_encoding.write_prompt_embedding_cache(
        path,
        embeddings=embeddings,
        split="train759",
        manifest_sha256="1" * 64,
        conversion_report_sha256="2" * 64,
        encoder_source_identity_sha256="3" * 64,
        encoding_contract_sha256="4" * 64,
        max_sequence_length=4,
    )

    loaded = hcu_prompt_encoding.load_prompt_embedding_cache(
        path,
        expected_prompts=prompts,
        expected_split="train759",
        expected_manifest_sha256="1" * 64,
        expected_conversion_report_sha256="2" * 64,
        expected_encoder_source_identity_sha256="3" * 64,
        expected_encoding_contract_sha256="4" * 64,
        max_sequence_length=4,
    )

    assert set(loaded) == prompts
    assert all(value.device.type == "cpu" for value in loaded.values())
    assert all(value.dtype == torch.bfloat16 for value in loaded.values())


def test_formal_prompt_cache_rejects_source_identity_drift(tmp_path: Path) -> None:
    path = tmp_path / "prompt-cache.pt"
    embeddings = {
        "insert hdmi": torch.zeros(
            (4, hcu_prompt_encoding.PROMPT_CACHE_EMBEDDING_WIDTH),
            dtype=torch.bfloat16,
        )
    }
    hcu_prompt_encoding.write_prompt_embedding_cache(
        path,
        embeddings=embeddings,
        split="train759",
        manifest_sha256="1" * 64,
        conversion_report_sha256="2" * 64,
        encoder_source_identity_sha256="3" * 64,
        encoding_contract_sha256="4" * 64,
        max_sequence_length=4,
    )

    with pytest.raises(ValueError, match="metadata"):
        hcu_prompt_encoding.load_prompt_embedding_cache(
            path,
            expected_prompts=embeddings,
            expected_split="train759",
            expected_manifest_sha256="1" * 64,
            expected_conversion_report_sha256="2" * 64,
            expected_encoder_source_identity_sha256="5" * 64,
            expected_encoding_contract_sha256="4" * 64,
            max_sequence_length=4,
        )


def _video_provenance(
    execution_device: str,
    text_encoder_device: str,
) -> dict[str, object]:
    return build_track31_payload_provenance(
        split="train759",
        manifest_sha256="1" * 64,
        conversion_report_sha256="2" * 64,
        encoder_source_identity_sha256="3" * 64,
        kind="video",
        episode_index=0,
        start_frame=0,
        end_frame=5,
        episode_task="insert_HDMI",
        stream_key="observation.images.top",
        tactile_mode=None,
        prompt="insert_HDMI",
        source_video_identity={
            "relative_path": (
                "videos/chunk-000/observation.images.top/episode_000000.mp4"
            ),
            "size_bytes": 1,
            "sha256": "4" * 64,
        },
        execution_device=execution_device,
        text_encoder_device=text_encoder_device,
    )


def test_formal_video_provenance_pins_logical_cuda_zero() -> None:
    provenance = _video_provenance("cuda:0", "cuda:0")
    assert provenance["execution_device"] == "cuda:0"
    assert provenance["text_encoder_device"] == "cuda:0"
    assert provenance["text_encoder_device_type"] == "cuda"


@pytest.mark.parametrize(
    ("execution_device", "text_encoder_device"),
    (
        ("cpu", "cpu"),
        ("cpu", "cuda:0"),
        ("cuda:0", "cpu"),
        ("cuda", "cuda"),
        ("cuda:1", "cuda:1"),
        ("cuda:0", "cuda:1"),
    ),
)
def test_formal_video_provenance_rejects_noncanonical_devices(
    execution_device: str,
    text_encoder_device: str,
) -> None:
    with pytest.raises(ValueError, match="HCU-only"):
        _video_provenance(execution_device, text_encoder_device)


def test_documented_formal_video_launcher_pins_umt5_to_hcu() -> None:
    document = (
        Path(__file__).resolve().parents[2] / "docs" / "TRACK31_UNIVTAC.md"
    ).read_text(encoding="utf-8")
    assert "--device cuda:0" in document
    assert "--text-encoder-device cuda:0" in document
    assert "--text-encoder-device cpu" not in document
