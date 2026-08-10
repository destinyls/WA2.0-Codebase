"""Regression tests for the FlexAttention mask contract."""

from __future__ import annotations

import os
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from n0_twam.models.model import (
    FlexAttnFunc,
    GroupedAttentionMask,
    WanAttention,
    _grouped_sdpa_query_token_limit,
    _resolve_flex_attention_backend,
    _resolve_mot_cross_attention_backend,
    _uses_eager_block_mask_creation,
    capture_attention_execution_contract,
)


def _evaluate_with_nested_vmap(
    mask_mod: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    query_length: int,
    key_value_length: int,
) -> torch.Tensor:
    """Match FlexAttention's q/kv vmaps without its getitem compatibility mode."""
    # Vendor Torch 2.5's compiled Flex HOP supplies int32 q/kv indices even
    # though eager create_block_mask uses int64.  Exercise the stricter path.
    query_indices = torch.arange(query_length, dtype=torch.int32)
    key_value_indices = torch.arange(key_value_length, dtype=torch.int32)
    batch = torch.tensor(0)
    head = torch.tensor(0)
    over_keys = torch.vmap(mask_mod, in_dims=(None, None, None, 0))
    over_queries = torch.vmap(over_keys, in_dims=(None, None, 0, None))
    return over_queries(batch, head, query_indices, key_value_indices)


def _grouped_mask_to_dense(mask: GroupedAttentionMask) -> torch.Tensor:
    """Materialize a grouped mask only for small reference tests."""
    dense = torch.zeros(mask.shape[-2:], dtype=torch.bool)
    for query_indices, key_value_indices in mask.groups:
        dense[query_indices[:, None], key_value_indices[None, :]] = True
    return dense


def test_self_attention_mask_matches_dense_reference_under_vmap() -> None:
    seq_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, -1])
    frame_ids = torch.tensor([0, 0, 2, 2, 6, 0, 2, -1])
    noise_ids = torch.tensor([1, 0, 1, 0, 1, 1, 0, -1])
    modality_ids = torch.tensor([0, 2, 1, 0, 0, 0, 2, -1])
    window_size = 2

    mask_mod = FlexAttnFunc._get_mask_mod(
        seq_ids,
        frame_ids,
        noise_ids,
        modality_ids,
        window_size,
    )
    actual = _evaluate_with_nested_vmap(mask_mod, len(seq_ids), len(seq_ids))

    query_frames = frame_ids[:, None]
    key_value_frames = frame_ids[None, :]
    query_noise = noise_ids[:, None]
    key_value_noise = noise_ids[None, :]
    clean_to_clean = (
        (query_noise == 1) & (key_value_noise == 1) & (key_value_frames <= query_frames)
    )
    noise_to_clean = (
        (query_noise == 0) & (key_value_noise == 1) & (key_value_frames < query_frames)
    )
    noise_to_noise = (
        (query_noise == 0) & (key_value_noise == 0) & (key_value_frames == query_frames)
    )
    same_valid_sequence = (
        (seq_ids[:, None] == seq_ids[None, :])
        & (seq_ids[:, None] >= 0)
        & (seq_ids[None, :] >= 0)
    )
    within_window = (query_frames - key_value_frames).abs() <= window_size
    expected = (
        (clean_to_clean | noise_to_clean | noise_to_noise)
        & same_valid_sequence
        & within_window
    )

    torch.testing.assert_close(actual, expected)
    assert actual[2, 0]  # clean frame 2 can attend clean frame 0
    assert not actual[3, 2]  # noisy frame 2 cannot attend same-frame clean
    assert actual[3, 0]  # noisy frame 2 can attend past clean
    assert actual[3, 3]  # noisy frame 2 can attend same-frame noise
    assert not actual[4, 0]  # window excludes clean frame 0 from frame 6
    assert not actual[:, 7].any()  # padding is never a key
    assert not actual[7].any()  # padding is never a query


def test_cross_attention_mask_matches_text_and_padding_reference_under_vmap() -> None:
    query_seq_ids = torch.tensor([0, 0, 1, -1])
    query_modality_ids = torch.tensor([0, 2, 1, -1])
    text_seq_ids = torch.tensor([0, 0, 1, 1, -1])
    text_type_ids = torch.zeros_like(text_seq_ids)
    mask_mod = FlexAttnFunc._get_cross_mask_mod(
        query_seq_ids,
        query_modality_ids,
        text_seq_ids,
        text_type_ids,
    )

    actual = _evaluate_with_nested_vmap(
        mask_mod,
        len(query_seq_ids),
        len(text_seq_ids),
    )
    expected = (
        (query_seq_ids[:, None] == text_seq_ids[None, :])
        & (query_seq_ids[:, None] >= 0)
        & (text_seq_ids[None, :] >= 0)
        & (query_modality_ids[:, None] != 2)
    )

    torch.testing.assert_close(actual, expected)
    assert actual[0, :2].all()  # video query attends same-sequence text
    assert not actual[1].any()  # tactile query never cross-attends text
    assert actual[2, 2:4].all()  # action query attends its sequence text
    assert not actual[:, 4].any()  # padded text is never a key
    assert not actual[3].any()  # padded query is never allowed


def test_grouped_self_attention_mask_matches_dense_reference() -> None:
    seq_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, -1])
    frame_ids = torch.tensor([0, 0, 2, 2, 6, 0, 2, -1])
    noise_ids = torch.tensor([1, 0, 1, 0, 1, 1, 0, -1])
    window_size = 2

    grouped_mask = FlexAttnFunc._create_grouped_self_mask(
        seq_ids,
        frame_ids,
        noise_ids,
        window_size,
    )
    actual = _grouped_mask_to_dense(grouped_mask)

    query_frames = frame_ids[:, None]
    key_value_frames = frame_ids[None, :]
    query_noise = noise_ids[:, None]
    key_value_noise = noise_ids[None, :]
    clean_to_clean = (
        (query_noise == 1) & (key_value_noise == 1) & (key_value_frames <= query_frames)
    )
    noise_to_clean = (
        (query_noise == 0) & (key_value_noise == 1) & (key_value_frames < query_frames)
    )
    noise_to_noise = (
        (query_noise == 0) & (key_value_noise == 0) & (key_value_frames == query_frames)
    )
    same_valid_sequence = (
        (seq_ids[:, None] == seq_ids[None, :])
        & (seq_ids[:, None] >= 0)
        & (seq_ids[None, :] >= 0)
    )
    within_window = (query_frames - key_value_frames).abs() <= window_size
    expected = (
        (clean_to_clean | noise_to_clean | noise_to_noise)
        & same_valid_sequence
        & within_window
    )

    assert grouped_mask.shape == (1, 1, len(seq_ids), len(seq_ids))
    torch.testing.assert_close(actual, expected)


def test_grouped_cross_attention_mask_matches_dense_reference() -> None:
    query_seq_ids = torch.tensor([0, 0, 1, 2, -1])
    query_modality_ids = torch.tensor([0, 2, 1, 0, -1])
    text_seq_ids = torch.tensor([0, 0, 1, 1, -1])
    grouped_mask = FlexAttnFunc._create_grouped_cross_mask(
        query_seq_ids,
        query_modality_ids,
        text_seq_ids,
    )

    actual = _grouped_mask_to_dense(grouped_mask)
    expected = (
        (query_seq_ids[:, None] == text_seq_ids[None, :])
        & (query_seq_ids[:, None] >= 0)
        & (text_seq_ids[None, :] >= 0)
        & (query_modality_ids[:, None] != 2)
    )

    assert grouped_mask.shape == (
        1,
        1,
        len(query_seq_ids),
        len(text_seq_ids),
    )
    torch.testing.assert_close(actual, expected)
    assert not actual[1].any()  # tactile query remains all-masked
    assert not actual[3].any()  # valid query with no matching text remains all-masked
    assert not actual[4].any()  # padded query remains all-masked


def test_grouped_sdpa_output_and_gradients_match_dense_masked_reference() -> None:
    torch.manual_seed(20260801)
    seq_ids = torch.tensor([0, 0, 0, 0, 1, 1, -1])
    frame_ids = torch.tensor([0, 0, 2, 2, 0, 2, -1])
    noise_ids = torch.tensor([1, 0, 1, 0, 1, 0, -1])
    grouped_mask = FlexAttnFunc._create_grouped_self_mask(
        seq_ids,
        frame_ids,
        noise_ids,
        window_size=2,
    )
    dense_mask = _grouped_mask_to_dense(grouped_mask)
    shape = (1, len(seq_ids), 2, 8)
    query_data = torch.randn(shape, dtype=torch.bfloat16)
    key_data = torch.randn(shape, dtype=torch.bfloat16)
    value_data = torch.randn(shape, dtype=torch.bfloat16)

    grouped_inputs = [
        tensor.clone().requires_grad_(True)
        for tensor in (query_data, key_data, value_data)
    ]
    reference_inputs = [
        tensor.clone().requires_grad_(True)
        for tensor in (query_data, key_data, value_data)
    ]
    with patch.dict(
        os.environ,
        {"N0_FLEX_ATTENTION_BACKEND": "grouped_sdpa"},
    ):
        operation = FlexAttnFunc()
    operation.set_block_mask(grouped_mask)
    grouped_output = operation(*grouped_inputs)
    reference_output = F.scaled_dot_product_attention(
        reference_inputs[0].transpose(1, 2),
        reference_inputs[1].transpose(1, 2),
        reference_inputs[2].transpose(1, 2),
        attn_mask=dense_mask[None, None],
    ).transpose(1, 2)

    weights = torch.linspace(
        0.25,
        1.25,
        steps=grouped_output.numel(),
    ).reshape_as(grouped_output)
    (grouped_output.float() * weights).sum().backward()
    (reference_output.float() * weights).sum().backward()

    torch.testing.assert_close(grouped_output, reference_output, rtol=0.02, atol=0.02)
    assert torch.count_nonzero(grouped_output[:, -1]) == 0
    assert torch.count_nonzero(reference_output[:, -1]) == 0
    for grouped_input, reference_input in zip(grouped_inputs, reference_inputs):
        torch.testing.assert_close(
            grouped_input.grad,
            reference_input.grad,
            rtol=0.03,
            atol=0.03,
        )


def test_grouped_flash_attention_preserves_grouped_mask_semantics() -> None:
    torch.manual_seed(20260801)
    seq_ids = torch.tensor([0, 0, 0, 0, -1])
    frame_ids = torch.tensor([0, 0, 2, 2, -1])
    noise_ids = torch.tensor([1, 0, 1, 0, -1])
    grouped_mask = FlexAttnFunc._create_grouped_self_mask(
        seq_ids,
        frame_ids,
        noise_ids,
        window_size=2,
    )
    dense_mask = _grouped_mask_to_dense(grouped_mask)
    shape = (1, len(seq_ids), 2, 8)
    input_data = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(3)]
    grouped_inputs = [tensor.clone().requires_grad_(True) for tensor in input_data]
    reference_inputs = [tensor.clone().requires_grad_(True) for tensor in input_data]

    def fake_flash_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        dropout_p: float,
        causal: bool,
    ) -> torch.Tensor:
        assert dropout_p == 0.0
        assert not causal
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)

    with (
        patch.dict(
            os.environ,
            {"N0_FLEX_ATTENTION_BACKEND": "grouped_flash_attn"},
        ),
        patch(
            "n0_twam.models.model.flash_attn_func",
            fake_flash_attention,
        ),
    ):
        operation = FlexAttnFunc()
        operation.set_block_mask(grouped_mask)
        grouped_output = operation(*grouped_inputs)

    reference_output = F.scaled_dot_product_attention(
        reference_inputs[0].transpose(1, 2),
        reference_inputs[1].transpose(1, 2),
        reference_inputs[2].transpose(1, 2),
        attn_mask=dense_mask[None, None],
    ).transpose(1, 2)
    grouped_output.float().sum().backward()
    reference_output.float().sum().backward()

    torch.testing.assert_close(grouped_output, reference_output, rtol=0.02, atol=0.02)
    assert torch.count_nonzero(grouped_output[:, -1]) == 0
    for grouped_input, reference_input in zip(grouped_inputs, reference_inputs):
        torch.testing.assert_close(
            grouped_input.grad,
            reference_input.grad,
            rtol=0.03,
            atol=0.03,
        )


def test_fully_masked_grouped_sdpa_is_differentiable_zero() -> None:
    empty_mask = GroupedAttentionMask(shape=(1, 1, 3, 4), groups=())
    tensors = [
        torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
        for shape in ((1, 3, 1, 4), (1, 4, 1, 4), (1, 4, 1, 4))
    ]
    with patch.dict(
        os.environ,
        {"N0_FLEX_ATTENTION_BACKEND": "grouped_sdpa"},
    ):
        operation = FlexAttnFunc(is_cross=True)
    operation.set_block_mask(empty_mask)
    output = operation(*tensors)
    output.float().sum().backward()

    assert torch.count_nonzero(output) == 0
    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) == 0


def test_grouped_sdpa_rejects_overlapping_query_groups() -> None:
    overlapping_mask = GroupedAttentionMask(
        shape=(1, 1, 2, 2),
        groups=(
            (torch.tensor([0]), torch.tensor([0])),
            (torch.tensor([0, 1]), torch.tensor([1])),
        ),
    )
    tensors = [torch.randn((1, 2, 1, 4), dtype=torch.bfloat16) for _ in range(3)]
    with patch.dict(
        os.environ,
        {"N0_FLEX_ATTENTION_BACKEND": "grouped_sdpa"},
    ):
        operation = FlexAttnFunc()
    operation.set_block_mask(overlapping_mask)

    try:
        operation(*tensors)
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("overlapping grouped queries did not fail")


def test_grouped_sdpa_rejects_out_of_bounds_key_indices() -> None:
    invalid_mask = GroupedAttentionMask(
        shape=(1, 1, 2, 2),
        groups=((torch.tensor([0]), torch.tensor([2])),),
    )
    tensors = [torch.randn((1, 2, 1, 4), dtype=torch.bfloat16) for _ in range(3)]
    with patch.dict(os.environ, {"N0_FLEX_ATTENTION_BACKEND": "grouped_sdpa"}):
        operation = FlexAttnFunc()
    operation.set_block_mask(invalid_mask)

    try:
        operation(*tensors)
    except IndexError as error:
        assert "key/value" in str(error)
    else:
        raise AssertionError("out-of-bounds grouped key index did not fail")


def test_grouped_sdpa_query_limit_is_bounded_and_fail_closed() -> None:
    with patch.dict(os.environ):
        os.environ.pop("N0_GROUPED_SDPA_MAX_QUERY_TOKENS", None)
        assert _grouped_sdpa_query_token_limit() == 16384
    with patch.dict(os.environ, {"N0_GROUPED_SDPA_MAX_QUERY_TOKENS": "2048"}):
        assert _grouped_sdpa_query_token_limit() == 2048
    for invalid in ("", "0", "-1", "1.5", " 2048"):
        with patch.dict(
            os.environ,
            {"N0_GROUPED_SDPA_MAX_QUERY_TOKENS": invalid},
        ):
            try:
                _grouped_sdpa_query_token_limit()
            except ValueError as error:
                assert "N0_GROUPED_SDPA_MAX_QUERY_TOKENS" in str(error)
            else:
                raise AssertionError(
                    f"invalid grouped SDPA limit accepted: {invalid!r}"
                )


def test_torch_25_hip_routes_to_vendor_flash_by_default() -> None:
    with (
        patch.dict(os.environ),
        patch("n0_twam.models.model.torch.__version__", "2.5.1+vendor"),
        patch("n0_twam.models.model.torch.version.hip", "6.3.0"),
        patch("n0_twam.models.model.flash_attn_func", object()),
    ):
        os.environ.pop("N0_FLEX_ATTENTION_BACKEND", None)
        assert _resolve_flex_attention_backend() == "grouped_flash_attn"


def test_torch_25_hip_falls_back_to_grouped_sdpa_without_vendor_flash() -> None:
    with (
        patch.dict(os.environ),
        patch("n0_twam.models.model.torch.__version__", "2.5.1+vendor"),
        patch("n0_twam.models.model.torch.version.hip", "6.3.0"),
        patch("n0_twam.models.model.flash_attn_func", None),
    ):
        os.environ.pop("N0_FLEX_ATTENTION_BACKEND", None)
        assert _resolve_flex_attention_backend() == "grouped_sdpa"


def test_nvidia_and_upstream_torch_route_to_flex_by_default() -> None:
    with patch.dict(os.environ):
        os.environ.pop("N0_FLEX_ATTENTION_BACKEND", None)
        with (
            patch("n0_twam.models.model.torch.__version__", "2.5.1+cu124"),
            patch("n0_twam.models.model.torch.version.hip", None),
        ):
            assert _resolve_flex_attention_backend() == "flex"
        with (
            patch("n0_twam.models.model.torch.__version__", "2.9.0"),
            patch("n0_twam.models.model.torch.version.hip", "6.3.0"),
        ):
            assert _resolve_flex_attention_backend() == "flex"


def test_attention_backend_override_is_explicit_and_fail_closed() -> None:
    with patch.dict(
        os.environ,
        {"N0_FLEX_ATTENTION_BACKEND": "grouped_sdpa"},
    ):
        assert _resolve_flex_attention_backend() == "grouped_sdpa"
    with patch.dict(os.environ, {"N0_FLEX_ATTENTION_BACKEND": "flex"}):
        assert _resolve_flex_attention_backend() == "flex"
    with (
        patch.dict(
            os.environ,
            {"N0_FLEX_ATTENTION_BACKEND": "grouped_flash_attn"},
        ),
        patch("n0_twam.models.model.flash_attn_func", object()),
    ):
        assert _resolve_flex_attention_backend() == "grouped_flash_attn"
    with (
        patch.dict(
            os.environ,
            {"N0_FLEX_ATTENTION_BACKEND": "grouped_flash_attn"},
        ),
        patch("n0_twam.models.model.flash_attn_func", None),
    ):
        try:
            _resolve_flex_attention_backend()
        except RuntimeError as error:
            assert "vendor flash-attn" in str(error)
        else:
            raise AssertionError("missing vendor flash-attn did not fail closed")
    with patch.dict(os.environ, {"N0_FLEX_ATTENTION_BACKEND": "sdpa"}):
        try:
            _resolve_flex_attention_backend()
        except ValueError as error:
            assert "N0_FLEX_ATTENTION_BACKEND" in str(error)
        else:
            raise AssertionError("unknown attention backend override did not fail")


def test_attention_execution_contract_records_resolved_grouped_limit() -> None:
    with patch.dict(
        os.environ,
        {
            "N0_FLEX_ATTENTION_BACKEND": "grouped_sdpa",
            "N0_GROUPED_SDPA_MAX_QUERY_TOKENS": "8192",
            "N0_MOT_CROSS_ATTENTION_BACKEND": "sdpa",
        },
    ):
        assert capture_attention_execution_contract() == {
            "attention_backend": "grouped_sdpa",
            "grouped_sdpa_max_query_tokens": 8192,
            "mot_cross_attention_backend": "sdpa",
        }


def test_mot_cross_attention_flash_matches_sdpa_and_mask_falls_back() -> None:
    def fake_flash_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)

    torch.manual_seed(13)
    with patch("n0_twam.models.model.flash_attn_func", fake_flash_attention):
        flash = WanAttention(
            dim=8,
            heads=2,
            dim_head=4,
            cross_attention_dim_head=4,
            attn_mode="flashattn",
        )
    sdpa = WanAttention(
        dim=8,
        heads=2,
        dim_head=4,
        cross_attention_dim_head=4,
        attn_mode="torch",
    )
    sdpa.load_state_dict(flash.state_dict())
    query = torch.randn(1, 3, 8)
    context = torch.randn(1, 5, 8)

    torch.testing.assert_close(
        flash(query, context, context, None),
        sdpa(query, context, context, None),
    )
    mask = torch.ones(1, 1, 3, 5, dtype=torch.bool)
    torch.testing.assert_close(
        flash(query, context, context, None, attn_mask=mask),
        sdpa(query, context, context, None, attn_mask=mask),
    )


def test_mot_cross_attention_backend_fails_closed_without_vendor_kernel() -> None:
    with (
        patch.dict(
            os.environ,
            {"N0_MOT_CROSS_ATTENTION_BACKEND": "flash_attn"},
        ),
        patch("n0_twam.models.model.flash_attn_func", None),
    ):
        with pytest.raises(RuntimeError, match="vendor flash-attn"):
            _resolve_mot_cross_attention_backend()


def test_vendor_torch_25_uses_eager_mask_creation_only() -> None:
    assert _uses_eager_block_mask_creation("2.5.1+vendor")
    assert _uses_eager_block_mask_creation("2.5.1a0+gitunknown")
    assert not _uses_eager_block_mask_creation("2.9.0")
    assert not _uses_eager_block_mask_creation("2.10.0.dev20260801")


def test_vendor_torch_25_routes_block_mask_creation_to_eager() -> None:
    sentinel = object()

    def mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        key_value_index: torch.Tensor,
    ) -> torch.Tensor:
        return query_index >= key_value_index

    with (
        patch("n0_twam.models.model.torch.__version__", "2.5.1+vendor"),
        patch("n0_twam.models.model.create_block_mask") as eager_create,
        patch.object(FlexAttnFunc, "compiled_create_block_mask") as compiled_create,
    ):
        eager_create.return_value = sentinel
        result = FlexAttnFunc._create_block_mask(mask_mod, 8, 8, torch.device("cpu"))

    assert result is sentinel
    compiled_create.assert_not_called()
    eager_create.assert_called_once()
    assert eager_create.call_args.kwargs["_compile"] is False


def test_upstream_torch_29_routes_block_mask_creation_to_compiled() -> None:
    sentinel = object()

    def mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        key_value_index: torch.Tensor,
    ) -> torch.Tensor:
        return query_index >= key_value_index

    with (
        patch("n0_twam.models.model.torch.__version__", "2.9.0"),
        patch("n0_twam.models.model.create_block_mask") as eager_create,
        patch.object(FlexAttnFunc, "compiled_create_block_mask") as compiled_create,
    ):
        compiled_create.return_value = sentinel
        result = FlexAttnFunc._create_block_mask(mask_mod, 8, 8, torch.device("cpu"))

    assert result is sentinel
    eager_create.assert_not_called()
    compiled_create.assert_called_once()
    assert compiled_create.call_args.kwargs["_compile"] is False
