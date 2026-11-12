# Copyright (c) 2026, BAAI. All rights reserved.
"""MTP loss plumbing, roll_tensor under CP and MTP rank placement helpers."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_pipeline_model_parallel_rank,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionLayer,
    get_mtp_layer_spec,
    get_mtp_num_layers_to_build,
    get_mtp_ranks,
    mtp_on_this_rank,
    process_mtp_loss,
    roll_tensor,
    tie_output_layer_state_dict,
)
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()
    MTPLossLoggingHelper.tracker = {}
    MTPLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0)


def _config(**overrides):
    defaults = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        mtp_num_layers=2,
        mtp_loss_scaling_factor=0.5,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _output_layer(hidden_states, weight=None, runtime_gather_output=None):
    return hidden_states * weight, None


def _compute_language_model_loss(labels, logits):
    return (labels.unsqueeze(-1) * logits.permute(1, 0, 2)).mean(-1)


def _rolled_last_dim_zeroed(tensor):
    rolled = torch.roll(tensor, shifts=-1, dims=-1)
    rolled[..., -1] = 0
    return rolled


def _expected_weight_grad(chunks, labels, loss_mask, scale, num_layers, main_scale, per_token):
    grad = torch.zeros(chunks[0].shape[-1])
    rolled_labels = labels.clone()
    rolled_mask = loss_mask.clone()
    original_num_tokens = loss_mask.sum()
    for layer in range(num_layers):
        rolled_labels = _rolled_last_dim_zeroed(rolled_labels)
        rolled_mask = _rolled_last_dim_zeroed(rolled_mask)
        num_tokens = rolled_mask.sum()
        term = torch.einsum('bs,sbh->h', rolled_mask * rolled_labels, chunks[layer + 1]) / (
            chunks[0].shape[-1]
        )
        if per_token:
            # calculate_per_token_loss re-scales by the original token count; the global
            # division by the token count happens later in finalize_model_grads.
            term = term * (original_num_tokens / num_tokens)
        else:
            term = term / num_tokens
        grad += scale / num_layers * term
    return main_scale * grad


def test_process_mtp_loss_returns_first_chunk_without_labels(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    hidden_states = torch.randn(12, 2, 8)

    output = process_mtp_loss(
        hidden_states,
        None,
        None,
        _output_layer,
        None,
        None,
        True,
        _compute_language_model_loss,
        _config(),
    )

    assert torch.equal(output, hidden_states[:4])


def test_process_mtp_loss_backward_scales_mtp_gradient(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    config = _config()
    torch.manual_seed(0)
    chunks = list(torch.randn(3, 4, 2, 8))
    hidden_states = torch.cat(chunks, dim=0)
    labels = torch.randint(1, 5, (2, 4))
    loss_mask = torch.ones(2, 4)

    weight = torch.randn(8, requires_grad=True)
    output = process_mtp_loss(
        hidden_states,
        labels,
        loss_mask,
        _output_layer,
        weight,
        None,
        True,
        _compute_language_model_loss,
        config,
    )
    MTPLossAutoScaler.main_loss_backward_scale = torch.tensor(2.0)
    output.sum().backward()

    expected = _expected_weight_grad(
        chunks, labels, loss_mask, 0.5, 2, main_scale=2.0, per_token=False
    )
    assert output.shape == (4, 2, 8)
    torch.testing.assert_close(weight.grad, expected, rtol=1e-5, atol=1e-6)


def test_process_mtp_loss_per_token_loss_normalizes_by_original_tokens(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    config = _config(calculate_per_token_loss=True)
    torch.manual_seed(1)
    chunks = list(torch.randn(3, 4, 2, 8))
    hidden_states = torch.cat(chunks, dim=0)
    labels = torch.randint(1, 5, (2, 4))
    loss_mask = torch.ones(2, 4)

    weight = torch.randn(8, requires_grad=True)
    output = process_mtp_loss(
        hidden_states,
        labels,
        loss_mask,
        lambda states, weight=None, runtime_gather_output=None: (states * weight, None),
        weight,
        None,
        True,
        _compute_language_model_loss,
        config,
    )
    output.sum().backward()

    expected = _expected_weight_grad(
        chunks, labels, loss_mask, 0.5, 2, main_scale=1.0, per_token=True
    )
    torch.testing.assert_close(weight.grad, expected, rtol=1e-5, atol=1e-6)


def test_mtp_loss_logging_helper_reduce_and_track_metrics(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    from megatron.core import mpu

    dp_group = mpu.get_data_parallel_group(with_context_parallel=True)

    MTPLossLoggingHelper.reduce_loss_in_tracker()
    assert MTPLossLoggingHelper.tracker == {}

    MTPLossLoggingHelper.save_loss_to_tracker(
        loss=torch.tensor(1.5), layer_number=0, num_layers=2, reduce_group=dp_group
    )
    MTPLossLoggingHelper.save_loss_to_tracker(
        loss=torch.tensor(2.5), layer_number=1, num_layers=2, avg_group=dp_group
    )
    MTPLossLoggingHelper.reduce_loss_in_tracker()
    values = MTPLossLoggingHelper.tracker["values"]
    torch.testing.assert_close(values, torch.tensor([1.5, 2.5], device=values.device))

    total_loss_dict = {}
    MTPLossLoggingHelper.track_mtp_metrics(
        loss_scale=2.0, iteration=7, writer=None, total_loss_dict=total_loss_dict
    )
    assert total_loss_dict["mtp_1 loss"] == torch.tensor(3.0)
    assert total_loss_dict["mtp_2 loss"] == torch.tensor(5.0)
    values = MTPLossLoggingHelper.tracker["values"]
    assert torch.equal(values, torch.zeros_like(values))
    assert MTPLossLoggingHelper.tracker["reduce_group"] is None


def test_mtp_loss_logging_helper_skips_without_saved_values(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    MTPLossLoggingHelper.save_loss_to_tracker(
        loss=torch.tensor(1.0), layer_number=None, num_layers=2
    )
    assert MTPLossLoggingHelper.tracker == {}

    MTPLossLoggingHelper.reduce_loss_in_tracker()
    MTPLossLoggingHelper.track_mtp_metrics(loss_scale=1.0, iteration=1, writer=None)
    assert MTPLossLoggingHelper.tracker == {}


def test_roll_tensor_exchanges_boundaries_across_cp_ranks(model_parallel):
    model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    cp_group = get_context_parallel_group()
    cp_rank = torch.distributed.get_rank(group=cp_group)
    full = torch.arange(2, 18, dtype=torch.float32, device="cuda").view(1, 16)
    if cp_rank == 0:
        local = full[:, [0, 1, 2, 3, 12, 13, 14, 15]].contiguous()
        expected = full[:, [1, 2, 3, 4, 13, 14, 15, 0]].clone()
        expected[:, -1] = 0
    else:
        local = full[:, [4, 5, 6, 7, 8, 9, 10, 11]].contiguous()
        expected = full[:, [5, 6, 7, 8, 9, 10, 11, 12]].contiguous()

    rolled, rolled_sum = roll_tensor(local, shifts=-1, dims=-1, cp_group=cp_group)

    torch.testing.assert_close(rolled, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(rolled_sum, expected.sum(), rtol=0.0, atol=0.0)


def test_roll_tensor_packed_seq_skips_empty_sequence(model_parallel):
    model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    cp_group = get_context_parallel_group()
    cp_rank = torch.distributed.get_rank(group=cp_group)
    cu_seqlens = torch.tensor([0, 8, 9, 13], dtype=torch.int32, device="cuda")
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=8,
        max_seqlen_kv=8,
        qkv_format="thd",
    )
    sequence = torch.arange(13, dtype=torch.float32, device="cuda")
    if cp_rank == 0:
        local = sequence[[0, 1, 6, 7, 9, 12]].unsqueeze(0)
        expected = torch.tensor([[1.0, 2.0, 7.0, 0.0, 10.0, 0.0]], device="cuda")
    else:
        local = sequence[[2, 3, 4, 5, 10, 11]].unsqueeze(0)
        expected = torch.tensor([[3.0, 4.0, 5.0, 6.0, 11.0, 12.0]], device="cuda")

    rolled, rolled_sum = roll_tensor(
        local, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
    )

    torch.testing.assert_close(rolled, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(rolled_sum, expected.sum(), rtol=0.0, atol=0.0)


def test_get_mtp_layer_spec_uses_local_backend():
    spec = get_mtp_layer_spec(get_gpt_layer_local_spec(), use_transformer_engine=False)

    assert spec.module is MultiTokenPredictionLayer
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    assert spec.submodules.eh_proj is ColumnParallelLinear


def test_get_mtp_ranks_without_layout_or_mtp_layers():
    config = _config(mtp_num_layers=None)
    assert get_mtp_ranks([0, 1, 2], config) == []

    config = _config()
    assert get_mtp_ranks([3, 5, 7], config) == [7]


def test_get_mtp_ranks_collects_layout_stages_with_mtp():
    layout = PipelineParallelLayerLayout(
        [["embedding", "decoder", "decoder"], ["mtp", "mtp", "loss"]],
        pipeline_model_parallel_size=2,
    )
    config = _config(pipeline_model_parallel_layout=layout)

    assert get_mtp_ranks([0, 1], config) == [1]


def test_mtp_on_this_rank_with_layout_reports_own_stage(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    layout = PipelineParallelLayerLayout(
        [["embedding", "decoder", "decoder"], ["mtp", "mtp", "loss"]],
        pipeline_model_parallel_size=2,
    )

    on_this_rank = mtp_on_this_rank(layout=layout)

    assert on_this_rank == (get_pipeline_model_parallel_rank() == 1)


def test_mtp_on_this_rank_with_layout_spans_virtual_stages(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    layout = PipelineParallelLayerLayout(
        [["embedding", "decoder"], ["decoder", "mtp", "mtp", "loss"]],
        pipeline_model_parallel_size=1,
    )

    assert mtp_on_this_rank(layout=layout) is True


def test_get_mtp_num_layers_to_build_zero_on_non_last_stage(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)

    built = get_mtp_num_layers_to_build(_config())

    assert built == (2 if get_pipeline_model_parallel_rank() == 1 else 0)


def test_tie_output_layer_state_dict_marks_post_process_replica(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    from megatron.core import mpu
    from megatron.core.utils import get_pg_rank, make_tp_sharded_tensor_for_checkpoint

    key = "mtp.layers.0.final_logit_bias"
    weight = torch.randn(4, 8)
    sharded_state_dict = {
        key: make_tp_sharded_tensor_for_checkpoint(
            weight, key, replica_id=(0, 0, 0), tp_group=mpu.get_tensor_model_parallel_group()
        )
    }

    tie_output_layer_state_dict(
        sharded_state_dict,
        weight,
        key,
        mpu.get_tensor_model_parallel_group(),
        mpu.get_data_parallel_group(with_context_parallel=True),
    )

    tied = sharded_state_dict[key]
    assert tied.replica_id == (
        1,
        0,
        get_pg_rank(mpu.get_data_parallel_group(with_context_parallel=True)),
    )
    assert tied.data is weight
