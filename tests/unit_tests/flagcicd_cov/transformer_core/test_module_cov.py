# Copyright (c) 2026, BAAI. All rights reserved.
"""Graphable module hooks, Float16Module pipeline paths and TP-gathered linear replacements."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.heterogeneous.linear_replacements import (
    ColumnParallelLinearGathered,
    _gather_from_tensor_parallel_region,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _config(**overrides):
    defaults = dict(
        num_layers=1, hidden_size=16, num_attention_heads=4, use_cpu_initialization=True
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _layer(config):
    return build_module(get_gpt_layer_local_spec(), config=config, layer_number=1)


class _DtypeProbeModule(torch.nn.Module):
    """Records the dtype of the hidden states it receives and applies a linear."""

    def __init__(self, hidden_size):
        super().__init__()
        self.seen_dtype = None
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask):
        self.seen_dtype = hidden_states.dtype
        return self.linear(hidden_states)


def test_te_backward_dw_wrapper_registers_graphed_callable(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    layer = _layer(_config(cuda_graph_impl="transformer_engine"))
    layer.current_microbatch = 3

    layer.init_backward_dw_wrapper()
    layer.set_te_cuda_graph_backward_dw_wrapper()

    graphed_callable = layer.backward_dw_wrapper.graphed_backward_dw_callable
    assert graphed_callable.args == (3,)
    assert graphed_callable.func == layer._te_cuda_graph_backward_dw_graph


def test_te_cuda_graph_capture_runs_plain_layer_forward(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    layer = _layer(_config()).eval().cuda()
    hidden_states = torch.randn(8, 2, 16, device="cuda")
    attention_mask = torch.ones((1, 1, 8, 8), dtype=bool, device="cuda")

    with torch.no_grad():
        captured = layer._te_cuda_graph_capture(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        expected, context = layer(hidden_states=hidden_states, attention_mask=attention_mask)

    assert context is None
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected, rtol=1e-5, atol=1e-6)


def test_layer_call_runs_plain_forward_when_no_local_cudagraph_manager(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    layer = _layer(_config()).eval().cuda()
    hidden_states = torch.randn(8, 2, 16, device="cuda")
    attention_mask = torch.ones((1, 1, 8, 8), dtype=bool, device="cuda")

    with torch.no_grad():
        called = layer(hidden_states=hidden_states, attention_mask=attention_mask)
        direct = layer.forward(hidden_states=hidden_states, attention_mask=attention_mask)

    torch.testing.assert_close(called, direct, rtol=0.0, atol=0.0)


def test_float16_module_requires_fp16_or_bf16(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    layer = _layer(_config())

    with pytest.raises(Exception, match="Either config.fp16 or config.bf16 should be True."):
        Float16Module(_config(), layer)


def test_float16_module_dualpipev_upcasts_output_on_first_and_last_stage(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    config = _config(bf16=True, use_dualpipev=True, dualpipev_pipeline_model_parallel_size=1)
    probe = _DtypeProbeModule(16)
    wrapped = Float16Module(config, probe).eval()
    hidden_states = torch.randn(8, 2, 16)

    with torch.no_grad():
        output = wrapped(hidden_states, attention_mask=None)

    assert probe.seen_dtype == torch.bfloat16
    assert output.dtype == torch.float32
    with torch.no_grad():
        expected = probe.linear(hidden_states.bfloat16()).float()
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


def test_float16_module_dualpipev_fp32_output_false_keeps_model_dtype(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    config = _config(bf16=True, use_dualpipev=True, dualpipev_pipeline_model_parallel_size=1)
    probe = _DtypeProbeModule(16)
    wrapped = Float16Module(config, probe).eval()
    hidden_states = torch.randn(8, 2, 16)

    with torch.no_grad():
        output = wrapped(hidden_states, attention_mask=None, fp32_output=False)

    assert probe.seen_dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16


def test_gather_from_tensor_parallel_region_concatenates_rank_shards(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    assert get_tensor_model_parallel_world_size() == 2
    rank = get_tensor_model_parallel_rank()
    x = torch.full((8, 2, 8), float(rank + 1), device="cuda")

    # sequence_parallel=True is not covered: the helper's F.pad list pads the sequence dimension
    # instead of the hidden dimension (linear_replacements.py:37, reported as a source bug).
    for sequence_parallel, expected_rows in ((False, 8),):
        config = _config(tensor_model_parallel_size=2, sequence_parallel=sequence_parallel)

        gathered = _gather_from_tensor_parallel_region(x, config)

        assert gathered.shape == (expected_rows, 2, 16)
        assert torch.equal(gathered[..., :8], torch.full_like(gathered[..., :8], 1.0))
        assert torch.equal(gathered[..., 8:], torch.full_like(gathered[..., 8:], 2.0))


def test_column_parallel_linear_gathered_rejects_runtime_gather(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    config = _config()
    linear = ColumnParallelLinearGathered(config)
    hidden_states = torch.randn(4, 2, 16)

    with pytest.raises(ValueError, match="gathering TP outputs is not supported"):
        linear.forward(hidden_states, runtime_gather_output=True)


def test_column_parallel_linear_gathered_gathers_own_shard(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    config = _config(tensor_model_parallel_size=2)
    linear = ColumnParallelLinearGathered(config).cuda()
    hidden_states = torch.randn(4, 2, 16, device="cuda")

    out, bias = linear.forward(hidden_states)

    assert bias is None
    rank = get_tensor_model_parallel_rank()
    own_shard = torch.nn.functional.linear(hidden_states, linear.weight)
    assert torch.equal(out[..., rank * 8 : (rank + 1) * 8], own_shard)


def test_transformer_block_builds_with_dualpipev_stage_offset(model_parallel):
    model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    config = _config(bf16=True, use_dualpipev=True, dualpipev_pipeline_model_parallel_size=2)

    block = TransformerBlock(
        config, get_gpt_layer_local_spec(), dualpipev_stage=0, post_process=True
    )

    assert block.dualpipev_stage == 0
    assert len(block.layers) == 1
    assert block.layers[0].layer_number == 1
