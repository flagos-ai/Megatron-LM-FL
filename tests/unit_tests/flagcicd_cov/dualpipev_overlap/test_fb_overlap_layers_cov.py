# Copyright (c) 2026, BAAI. All rights reserved.
"""fb_overlap TE layers, blocks and GPT model: overlapped runs match the ordinary modules."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.plugin.dualpipev.fb_overlap.gpt_model import (
    gpt_model_backward,
    gpt_model_forward,
    gpt_model_forward_backward_overlapping,
)
from megatron.plugin.dualpipev.fb_overlap.modules import utils
from megatron.plugin.dualpipev.fb_overlap.transformer_block import (
    transformer_block_backward,
    transformer_block_forward,
    transformer_block_forward_backward_overlapping,
)
from megatron.plugin.dualpipev.fb_overlap.transformer_layer import (
    transformer_layer_backward,
    transformer_layer_forward,
    transformer_layer_forward_backward_overlapping,
)
from megatron.training import global_vars
from tests.unit_tests.test_utilities import Utils

SEQ_LEN = 32
BATCH_SIZE = 2
HIDDEN_SIZE = 64
VOCAB_SIZE = 128
RTOL = 1e-4
ATOL = 1e-5


@pytest.fixture
def model_parallel(monkeypatch):
    monkeypatch.setattr(utils, 'COMM_STREAM', None)
    monkeypatch.setattr(global_vars, '_GLOBAL_ARGS', SimpleNamespace())
    Utils.initialize_model_parallel(expert_model_parallel_size=8)
    yield
    Utils.destroy_model_parallel()


def _config(num_layers):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=128,
        num_attention_heads=4,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        attention_backend=AttnBackend.unfused,
        delay_wgrad_compute=True,
        gradient_accumulation_fusion=False,
        bf16=True,
        overlap_moe_expert_parallel_comm=True,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_token_dispatcher_type='alltoall',
        moe_router_load_balancing_type='none',
        moe_grouped_gemm=True,
        moe_shared_expert_intermediate_size=64,
        expert_model_parallel_size=8,
    )


def _layer_spec(kind):
    return get_gpt_layer_with_transformer_engine_spec(
        num_experts=8 if kind == 'moe' else None, moe_grouped_gemm=True
    )


def _reference_pair(build):
    model_parallel_cuda_manual_seed(123)
    layer = build().cuda()
    reference = build().cuda()
    reference.load_state_dict(layer.state_dict())
    for module in reference.modules():
        if getattr(module, 'wgrad_store', None):
            module.wgrad_store.disable_delay_wgrad_compute()
    return layer, reference


def _build_layers(kind):
    def build():
        return build_module(_layer_spec(kind), config=_config(num_layers=1), layer_number=1)

    return _reference_pair(build)


def _build_blocks(kind, post_process):
    def build():
        return TransformerBlock(
            config=_config(num_layers=2),
            spec=_layer_spec(kind),
            pre_process=True,
            post_process=post_process,
        )

    return _reference_pair(build)


def _build_gpt_models(post_process):
    def build():
        return GPTModel(
            config=_config(num_layers=2),
            transformer_layer_spec=_layer_spec('dense'),
            vocab_size=VOCAB_SIZE,
            max_sequence_length=SEQ_LEN,
            pre_process=True,
            post_process=post_process,
        )

    return _reference_pair(build)


def _bind_overlap_forward(block, monkeypatch):
    for layer in block.layers:
        monkeypatch.setattr(
            layer, 'forward', transformer_layer_forward.__get__(layer, TransformerLayer)
        )


def _input_tensor(seed):
    torch.manual_seed(seed)
    return torch.randn(
        SEQ_LEN,
        BATCH_SIZE,
        HIDDEN_SIZE,
        dtype=torch.float32,
        device=torch.cuda.current_device(),
        requires_grad=True,
    )


def _model_inputs():
    torch.manual_seed(3)
    device = torch.cuda.current_device()
    input_ids = torch.randint(
        0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.int64, device=device
    )
    position_ids = torch.arange(SEQ_LEN, dtype=torch.int64, device=device).repeat(BATCH_SIZE, 1)
    return input_ids, position_ids


def assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def assert_same_gradients(layer, reference):
    for (name, param), (_, reference_param) in zip(
        layer.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(param.grad, reference_param.grad, rtol=RTOL, atol=ATOL, msg=name)


def test_dense_layer_forward_matches_layer_forward(model_parallel):
    layer, reference = _build_layers('dense')
    reference_output = reference(_input_tensor(0), attention_mask=None)[0]
    output, _, graph = transformer_layer_forward(layer, _input_tensor(0), None)
    assert_close(output, reference_output)
    assert graph.is_moe_layer is False


def test_dense_layer_backward_matches_autograd(model_parallel):
    layer, reference = _build_layers('dense')
    reference_hidden = _input_tensor(0)
    reference_output = reference(reference_hidden, attention_mask=None)[0]
    grad = _input_tensor(1)
    torch.autograd.backward(reference_output, grad_tensors=grad)
    _, _, graph = transformer_layer_forward(layer, _input_tensor(0), None)
    input_grad = transformer_layer_backward(grad, graph)
    assert_close(input_grad, reference_hidden.grad)
    assert_same_gradients(layer, reference)


def test_moe_layer_forward_matches_layer_forward(model_parallel):
    layer, reference = _build_layers('moe')
    reference_output = reference(_input_tensor(0), attention_mask=None)[0]
    output, _, graph = transformer_layer_forward(layer, _input_tensor(0), None)
    assert_close(output, reference_output)
    assert graph.is_moe_layer is True


def test_moe_layer_backward_matches_autograd(model_parallel):
    layer, reference = _build_layers('moe')
    reference_hidden = _input_tensor(0)
    reference_output = reference(reference_hidden, attention_mask=None)[0]
    grad = _input_tensor(1)
    torch.autograd.backward(reference_output, grad_tensors=grad)
    _, _, graph = transformer_layer_forward(layer, _input_tensor(0), None)
    input_grad = transformer_layer_backward(grad, graph)
    assert_close(input_grad, reference_hidden.grad)
    assert_same_gradients(layer, reference)


@pytest.mark.parametrize(
    ('fwd_kind', 'bwd_kind'),
    [
        pytest.param('dense', 'dense', id='fwd-dense-bwd-dense'),
        pytest.param('dense', 'moe', id='fwd-dense-bwd-moe'),
        pytest.param('moe', 'dense', id='fwd-moe-bwd-dense'),
        pytest.param('moe', 'moe', id='fwd-moe-bwd-moe'),
    ],
)
def test_overlapped_forward_and_backward_match_separate_runs(model_parallel, fwd_kind, bwd_kind):
    fwd_layer, fwd_reference = _build_layers(fwd_kind)
    bwd_layer, bwd_reference = _build_layers(bwd_kind)
    reference_hidden = _input_tensor(0)
    reference_output = bwd_reference(reference_hidden, attention_mask=None)[0]
    grad = _input_tensor(1)
    torch.autograd.backward(reference_output, grad_tensors=grad)
    _, _, bwd_graph = transformer_layer_forward(bwd_layer, _input_tensor(0), None)
    fwd_output, _, _, _, comm_output = transformer_layer_forward_backward_overlapping(
        fwd_layer, _input_tensor(2), None, bwd_layer_output_grad=grad, bwd_layer_graph=bwd_graph
    )
    assert_close(bwd_graph.layer_input.grad, reference_hidden.grad)
    assert_close(comm_output.input_tensor_grad, reference_hidden.grad)
    assert_same_gradients(bwd_layer, bwd_reference)
    reference_fwd_output = fwd_reference(_input_tensor(2), attention_mask=None)[0]
    assert_close(fwd_output, reference_fwd_output)


def test_checkpointed_forward_records_inputs_and_recomputes_on_backward(model_parallel):
    layer, reference = _build_layers('dense')
    reference_hidden = _input_tensor(0)
    reference_output = reference(reference_hidden, attention_mask=None)[0]
    grad = _input_tensor(1)
    torch.autograd.backward(reference_output, grad_tensors=grad)
    output, _, graph = transformer_layer_forward_backward_overlapping(
        layer, _input_tensor(0), None, checkpoint=True
    )
    assert graph.checkpointed is True
    assert graph.layer_inputs[0] is None
    input_grad = transformer_layer_backward(grad, graph)
    assert_close(output, reference_output)
    assert_close(input_grad, reference_hidden.grad)
    assert_same_gradients(layer, reference)


@pytest.mark.parametrize(
    'block_kind', [pytest.param('dense', id='dense'), pytest.param('moe', id='moe')]
)
def test_block_forward_and_backward_match_ordinary_block(model_parallel, monkeypatch, block_kind):
    block, reference = _build_blocks(block_kind, post_process=False)
    _bind_overlap_forward(block, monkeypatch)
    reference_hidden = _input_tensor(0)
    reference_output = reference(reference_hidden, attention_mask=None)
    grad = _input_tensor(1)
    torch.autograd.backward(reference_output, grad_tensors=grad)
    output, layer_graphs = transformer_block_forward(
        block, hidden_states=_input_tensor(0), attention_mask=None
    )
    assert_close(output, reference_output)
    input_grad = transformer_block_backward(grad, layer_graphs)
    assert_close(input_grad, reference_hidden.grad)
    assert_same_gradients(block, reference)


@pytest.mark.parametrize(
    'block_kind', [pytest.param('dense', id='dense'), pytest.param('moe', id='moe')]
)
def test_block_overlap_forward_and_backward_match_separate_runs(
    model_parallel, monkeypatch, block_kind
):
    fwd_block, fwd_reference = _build_blocks(block_kind, post_process=True)
    bwd_block, bwd_reference = _build_blocks(block_kind, post_process=False)
    _bind_overlap_forward(bwd_block, monkeypatch)
    reference_hidden = _input_tensor(0)
    reference_output = bwd_reference(reference_hidden, attention_mask=None)
    grad = _input_tensor(1)
    torch.autograd.backward(reference_output, grad_tensors=grad)
    _, bwd_layer_graphs = transformer_block_forward(
        bwd_block, hidden_states=_input_tensor(0), attention_mask=None
    )
    (fwd_output, _), bwd_input_grad, _ = transformer_block_forward_backward_overlapping(
        fwd_block, _input_tensor(2), None, grad, bwd_layer_graphs
    )
    assert_close(bwd_input_grad, reference_hidden.grad)
    assert_same_gradients(bwd_block, bwd_reference)
    reference_fwd_output = fwd_reference(_input_tensor(2), attention_mask=None)
    assert_close(fwd_output, reference_fwd_output)


def test_gpt_model_forward_returns_reference_logits(model_parallel, monkeypatch):
    model, reference = _build_gpt_models(post_process=True)
    _bind_overlap_forward(model.decoder, monkeypatch)
    input_ids, position_ids = _model_inputs()
    reference_logits = reference(input_ids, position_ids, None)
    logits, graph = gpt_model_forward(model, input_ids, position_ids, None)
    assert_close(logits, reference_logits)
    assert [layer_graph.layer.layer_number for layer_graph in graph.layer_graphs] == [1, 2]
    assert graph.preprocess_graph[0].shape == (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)


def test_gpt_model_backward_matches_autograd(model_parallel, monkeypatch):
    model, reference = _build_gpt_models(post_process=False)
    _bind_overlap_forward(model.decoder, monkeypatch)
    input_ids, position_ids = _model_inputs()
    reference_hidden = reference(input_ids, position_ids, None)
    grad = _input_tensor(1)
    torch.autograd.backward(reference_hidden, grad_tensors=grad)
    hidden_states, graph = gpt_model_forward(model, input_ids, position_ids, None)
    assert_close(hidden_states, reference_hidden)
    returned = gpt_model_backward(grad, graph)
    assert returned is None
    assert_same_gradients(model, reference)


def test_gpt_model_overlap_without_bwd_graph_forwards_normally(model_parallel, monkeypatch):
    model, reference = _build_gpt_models(post_process=True)
    _bind_overlap_forward(model.decoder, monkeypatch)
    input_ids, position_ids = _model_inputs()
    reference_logits = reference(input_ids, position_ids, None)
    logits, graph = gpt_model_forward_backward_overlapping(model, input_ids, position_ids, None)
    assert_close(logits, reference_logits)
    assert [layer_graph.layer.layer_number for layer_graph in graph.layer_graphs] == [1, 2]


def test_gpt_model_overlap_forward_and_backward_match_separate_runs(model_parallel, monkeypatch):
    fwd_model, fwd_reference = _build_gpt_models(post_process=True)
    bwd_model, bwd_reference = _build_gpt_models(post_process=False)
    _bind_overlap_forward(bwd_model.decoder, monkeypatch)
    input_ids, position_ids = _model_inputs()
    reference_hidden = bwd_reference(input_ids, position_ids, None)
    grad = _input_tensor(1)
    torch.autograd.backward(reference_hidden, grad_tensors=grad)
    bwd_hidden_states, bwd_graph = gpt_model_forward(bwd_model, input_ids, position_ids, None)
    assert_close(bwd_hidden_states, reference_hidden)
    logits, _, pp_comm_output = gpt_model_forward_backward_overlapping(
        fwd_model,
        input_ids,
        position_ids,
        None,
        extra_block_kwargs={
            'bwd_model_grad': grad,
            'bwd_model_graph': bwd_graph,
            'pp_comm_params': None,
            'bwd_pp_comm_params': None,
        },
    )
    reference_logits = fwd_reference(input_ids, position_ids, None)
    assert_close(logits, reference_logits)
    assert_same_gradients(bwd_model, bwd_reference)
