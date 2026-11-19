# Copyright (c) 2026, BAAI. All rights reserved.
"""fb_overlap graph bookkeeping: detached inputs, graph backward, delay-wgrad switches, p2p params."""

import pytest
import torch

from megatron.plugin.dualpipev.fb_overlap.modules import utils


class _WgradStore:
    def __init__(self, log, name):
        self.log = log
        self.name = name

    def enable_delay_wgrad_compute(self):
        self.log.append(('enable', self.name))

    def disable_delay_wgrad_compute(self):
        self.log.append(('disable', self.name))


class _Linear(torch.nn.Module):
    """A leaf module exposing the two hooks the switches call: wgrad_store and backward_dw."""

    def __init__(self, log, name):
        super().__init__()
        self.wgrad_store = _WgradStore(log, name)
        self.log = log
        self.name = name

    def backward_dw(self):
        self.log.append(('backward_dw', self.name))


def _container(log, prefix, names):
    module = torch.nn.Module()
    for name in names:
        setattr(module, name, _Linear(log, f'{prefix}.{name}'))
    return module


def _layer_graph(log, attention_names, mlp=None):
    """A checkpointed LayerGraph around a module tree; the switches only read `.layer`."""
    layer = torch.nn.Module()
    layer.self_attention = _container(log, 'attn', attention_names)
    layer.mlp = mlp if mlp is not None else _container(log, 'mlp', ['linear_fc1', 'linear_fc2'])
    return utils.LayerGraph([None, None], None, None, None, layer, checkpointed=True)


def _moe_mlp(log):
    mlp = torch.nn.Module()
    mlp.experts = _container(log, 'experts', ['linear_fc1', 'linear_fc2'])
    mlp.shared_experts = _container(log, 'shared', ['linear_fc1', 'linear_fc2'])
    return mlp


SELF_ATTENTION = ['linear_proj', 'linear_qkv']
LATENT_ATTENTION = ['linear_proj', 'linear_kv_up_proj', 'linear_kv_down_proj', 'linear_q_proj']


def test_detach_tensor_returns_input_when_checkpointing_forward():
    tensor = torch.ones(2)
    assert utils.detach_tensor(tensor, checkpoint_forward=True) is tensor


def test_detach_tensor_passes_none_through():
    assert utils.detach_tensor(None) is None


def test_detach_tensor_makes_a_grad_leaf_sharing_storage():
    source = torch.ones(2, requires_grad=True)
    output = source * 2
    detached = utils.detach_tensor(output)
    assert detached.requires_grad
    assert detached.is_leaf
    assert detached.grad_fn is None
    assert detached.data_ptr() == output.data_ptr()


def test_graph_backward_uses_given_grad_and_frees_both_buffers():
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    output = source * 3
    grad = torch.tensor([1.0, 1.0])
    utils.run_graph_backward((output, None), grad)
    torch.testing.assert_close(source.grad, torch.tensor([3.0, 3.0]))
    assert output.untyped_storage().size() == 0
    assert grad.untyped_storage().size() == 0


def test_graph_backward_falls_back_to_detached_output_grad_and_keeps_buffers():
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    output = source * 3
    detached = torch.zeros(2, requires_grad=True)
    detached.grad = torch.tensor([2.0, 4.0])
    utils.run_graph_backward((output, detached), keep_graph=True, keep_grad=True)
    torch.testing.assert_close(source.grad, torch.tensor([6.0, 12.0]))
    assert output.untyped_storage().size() == 8
    assert detached.grad.untyped_storage().size() == 8


@pytest.mark.parametrize(
    ('attention_names', 'enable'),
    [
        pytest.param(SELF_ATTENTION, True, id='self-attention-enable'),
        pytest.param(SELF_ATTENTION, False, id='self-attention-disable'),
        pytest.param(LATENT_ATTENTION, True, id='latent-attention-enable'),
        pytest.param(LATENT_ATTENTION, False, id='latent-attention-disable'),
    ],
)
def test_attention_delay_wgrad_switch_covers_attention_projections(attention_names, enable):
    log = []
    utils.turn_attention_delay_wgrad_compute(_layer_graph(log, attention_names), enable=enable)
    action = 'enable' if enable else 'disable'
    assert log == [(action, f'attn.{name}') for name in attention_names]


@pytest.mark.parametrize(
    'attention_names',
    [
        pytest.param(SELF_ATTENTION, id='self-attention'),
        pytest.param(LATENT_ATTENTION, id='latent-attention'),
    ],
)
def test_attention_backward_dw_runs_every_attention_projection(attention_names):
    log = []
    utils.call_attention_backward_dw(_layer_graph(log, attention_names))
    assert log == [('backward_dw', f'attn.{name}') for name in attention_names]


@pytest.mark.parametrize(
    ('switch', 'prefix', 'enable'),
    [
        pytest.param('turn_dense_mlp_delay_wgrad_compute', 'mlp', True, id='dense-enable'),
        pytest.param('turn_dense_mlp_delay_wgrad_compute', 'mlp', False, id='dense-disable'),
        pytest.param('turn_experts_delay_wgrad_compute', 'experts', True, id='experts-enable'),
        pytest.param('turn_experts_delay_wgrad_compute', 'experts', False, id='experts-disable'),
        pytest.param('turn_shared_experts_delay_wgrad_compute', 'shared', True, id='shared-enable'),
        pytest.param(
            'turn_shared_experts_delay_wgrad_compute', 'shared', False, id='shared-disable'
        ),
    ],
)
def test_mlp_delay_wgrad_switches_touch_fc2_then_fc1(switch, prefix, enable):
    log = []
    mlp = None if prefix == 'mlp' else _moe_mlp(log)
    getattr(utils, switch)(_layer_graph(log, SELF_ATTENTION, mlp), enable=enable)
    action = 'enable' if enable else 'disable'
    assert log == [(action, f'{prefix}.linear_fc2'), (action, f'{prefix}.linear_fc1')]


@pytest.mark.parametrize(
    ('call', 'prefix'),
    [
        pytest.param('call_dense_mlp_backward_dw', 'mlp', id='dense'),
        pytest.param('call_experts_backward_dw', 'experts', id='experts'),
        pytest.param('call_shared_experts_backward_dw', 'shared', id='shared'),
    ],
)
def test_mlp_backward_dw_runs_fc2_then_fc1(call, prefix):
    log = []
    mlp = None if prefix == 'mlp' else _moe_mlp(log)
    getattr(utils, call)(_layer_graph(log, SELF_ATTENTION, mlp))
    assert log == [('backward_dw', f'{prefix}.linear_fc2'), ('backward_dw', f'{prefix}.linear_fc1')]


def test_layer_graph_unpacks_saved_graphs_in_order():
    layer = torch.nn.Linear(1, 1)
    graph = utils.LayerGraph(list(range(15)), 'recompute', 'in', 'out', layer)
    assert (
        graph.attn_graph,
        graph.router_graph,
        graph.grouped_mlp_graph,
        graph.unperm2_graph,
        graph.shared_experts_graph,
    ) == (0, 2, 9, 12, 13)
    assert graph.layer_input == 14
    assert (graph.recompute_needed_tensors, graph.input_splits, graph.output_splits) == (
        'recompute',
        'in',
        'out',
    )
    assert graph.checkpointed is False
    assert graph.is_moe_layer is False


def test_checkpointed_layer_graph_keeps_only_the_input():
    graph = utils.LayerGraph(
        [None, 'layer-input'], None, None, None, torch.nn.Linear(1, 1), checkpointed=True
    )
    assert graph.unperm2_graph == (None, None)
    assert graph.layer_input == 'layer-input'
    assert not hasattr(graph, 'attn_graph')


def test_layer_graph_detects_moe_layers_and_records_inputs():
    log = []
    graph = _layer_graph(log, SELF_ATTENTION, _moe_mlp(log))
    graph.record_layer_inputs('mask', None)
    assert graph.is_moe_layer is True
    assert graph.layer_inputs == ('mask', None)


def test_model_graph_pairs_preprocess_graph_with_its_output():
    model_graph = utils.ModelGraph(['layer'], 'block', 'pre', 'pre-out')
    assert model_graph.preprocess_graph == ('pre', 'pre-out')
    assert model_graph.layer_graphs == ['layer']
    assert model_graph.block_output == 'block'
    assert utils.ModelGraph([], 'block').preprocess_graph == (None, None)


def test_p2p_params_describe_directions():
    params = utils.P2PCommParams(send_next=True)
    assert str(params) == 'send next:True send_prev:False recv_next:False recv_prev:False'


@pytest.mark.parametrize(
    ('params', 'needed'),
    [
        pytest.param(None, False, id='no-params'),
        pytest.param(utils.P2PCommParams(), False, id='no-direction'),
        pytest.param(utils.P2PCommParams(recv_prev=True), True, id='recv-prev'),
        pytest.param(utils.P2PCommParams(send_prev=True), True, id='send-prev'),
    ],
)
def test_p2p_comm_is_needed_only_for_some_direction(params, needed):
    assert utils.is_p2p_comm_needed(params) is needed


def test_p2p_comm_output_keeps_every_field():
    output = utils.P2PCommOutput(
        input_tensor=1,
        output_tensor_grad=2,
        fwd_wait_handles=3,
        bwd_wait_handles=4,
        input_tensor_grad=5,
    )
    assert (
        output.input_tensor,
        output.output_tensor_grad,
        output.fwd_wait_handles,
        output.bwd_wait_handles,
        output.input_tensor_grad,
    ) == (1, 2, 3, 4, 5)


@pytest.mark.parametrize(
    'params',
    [
        pytest.param(utils.P2PCommParams(send_next=True, send_prev=True), id='send-both'),
        pytest.param(utils.P2PCommParams(recv_next=True, recv_prev=True), id='recv-both'),
    ],
)
def test_p2p_helper_rejects_two_directions_at_once(params):
    with pytest.raises(AssertionError):
        utils.p2p_comm_helper(params, None)
