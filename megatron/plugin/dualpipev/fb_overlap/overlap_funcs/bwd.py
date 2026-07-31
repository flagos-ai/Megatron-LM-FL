# Mainly adopted from https://gitee.com/ascend/MindSpeed/blob/master/mindspeed/core/transformer/moe/moe_feature/fb_overlap/overlap_funcs/bwd.py

import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import permute
from megatron.plugin.dualpipev.fb_overlap.modules.utils import (
    async_all_to_all,
    call_attention_backward_dw,
    call_dense_mlp_backward_dw,
    call_experts_backward_dw,
    call_shared_experts_backward_dw,
    run_graph_backward,
    turn_attention_delay_wgrad_compute,
    turn_dense_mlp_delay_wgrad_compute,
    turn_experts_delay_wgrad_compute,
    turn_shared_experts_delay_wgrad_compute,
)
from megatron.plugin.dualpipev.observability import wait_async_all_to_all


def transformer_layer_backward_moe(layer_output_grad, layer_graph):
    """Backward function of transformer layer, for moe models"""
    self = layer_graph
    dispached_input, fc1_out, act_out, probs, indices, global_input_tokens_local_experts_indices = (
        self.recompute_needed_tensors
    )
    ep_group = parallel_state.get_expert_model_parallel_group()

    backward_shared = (
        layer_output_grad if layer_output_grad is not None else self.unperm2_graph[1].grad
    )

    run_graph_backward(self.unperm2_graph, layer_output_grad, keep_grad=True)

    _, unperm1_out_grad, bwd_unperm_a2a_handle = async_all_to_all(
        self.unperm_a2a_graph[1].grad,
        self.output_splits,
        self.input_splits,
        ep_group,
        trace_owner=self.layer,
        pass_direction="backward",
        logical_phase="combine",
        payload_role="expert_output_gradient",
    )
    # overlap alltoall by shared experts backward
    if self.shared_experts_graph[0] is not None:
        turn_shared_experts_delay_wgrad_compute(self, enable=True)
        run_graph_backward(self.shared_experts_graph, backward_shared, keep_grad=True)
        call_shared_experts_backward_dw(self)
        turn_shared_experts_delay_wgrad_compute(self, enable=False)

    wait_async_all_to_all(bwd_unperm_a2a_handle, completion_site="backward_combine_gradient_ready")
    bwd_unperm_a2a_handle = None

    run_graph_backward(self.unperm1_graph, unperm1_out_grad)

    turn_experts_delay_wgrad_compute(self, enable=True)
    run_graph_backward(self.grouped_mlp_graph, keep_grad=True)  # keep for dw commputation

    run_graph_backward(self.perm2_graph, keep_graph=True)  # keep for dw commutation
    run_graph_backward(self.perm2_append_graph, keep_graph=True)

    _, perm1_out1_grad, bwd_perm_a2a_handle1 = async_all_to_all(
        self.perm_a2a_graph[1].grad,
        self.input_splits,
        self.output_splits,
        ep_group,
        trace_owner=self.layer,
        pass_direction="backward",
        logical_phase="dispatch",
        payload_role="token_hidden_states_gradient",
    )

    _, perm1_out2_grad, bwd_perm_a2a_handle2 = async_all_to_all(
        self.perm_a2a_append_graph[1].grad,
        self.input_splits,
        self.output_splits,
        ep_group,
        trace_owner=self.layer,
        pass_direction="backward",
        logical_phase="dispatch",
        payload_role="routing_probabilities_gradient",
    )

    # dw computation
    call_experts_backward_dw(self)
    turn_experts_delay_wgrad_compute(self, enable=False)

    wait_async_all_to_all(
        bwd_perm_a2a_handle1, completion_site="backward_dispatch_hidden_gradient_ready"
    )
    bwd_perm_a2a_handle1 = None
    wait_async_all_to_all(
        bwd_perm_a2a_handle2, completion_site="backward_dispatch_probability_gradient_ready"
    )
    bwd_perm_a2a_handle2 = None

    run_graph_backward(self.perm1_graph, perm1_out1_grad)
    run_graph_backward(self.perm1_append_graph, perm1_out2_grad)
    perm1_out1_grad.untyped_storage().resize_(0)
    perm1_out2_grad.untyped_storage().resize_(0)
    run_graph_backward(self.router_graph)
    run_graph_backward(self.pre_mlp_layernorm_graph)

    turn_attention_delay_wgrad_compute(self, enable=True)
    run_graph_backward(self.attn_graph, keep_grad=True)
    call_attention_backward_dw(self)
    turn_attention_delay_wgrad_compute(self, enable=False)

    self.recompute_needed_tensors = [None for _ in range(len(self.recompute_needed_tensors))]

    return self.layer_input.grad


def transformer_layer_backward_dense(layer_output_grad, layer_graph):
    """Backward function of transformer layer, for dense models"""
    turn_dense_mlp_delay_wgrad_compute(layer_graph, enable=True)
    run_graph_backward(layer_graph.unperm2_graph, layer_output_grad, keep_grad=True)
    run_graph_backward(layer_graph.pre_mlp_layernorm_graph)
    call_dense_mlp_backward_dw(layer_graph)
    turn_dense_mlp_delay_wgrad_compute(layer_graph, enable=False)

    turn_attention_delay_wgrad_compute(layer_graph, enable=True)
    run_graph_backward(layer_graph.attn_graph, keep_grad=True)
    call_attention_backward_dw(layer_graph)
    turn_attention_delay_wgrad_compute(layer_graph, enable=False)
    return layer_graph.layer_input.grad
