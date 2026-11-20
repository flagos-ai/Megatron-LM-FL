import logging
import os

import torch
import torch.distributed as dist

from megatron.core.tensor_parallel import all_to_all
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import permute, sort_chunks_by_idxs
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher


def _supported_dispatcher(dispatcher):
    config = dispatcher.config
    # Expert TP is independent of ordinary model TP.
    return (
        os.getenv("NPU_MOE_MC2", "0") == "1"
        and dispatcher.tp_size == 1
        and config.cuda_graph_impl == "none"
        and dispatcher.ep_size in (8, 16, 32, 64, 128)
        and dispatcher.num_local_experts <= 32
        and config.num_moe_experts <= 256
        and 2 <= config.moe_router_topk <= 8
        and config.moe_expert_capacity_factor is None
    )


def _supported(layer, x):
    dispatcher, config = layer.token_dispatcher, layer.config
    if x.device.type != "npu" or x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not isinstance(dispatcher, MoEAlltoAllTokenDispatcher) or not _supported_dispatcher(
        dispatcher
    ):
        return False
    if not isinstance(layer.experts, TEGroupedMLP) or layer.experts._with_fused_impl:
        return False
    if layer.experts.linear_fc1.single_grouped_weight or layer.experts.linear_fc2.single_grouped_weight:
        return False
    if dispatcher.ep_group._get_backend(x.device).name() not in ("hccl", "flagcx"):
        return False
    return not any(
        (
            config.fp8,
            config.fp4,
            config.cpu_offloading,
            config.add_bias_linear,
            config.gradient_accumulation_fusion,
            config.delay_wgrad_compute,
            config.overlap_dispatch_backward_with_experts_wgrad,
            config.moe_router_padding_for_quantization,
            layer.experts.offload_expert_fc1,
            layer.experts.offload_moe_act,
        )
    )


class _ReorderProbabilities(torch.autograd.Function):
    """Differentiate a bijective expert-column permutation with its inverse."""

    @staticmethod
    def forward(ctx, probs, columns, inverse):
        ctx.save_for_backward(inverse)
        return probs.index_select(1, columns)

    @staticmethod
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        return grad.index_select(1, inverse), None, None


def preprocess(self, hidden_states, probs, routing_map):
    """Enable round scheduling only after the whole layer and EP group agree."""
    dispatcher = self.token_dispatcher
    dispatcher._npu_moe_schedule = _supports_group(self, hidden_states)
    return MoELayer.preprocess.__wrapped__(self, hidden_states, probs, routing_map)


def dispatch_preprocess(self, hidden_states, routing_map, probs):
    """Write the first token permutation directly in native MC2 round order."""
    self._npu_moe_permutation_pair = None
    self._npu_moe_expert_order = None
    if not getattr(self, "_npu_moe_schedule", False):
        return MoEAlltoAllTokenDispatcher.dispatch_preprocess.__wrapped__(
            self, hidden_states, routing_map, probs
        )
    tokens_per_expert = self.preprocess(routing_map)
    for point in (
        "before_permutation_1",
        "before_ep_alltoall",
        "before_permutation_2",
        "before_finish",
    ):
        tokens_per_expert = self._maybe_dtoh_and_synchronize(point, tokens_per_expert)
    counts = getattr(self, "_npu_moe_counts", None)
    if counts is None:
        raise RuntimeError("MC2 dispatch requires globally gathered expert counts from preprocess.")
    counts = counts.reshape(self.ep_size, self.num_experts)
    ordered_probs = probs
    self.routing_map = routing_map
    self.tokens_per_expert = tokens_per_expert
    counts_supported = supports_moe_counts(counts)
    if counts_supported:
        orders = (
            counts.sum(0)
            .reshape(self.ep_size, self.num_local_experts)
            .argsort(dim=1, descending=True, stable=True)
        )
        columns = (orders + torch.arange(self.ep_size)[:, None] * self.num_local_experts).flatten()
        local_order = orders[self.ep_rank]
        self._npu_moe_counts = counts.index_select(1, columns)
        self._npu_moe_expert_order = local_order.tolist()
        inverse = columns.argsort().to(hidden_states.device)
        columns = columns.to(hidden_states.device)
        self.routing_map = routing_map.index_select(1, columns)
        ordered_probs = _ReorderProbabilities.apply(probs, columns, inverse)
        self.tokens_per_expert = tokens_per_expert.index_select(0, local_order)
    self.hidden_shape = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    self.hidden_shape_before_permute = hidden_states.shape
    self.probs = probs
    if self.shared_experts is not None:
        self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))
    if counts_supported and self.config.moe_permute_fusion:
        from transformer_engine.plugin import tefl

        get_pair = getattr(tefl, "get_moe_permutation_pair", None)
        if get_pair is not None:
            self._npu_moe_permutation_pair = get_pair(
                hidden_states,
                self.routing_map,
                ordered_probs,
                self.num_out_tokens,
                self.config.moe_router_topk,
            )
    if self._npu_moe_permutation_pair is None:
        output, permuted_probs, mapping, _, _ = permute(
            hidden_states,
            self.routing_map,
            probs=ordered_probs,
            num_out_tokens=self.num_out_tokens,
            fused=self.config.moe_permute_fusion,
        )
    else:
        self.routing_map = self.routing_map.contiguous()
        output, permuted_probs, mapping = self._npu_moe_permutation_pair[0](
            hidden_states, self.routing_map, ordered_probs, self.num_out_tokens
        )
    self.reversed_local_input_permutation_mapping = mapping
    return output, permuted_probs


def combine_postprocess(self, permutated_local_input_tokens: torch.Tensor) -> torch.Tensor:
    """Consume the permutation pair selected by this dispatch, then restore the layer shape."""
    pair = getattr(self, "_npu_moe_permutation_pair", None)
    if pair is None:
        return MoEAlltoAllTokenDispatcher.combine_postprocess.__wrapped__(
            self, permutated_local_input_tokens
        )
    output = pair[1](
        permutated_local_input_tokens,
        self.reversed_local_input_permutation_mapping,
        self.hidden_shape_before_permute,
        self.routing_map,
    ).view(self.hidden_shape)
    if self.shared_experts is not None:
        output += self.shared_experts.get_output()
    return output


def gather_tokens_per_expert(self, counts):
    """Copy existing global counts on the dispatcher's metadata stream."""
    gathered = MoEAlltoAllTokenDispatcher._gather_tokens_per_expert.__wrapped__(self, counts)
    self._npu_moe_counts = None
    if counts.device.type == "npu" and getattr(self, "_npu_moe_schedule", False):
        stream = self.cuda_dtoh_stream
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            self._npu_moe_counts = gathered.to("cpu", non_blocking=True)
            gathered.record_stream(stream)
    return gathered


def _supports_group(layer, x):
    dispatcher = layer.token_dispatcher
    if x.device.type != "npu" or not isinstance(dispatcher, MoEAlltoAllTokenDispatcher):
        return False
    device = x.device
    # All EP ranks vote, including locally unsupported layers. Configuration and
    # TE policy stay fixed after this first vote; a uniform runtime disable is safe.
    key = (dispatcher.ep_group, device)
    cached = getattr(dispatcher, "_npu_moe_group_support", None)
    if cached is None or cached[0] != key:
        supported = _supported(layer, x)
        if supported:
            from transformer_engine.plugin import tefl

            # This local probe must not initialize a communicator.
            supported = tefl.is_moe_fusion_available()
        flag = torch.tensor(int(supported), dtype=torch.int64, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=dispatcher.ep_group)
        if flag.item():
            # Communicator lookup may be collective: enter only after all ranks agree.
            flag.fill_(int(tefl.is_moe_fusion_available(dispatcher.ep_group, device)))
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=dispatcher.ep_group)
        cached = (key, bool(flag.item()))
        dispatcher._npu_moe_group_support = cached
    return cached[1] and _supported(layer, x)


def supports_moe_counts(counts):
    """Check the complete host count matrix shared by all EP ranks."""
    # CANN 9.0.1 GroupedMatMulAlltoAllv stores per-source/expert counts in uint16_t.
    # This operator runs in FC2 forward and FC1 input backward.
    return counts.max().item() < 65536


def _weights(linear, order=None):
    # MC2 bypasses Linear.forward: invoke only its still-registered native DDP hook.
    # disable_forward_pre_hook removes this id; re-enabling installs the new one.
    hook = linear._forward_pre_hooks.get(getattr(linear, "_npu_mc2_param_sync_hook_id", None))
    if hook is not None:
        hook(linear)
    weights = tuple(linear._get_weight_tensors())
    return weights if order is None else tuple(weights[i] for i in order)


def compute_experts(self, hidden_states, probs):
    dispatcher = self.token_dispatcher
    if not _supports_group(self, hidden_states):
        return MoELayer.compute_experts.__wrapped__(self, hidden_states, probs)

    from transformer_engine.plugin import tefl

    counts = dispatcher._npu_moe_counts.reshape(dispatcher.ep_size, dispatcher.num_experts)
    tokens_per_expert = dispatcher.tokens_per_expert
    for point in ("before_ep_alltoall", "before_permutation_2", "before_finish"):
        tokens_per_expert = dispatcher._maybe_dtoh_and_synchronize(point, tokens_per_expert)
    dispatcher.tokens_per_expert = tokens_per_expert
    # Every EP rank sees the full current matrix, including remote-only overflows.
    if not supports_moe_counts(counts):
        if getattr(self, "_npu_moe_mc2_logged", None) is not False:
            logging.getLogger(__name__).warning(
                "NPU_MOE_MC2 fallback: per-source/expert count=%s exceeds uint16 limit; layer=%s",
                counts.max().item(),
                self.layer_number,
            )
        self._npu_moe_mc2_logged = False
        return MoELayer.compute_experts.__wrapped__(self, hidden_states, probs)
    send_counts = counts[dispatcher.ep_rank].tolist()
    recv_counts = counts[:, dispatcher.local_expert_indices].reshape(-1).tolist()
    sizes = torch.tensor(recv_counts, dtype=torch.int64).reshape(-1, dispatcher.num_local_experts)
    dispatcher.tokens_per_expert = None
    shared = dispatcher.shared_experts
    if shared is not None:
        shared.wait_current_stream()
    global_probs = all_to_all(
        dispatcher.ep_group,
        probs,
        dispatcher.output_splits,
        dispatcher.input_splits,
        use_nccl_stream=dispatcher.use_nccl_stream,
    )
    if shared is not None:
        shared.linear_fc1_forward_and_act(hidden_states)
    if dispatcher.num_local_experts > 1:
        fuse_probs = self.config.moe_permute_fusion and global_probs.dtype == torch.float32
        global_probs, _ = sort_chunks_by_idxs(
            global_probs[:, None] if fuse_probs else global_probs,
            sizes.reshape(-1).to(global_probs.device) if fuse_probs else sizes.reshape(-1),
            dispatcher.sort_input_by_local_experts,
            fused=fuse_probs,
        )
        if fuse_probs:
            global_probs = global_probs[:, 0]

    order = getattr(dispatcher, "_npu_moe_expert_order", None)

    def fc1(x, _splits):
        return (
            tefl.moe_alltoallv_grouped_matmul(
                x,
                _weights(self.experts.linear_fc1, order),
                dispatcher.ep_group,
                send_counts,
                recv_counts,
            ),
            None,
        )

    def fc2(x, _splits):
        return (
            tefl.moe_grouped_matmul_alltoallv(
                x,
                _weights(self.experts.linear_fc2, order),
                dispatcher.ep_group,
                send_counts,
                recv_counts,
            ),
            None,
        )

    output, bias = self.experts(
        hidden_states, tokens_per_expert, global_probs, linear_fc1=fc1, linear_fc2=fc2
    )
    if shared is not None:
        shared.linear_fc2_forward(output)
        shared.post_forward_comm()
    if not getattr(self, "_npu_moe_mc2_logged", False):
        logging.getLogger(__name__).info(
            "NPU_MOE_MC2 ep=%s local_experts=%s", dispatcher.ep_size, dispatcher.num_local_experts
        )
        self._npu_moe_mc2_logged = True
    return output, bias
