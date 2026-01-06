"""
Plugin implementation for megatron.core.optimizer.clip_grads.

This file mirrors the path structure of the original file:
- Original: megatron/core/optimizer/clip_grads.py
- Plugin:   plugins/core/optimizer/clip_grads.py
"""

import logging
from typing import List, Optional, Union

import torch
from torch import inf

from megatron.plugin.decorators import plugin_implementation
from megatron.core.utils import get_data_parallel_group_if_dtensor
from megatron.core.utils import to_local_if_dtensor
from megatron.core.transformer.module import param_is_not_shared
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate

try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
    )
    l2_norm_impl = multi_tensor_l2norm
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier
        l2_norm_impl = amp_C.multi_tensor_l2norm
    except ImportError:
        from megatron.core.utils import local_multi_tensor_applier as multi_tensor_applier
        from megatron.core.utils import local_multi_tensor_l2_norm as l2_norm_impl

try:
    from megatron.plugin.utils import get_device_type_for_comm
except ImportError:
    def get_device_type_for_comm(group):
        return "cuda"

logger = logging.getLogger(__name__)

@plugin_implementation("clip_grads", "get_grad_norm_fp32")
def get_grad_norm_fp32(
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    norm_type: Union[int, float] = 2,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Calculate the norm of gradients in fp32.

    Plugin implementation that supports:
    - List-based grad_stats_parallel_group (for heterogeneous mode)
    - CPU communication support

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters.

    Arguments:
        grads_for_norm (Iterable[Tensor] or Tensor): an iterable of Tensors or a single
            Tensor that will be used for calculating the grad norm.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        grad_stats_parallel_group (group or list): Process group(s) for reducing the grad norms. This is
            generally the model-parallel group for non-distributed optimizers, and the entire
            world for the distributed optimizer. Can be a list for heterogeneous mode.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    logger.debug(f"Megatron-LM-FL Plugins: get_grad_norm_fp32")
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    data_parallel_group = None
    for grad in grads_for_norm:
        data_parallel_group = get_data_parallel_group_if_dtensor(grad, data_parallel_group)

    grads_for_norm = [to_local_if_dtensor(grad) for grad in grads_for_norm]

    # Norm parameters.
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
            )

        # Take max across all model-parallel GPUs.
        # For cpu communication
        tensor_device = get_device_type_for_comm(grad_stats_parallel_group)
        total_norm_cuda = total_norm_cuda.to(tensor_device)
        if isinstance(grad_stats_parallel_group, list):
            for group in grad_stats_parallel_group:
                torch.distributed.all_reduce(
                    total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=group
                )
        else:
            torch.distributed.all_reduce(
                total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=grad_stats_parallel_group
            )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:
                grad_norm, _ = multi_tensor_applier(
                    l2_norm_impl,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False,  # no per-parameter norm
                )
            else:
                grad_norm = torch.zeros(1, dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm**norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm**norm_type

        # Sum across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
            )

        # For cpu communication
        tensor_device = get_device_type_for_comm(grad_stats_parallel_group)
        total_norm = total_norm.to(tensor_device)
        if isinstance(grad_stats_parallel_group, list):
            for group in grad_stats_parallel_group:
                torch.distributed.all_reduce(
                    total_norm, op=torch.distributed.ReduceOp.SUM, group=group
                )
        else:
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
            )
        total_norm = total_norm.item() ** (1.0 / norm_type)

    return total_norm

@plugin_implementation("clip_grads", "count_zeros_fp32")
def count_zeros_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    grad_stats_parallel_group: torch.distributed.ProcessGroup,
    use_decoupled_grad: bool = False,
) -> float:
    logger.debug(f"Megatron-LM-FL Plugins: count_zeros_fp32")
    """Counts the number of zeros in gradients associated with the passed-in list of
    parameters.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have the number of zeros in its corresponding
            gradient counted.
        grad_stats_parallel_group (group): Process group for reducing the num_zeros count. This is
            generally the model-parallel group for non-distributed optimizers, and the entire
            world for the distributed optimizer.
        use_decoupled_grad (bool, optional) whether to read grad from ".grad" or ".decoupled_grad",
            default value is False.
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = torch.zeros(1, dtype=torch.float, device='cuda')
    data_parallel_group = None
    use_megatron_fsdp = False
    for param in parameters:
        if getattr(param, "__fsdp_param__", False) and param.grad is not None:
            # If the parameter is managed by Megatron FSDP, we need to handle it differently.
            use_megatron_fsdp = True
            grad = param.grad._local_tensor
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros += num_zeros
            continue

        grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
        grad_not_none = hasattr(param, grad_attr) and getattr(param, grad_attr) is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grad_obj = getattr(param, grad_attr)
            data_parallel_group = get_data_parallel_group_if_dtensor(grad_obj, data_parallel_group)
            grad = to_local_if_dtensor(grad_obj).detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    if use_megatron_fsdp and data_parallel_group is not None:
        raise ValueError(
            "Unexpected use of Megatron FSDP with data parallel group. "
            "Please ensure that the parameters are properly managed by Megatron FSDP."
        )

    # Sum across all data-parallel GPUs if using FSDP.
    if data_parallel_group:
        torch.distributed.all_reduce(
            total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )
    # Sum across all model-parallel GPUs.
    comm_device = get_device_type_for_comm(grad_stats_parallel_group)
    if comm_device == "cpu":
        total_num_zeros = total_num_zeros.cpu()

    if isinstance(grad_stats_parallel_group, list):
        original_total_num_zeros = total_num_zeros.clone().detach()
        for group in grad_stats_parallel_group:
            total_num_zeros.data = original_total_num_zeros.data.clone()
            torch.distributed.all_reduce(
                total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=group
            )
    else:
        torch.distributed.all_reduce(
            total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
        )

    total_num_zeros = total_num_zeros.item()

    return total_num_zeros
