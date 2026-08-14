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

from megatron.core.utils import get_data_parallel_group_if_dtensor
from megatron.core.utils import to_local_if_dtensor
from megatron.core.transformer.module import param_is_not_shared
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate

from megatron.plugin.platform import get_platform
cur_platform = get_platform()

try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_l2norm,
        multi_tensor_scale_tensor,
    )
    l2_norm_impl = multi_tensor_l2norm
    multi_tensor_scale_tensor_impl = multi_tensor_scale_tensor
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier
        l2_norm_impl = amp_C.multi_tensor_l2norm
        multi_tensor_scale_tensor_impl = None
    except ImportError:
        from megatron.core.utils import local_multi_tensor_applier as multi_tensor_applier
        from megatron.core.utils import local_multi_tensor_l2_norm as l2_norm_impl
        multi_tensor_scale_tensor_impl = None

try:
    from megatron.plugin.utils import get_device_type_for_comm
except ImportError:
    def get_device_type_for_comm(group):
        return cur_platform.device_name()

logger = logging.getLogger(__name__)

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
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device=cur_platform.device_name())
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
            dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=cur_platform.device_name())
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
                grad_norm = torch.zeros(1, dtype=torch.float, device=cur_platform.device_name())
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
        if multi_tensor_scale_tensor_impl is not None:
            total_norm = total_norm.pow(1.0 / norm_type)
        else:
            total_norm = total_norm.item() ** (1.0 / norm_type)

    return total_norm

