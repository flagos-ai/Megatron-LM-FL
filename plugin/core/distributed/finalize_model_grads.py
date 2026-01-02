import logging
from typing import Callable, List, Optional, Union

import torch
try:
    from torch.distributed._tensor import DTensor, distribute_tensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False


from megatron.core import parallel_state
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
)
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_pg_size,
)
from megatron.core.distributed.finalize_model_grads import _get_main_grad_attr, _unshard_if_dtensor, _reshard_if_dtensor


from plugin.utils import get_device_type_for_comm
from plugin.decorators import plugin_implementation

logger = logging.getLogger(__name__)

@plugin_implementation("finalize_model_grads", "_allreduce_embedding_grad")
def _allreduce_embedding_grad(
    model: List[torch.nn.Module],
    embd_group: torch.distributed.ProcessGroup,
    pp_group: torch.distributed.ProcessGroup,
    weight_getter: Callable[[torch.nn.Module], Optional[torch.nn.Parameter]],
    skip_if_none: bool = True,
):
    """Unified helper to all-reduce embedding parameters across pipeline stages.

    Args:
        model (List[torch.nn.Module]): A list of model chunks (PP/VPP).
        embd_group (torch.distributed.ProcessGroup): The process group over which to reduce.
        pp_group (torch.distributed.ProcessGroup): The pipeline parallel process group for
            first/last stage detection.
        weight_getter (Callable[[torch.nn.Module], Optional[torch.nn.Parameter]]): A function
            that takes the *pre-process* model chunk and returns the parameter to be reduced
            (or ``None`` if not applicable).
        skip_if_none (bool, optional): If True, quietly returns when the parameter or its
            gradient is ``None``. Defaults to True.
    """
    
    logger.debug(f"Megatron-LM-FL Plugins: _allreduce_embedding_grad")
    embd_group_is_list = isinstance(embd_group, list)
    if (
        not embd_group_is_list and
        # embd_group can be None in cases there is no embd_group
        # get_pg_size(embd_group) will return 1 and the all-reduce will be skipped.
        get_pg_size(embd_group) > 1
        and torch.distributed.get_rank() in torch.distributed.get_process_group_ranks(embd_group)
    ):

        if is_pp_first_stage(pp_group):
            model_module = model[0]
        elif is_pp_last_stage(pp_group):
            model_module = model[-1]
        else:  # We do not support an interleaved schedule for models with encoders yet.
            model_module = model[0]

        ddp_config = model_module.ddp_config
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)

        weight = weight_getter(model_module)
        if weight is None and skip_if_none:
            return

        grad_attr = _get_main_grad_attr(weight)
        orig_grad = getattr(weight, grad_attr)
        if ddp_config.use_megatron_fsdp:
            orig_grad = orig_grad._local_tensor if orig_grad is not None else None
        grad = _unshard_if_dtensor(orig_grad)
        # When the embedding is frozen, the grad is None.
        if grad is None and skip_if_none:
            return
        torch.distributed.all_reduce(grad, group=embd_group)
        setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))

    ######## FlagScale Begin ########
    elif (embd_group_is_list and
        get_pg_size(embd_group) > 1
        and torch.distributed.get_rank() in torch.distributed.get_process_group_ranks(embd_group[0])):
        if is_pp_first_stage(pp_group):
            model_module = model[0]
        elif is_pp_last_stage(pp_group):
            model_module = model[-1]
        else:  # We do not support an interleaved schedule for models with encoders yet.
            model_module = model[0]

        ddp_config = model_module.ddp_config
        use_dist_opt = ddp_config.use_distributed_optimizer
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)

        weight = weight_getter(model_module)
        if weight is None and skip_if_none:
            return

        grad_attr = _get_main_grad_attr(weight)
        orig_grad = getattr(weight, grad_attr)
        if ddp_config.use_megatron_fsdp:
            orig_grad = orig_grad._local_tensor if orig_grad is not None else None
        grad = _unshard_if_dtensor(orig_grad)
        # When the embedding is frozen, the grad is None.
        if grad is None and skip_if_none:
            return
        com_device = get_device_type_for_comm(embd_group)
        if com_device == "cpu":
            grad = grad.cpu()
        if use_dist_opt:
            if ddp_config.use_partial_reduce_for_shared_embedding:
                dp_world_size = parallel_state.get_data_parallel_world_size()
                dp_rank = parallel_state.get_data_parallel_rank()
                assert grad.shape[0] % dp_world_size == 0, f"grad shape: {grad.shape[0]}, dp_world_size: {dp_world_size}"
                per_partion_size = grad.shape[0] // dp_world_size
                if len(embd_group) == 1:
                    offset = per_partion_size * dp_rank
                    torch.distributed.all_reduce(grad[offset:offset+per_partion_size, :], group=embd_group[0])
                else:
                    group_idx = 0
                    per_partion_size = per_partion_size // len(embd_group)
                    for group in embd_group:
                        offset = per_partion_size * (dp_rank * len(embd_group) + group_idx)
                        torch.distributed.all_reduce(grad[offset : offset + per_partion_size, :], group=group)
                        group_idx += 1
            else: # megartron default method
                torch.distributed.all_reduce(grad, group=embd_group[0])
        else:
            if len(embd_group) == 1: # megartron default method
                torch.distributed.all_reduce(grad, group=embd_group[0])
            else:
                original_grad_data = grad.clone().detach().data
                for group in embd_group:
                    grad.data.copy_(original_grad_data)
                    torch.distributed.all_reduce(grad, group=group)
        if grad.device == torch.device('cpu'):
            grad.to(torch.cuda.current_device())
        setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))
    ######## FlagScale End ########

