import torch
from typing import List, Optional
from megatron.core import parallel_state


def get_device_type_for_comm(model_parallel_group=None):
    device = 'cuda'
    # "cpu:gloo": gloo only supports cpu tensor.
    # "gloo" & "cpu:gloo,cuda:gloo": gloo supports both cpu and cuda tensor.
    if isinstance(model_parallel_group, list):
        if 'cpu:gloo' == torch.distributed.get_backend(model_parallel_group[0]):
            device = 'cpu'
    else:
        if 'cpu:gloo' == torch.distributed.get_backend(model_parallel_group):
            device = 'cpu'
    return device


def is_built_on_zero_rank():
    """
    Determines if the current distributed rank is the one responsible for building datasets.

    Returns:
        bool: True if the current rank is responsible for building resources, False otherwise.
    """
    
    from megatron.training import get_args
    #TODO: We should not depend on get_args in megatron core, the args belong to training.
    try: ### for unit tests
        args = get_args()
    except Exception:
        if torch.distributed.get_rank() == 0 or int(os.environ["LOCAL_RANK"]) == 0:
            return True
        else:
            return False

    is_built = False
    if not args.no_shared_fs \
        and torch.distributed.get_rank() == 0:
        is_built = True
    elif args.no_shared_fs \
        and int(os.environ["LOCAL_RANK"]) == 0:
        is_built = True
    else:
        is_built = False

    return is_built


def reduce_aux_losses_tracker_across_ranks_hetero(
    track_names: Optional[List[str]] = None,
):
    """Collect and reduce the auxiliary losses across ranks."""
    # Lazy import inside function to avoid circular import
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    if track_names is None:
        track_names = tracker.keys()
    for name in track_names:
        values = tracker[name]["values"]
        # Reduce aux losses across ranks.
        if tracker[name].get("reduce_group") is not None:
            torch.distributed.all_reduce(
                values, group=tracker[name].get("reduce_group")
            )
        if tracker[name].get("avg_group") is not None:
            torch.distributed.all_reduce(
                values,
                group=tracker[name]["avg_group"],
                op=torch.distributed.ReduceOp.AVG,
            )
        pp_groups = parallel_state.get_pipeline_model_parallel_group()
        if "cpu:gloo" == torch.distributed.get_backend(pp_groups[0]):
            values = values.cpu()
        assert isinstance(pp_groups, list), "pp_groups should be a list for hetero."
        if len(pp_groups) > 1:
            origin_values = values.clone().detach()
            for pp_group in pp_groups:
                values.copy_(origin_values)
                torch.distributed.all_reduce(values, group=pp_group)
        else:
            torch.distributed.all_reduce(values, group=pp_groups[0])
