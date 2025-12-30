from typing import List, Optional

import torch

from megatron.core import parallel_state

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
