"""KunLunXin plugin implementations for multi-token prediction."""

import torch

from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.plugin.kunlunxin.debug import debug_patch


@debug_patch("transformer.multi_token_prediction.reduce_loss_in_tracker")
def reduce_loss_in_tracker():
    """Collect and reduce the MTP losses across ranks for KunLunXin."""
    tracker = MTPLossLoggingHelper.tracker
    if "values" not in tracker:
        return

    values = tracker["values"]
    if tracker.get("reduce_group") is not None:
        torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))

    if tracker.get("avg_group") is not None:
        torch.distributed.all_reduce(
            values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.SUM
        )
        group_size = torch.distributed.get_world_size(group=tracker["avg_group"])
        values.div_(group_size)
