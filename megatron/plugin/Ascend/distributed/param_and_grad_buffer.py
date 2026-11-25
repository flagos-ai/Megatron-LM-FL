# Copyright 2026 FlagOS Contributors

from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup


class _MultiCoalescingHandle:
    def __init__(self, handles):
        self.handles = handles

    def wait(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


def _gather_param_buckets(self, indices, async_op):
    """Batch Ascend parameter all-gathers by storage dtype."""
    gather = _ParamAndGradBucketGroup._gather_param_buckets.__wrapped__
    if not self.buckets or self.buckets[0].param_data.device.type != "npu":
        return gather(self, indices, async_op)

    batches = {}
    for idx in indices:
        batches.setdefault(self.buckets[idx].param_data.dtype, []).append(idx)
    handles = [gather(self, batches[dtype], async_op) for dtype in sorted(batches, key=str)]
    if async_op:
        return handles[0] if len(handles) == 1 else _MultiCoalescingHandle(handles)
    return None
