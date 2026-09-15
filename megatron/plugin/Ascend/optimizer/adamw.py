"""Select TorchNPU fused AdamW through the Megatron override interface."""

import torch

from megatron.core.optimizer import _get_adam_class


def get_adam_class(config, kwargs):
    """Use native fused AdamW only for supported, device-resident NPU parameters.

    Parameter groups, distributed optimizer wrapping, and checkpoint handling
    remain owned by core. Unsupported configurations retain the original path.
    """
    npu = getattr(torch, "npu", None)
    supported = (
        npu is not None
        and npu.is_available()
        and config.decoupled_weight_decay
        and not config.optimizer_cuda_graph
        and not config.use_precision_aware_optimizer
        and not config.optimizer_cpu_offload
    )
    if supported:
        params = [param for group in kwargs["params"] for param in group["params"]]
        supported = bool(params) and all(param.device.type == "npu" for param in params)
    if not supported:
        return _get_adam_class.__wrapped__(config, kwargs)

    kwargs.update(fused=True, foreach=False, capturable=False)
    return torch.optim.AdamW
