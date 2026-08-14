import torch
from torch import _C
import torch_npu
from torch_npu.npu import _lazy_call, device as device_ctx_manager


def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1, graph_safe: bool = False):
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):

        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.npu.default_generators[idx]
            if graph_safe:
                default_generator.graphsafe_set_state(new_state)
            else:
                default_generator.set_state(new_state)

    _lazy_call(cb)