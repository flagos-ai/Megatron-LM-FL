import torch


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