# Copyright (c) 2026, Biren Technology. All rights reserved.
"""Biren SUPA Platform for Megatron-LM-FL.

SUPA is an independent torch device type ('supa', via torch_supa renaming
PrivateUse1). transfer_to_supa patches torch.cuda so fl core's hardcoded
torch.cuda.* call sites resolve to the SUPA device. The distributed
backend is BCCL (torch_supa remaps 'nccl' -> 'bccl').
"""

import logging

from .platform_base import PlatformBase

try:
    import torch
except ImportError:
    pass

logger = logging.getLogger(__name__)


class PlatformSUPA(PlatformBase):

    def __init__(self):
        self._name = "supa"

    def is_available(self):
        try:
            import torch
            # Determine if we are on a SUPA device
            if torch.supa.device_count() > 0 and torch.supa.is_available():
                return True
            else:
                return False
        except Exception:
            return False

    def get_device_properties(self, device_index=None):
        return torch.supa.get_device_properties(device_index)

    def get_device_capability(self, device_index=None):
        return torch.supa.get_device_capability(device_index)

    def is_synchronized_device(self):
        return False

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return "supa"
        return "supa:{}".format(device_index)

    def device(self, device_index=None):
        return torch.device("supa", device_index)

    def set_device(self, device_index):
        torch.supa.set_device(device_index)

    def current_device(self):
        return torch.supa.current_device()

    def current_device_name(self):
        return "supa:{}".format(self.current_device())

    def device_count(self):
        return torch.supa.device_count()

    def synchronize(self, device_index=None):
        return torch.supa.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.supa.set_rng_state(new_state)
        return torch.supa.set_rng_state(new_state, device_index)

    def get_rng_state(self, device=None):
        if device is None:
            return torch.supa.get_rng_state()
        return torch.supa.get_rng_state(device)

    def manual_seed(self, seed):
        return torch.supa.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.supa.manual_seed_all(seed)

    def initial_seed(self):
        return torch.supa.initial_seed()

    @property
    def default_generators(self):
        return torch.supa.default_generators

    # Streams/Events
    @property
    def Stream(self):
        return torch.supa.Stream

    def stream(self, stream):
        return torch.supa.stream(stream)

    def set_stream(self, stream):
        return torch.supa.set_stream(stream)

    def current_stream(self, device_index=None):
        return torch.supa.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.supa.default_stream(device_index)

    @property
    def MemPool(self):
        return torch.supa.MemPool

    def use_mem_pool(self, pool):
        return torch.supa.use_mem_pool(pool)

    @property
    def Event(self):
        return torch.supa.Event

    # Memory management
    def empty_cache(self):
        return torch.supa.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.supa.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.supa.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.supa.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.supa.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.supa.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.supa.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.supa, "memory_stats"):
            return torch.supa.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.supa, "reset_peak_memory_stats"):
            return torch.supa.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.supa, "memory_reserved"):
            return torch.supa.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.supa, "max_memory_reserved"):
            return torch.supa.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.supa.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        try:
            return bool(torch.supa.is_bf16_supported())
        except Exception:
            return False

    def is_fp16_supported(self):
        if not torch.supa.is_available():
            return False
        return True

    def supported_dtypes(self):
        supported_dtypes = [torch.float]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc
    def amp(self):
        if hasattr(torch.supa, "amp"):
            return torch.supa.amp
        return None

    def range(self, msg):
        if hasattr(torch.supa.brtx, "range"):
            return torch.supa.brtx.range(msg)

    def range_push(self, msg):
        if hasattr(torch.supa.brtx, "range_push"):
            return torch.supa.brtx.range_push(msg)

    def range_pop(self):
        if hasattr(torch.supa.brtx, "range_pop"):
            return torch.supa.brtx.range_pop()

    def lazy_call(self, callback):
        pass

    def is_triton_supported(self):
        return False

    # Graph operations
    def create_graph(self):
        return torch.supa.SUPAGraph()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.supa.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        return torch.supa.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.supa.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.supa.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.supa.FloatTensor

    @property
    def HalfTensor(self):
        return torch.supa.HalfTensor

    @property
    def IntTensor(self):
        return torch.supa.IntTensor

    @property
    def LongTensor(self):
        return torch.supa.LongTensor

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        return str(tensor.device).startswith("supa")

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def visible_devices_envs(self):
        return ["SUPA_VISIBLE_DEVICES"]

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))

    def get_compile_backend(self):
        pass

    def set_compile_backend(self, backend):
        pass

    def temperature(self):
        pass

    def power_draw(self):
        pass

    def utilization(self):
        pass

    def clock_rate(self):
        pass
