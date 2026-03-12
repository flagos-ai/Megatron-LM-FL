# Adopted from DeepSpeed Accelerator, https://github.com/deepspeedai/DeepSpeed/

import os
import sys

from .platform_base import PlatformBase

try:
    import torch.musa
except ImportError:
    pass

pynvml = None
try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None


class PlatformMUSA(PlatformBase):

    def __init__(self):
        self._name = 'musa'

    def is_available(self):
        try:
            import torch
            # Determine if we are on a MUSA device
            if torch.musa.device_count() > 0 and torch.musa.is_available():
                return True
            else:
                return False
        except (RuntimeError, ImportError) as e:
            return False

    def get_device_properties(self, device_index=None):
        return torch.musa.get_device_properties(device_index)

    def get_device_capability(self, device_index=None):
        return torch.musa.get_device_capability(device_index)

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
            return 'musa'
        return 'musa:{}'.format(device_index)

    def communication_backend_version(self):
        return torch.musa.mccl.version()

    def device(self, device_index=None):
        return torch.device('musa', device_index)

    def set_device(self, device_index):
        torch.musa.set_device(device_index)

    def current_device(self):
        return torch.musa.current_device()

    def current_device_name(self):
        return 'musa:{}'.format(torch.musa.current_device())

    def device_count(self):
        return torch.musa.device_count()

    def synchronize(self, device_index=None):
        return torch.musa.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.musa.set_rng_state(new_state)

        return torch.musa.set_rng_state(new_state, device_index)

    def get_rng_state(self, device=None):
        if device is None:
            return torch.musa.get_rng_state()

        return torch.musa.get_rng_state(device)

    def manual_seed(self, seed):
        return torch.musa.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.musa.manual_seed_all(seed)

    def initial_seed(self):
        return torch.musa.initial_seed()

    @property
    def default_generators(self):
        return torch.musa.default_generators

    # Streams/Events
    @property
    def Stream(self):
        return torch.musa.Stream

    def stream(self, stream):
        return torch.musa.stream(stream)
    
    def set_stream(self, stream):
        return torch.musa.set_stream(stream)

    def current_stream(self, device_index=None):
        return torch.musa.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.musa.default_stream(device_index)

    @property
    def MemPool(self):
        return torch.musa.MemPool

    def use_mem_pool(self, pool):
        return torch.musa.use_mem_pool(pool)

    @property
    def Event(self):
        return torch.musa.Event

    # Memory management
    def empty_cache(self):
        return torch.musa.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.musa.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.musa.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.musa.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.musa.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.musa.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.musa.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.musa, 'memory_stats'):
            return torch.musa.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.musa, 'reset_peak_memory_stats'):
            return torch.musa.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.musa, 'memory_reserved'):
            return torch.musa.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.musa, 'max_memory_reserved'):
            return torch.musa.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.musa.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        if not torch.musa.is_available():
            return False
        return torch.musa.is_bf16_supported()

    def supported_dtypes(self):
        supported_dtypes = [torch.float]
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc
    def amp(self):
        if hasattr(torch.musa, 'amp'):
            return torch.musa.amp
        return None

    def is_available(self):
        return torch.musa.is_available()

    def range(self, msg):
        pass

    def range_push(self, msg):
        pass

    def range_pop(self):
        pass

    def lazy_call(self, callback):
        pass

    def is_triton_supported(self):
        pass

    # Graph operations
    def create_graph(self):
        return torch.musa.MUSAGraph()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.musa.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        return torch.musa.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.musa.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.musa.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.musa.FloatTensor

    @property
    def HalfTensor(self):
        return torch.musa.HalfTensor

    @property
    def IntTensor(self):
        return torch.musa.IntTensor

    @property
    def LongTensor(self):
        return torch.musa.LongTensor

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('musa:'):
            return True
        else:
            return False

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def visible_devices_envs(self):
        return ['MUSA_VISIBLE_DEVICES']

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