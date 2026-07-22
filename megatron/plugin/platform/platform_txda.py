# Adopted from DeepSpeed Accelerator, https://github.com/deepspeedai/DeepSpeed/

import os
import sys

from .platform_base import PlatformBase

try:
    import torch
    import flag_gems
    from torch_txda import transfer_to_txda
except Exception:
    pass


class PlatformTXDA(PlatformBase):

    def __init__(self):
        self._name = 'txda'

    def is_available(self):
        try:
            import torch
            import torch_txda
            # Determine if we are on a TXDA device
            if torch.txda.device_count() > 0 and torch.txda.is_available():
                return True
            else:
                return False
        except Exception as e:
            return False

    def get_device_properties(self, device_index=None):
        return torch.cuda.get_device_properties(device_index)

    def get_device_capability(self, device_index=None):
        return torch.cuda.get_device_capability(device_index)

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
            return 'txda'
        return 'txda:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.device('txda', device_index)

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)

    def current_device(self):
        return torch.cuda.current_device()

    def current_device_name(self):
        return 'txda:{}'.format(torch.cuda.current_device())

    def device_count(self):
        return torch.cuda.device_count()

    def synchronize(self, device_index=None):
        return torch.cuda.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.cuda.set_rng_state(new_state)

        return torch.cuda.set_rng_state(new_state, device_index)

    def get_rng_state(self, device=None):
        if device is None:
            return torch.cuda.get_rng_state()

        return torch.cuda.get_rng_state(device)

    def manual_seed(self, seed):
        return torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.cuda.manual_seed_all(seed)

    def initial_seed(self):
        return torch.cuda.initial_seed()

    @property
    def default_generators(self):
        return torch.cuda.default_generators

    # Streams/Events
    @property
    def Stream(self):
        return torch.cuda.Stream

    def stream(self, stream):
        return torch.cuda.stream(stream)
    
    def set_stream(self, stream):
        return torch.cuda.set_stream(stream)

    def current_stream(self, device_index=None):
        return torch.cuda.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.cuda.default_stream(device_index)

    @property
    def MemPool(self):
        return torch.cuda.MemPool

    def use_mem_pool(self, pool):
        return torch.cuda.use_mem_pool(pool)

    @property
    def Event(self):
        return torch.cuda.Event

    # Memory management
    def empty_cache(self):
        return torch.cuda.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.cuda.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.cuda.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch.cuda.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch.cuda.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch.cuda.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch.cuda.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.cuda, 'memory_stats'):
            return torch.cuda.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            return torch.cuda.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch.cuda, 'memory_reserved'):
            return torch.cuda.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.cuda, 'max_memory_reserved'):
            return torch.cuda.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.cuda.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        if not torch.cuda.is_available():
            return False
        return torch.cuda.is_bf16_supported()

    def is_fp16_supported(self):
        if not torch.cuda.is_available():
            return False
        if hasattr(torch.cuda, "is_fp16_supported"):
            return torch.cuda.is_fp16_supported()
        return False

    def supported_dtypes(self):
        supported_dtypes = [torch.float]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc
    def amp(self):
        if hasattr(torch.cuda, 'amp'):
            return torch.cuda.amp
        return None

    def range(self, msg):
        if hasattr(torch.cuda.nvtx, 'range'):
            return torch.cuda.nvtx.range(msg)

    def range_push(self, msg):
        if hasattr(torch.cuda.nvtx, 'range_push'):
            return torch.cuda.nvtx.range_push(msg)

    def range_pop(self):
        if hasattr(torch.cuda.nvtx, 'range_pop'):
            return torch.cuda.nvtx.range_pop()

    def lazy_call(self, callback):
        #return torch.cuda._lazy_call(callback)
        pass

    def is_triton_supported(self):
        pass

    # Graph operations
    def create_graph(self):
        #return torch.cuda.CUDAGraph()
        return None

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.cuda.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return torch.cuda.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.cuda.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.cuda.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.cuda.FloatTensor

    @property
    def HalfTensor(self):
        return torch.cuda.HalfTensor

    @property
    def IntTensor(self):
        return torch.cuda.IntTensor

    @property
    def LongTensor(self):
        return torch.cuda.LongTensor

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('txda:'):
            return True
        else:
            return False

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def visible_devices_envs(self):
        return ['TXDA_VISIBLE_DEVICES']

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))

    def get_compile_backend(self):
        #return self._compile_backend
        pass

    def set_compile_backend(self, backend):
        pass

    def temperature(self):
        #return torch.cuda.temperature()
        pass

    def power_draw(self):
        #return torch.cuda.power_draw()
        pass

    def utilization(self):
        #return torch.cuda.utilization()
        pass

    def clock_rate(self):
        #return torch.cuda.clock_rate()
        pass
