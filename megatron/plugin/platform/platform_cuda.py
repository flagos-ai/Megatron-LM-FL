# Adopted from DeepSpeed Accelerator, https://github.com/deepspeedai/DeepSpeed/

import os
import sys

from .platform_base import PlatformBase

try:
    import torch.cuda
except ImportError:
    pass

pynvml = None
try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None


class PlatformCUDA(PlatformBase):

    def __init__(self):
        self._name = 'cuda'

    def is_available(self):
        try:
            import torch
            # Determine if we are on a GPU or x86 CPU with torch.
            if torch.cuda.device_count() > 0 and torch.cuda.is_available():  #ignore-cuda
                return True
            else:
                return False
        except (RuntimeError, ImportError) as e:
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
            return 'cuda'
        return 'cuda:{}'.format(device_index)

    def communication_backend_version(self):
        return torch.cuda.nccl.version()

    def device(self, device_index=None):
        return torch.device('cuda', device_index)

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)

    def current_device(self):
        return torch.cuda.current_device()

    def current_device_name(self):
        return 'cuda:{}'.format(torch.cuda.current_device())

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

    def _get_nvml_gpu_id(self, torch_gpu_id):
        """
        credit: https://discuss.pytorch.org/t/making-pynvml-match-torch-device-ids-cuda-visible-devices/103020

        Remap torch device id to nvml device id, respecting CUDA_VISIBLE_DEVICES.

        If the latter isn't set return the same id
        """
        # if CUDA_VISIBLE_DEVICES is used automagically remap the id since pynvml ignores this env var
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
            return ids[torch_gpu_id]  # remap
        else:
            return torch_gpu_id

    def available_memory(self, device_index=None):
        if pynvml:
            if device_index is None:
                device_index = self.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._get_nvml_gpu_id(device_index))
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.free
        else:
            return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        if not torch.cuda.is_available():
            return False
        return torch.cuda.is_bf16_supported()

    def is_fp16_supported(self):
        if not torch.cuda.is_available():
            return False
        # See https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix
        # FP16 on compute capability 6.x is deprecated
        allow_deprecated_fp16 = os.environ.get('DS_ALLOW_DEPRECATED_FP16', '0') == '1'
        major, _ = torch.cuda.get_device_capability()
        if major >= 7:
            return True
        elif major == 6 and allow_deprecated_fp16:
            return True
        else:
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

    def is_available(self):
        return torch.cuda.is_available()

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
        return torch.cuda._lazy_call(callback)

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        if not self.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return True
        else:
            return False

    # Graph operations
    def create_graph(self):
        return torch.cuda.CUDAGraph()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.cuda.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return torch.cuda.BFloat16Tensor
        # return functools.partial(torch.tensor, dtype=torch.bfloat16, device='cuda')

    @property
    def ByteTensor(self):
        return torch.cuda.ByteTensor
        # return functools.partial(torch.tensor, dtype=torch.uint8, device='cuda')

    @property
    def DoubleTensor(self):
        return torch.cuda.DoubleTensor
        # return functools.partial(torch.tensor, dtype=torch.double, device='cuda')

    @property
    def FloatTensor(self):
        return torch.cuda.FloatTensor
        # return functools.partial(torch.tensor, dtype=torch.float, device='cuda')

    @property
    def HalfTensor(self):
        return torch.cuda.HalfTensor
        # return functools.partial(torch.tensor, dtype=torch.half, device='cuda')

    @property
    def IntTensor(self):
        return torch.cuda.IntTensor
        # return functools.partial(torch.tensor, dtype=torch.int, device='cuda')

    @property
    def LongTensor(self):
        return torch.cuda.LongTensor
        # return functools.partial(torch.tensor, dtype=torch.long, device='cuda')

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('cuda:'):
            return True
        else:
            return False

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return ['NCCL']

    def visible_devices_envs(self):
        return ['CUDA_VISIBLE_DEVICES']

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))

    def get_compile_backend(self):
        return self._compile_backend

    def set_compile_backend(self, backend):
        supported_backends = torch._dynamo.list_backends(exclude_tags=())
        if backend in supported_backends:
            self._compile_backend = backend
        else:
            raise ValueError(
                f"{backend} not supported by {self.device_name()}. Supported Backends are {supported_backends}")

    def temperature(self):
        return torch.cuda.temperature()

    def power_draw(self):
        return torch.cuda.power_draw()

    def utilization(self):
        return torch.cuda.utilization()

    def clock_rate(self):
        return torch.cuda.clock_rate()