from .abstract_accelerator import MegatronAccelerator

# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch
except ImportError as e:
    pass

try:
    import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore
    oneccl_imported_p = True
except ImportError as e:
    oneccl_imported_p = False

import os


# accelerator for Intel CPU
class CPU_Accelerator(MegatronAccelerator):

    def __init__(self):
        self._name = 'cpu'
        self._compile_backend = "inductor"
        if oneccl_imported_p:
            self._communication_backend_name = 'ccl'
        else:
            # fallback to gloo if oneccl_binding_for_pytorch is not installed
            self._communication_backend_name = 'gloo'
        try:
            import psutil
            mem = psutil.Process().memory_info().rss
            self.max_mem = mem
        except ImportError as e:
            self.max_mem = 0

    def get_device_properties(self, device_index=None):
        raise NotImplementedError("CPU does not have device properties")

    def get_device_capability(self, device_index=None):
        raise NotImplementedError("CPU does not have device capability")

    def is_synchronized_device(self):
        return True

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        return 'cpu'

    def device(self, device_index=None):
        return None

    def set_device(self, device_index):
        return

    def current_device(self):
        return os.environ.get('LOCAL_RANK', 0)

    def current_device_name(self):
        return 'cpu'

    def device_count(self):
        device_count = int(os.environ.get('LOCAL_SIZE', 0))
        if device_count > 0:
            return device_count
        else:
            return 0

    def synchronize(self, device_index=None):
        return

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.set_rng_state(new_state)
        return torch.set_rng_state(new_state, device_index)

    def get_rng_state(self, device=None):
        return torch.get_rng_state()

    def manual_seed(self, seed):
        return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.manual_seed(seed)

    def initial_seed(self):
        return torch.initial_seed()

    @property
    def default_generators(self):
        return torch.default_generator

    # Streams/Events
    @property
    def Stream(self):
        return None

    def stream(self, stream):
        from .utils import noop_context
        return noop_context()

    def set_stream(self, stream):
        return

    def current_stream(self, device_index=None):
        return None

    def default_stream(self, device_index=None):
        return None

    @property
    def MemPool(self):
        return None

    def use_mem_pool(self, pool):
        return

    @property
    def Event(self):
        return None

    # Memory management
    def empty_cache(self):
        return

    def get_rss(self):
        import psutil
        mem = psutil.Process().memory_info().rss
        if mem > self.max_mem:
            self.max_mem = mem
        return mem

    def reset_rss(self):
        import psutil
        mem = psutil.Process().memory_info().rss
        self.max_mem = mem
        return mem

    def memory_allocated(self, device_index=None):
        return self.get_rss()

    def max_memory_allocated(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_allocated(self, device_index=None):
        self.reset_rss()
        return

    def memory_cached(self, device_index=None):
        return self.get_rss()

    def max_memory_cached(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_cached(self, device_index=None):
        self.reset_rss()
        return

    def memory_stats(self, device_index=None):
        mem = self.get_rss()
        mem_stat = {}
        mem_stat['allocated_bytes.all.current'] = mem
        mem_stat['allocated_bytes.all.peak'] = self.max_mem
        return mem_stat

    def reset_peak_memory_stats(self, device_index=None):
        self.reset_rss()
        return

    def memory_reserved(self, device_index=None):
        return self.get_rss()

    def max_memory_reserved(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def total_memory(self, device_index=None):
        import psutil
        return psutil.virtual_memory().total

    def available_memory(self, device_index=None):
        import psutil
        return psutil.virtual_memory().available

    # Misc
    def amp(self):
        return torch.cpu.amp

    def is_available(self):
        return True

    def range_push(self, msg):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_push(msg)
        return

    def range_pop(self):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_pop()
        return

    def lazy_call(self, callback):
        return callback()

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        try:
            if torch.ops.mkldnn._is_mkldnn_fp16_supported():
                return True
        except:
            return False

    def supported_dtypes(self):
        supported_dtypes = [torch.float, torch.bfloat16]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.float16)
        return supported_dtypes

    # Graph operations
    def create_graph(self):
        return None

    def capture_to_graph(self, graph, pool=None, stream=None):
        from utils import noop_context
        return noop_context()

    def replay_graph(self, graph):
        return
    
    # Tensor operations
    @property
    def BFloat16Tensor(self):
        return torch.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.FloatTensor

    @property
    def HalfTensor(self):
        return torch.HalfTensor

    @property
    def IntTensor(self):
        return torch.IntTensor

    @property
    def LongTensor(self):
        return torch.LongTensor

    def pin_memory(self, tensor, align_bytes=1):
        return tensor

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('cpu'):
            return True
        else:
            return False

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return []

    # TODO: cpu's visible envs is confirmed, keep as CUDA_VISIBLE_DEVICES
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
