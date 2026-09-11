# Adopted from DeepSpeed Accelerator, https://github.com/deepspeedai/DeepSpeed/

import abc
from abc import ABC


class PlatformBase(ABC):

    def __init__(self):
        self._name = None

    @abc.abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_device_properties(self, device_index=None):
        ...

    @abc.abstractmethod
    def get_device_capability(self, device_index=None):
        ...

    @abc.abstractmethod
    def is_synchronized_device(self):
        ...

    @abc.abstractmethod
    def use_host_timers(self):
        ...

    @abc.abstractmethod
    def resolves_data_dependency(self):
        ...

    @abc.abstractmethod
    def handles_memory_backpressure(self):
        ...

    def platform_name(self):
        """Return the ME-FL platform identity, e.g. 'cuda', 'npu', 'kunlunxin'.

        This is the key the platform is registered under in ``PLATFORMS`` and the
        vendor key the override registry matches against. It is NOT necessarily a
        torch device type: a platform may drive its hardware through another
        vendor's torch backend. KunLunXin runs XPU through torch_xmlir, which
        exposes it via the torch CUDA API, so its ``platform_name()`` is
        'kunlunxin' while its ``device_name()`` is 'cuda'.

        Use this for identity decisions (platform lookup, vendor override
        selection, user-facing messages). Never pass it to torch.
        """
        return self._name

    # Device APIs
    @abc.abstractmethod
    def device_name(self, device_index):
        """Return a torch-valid device string, e.g. 'cuda' or 'cuda:0'.

        Callers across megatron/core pass this straight into
        ``torch.tensor(device=...)`` / ``torch.device(...)`` and compare it
        against ``tensor.device.type``. It MUST therefore be a device type this
        torch build actually recognises, and it must match the namespace the
        rest of this class drives (``device()``, ``current_device()``, RNG,
        streams). Returning a vendor-branded name that torch has no backend for
        breaks both: torch raises on the tensor-creation sites, and the
        ``device.type ==`` comparisons silently go false. Use
        ``platform_name()`` when you want the vendor identity.
        """
        ...

    @abc.abstractmethod
    def device(self, device_index):
        ...

    @abc.abstractmethod
    def set_device(self, device_index):
        ...

    @abc.abstractmethod
    def current_device(self):
        ...

    @abc.abstractmethod
    def current_device_name(self):
        ...

    @abc.abstractmethod
    def device_count(self):
        ...

    @abc.abstractmethod
    def synchronize(self, device_index=None):
        ...

    # RNG APIs
    @abc.abstractmethod
    def random(self):
        ...

    @abc.abstractmethod
    def set_rng_state(self, new_state, device_index=None):
        ...

    @abc.abstractmethod
    def get_rng_state(self, device=None):
        ...

    @abc.abstractmethod
    def manual_seed(self, seed):
        ...

    @abc.abstractmethod
    def manual_seed_all(self, seed):
        ...

    @abc.abstractmethod
    def initial_seed(self):
        ...

    @property
    def default_generators(self):
        ...

    @property
    @abc.abstractmethod
    def MemPool(self):
        ...

    @abc.abstractmethod
    def use_mem_pool(self, pool):
        ...

    # Streams/Events
    @property
    @abc.abstractmethod
    def Stream(self):
        ...

    @abc.abstractmethod
    def stream(self, stream):
        ...

    @abc.abstractmethod
    def set_stream(self, stream):
        ...

    @abc.abstractmethod
    def current_stream(self, device_index=None):
        ...

    @abc.abstractmethod
    def default_stream(self, device_index=None):
        ...

    @property
    @abc.abstractmethod
    def Event(self):
        ...

    # Memory management
    @abc.abstractmethod
    def empty_cache(self):
        ...

    @abc.abstractmethod
    def memory_allocated(self, device_index=None):
        ...

    @abc.abstractmethod
    def max_memory_allocated(self, device_index=None):
        ...

    @abc.abstractmethod
    def reset_max_memory_allocated(self, device_index=None):
        ...

    @abc.abstractmethod
    def memory_cached(self, device_index=None):
        ...

    @abc.abstractmethod
    def max_memory_cached(self, device_index=None):
        ...

    @abc.abstractmethod
    def reset_max_memory_cached(self, device_index=None):
        ...

    @abc.abstractmethod
    def memory_stats(self, device_index=None):
        ...

    @abc.abstractmethod
    def reset_peak_memory_stats(self, device_index=None):
        ...

    @abc.abstractmethod
    def memory_reserved(self, device_index=None):
        ...

    @abc.abstractmethod
    def max_memory_reserved(self, device_index=None):
        ...

    @abc.abstractmethod
    def total_memory(self, device_index=None):
        ...

    @abc.abstractmethod
    def available_memory(self, device_index=None):
        ...

    # Data types
    @abc.abstractmethod
    def is_bf16_supported(self):
        ...

    @abc.abstractmethod
    def is_fp16_supported(self):
        ...

    @abc.abstractmethod
    def supported_dtypes(self):
        ...

    # Misc
    @abc.abstractmethod
    def amp(self):
        ...

    @abc.abstractmethod
    def range(self, msg):
        ...

    @abc.abstractmethod
    def range_push(self, msg):
        ...

    @abc.abstractmethod
    def range_pop(self):
        ...

    @abc.abstractmethod
    def lazy_call(self, callback):
        ...

    @abc.abstractmethod
    def is_triton_supported(self):
        ...

    # Graph operations
    @abc.abstractmethod
    def create_graph(self):
        ...

    @abc.abstractmethod
    def capture_to_graph(self, graph, pool=None, stream=None):
        ...

    @abc.abstractmethod
    def replay_graph(self, graph):
        ...

    # Tensor operations
    @property
    @abc.abstractmethod
    def BFloat16Tensor(self):
        ...

    @property
    @abc.abstractmethod
    def ByteTensor(self):
        ...

    @property
    @abc.abstractmethod
    def DoubleTensor(self):
        ...

    @property
    @abc.abstractmethod
    def FloatTensor(self):
        ...

    @property
    @abc.abstractmethod
    def HalfTensor(self):
        ...

    @property
    @abc.abstractmethod
    def IntTensor(self):
        ...

    @property
    @abc.abstractmethod
    def LongTensor(self):
        ...

    @abc.abstractmethod
    def pin_memory(self, tensor, align_bytes=1):
        ...

    @abc.abstractmethod
    def is_pinned(self, tensor):
        ...

    @abc.abstractmethod
    def on_accelerator(self, tensor):
        ...

    @abc.abstractmethod
    def build_extension(self):
        ...

    @abc.abstractmethod
    def visible_devices_envs(self):
        ...

    @abc.abstractmethod
    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        ...

    @abc.abstractmethod
    def get_compile_backend(self):
        ...

    @abc.abstractmethod
    def set_compile_backend(self, backend):
        ...
    
    @abc.abstractmethod
    def temperature(self):
        ...

    @abc.abstractmethod
    def power_draw(self):
        ...

    @abc.abstractmethod
    def utilization(self):
        ...

    @abc.abstractmethod
    def clock_rate(self):
        ...