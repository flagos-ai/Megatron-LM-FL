# Adopted from DeepSpeed Accelerator, https://github.com/deepspeedai/DeepSpeed/

import warnings

import torch

from .platform_base import PlatformBase

try:
    import torch_ptpu
except Exception:
    pass


class _PTPUTensorMetatype(type):
    """isinstance() predicate standing in for a vendor per-dtype tensor class.

    torch_ptpu does not publish torch.cuda.FloatTensor-style classes: every PPU
    tensor is a plain torch.Tensor whose device type is 'ptpu', so a real class
    cannot express "fp32 on this device". The dtype classification in
    megatron/core/transformer/module.py is isinstance-only, so the properties
    below return these predicates instead.
    """

    _dtype = None

    def __instancecheck__(cls, instance):
        return (
            isinstance(instance, torch.Tensor)
            and instance.device.type == 'ptpu'
            and instance.dtype == cls._dtype
        )


def _ptpu_tensor_type(dtype):
    return _PTPUTensorMetatype('ptpu.{}Tensor'.format(str(dtype).split('.')[-1]), (), {'_dtype': dtype})


_PTPU_TENSOR_TYPES = {
    'bfloat16': _ptpu_tensor_type(torch.bfloat16),
    'uint8': _ptpu_tensor_type(torch.uint8),
    'float64': _ptpu_tensor_type(torch.float64),
    'float32': _ptpu_tensor_type(torch.float32),
    'float16': _ptpu_tensor_type(torch.float16),
    'int32': _ptpu_tensor_type(torch.int32),
    'int64': _ptpu_tensor_type(torch.int64),
}


class PlatformPTPU(PlatformBase):
    """Sunrise (T-Head PPU / tangrt) platform.

    torch_ptpu registers 'ptpu' as a PrivateUse1 backend
    (torch.utils.rename_privateuse1_backend("ptpu")), so the device API lives
    under torch.ptpu. Unlike the MLU/TXDA bridges it does *not* alias that
    surface onto torch.cuda, and torch.cuda.is_available() stays False on a
    PPU host — without this platform the selection chain falls through to the
    CPU platform and initialize.py aborts with "Megatron requires an
    accelerator."

    The methods mirror the vendor surface 1:1. The names torch_ptpu does not
    export (manual_seed family, memory_cached/memory_stats, MemPool, amp, nvtx
    ranges) are guarded and degrade to the generic torch call or to None, the
    same shape platform_txda/platform_enflame use.
    """

    def __init__(self):
        self._name = 'ptpu'

    def is_available(self):
        try:
            import torch
            if torch.ptpu.device_count() > 0 and torch.ptpu.is_available():
                return True
            else:
                return False
        except Exception as e:
            return False

    def get_device_properties(self, device_index=None):
        return torch.ptpu.get_device_properties(device_index)

    def get_device_capability(self, device_index=None):
        return torch.ptpu.get_device_capability(device_index)

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
            return 'ptpu'
        return 'ptpu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.device('ptpu', device_index)

    def set_device(self, device_index):
        torch.ptpu.set_device(device_index)

    def current_device(self):
        return torch.ptpu.current_device()

    def current_device_name(self):
        return 'ptpu:{}'.format(torch.ptpu.current_device())

    def device_count(self):
        return torch.ptpu.device_count()

    def synchronize(self, device_index=None):
        return torch.ptpu.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.ptpu.set_rng_state(new_state)

        return torch.ptpu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device=None):
        if device is None:
            return torch.ptpu.get_rng_state()

        return torch.ptpu.get_rng_state(device)

    # torch_ptpu exposes the per-device state but not the seeding entry points;
    # the generic torch calls are device-agnostic (they forward to the
    # registered PrivateUse1 module).
    def manual_seed(self, seed):
        if hasattr(torch.ptpu, 'manual_seed'):
            return torch.ptpu.manual_seed(seed)
        return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        if hasattr(torch.ptpu, 'manual_seed_all'):
            return torch.ptpu.manual_seed_all(seed)
        return torch.manual_seed_all(seed)

    def initial_seed(self):
        if hasattr(torch.ptpu, 'initial_seed'):
            return torch.ptpu.initial_seed()
        return torch.initial_seed()

    @property
    def default_generators(self):
        return torch.ptpu.default_generators

    # Streams/Events
    @property
    def Stream(self):
        return torch.ptpu.Stream

    def stream(self, stream):
        return torch.ptpu.stream(stream)

    def set_stream(self, stream):
        return torch.ptpu.set_stream(stream)

    def current_stream(self, device_index=None):
        return torch.ptpu.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch.ptpu.default_stream(device_index)

    @property
    def MemPool(self):
        return getattr(torch.ptpu, 'MemPool', None)

    def use_mem_pool(self, pool):
        if hasattr(torch.ptpu, 'use_mem_pool'):
            return torch.ptpu.use_mem_pool(pool)

    @property
    def Event(self):
        return torch.ptpu.Event

    # Memory management
    #
    # torch_ptpu tracks the resident and peak-allocated counters but publishes
    # no cached/peak-reserved pair. report_memory() (training/utils.py) divides
    # every one of these by 1 MiB with no None guard, so the missing ones have
    # to answer with the closest counter the vendor does keep — returning None
    # aborts the first iteration's memory log.
    def empty_cache(self):
        return torch.ptpu.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch.ptpu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch.ptpu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        if hasattr(torch.ptpu, 'reset_max_memory_allocated'):
            return torch.ptpu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        if hasattr(torch.ptpu, 'memory_cached'):
            return torch.ptpu.memory_cached(device_index)
        return self.memory_reserved(device_index)

    def max_memory_cached(self, device_index=None):
        if hasattr(torch.ptpu, 'max_memory_cached'):
            return torch.ptpu.max_memory_cached(device_index)
        return self.max_memory_reserved(device_index)

    def reset_max_memory_cached(self, device_index=None):
        if hasattr(torch.ptpu, 'reset_max_memory_cached'):
            return torch.ptpu.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch.ptpu, 'memory_stats'):
            return torch.ptpu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        return torch.ptpu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        return torch.ptpu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch.ptpu, 'max_memory_reserved'):
            return torch.ptpu.max_memory_reserved(device_index)
        return self.memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch.ptpu.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    #
    # torch_ptpu publishes no capability query, so the dtype support is probed
    # by allocating the smallest tensor of that dtype; a vendor build that
    # gains a query is used directly.
    def _dtype_supported(self, dtype):
        if not torch.ptpu.is_available():
            return False
        try:
            torch.empty(1, dtype=dtype, device='ptpu')
            return True
        except Exception:
            return False

    def is_bf16_supported(self):
        if not torch.ptpu.is_available():
            return False
        if hasattr(torch.ptpu, 'is_bf16_supported'):
            return torch.ptpu.is_bf16_supported()
        return self._dtype_supported(torch.bfloat16)

    def is_fp16_supported(self):
        if not torch.ptpu.is_available():
            return False
        if hasattr(torch.ptpu, 'is_fp16_supported'):
            return torch.ptpu.is_fp16_supported()
        return self._dtype_supported(torch.float16)

    def supported_dtypes(self):
        supported_dtypes = [torch.float]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc
    def amp(self):
        return getattr(torch.ptpu, 'amp', None)

    def range(self, msg):
        if hasattr(torch.cuda, 'nvtx') and hasattr(torch.cuda.nvtx, 'range'):
            return torch.cuda.nvtx.range(msg)

    def range_push(self, msg):
        if hasattr(torch.cuda, 'nvtx') and hasattr(torch.cuda.nvtx, 'range_push'):
            return torch.cuda.nvtx.range_push(msg)

    def range_pop(self):
        if hasattr(torch.cuda, 'nvtx') and hasattr(torch.cuda.nvtx, 'range_pop'):
            return torch.cuda.nvtx.range_pop()

    def lazy_call(self, callback):
        pass

    def is_triton_supported(self):
        pass

    # Graph operations
    def create_graph(self):
        # torch_ptpu exposes PTPUGraph; guarded so a vendor build without it
        # degrades to "no graph" the way platform_txda does.
        graph_cls = getattr(torch.ptpu, 'PTPUGraph', None)
        if graph_cls is None:
            return None
        return graph_cls()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.ptpu.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return getattr(torch.ptpu, 'BFloat16Tensor', _PTPU_TENSOR_TYPES['bfloat16'])

    @property
    def ByteTensor(self):
        return getattr(torch.ptpu, 'ByteTensor', _PTPU_TENSOR_TYPES['uint8'])

    @property
    def DoubleTensor(self):
        return getattr(torch.ptpu, 'DoubleTensor', _PTPU_TENSOR_TYPES['float64'])

    @property
    def FloatTensor(self):
        return getattr(torch.ptpu, 'FloatTensor', _PTPU_TENSOR_TYPES['float32'])

    @property
    def HalfTensor(self):
        return getattr(torch.ptpu, 'HalfTensor', _PTPU_TENSOR_TYPES['float16'])

    @property
    def IntTensor(self):
        return getattr(torch.ptpu, 'IntTensor', _PTPU_TENSOR_TYPES['int32'])

    @property
    def LongTensor(self):
        return getattr(torch.ptpu, 'LongTensor', _PTPU_TENSOR_TYPES['int64'])

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('ptpu:'):
            return True
        else:
            return False

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def visible_devices_envs(self):
        return ['TANG_VISIBLE_DEVICES']

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


def enable_flag_gems():
    """Bind the device's operator library.

    torch_ptpu registers only a slice of aten for PrivateUse1 — matmul, softmax
    and the elementwise kernels it has taBLAS/taDNN support for. The rest
    (torch.tanh, which the jit-fuser warmup calls, among them) raises
    "Could not run 'aten::tanh.out' with arguments from the 'ptpu' backend",
    so a training run cannot get through initialization without it. FlagGems is
    the FlagOS operator library for this device; the registration path is where
    it gets switched on, the same way platform_npu calls registry_patch() from
    there.

    Called only when the platform is available, and tolerant of a device whose
    torch_ptpu build is complete enough not to need it.

    The pow family is excluded: its scalar-tensor entry reaches
    flag_gems' _fallback_pow, which compiles `x ** exponent` on a tl.tensor and
    therefore cannot build on either compiler (`tl.tensor` has no __pow__). The
    vendor kernel covers these, and the same exclusion is what the sglang line
    for this device carries. Filed upstream as FlagGems #6173.
    """
    try:
        import flag_gems

        flag_gems.enable(unused=["pow_scalar", "pow_tensor_scalar", "pow_tensor_tensor"])
    except Exception as e:  # noqa: BLE001 - a missing op library must not be fatal here
        warnings.warn(f"flag_gems could not be enabled for ptpu: {e}")
