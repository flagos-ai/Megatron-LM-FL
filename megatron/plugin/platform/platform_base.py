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

    # Device APIs
    @abc.abstractmethod
    def device_name(self, device_index):
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

    # Attention backend capabilities
    def requires_flash_attn_for_dynamic_batching(self) -> bool:
        """Whether the dynamic-batching inference path needs flash-attn >= 2.7.3.

        Platforms with a native paged-attention op (e.g. NPU via CANN) override
        this to False; the flash-attn version gate is then skipped. Default True.
        """
        return True

    def paged_decode_attention(
        self,
        query,
        key_cache,
        block_table,
        seqlens_k,
        *,
        value_cache=None,
        num_heads=None,
        num_kv_heads=None,
        scale_value=None,
    ):
        """Run decode-phase attention (one query row per token) on a paged KV cache.

        Default: no native op available, raise. Platforms with a vendor op
        (NPU: torch_npu.atb._npu_paged_attention_v2) implement this; the caller
        (Attention.flash_decode_and_prefill) dispatches on the platform and
        never reaches here on unsupported platforms.
        """
        raise NotImplementedError(
            "paged_decode_attention is not implemented for platform "
            + self.__class__.__name__
        )

    def paged_prefill_attention(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlens_k,
        block_table,
        *,
        num_heads,
    ):
        """Run prefill-phase attention (varlen, causal) on a paged KV cache.

        Default: no native op available, raise. Platforms with a vendor op
        (NPU: gather the paged cache into contiguous TND rows +
        torch_npu.npu_fusion_attention) implement this; the caller dispatches on
        the platform and never reaches here on unsupported platforms.
        """
        raise NotImplementedError(
            "paged_prefill_attention is not implemented for platform "
            + self.__class__.__name__
        )

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