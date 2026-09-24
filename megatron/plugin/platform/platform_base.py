# Adopted from DeepSpeed Accelerator, https://github.com/deepspeedai/DeepSpeed/

import abc
from abc import ABC

import torch


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
    def supports_paged_attention(self) -> bool:
        """Whether paged_decode_attention/paged_prefill_attention are usable.

        The dynamic-batching attention path needs a kernel that reads a paged KV
        cache: either the flash-attn varlen kernels with a block_table, or the
        platform's own paged op. Platforms without the former (the NPU and the
        devices with no flash-attn build) override this to True; the default
        False keeps a platform with working flash-attn kernels on that path, and
        is also what keeps the flash-attn >= 2.7.3 gate in force there.
        """
        return False

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

        Default: the flag_gems paged kernel. Platforms with a vendor op (NPU:
        torch_npu.atb._npu_paged_attention_v2) override this; the caller reaches
        here only after supports_paged_attention() answered True.
        """
        flash_attn_varlen_func = flag_gems_paged_attention()
        if flash_attn_varlen_func is None:
            raise NotImplementedError(
                "paged_decode_attention is not implemented for platform "
                + self.__class__.__name__
                + " and flag_gems is not installed"
            )
        orig_shape = query.shape
        flat_q = query.reshape(-1, num_heads, query.shape[-1])
        rows = flat_q.shape[0]
        # The metadata buffers are sized for the padded batch: drop surplus rows.
        if block_table.shape[0] != rows:
            block_table = block_table[:rows]
        if isinstance(seqlens_k, torch.Tensor):
            seqlens_k = seqlens_k[:rows]
            max_seqlen_k = int(seqlens_k.max())
        else:
            max_seqlen_k = int(max(seqlens_k))
        cu_seqlens_q = torch.arange(rows + 1, device=query.device, dtype=torch.int32)
        out = flash_attn_varlen_func(
            q=flat_q.contiguous(),
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seqlens_k,
            max_seqlen_k=max_seqlen_k,
            block_table=block_table,
            causal=True,
            softmax_scale=scale_value,
            fa_version=2,
        )
        return out.reshape(orig_shape)

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

        Default: gather the paged cache into contiguous TND rows and run the
        flag_gems varlen kernel. Platforms with a vendor op (NPU:
        torch_npu.npu_fusion_attention) override this; the caller reaches here
        only after supports_paged_attention() answered True.
        """
        flash_attn_varlen_func = flag_gems_paged_attention()
        if flash_attn_varlen_func is None:
            raise NotImplementedError(
                "paged_prefill_attention is not implemented for platform "
                + self.__class__.__name__
                + " and flag_gems is not installed"
            )
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
        if isinstance(seqlens_k, torch.Tensor):
            max_seqlen_k = int(seqlens_k.max())
        else:
            max_seqlen_k = int(max(seqlens_k))
        return flash_attn_varlen_func(
            q=q.contiguous(),
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqlens_k,
            max_seqlen_k=max_seqlen_k,
            block_table=block_table,
            causal=True,
            fa_version=2,
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


def flag_gems_paged_attention():
    """The flag_gems paged-attention kernel, or None when flag_gems is absent.

    flag_gems is the vendor-neutral kernel layer, so one implementation serves
    every device without a vendor paged op. The import is deferred because
    flag_gems is optional at import time.
    """
    try:
        from flag_gems import flash_attn_varlen_func
    except Exception:
        return None
    return flash_attn_varlen_func
