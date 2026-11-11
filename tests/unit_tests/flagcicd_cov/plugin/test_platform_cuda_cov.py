# Copyright (c) 2026, BAAI. All rights reserved.
"""PlatformCUDA: availability probing, torch.cuda forwarding, nvml memory, capability-based dtype support."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.cpp_extension import BuildExtension

from megatron.plugin.platform import platform_cuda
from megatron.plugin.platform.platform_cuda import PlatformCUDA


class _FakeCuda:
    def __init__(self, capability=(8, 0), with_amp=True):
        self.calls = []
        self.capability = capability
        if with_amp:
            self.amp = "cuda-amp"
        self.MemPool = "pool-type"
        self.ByteTensor = "byte-tensor"
        self.DoubleTensor = "double-tensor"
        self.IntTensor = "int-tensor"
        self.LongTensor = "long-tensor"
        self.nvtx = SimpleNamespace(
            range=lambda msg: ("nvtx_range", msg),
            range_push=lambda msg: ("nvtx_push", msg),
            range_pop=lambda: "nvtx_pop",
        )

    def get_device_capability(self, device_index=None):
        self.calls.append(("get_device_capability", (device_index,), {}))
        return self.capability

    def get_device_properties(self, device_index=None):
        self.calls.append(("get_device_properties", (device_index,), {}))
        return SimpleNamespace(total_memory=1024)

    def current_device(self):
        return 1

    def set_rng_state(self, new_state, device_index=None):
        self.calls.append(("set_rng_state", (new_state, device_index), {}))

    def get_rng_state(self, device=None):
        self.calls.append(("get_rng_state", (device,), {}))
        return "rng-state"

    def manual_seed_all(self, seed):
        self.calls.append(("manual_seed_all", (seed,), {}))
        return "seed-all"

    def initial_seed(self):
        return 1234

    def default_stream(self, device_index=None):
        self.calls.append(("default_stream", (device_index,), {}))
        return "default-stream"

    def use_mem_pool(self, pool):
        self.calls.append(("use_mem_pool", (pool,), {}))
        return "pool-ok"

    def reset_max_memory_allocated(self, device_index=None):
        self.calls.append(("reset_max_memory_allocated", (device_index,), {}))
        return "reset-alloc"

    def memory_cached(self, device_index=None):
        self.calls.append(("memory_cached", (device_index,), {}))
        return 64

    def max_memory_cached(self, device_index=None):
        self.calls.append(("max_memory_cached", (device_index,), {}))
        return 128

    def reset_max_memory_cached(self, device_index=None):
        self.calls.append(("reset_max_memory_cached", (device_index,), {}))
        return "reset-cached"

    def max_memory_reserved(self, device_index=None):
        self.calls.append(("max_memory_reserved", (device_index,), {}))
        return 192

    def memory_allocated(self, device_index=None):
        self.calls.append(("memory_allocated", (device_index,), {}))
        return 24

    def graph(self, graph, pool=None, stream=None, capture_error_mode=None):
        self.calls.append(("graph", (graph, pool, stream, capture_error_mode), {}))
        return "captured"

    def _lazy_call(self, callback):
        self.calls.append(("_lazy_call", (callback,), {}))
        return "lazy"


def _fake_torch(accelerator):
    return SimpleNamespace(cuda=accelerator)


def _patch_torch(monkeypatch, accelerator):
    fake = _fake_torch(accelerator)
    monkeypatch.setattr(platform_cuda, "torch", fake)
    return fake


def test_module_import_tolerates_missing_torch_cuda(monkeypatch):
    with pytest.MonkeyPatch.context() as context:
        context.setitem(sys.modules, "torch.cuda", None)
        importlib.reload(platform_cuda)
        assert platform_cuda.PlatformCUDA().device_name() == "cuda"
    importlib.reload(platform_cuda)


def test_is_available_false_when_no_cuda_devices(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    assert PlatformCUDA().is_available() is False


def test_is_available_false_when_torch_import_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert PlatformCUDA().is_available() is False


def test_is_available_true_when_native_device_type_matches(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert PlatformCUDA().is_available() is True


def test_rng_stream_and_device_forwarding(monkeypatch):
    accelerator = _FakeCuda()
    _patch_torch(monkeypatch, accelerator)
    platform = PlatformCUDA()
    assert platform.get_device_capability(1) == (8, 0)
    assert accelerator.calls == [("get_device_capability", (1,), {})]
    platform.set_rng_state("state")
    platform.set_rng_state("state", 1)
    assert accelerator.calls[-2:] == [
        ("set_rng_state", ("state", None), {}),
        ("set_rng_state", ("state", 1), {}),
    ]
    assert platform.get_rng_state() == "rng-state"
    assert platform.manual_seed_all(5) == "seed-all"
    assert platform.initial_seed() == 1234
    assert platform.default_stream(0) == "default-stream"
    assert platform.MemPool == "pool-type"
    assert platform.use_mem_pool("pool") == "pool-ok"


def test_memory_accessors_forward_to_cuda(monkeypatch):
    accelerator = _FakeCuda()
    _patch_torch(monkeypatch, accelerator)
    monkeypatch.setattr(platform_cuda, "pynvml", None)
    platform = PlatformCUDA()
    assert platform.reset_max_memory_allocated(0) == "reset-alloc"
    assert platform.memory_cached(0) == 64
    assert platform.max_memory_cached(0) == 128
    assert platform.reset_max_memory_cached(0) == "reset-cached"
    assert platform.max_memory_reserved(0) == 192
    assert platform.total_memory(0) == 1024
    assert platform.available_memory(0) == 1000
    assert accelerator.calls == [
        ("reset_max_memory_allocated", (0,), {}),
        ("memory_cached", (0,), {}),
        ("max_memory_cached", (0,), {}),
        ("reset_max_memory_cached", (0,), {}),
        ("max_memory_reserved", (0,), {}),
        ("get_device_properties", (0,), {}),
        ("get_device_properties", (0,), {}),
        ("memory_allocated", (0,), {}),
    ]


def test_get_nvml_gpu_id_remaps_through_cuda_visible_devices(monkeypatch):
    platform = PlatformCUDA()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6")
    assert platform._get_nvml_gpu_id(1) == 5
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert platform._get_nvml_gpu_id(1) == 1


def test_available_memory_uses_pynvml_when_installed(monkeypatch):
    accelerator = _FakeCuda()
    _patch_torch(monkeypatch, accelerator)
    nvml_calls = []

    def get_handle(index):
        nvml_calls.append(("handle", index))
        return ("handle", index)

    def get_memory_info(handle):
        nvml_calls.append(("info", handle))
        return SimpleNamespace(free=12345)

    fake_pynvml = SimpleNamespace(
        nvmlDeviceGetHandleByIndex=get_handle, nvmlDeviceGetMemoryInfo=get_memory_info
    )
    monkeypatch.setattr(platform_cuda, "pynvml", fake_pynvml)
    platform = PlatformCUDA()
    assert platform.available_memory(0) == 12345
    assert platform.available_memory() == 12345
    assert nvml_calls == [
        ("handle", 0),
        ("info", ("handle", 0)),
        ("handle", 1),
        ("info", ("handle", 1)),
    ]


@pytest.mark.parametrize(
    "available, capability, deprecated_fp16, bf16_supported, expected_dtypes",
    [
        pytest.param(False, (8, 0), False, True, [torch.float], id="cuda-unavailable"),
        pytest.param(True, (5, 0), False, True, [torch.float, torch.bfloat16], id="sm50-bf16-only"),
        pytest.param(
            True, (6, 0), True, False, [torch.float, torch.half], id="sm60-deprecated-fp16"
        ),
        pytest.param(
            True,
            (8, 0),
            False,
            True,
            [torch.float, torch.half, torch.bfloat16],
            id="sm80-fp16-bf16",
        ),
    ],
)
def test_dtype_support_follows_capability_and_availability(
    monkeypatch, available, capability, deprecated_fp16, bf16_supported, expected_dtypes
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: available)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: capability)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: bf16_supported)
    if deprecated_fp16:
        monkeypatch.setenv("MG_ALLOW_DEPRECATED_FP16", "1")
    platform = PlatformCUDA()
    assert platform.is_bf16_supported() is (bf16_supported and available)
    assert platform.is_fp16_supported() is (available and (deprecated_fp16 or capability[0] >= 7))
    assert platform.supported_dtypes() == expected_dtypes


def test_amp_returns_cuda_amp_or_none(monkeypatch):
    with_amp = _FakeCuda(with_amp=True)
    _patch_torch(monkeypatch, with_amp)
    assert PlatformCUDA().amp() == "cuda-amp"

    without_amp = _FakeCuda(with_amp=False)
    _patch_torch(monkeypatch, without_amp)
    assert PlatformCUDA().amp() is None


def test_range_helpers_forward_to_nvtx(monkeypatch):
    accelerator = _FakeCuda()
    _patch_torch(monkeypatch, accelerator)
    platform = PlatformCUDA()
    assert platform.range("msg") == ("nvtx_range", "msg")
    assert platform.range_push("msg") == ("nvtx_push", "msg")
    assert platform.range_pop() == "nvtx_pop"
    assert platform.lazy_call("callback") == "lazy"
    assert accelerator.calls == [("_lazy_call", ("callback",), {})]


@pytest.mark.parametrize(
    "capability, expected",
    [
        pytest.param((6, 0), False, id="sm60-unsupported"),
        pytest.param((8, 0), True, id="sm80-supported"),
    ],
)
def test_is_triton_supported_follows_device_capability(monkeypatch, capability, expected):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: capability)
    assert PlatformCUDA().is_triton_supported() is expected


def test_is_triton_supported_false_when_no_devices(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    assert PlatformCUDA().is_triton_supported() is False


def test_random_forwards_to_torch_module():
    assert PlatformCUDA().random() is torch.random


def test_graph_capture_and_tensor_helpers(monkeypatch):
    accelerator = _FakeCuda()
    _patch_torch(monkeypatch, accelerator)
    platform = PlatformCUDA()
    assert platform.capture_to_graph("graph", "pool", "stream") == "captured"
    assert accelerator.calls == [("graph", ("graph", "pool", "stream", None), {})]
    assert platform.ByteTensor == "byte-tensor"
    assert platform.DoubleTensor == "double-tensor"
    assert platform.IntTensor == "int-tensor"
    assert platform.LongTensor == "long-tensor"

    class _FakeTensor:
        def __init__(self, device):
            self.device = device
            self.pinned = False

        def pin_memory(self):
            self.pinned = True
            return self

        def is_pinned(self):
            return self.pinned

    tensor = _FakeTensor(torch.device("cuda:0"))
    assert platform.pin_memory(tensor) is tensor
    assert platform.is_pinned(tensor) is True
    assert platform.on_accelerator(tensor) is True
    assert platform.on_accelerator(_FakeTensor(torch.device("cpu"))) is False


def test_extension_and_env_helpers(monkeypatch):
    accelerator = _FakeCuda()
    _patch_torch(monkeypatch, accelerator)
    platform = PlatformCUDA()
    assert platform.build_extension() is BuildExtension
    current_env = {}
    platform.set_visible_devices_envs(current_env, [1, 3])
    assert current_env == {"CUDA_VISIBLE_DEVICES": "1,3"}


def test_set_compile_backend_rejects_unsupported_backend(monkeypatch):
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    platform = PlatformCUDA()
    platform.set_compile_backend("eager")
    assert platform.get_compile_backend() == "eager"
    with pytest.raises(ValueError, match="bogus not supported by cuda"):
        platform.set_compile_backend("bogus")
