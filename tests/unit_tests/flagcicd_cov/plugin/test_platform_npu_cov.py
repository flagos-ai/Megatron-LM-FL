# Copyright (c) 2026, BAAI. All rights reserved.
"""PlatformNPU adapter: torch_npu forwarding, mstx range stack, graph and monitoring placeholders."""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from megatron.plugin.platform import platform_npu
from megatron.plugin.platform.platform_npu import PlatformNPU


class _FakeMstx:
    def __init__(self):
        self.calls = []

    def mstx_range(self, msg, *args, **kwargs):
        self.calls.append(("range", msg))
        return ("range", msg)

    def range_start(self, msg, range_id):
        self.calls.append(("start", msg, range_id))
        return ("push", msg)

    def range_end(self, range_id):
        self.calls.append(("end", range_id))
        return "pop"


class _FakeGraph:
    def __init__(self):
        self.replayed = False

    def replay(self):
        self.replayed = True


class _FakeNpuRuntime:
    def __init__(self, device_count=2, available=True, mstx=None, with_amp=True):
        self.calls = []
        self.num_devices = device_count
        self.available = available
        self.mstx = mstx or _FakeMstx()
        if with_amp:
            self.amp = "npu-amp"
        self.Stream = "stream-type"
        self.Event = "event-type"

    def _forward(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return None

    def is_available(self):
        return self.available

    def device_count(self):
        return self.num_devices

    def get_device_properties(self, device_index=None):
        self.calls.append(("get_device_properties", (device_index,), {}))
        return SimpleNamespace(total_memory=2048)

    def set_device(self, device_index):
        self.calls.append(("set_device", (device_index,), {}))

    def current_device(self):
        return 1

    def synchronize(self, device_index=None):
        self.calls.append(("synchronize", (device_index,), {}))
        return ("sync", device_index)

    def set_rng_state(self, new_state, device_index=None):
        self.calls.append(("set_rng_state", (new_state, device_index), {}))

    def get_rng_state(self, device=None):
        self.calls.append(("get_rng_state", (device,), {}))
        return "rng-state"

    def manual_seed(self, seed):
        self.calls.append(("manual_seed", (seed,), {}))
        return ("seed", seed)

    def manual_seed_all(self, seed):
        self.calls.append(("manual_seed_all", (seed,), {}))
        return ("seed-all", seed)

    def initial_seed(self):
        return 1234

    def stream(self, stream):
        self.calls.append(("stream", (stream,), {}))
        return ("stream", stream)

    def set_stream(self, stream):
        self.calls.append(("set_stream", (stream,), {}))
        return ("set_stream", stream)

    def current_stream(self, device_index=None):
        self.calls.append(("current_stream", (device_index,), {}))
        return ("current_stream", device_index)

    def default_stream(self, device_index=None):
        self.calls.append(("default_stream", (device_index,), {}))
        return ("default_stream", device_index)

    def empty_cache(self):
        return "empty"

    def memory_allocated(self, device_index=None):
        self.calls.append(("memory_allocated", (device_index,), {}))
        return 24

    def max_memory_allocated(self, device_index=None):
        return 48

    def reset_max_memory_allocated(self, device_index=None):
        return "reset-alloc"

    def memory_cached(self, device_index=None):
        return 64

    def max_memory_cached(self, device_index=None):
        return 128

    def reset_max_memory_cached(self, device_index=None):
        return "reset-cached"

    def memory_stats(self, device_index=None):
        return {"device": device_index}

    def reset_peak_memory_stats(self, device_index=None):
        return "reset-peak"

    def memory_reserved(self, device_index=None):
        return 96

    def max_memory_reserved(self, device_index=None):
        return 192

    def graph_pool_handle(self):
        return "pool-handle"


def _fake_torch(npu_runtime):
    return SimpleNamespace(
        device=lambda name, index=None: SimpleNamespace(type=name, index=index),
        float=torch.float,
        half=torch.half,
        bfloat16=torch.bfloat16,
        npu=SimpleNamespace(
            MemPool="pool-type",
            use_mem_pool=lambda pool: ("pool", pool),
            NPUGraph=_FakeGraph,
            graph=lambda graph, pool=None, stream=None: ("capture", graph, pool, stream),
            BFloat16Tensor="bf16-tensor",
            ByteTensor="byte-tensor",
            DoubleTensor="double-tensor",
            FloatTensor="float-tensor",
            HalfTensor="half-tensor",
            IntTensor="int-tensor",
            LongTensor="long-tensor",
        ),
    )


def _patch_npu(monkeypatch, npu_runtime):
    monkeypatch.setattr(platform_npu, "torch_npu", SimpleNamespace(npu=npu_runtime), raising=False)
    monkeypatch.setattr(platform_npu, "torch", _fake_torch(npu_runtime), raising=False)


def test_is_available_false_when_no_npu_devices(monkeypatch):
    runtime = _FakeNpuRuntime(device_count=0)
    _patch_npu(monkeypatch, runtime)
    assert PlatformNPU().is_available() is False


def test_device_and_synchronization_forward_to_npu(monkeypatch):
    runtime = _FakeNpuRuntime()
    _patch_npu(monkeypatch, runtime)
    platform = PlatformNPU()
    assert platform.get_device_properties(0).total_memory == 2048
    assert platform.get_device_capability(0) == (9, 0)
    assert platform.is_synchronized_device() is False
    assert platform.use_host_timers() is False
    assert platform.resolves_data_dependency() is False
    assert platform.handles_memory_backpressure() is False
    assert platform.device_name() == "npu"
    assert platform.device_name(3) == "npu:3"
    assert platform.device(2) == SimpleNamespace(type="npu", index=2)
    platform.set_device(4)
    assert platform.current_device() == 1
    assert platform.current_device_name() == "npu:1"
    assert platform.device_count() == 2
    assert platform.synchronize(0) == ("sync", 0)
    assert runtime.calls == [
        ("get_device_properties", (0,), {}),
        ("set_device", (4,), {}),
        ("synchronize", (0,), {}),
    ]


def test_rng_and_stream_forward_to_npu(monkeypatch):
    runtime = _FakeNpuRuntime()
    _patch_npu(monkeypatch, runtime)
    platform = PlatformNPU()
    platform.set_rng_state("state")
    platform.set_rng_state("state", 1)
    assert runtime.calls == [
        ("set_rng_state", ("state", None), {}),
        ("set_rng_state", ("state", 1), {}),
    ]
    assert platform.get_rng_state() == "rng-state"
    assert platform.get_rng_state(1) == "rng-state"
    assert runtime.calls[-1] == ("get_rng_state", (1,), {})
    assert platform.manual_seed(7) == ("seed", 7)
    assert platform.manual_seed_all(8) == ("seed-all", 8)
    assert platform.initial_seed() == 1234
    assert platform.Stream == "stream-type"
    assert platform.stream("s") == ("stream", "s")
    assert platform.set_stream("s") == ("set_stream", "s")
    assert platform.current_stream(0) == ("current_stream", 0)
    assert platform.default_stream(0) == ("default_stream", 0)
    assert platform.Event == "event-type"


def test_memory_accessors_forward_to_npu(monkeypatch):
    runtime = _FakeNpuRuntime()
    _patch_npu(monkeypatch, runtime)
    platform = PlatformNPU()
    assert platform.empty_cache() == "empty"
    assert platform.memory_allocated(0) == 24
    assert platform.max_memory_allocated(0) == 48
    assert platform.reset_max_memory_allocated(0) == "reset-alloc"
    assert platform.memory_cached(0) == 64
    assert platform.max_memory_cached(0) == 128
    assert platform.reset_max_memory_cached(0) == "reset-cached"
    assert platform.memory_stats(0) == {"device": 0}
    assert platform.reset_peak_memory_stats(0) == "reset-peak"
    assert platform.memory_reserved(0) == 96
    assert platform.max_memory_reserved(0) == 192
    assert platform.total_memory(0) == 2048
    assert platform.available_memory(0) == 2024
    assert runtime.calls == [
        ("memory_allocated", (0,), {}),
        ("get_device_properties", (0,), {}),
        ("get_device_properties", (0,), {}),
        ("memory_allocated", (0,), {}),
    ]


def test_dtype_and_amp_support_follow_npu_availability(monkeypatch):
    unavailable = _FakeNpuRuntime(available=False)
    _patch_npu(monkeypatch, unavailable)
    platform = PlatformNPU()
    assert platform.is_bf16_supported() is False
    assert platform.is_fp16_supported() is False
    assert platform.supported_dtypes() == [torch.float]

    available = _FakeNpuRuntime()
    _patch_npu(monkeypatch, available)
    assert platform.is_bf16_supported() is True
    assert platform.is_fp16_supported() is True
    assert platform.supported_dtypes() == [torch.float, torch.half, torch.bfloat16]
    assert platform.amp() == "npu-amp"

    without_amp = _FakeNpuRuntime(with_amp=False)
    _patch_npu(monkeypatch, without_amp)
    assert platform.amp() is None


def test_random_forwards_to_torch_module():
    assert PlatformNPU().random() is torch.random


def test_range_pop_logs_warning_on_empty_stack(monkeypatch, caplog):
    runtime = _FakeNpuRuntime()
    _patch_npu(monkeypatch, runtime)
    platform = PlatformNPU()
    assert platform.range_pop() is None
    assert any(
        "Attempted to pop NVTX range from empty stack" in record.message
        for record in caplog.records
    )


def test_mstx_ranges_push_and_pop_range_ids(monkeypatch):
    runtime = _FakeNpuRuntime()
    _patch_npu(monkeypatch, runtime)
    platform = PlatformNPU()
    assert platform.range("msg") == ("range", "msg")
    assert platform.range_push("first") == ("push", "first")
    assert platform.range_push("second") == ("push", "second")
    assert platform.range_pop() == "pop"
    assert runtime.mstx.calls == [
        ("range", "msg"),
        ("start", "first", 1),
        ("start", "second", 2),
        ("end", 2),
    ]


def test_lazy_call_is_triton_and_graph_helpers(monkeypatch):
    runtime = _FakeNpuRuntime()
    _patch_npu(monkeypatch, runtime)
    platform = PlatformNPU()
    assert platform.lazy_call("callback") is None
    assert platform.is_triton_supported() is None
    assert platform.graph_pool_handle() == "pool-handle"
    graph = platform.create_graph()
    assert platform.capture_to_graph(graph) == ("capture", graph, None, None)
    platform.replay_graph(graph)
    assert graph.replayed is True


def test_tensor_type_properties_and_device_checks(monkeypatch):
    runtime = _FakeNpuRuntime()
    _patch_npu(monkeypatch, runtime)
    platform = PlatformNPU()
    assert platform.BFloat16Tensor == "bf16-tensor"
    assert platform.ByteTensor == "byte-tensor"
    assert platform.DoubleTensor == "double-tensor"
    assert platform.FloatTensor == "float-tensor"
    assert platform.HalfTensor == "half-tensor"
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

    tensor = _FakeTensor("npu:0")
    assert platform.pin_memory(tensor) is tensor
    assert platform.is_pinned(tensor) is True
    assert platform.on_accelerator(tensor) is True
    assert platform.on_accelerator(torch.tensor([1.0])) is False


def test_extension_env_and_monitoring_helpers(monkeypatch):
    runtime = _FakeNpuRuntime()
    _patch_npu(monkeypatch, runtime)
    platform = PlatformNPU()
    from torch.utils.cpp_extension import BuildExtension

    assert platform.build_extension() is BuildExtension
    current_env = {}
    platform.set_visible_devices_envs(current_env, [1, 3])
    assert current_env == {"ASCEND_RT_VISIBLE_DEVICES": "1,3"}
    assert platform.get_compile_backend() is None
    assert platform.set_compile_backend("inductor") is None
    assert platform.temperature() is None
    assert platform.power_draw() is None
    assert platform.utilization() is None
    assert platform.clock_rate() is None


def test_module_import_binds_torch_npu_contrib_transfer():
    torch_npu = types.ModuleType("torch_npu")
    contrib = types.ModuleType("torch_npu.contrib")
    contrib.transfer_to_npu = "transfer-hook"
    torch_npu.contrib = contrib
    with pytest.MonkeyPatch.context() as context:
        context.setitem(sys.modules, "torch_npu", torch_npu)
        context.setitem(sys.modules, "torch_npu.contrib", contrib)
        importlib.reload(platform_npu)
        assert platform_npu.PlatformNPU().device_name() == "npu"
    importlib.reload(platform_npu)
