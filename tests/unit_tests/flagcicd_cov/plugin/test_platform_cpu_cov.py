# Copyright (c) 2026, BAAI. All rights reserved.
"""PlatformCPU: import guards, psutil-backed memory accounting, RNG/stream/graph placeholders, env and compile-backend helpers."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.plugin.platform import platform_cpu
from megatron.plugin.platform.platform_cpu import PlatformCPU


class _FakePsutil:
    def __init__(self, rss, total, available):
        self.rss = rss
        self.total = total
        self.available = available

    def Process(self):
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=self.rss))

    def virtual_memory(self):
        return SimpleNamespace(total=self.total, available=self.available)


class _FakeTensor:
    def __init__(self, device):
        self.device = device

    def pin_memory(self):
        return self

    def is_pinned(self):
        return True


def test_module_import_tolerates_missing_torch():
    with pytest.MonkeyPatch.context() as context:
        context.setitem(sys.modules, "torch", None)
        importlib.reload(platform_cpu)
        assert platform_cpu.PlatformCPU().device_name() == "cpu"
    importlib.reload(platform_cpu)


def test_module_import_tracks_oneccl_availability():
    with pytest.MonkeyPatch.context() as context:
        context.setitem(sys.modules, "oneccl_bindings_for_pytorch", SimpleNamespace())
        importlib.reload(platform_cpu)
        assert platform_cpu.oneccl_imported_p is True
    importlib.reload(platform_cpu)
    assert platform_cpu.oneccl_imported_p is False


def test_init_records_process_rss_when_psutil_importable(monkeypatch):
    fake = _FakePsutil(rss=8000, total=1000000, available=400000)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert PlatformCPU().max_mem == 8000


def test_memory_accessors_track_rss_and_peak(monkeypatch):
    fake = _FakePsutil(rss=8000, total=1000000, available=400000)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    platform = PlatformCPU()
    fake.rss = 16000
    assert platform.get_rss() == 16000
    assert platform.max_mem == 16000
    fake.rss = 4000
    assert platform.memory_allocated() == 4000
    assert platform.memory_cached() == 4000
    assert platform.memory_reserved() == 4000
    assert platform.max_memory_allocated() == 16000
    assert platform.max_memory_cached() == 16000
    assert platform.max_memory_reserved() == 16000
    platform.reset_max_memory_allocated()
    assert platform.max_mem == 4000
    fake.rss = 12000
    assert platform.reset_rss() == 12000
    assert platform.max_mem == 12000
    platform.reset_max_memory_cached()
    platform.reset_peak_memory_stats()
    assert platform.max_mem == 12000


def test_memory_stats_reports_current_and_peak_rss(monkeypatch):
    fake = _FakePsutil(rss=8000, total=1000000, available=400000)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    platform = PlatformCPU()
    fake.rss = 12000
    assert platform.memory_stats() == {
        "allocated_bytes.all.current": 12000,
        "allocated_bytes.all.peak": 12000,
    }


def test_total_and_available_memory_read_virtual_memory(monkeypatch):
    fake = _FakePsutil(rss=8000, total=1000000, available=400000)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    platform = PlatformCPU()
    assert platform.total_memory() == 1000000
    assert platform.available_memory() == 400000


def test_rng_methods_forward_to_torch():
    platform = PlatformCPU()
    assert platform.random() is torch.random
    state = platform.get_rng_state()
    assert isinstance(state, torch.ByteTensor)
    platform.set_rng_state(state)
    assert torch.equal(platform.get_rng_state(), state)
    platform.manual_seed(7)
    assert platform.initial_seed() == 7
    platform.manual_seed_all(9)
    assert platform.initial_seed() == 9
    assert platform.default_generators is torch.default_generator
    assert platform.amp() is torch.cpu.amp


def test_stream_pool_and_event_helpers_return_placeholders():
    platform = PlatformCPU()
    assert platform.Stream is None
    assert platform.stream("s").__enter__() is None
    assert platform.set_stream("s") is None
    assert platform.current_stream() is None
    assert platform.default_stream() is None
    assert platform.MemPool is None
    assert platform.use_mem_pool("pool") is None
    assert platform.Event is None


def test_range_helpers_and_lazy_call():
    platform = PlatformCPU()
    assert platform.range("msg") is None
    assert platform.range_push("msg") is None
    assert platform.range_pop() is None
    calls = []
    assert platform.lazy_call(lambda: calls.append("ran") or "value") == "value"
    assert calls == ["ran"]


def test_supported_dtypes_follow_mkldnn_fp16_probe():
    probe = getattr(torch.ops.mkldnn, "_is_mkldnn_fp16_supported", None)
    fp16_supported = probe is not None and probe()
    platform = PlatformCPU()
    expected = [torch.float, torch.bfloat16]
    if fp16_supported:
        expected.append(torch.float16)
    assert platform.supported_dtypes() == expected


def test_graph_operations_return_placeholders():
    platform = PlatformCPU()
    assert platform.graph_pool_handle() is None
    assert platform.create_graph() is None
    with platform.capture_to_graph(platform.create_graph()):
        pass
    assert platform.replay_graph("graph") is None


def test_tensor_type_properties_forward_to_torch():
    platform = PlatformCPU()
    assert platform.BFloat16Tensor is torch.BFloat16Tensor
    assert platform.ByteTensor is torch.ByteTensor
    assert platform.DoubleTensor is torch.DoubleTensor
    assert platform.FloatTensor is torch.FloatTensor
    assert platform.HalfTensor is torch.HalfTensor
    assert platform.IntTensor is torch.IntTensor
    assert platform.LongTensor is torch.LongTensor


def test_on_accelerator_checks_tensor_device_prefix():
    platform = PlatformCPU()
    assert platform.on_accelerator(torch.tensor([1.0])) is True
    assert platform.on_accelerator(_FakeTensor(torch.device("cuda:0"))) is False
    assert platform.is_pinned(_FakeTensor(torch.device("cpu"))) is True
    tensor = _FakeTensor(torch.device("cpu"))
    assert platform.pin_memory(tensor) is tensor


def test_build_extension_returns_torch_build_extension():
    from torch.utils.cpp_extension import BuildExtension

    assert PlatformCPU().build_extension() is BuildExtension


def test_set_visible_devices_envs_writes_cuda_ids():
    current_env = {}
    PlatformCPU().set_visible_devices_envs(current_env, [1, 2])
    assert current_env == {"CUDA_VISIBLE_DEVICES": "1,2"}


def test_set_compile_backend_rejects_unsupported_backend(monkeypatch):
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    platform = PlatformCPU()
    platform.set_compile_backend("eager")
    assert platform.get_compile_backend() == "eager"
    with pytest.raises(ValueError, match="bogus not supported by cpu"):
        platform.set_compile_backend("bogus")


def test_current_device_reads_local_rank_environment(monkeypatch):
    platform = PlatformCPU()
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert platform.current_device() == "3"
    monkeypatch.delenv("LOCAL_RANK")
    assert platform.current_device() == 0
    assert platform.set_device(0) is None
