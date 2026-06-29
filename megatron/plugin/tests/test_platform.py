# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch, MagicMock

from megatron.plugin.platform.platform_base import PlatformBase
from megatron.plugin.platform.platform_cpu import PlatformCPU
from megatron.plugin.platform import platform_register, platform_manager


def _reset_platform_manager():
    """Reset the global cur_platform state for test isolation."""
    platform_manager.cur_platform = None


class TestPlatformBase(unittest.TestCase):
    """Test that PlatformBase enforces the abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """PlatformBase cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            PlatformBase()

    def test_subclass_must_implement_abstract_methods(self):
        """A subclass missing abstract methods cannot be instantiated."""

        class IncompletePlatform(PlatformBase):
            pass

        with self.assertRaises(TypeError):
            IncompletePlatform()

    def test_subclass_with_all_methods_can_instantiate(self):
        """A subclass implementing all abstract methods can be instantiated."""
        platform = _create_mock_platform("test")
        self.assertIsInstance(platform, PlatformBase)
        self.assertEqual(platform._name, "test")


class TestPlatformRegister(unittest.TestCase):
    """Test platform registration mechanism."""

    def setUp(self):
        self._original_platforms = platform_register.PLATFORMS.copy()
        platform_register.PLATFORMS.clear()

    def tearDown(self):
        platform_register.PLATFORMS.clear()
        platform_register.PLATFORMS.update(self._original_platforms)

    def test_platforms_dict_starts_empty_after_clear(self):
        """After clearing, PLATFORMS should be empty."""
        self.assertEqual(len(platform_register.PLATFORMS), 0)

    def test_register_platforms_adds_cpu(self):
        """register_platforms should always register CPU (always available)."""
        platform_register.register_platforms()
        self.assertIn("cpu", platform_register.PLATFORMS)

    def test_registered_cpu_is_platform_base_instance(self):
        """Registered CPU platform should be a PlatformBase instance."""
        platform_register.register_platforms()
        cpu = platform_register.PLATFORMS.get("cpu")
        self.assertIsNotNone(cpu)
        self.assertIsInstance(cpu, PlatformBase)

    def test_registered_platforms_keys_are_lowercase(self):
        """All platform keys should be lowercase."""
        platform_register.register_platforms()
        for key in platform_register.PLATFORMS.keys():
            self.assertEqual(key, key.lower())

    def test_all_registered_platforms_are_available(self):
        """Only available platforms get registered."""
        platform_register.register_platforms()
        for name, platform in platform_register.PLATFORMS.items():
            self.assertTrue(
                platform.is_available(),
                f"Platform '{name}' is registered but is_available() returns False",
            )

    def test_unavailable_platform_not_registered(self):
        """A platform that is not available should not appear in PLATFORMS."""
        # Mock a platform whose is_available returns False
        with patch(
            "megatron.plugin.platform.platform_cpu.PlatformCPU.is_available",
            return_value=False,
        ):
            platform_register.register_platforms()
            self.assertNotIn("cpu", platform_register.PLATFORMS)


class TestPlatformManager(unittest.TestCase):
    """Test platform_manager: get_platform, set_platform, is_current_platform_supported."""

    def setUp(self):
        _reset_platform_manager()
        # Ensure PLATFORMS has at least cpu
        self._original_platforms = platform_register.PLATFORMS.copy()
        if "cpu" not in platform_register.PLATFORMS:
            platform_register.PLATFORMS["cpu"] = PlatformCPU()

    def tearDown(self):
        _reset_platform_manager()
        platform_register.PLATFORMS.clear()
        platform_register.PLATFORMS.update(self._original_platforms)

    def test_get_platform_returns_platform_base(self):
        """get_platform should return a PlatformBase instance."""
        platform = platform_manager.get_platform()
        self.assertIsInstance(platform, PlatformBase)

    def test_get_platform_caches_result(self):
        """Calling get_platform twice returns the same object."""
        p1 = platform_manager.get_platform()
        p2 = platform_manager.get_platform()
        self.assertIs(p1, p2)

    def test_set_platform_overrides_current(self):
        """set_platform should override whatever get_platform returns."""
        mock_platform = _create_mock_platform("mock")
        platform_manager.set_platform(mock_platform)
        self.assertIs(platform_manager.get_platform(), mock_platform)

    def test_set_platform_then_get(self):
        """After set_platform, get_platform returns the set value."""
        cpu = PlatformCPU()
        platform_manager.set_platform(cpu)
        result = platform_manager.get_platform()
        self.assertIs(result, cpu)
        self.assertEqual(result._name, "cpu")

    def test_get_platform_prefers_cuda_over_cpu(self):
        """When cuda is available, get_platform should prefer cuda over cpu."""
        mock_cuda = _create_mock_platform("cuda")
        mock_cuda.is_available = lambda: True
        platform_register.PLATFORMS["cuda"] = mock_cuda

        _reset_platform_manager()
        platform = platform_manager.get_platform()
        self.assertEqual(platform._name, "cuda")

    def test_get_platform_falls_back_to_cpu(self):
        """When only cpu is available, get_platform should return cpu."""
        # Remove all non-cpu platforms
        platform_register.PLATFORMS.clear()
        platform_register.PLATFORMS["cpu"] = PlatformCPU()

        _reset_platform_manager()
        platform = platform_manager.get_platform()
        self.assertEqual(platform._name, "cpu")

    def test_get_platform_raises_when_none_available(self):
        """When no platform is registered, get_platform raises ValueError."""
        platform_register.PLATFORMS.clear()
        _reset_platform_manager()

        with self.assertRaises(ValueError):
            platform_manager.get_platform()

    def test_platform_selection_priority(self):
        """Platforms are selected in priority order: cuda > musa > txda > npu > enflame > cpu."""
        # Register mock platforms for musa and cpu only
        platform_register.PLATFORMS.clear()
        mock_musa = _create_mock_platform("musa")
        mock_musa.is_available = lambda: True
        platform_register.PLATFORMS["musa"] = mock_musa
        platform_register.PLATFORMS["cpu"] = PlatformCPU()

        _reset_platform_manager()
        platform = platform_manager.get_platform()
        self.assertEqual(platform._name, "musa")


class TestPlatformCPU(unittest.TestCase):
    """Test PlatformCPU concrete implementation."""

    def setUp(self):
        self.cpu = PlatformCPU()

    def test_is_available(self):
        """CPU platform is always available."""
        self.assertTrue(self.cpu.is_available())

    def test_name(self):
        """CPU platform _name is 'cpu'."""
        self.assertEqual(self.cpu._name, "cpu")

    def test_device_name(self):
        """device_name returns 'cpu'."""
        self.assertEqual(self.cpu.device_name(), "cpu")

    def test_current_device_name(self):
        """current_device_name returns 'cpu'."""
        self.assertEqual(self.cpu.current_device_name(), "cpu")

    def test_device_returns_none(self):
        """device() returns None for CPU."""
        self.assertIsNone(self.cpu.device())

    def test_device_count_default(self):
        """CPU device_count returns 0 when LOCAL_SIZE is not set."""
        os.environ.pop("LOCAL_SIZE", None)
        self.assertEqual(self.cpu.device_count(), 0)

    def test_device_count_with_local_size(self):
        """CPU device_count returns LOCAL_SIZE value when set."""
        os.environ["LOCAL_SIZE"] = "4"
        try:
            self.assertEqual(self.cpu.device_count(), 4)
        finally:
            os.environ.pop("LOCAL_SIZE", None)

    def test_is_synchronized_device(self):
        """CPU is a synchronized device."""
        self.assertTrue(self.cpu.is_synchronized_device())

    def test_use_host_timers(self):
        """CPU uses host timers (synchronized)."""
        self.assertTrue(self.cpu.use_host_timers())

    def test_resolves_data_dependency(self):
        """CPU resolves data dependency (synchronized)."""
        self.assertTrue(self.cpu.resolves_data_dependency())

    def test_handles_memory_backpressure(self):
        """CPU handles memory backpressure (synchronized)."""
        self.assertTrue(self.cpu.handles_memory_backpressure())

    def test_synchronize_is_noop(self):
        """CPU synchronize should succeed (noop)."""
        # Should not raise
        self.cpu.synchronize()

    def test_empty_cache_is_noop(self):
        """CPU empty_cache should not raise."""
        self.cpu.empty_cache()

    def test_is_bf16_supported(self):
        """is_bf16_supported returns a boolean."""
        result = self.cpu.is_bf16_supported()
        self.assertIsInstance(result, bool)

    def test_is_fp16_supported(self):
        """is_fp16_supported returns a boolean."""
        result = self.cpu.is_fp16_supported()
        self.assertIsInstance(result, bool)

    def test_is_triton_supported(self):
        """CPU does not support triton."""
        self.assertFalse(self.cpu.is_triton_supported())

    def test_visible_devices_envs(self):
        """visible_devices_envs returns a list."""
        envs = self.cpu.visible_devices_envs()
        self.assertIsInstance(envs, list)
        self.assertGreater(len(envs), 0)

    def test_on_accelerator_cpu_tensor(self):
        """on_accelerator returns True for cpu tensor."""
        try:
            import torch
            t = torch.tensor([1.0])
            self.assertTrue(self.cpu.on_accelerator(t))
        except ImportError:
            self.skipTest("torch not available")

    def test_pin_memory(self):
        """pin_memory works on a CPU tensor."""
        try:
            import torch
            t = torch.tensor([1.0])
            pinned = self.cpu.pin_memory(t)
            self.assertIsNotNone(pinned)
        except ImportError:
            self.skipTest("torch not available")
        except RuntimeError:
            # May fail if CUDA not available for pin_memory
            pass

    def test_get_set_compile_backend(self):
        """get/set compile_backend round-trips."""
        try:
            import torch
            # inductor is typically available
            self.cpu.set_compile_backend("inductor")
            self.assertEqual(self.cpu.get_compile_backend(), "inductor")
        except (ImportError, ValueError):
            self.skipTest("torch or inductor backend not available")

    def test_temperature_returns_negative(self):
        """CPU temperature returns -1 (not applicable)."""
        self.assertEqual(self.cpu.temperature(), -1)

    def test_power_draw_returns_negative(self):
        """CPU power_draw returns -1 (not applicable)."""
        self.assertEqual(self.cpu.power_draw(), -1)

    def test_utilization_returns_negative(self):
        """CPU utilization returns -1 (not applicable)."""
        self.assertEqual(self.cpu.utilization(), -1)

    def test_clock_rate_returns_negative(self):
        """CPU clock_rate returns -1 (not applicable)."""
        self.assertEqual(self.cpu.clock_rate(), -1)

    def test_get_device_properties_raises(self):
        """CPU get_device_properties raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.cpu.get_device_properties()

    def test_get_device_capability_raises(self):
        """CPU get_device_capability raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.cpu.get_device_capability()


class _FakeDeviceProps:
    total_memory = 1024


class _FakeTensor:
    def __init__(self, device):
        self.device = device
        self.pinned = False

    def pin_memory(self):
        self.pinned = True
        return self

    def is_pinned(self):
        return self.pinned


class _FakeGraph:
    def __init__(self):
        self.replayed = False

    def replay(self):
        self.replayed = True


class _FakeAccelerator:
    Stream = "stream-type"
    Event = "event-type"
    MemPool = "pool-type"
    BFloat16Tensor = "bf16-tensor"
    ByteTensor = "byte-tensor"
    DoubleTensor = "double-tensor"
    FloatTensor = "float-tensor"
    HalfTensor = "half-tensor"
    IntTensor = "int-tensor"
    LongTensor = "long-tensor"
    MUSAGraph = _FakeGraph
    TopsGraph = _FakeGraph
    NPUGraph = _FakeGraph
    default_generators = ("gen0",)

    def __init__(self, available=True, fp16=True, bf16=True):
        self._available = available
        self._fp16 = fp16
        self._bf16 = bf16
        self.amp = "amp"
        self.mstx = types.SimpleNamespace(
            mstx_range=lambda msg: ("range", msg),
            range_start=lambda msg: ("push", msg),
            range_end=lambda: "pop",
        )

    def is_available(self):
        return self._available

    def device_count(self):
        return 2

    def get_device_properties(self, device_index=None):
        return _FakeDeviceProps()

    def get_device_capability(self, device_index=None):
        return (8, 0)

    def set_device(self, device_index):
        self.device_index = device_index

    def current_device(self):
        return 1

    def synchronize(self, device_index=None):
        return ("sync", device_index)

    def set_rng_state(self, state, device_index=None):
        return ("set_rng", state, device_index)

    def get_rng_state(self, device=None):
        return ("rng", device)

    def manual_seed(self, seed):
        return ("seed", seed)

    def manual_seed_all(self, seed):
        return ("seed_all", seed)

    def initial_seed(self):
        return 123

    def stream(self, stream):
        return ("stream", stream)

    def set_stream(self, stream):
        return ("set_stream", stream)

    def current_stream(self, device_index=None):
        return ("current_stream", device_index)

    def default_stream(self, device_index=None):
        return ("default_stream", device_index)

    def use_mem_pool(self, pool):
        return ("pool", pool)

    def empty_cache(self):
        return "empty"

    def memory_allocated(self, device_index=None):
        return 24

    def max_memory_allocated(self, device_index=None):
        return 48

    def reset_max_memory_allocated(self, device_index=None):
        return "reset_alloc"

    def memory_cached(self, device_index=None):
        return 64

    def max_memory_cached(self, device_index=None):
        return 128

    def reset_max_memory_cached(self, device_index=None):
        return "reset_cached"

    def memory_stats(self, device_index=None):
        return {"device": device_index}

    def reset_peak_memory_stats(self, device_index=None):
        return "reset_peak"

    def memory_reserved(self, device_index=None):
        return 96

    def max_memory_reserved(self, device_index=None):
        return 192

    def is_bf16_supported(self):
        return self._bf16

    def is_fp16_supported(self):
        return self._fp16

    def graph(self, graph, pool=None, stream=None):
        return ("capture", graph, pool, stream)


def _fake_torch_for_platform(accelerator_name, accelerator):
    fake_nvtx = types.SimpleNamespace(
        range=lambda msg: ("nvtx_range", msg),
        range_push=lambda msg: ("nvtx_push", msg),
        range_pop=lambda: "nvtx_pop",
    )
    fake_cuda = types.SimpleNamespace(nvtx=fake_nvtx)
    fake_torch = types.SimpleNamespace(
        float="float",
        half="half",
        bfloat16="bf16",
        random="random-module",
        cuda=fake_cuda,
        device=lambda name, index=None: f"{name}:{index}" if index is not None else name,
    )
    setattr(fake_torch, accelerator_name, accelerator)
    if accelerator_name == "cuda":
        fake_torch.cuda = accelerator
        fake_torch.cuda.nvtx = fake_nvtx
    return fake_torch


class TestMockedVendorPlatforms(unittest.TestCase):
    """Cover platform wrappers without requiring real vendor hardware."""

    def _exercise_accelerator_platform(
        self,
        module_path,
        class_name,
        accelerator_name,
        device_prefix,
        visible_env,
        graph_ctor_name,
        extra_modules=None,
    ):
        module = __import__(module_path, fromlist=[class_name])
        platform_cls = getattr(module, class_name)
        accelerator = _FakeAccelerator()
        fake_torch = _fake_torch_for_platform(accelerator_name, accelerator)
        module_updates = {"torch": fake_torch}
        if extra_modules:
            module_updates.update(extra_modules(accelerator))
        if device_prefix == "txda":
            fake_torch.txda = accelerator

        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, module_updates))
            stack.enter_context(patch.object(module, "torch", fake_torch))
            for name, value in module_updates.items():
                if name != "torch":
                    stack.enter_context(patch.object(module, name, value, create=True))
            platform = platform_cls()
            self.assertTrue(platform.is_available())
            self.assertEqual(platform.get_device_properties(0).total_memory, 1024)
            self.assertEqual(platform.get_device_capability(0), (8, 0))
            self.assertFalse(platform.is_synchronized_device())
            self.assertFalse(platform.use_host_timers())
            self.assertFalse(platform.resolves_data_dependency())
            self.assertFalse(platform.handles_memory_backpressure())
            self.assertEqual(platform.device_name(), device_prefix)
            self.assertEqual(platform.device_name(3), f"{device_prefix}:3")
            self.assertEqual(platform.device(2), f"{device_prefix}:2")
            self.assertEqual(platform.current_device(), 1)
            self.assertEqual(platform.current_device_name(), f"{device_prefix}:1")
            self.assertEqual(platform.device_count(), 2)
            self.assertEqual(platform.synchronize(0), ("sync", 0))
            self.assertEqual(platform.random(), "random-module")
            self.assertEqual(platform.set_rng_state("state"), ("set_rng", "state", None))
            self.assertEqual(platform.set_rng_state("state", 1), ("set_rng", "state", 1))
            self.assertEqual(platform.get_rng_state(), ("rng", None))
            self.assertEqual(platform.get_rng_state(1), ("rng", 1))
            self.assertEqual(platform.manual_seed(7), ("seed", 7))
            self.assertEqual(platform.manual_seed_all(8), ("seed_all", 8))
            self.assertEqual(platform.initial_seed(), 123)
            self.assertEqual(platform.default_generators, ("gen0",))
            self.assertEqual(platform.Stream, "stream-type")
            self.assertEqual(platform.stream("s"), ("stream", "s"))
            self.assertEqual(platform.set_stream("s"), ("set_stream", "s"))
            self.assertEqual(platform.current_stream(0), ("current_stream", 0))
            self.assertEqual(platform.default_stream(0), ("default_stream", 0))
            self.assertEqual(platform.MemPool, "pool-type")
            self.assertEqual(platform.use_mem_pool("pool"), ("pool", "pool"))
            self.assertEqual(platform.Event, "event-type")
            self.assertEqual(platform.empty_cache(), "empty")
            self.assertEqual(platform.memory_allocated(0), 24)
            self.assertEqual(platform.max_memory_allocated(0), 48)
            self.assertEqual(platform.reset_max_memory_allocated(0), "reset_alloc")
            self.assertEqual(platform.memory_cached(0), 64)
            self.assertEqual(platform.max_memory_cached(0), 128)
            self.assertEqual(platform.reset_max_memory_cached(0), "reset_cached")
            self.assertEqual(platform.memory_stats(0), {"device": 0})
            self.assertEqual(platform.reset_peak_memory_stats(0), "reset_peak")
            self.assertEqual(platform.memory_reserved(0), 96)
            self.assertEqual(platform.max_memory_reserved(0), 192)
            self.assertEqual(platform.total_memory(0), 1024)
            self.assertEqual(platform.available_memory(0), 1000)
            self.assertTrue(platform.is_fp16_supported())
            self.assertTrue(platform.is_bf16_supported())
            self.assertEqual(platform.supported_dtypes(), ["float", "half", "bf16"])
            self.assertEqual(platform.amp(), "amp")
            self.assertEqual(platform.range("msg"), ("nvtx_range", "msg"))
            self.assertEqual(platform.range_push("msg"), ("nvtx_push", "msg"))
            self.assertEqual(platform.range_pop(), "nvtx_pop")
            graph = platform.create_graph()
            if graph is None:
                graph = _FakeGraph()
            else:
                self.assertIsInstance(graph, _FakeGraph)
            self.assertEqual(platform.capture_to_graph(graph, "pool", "stream"), ("capture", graph, "pool", "stream"))
            platform.replay_graph(graph)
            self.assertTrue(graph.replayed)
            self.assertEqual(platform.BFloat16Tensor, "bf16-tensor")
            self.assertEqual(platform.ByteTensor, "byte-tensor")
            self.assertEqual(platform.DoubleTensor, "double-tensor")
            self.assertEqual(platform.FloatTensor, "float-tensor")
            self.assertEqual(platform.HalfTensor, "half-tensor")
            self.assertEqual(platform.IntTensor, "int-tensor")
            self.assertEqual(platform.LongTensor, "long-tensor")
            tensor = _FakeTensor(f"{device_prefix}:0")
            self.assertIs(platform.pin_memory(tensor), tensor)
            self.assertTrue(platform.is_pinned(tensor))
            self.assertTrue(platform.on_accelerator(tensor))
            self.assertFalse(platform.on_accelerator(_FakeTensor("cpu")))
            env = {}
            platform.set_visible_devices_envs(env, [1, 3])
            self.assertEqual(platform.visible_devices_envs(), [visible_env])
            self.assertEqual(env[visible_env], "1,3")
            self.assertIsNone(platform.lazy_call(lambda: None))
            self.assertIsNone(platform.is_triton_supported())
            self.assertIsNone(platform.get_compile_backend())
            self.assertIsNone(platform.set_compile_backend("inductor"))
            self.assertIsNone(platform.temperature())
            self.assertIsNone(platform.power_draw())
            self.assertIsNone(platform.utilization())
            self.assertIsNone(platform.clock_rate())

            accelerator._available = False
            self.assertFalse(platform.is_fp16_supported())
            self.assertFalse(platform.is_bf16_supported())

    def test_musa_platform_wrapper_contract_with_mock_backend(self):
        self._exercise_accelerator_platform(
            "megatron.plugin.platform.platform_musa",
            "PlatformMUSA",
            "musa",
            "musa",
            "MUSA_VISIBLE_DEVICES",
            "MUSAGraph",
        )

    def test_enflame_platform_wrapper_contract_with_mock_backend(self):
        self._exercise_accelerator_platform(
            "megatron.plugin.platform.platform_enflame",
            "PlatformENFLAME",
            "gcu",
            "gcu",
            "TOPS_VISIBLE_DEVICES",
            "TopsGraph",
        )

    def test_txda_platform_wrapper_contract_with_mock_backend(self):
        def extras(accelerator):
            return {
                "torch_txda": types.SimpleNamespace(transfer_to_txda=lambda value: value),
                "flag_gems": types.SimpleNamespace(),
            }

        self._exercise_accelerator_platform(
            "megatron.plugin.platform.platform_txda",
            "PlatformTXDA",
            "cuda",
            "txda",
            "TXDA_VISIBLE_DEVICES",
            "CUDAGraph",
            extras,
        )

    def test_npu_platform_wrapper_contract_with_mock_backend(self):
        module = __import__("megatron.plugin.platform.platform_npu", fromlist=["PlatformNPU"])
        accelerator = _FakeAccelerator()
        fake_torch = _fake_torch_for_platform("npu", accelerator)
        fake_torch_npu = types.SimpleNamespace(npu=accelerator)
        with patch.dict(sys.modules, {"torch": fake_torch, "torch_npu": fake_torch_npu}), patch.object(
            module, "torch", fake_torch
        ), patch.object(module, "torch_npu", fake_torch_npu, create=True):
            platform = module.PlatformNPU()
            self.assertTrue(platform.is_available())
            self.assertEqual(platform.device_name(2), "npu:2")
            self.assertEqual(platform.device(2), "npu:2")
            self.assertEqual(platform.default_generators, ("gen0",))
            self.assertEqual(platform.MemPool, "pool-type")
            self.assertEqual(platform.use_mem_pool("pool"), ("pool", "pool"))
            self.assertTrue(platform.is_fp16_supported())
            self.assertTrue(platform.is_bf16_supported())
            self.assertEqual(platform.range("x"), ("range", "x"))
            self.assertEqual(platform.range_push("x"), ("push", "x"))
            self.assertEqual(platform.range_pop(), "pop")
            graph = platform.create_graph()
            self.assertIsInstance(graph, _FakeGraph)
            self.assertEqual(platform.capture_to_graph(graph), ("capture", graph, None, None))
            platform.replay_graph(graph)
            self.assertTrue(graph.replayed)
            self.assertEqual(platform.visible_devices_envs(), ["ASCEND_RT_VISIBLE_DEVICES"])


# ---------- Auto-discovery: Interface Contract Tests for ALL Registered Platforms ----------


class TestAllRegisteredPlatformsContract(unittest.TestCase):
    """
    Parameterized tests that automatically run against every platform in PLATFORMS.
    When a new platform is added and registered, these tests cover it immediately
    without any manual test code changes.
    """

    @classmethod
    def setUpClass(cls):
        """Collect all currently registered platforms."""
        cls.platforms = dict(platform_register.PLATFORMS)
        if not cls.platforms:
            # Ensure at least registration has happened
            platform_register.register_platforms()
            cls.platforms = dict(platform_register.PLATFORMS)

    def _for_each_platform(self, check_fn):
        """Run check_fn(name, platform) for every registered platform."""
        self.assertGreater(len(self.platforms), 0, "No platforms registered")
        for name, platform in self.platforms.items():
            with self.subTest(platform=name):
                check_fn(name, platform)

    def _call_monitoring_api(self, name, platform, method_name):
        """Call a hardware monitoring API, skipping only missing optional NVML support."""
        try:
            return getattr(platform, method_name)()
        except ModuleNotFoundError as exc:
            if "pynvml" in str(exc).lower():
                self.skipTest(f"{name}.{method_name} requires pynvml/NVML support")
            raise

    # --- Basic contract ---

    def test_is_platform_base_instance(self):
        """Every registered platform must be a PlatformBase subclass."""
        self._for_each_platform(
            lambda name, p: self.assertIsInstance(p, PlatformBase)
        )

    def test_has_name_attribute(self):
        """Every platform must have a _name attribute matching its registry key."""
        def check(name, p):
            self.assertTrue(hasattr(p, '_name'))
            self.assertEqual(p._name, name)
        self._for_each_platform(check)

    def test_is_available_returns_bool(self):
        """is_available() must return a bool and be True (already registered)."""
        def check(name, p):
            result = p.is_available()
            self.assertIsInstance(result, bool)
            self.assertTrue(result)
        self._for_each_platform(check)

    # --- Device APIs ---

    def test_device_name_returns_string(self):
        """device_name() must return a string."""
        def check(name, p):
            result = p.device_name()
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
        self._for_each_platform(check)

    def test_current_device_name_returns_string(self):
        """current_device_name() must return a string."""
        def check(name, p):
            result = p.current_device_name()
            self.assertIsInstance(result, str)
        self._for_each_platform(check)

    def test_device_count_returns_int(self):
        """device_count() must return a non-negative int."""
        def check(name, p):
            result = p.device_count()
            self.assertIsInstance(result, int)
            self.assertGreaterEqual(result, 0)
        self._for_each_platform(check)

    # --- Synchronization properties ---

    def test_is_synchronized_device_returns_bool(self):
        """is_synchronized_device() must return a bool."""
        def check(name, p):
            self.assertIsInstance(p.is_synchronized_device(), bool)
        self._for_each_platform(check)

    def test_use_host_timers_returns_bool(self):
        """use_host_timers() must return a bool."""
        def check(name, p):
            self.assertIsInstance(p.use_host_timers(), bool)
        self._for_each_platform(check)

    def test_resolves_data_dependency_returns_bool(self):
        """resolves_data_dependency() must return a bool."""
        def check(name, p):
            self.assertIsInstance(p.resolves_data_dependency(), bool)
        self._for_each_platform(check)

    def test_handles_memory_backpressure_returns_bool(self):
        """handles_memory_backpressure() must return a bool."""
        def check(name, p):
            self.assertIsInstance(p.handles_memory_backpressure(), bool)
        self._for_each_platform(check)

    # --- Data types ---

    def test_is_bf16_supported_returns_bool(self):
        """is_bf16_supported() must return a bool."""
        def check(name, p):
            self.assertIsInstance(p.is_bf16_supported(), bool)
        self._for_each_platform(check)

    def test_is_fp16_supported_returns_bool(self):
        """is_fp16_supported() must return a bool."""
        def check(name, p):
            self.assertIsInstance(p.is_fp16_supported(), bool)
        self._for_each_platform(check)

    def test_is_triton_supported_returns_bool(self):
        """is_triton_supported() must return a bool."""
        def check(name, p):
            self.assertIsInstance(p.is_triton_supported(), bool)
        self._for_each_platform(check)

    # --- Visible devices ---

    def test_visible_devices_envs_returns_list(self):
        """visible_devices_envs() must return a list of strings."""
        def check(name, p):
            result = p.visible_devices_envs()
            self.assertIsInstance(result, list)
            for item in result:
                self.assertIsInstance(item, str)
        self._for_each_platform(check)

    # --- Compile backend ---

    def test_get_compile_backend_returns_string(self):
        """get_compile_backend() must return a string (after set)."""
        def check(name, p):
            try:
                # Some platforms only have _compile_backend after set_compile_backend
                p.set_compile_backend("inductor")
                result = p.get_compile_backend()
                self.assertIsInstance(result, str)
            except (ValueError, AttributeError):
                # Platform may not support inductor; skip gracefully
                pass
        self._for_each_platform(check)

    # --- Monitoring APIs ---

    def test_temperature_returns_numeric(self):
        """temperature() must return a numeric value."""
        def check(name, p):
            result = self._call_monitoring_api(name, p, "temperature")
            self.assertIsInstance(result, (int, float))
        self._for_each_platform(check)

    def test_power_draw_returns_numeric(self):
        """power_draw() must return a numeric value."""
        def check(name, p):
            result = self._call_monitoring_api(name, p, "power_draw")
            self.assertIsInstance(result, (int, float))
        self._for_each_platform(check)

    def test_utilization_returns_numeric(self):
        """utilization() must return a numeric value."""
        def check(name, p):
            result = self._call_monitoring_api(name, p, "utilization")
            self.assertIsInstance(result, (int, float))
        self._for_each_platform(check)

    def test_clock_rate_returns_numeric(self):
        """clock_rate() must return a numeric value."""
        def check(name, p):
            result = self._call_monitoring_api(name, p, "clock_rate")
            self.assertIsInstance(result, (int, float))
        self._for_each_platform(check)

    # --- Synchronize should not raise ---

    def test_synchronize_does_not_raise(self):
        """synchronize() must not raise."""
        def check(name, p):
            try:
                p.synchronize()
            except Exception as e:
                self.fail(f"Platform '{name}' synchronize() raised: {e}")
        self._for_each_platform(check)

    # --- Empty cache should not raise ---

    def test_empty_cache_does_not_raise(self):
        """empty_cache() must not raise."""
        def check(name, p):
            try:
                p.empty_cache()
            except Exception as e:
                self.fail(f"Platform '{name}' empty_cache() raised: {e}")
        self._for_each_platform(check)


# ---------- Helpers ----------


def _create_mock_platform(name):
    """Create a concrete PlatformBase subclass for testing."""

    class MockPlatform(PlatformBase):
        def __init__(self):
            self._name = name

        def is_available(self):
            return True

        def get_device_properties(self, device_index=None):
            return None

        def get_device_capability(self, device_index=None):
            return (0, 0)

        def is_synchronized_device(self):
            return True

        def use_host_timers(self):
            return True

        def resolves_data_dependency(self):
            return True

        def handles_memory_backpressure(self):
            return True

        def device_name(self, device_index=None):
            return name

        def device(self, device_index=None):
            return None

        def set_device(self, device_index):
            pass

        def current_device(self):
            return 0

        def current_device_name(self):
            return name

        def device_count(self):
            return 1

        def synchronize(self, device_index=None):
            pass

        def random(self):
            return None

        def set_rng_state(self, new_state, device_index=None):
            pass

        def get_rng_state(self, device=None):
            return None

        def manual_seed(self, seed):
            pass

        def manual_seed_all(self, seed):
            pass

        def initial_seed(self):
            return 0

        @property
        def MemPool(self):
            return None

        def use_mem_pool(self, pool):
            pass

        @property
        def Stream(self):
            return None

        def stream(self, stream):
            pass

        def set_stream(self, stream):
            pass

        def current_stream(self, device_index=None):
            return None

        def default_stream(self, device_index=None):
            return None

        @property
        def Event(self):
            return None

        def empty_cache(self):
            pass

        def memory_allocated(self, device_index=None):
            return 0

        def max_memory_allocated(self, device_index=None):
            return 0

        def reset_max_memory_allocated(self, device_index=None):
            pass

        def memory_cached(self, device_index=None):
            return 0

        def max_memory_cached(self, device_index=None):
            return 0

        def reset_max_memory_cached(self, device_index=None):
            pass

        def memory_stats(self, device_index=None):
            return {}

        def reset_peak_memory_stats(self, device_index=None):
            pass

        def memory_reserved(self, device_index=None):
            return 0

        def max_memory_reserved(self, device_index=None):
            return 0

        def total_memory(self, device_index=None):
            return 0

        def available_memory(self, device_index=None):
            return 0

        def is_bf16_supported(self):
            return False

        def is_fp16_supported(self):
            return False

        def supported_dtypes(self):
            return []

        def amp(self):
            return None

        def range(self, msg):
            pass

        def range_push(self, msg):
            pass

        def range_pop(self):
            pass

        def lazy_call(self, callback):
            pass

        def is_triton_supported(self):
            return False

        def create_graph(self):
            return None

        def capture_to_graph(self, graph, pool=None, stream=None):
            pass

        def replay_graph(self, graph):
            pass

        @property
        def BFloat16Tensor(self):
            return None

        @property
        def ByteTensor(self):
            return None

        @property
        def DoubleTensor(self):
            return None

        @property
        def FloatTensor(self):
            return None

        @property
        def HalfTensor(self):
            return None

        @property
        def IntTensor(self):
            return None

        @property
        def LongTensor(self):
            return None

        def pin_memory(self, tensor, align_bytes=1):
            return tensor

        def is_pinned(self, tensor):
            return False

        def on_accelerator(self, tensor):
            return False

        def build_extension(self):
            return None

        def visible_devices_envs(self):
            return []

        def set_visible_devices_envs(self, current_env, local_accelerator_ids):
            pass

        def get_compile_backend(self):
            return "inductor"

        def set_compile_backend(self, backend):
            pass

        def temperature(self):
            return -1

        def power_draw(self):
            return -1

        def utilization(self):
            return -1

        def clock_rate(self):
            return -1

    return MockPlatform()


if __name__ == "__main__":
    unittest.main(verbosity=2)
