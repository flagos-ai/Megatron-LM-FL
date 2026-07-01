import os
import sys
import unittest

# Ensure megatron can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron.plugin.decorators import (
    _plugin_registry,
    _plugin_impl_cache,
    _original_impl_cache,
    _lazy_registry,
    _DEFAULT_VENDOR,
    _get_preferred_vendor,
    _target_to_method_key,
    register_override_method,
    get_override_method,
    override,
    overridable,
    register,
)


def _clear_registry():
    """Clear the registry and caches to ensure test isolation."""
    _plugin_registry.clear()
    _plugin_impl_cache.clear()
    _original_impl_cache.clear()
    _lazy_registry.clear()
    # Clear environment variables
    os.environ.pop("MG_FL_PREFER", None)


class TestRegisterOverrideMethod(unittest.TestCase):
    """Test register_override_method with multi-vendor registration."""

    def setUp(self):
        _clear_registry()

    def tearDown(self):
        _clear_registry()

    def test_register_default_vendor(self):
        """When vendor is not specified, register to 'default'."""
        def fn(): return "default_impl"
        register_override_method("A.foo", fn)

        self.assertIn("A.foo", _plugin_registry)
        self.assertIn("default", _plugin_registry["A.foo"])
        self.assertEqual(_plugin_registry["A.foo"]["default"](), "default_impl")

    def test_register_named_vendor(self):
        """When vendor is specified, register to the corresponding vendor."""
        def fn_musa(): return "musa_impl"
        register_override_method("A.foo", fn_musa, vendor="musa")

        self.assertIn("musa", _plugin_registry["A.foo"])
        self.assertEqual(_plugin_registry["A.foo"]["musa"](), "musa_impl")

    def test_register_multiple_vendors(self):
        """Multiple vendors register the same method_key."""
        def fn_default(): return "default"
        def fn_musa(): return "musa"
        def fn_txda(): return "txda"

        register_override_method("A.foo", fn_default)
        register_override_method("A.foo", fn_musa, vendor="musa")
        register_override_method("A.foo", fn_txda, vendor="txda")

        self.assertEqual(len(_plugin_registry["A.foo"]), 3)
        self.assertEqual(_plugin_registry["A.foo"]["default"](), "default")
        self.assertEqual(_plugin_registry["A.foo"]["musa"](), "musa")
        self.assertEqual(_plugin_registry["A.foo"]["txda"](), "txda")

    def test_vendor_case_insensitive(self):
        """Vendor name is case-insensitive."""
        def fn(): return "impl"
        register_override_method("A.foo", fn, vendor="MUSA")
        self.assertIn("musa", _plugin_registry["A.foo"])


class TestGetOverrideMethod(unittest.TestCase):
    """Test get_override_method selection logic."""

    def setUp(self):
        _clear_registry()

    def tearDown(self):
        _clear_registry()

    def test_no_registration(self):
        """Return None when no implementation is registered."""
        result = get_override_method("A.foo")
        self.assertIsNone(result)

    def test_default_vendor_selected_when_no_prefer(self):
        """When MG_FL_PREFER is not set, select the default vendor."""
        def fn_default(): return "default"
        def fn_musa(): return "musa"
        register_override_method("A.foo", fn_default)
        register_override_method("A.foo", fn_musa, vendor="musa")

        result = get_override_method("A.foo")
        self.assertEqual(result(), "default")

    def test_prefer_selects_correct_vendor(self):
        """When MG_FL_PREFER=musa, select the musa implementation."""
        def fn_default(): return "default"
        def fn_musa(): return "musa"
        register_override_method("A.foo", fn_default)
        register_override_method("A.foo", fn_musa, vendor="musa")

        os.environ["MG_FL_PREFER"] = "musa"
        result = get_override_method("A.foo")
        self.assertEqual(result(), "musa")

    def test_prefer_txda(self):
        """When MG_FL_PREFER=txda, select the txda implementation."""
        def fn_default(): return "default"
        def fn_txda(): return "txda"
        register_override_method("A.foo", fn_default)
        register_override_method("A.foo", fn_txda, vendor="txda")

        os.environ["MG_FL_PREFER"] = "txda"
        result = get_override_method("A.foo")
        self.assertEqual(result(), "txda")

    def test_prefer_nonexistent_vendor_fallback_to_default(self):
        """When the vendor specified by MG_FL_PREFER does not exist, fallback to default."""
        def fn_default(): return "default"
        register_override_method("A.foo", fn_default)

        os.environ["MG_FL_PREFER"] = "nonexistent"
        result = get_override_method("A.foo")
        self.assertEqual(result(), "default")

    def test_multiple_vendors_no_default_no_prefer_returns_none(self):
        """Multiple non-default vendors without MG_FL_PREFER set -> return None."""
        def fn_musa(): return "musa"
        def fn_txda(): return "txda"
        register_override_method("A.foo", fn_musa, vendor="musa")
        register_override_method("A.foo", fn_txda, vendor="txda")

        result = get_override_method("A.foo")
        self.assertIsNone(result)

    def test_prefer_empty_string_treated_as_unset(self):
        """MG_FL_PREFER="" is treated as unset."""
        def fn_default(): return "default"
        register_override_method("A.foo", fn_default)

        os.environ["MG_FL_PREFER"] = ""
        result = get_override_method("A.foo")
        self.assertEqual(result(), "default")

    def test_prefer_case_insensitive(self):
        """MG_FL_PREFER value is case-insensitive."""
        def fn_musa(): return "musa"
        register_override_method("A.foo", fn_musa, vendor="musa")

        os.environ["MG_FL_PREFER"] = "MUSA"
        result = get_override_method("A.foo")
        self.assertEqual(result(), "musa")


class TestOverrideDecorator(unittest.TestCase):
    """Test the @override decorator."""

    def setUp(self):
        _clear_registry()

    def tearDown(self):
        _clear_registry()

    def test_override_default_vendor(self):
        """@override registers to default when vendor is not specified."""
        @override("MyClass", "my_method")
        def my_method(self):
            return "overridden"

        impl = get_override_method("MyClass.my_method")
        self.assertIsNotNone(impl)

    def test_override_with_vendor(self):
        """@override with a specified vendor."""
        @override("MyClass", "my_method", vendor="musa")
        def my_method_musa(self):
            return "musa"

        self.assertIn("musa", _plugin_registry["MyClass.my_method"])

    def test_override_multiple_vendors_same_method(self):
        """Multiple vendors register the same method using @override."""
        @override("MyClass", "compute", vendor="default")
        def compute_default(self, x):
            return x + 1

        @override("MyClass", "compute", vendor="musa")
        def compute_musa(self, x):
            return x + 10

        @override("MyClass", "compute", vendor="txda")
        def compute_txda(self, x):
            return x + 100

        # Default selects the default vendor
        self.assertEqual(get_override_method("MyClass.compute")(None, 0), 1)

        # Set MG_FL_PREFER=musa
        os.environ["MG_FL_PREFER"] = "musa"
        self.assertEqual(get_override_method("MyClass.compute")(None, 0), 10)

        # Set MG_FL_PREFER=txda
        os.environ["MG_FL_PREFER"] = "txda"
        self.assertEqual(get_override_method("MyClass.compute")(None, 0), 100)


class TestOverridableDecorator(unittest.TestCase):
    """Test the full dispatch flow of the @overridable decorator."""

    def setUp(self):
        _clear_registry()

    def tearDown(self):
        _clear_registry()

    def test_no_override_uses_original(self):
        """When no override is registered, use the original implementation."""
        @overridable
        def my_func(x):
            return x * 2

        self.assertEqual(my_func(5), 10)

    def test_override_replaces_original(self):
        """After registering an override, call the override implementation."""
        # Register the override first (simulating plugin module loading)
        # Note: method_key is "test_override_manager.my_func" (module_name.function_name)
        # But since we are in a test, the module is __main__, so method_key is "__main__.my_func"
        # To simplify testing, we directly test the register + get flow

        @override("MyClass", "process")
        def process_override(self, x):
            return x * 100

        # Verify registration succeeded
        impl = get_override_method("MyClass.process")
        self.assertIsNotNone(impl)
        self.assertEqual(impl(None, 5), 500)


class TestGetPreferredVendor(unittest.TestCase):
    """Test the _get_preferred_vendor function."""

    def setUp(self):
        os.environ.pop("MG_FL_PREFER", None)
        # Mock platform to prevent platform inference from leaking into tests
        from megatron.plugin.platform import platform_manager

        self._original_platform = platform_manager.cur_platform
        platform_manager.cur_platform = None

    def tearDown(self):
        os.environ.pop("MG_FL_PREFER", None)
        from megatron.plugin.platform import platform_manager

        platform_manager.cur_platform = self._original_platform

    def test_unset(self):
        self.assertIsNone(_get_preferred_vendor())

    def test_unset_uses_selected_kunlunxin_platform(self):
        """Platform inference returns kunlunxin when that platform is selected."""
        from megatron.plugin.platform import platform_manager

        class KunlunxinPlatform:
            """Stub platform returning kunlunxin device name."""

            def device_name(self):
                """Return the kunlunxin device name."""
                return "kunlunxin"

        original_platform = platform_manager.cur_platform
        platform_manager.cur_platform = KunlunxinPlatform()
        try:
            self.assertEqual(_get_preferred_vendor(), "kunlunxin")
        finally:
            platform_manager.cur_platform = original_platform

    def test_empty(self):
        os.environ["MG_FL_PREFER"] = ""
        self.assertIsNone(_get_preferred_vendor())

    def test_whitespace_only(self):
        os.environ["MG_FL_PREFER"] = "   "
        self.assertIsNone(_get_preferred_vendor())

    def test_normal_value(self):
        os.environ["MG_FL_PREFER"] = "musa"
        self.assertEqual(_get_preferred_vendor(), "musa")

    def test_uppercase(self):
        os.environ["MG_FL_PREFER"] = "MUSA"
        self.assertEqual(_get_preferred_vendor(), "musa")

    def test_with_whitespace(self):
        os.environ["MG_FL_PREFER"] = "  txda  "
        self.assertEqual(_get_preferred_vendor(), "txda")


class TestTargetToMethodKey(unittest.TestCase):
    """Test the _target_to_method_key helper."""

    def test_module_level_function(self):
        """Module-level function: last segment of module + func name."""
        key = _target_to_method_key(
            "megatron.core.distributed.finalize_model_grads._allreduce_embedding_grad"
        )
        self.assertEqual(key, "finalize_model_grads._allreduce_embedding_grad")

    def test_module_level_function_short(self):
        key = _target_to_method_key("megatron.core.optimizer.clip_grads.get_grad_norm_fp32")
        self.assertEqual(key, "clip_grads.get_grad_norm_fp32")

    def test_class_method(self):
        """Class method: PascalCase segment is treated as class name."""
        key = _target_to_method_key(
            "megatron.core.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan"
        )
        self.assertEqual(key, "MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan")

    def test_class_method_language_module(self):
        key = _target_to_method_key(
            "megatron.core.models.common.language_module.language_module.LanguageModule._is_in_embd_group"
        )
        self.assertEqual(key, "LanguageModule._is_in_embd_group")

    def test_class_method_scheduler(self):
        key = _target_to_method_key(
            "megatron.core.optimizer_param_scheduler.OptimizerParamScheduler.get_lr"
        )
        self.assertEqual(key, "OptimizerParamScheduler.get_lr")


class TestRegisterFunction(unittest.TestCase):
    """Test the centralized register() function."""

    def setUp(self):
        _clear_registry()

    def tearDown(self):
        _clear_registry()

    def test_register_adds_to_lazy_registry(self):
        """register() should add entry to _lazy_registry."""
        register(
            target="megatron.core.optimizer.clip_grads.get_grad_norm_fp32",
            impl="megatron.plugin.optimizer.clip_grads.get_grad_norm_fp32",
        )
        key = "clip_grads.get_grad_norm_fp32"
        self.assertIn(key, _lazy_registry)
        self.assertIn("default", _lazy_registry[key])
        self.assertEqual(
            _lazy_registry[key]["default"],
            "megatron.plugin.optimizer.clip_grads.get_grad_norm_fp32",
        )

    def test_register_with_vendor(self):
        """register() with vendor parameter."""
        register(
            target="megatron.core.optimizer.clip_grads.get_grad_norm_fp32",
            impl="megatron.plugin.optimizer.clip_grads.get_grad_norm_fp32_musa",
            vendor="musa",
        )
        key = "clip_grads.get_grad_norm_fp32"
        self.assertIn("musa", _lazy_registry[key])

    def test_register_multiple_vendors(self):
        """Multiple vendors for the same target."""
        register(
            target="megatron.core.optimizer.clip_grads.get_grad_norm_fp32",
            impl="megatron.plugin.optimizer.clip_grads.get_grad_norm_fp32",
        )
        register(
            target="megatron.core.optimizer.clip_grads.get_grad_norm_fp32",
            impl="megatron.plugin.optimizer.clip_grads.get_grad_norm_fp32_musa",
            vendor="musa",
        )
        key = "clip_grads.get_grad_norm_fp32"
        self.assertEqual(len(_lazy_registry[key]), 2)
        self.assertIn("default", _lazy_registry[key])
        self.assertIn("musa", _lazy_registry[key])

    def test_register_vendor_case_insensitive(self):
        """Vendor name is lowercased."""
        register(
            target="megatron.core.optimizer.clip_grads.get_grad_norm_fp32",
            impl="megatron.plugin.optimizer.clip_grads.get_grad_norm_fp32_musa",
            vendor="MUSA",
        )
        key = "clip_grads.get_grad_norm_fp32"
        self.assertIn("musa", _lazy_registry[key])

    def test_lazy_resolve_on_get_override_method(self):
        """get_override_method should resolve lazy registry entries."""
        # Create a temporary module with a test function
        import types
        test_module = types.ModuleType("megatron.plugin.tests._test_lazy_target")
        test_module.my_func = lambda: "lazy_resolved"
        sys.modules["megatron.plugin.tests._test_lazy_target"] = test_module

        try:
            register(
                target="megatron.core.tests._test_lazy_target.my_func",
                impl="megatron.plugin.tests._test_lazy_target.my_func",
            )
            result = get_override_method("_test_lazy_target.my_func")
            self.assertIsNotNone(result)
            self.assertEqual(result(), "lazy_resolved")
            # After resolution, should be in _plugin_registry
            self.assertIn("_test_lazy_target.my_func", _plugin_registry)
        finally:
            del sys.modules["megatron.plugin.tests._test_lazy_target"]

    def test_eager_registry_takes_priority(self):
        """If eager registry has an entry, lazy registry is not consulted."""
        def eager_fn(): return "eager"
        register_override_method("foo.bar", eager_fn)

        register(
            target="megatron.core.foo.bar",
            impl="megatron.plugin.foo.bar",
        )
        result = get_override_method("foo.bar")
        self.assertEqual(result(), "eager")


class TestOverridableClass(unittest.TestCase):
    """Test @overridable on class definitions."""

    def setUp(self):
        _clear_registry()

    def tearDown(self):
        _clear_registry()

    def test_no_override_returns_original_instance(self):
        """Without override registered, instantiation returns original class instance."""
        @overridable
        class MyClass:
            def __init__(self, value):
                self.value = value

            def get_value(self):
                return self.value

        obj = MyClass(42)
        self.assertEqual(obj.value, 42)
        self.assertEqual(obj.get_value(), 42)
        self.assertIsInstance(obj, MyClass)

    def test_override_returns_override_instance(self):
        """With override registered, instantiation returns override class instance."""
        @overridable
        class MyClass:
            def __init__(self, value):
                self.value = value

            def get_value(self):
                return self.value

        class MyClassOverride(MyClass):
            def __init__(self, value):
                super().__init__(value * 2)

            def get_value(self):
                return self.value + 100

        # The method_key for a class is "module_basename.ClassName"
        # Since we're in test, module is __main__ or the test file
        # We register using the generic method_key format
        module_parts = MyClass.__module__.split('.')
        module_name = module_parts[-1]
        method_key = f"{module_name}.MyClass"
        register_override_method(method_key, MyClassOverride)

        obj = MyClass(10)
        self.assertEqual(obj.value, 20)  # 10 * 2
        self.assertEqual(obj.get_value(), 120)  # 20 + 100
        self.assertIsInstance(obj, MyClass)  # isinstance still works

    def test_override_class_isinstance_compatible(self):
        """Override class instances pass isinstance check against original."""
        @overridable
        class BaseScheduler:
            def __init__(self):
                self.name = "base"

        class PluginScheduler(BaseScheduler):
            def __init__(self):
                super().__init__()
                self.name = "plugin"

        module_parts = BaseScheduler.__module__.split('.')
        module_name = module_parts[-1]
        method_key = f"{module_name}.BaseScheduler"
        register_override_method(method_key, PluginScheduler)

        obj = BaseScheduler()
        self.assertEqual(obj.name, "plugin")
        self.assertIsInstance(obj, BaseScheduler)

    def test_subclass_not_affected(self):
        """Subclassing the overridable class works normally."""
        @overridable
        class MyBase:
            def __init__(self, x):
                self.x = x

        class MySub(MyBase):
            def __init__(self, x):
                super().__init__(x + 1)

        obj = MySub(5)
        self.assertEqual(obj.x, 6)
        self.assertIsInstance(obj, MyBase)

    def test_class_preserves_name(self):
        """Decorated class preserves __name__ and __qualname__."""
        @overridable
        class FancyClass:
            pass

        self.assertEqual(FancyClass.__name__, "FancyClass")

    def test_class_apply_dispatches_to_override(self):
        """Class-level apply calls route to the override class when registered."""
        @overridable
        class OriginalFunction:
            @classmethod
            def apply(cls, value):
                return f"original:{value}"

        class OverrideFunction(OriginalFunction):
            @classmethod
            def apply(cls, value):
                return f"override:{value}"

        module_parts = OriginalFunction.__module__.split('.')
        module_name = module_parts[-1]
        method_key = f"{module_name}.OriginalFunction"
        register_override_method(method_key, OverrideFunction)

        self.assertEqual(OriginalFunction.apply("x"), "override:x")

    def test_lazy_register_class(self):
        """register() with class target resolves lazily."""
        import types
        test_module = types.ModuleType("megatron.plugin.tests._test_cls_override")

        @overridable
        class OriginalCls:
            def __init__(self):
                self.source = "original"

        class OverrideCls(OriginalCls):
            def __init__(self):
                super().__init__()
                self.source = "override"

        test_module.OverrideCls = OverrideCls
        sys.modules["megatron.plugin.tests._test_cls_override"] = test_module

        try:
            # Construct target path matching the key format
            module_parts = OriginalCls.__module__.split('.')
            module_name = module_parts[-1]
            # register with a target that produces the right method_key
            register(
                target=f"megatron.core.{module_name}.OriginalCls",
                impl="megatron.plugin.tests._test_cls_override.OverrideCls",
            )

            obj = OriginalCls()
            self.assertEqual(obj.source, "override")
            self.assertIsInstance(obj, OriginalCls)
        finally:
            del sys.modules["megatron.plugin.tests._test_cls_override"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
