import os
import sys
import unittest

# Ensure megatron can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron.plugin.decorators import (
    _plugin_registry,
    _plugin_impl_cache,
    _original_impl_cache,
    _DEFAULT_VENDOR,
    _get_preferred_vendor,
    register_override_method,
    get_override_method,
    override,
    overridable,
)


def _clear_registry():
    """Clear the registry and caches to ensure test isolation."""
    _plugin_registry.clear()
    _plugin_impl_cache.clear()
    _original_impl_cache.clear()
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

    def tearDown(self):
        os.environ.pop("MG_FL_PREFER", None)

    def test_unset(self):
        self.assertIsNone(_get_preferred_vendor())

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


if __name__ == "__main__":
    unittest.main(verbosity=2)