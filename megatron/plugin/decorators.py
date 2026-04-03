"""
Plugin decorator system for method replacement.

The decorator automatically detects the class and method context,
and looks up the implementation in plugin.

Multi-vendor support:
    Multiple vendors can register overrides for the same method via the
    ``vendor`` parameter of :func:`override`.  At runtime the environment
    variable ``MG_FL_PREFER`` selects which vendor's implementation is used.

    Example::

        export MG_FL_PREFER=musa      # prefer MUSA vendor implementations
        export MG_FL_PREFER=txda      # prefer TXDA vendor implementations

    When ``MG_FL_PREFER`` is unset (or empty), the "default" vendor is used.
"""

import functools
import importlib
import inspect
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Default vendor name used when @override does not specify a vendor
_DEFAULT_VENDOR = "default"

# Registry to store override methods
# Key format: "ClassName.method_name" -> { vendor_name: implementation }
_plugin_registry: dict[str, dict[str, Callable]] = {}

# Cache for override methods lookup results
# _plugin_impl_cache: stores functions that have override methods
# _original_impl_cache: stores functions that should use original implementation (no plugin found)
_plugin_impl_cache: dict[Callable, Callable] = {}
_original_impl_cache: set[Callable] = set()


def _get_preferred_vendor() -> Optional[str]:
    """Get the preferred vendor from MG_FL_PREFER environment variable.

    Returns:
        The vendor name string (lowercased), or None if not set.
    """
    vendor = os.environ.get("MG_FL_PREFER")
    if vendor is not None:
        vendor = vendor.strip().lower()
        if vendor == "":
            return None
    return vendor


def register_override_method(method_key: str, implementation: Callable,
                             vendor: str = _DEFAULT_VENDOR) -> None:
    """
    Register an override method for a method or function.

    Args:
        method_key: Unique key for the method/function
                    (e.g., "LanguageModule._is_in_embd_group" or "clip_grads.get_grad_norm_fp32")
        implementation: The implementation function
        vendor: Vendor name that provides this implementation (e.g., "musa", "txda").
                Defaults to "default".
    """
    vendor = vendor.lower()
    if method_key not in _plugin_registry:
        _plugin_registry[method_key] = {}
    _plugin_registry[method_key][vendor] = implementation
    logger.debug(f"Registered override method: {method_key} (vendor={vendor})")


def get_override_method(method_key: str) -> Optional[Callable]:
    """
    Get an override method for a method or function.

    Selection priority:
    1. If MG_FL_PREFER is set and a matching vendor implementation exists, use it.
    2. Otherwise fall back to the "default" vendor.
    3. Otherwise return None.

    Args:
        method_key: Unique key for the method/function

    Returns:
        The override method if available, None otherwise
    """
    vendor_map = _plugin_registry.get(method_key)
    if vendor_map is None:
        return None

    preferred = _get_preferred_vendor()

    # 1. Preferred vendor
    if preferred is not None and preferred in vendor_map:
        logger.debug(f"Using vendor '{preferred}' for {method_key}")
        return vendor_map[preferred]

    # 2. Default vendor
    if _DEFAULT_VENDOR in vendor_map:
        logger.debug(f"Using vendor '{_DEFAULT_VENDOR}' for {method_key}")
        return vendor_map[_DEFAULT_VENDOR]

    # 3. Multiple vendors but no preference / no default -- warn and return None
    if preferred is not None:
        logger.warning(
            f"MG_FL_PREFER='{preferred}' but no matching vendor for {method_key}. "
            f"Available vendors: {list(vendor_map.keys())}"
        )
    return None


def overridable(func: Callable) -> Callable:
    """
    Decorator to mark a method or function as replaceable by plugin.

    Usage in core code (for methods):
        @overridable
        def _is_in_embd_group(self):
            # Original implementation (fallback if no plugin)
            ...

    Usage in core code (for module-level functions):
        @overridable
        def get_grad_norm_fp32(...):
            # Original implementation (fallback if no plugin)
            ...

    The decorator automatically:
    1. For methods: Detects the class name and method name
    2. For functions: Uses module name and function name
    3. Looks up override method using the key
    4. Uses plugin if found, otherwise uses original implementation

    No parameters needed - everything is auto-detected!
    """
    # Save the original qualname at decoration time
    # This is crucial for inheritance: when a subclass calls a parent's method,
    # we need the qualname of the method as defined in the parent class, not the subclass
    # Example: If A defines m1() and B inherits A, B().m1() should use "A.m1" as the key
    original_qualname = func.__qualname__
    original_module = func.__module__

    # Determine if this is a method or function at decoration time
    # by inspecting the function signature
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    is_method = params and params[0] == 'self'

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check cache first - use func as key
        if func in _plugin_impl_cache:
            # Plugin implementation found and cached
            return _plugin_impl_cache[func](*args, **kwargs)
        elif func in _original_impl_cache:
            # Already checked, no plugin found - use original implementation
            return func(*args, **kwargs)

        # Cache miss: first time calling this function, need to compute method_key and lookup
        # Compute method_key only when needed (first call)
        if is_method:
            # It's a method - use the original qualname
            if '.' in original_qualname:
                # Extract class name from qualname (e.g., "A.m1" -> "A", "Outer.Inner.method" -> "Inner")
                parts = original_qualname.rsplit('.', 1)
                if len(parts) == 2:
                    class_path = parts[0]
                    method_name = parts[1]
                    # Get the actual class name (last part of class path, handles nested classes)
                    class_name = class_path.split('.')[-1]
                    method_key = f"{class_name}.{method_name}"
                else:
                    # Fallback if qualname format is unexpected
                    method_key = f"unknown.{func.__name__}"
            else:
                # Fallback if no class in qualname (shouldn't happen for methods)
                method_key = f"unknown.{func.__name__}"
        else:
            # It's a module-level function
            # Get the module name from the function's module
            # For megatron.core.optimizer.clip_grads, we want "clip_grads"
            module_parts = original_module.split('.')
            module_name = module_parts[-1] if module_parts else "unknown"
            function_name = func.__name__
            method_key = f"{module_name}.{function_name}"

        plugin_impl = get_override_method(method_key)

        # If not found, try to lazy import the plugin module
        if plugin_impl is None:
            try:
                # Try to import the corresponding plugin module
                # For megatron.core.distributed.finalize_model_grads -> megatron.plugin.distributed.finalize_model_grads
                # For megatron.core.optimizer.clip_grads -> megatron.plugin.optimizer.clip_grads
                if original_module.startswith("megatron.core."):
                    # Replace "megatron.core." with "megatron.plugin."
                    # e.g., megatron.core.distributed.xxx -> megatron.plugin.distributed.xxx
                    plugin_module = original_module.replace("megatron.core.", "megatron.plugin.", 1)
                    try:
                        importlib.import_module(plugin_module)
                        # Try again after import
                        plugin_impl = get_override_method(method_key)
                        if plugin_impl is not None:
                            logger.debug(f"Lazy loaded override method for {method_key}")
                    except (ImportError, ModuleNotFoundError):
                        # Plugin module doesn't exist, that's okay
                        pass
            except Exception as e:
                # Ignore any errors during lazy import
                logger.debug(f"Failed to lazy import plugin for {method_key}: {e}")

        # Cache the result
        if plugin_impl is not None:
            _plugin_impl_cache[func] = plugin_impl
            logger.debug(f"Using override method for {method_key}")
            return plugin_impl(*args, **kwargs)
        else:
            # Cache "not found" result to avoid repeated lookup
            _original_impl_cache.add(func)
            logger.debug(f"Using original implementation for {method_key}")
            # Use original implementation
            return func(*args, **kwargs)

    return wrapper


def override(class_or_module_name: str, method_or_function_name: str,
             vendor: str = _DEFAULT_VENDOR):
    """
    Decorator to register an override method.

    Usage in plugins (for methods, default vendor):
        @override("LanguageModule", "_is_in_embd_group")
        def _is_in_embd_group(self):
            # Plugin implementation
            ...

    Usage in plugins (for functions, with vendor):
        @override("clip_grads", "get_grad_norm_fp32", vendor="musa")
        def get_grad_norm_fp32(...):
            # MUSA-specific implementation
            ...

    When multiple vendors register the same method, set ``MG_FL_PREFER``
    to choose which vendor to use at runtime::

        export MG_FL_PREFER=musa

    Args:
        class_or_module_name: Class name (e.g., "LanguageModule") or module name (e.g., "clip_grads")
        method_or_function_name: Method name (e.g., "_is_in_embd_group") or function name
        vendor: Vendor name (e.g., "musa", "txda"). Defaults to "default".
    """
    def decorator(impl_func: Callable) -> Callable:
        method_key = f"{class_or_module_name}.{method_or_function_name}"
        register_override_method(method_key, impl_func, vendor=vendor)
        logger.info(f"Registered override method: {method_key} (vendor={vendor})")
        return impl_func
    return decorator
