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

Centralized registration (recommended):
    Instead of decorating each plugin function with ``@override``, declare all
    mappings in a single registry file using :func:`register`::

        # megatron/plugin/override_registry.py
        from megatron.plugin.decorators import register

        register(
            target="megatron.core.distributed.finalize_model_grads._allreduce_embedding_grad",
            impl="megatron.plugin.distributed.finalize_model_grads._allreduce_embedding_grad",
        )

    The ``impl`` module is lazily imported only when the overridden function is
    first called.
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

# Registry to store override methods (eagerly loaded implementations)
# Key format: "ClassName.method_name" -> { vendor_name: implementation }
_plugin_registry: dict[str, dict[str, Callable]] = {}

# Lazy registry: stores module paths for deferred import
# Key format: "ClassName.method_name" -> { vendor_name: "full.module.path.func_name" }
_lazy_registry: dict[str, dict[str, str]] = {}

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


def _resolve_lazy_impl(method_key: str) -> Optional[Callable]:
    """Resolve a lazy-registered implementation by importing its module.

    If the method_key exists in ``_lazy_registry``, select the appropriate
    vendor implementation (using the same priority as ``get_override_method``),
    import its module, retrieve the function object, register it into
    ``_plugin_registry``, and return it.

    Returns:
        The resolved implementation callable, or None if not in lazy registry.
    """
    if method_key not in _lazy_registry:
        return None

    vendor_map = _lazy_registry[method_key]

    # Determine which vendor to load
    preferred = _get_preferred_vendor()
    if preferred and preferred in vendor_map:
        impl_path = vendor_map[preferred]
    elif _DEFAULT_VENDOR in vendor_map:
        impl_path = vendor_map[_DEFAULT_VENDOR]
    elif preferred is None and len(vendor_map) == 1:
        impl_path = next(iter(vendor_map.values()))
    else:
        return None

    # Import the module and get the function
    try:
        rsplit = impl_path.rsplit(".", 1)
        if len(rsplit) != 2:
            logger.warning(f"Invalid impl path in lazy registry: {impl_path}")
            return None
        module_path, func_name = rsplit
        mod = importlib.import_module(module_path)
        impl_func = getattr(mod, func_name, None)
        if impl_func is None:
            logger.warning(
                f"Function '{func_name}' not found in module '{module_path}'"
            )
            return None

        # Register into the eager registry so future lookups skip lazy loading
        vendor_for_reg = preferred if (preferred and preferred in vendor_map) else _DEFAULT_VENDOR
        register_override_method(method_key, impl_func, vendor=vendor_for_reg)
        return impl_func
    except (ImportError, ModuleNotFoundError) as e:
        logger.debug(f"Failed to lazy-import '{impl_path}' for {method_key}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Error resolving lazy impl for {method_key}: {e}")
        return None


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
    if vendor_map is not None:
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

    # Fallback: check the lazy registry for deferred imports
    return _resolve_lazy_impl(method_key)


def overridable(func_or_class):
    """
    Decorator to mark a method, function, or class as replaceable by plugin.

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

    Usage in core code (for classes):
        @overridable
        class MyScheduler:
            def __init__(self, ...):
                ...

        # Instantiation: MyScheduler(...) will return an instance of the
        # override class if registered, otherwise the original class.
        # The override class MUST inherit from the original class to
        # ensure isinstance() compatibility.

    The decorator automatically:
    1. For methods: Detects the class name and method name
    2. For functions: Uses module name and function name
    3. For classes: Uses module name and class name
    4. Looks up override in the plugin registry
    5. Uses plugin if found, otherwise uses original implementation

    No parameters needed - everything is auto-detected!
    """
    if inspect.isclass(func_or_class):
        return _overridable_class(func_or_class)
    else:
        return _overridable_func(func_or_class)


def _overridable_class(cls):
    """Handle @overridable on a class definition."""
    original_module = cls.__module__
    class_name = cls.__name__

    # method_key: "module_basename.ClassName"
    module_parts = original_module.split('.')
    module_name = module_parts[-1] if module_parts else "unknown"
    method_key = f"{module_name}.{class_name}"

    # State for caching the resolved class (None = not resolved yet)
    _resolved = {}  # Use dict to allow mutation in closure

    def _resolve_override_class():
        """Resolve and cache the override class (or mark as not found)."""
        if 'cls' in _resolved:
            return _resolved['cls']

        plugin_cls = get_override_method(method_key)

        # If not found, try to lazy import the plugin module
        if plugin_cls is None:
            try:
                if original_module.startswith("megatron.core."):
                    plugin_module = original_module.replace("megatron.core.", "megatron.plugin.", 1)
                    try:
                        importlib.import_module(plugin_module)
                        plugin_cls = get_override_method(method_key)
                    except (ImportError, ModuleNotFoundError):
                        pass
            except Exception as e:
                logger.debug(f"Failed to lazy import plugin for {method_key}: {e}")

        if plugin_cls is not None:
            logger.info(f"Using override class for {method_key}")
        else:
            logger.debug(f"Using original class for {method_key}")

        _resolved['cls'] = plugin_cls
        return plugin_cls

    class OverridableClassProxy(cls):
        """Proxy class that dispatches instantiation to the override class if registered."""

        def __new__(proxy_cls, *args, **kwargs):
            # Only dispatch when instantiating the proxy class itself.
            # If a subclass (including the override class) calls __new__,
            # skip dispatching to avoid infinite recursion.
            if proxy_cls is not OverridableClassProxy:
                return object.__new__(proxy_cls)
            override_cls = _resolve_override_class()
            if override_cls is not None:
                # Create instance of the override class.
                # Use object.__new__ to avoid re-entering this __new__.
                instance = object.__new__(override_cls)
                return instance
            else:
                # No override: create instance of the proxy itself
                # (which inherits cls's __init__ and all methods)
                return object.__new__(OverridableClassProxy)

        def __init_subclass__(subcls, **kwargs):
            # Allow normal subclassing without interference
            super().__init_subclass__(**kwargs)

    # Preserve the original class identity as much as possible
    OverridableClassProxy.__name__ = cls.__name__
    OverridableClassProxy.__qualname__ = cls.__qualname__
    OverridableClassProxy.__module__ = cls.__module__
    OverridableClassProxy.__doc__ = cls.__doc__

    return OverridableClassProxy


def _overridable_func(func):
    """Handle @overridable on a function or method."""
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
            logger.info(f"Using override method for {method_key}")
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


def _target_to_method_key(target: str) -> str:
    """Convert a full target path to the method_key format used by the registry.

    Rules (matching how ``@overridable`` generates keys):
    - Module-level function:
        "megatron.core.distributed.finalize_model_grads._allreduce_embedding_grad"
        -> method_key = "finalize_model_grads._allreduce_embedding_grad"
    - Class method (identified by PascalCase segment before the last dot):
        "megatron.core.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan"
        -> method_key = "MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan"

    Args:
        target: Full dotted path to the original function/method.

    Returns:
        The method_key string.
    """
    parts = target.rsplit(".", 1)
    if len(parts) != 2:
        return target

    prefix, func_name = parts
    # Get the last segment of the prefix
    prefix_parts = prefix.rsplit(".", 1)
    last_segment = prefix_parts[-1] if len(prefix_parts) == 2 else prefix_parts[0]

    # Heuristic: if last_segment starts with uppercase -> class method
    # otherwise -> module-level function (use module basename as key prefix)
    if last_segment and last_segment[0].isupper():
        # Class method: key is "ClassName.method_name"
        return f"{last_segment}.{func_name}"
    else:
        # Module-level function: key is "module_basename.func_name"
        return f"{last_segment}.{func_name}"


def register(target: str, impl: str, vendor: str = _DEFAULT_VENDOR) -> None:
    """Register an override mapping from target to implementation (lazy).

    This is the recommended way to declare overrides. All mappings can be
    collected in a single registry file (e.g., ``megatron/plugin/override_registry.py``).
    The ``impl`` module will only be imported when the overridden function is
    first called.

    Args:
        target: Full dotted path to the original function/method in megatron core.
            Examples:
                "megatron.core.distributed.finalize_model_grads._allreduce_embedding_grad"
                "megatron.core.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan"
        impl: Full dotted path to the plugin implementation function.
            Examples:
                "megatron.plugin.distributed.finalize_model_grads._allreduce_embedding_grad"
                "megatron.plugin.optimizer.optimizer._unscale_main_grads_and_check_for_nan"
        vendor: Vendor name (e.g., "musa", "txda"). Defaults to "default".

    Example::

        from megatron.plugin.decorators import register

        register(
            target="megatron.core.distributed.finalize_model_grads._allreduce_embedding_grad",
            impl="megatron.plugin.distributed.finalize_model_grads._allreduce_embedding_grad",
        )
        register(
            target="megatron.core.optimizer.clip_grads.get_grad_norm_fp32",
            impl="megatron.plugin.optimizer.clip_grads.get_grad_norm_fp32",
            vendor="musa",
        )
    """
    vendor = vendor.lower()
    method_key = _target_to_method_key(target)

    if method_key not in _lazy_registry:
        _lazy_registry[method_key] = {}
    _lazy_registry[method_key][vendor] = impl
    logger.debug(f"Registered lazy override: {method_key} -> {impl} (vendor={vendor})")
