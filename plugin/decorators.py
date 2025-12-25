"""
Plugin decorator system for method replacement.

The decorator automatically detects the class and method context,
and looks up the implementation in plugin.
"""

import functools
import importlib
import inspect
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Registry to store plugin implementations
# Key format: "ClassName.method_name" for methods, "module_name.function_name" for functions
_plugin_registry: dict[str, Callable] = {}


def register_plugin_method(method_key: str, implementation: Callable) -> None:
    """
    Register a plugin implementation for a method or function.
    
    Args:
        method_key: Unique key for the method/function (e.g., "LanguageModule._is_in_embd_group" or "clip_grads.get_grad_norm_fp32")
        implementation: The implementation function
    """
    _plugin_registry[method_key] = implementation
    logger.debug(f"Registered plugin method: {method_key}")


def get_plugin_method(method_key: str) -> Optional[Callable]:
    """
    Get a plugin implementation for a method or function.
    
    Args:
        method_key: Unique key for the method/function
        
    Returns:
        The plugin implementation if available, None otherwise
    """
    return _plugin_registry.get(method_key)


def plugin_method(func: Callable) -> Callable:
    """
    Decorator to mark a method or function as replaceable by plugin.
    
    Usage in core code (for methods):
        @plugin_method
        def _is_in_embd_group(self):
            # Original implementation (fallback if no plugin)
            ...
    
    Usage in core code (for module-level functions):
        @plugin_method
        def get_grad_norm_fp32(...):
            # Original implementation (fallback if no plugin)
            ...
    
    The decorator automatically:
    1. For methods: Detects the class name and method name
    2. For functions: Uses module name and function name
    3. Looks up plugin implementation using the key
    4. Uses plugin if found, otherwise uses original implementation
    
    No parameters needed - everything is auto-detected!
    """
    # Determine if this is a method or function at decoration time
    # by inspecting the function signature
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    is_method = params and params[0] == 'self'
    
    # Save the original qualname at decoration time
    # This is crucial for inheritance: when a subclass calls a parent's method,
    # we need the qualname of the method as defined in the parent class, not the subclass
    # Example: If A defines m1() and B inherits A, B().m1() should use "A.m1" as the key
    original_qualname = func.__qualname__
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_method:
            # It's a method - use the original qualname saved at decoration time
            # This ensures we get the class where the method is actually defined,
            # not the class of the instance calling it
            # Example: If A defines m1() and B inherits A, B().m1() will use "A.m1"
            qualname = original_qualname
            if '.' in qualname:
                # Extract class name from qualname (e.g., "A.m1" -> "A", "Outer.Inner.method" -> "Inner")
                parts = qualname.rsplit('.', 1)
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
            module_parts = func.__module__.split('.')
            module_name = module_parts[-1] if module_parts else "unknown"
            function_name = func.__name__
            method_key = f"{module_name}.{function_name}"
        
        # print(f"Megatron-LM-FL Decorators PluginMethod: method_key = {method_key}")
        # Check if plugin implementation exists
        plugin_impl = get_plugin_method(method_key)
        
        # If not found, try to lazy import the plugin module
        if plugin_impl is None:
            try:
                # Try to import the corresponding plugin module
                # For megatron.core.distributed.finalize_model_grads -> plugin.core.distributed.finalize_model_grads
                megatron_module = func.__module__
                if megatron_module.startswith("megatron."):
                    plugin_module = megatron_module.replace("megatron.", "plugin.", 1)
                    try:
                        importlib.import_module(plugin_module)
                        # Try again after import
                        plugin_impl = get_plugin_method(method_key)
                        if plugin_impl is not None:
                            logger.debug(f"Lazy loaded plugin implementation for {method_key}")
                    except (ImportError, ModuleNotFoundError):
                        # Plugin module doesn't exist, that's okay
                        pass
            except Exception as e:
                # Ignore any errors during lazy import
                logger.debug(f"Failed to lazy import plugin for {method_key}: {e}")
        
        # print(f"Megatron-LM-FL Decorators PluginMethod: plugin_impl = {plugin_impl}")
        if plugin_impl is not None:
            logger.debug(f"Using plugin implementation for {method_key}")
            return plugin_impl(*args, **kwargs)
        else:
            # Use original implementation
            return func(*args, **kwargs)
    
    return wrapper


def plugin_implementation(class_or_module_name: str, method_or_function_name: str):
    """
    Decorator to register a plugin implementation.
    
    Usage in plugins (for methods):
        @plugin_implementation("LanguageModule", "_is_in_embd_group")
        def _is_in_embd_group(self):
            # Plugin implementation
            ...
    
    Usage in plugins (for functions):
        @plugin_implementation("clip_grads", "get_grad_norm_fp32")
        def get_grad_norm_fp32(...):
            # Plugin implementation
            ...
    
    This decorator automatically registers the function as a plugin implementation.
    
    Args:
        class_or_module_name: Class name (e.g., "LanguageModule") or module name (e.g., "clip_grads")
        method_or_function_name: Method name (e.g., "_is_in_embd_group") or function name (e.g., "get_grad_norm_fp32")
    """
    def decorator(impl_func: Callable) -> Callable:
        method_key = f"{class_or_module_name}.{method_or_function_name}"
        # print(f"Megatron-LM-FL Decorators PluginImplementation: method_key = {method_key}")
        register_plugin_method(method_key, impl_func)
        logger.info(f"Registered plugin implementation: {method_key}")
        return impl_func
    return decorator
