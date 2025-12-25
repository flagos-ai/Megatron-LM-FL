"""
Plugin decorator system for method replacement.

The decorator automatically detects the class and method context,
and looks up the implementation in plugins.
"""

import functools
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Registry to store plugin implementations
# Key format: "ClassName.method_name"
_plugin_registry: dict[str, Callable] = {}


def register_plugin_method(method_key: str, implementation: Callable) -> None:
    """
    Register a plugin implementation for a method.
    
    Args:
        method_key: Unique key for the method (e.g., "LanguageModule._is_in_embd_group")
        implementation: The implementation function
    """
    _plugin_registry[method_key] = implementation
    logger.debug(f"Registered plugin method: {method_key}")


def get_plugin_method(method_key: str) -> Optional[Callable]:
    """
    Get a plugin implementation for a method.
    
    Args:
        method_key: Unique key for the method
        
    Returns:
        The plugin implementation if available, None otherwise
    """
    return _plugin_registry.get(method_key)


def plugin_method(func: Callable) -> Callable:
    """
    Decorator to mark a method as replaceable by plugins.
    
    Usage in core code:
        @plugin_method
        def _is_in_embd_group(self):
            # Original implementation (fallback if no plugin)
            ...
    
    The decorator automatically:
    1. Detects the class name and method name
    2. Looks up plugin implementation using "ClassName.method_name" as key
    3. Uses plugin if found, otherwise uses original implementation
    
    No parameters needed - everything is auto-detected!
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Auto-detect class name and method name
        class_name = self.__class__.__name__
        method_name = func.__name__
        method_key = f"{class_name}.{method_name}"
        
        # Check if plugin implementation exists
        plugin_impl = get_plugin_method(method_key)
        if plugin_impl is not None:
            logger.debug(f"Using plugin implementation for {method_key}")
            return plugin_impl(self, *args, **kwargs)
        else:
            # Use original implementation
            return func(self, *args, **kwargs)
    
    return wrapper


def plugin_implementation(class_name: str, method_name: str):
    """
    Decorator to register a plugin implementation.
    
    Usage in plugins:
        @plugin_implementation("LanguageModule", "_is_in_embd_group")
        def _is_in_embd_group(self):
            # Plugin implementation
            ...
    
    This decorator automatically registers the function as a plugin implementation.
    
    Args:
        class_name: Class name (e.g., "LanguageModule")
        method_name: Method name (e.g., "_is_in_embd_group")
    """
    def decorator(impl_func: Callable) -> Callable:
        method_key = f"{class_name}.{method_name}"
        register_plugin_method(method_key, impl_func)
        logger.info(f"Registered plugin implementation: {method_key}")
        return impl_func
    return decorator

