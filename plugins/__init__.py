"""
Plugin system for method replacement.

This module automatically registers plugin implementations when imported.
"""

# Import plugin implementations
# The implementations will be registered automatically when the module is imported
from . import core  # noqa: F401

__all__ = ["plugin_method", "plugin_implementation"]

# Export the decorators for use in core code and plugins
from .decorators import plugin_method, plugin_implementation

