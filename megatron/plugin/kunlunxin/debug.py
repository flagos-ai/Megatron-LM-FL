"""Debug logging helpers for KunLunXin plugin implementations."""

import functools
import logging
import os
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable)
_TRUE_VALUES = ("1", "true", "yes", "on")


def debug_enabled() -> bool:
    """Return whether KunLunXin patch debug logging is enabled."""
    return os.getenv("MG_FL_KUNLUNXIN_DEBUG", "").strip().lower() in _TRUE_VALUES


def log_patch(name: str) -> None:
    """Emit a debug log when execution enters a KunLunXin patch implementation."""
    if debug_enabled():
        logger.warning("[KunLunXin Override] %s", name)


def debug_patch(name: str):
    """Wrap a KunLunXin patch implementation with entry logging.

    Args:
        name: Human-readable patch name to include in the debug log.

    Returns:
        A decorator that logs before calling the wrapped patch implementation.
    """
    def decorator(func: _F) -> _F:
        """Decorate a KunLunXin patch implementation."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Log patch entry and call the original implementation."""
            log_patch(name)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
