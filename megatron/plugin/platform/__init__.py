# Copyright (c) BAAI Corporation.

from .platform_base import PlatformBase
from .platform_register import register_platforms
from .platform_manager import get_platform, set_platform, is_current_platform_supported

register_platforms()