# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""Implicit training probes without loading the tracing runtime or analyzers.

Initialize the shared Core facade before importing a concrete probe. Core's
compatibility exports import these probes during package initialization, so
starting a concrete probe first would otherwise expose an incomplete module.
"""

from megatron.core import observability as _observability  # noqa: F401
