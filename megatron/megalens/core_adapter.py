"""Install MegaLens as the runtime backend for Megatron Core probes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from megatron.core.observability import install_trace_sink, reset_trace_sink
from megatron.core.transformer.cuda_graphs import is_graph_capturing, is_graph_warmup
from megatron.megalens.trace import Tracer

_CUDA_CAPTURE_QUERY = (
    getattr(torch.cuda, "is_current_stream_capturing", None) if torch.cuda.is_available() else None
)


class MegaLensTraceSink:
    """Adapt :class:`Tracer` to the dependency-light Core trace contract."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def is_enabled(self, name: str) -> bool:
        return self._tracer.is_event_enabled(name)

    def scope(
        self,
        name: str,
        *,
        ctx: Mapping[str, Any] | None = None,
        slots: Sequence[str] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ):
        return self._tracer.scope(
            name, ctx=dict(ctx or {}), slots=list(slots or ()), attrs=dict(attrs or {})
        )


def should_suppress_core_scope() -> bool:
    """Keep Python/CUDA event creation out of graph and compiler capture."""
    if is_graph_capturing() or is_graph_warmup():
        return True

    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if is_compiling is not None and is_compiling():
        return True

    if not callable(_CUDA_CAPTURE_QUERY):
        return False
    try:
        return bool(_CUDA_CAPTURE_QUERY())
    except RuntimeError:
        # Some torch-compatible backends expose the API without implementing it.
        # Megatron-managed capture is already covered by the flags above.
        return False


def install_megalens_core_sink(tracer: Tracer) -> MegaLensTraceSink:
    """Bind one configured tracer to Core probes during process startup."""
    sink = MegaLensTraceSink(tracer)
    install_trace_sink(sink, suppress_scope=should_suppress_core_scope)
    return sink


def uninstall_megalens_core_sink() -> None:
    """Restore Core's Null sink for in-process teardown and restart."""
    reset_trace_sink()


__all__ = [
    "MegaLensTraceSink",
    "install_megalens_core_sink",
    "should_suppress_core_scope",
    "uninstall_megalens_core_sink",
]
