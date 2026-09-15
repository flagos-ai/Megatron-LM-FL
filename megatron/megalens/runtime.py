"""Training-layer lifecycle for one rank-local MegaLens tracer."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Iterator

from megatron.megalens.core_adapter import install_megalens_core_sink, uninstall_megalens_core_sink
from megatron.megalens.hardware_monitor import get_hardware_monitor_capabilities
from megatron.megalens.trace import Tracer


class MegaLensRuntime:
    """Own a fresh Tracer from argument binding through process teardown.

    The runtime is designed for Megatron's single training thread on each rank.
    It keeps incomplete iterations out of persisted traces and resets the Core
    sink during both graceful shutdown and in-process restart cleanup.
    """

    def __init__(self, args, *, tracer: Tracer | None = None) -> None:
        self.tracer = tracer if tracer is not None else Tracer()
        self.tracer.configure(args)
        self.hardware_monitor_capabilities = get_hardware_monitor_capabilities()
        if (
            getattr(args, "hardware_monitor", False)
            and not self.hardware_monitor_capabilities.any_provider_available
        ):
            warnings.warn(
                "MegaLens hardware monitoring was requested, but no metric provider is "
                "available (psutil=unavailable, NVML=unavailable). Tracing will continue "
                "without hardware samples. Install the 'megalens-monitor' extra and make "
                "NVML available to enable CPU/GPU metrics.",
                RuntimeWarning,
                stacklevel=2,
            )
        install_megalens_core_sink(self.tracer)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @contextmanager
    def iteration(self, iteration_id: int, *, enable_hw_monitor: bool = False) -> Iterator[None]:
        """Trace one non-skipped training step with exception-safe cleanup."""
        if self._closed:
            raise RuntimeError("MegaLens runtime is closed")
        if iteration_id <= 0:
            raise ValueError("MegaLens iteration IDs must be positive")

        self.tracer.iteration_begin(iteration_id, enable_hw_monitor)
        try:
            yield
        except BaseException:
            try:
                self.tracer.abort_iteration()
            except Exception as cleanup_error:
                warnings.warn(
                    f"MegaLens failed to abort an incomplete iteration: {cleanup_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            raise
        else:
            try:
                self.tracer.iteration_end(enable_hw_monitor)
            except BaseException:
                try:
                    self.tracer.abort_iteration()
                except Exception as cleanup_error:
                    warnings.warn(
                        f"MegaLens failed to clean up a finalization error: {cleanup_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                raise

    def shutdown(self, *, graceful: bool) -> None:
        """Close resources and always restore Core's Null sink."""
        if self._closed:
            return
        try:
            self.tracer.shutdown(graceful=graceful)
        finally:
            uninstall_megalens_core_sink()
            self._closed = True


__all__ = ["MegaLensRuntime"]
