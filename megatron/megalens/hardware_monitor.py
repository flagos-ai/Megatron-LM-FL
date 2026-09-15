"""Hardware metrics collector using NVML and psutil.

Runs a background thread that samples GPU and CPU metrics at a configurable
interval and stores them as Chrome Trace Counter events (ph: "C") aligned to
the CUDA event timeline of the parent Tracer.

Metrics collected per GPU:
    SM_Util_pct, Mem_Util_pct, SM_Clock_MHz, SM_Base_Clock_MHz,
    Temp_C, Power_W, NVLink_Tx_MBs, NVLink_Rx_MBs

Metrics collected globally:
    CPU_Util_pct, Sys_Mem_Used_GB
"""

import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import pynvml

    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False

# NVML field IDs for NVLink cumulative throughput counters (KiB).
# These are stable across driver versions ≥ 470.
_NVML_FI_NVLINK_TX: int = 141  # NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX
_NVML_FI_NVLINK_RX: int = 142  # NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX


@dataclass(frozen=True)
class HardwareMonitorCapabilities:
    """Metric providers available to a rank-local hardware monitor."""

    psutil_available: bool
    nvml_available: bool

    @property
    def any_provider_available(self) -> bool:
        return self.psutil_available or self.nvml_available


def get_hardware_monitor_capabilities() -> HardwareMonitorCapabilities:
    """Return dependency-level capabilities without initializing NVML."""

    return HardwareMonitorCapabilities(psutil_available=_HAS_PSUTIL, nvml_available=_HAS_NVML)


class _NVLinkThroughputTracker:
    """Converts cumulative NVLink KiB counters into instantaneous MB/s."""

    def __init__(self) -> None:
        self._prev_tx: Optional[int] = None
        self._prev_rx: Optional[int] = None
        self._prev_time: float = 0.0
        self._available: Optional[bool] = None  # tri-state: None = untested

    def sample(self, handle: Any) -> Tuple[Optional[float], Optional[float]]:
        """Return (tx_MB_s, rx_MB_s) or (None, None) if unsupported."""
        if self._available is False:
            return None, None

        try:
            field_values = pynvml.nvmlDeviceGetFieldValues(
                handle, [_NVML_FI_NVLINK_TX, _NVML_FI_NVLINK_RX]
            )
            tx_kib = self._extract_counter(field_values[0])
            rx_kib = self._extract_counter(field_values[1])

            if tx_kib is None or rx_kib is None:
                self._available = False
                return None, None

            self._available = True
            now = time.monotonic()

            if self._prev_tx is not None:
                dt = now - self._prev_time
                if dt > 0:
                    tx_mbs = (tx_kib - self._prev_tx) / 1024.0 / dt
                    rx_mbs = (rx_kib - self._prev_rx) / 1024.0 / dt
                else:
                    tx_mbs, rx_mbs = 0.0, 0.0
            else:
                # First sample – no delta yet
                tx_mbs, rx_mbs = 0.0, 0.0

            self._prev_tx = tx_kib
            self._prev_rx = rx_kib
            self._prev_time = now
            return tx_mbs, rx_mbs

        except Exception:
            self._available = False
            return None, None

    @staticmethod
    def _extract_counter(field_value: Any) -> Optional[int]:
        """Robustly extract the integer counter from an NVML field value."""
        try:
            if hasattr(field_value, "nvmlReturn") and field_value.nvmlReturn != 0:
                return None
        except Exception:
            pass
        for attr in ("ullVal", "ulVal", "uiVal"):
            try:
                return int(getattr(field_value.value, attr))
            except (AttributeError, TypeError, ValueError):
                continue
        try:
            return int(field_value.value)
        except Exception:
            return None


class HardwareMonitor:
    """Background thread that samples GPU and CPU hardware counters.

    Args:
        interval: Seconds between samples (default 0.1 = 10 Hz = 100ms).
            Paper specifies 100ms sampling interval for <0.5% overhead.
        nvlink_every_n: Sample NVLink counters every N-th iteration
            (NVLink field-value queries are ~10x more expensive than
            other NVML calls).  Default 5 means NVLink at 2 Hz.
        cpu_every_n: Sample CPU/memory metrics every N-th iteration.
            Default 5 means CPU metrics at 2 Hz.
    """

    def __init__(
        self, interval: float = 0.1, nvlink_every_n: int = 5, cpu_every_n: int = 5
    ) -> None:
        self._trace_window_start_ns: int = 0
        self.interval: float = interval
        self._nvlink_every_n: int = nvlink_every_n
        self._cpu_every_n: int = cpu_every_n
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._records: List[Dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()

        self._handle: Any = None
        self._device_idx: int = -1
        self._base_clock_mhz: int = 0
        self._nvlink_tracker: Optional[_NVLinkThroughputTracker] = None
        self._sample_count: int = 0  # loop iteration counter

        self._nvml_initialised: bool = False
        self._nvml_init_attempted: bool = False
        self._unavailable_warning_emitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, trace_window_start_ns: int) -> None:
        """Begin collecting.  ``trace_window_start_ns`` is the wall-clock
        ``time.time_ns()`` captured by the tracer at iteration start so that
        hardware timestamps can be aligned to the CUDA event timeline."""
        if self._running:
            return
        self._trace_window_start_ns = trace_window_start_ns

        # Keep trace-only runs free of NVML side effects. Initialization happens
        # once, when hardware monitoring is explicitly started.
        self._initialise_nvml()

        # Resolve local GPU handle once (avoids per-sample overhead)
        if self._nvml_initialised and self._handle is None:
            try:
                self._device_idx = torch.cuda.current_device()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_idx)
                self._base_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(
                    self._handle, pynvml.NVML_CLOCK_SM
                )
                self._nvlink_tracker = _NVLinkThroughputTracker()
            except Exception:
                self._handle = None

        if not _HAS_PSUTIL and self._handle is None:
            if _HAS_NVML and not self._unavailable_warning_emitted:
                warnings.warn(
                    "MegaLens hardware monitoring has no usable metric provider: "
                    "psutil is unavailable and NVML initialization/device lookup failed. "
                    "Hardware samples are disabled for this rank.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._unavailable_warning_emitted = True
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def shutdown(self) -> None:
        """Release NVML resources (call once at program exit)."""
        self.stop()
        if self._nvml_initialised:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialised = False

    def collect_and_clear(self) -> List[Dict[str, Any]]:
        """Drain accumulated records (thread-safe)."""
        with self._lock:
            data = self._records
            self._records = []
        return data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _initialise_nvml(self) -> None:
        """Initialize NVML at most once, on the first explicit monitor start."""

        if self._nvml_init_attempted:
            return
        self._nvml_init_attempted = True
        if not _HAS_NVML:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_initialised = True
        except Exception:
            self._nvml_initialised = False

    def _monitor_loop(self) -> None:
        try:
            g_rk: int = torch.distributed.get_rank()
        except RuntimeError:
            g_rk = 0

        while self._running:
            loop_start = time.monotonic()
            ts_ns = time.time_ns() - self._trace_window_start_ns
            new_events: List[Dict[str, Any]] = []
            self._sample_count += 1

            # --- CPU / System metrics (low frequency) ---
            if _HAS_PSUTIL and self._sample_count % self._cpu_every_n == 0:
                try:
                    cpu_pct = psutil.cpu_percent(interval=None)
                    mem = psutil.virtual_memory()
                    new_events.append(
                        {
                            "name": "CPU_Metrics",
                            "ph": "C",
                            "rel_ts": ts_ns,
                            "g_rk": g_rk,
                            "tid": "Hardware Monitor",
                            "args": {
                                "CPU_Util_pct": cpu_pct,
                                "Sys_Mem_Used_GB": round(mem.used / 1024**3, 2),
                            },
                        }
                    )
                except Exception:
                    pass

            # --- GPU metrics (local device only) ---
            if self._handle is not None:
                sample_nvlink = self._sample_count % self._nvlink_every_n == 0
                gpu_args = self._collect_gpu_metrics(include_nvlink=sample_nvlink)
                if gpu_args:
                    new_events.append(
                        {
                            "name": "GPU_Metrics",
                            "ph": "C",
                            "rel_ts": ts_ns,
                            "g_rk": g_rk,
                            "tid": "Hardware Monitor",
                            "dev": self._device_idx,
                            "args": gpu_args,
                        }
                    )

            if new_events:
                with self._lock:
                    self._records.extend(new_events)

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, self.interval - elapsed))

    def _collect_gpu_metrics(self, include_nvlink: bool = True) -> Optional[Dict[str, Any]]:
        """Read GPU sensors from NVML.

        Args:
            include_nvlink: Whether to query NVLink throughput counters
                (expensive: ~1-5ms via ``nvmlDeviceGetFieldValues``).
                Set False on most samples to reduce overhead.
        """
        handle = self._handle
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        except pynvml.NVMLError:
            return None

        args: Dict[str, Any] = {
            "SM_Util_pct": util.gpu,
            "Mem_Util_pct": util.memory,
            "SM_Clock_MHz": clock,
            "SM_Base_Clock_MHz": self._base_clock_mhz,
            "Temp_C": temp,
            "Power_W": round(power, 1),
            "Mem_Used_MB": round(mem_info.used / 1024**2, 1),
        }

        # NVLink throughput — only when requested (expensive query)
        if include_nvlink and self._nvlink_tracker is not None:
            tx_mbs, rx_mbs = self._nvlink_tracker.sample(handle)
            if tx_mbs is not None:
                args["NVLink_Tx_MBs"] = round(tx_mbs, 1)
                args["NVLink_Rx_MBs"] = round(rx_mbs, 1)

        return args
