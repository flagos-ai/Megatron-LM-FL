# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""C2 parity check — hardware monitor on Megatron-LM-FL.

HardwareMonitor is the (currently NVIDIA/NVML-based) background sampler that
feeds NVLink/SM/HBM counters into the trace. This check verifies it ports and
constructs cleanly and its buffer API works, without requiring a live GPU
sample loop (live sampling is validated at training-time end-to-end).

Scope / platform note: this module is the single heaviest NVIDIA coupling point
in MegaLens (pynvml). It is ported as-is for the CUDA phase; the FlagOS
multi-chip platform seam (HardwareBackend protocol + CudaNvmlBackend +
get_hardware_backend factory) lands in a dedicated later commit and will not
touch the collection call sites.

Run:  python -m megatron.megalens.migration_checks.check_c2_hardware_monitor
Exit code 0 = parity holds.
"""
from __future__ import annotations

from megatron.megalens.hardware_monitor import _HAS_NVML, HardwareMonitor


def main() -> int:
    ok = True

    # Constructs regardless of GPU/NVML availability (NVML init is guarded).
    mon = HardwareMonitor(interval=0.1, nvlink_every_n=5, cpu_every_n=5)
    print(
        "HardwareMonitor constructed; _HAS_NVML =",
        _HAS_NVML,
        "nvml_initialised =",
        mon._nvml_initialised,
    )

    # Buffer API works with no samples collected yet.
    records = mon.collect_and_clear()
    ok &= isinstance(records, list) and len(records) == 0
    print("collect_and_clear() on fresh monitor:", records)

    # Idempotent stop/shutdown must not raise even when never started.
    mon.stop()
    mon.shutdown()
    print("stop()/shutdown() on unstarted monitor: OK")

    print("\nC2 hardware_monitor parity:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
