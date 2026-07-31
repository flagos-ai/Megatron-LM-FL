# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""C2 parity check — MegaLens shared utils on Megatron-LM-FL.

Verifies the torch-free core of ``utils.py`` — parallel-size inference from
trace metadata — which every analyzer relies on to auto-detect DP/PP/TP/EP
sizes without user input.

Run:  python -m megatron.megalens.migration_checks.check_c2_utils
Exit code 0 = parity holds. No torch / GPU required.
"""
from __future__ import annotations

import sys

from megatron.megalens.utils import get_tensor_bytes, infer_parallel_sizes_from_traces


def main() -> int:
    # Synthetic Chrome-trace events carrying rank metadata in args.dp_rk/pp_rk/tp_rk,
    # plus a moe-router event that advertises ep_size (how EP size is inferred).
    traces = [
        {"name": "forward-step", "args": {"dp_rk": 0, "pp_rk": 0, "tp_rk": 0}},
        {"name": "forward-step", "args": {"dp_rk": 1, "pp_rk": 0, "tp_rk": 0}},
        {"name": "tp-allreduce", "args": {"dp_rk": 0, "pp_rk": 1, "tp_rk": 1}},
        {"name": "moe-router", "args": {"dp_rk": 0, "pp_rk": 0, "tp_rk": 0, "ep_size": 4}},
    ]
    sizes = infer_parallel_sizes_from_traces(traces)
    expected = {"dp": 2, "pp": 2, "tp": 2, "ep": 4}

    ok = sizes == expected
    # get_tensor_bytes must degrade gracefully without torch tensors.
    ok &= get_tensor_bytes(None) == 0
    ok &= get_tensor_bytes([None, [None]]) == 0

    print("infer_parallel_sizes_from_traces:", sizes, "expected", expected)
    print("\nC2 utils parity:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
