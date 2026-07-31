# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""C2 parity check — trace aggregation core on Megatron-LM-FL.

Verifies the deterministic ``--bench-dir`` ingestion core on synthetic data:
  1. collect_benchmark_files  — parse benchmark-<...>.json filenames -> Rank
  2. read_benchmark_file      — raw per-rank rows -> Iteration list
  3. aggregate_benchmark_data — align iterations across ranks; derive DP/PP/TP

Scope note: the final ``benchmark_to_chrome_trace`` -> ``transform`` emission
assumes realistically-structured per-event args (every B/E event carries
dp_rk/pp_rk/tp_rk) and is exercised end-to-end against a real trace at the
analyzer-level step, not with hand-crafted synthetic rows here. The code itself
is byte-identical to the source; this check covers the parsing/aggregation core.

Run:  python -m megatron.megalens.migration_checks.check_c2_trace_aggregate
Exit code 0 = parity holds. No torch / GPU required.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from megatron.megalens.trace_aggregate import (
    Rank,
    aggregate_benchmark_data,
    collect_benchmark_files,
    read_benchmark_file,
)


def _raw_rows(g_rk: int) -> list:
    """One iteration with a single forward B/E pair for a given global rank."""
    return [
        {"name": "iteration", "ph": "B", "pad_before": 0, "iteration": 0},
        {"name": "forward", "ph": "B", "rel_ts": 100, "g_rk": g_rk},
        {"name": "forward", "ph": "E", "rel_ts": 300, "g_rk": g_rk},
        {"name": "iteration", "ph": "E", "duration_wall": 1000, "iteration": 0},
    ]


def main() -> int:
    ok = True

    # --- 1) filename -> Rank parsing via a temp bench dir ---
    with tempfile.TemporaryDirectory() as d:
        for dp in (0, 1):
            p = Path(d) / f"benchmark-data-{dp}-pipeline-0-tensor-0.json"
            p.write_text(json.dumps(_raw_rows(dp)), encoding="utf-8")
        files = collect_benchmark_files(d)
        ranks = sorted((r.data, r.pipeline, r.tensor) for r, _ in files)
        ok &= ranks == [(0, 0, 0), (1, 0, 0)]
        print("collect_benchmark_files ranks:", ranks)

        # --- 2-3) read -> aggregate; DP/PP/TP derived from rank metadata ---
        contents = [read_benchmark_file(r, c) for r, c in files]
        ok &= all(len(c) == 1 for c in contents)  # one iteration per rank
        iterations, dp, pp, tp = aggregate_benchmark_data(contents)
        print(f"aggregate parallelism: dp={dp} pp={pp} tp={tp}")
        ok &= (dp, pp, tp) == (2, 1, 1)
        ok &= len(iterations) == 1

    print("\nC2 trace_aggregate parity:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
