# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""C2 parity check — trace data loader + paper style on Megatron-LM-FL.

TraceDataLoader is the indexed query layer every analyzer consumes: it ingests
the post-transform Chrome-trace events (ph=X spans, ph=C counters, ph=M topology)
and answers name/window/rank queries.

Verifies on synthetic events:
  - from_traces ingests ph=X spans and ph=M process_name topology
  - get_events_by_name / get_ranks return the expected spans/ranks
  - infer_parallel_sizes_from_loader (utils) reads the loader topology -> dp=2
  - paper_style.apply_global_rcparams() runs (matplotlib styling entrypoint)

Run:  python -m megatron.megalens.migration_checks.check_c2_data_loader
Exit code 0 = parity holds. No torch / GPU required.
"""
from __future__ import annotations

from megatron.megalens.data_loader import TraceDataLoader
from megatron.megalens.utils import infer_parallel_sizes_from_loader


def main() -> int:
    events = [
        {"ph": "M", "name": "process_name", "pid": 0, "args": {"name": "DP0-PP0-TP0"}},
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "DP1-PP0-TP0"}},
        {
            "ph": "X",
            "name": "tp-allreduce",
            "ts": 100,
            "dur": 200,
            "pid": 0,
            "args": {"dp_rk": 0, "pp_rk": 0, "tp_rk": 0, "data_bytes": 1024},
        },
        {
            "ph": "X",
            "name": "tp-allreduce",
            "ts": 150,
            "dur": 180,
            "pid": 1,
            "args": {"dp_rk": 1, "pp_rk": 0, "tp_rk": 0, "data_bytes": 1024},
        },
        {
            "ph": "X",
            "name": "forward",
            "ts": 50,
            "dur": 300,
            "pid": 0,
            "args": {"dp_rk": 0, "pp_rk": 0, "tp_rk": 0},
        },
    ]

    loader = TraceDataLoader.from_traces(events)
    ok = True

    ok &= len(loader.span_events) == 3
    print("span_events:", len(loader.span_events))

    ar = loader.get_events_by_name("tp-allreduce")
    ok &= len(ar) == 2
    print("tp-allreduce spans:", len(ar))

    ranks = sorted(loader.get_ranks())
    ok &= ranks == [0, 1]
    print("ranks:", ranks)

    # chain into utils: topology (from ph=M) -> parallel sizes
    sizes = infer_parallel_sizes_from_loader(loader)
    print("inferred sizes:", sizes)
    ok &= sizes["dp"] == 2 and sizes["pp"] == 1 and sizes["tp"] == 1

    # paper_style entrypoint must import and run
    from megatron.megalens import paper_style

    paper_style.apply_global_rcparams()
    print("paper_style.apply_global_rcparams() OK (PAPER_MODE=%s)" % paper_style.PAPER_MODE)

    print("\nC2 data_loader + paper_style parity:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
