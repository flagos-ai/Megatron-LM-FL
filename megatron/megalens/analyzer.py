# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""MegaLens trace aggregation and parallelism analysis CLI.

This module is the recommended entry point for:

1. **Aggregating** per-rank benchmark JSON files into a single Chrome-trace-style
   event list (post-``transform``), suitable for Perfetto / downstream tools.
2. **Running** Pipeline (PP), Data Parallel (DP), Tensor / Sequence Parallel
   (TP/SP), and Expert Parallel (EP) diagnostic analyses on that trace.

Typical usage::

    # From a directory of per-rank benchmark files: aggregate + run all analyses
    python -m megatron.megalens.analyzer \\
        --bench-dir ./trace_output \\
        --output-dir ./megalens_out

    # Analyze an already-aggregated trace JSON
    python -m megatron.megalens.analyzer \\
        --trace ./megalens_out/aggregated_trace.json \\
        --run pp tp \\
        --output-dir ./reports

    # Only aggregate to JSON (no analyzers)
    python -m megatron.megalens.analyzer \\
        --bench-dir ./trace_output \\
        --aggregate-only \\
        --trace-output ./aggregated.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from megatron.megalens.trace_aggregate import (
    Iteration,
    aggregate_benchmark_data,
    benchmark_to_chrome_trace,
    collect_benchmark_files,
    read_benchmark_file,
)
from megatron.megalens.utils import infer_parallel_sizes_from_traces

logger = logging.getLogger(__name__)


def _expand_run_modes(modes: Sequence[str]) -> List[str]:
    """Turn CLI ``--run`` values into an ordered list without duplicates."""
    if not modes:
        return ["pp", "dp", "tp", "ep", "hybrid"]
    if "all" in modes:
        return ["pp", "dp", "tp", "ep", "hybrid"]
    seen: set[str] = set()
    out: List[str] = []
    for m in modes:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _infer_parallel_sizes(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    """Infer DP/PP/TP/EP sizes from trace metadata."""
    return infer_parallel_sizes_from_traces(traces)


def aggregate_traces_from_benchmark_dir(bench_dir: Path) -> Tuple[List[Dict[str, Any]], Path]:
    """Load per-rank benchmark files, aggregate, and convert to Chrome trace events.

    Args:
        bench_dir: Directory containing ``benchmark-*.json`` (see
            :func:`megatron.megalens.trace_aggregate.collect_benchmark_files`).

    Returns:
        ``(traces, bench_dir)`` where ``traces`` is the list returned by
        :func:`megatron.megalens.trace_aggregate.benchmark_to_chrome_trace`.
    """
    bench_dir = bench_dir.resolve()
    files = collect_benchmark_files(bench_dir)
    if not files:
        raise FileNotFoundError(
            f"No benchmark files found under {bench_dir} "
            "(expected names like benchmark-*.json)."
        )
    contents: List[List[Iteration]] = [
        read_benchmark_file(rank, content) for rank, content in files
    ]
    iterations, _dp, _pp, _tp = aggregate_benchmark_data(contents)
    traces = benchmark_to_chrome_trace(iterations)
    return traces, bench_dir


def load_traces_from_json(path: Path) -> List[Dict[str, Any]]:
    """Load an aggregated Chrome trace JSON (list of event dicts)."""
    path = path.resolve()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of trace events.")
    return data


def write_trace_json(traces: List[Dict[str, Any]], path: Path) -> None:
    """Write trace events to JSON (UTF-8, indented)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2)
    logger.info("Wrote aggregated trace (%d events) -> %s", len(traces), path)


def run_parallelism_analyses(
    traces: List[Dict[str, Any]],
    modes: Sequence[str],
    output_dir: Path,
    *,
    pp_theory_bw_gbps: Optional[float] = None,
    tp_nvlink_peak_gbps: Optional[float] = None,
) -> None:
    """Run selected PP / DP / TP / EP / Hybrid analyzers under ``output_dir``."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded = _expand_run_modes(modes)
    sizes = _infer_parallel_sizes(traces)
    logger.info(
        "Detected parallel sizes from trace: DP=%d PP=%d TP=%d EP=%d",
        sizes["dp"], sizes["pp"], sizes["tp"], sizes["ep"],
    )

    pp_result: Optional[Any] = None
    dp_result: Optional[Any] = None
    tp_result: Optional[Any] = None
    ep_result: Optional[Any] = None

    if "pp" in expanded:
        if sizes["pp"] <= 1:
            logger.info("Skipping PP analysis: pipeline parallel size <= 1")
        else:
            from megatron.megalens.pp_analyzer import analyze_pp_traces

            pp_dir = output_dir / "pp"
            logger.info("Running PP analysis -> %s", pp_dir)
            pp_result = analyze_pp_traces(
                traces,
                theory_bw_gbps=pp_theory_bw_gbps,
                output_dir=str(pp_dir),
            )

    if "dp" in expanded:
        if sizes["dp"] <= 1:
            logger.info("Skipping DP analysis: data parallel size <= 1")
        else:
            from megatron.megalens.dp_analyzer import analyze_dp_traces

            dp_dir = output_dir / "dp"
            logger.info("Running DP analysis -> %s", dp_dir)
            dp_result = analyze_dp_traces(traces, output_dir=str(dp_dir))

    if "tp" in expanded:
        if sizes["tp"] <= 1:
            logger.info("Skipping TP/SP analysis: tensor parallel size <= 1")
        else:
            from megatron.megalens.tp_analyzer import analyze_tp_traces

            tp_dir = output_dir / "tp"
            logger.info("Running TP/SP analysis -> %s", tp_dir)
            tp_result = analyze_tp_traces(
                traces,
                nvlink_theory_peak_gbps=tp_nvlink_peak_gbps,
                output_dir=str(tp_dir),
            )

    if "ep" in expanded:
        if sizes["ep"] <= 1:
            logger.info("Skipping EP analysis: expert parallel size <= 1")
        else:
            from megatron.megalens.ep_analyzer import analyze_ep_traces

            ep_dir = output_dir / "ep"
            logger.info("Running EP analysis -> %s", ep_dir)
            ep_result = analyze_ep_traces(traces, output_dir=str(ep_dir))

    if "hybrid" in expanded:
        if sizes["pp"] <= 1 and sizes["dp"] <= 1 and sizes["tp"] <= 1 and sizes["ep"] <= 1:
            logger.info("Skipping Hybrid analysis: all parallel sizes are <= 1")
        else:
            from megatron.megalens.hybrid_analyzer import analyze_hybrid_traces

            hybrid_dir = output_dir / "hybrid"
            logger.info("Running Hybrid cross-dimension analysis -> %s", hybrid_dir)
            analyze_hybrid_traces(
                traces,
                output_dir=str(hybrid_dir),
                pp_result=pp_result,
                dp_result=dp_result,
                tp_result=tp_result,
                ep_result=ep_result,
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MegaLens: aggregate training traces and run PP / DP / TP / EP / Hybrid analyses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "-b",
        "--bench-dir",
        type=Path,
        help="Directory with per-rank benchmark JSON files to aggregate.",
    )
    src.add_argument(
        "-t",
        "--trace",
        type=Path,
        help="Path to an already-aggregated Chrome trace JSON (event list).",
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate (requires --bench-dir); do not run analyzers.",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        choices=("pp", "dp", "tp", "ep", "hybrid", "all"),
        metavar="MODE",
        help="Which analyses to run: pp, dp, tp, ep, hybrid, or all. Ignored with --aggregate-only.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Base directory for analyzer outputs (subdirs pp/, dp/, tp/, ep/, hybrid/).",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help=(
            "Write aggregated trace JSON to this path. "
            "With --bench-dir, default: <output-dir>/aggregated_trace.json if "
            "--output-dir is set, else ./megalens_trace_<timestamp>.json"
        ),
    )
    parser.add_argument(
        "--pp-theory-bw",
        type=float,
        default=None,
        help="Override unidirectional P2P theoretical bandwidth (GB/s) for PP analyzer.",
    )
    parser.add_argument(
        "--tp-nvlink-peak",
        type=float,
        default=None,
        help=(
            "NVLink theoretical peak (GB/s) for TP analyzer saturation heuristics. "
            "If omitted, auto-detects a unidirectional peer-to-peer peak from NVML/topology."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.aggregate_only and args.trace is not None:
        parser.error("--aggregate-only requires --bench-dir (not --trace).")

    traces: List[Dict[str, Any]]
    default_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.bench_dir is not None:
        traces, _ = aggregate_traces_from_benchmark_dir(args.bench_dir)
        if args.trace_output is not None:
            trace_out = args.trace_output.resolve()
        elif args.output_dir is not None:
            trace_out = args.output_dir.resolve() / "aggregated_trace.json"
        else:
            trace_out = Path.cwd() / f"megalens_trace_{default_stamp}.json"
        write_trace_json(traces, trace_out)
    else:
        assert args.trace is not None
        traces = load_traces_from_json(args.trace)

    if args.aggregate_only:
        logger.info("Aggregate-only mode; skipping analyzers.")
        return 0

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path.cwd() / f"megalens_analysis_{default_stamp}"

    modes: Sequence[str] = args.run if args.run else ["pp", "dp", "tp", "ep", "hybrid"]
    run_parallelism_analyses(
        traces,
        modes,
        out_dir,
        pp_theory_bw_gbps=args.pp_theory_bw,
        tp_nvlink_peak_gbps=args.tp_nvlink_peak,
    )
    logger.info("Done. Analysis outputs under %s", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
