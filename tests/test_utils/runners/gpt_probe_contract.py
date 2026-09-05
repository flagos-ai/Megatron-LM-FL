# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exact trace contracts for the controlled GPT training profiles."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from megatron.megalens.trace_aggregate import (
    Event,
    Iteration,
    collect_benchmark_files,
    read_benchmark_file,
)
from tests.test_utils.runners.megalens_run_manifest import Failure

_MODEL_PHASES = frozenset(
    ("forward-step", "decoder", "decoder-postprocess", "output_layer", "loss")
)

_EAGER_TREE_PHASES = frozenset(
    (
        "decoder",
        "transformer_layer",
        "_forward_attention",
        "attention",
        "_forward_mlp",
        "MLP.forward",
    )
)

_EAGER_LAYER_SEQUENCE = (
    ("transformer_layer", "B"),
    ("_forward_attention", "B"),
    ("attention", "B"),
    ("attention", "E"),
    ("_forward_attention", "E"),
    ("_forward_mlp", "B"),
    ("MLP.forward", "B"),
    ("MLP.forward", "E"),
    ("_forward_mlp", "E"),
    ("transformer_layer", "E"),
)

_OPTIMIZER_PHASES = frozenset(("optimizer", "optimizer-step", "optimizer-postprocess"))

_OPTIMIZER_SEQUENCE = (
    ("optimizer", "B"),
    ("optimizer-step", "B"),
    ("optimizer-step", "E"),
    ("optimizer", "E"),
    ("optimizer-postprocess", "B"),
    ("optimizer-postprocess", "E"),
)

_CUDA_GRAPH_INNER_PHASES = frozenset(
    ("transformer_layer", "_forward_attention", "attention", "_forward_mlp", "MLP.forward")
)

_CUDA_KERNEL_REQUIRED_FIELDS = frozenset(
    (
        "record_type",
        "name",
        "start_us",
        "end_us",
        "wall_start_us",
        "wall_end_us",
        "iter_rel_start_us",
        "iter_rel_end_us",
        "duration_us",
        "device",
        "iteration",
        "g_rk",
        "dp_rk",
        "pp_rk",
        "tp_rk",
    )
)

_CUDA_GRAPH_MODEL_KERNEL_MARKERS = ("gemm", "nvjet_", "sdpa", "transformer_engine::")


@dataclass(frozen=True)
class _Span:
    begin: Event
    end: Event


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    by_rank: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None:
            raise ValueError(f"trace shard {rank} has no global rank")
        if rank.global_rank in by_rank:
            raise ValueError(f"duplicate trace shard for global rank {rank.global_rank}")
        by_rank[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return by_rank


def _failure(code: str, message: str, *, rank: int, iteration: int) -> Failure:
    return Failure(code, message, f"rank={rank} iteration={iteration}")


def _pair_spans(
    iteration: Iteration, names: Iterable[str], *, rank: int
) -> tuple[Mapping[str, Sequence[_Span]], list[Failure]]:
    selected = frozenset(names)
    open_events: dict[str, list[Event]] = defaultdict(list)
    spans: dict[str, list[_Span]] = defaultdict(list)
    failures: list[Failure] = []
    iteration_id = int(iteration.iteration_id)

    for event in iteration.events:
        if event.name not in selected:
            continue
        if event.ph == "B":
            open_events[event.name].append(event)
        elif event.ph == "E":
            if not open_events[event.name]:
                failures.append(
                    _failure(
                        "trace.gpt.unmatched_end",
                        f"event {event.name!r} has an unmatched end",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
                continue
            spans[event.name].append(_Span(open_events[event.name].pop(), event))
        else:
            failures.append(
                _failure(
                    "trace.gpt.phase",
                    f"event {event.name!r} uses unsupported phase {event.ph!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    for name, pending in open_events.items():
        if pending:
            failures.append(
                _failure(
                    "trace.gpt.unmatched_begin",
                    f"event {name!r} has {len(pending)} unmatched begin record(s)",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    return spans, failures


def _direct_parent(child: _Span, spans: Mapping[str, Sequence[_Span]]) -> str | None:
    enclosing = [
        candidate
        for candidates in spans.values()
        for candidate in candidates
        if candidate is not child
        and candidate.begin.rel_ts <= child.begin.rel_ts
        and child.end.rel_ts <= candidate.end.rel_ts
    ]
    if not enclosing:
        return None
    return min(
        enclosing, key=lambda span: (span.end.rel_ts - span.begin.rel_ts, -span.begin.rel_ts)
    ).begin.name


def _validate_iteration(
    iteration: Iteration, *, rank: int, expected_counts: Mapping[str, int]
) -> list[Failure]:
    iteration_id = int(iteration.iteration_id)
    failures: list[Failure] = []
    phase_counts = Counter(
        (event.name, event.ph) for event in iteration.events if event.name in _MODEL_PHASES
    )
    spans, pairing_failures = _pair_spans(iteration, _MODEL_PHASES, rank=rank)
    failures.extend(pairing_failures)

    for name, expected in expected_counts.items():
        observed = (phase_counts[(name, "B")], phase_counts[(name, "E")], len(spans.get(name, ())))
        if observed != (expected, expected, expected):
            failures.append(
                _failure(
                    "trace.gpt.count",
                    f"event {name!r} has B/E/spans={observed}, "
                    f"expected {expected}/{expected}/{expected}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    expected_parents = {
        "decoder": "forward-step",
        "decoder-postprocess": "forward-step",
        "output_layer": "decoder-postprocess",
        "loss": "decoder-postprocess",
    }
    for name, parent_name in expected_parents.items():
        for span in spans.get(name, ()):
            parent = _direct_parent(span, spans)
            if parent != parent_name:
                failures.append(
                    _failure(
                        "trace.gpt.parent",
                        f"event {name!r} has direct model-phase parent {parent!r}, "
                        f"expected {parent_name!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )

    decoder = spans.get("decoder", ())
    postprocess = spans.get("decoder-postprocess", ())
    if (
        len(decoder) == len(postprocess) == 1
        and decoder[0].end.rel_ts > postprocess[0].begin.rel_ts
    ):
        failures.append(
            _failure(
                "trace.gpt.order",
                "decoder-postprocess begins before decoder ends",
                rank=rank,
                iteration=iteration_id,
            )
        )

    output_layer = spans.get("output_layer", ())
    loss = spans.get("loss", ())
    if len(output_layer) == len(loss) == 1 and output_layer[0].end.rel_ts > loss[0].begin.rel_ts:
        failures.append(
            _failure(
                "trace.gpt.order",
                "loss begins before output_layer ends",
                rank=rank,
                iteration=iteration_id,
            )
        )
    return failures


def _validate_eager_layers(
    iteration: Iteration, *, rank: int, expected_layers: int, expected_calls: int = 1
) -> list[Failure]:
    observed = tuple(
        (event.name, event.ph) for event in iteration.events if event.name in _EAGER_TREE_PHASES
    )
    expected_call = (
        (("decoder", "B"),) + _EAGER_LAYER_SEQUENCE * expected_layers + (("decoder", "E"),)
    )
    expected = expected_call * expected_calls
    if observed == expected:
        return []
    return [
        _failure(
            "trace.gpt.eager_layers",
            f"eager Transformer sequence has {len(observed)} records, "
            f"expected {len(expected)} records for {expected_layers} layers "
            f"across {expected_calls} call(s)",
            rank=rank,
            iteration=int(iteration.iteration_id),
        )
    ]


def _validate_optimizer_phases(iteration: Iteration, *, rank: int) -> list[Failure]:
    observed = tuple(
        (event.name, event.ph) for event in iteration.events if event.name in _OPTIMIZER_PHASES
    )
    if observed == _OPTIMIZER_SEQUENCE:
        return []
    return [
        _failure(
            "trace.optimizer.sequence",
            f"optimizer phase sequence is {observed!r}, " f"expected {_OPTIMIZER_SEQUENCE!r}",
            rank=rank,
            iteration=int(iteration.iteration_id),
        )
    ]


def _validate_single_scope(
    iteration: Iteration, name: str, *, rank: int, expected: int = 1
) -> list[Failure]:
    spans, failures = _pair_spans(iteration, (name,), rank=rank)
    observed = len(spans.get(name, ()))
    if observed != expected:
        failures.append(
            _failure(
                "trace.gpt.count",
                f"event {name!r} has {observed} complete scope(s), expected {expected}",
                rank=rank,
                iteration=int(iteration.iteration_id),
            )
        )
    return failures


def _validate_cuda_graph_replay_inner_phases_absent(
    iteration: Iteration, *, rank: int
) -> list[Failure]:
    observed = Counter(
        event.name for event in iteration.events if event.name in _CUDA_GRAPH_INNER_PHASES
    )
    if not observed:
        return []
    return [
        _failure(
            "trace.gpt.cuda_graph_replay_inner",
            f"CUDA Graph replay contains inner Core events {dict(observed)!r}",
            rank=rank,
            iteration=int(iteration.iteration_id),
        )
    ]


def _validate_gpt_model_phases(
    trace_root: Path,
    *,
    expected_pipeline_ranks: Mapping[int, int],
    postprocess_ranks: frozenset[int],
    eager_layers_by_rank: Mapping[int, int] | None = None,
    validate_optimizer: bool = False,
    expected_iterations: tuple[int, ...] = (1, 2),
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    expected_ranks = tuple(sorted(expected_pipeline_ranks))
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != expected_ranks:
        failures.append(
            Failure(
                "trace.gpt.ranks",
                f"GPT phase contract expects ranks {list(expected_ranks)}, "
                f"observed {list(observed_ranks)}",
                "gpt-model-phases",
            )
        )

    for rank in expected_ranks:
        iterations = by_rank.get(rank, ())
        iteration_ids = tuple(item.iteration_id for item in iterations)
        if iteration_ids != expected_iterations:
            failures.append(
                Failure(
                    "trace.gpt.iterations",
                    f"rank {rank} expects iterations {list(expected_iterations)}, "
                    f"observed {list(iteration_ids)}",
                    f"rank={rank}",
                )
            )
        expected_pp_rank = expected_pipeline_ranks[rank]
        for iteration in iterations:
            iteration_id = int(iteration.iteration_id)
            observed_pp_ranks = {event.rank.pipeline for event in iteration.events}
            if observed_pp_ranks != {expected_pp_rank}:
                failures.append(
                    _failure(
                        "trace.gpt.pipeline_rank",
                        f"events use pipeline ranks {sorted(observed_pp_ranks)}, "
                        f"expected [{expected_pp_rank}]",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
            has_postprocess = rank in postprocess_ranks
            failures.extend(
                _validate_iteration(
                    iteration,
                    rank=rank,
                    expected_counts={
                        "forward-step": 1,
                        "decoder": 1,
                        "decoder-postprocess": 1,
                        "output_layer": int(has_postprocess),
                        "loss": int(has_postprocess),
                    },
                )
            )
            if eager_layers_by_rank is not None and rank in eager_layers_by_rank:
                failures.extend(
                    _validate_eager_layers(
                        iteration, rank=rank, expected_layers=eager_layers_by_rank[rank]
                    )
                )
            if validate_optimizer:
                failures.extend(_validate_optimizer_phases(iteration, rank=rank))
    return tuple(failures)


def validate_gpt_pp2_training_phases(trace_root: Path) -> tuple[Failure, ...]:
    """Validate PP2 GPT model and optimizer phase structure."""

    return _validate_gpt_model_phases(
        trace_root,
        expected_pipeline_ranks={0: 0, 1: 1},
        postprocess_ranks=frozenset((1,)),
        validate_optimizer=True,
    )


def validate_gpt_pp1_eager_phases(trace_root: Path) -> tuple[Failure, ...]:
    """Validate GPT model phases and two local eager Transformer layers."""

    return _validate_gpt_model_phases(
        trace_root,
        expected_pipeline_ranks={0: 0},
        postprocess_ranks=frozenset((0,)),
        eager_layers_by_rank={0: 2},
    )


def _validate_gpt_pp1_eager_kernel_windows(
    trace_root: Path, *, expected_iterations: tuple[int, int], shared_profiler_window: bool
) -> tuple[Failure, ...]:
    """Validate CUPTI ownership across two eager GPT iterations."""

    failures = list(
        _validate_gpt_model_phases(
            trace_root,
            expected_pipeline_ranks={0: 0},
            postprocess_ranks=frozenset((0,)),
            eager_layers_by_rank={0: 2},
            expected_iterations=expected_iterations,
        )
    )
    iterations = _load_iterations(trace_root).get(0, ())
    profiler_anchors: dict[int, set[int]] = defaultdict(set)
    iteration_anchors: dict[int, set[int]] = defaultdict(set)
    kernel_starts: dict[int, list[int]] = defaultdict(list)

    for iteration in iterations:
        iteration_id = int(iteration.iteration_id)
        kernels = [event for event in iteration.events if event.name == "cuda_kernel"]
        if not kernels:
            failures.append(
                _failure(
                    "trace.gpt.continuous_kernel_capture",
                    "continuous CUPTI window has no CUDA kernel records",
                    rank=0,
                    iteration=iteration_id,
                )
            )
            continue

        missing_fields: set[str] = set()
        ownership_errors = 0
        timeline_errors = 0
        positive_duration_kernels = 0
        for event in kernels:
            attrs = event.attrs
            event_missing = _CUDA_KERNEL_REQUIRED_FIELDS - attrs.keys()
            missing_fields.update(event_missing)
            if event_missing:
                continue
            if (
                event.ph != "X"
                or attrs["record_type"] != "cuda_kernel"
                or attrs["device"] != 0
                or attrs["iteration"] != iteration_id
                or tuple(attrs[field] for field in ("g_rk", "dp_rk", "pp_rk", "tp_rk"))
                != (0, 0, 0, 0)
            ):
                ownership_errors += 1
            try:
                start = int(attrs["start_us"])
                end = int(attrs["end_us"])
                wall_start = int(attrs["wall_start_us"])
                wall_end = int(attrs["wall_end_us"])
                iter_start = int(attrs["iter_rel_start_us"])
                iter_end = int(attrs["iter_rel_end_us"])
                duration = int(attrs["duration_us"])
            except (TypeError, ValueError):
                timeline_errors += 1
                continue
            if (
                end < start
                or wall_end < wall_start
                or iter_end < iter_start
                or duration < 0
                or wall_start - start != wall_end - end
                or wall_start - iter_start != wall_end - iter_end
            ):
                timeline_errors += 1
                continue
            profiler_anchors[iteration_id].add(wall_start - start)
            iteration_anchors[iteration_id].add(wall_start - iter_start)
            kernel_starts[iteration_id].append(wall_start)
            positive_duration_kernels += duration > 0

        if missing_fields:
            failures.append(
                _failure(
                    "trace.gpt.continuous_kernel_fields",
                    "CUDA kernel records lack required ownership or timeline fields: "
                    f"{sorted(missing_fields)}",
                    rank=0,
                    iteration=iteration_id,
                )
            )
        if ownership_errors:
            failures.append(
                _failure(
                    "trace.gpt.continuous_kernel_ownership",
                    f"{ownership_errors} CUDA kernel record(s) use an unexpected "
                    "rank, device, iteration, phase, or record type",
                    rank=0,
                    iteration=iteration_id,
                )
            )
        if timeline_errors or positive_duration_kernels == 0:
            failures.append(
                _failure(
                    "trace.gpt.continuous_kernel_timeline",
                    f"{timeline_errors} CUDA kernel record(s) have an invalid "
                    "rank-local device interval or no positive duration",
                    rank=0,
                    iteration=iteration_id,
                )
            )

    expected_iteration_set = set(expected_iterations)
    anchors_are_valid = (
        set(profiler_anchors) == expected_iteration_set
        and set(iteration_anchors) == expected_iteration_set
        and all(len(anchors) == 1 for anchors in profiler_anchors.values())
        and all(len(anchors) == 1 for anchors in iteration_anchors.values())
    )
    if anchors_are_valid:
        profiler_values = [
            next(iter(profiler_anchors[iteration])) for iteration in expected_iterations
        ]
        iteration_values = [
            next(iter(iteration_anchors[iteration])) for iteration in expected_iterations
        ]
        anchors_are_valid = iteration_values[0] < iteration_values[1]
        if shared_profiler_window:
            anchors_are_valid = (
                anchors_are_valid
                and len(set(profiler_values)) == 1
                and all(
                    start < iteration_values[1] for start in kernel_starts[expected_iterations[0]]
                )
                and all(
                    start >= iteration_values[1] for start in kernel_starts[expected_iterations[1]]
                )
            )
        else:
            anchors_are_valid = anchors_are_valid and len(set(profiler_values)) == len(
                expected_iterations
            )
    if not anchors_are_valid:
        failures.append(
            Failure(
                "trace.gpt.continuous_kernel_window",
                "CUPTI records do not use the expected profiler windows and "
                "ordered iteration anchors",
                f"rank=0 iterations={list(expected_iterations)}",
            )
        )
    return tuple(failures)


def validate_gpt_pp1_eager_continuous_kernel_phases(trace_root: Path) -> tuple[Failure, ...]:
    """Validate one CUPTI window spanning two eager GPT iterations."""

    return _validate_gpt_pp1_eager_kernel_windows(
        trace_root, expected_iterations=(1, 2), shared_profiler_window=True
    )


def _validate_layerwise_full_cuda_graph_phases(
    trace_root: Path, *, owner: str, profile_name: str
) -> tuple[Failure, ...]:
    """Validate eager then replay visibility for a two-layer whole-layer graph."""

    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != (0,):
        failures.append(
            Failure(
                "trace.gpt.ranks",
                f"{owner} whole-layer CUDA Graph expects only global rank 0; "
                f"observed {list(observed_ranks)}",
                profile_name,
            )
        )

    iterations = by_rank.get(0, ())
    iteration_ids = tuple(int(item.iteration_id) for item in iterations)
    if iteration_ids != (1, 2):
        failures.append(
            Failure(
                "trace.gpt.iterations",
                f"{owner} whole-layer CUDA Graph expects iterations [1, 2]; "
                f"observed {list(iteration_ids)}",
                "rank=0",
            )
        )

    for iteration in iterations:
        iteration_id = int(iteration.iteration_id)
        observed_pp_ranks = {event.rank.pipeline for event in iteration.events}
        if observed_pp_ranks != {0}:
            failures.append(
                _failure(
                    "trace.gpt.pipeline_rank",
                    f"events use pipeline ranks {sorted(observed_pp_ranks)}, expected [0]",
                    rank=0,
                    iteration=iteration_id,
                )
            )
        failures.extend(
            _validate_iteration(
                iteration,
                rank=0,
                expected_counts={
                    "forward-step": 1,
                    "decoder": 1,
                    "decoder-postprocess": 1,
                    "output_layer": 1,
                    "loss": 1,
                },
            )
        )
        failures.extend(_validate_single_scope(iteration, "backward-step", rank=0))
        failures.extend(_validate_optimizer_phases(iteration, rank=0))
        if iteration_id == 1:
            failures.extend(_validate_eager_layers(iteration, rank=0, expected_layers=2))
        elif iteration_id == 2:
            failures.extend(_validate_cuda_graph_replay_inner_phases_absent(iteration, rank=0))
    return tuple(failures)


def validate_te_full_cuda_graph_phases(trace_root: Path) -> tuple[Failure, ...]:
    """Validate TE whole-layer eager and replay visibility."""

    return _validate_layerwise_full_cuda_graph_phases(
        trace_root, owner="TE", profile_name="te-full-cuda-graph"
    )


def _validate_layerwise_cuda_graph_kernel_phases(
    trace_root: Path, *, framework_contract: Callable[[Path], tuple[Failure, ...]], owner: str
) -> tuple[Failure, ...]:
    """Validate rank-local device work for eager and layerwise Graph replay."""

    failures = list(framework_contract(trace_root))
    iterations = _load_iterations(trace_root).get(0, ())

    for iteration in iterations:
        iteration_id = int(iteration.iteration_id)
        kernels = [event for event in iteration.events if event.name == "cuda_kernel"]
        if not kernels:
            failures.append(
                _failure(
                    "trace.gpt.cuda_graph_kernel_capture",
                    f"{owner} CUDA Graph iteration has no CUDA kernel records",
                    rank=0,
                    iteration=iteration_id,
                )
            )
            continue

        missing_fields: set[str] = set()
        ownership_errors = 0
        timeline_errors = 0
        positive_duration_kernels = 0
        replay_compute_kernels = 0
        profiler_anchors: set[int] = set()
        iteration_anchors: set[int] = set()
        for event in kernels:
            attrs = event.attrs
            event_missing = _CUDA_KERNEL_REQUIRED_FIELDS - attrs.keys()
            missing_fields.update(event_missing)
            if event_missing:
                continue

            if (
                event.ph != "X"
                or attrs["record_type"] != "cuda_kernel"
                or attrs["device"] != 0
                or attrs["iteration"] != iteration_id
                or tuple(attrs[field] for field in ("g_rk", "dp_rk", "pp_rk", "tp_rk"))
                != (0, 0, 0, 0)
            ):
                ownership_errors += 1

            duration = 0
            try:
                start = int(attrs["start_us"])
                end = int(attrs["end_us"])
                wall_start = int(attrs["wall_start_us"])
                wall_end = int(attrs["wall_end_us"])
                iter_start = int(attrs["iter_rel_start_us"])
                iter_end = int(attrs["iter_rel_end_us"])
                duration = int(attrs["duration_us"])
            except (TypeError, ValueError):
                timeline_errors += 1
            else:
                if (
                    end < start
                    or wall_end < wall_start
                    or iter_end < iter_start
                    or duration < 0
                    or wall_start - start != wall_end - end
                    or wall_start - iter_start != wall_end - iter_end
                ):
                    timeline_errors += 1
                else:
                    profiler_anchors.add(wall_start - start)
                    iteration_anchors.add(wall_start - iter_start)
                    if duration > 0:
                        positive_duration_kernels += 1

            kernel_name = str(attrs["name"]).lower()
            if (
                iteration_id == 2
                and duration > 0
                and "nccl" not in kernel_name
                and any(marker in kernel_name for marker in _CUDA_GRAPH_MODEL_KERNEL_MARKERS)
            ):
                replay_compute_kernels += 1

        if (
            len(profiler_anchors) > 1
            or len(iteration_anchors) > 1
            or positive_duration_kernels == 0
        ):
            timeline_errors += 1

        if missing_fields:
            failures.append(
                _failure(
                    "trace.gpt.cuda_graph_kernel_fields",
                    "CUDA kernel records lack required ownership or timeline fields: "
                    f"{sorted(missing_fields)}",
                    rank=0,
                    iteration=iteration_id,
                )
            )
        if ownership_errors:
            failures.append(
                _failure(
                    "trace.gpt.cuda_graph_kernel_ownership",
                    f"{ownership_errors} CUDA kernel record(s) use an unexpected "
                    "rank, device, iteration, phase, or record type",
                    rank=0,
                    iteration=iteration_id,
                )
            )
        if timeline_errors:
            failures.append(
                _failure(
                    "trace.gpt.cuda_graph_kernel_timeline",
                    f"{timeline_errors} CUDA kernel record(s) have an invalid "
                    "rank-local device interval",
                    rank=0,
                    iteration=iteration_id,
                )
            )
        if iteration_id == 2 and replay_compute_kernels == 0:
            failures.append(
                _failure(
                    "trace.gpt.cuda_graph_replay_compute_kernel",
                    f"{owner} replay has no recognized model-compute kernel",
                    rank=0,
                    iteration=iteration_id,
                )
            )

    return tuple(failures)


def validate_te_full_cuda_graph_kernel_phases(trace_root: Path) -> tuple[Failure, ...]:
    """Validate rank-local device work for eager and TE Graph replay."""

    return _validate_layerwise_cuda_graph_kernel_phases(
        trace_root, framework_contract=validate_te_full_cuda_graph_phases, owner="TE whole-layer"
    )


def validate_gpt_cp_eager_phases(
    trace_root: Path,
    *,
    context_parallel_size: int,
    data_parallel_size: int,
    expected_layers: int = 2,
) -> tuple[Failure, ...]:
    """Validate CP ranks across all DP replicas sharing one complete GPT stage."""

    ranks = range(context_parallel_size * data_parallel_size)
    return _validate_gpt_model_phases(
        trace_root,
        expected_pipeline_ranks={rank: 0 for rank in ranks},
        postprocess_ranks=frozenset(ranks),
        eager_layers_by_rank={rank: expected_layers for rank in ranks},
    )


def validate_gpt_cp_dp1_eager_phases(
    trace_root: Path, *, context_parallel_size: int, expected_layers: int = 2
) -> tuple[Failure, ...]:
    """Validate DP1 CP ranks sharing one complete eager GPT stage."""

    return validate_gpt_cp_eager_phases(
        trace_root,
        context_parallel_size=context_parallel_size,
        data_parallel_size=1,
        expected_layers=expected_layers,
    )
