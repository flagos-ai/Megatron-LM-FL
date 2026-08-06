# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact trace contracts for the controlled GPT training profiles."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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
    iteration: Iteration,
    names: Iterable[str],
    *,
    rank: int,
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
        enclosing,
        key=lambda span: (
            span.end.rel_ts - span.begin.rel_ts,
            -span.begin.rel_ts,
        ),
    ).begin.name


def _validate_iteration(
    iteration: Iteration,
    *,
    rank: int,
    expected_counts: Mapping[str, int],
) -> list[Failure]:
    iteration_id = int(iteration.iteration_id)
    failures: list[Failure] = []
    phase_counts = Counter(
        (event.name, event.ph)
        for event in iteration.events
        if event.name in _MODEL_PHASES
    )
    spans, pairing_failures = _pair_spans(iteration, _MODEL_PHASES, rank=rank)
    failures.extend(pairing_failures)

    for name, expected in expected_counts.items():
        observed = (
            phase_counts[(name, "B")],
            phase_counts[(name, "E")],
            len(spans.get(name, ())),
        )
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
    if len(decoder) == len(postprocess) == 1 and decoder[0].end.rel_ts > postprocess[0].begin.rel_ts:
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
    if (
        len(output_layer) == len(loss) == 1
        and output_layer[0].end.rel_ts > loss[0].begin.rel_ts
    ):
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
    iteration: Iteration,
    *,
    rank: int,
    expected_layers: int,
) -> list[Failure]:
    observed = tuple(
        (event.name, event.ph)
        for event in iteration.events
        if event.name in _EAGER_TREE_PHASES
    )
    expected = (("decoder", "B"),) + _EAGER_LAYER_SEQUENCE * expected_layers + (
        ("decoder", "E"),
    )
    if observed == expected:
        return []
    return [
        _failure(
            "trace.gpt.eager_layers",
            f"eager Transformer sequence has {len(observed)} records, "
            f"expected {len(expected)} records for {expected_layers} layers",
            rank=rank,
            iteration=int(iteration.iteration_id),
        )
    ]


def _validate_optimizer_phases(
    iteration: Iteration,
    *,
    rank: int,
) -> list[Failure]:
    observed = tuple(
        (event.name, event.ph)
        for event in iteration.events
        if event.name in _OPTIMIZER_PHASES
    )
    if observed == _OPTIMIZER_SEQUENCE:
        return []
    return [
        _failure(
            "trace.optimizer.sequence",
            f"optimizer phase sequence is {observed!r}, "
            f"expected {_OPTIMIZER_SEQUENCE!r}",
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
        if iteration_ids != (1, 2):
            failures.append(
                Failure(
                    "trace.gpt.iterations",
                    f"rank {rank} expects iterations [1, 2], "
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
                        iteration,
                        rank=rank,
                        expected_layers=eager_layers_by_rank[rank],
                    )
                )
            if validate_optimizer:
                failures.extend(_validate_optimizer_phases(iteration, rank=rank))
    return tuple(failures)


def validate_gpt_pp1_model_phases(trace_root: Path) -> tuple[Failure, ...]:
    """Validate one-microbatch GPT model phases on a single pipeline stage."""

    return _validate_gpt_model_phases(
        trace_root,
        expected_pipeline_ranks={0: 0},
        postprocess_ranks=frozenset((0,)),
    )


def validate_gpt_pp2_model_phases(trace_root: Path) -> tuple[Failure, ...]:
    """Validate one-microbatch GPT model phases across two pipeline stages."""

    return _validate_gpt_model_phases(
        trace_root,
        expected_pipeline_ranks={0: 0, 1: 1},
        postprocess_ranks=frozenset((1,)),
    )


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


def _validate_qwen3_tp_eager_phases(
    trace_root: Path, *, tensor_parallel_size: int
) -> tuple[Failure, ...]:
    """Validate one full 28-layer Qwen3 stage on every TP rank."""

    ranks = range(tensor_parallel_size)
    return _validate_gpt_model_phases(
        trace_root,
        expected_pipeline_ranks={rank: 0 for rank in ranks},
        postprocess_ranks=frozenset(ranks),
        eager_layers_by_rank={rank: 28 for rank in ranks},
    )


def validate_qwen3_tp2_eager_phases(trace_root: Path) -> tuple[Failure, ...]:
    """Validate one full 28-layer Qwen3 stage on both TP2 ranks."""

    return _validate_qwen3_tp_eager_phases(trace_root, tensor_parallel_size=2)


def validate_qwen3_tp4_eager_phases(trace_root: Path) -> tuple[Failure, ...]:
    """Validate one full 28-layer Qwen3 stage on all TP4 ranks."""

    return _validate_qwen3_tp_eager_phases(trace_root, tensor_parallel_size=4)


def validate_qwen3_tp8_eager_phases(trace_root: Path) -> tuple[Failure, ...]:
    """Validate one full 28-layer Qwen3 stage on all TP8 ranks."""

    return _validate_qwen3_tp_eager_phases(trace_root, tensor_parallel_size=8)


def validate_gpt_cp_dp1_eager_phases(
    trace_root: Path, *, context_parallel_size: int, expected_layers: int = 2
) -> tuple[Failure, ...]:
    """Validate DP1 CP ranks sharing one complete eager GPT stage."""

    ranks = range(context_parallel_size)
    return _validate_gpt_model_phases(
        trace_root,
        expected_pipeline_ranks={rank: 0 for rank in ranks},
        postprocess_ranks=frozenset(ranks),
        eager_layers_by_rank={rank: expected_layers for rank in ranks},
    )


def validate_gpt_cp2_eager_phases(trace_root: Path) -> tuple[Failure, ...]:
    """Validate two CP ranks sharing one complete two-layer GPT stage."""

    return validate_gpt_cp_dp1_eager_phases(
        trace_root, context_parallel_size=2
    )
