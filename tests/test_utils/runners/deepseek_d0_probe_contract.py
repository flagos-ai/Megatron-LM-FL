# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact offline Trace contract for the FlagOS DeepSeek D0 profile."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from megatron.megalens.trace_aggregate import (
    Event,
    Iteration,
    Rank,
    collect_benchmark_files,
    read_benchmark_file,
)
from tests.test_utils.runners.megalens_run_manifest import Failure

DEFAULT_MICROBATCHES_PER_ITERATION = 64

_RANKS = tuple(range(16))
_ITERATIONS = (1, 2)
_DATA_PARALLEL_SIZE = 8
_MODEL_SCOPE_NAMES = frozenset(
    ("forward-step", "decoder", "decoder-postprocess", "output_layer", "loss")
)
_MOE_NAMES = frozenset(
    (
        "moe-shared-expert",
        "moe-router",
        "moe-dispatch",
        "ep-alltoall-dispatch",
        "moe-experts",
        "moe-combine",
        "ep-alltoall-combine",
    )
)
_MOE_CALL_SEQUENCE = (
    ("moe-shared-expert", "B"),
    ("moe-shared-expert", "E"),
    ("moe-router", "B"),
    ("moe-router", "E"),
    ("moe-dispatch", "B"),
    ("ep-alltoall-dispatch", "B"),
    ("ep-alltoall-dispatch", "E"),
    ("moe-dispatch", "E"),
    ("moe-experts", "B"),
    ("moe-experts", "E"),
    ("moe-combine", "B"),
    ("ep-alltoall-combine", "B"),
    ("ep-alltoall-combine", "E"),
    ("moe-combine", "E"),
)
_MAIN_LAYERS = {
    0: tuple(range(2, 14)),
    1: tuple(range(14, 28)),
}
_MTP_LAYERS = {0: (), 1: (1,)}
_TOPOLOGY_FIELDS = {
    "ep_size": 4,
    "num_experts": 64,
    "num_local_experts": 16,
}
_ROUTER_HANDOFF_FIELDS = (
    "dropped_tokens",
    "drop_rate",
    "expert_cv",
    "top1_expert_share",
    "aux_loss",
    "z_loss",
)


@dataclass(frozen=True)
class _Span:
    begin_index: int
    end_index: int
    begin: Event
    end: Event


def _failure(
    code: str,
    message: str,
    *,
    rank: int,
    iteration: int | None,
    microbatch: int | None = None,
    region: str | None = None,
) -> Failure:
    evidence = f"rank={rank}"
    if iteration is not None:
        evidence += f" iteration={iteration}"
    if microbatch is not None:
        evidence += f" microbatch={microbatch}"
    if region is not None:
        evidence += f" region={region}"
    return Failure(f"trace.deepseek_d0.{code}", message, evidence)


def _load_iterations(
    trace_root: Path,
) -> Mapping[int, tuple[Rank, Sequence[Iteration]]]:
    result: dict[int, tuple[Rank, Sequence[Iteration]]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None or rank.global_rank in result:
            raise ValueError("DeepSeek D0 contract requires one shard per global rank")
        result[rank.global_rank] = (rank, tuple(read_benchmark_file(rank, content)))
    return result


def _pair_model_scopes(
    iteration: Iteration,
    *,
    rank: int,
) -> tuple[Mapping[str, Sequence[_Span]], list[Failure]]:
    open_scopes: list[tuple[int, Event]] = []
    spans: dict[str, list[_Span]] = defaultdict(list)
    failures: list[Failure] = []
    iteration_id = int(iteration.iteration_id)

    for index, event in enumerate(iteration.events):
        if event.name not in _MODEL_SCOPE_NAMES:
            continue
        if event.ph == "B":
            open_scopes.append((index, event))
            continue
        if event.ph != "E":
            failures.append(
                _failure(
                    "model_phase",
                    f"model scope {event.name!r} uses phase {event.ph!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        if not open_scopes or open_scopes[-1][1].name != event.name:
            expected = open_scopes[-1][1].name if open_scopes else None
            failures.append(
                _failure(
                    "model_nesting",
                    f"model scope {event.name!r} closes while {expected!r} is active",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        begin_index, begin = open_scopes.pop()
        spans[event.name].append(_Span(begin_index, index, begin, event))

    for _index, event in open_scopes:
        failures.append(
            _failure(
                "model_nesting",
                f"model scope {event.name!r} has no matching end",
                rank=rank,
                iteration=iteration_id,
            )
        )
    return {name: tuple(items) for name, items in spans.items()}, failures


def _inside(child: _Span, parent: _Span) -> bool:
    return parent.begin_index < child.begin_index and child.end_index < parent.end_index


def _moe_records(iteration: Iteration, span: _Span) -> list[Event]:
    return [
        event
        for event in iteration.events[span.begin_index + 1 : span.end_index]
        if event.name in _MOE_NAMES
    ]


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_field(
    failures: list[Failure],
    event: Event,
    field: str,
    expected: object,
    *,
    rank: int,
    iteration: int,
    microbatch: int,
    region: str,
    layer: int,
) -> None:
    observed = event.attrs.get(field)
    if observed == expected:
        return
    failures.append(
        _failure(
            "field",
            f"layer={layer} {event.name} field {field!r}={observed!r}, "
            f"expected {expected!r}",
            rank=rank,
            iteration=iteration,
            microbatch=microbatch,
            region=region,
        )
    )


def _validate_moe_call(
    records: Sequence[Event],
    *,
    layer: int,
    rank: int,
    iteration: int,
    microbatch: int,
    region: str,
) -> list[Failure]:
    failures: list[Failure] = []
    end_events = {event.name: event for event in records if event.ph == "E"}
    shared = end_events["moe-shared-expert"]
    router = end_events["moe-router"]
    dispatch = end_events["moe-dispatch"]
    experts = end_events["moe-experts"]
    combine = end_events["moe-combine"]

    for event in (shared, router, dispatch, experts, combine):
        _validate_field(
            failures,
            event,
            "layer",
            layer,
            rank=rank,
            iteration=iteration,
            microbatch=microbatch,
            region=region,
            layer=layer,
        )
        _validate_field(
            failures,
            event,
            "ep_size",
            4,
            rank=rank,
            iteration=iteration,
            microbatch=microbatch,
            region=region,
            layer=layer,
        )
    for event in (router, dispatch, experts, combine):
        for field, expected in _TOPOLOGY_FIELDS.items():
            _validate_field(
                failures,
                event,
                field,
                expected,
                rank=rank,
                iteration=iteration,
                microbatch=microbatch,
                region=region,
                layer=layer,
            )

    for event in (router, dispatch):
        _validate_field(
            failures,
            event,
            "router_topk",
            6,
            rank=rank,
            iteration=iteration,
            microbatch=microbatch,
            region=region,
            layer=layer,
        )
        _validate_field(
            failures,
            event,
            "num_tokens",
            4096,
            rank=rank,
            iteration=iteration,
            microbatch=microbatch,
            region=region,
            layer=layer,
        )
    _validate_field(
        failures,
        router,
        "routed_tokens",
        24576,
        rank=rank,
        iteration=iteration,
        microbatch=microbatch,
        region=region,
        layer=layer,
    )
    _validate_field(
        failures,
        dispatch,
        "dispatcher",
        "alltoall",
        rank=rank,
        iteration=iteration,
        microbatch=microbatch,
        region=region,
        layer=layer,
    )
    _validate_field(
        failures,
        dispatch,
        "capacity_factor",
        None,
        rank=rank,
        iteration=iteration,
        microbatch=microbatch,
        region=region,
        layer=layer,
    )
    _validate_field(
        failures,
        combine,
        "dispatcher",
        "alltoall",
        rank=rank,
        iteration=iteration,
        microbatch=microbatch,
        region=region,
        layer=layer,
    )
    _validate_field(
        failures,
        combine,
        "num_tokens",
        4096,
        rank=rank,
        iteration=iteration,
        microbatch=microbatch,
        region=region,
        layer=layer,
    )

    for field in _ROUTER_HANDOFF_FIELDS:
        router_value = router.attrs.get(field)
        dispatch_value = dispatch.attrs.get(field)
        if router_value != dispatch_value:
            failures.append(
                _failure(
                    "router_handoff",
                    f"layer={layer} field {field!r} differs between Router and Dispatch: "
                    f"{router_value!r} != {dispatch_value!r}",
                    rank=rank,
                    iteration=iteration,
                    microbatch=microbatch,
                    region=region,
                )
            )
    for field, expected in (("dropped_tokens", 0), ("drop_rate", 0.0), ("z_loss", None)):
        _validate_field(
            failures,
            router,
            field,
            expected,
            rank=rank,
            iteration=iteration,
            microbatch=microbatch,
            region=region,
            layer=layer,
        )
    for field in ("expert_cv", "top1_expert_share", "routing_entropy", "aux_loss"):
        if not _is_finite_number(router.attrs.get(field)):
            failures.append(
                _failure(
                    "router_metric",
                    f"layer={layer} Router field {field!r}="
                    f"{router.attrs.get(field)!r}, expected a finite number",
                    rank=rank,
                    iteration=iteration,
                    microbatch=microbatch,
                    region=region,
                )
            )

    tokens_per_expert = experts.attrs.get("tokens_per_expert")
    routed_tokens = experts.attrs.get("routed_tokens")
    valid_counts = (
        isinstance(tokens_per_expert, list)
        and len(tokens_per_expert) == 16
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in tokens_per_expert
        )
    )
    if not valid_counts or sum(tokens_per_expert) != routed_tokens:
        failures.append(
            _failure(
                "expert_workload",
                f"layer={layer} tokens_per_expert must contain 16 non-negative counts "
                "whose sum equals routed_tokens",
                rank=rank,
                iteration=iteration,
                microbatch=microbatch,
                region=region,
            )
        )
    for field in ("expert_cv", "top1_expert_share", "expert_max_over_mean"):
        if not _is_finite_number(experts.attrs.get(field)):
            failures.append(
                _failure(
                    "expert_metric",
                    f"layer={layer} Experts field {field!r}="
                    f"{experts.attrs.get(field)!r}, expected a finite number",
                    rank=rank,
                    iteration=iteration,
                    microbatch=microbatch,
                    region=region,
                )
            )

    for collective_name in ("ep-alltoall-dispatch", "ep-alltoall-combine"):
        collective = end_events[collective_name]
        for field, expected in (
            ("comm_type", "ep-alltoall"),
            ("dispatcher", "alltoall"),
            ("group_size", 4),
            ("ep_size", 4),
            ("tp_size", 1),
        ):
            _validate_field(
                failures,
                collective,
                field,
                expected,
                rank=rank,
                iteration=iteration,
                microbatch=microbatch,
                region=region,
                layer=layer,
            )
        data_bytes = collective.attrs.get("data_bytes")
        if not isinstance(data_bytes, int) or isinstance(data_bytes, bool) or data_bytes <= 0:
            failures.append(
                _failure(
                    "collective_bytes",
                    f"layer={layer} {collective_name} data_bytes="
                    f"{collective.attrs.get('data_bytes')!r}, expected a positive integer",
                    rank=rank,
                    iteration=iteration,
                    microbatch=microbatch,
                    region=region,
                )
            )
    return failures


def _validate_moe_region(
    iteration: Iteration,
    span: _Span,
    *,
    expected_layers: Sequence[int],
    rank: int,
    microbatch: int,
    region: str,
) -> list[Failure]:
    iteration_id = int(iteration.iteration_id)
    records = _moe_records(iteration, span)
    expected_sequence = _MOE_CALL_SEQUENCE * len(expected_layers)
    observed_sequence = tuple((event.name, event.ph) for event in records)
    if observed_sequence != expected_sequence:
        return [
            _failure(
                "moe_sequence",
                f"expected {len(expected_layers)} complete shared-expert/MoE calls "
                f"({len(expected_sequence)} records), observed {len(records)} records",
                rank=rank,
                iteration=iteration_id,
                microbatch=microbatch,
                region=region,
            )
        ]

    failures: list[Failure] = []
    records_per_call = len(_MOE_CALL_SEQUENCE)
    for occurrence, layer in enumerate(expected_layers):
        start = occurrence * records_per_call
        failures.extend(
            _validate_moe_call(
                records[start : start + records_per_call],
                layer=layer,
                rank=rank,
                iteration=iteration_id,
                microbatch=microbatch,
                region=region,
            )
        )
    return failures


def _contained(spans: Iterable[_Span], parent: _Span) -> list[_Span]:
    return sorted(
        (span for span in spans if _inside(span, parent)),
        key=lambda span: span.begin_index,
    )


def _validate_iteration(
    iteration: Iteration,
    *,
    rank: int,
    pipeline_rank: int,
    microbatches_per_iteration: int,
) -> list[Failure]:
    if iteration.iteration_id is None:
        return [
            _failure(
                "iteration",
                "D0 scopes are outside a numbered iteration",
                rank=rank,
                iteration=None,
            )
        ]
    iteration_id = int(iteration.iteration_id)
    spans, failures = _pair_model_scopes(iteration, rank=rank)
    phase_counts = Counter(
        (event.name, event.ph)
        for event in iteration.events
        if event.name in _MODEL_SCOPE_NAMES
    )
    expected_counts = {
        "forward-step": microbatches_per_iteration,
        "decoder": microbatches_per_iteration,
        "decoder-postprocess": microbatches_per_iteration,
        "output_layer": microbatches_per_iteration if pipeline_rank == 1 else 0,
        "loss": microbatches_per_iteration if pipeline_rank == 1 else 0,
    }
    for name, expected in expected_counts.items():
        observed = (phase_counts[(name, "B")], phase_counts[(name, "E")], len(spans.get(name, ())))
        if observed != (expected, expected, expected):
            failures.append(
                _failure(
                    "model_count",
                    f"event {name!r} has B/E/spans={observed}, expected "
                    f"{expected}/{expected}/{expected}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

    forward_spans = sorted(spans.get("forward-step", ()), key=lambda span: span.begin_index)
    for microbatch, forward in enumerate(forward_spans):
        decoder = _contained(spans.get("decoder", ()), forward)
        postprocess = _contained(spans.get("decoder-postprocess", ()), forward)
        if len(decoder) != 1 or len(postprocess) != 1:
            failures.append(
                _failure(
                    "forward_tree",
                    f"forward-step contains {len(decoder)} decoder and "
                    f"{len(postprocess)} decoder-postprocess scopes, expected one each",
                    rank=rank,
                    iteration=iteration_id,
                    microbatch=microbatch,
                )
            )
            continue
        if decoder[0].end_index > postprocess[0].begin_index:
            failures.append(
                _failure(
                    "forward_order",
                    "decoder-postprocess begins before decoder ends",
                    rank=rank,
                    iteration=iteration_id,
                    microbatch=microbatch,
                )
            )
        failures.extend(
            _validate_moe_region(
                iteration,
                decoder[0],
                expected_layers=_MAIN_LAYERS[pipeline_rank],
                rank=rank,
                microbatch=microbatch,
                region="decoder",
            )
        )
        failures.extend(
            _validate_moe_region(
                iteration,
                postprocess[0],
                expected_layers=_MTP_LAYERS[pipeline_rank],
                rank=rank,
                microbatch=microbatch,
                region="decoder-postprocess",
            )
        )
        output_layers = _contained(spans.get("output_layer", ()), postprocess[0])
        losses = _contained(spans.get("loss", ()), postprocess[0])
        expected_postprocess = 1 if pipeline_rank == 1 else 0
        if len(output_layers) != expected_postprocess or len(losses) != expected_postprocess:
            failures.append(
                _failure(
                    "postprocess_tree",
                    f"decoder-postprocess contains {len(output_layers)} output_layer and "
                    f"{len(losses)} loss scopes, expected {expected_postprocess} each",
                    rank=rank,
                    iteration=iteration_id,
                    microbatch=microbatch,
                    region="decoder-postprocess",
                )
            )
    return failures


def validate_deepseek_d0_trace(
    trace_root: Path,
    *,
    microbatches_per_iteration: int = DEFAULT_MICROBATCHES_PER_ITERATION,
) -> tuple[Failure, ...]:
    """Validate the guide D0 PP2/DP8/EP4 MoE, shared-expert, and MTP trace."""

    if microbatches_per_iteration < 1:
        raise ValueError("microbatches_per_iteration must be positive")
    by_rank = _load_iterations(trace_root)
    failures: list[Failure] = []
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != _RANKS:
        failures.append(
            Failure(
                "trace.deepseek_d0.ranks",
                f"expected ranks {_RANKS}, observed {observed_ranks}",
                "deepseek-d0",
            )
        )

    for global_rank in _RANKS:
        loaded = by_rank.get(global_rank)
        if loaded is None:
            continue
        shard_rank, iterations = loaded
        pipeline_rank = global_rank // _DATA_PARALLEL_SIZE
        data_rank = global_rank % _DATA_PARALLEL_SIZE
        expected_coordinates = (data_rank, pipeline_rank, 0)
        observed_coordinates = (shard_rank.data, shard_rank.pipeline, shard_rank.tensor)
        if observed_coordinates != expected_coordinates:
            failures.append(
                _failure(
                    "coordinates",
                    f"shard coordinates={observed_coordinates}, expected {expected_coordinates}",
                    rank=global_rank,
                    iteration=None,
                )
            )
        observed_iterations = tuple(iteration.iteration_id for iteration in iterations)
        if observed_iterations != _ITERATIONS:
            failures.append(
                _failure(
                    "iterations",
                    f"expected iteration IDs {_ITERATIONS}, observed {observed_iterations}",
                    rank=global_rank,
                    iteration=None,
                )
            )
        for iteration in iterations:
            failures.extend(
                _validate_iteration(
                    iteration,
                    rank=global_rank,
                    pipeline_rank=pipeline_rank,
                    microbatches_per_iteration=microbatches_per_iteration,
                )
            )
    return tuple(failures)


__all__ = ["DEFAULT_MICROBATCHES_PER_ITERATION", "validate_deepseek_d0_trace"]
