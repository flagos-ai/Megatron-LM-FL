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
D0_RANK_ORDER = "tp-cp-ep-dp-pp"

_RANKS = tuple(range(16))
_ITERATIONS = (1, 2)
_DATA_PARALLEL_SIZE = 8
_EXPERT_MODEL_PARALLEL_SIZE = 4
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
    valid_routed_tokens = (
        isinstance(routed_tokens, int)
        and not isinstance(routed_tokens, bool)
        and routed_tokens >= 0
    )
    expert_token_total = sum(tokens_per_expert) if valid_counts else None
    if (
        not valid_counts
        or not valid_routed_tokens
        or expert_token_total != routed_tokens
    ):
        failures.append(
            _failure(
                "expert_workload",
                f"layer={layer} tokens_per_expert must contain 16 non-negative counts "
                "whose sum equals a non-negative routed_tokens value",
                rank=rank,
                iteration=iteration,
                microbatch=microbatch,
                region=region,
            )
        )

    combine_num_tokens = combine.attrs.get("num_tokens")
    if (
        not isinstance(combine_num_tokens, int)
        or isinstance(combine_num_tokens, bool)
        or combine_num_tokens < 0
        or not valid_routed_tokens
        or combine_num_tokens != routed_tokens
        or combine_num_tokens != expert_token_total
    ):
        failures.append(
            _failure(
                "combine_workload",
                f"layer={layer} Combine num_tokens={combine_num_tokens!r} must equal "
                f"Experts routed_tokens={routed_tokens!r} and "
                f"sum(tokens_per_expert)={expert_token_total!r}",
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
        valid_data_bytes = isinstance(data_bytes, int) and not isinstance(data_bytes, bool)
        if collective_name == "ep-alltoall-combine" and routed_tokens == 0:
            valid_data_bytes = valid_data_bytes and data_bytes == 0
            expected_data_bytes = "zero for an empty local expert workload"
        else:
            valid_data_bytes = valid_data_bytes and data_bytes > 0
            expected_data_bytes = "a positive integer"
        if not valid_data_bytes:
            failures.append(
                _failure(
                    "collective_bytes",
                    f"layer={layer} {collective_name} data_bytes="
                    f"{data_bytes!r}, expected {expected_data_bytes}",
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


def _collect_moe_workloads(
    iteration: Iteration,
    *,
    rank: int,
) -> Mapping[tuple[int, int], tuple[int, int]]:
    """Return unique per-microbatch Router/Experts assignment totals."""

    spans, scope_failures = _pair_model_scopes(iteration, rank=rank)
    if scope_failures:
        return {}

    workloads: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    duplicate_keys: set[tuple[int, int]] = set()
    forward_spans = sorted(spans.get("forward-step", ()), key=lambda span: span.begin_index)
    for microbatch, forward in enumerate(forward_spans):
        for event in iteration.events[forward.begin_index + 1 : forward.end_index]:
            if event.ph != "E" or event.name not in ("moe-router", "moe-experts"):
                continue
            layer = event.attrs.get("layer")
            field = "routed_tokens"
            value = event.attrs.get(field)
            if (
                not isinstance(layer, int)
                or isinstance(layer, bool)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                continue
            key = (microbatch, layer)
            if event.name in workloads[key]:
                duplicate_keys.add(key)
                continue
            workloads[key][event.name] = value

    return {
        key: (values["moe-router"], values["moe-experts"])
        for key, values in workloads.items()
        if key not in duplicate_keys
        and "moe-router" in values
        and "moe-experts" in values
    }


def _validate_ep_workload_conservation(
    by_rank: Mapping[int, tuple[Rank, Sequence[Iteration]]],
    *,
    microbatches_per_iteration: int,
) -> list[Failure]:
    """Require routed assignments to be conserved within every D0 EP4 group."""

    workloads: dict[tuple[int, int], Mapping[tuple[int, int], tuple[int, int]]] = {}
    for rank, (_shard_rank, iterations) in by_rank.items():
        for iteration in iterations:
            if iteration.iteration_id is None:
                continue
            workloads[(rank, int(iteration.iteration_id))] = _collect_moe_workloads(
                iteration,
                rank=rank,
            )

    failures: list[Failure] = []
    expert_data_replicas = _DATA_PARALLEL_SIZE // _EXPERT_MODEL_PARALLEL_SIZE
    for pipeline_rank in (0, 1):
        expected_layers = _MAIN_LAYERS[pipeline_rank] + _MTP_LAYERS[pipeline_rank]
        stage_base = pipeline_rank * _DATA_PARALLEL_SIZE
        for expert_data_rank in range(expert_data_replicas):
            group_base = stage_base + expert_data_rank * _EXPERT_MODEL_PARALLEL_SIZE
            group_ranks = tuple(
                range(group_base, group_base + _EXPERT_MODEL_PARALLEL_SIZE)
            )
            for iteration in _ITERATIONS:
                for microbatch in range(microbatches_per_iteration):
                    for layer in expected_layers:
                        group_values = [
                            workloads.get((rank, iteration), {}).get((microbatch, layer))
                            for rank in group_ranks
                        ]
                        if any(value is None for value in group_values):
                            # The per-rank structural contract reports the missing or
                            # malformed call; conservation only evaluates complete groups.
                            continue
                        router_total = sum(value[0] for value in group_values if value is not None)
                        experts_total = sum(value[1] for value in group_values if value is not None)
                        if router_total == experts_total:
                            continue
                        failures.append(
                            _failure(
                                "ep_conservation",
                                f"EP group {group_ranks} layer={layer} Router routed_tokens "
                                f"total={router_total}, Experts routed_tokens "
                                f"total={experts_total}",
                                rank=group_ranks[0],
                                iteration=iteration,
                                microbatch=microbatch,
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
        # D0 relies on Megatron's target-side default tp-cp-ep-dp-pp rank order.
        # With TP=CP=1, PP is the slowest-varying axis and each PP stage owns
        # eight consecutive data-parallel ranks.
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
    failures.extend(
        _validate_ep_workload_conservation(
            by_rank,
            microbatches_per_iteration=microbatches_per_iteration,
        )
    )
    return tuple(failures)


__all__ = [
    "D0_RANK_ORDER",
    "DEFAULT_MICROBATCHES_PER_ITERATION",
    "validate_deepseek_d0_trace",
]
