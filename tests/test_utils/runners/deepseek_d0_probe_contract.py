# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact offline Trace contracts for the FlagOS DeepSeek D0/D2 profiles."""

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
from tests.test_utils.runners import dp_probe_contract
from tests.test_utils.runners.megalens_run_manifest import Failure

DEFAULT_MICROBATCHES_PER_ITERATION = 64
DEFAULT_DATA_PARALLEL_SIZE = 8
D2_DATA_PARALLEL_SIZE = 8
D2_EXPERT_MODEL_PARALLEL_SIZE = 8
D3_DATA_PARALLEL_SIZE = 8
D3_EXPERT_MODEL_PARALLEL_SIZE = 4
D0_RANK_ORDER = "tp-cp-ep-dp-pp"

_ITERATIONS = (1, 2)
_PIPELINE_MODEL_PARALLEL_SIZE = 2
_D3_PIPELINE_MODEL_PARALLEL_SIZE = 1
_D0_EXPERT_MODEL_PARALLEL_SIZE = 4
_SUPPORTED_DATA_PARALLEL_SIZES = frozenset((4, DEFAULT_DATA_PARALLEL_SIZE))
_REVIEWED_TOPOLOGIES = frozenset(
    (
        (_PIPELINE_MODEL_PARALLEL_SIZE, 4, _D0_EXPERT_MODEL_PARALLEL_SIZE),
        (
            _PIPELINE_MODEL_PARALLEL_SIZE,
            DEFAULT_DATA_PARALLEL_SIZE,
            _D0_EXPERT_MODEL_PARALLEL_SIZE,
        ),
        (
            _PIPELINE_MODEL_PARALLEL_SIZE,
            D2_DATA_PARALLEL_SIZE,
            D2_EXPERT_MODEL_PARALLEL_SIZE,
        ),
        (
            _D3_PIPELINE_MODEL_PARALLEL_SIZE,
            D3_DATA_PARALLEL_SIZE,
            D3_EXPERT_MODEL_PARALLEL_SIZE,
        ),
    )
)
_NUM_EXPERTS = 64
_EXPERT_TENSOR_PARALLEL_SIZE = 1
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
_D3_MAIN_LAYERS = {0: tuple(range(2, 28))}
_D3_MTP_LAYERS = {0: (1,)}
_ROUTER_HANDOFF_FIELDS = (
    "dropped_tokens",
    "drop_rate",
    "expert_cv",
    "top1_expert_share",
    "aux_loss",
    "z_loss",
)
_DP_GROUP_ROUTES = {
    "dp-reduce-scatter": "reduce_scatter",
    "dp-param-all-gather": "all_gather",
}
_DP_COMPLETION_EVENTS = frozenset(("dp-grad-sync-complete", "dp-param-sync-complete"))
_DP_LIFECYCLE_EVENTS = frozenset(_DP_GROUP_ROUTES) | _DP_COMPLETION_EVENTS
_ETP_TP_COLLECTIVES = frozenset(
    (
        "tp-all-gather-first",
        "tp-all-gather-last",
        "tp-reduce-scatter",
        "tp-reduce-scatter-last",
    )
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
    expert_model_parallel_size: int,
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
    num_local_experts = _NUM_EXPERTS // expert_model_parallel_size

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
            expert_model_parallel_size,
            rank=rank,
            iteration=iteration,
            microbatch=microbatch,
            region=region,
            layer=layer,
        )
    for event in (router, dispatch, experts, combine):
        for field, expected in (
            ("ep_size", expert_model_parallel_size),
            ("num_experts", _NUM_EXPERTS),
            ("num_local_experts", num_local_experts),
        ):
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
        router,
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
        "num_tokens",
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
    for field, expected in (
        ("dropped_tokens", 0),
        ("drop_rate", 0.0),
        ("z_loss", None),
    ):
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
        and len(tokens_per_expert) == num_local_experts
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
                f"layer={layer} tokens_per_expert must contain "
                f"{num_local_experts} non-negative counts "
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
    if (
        expert_model_parallel_size == D2_EXPERT_MODEL_PARALLEL_SIZE
        and valid_counts
        and valid_routed_tokens
        and expert_token_total == routed_tokens
    ):
        if routed_tokens > 0:
            mean_tokens = routed_tokens / len(tokens_per_expert)
            variance = sum(
                (value - mean_tokens) ** 2 for value in tokens_per_expert
            ) / len(tokens_per_expert)
            max_tokens = max(tokens_per_expert)
            expected_metrics = {
                "expert_cv": math.sqrt(variance) / mean_tokens,
                "top1_expert_share": max_tokens / routed_tokens,
                "expert_max_over_mean": max_tokens / mean_tokens,
            }
        else:
            expected_metrics = {
                "expert_cv": 0.0,
                "top1_expert_share": 0.0,
                "expert_max_over_mean": 0.0,
            }
        for field, expected in expected_metrics.items():
            observed = experts.attrs.get(field)
            if not _is_finite_number(observed) or not math.isclose(
                float(observed),
                expected,
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                failures.append(
                    _failure(
                        "expert_metric",
                        f"layer={layer} Experts field {field!r}={observed!r}, "
                        f"expected {expected!r} from tokens_per_expert",
                        rank=rank,
                        iteration=iteration,
                        microbatch=microbatch,
                        region=region,
                    )
                )
    else:
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
            ("group_size", expert_model_parallel_size),
            ("ep_size", expert_model_parallel_size),
            ("tp_size", _EXPERT_TENSOR_PARALLEL_SIZE),
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
        valid_data_bytes = isinstance(data_bytes, int) and not isinstance(
            data_bytes, bool
        )
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
    expert_model_parallel_size: int,
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
                expert_model_parallel_size=expert_model_parallel_size,
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
    pipeline_model_parallel_size: int,
    microbatches_per_iteration: int,
    expert_model_parallel_size: int,
    main_layers: Mapping[int, Sequence[int]],
    mtp_layers: Mapping[int, Sequence[int]],
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
        "output_layer": (
            microbatches_per_iteration
            if pipeline_rank == pipeline_model_parallel_size - 1
            else 0
        ),
        "loss": (
            microbatches_per_iteration
            if pipeline_rank == pipeline_model_parallel_size - 1
            else 0
        ),
    }
    for name, expected in expected_counts.items():
        observed = (
            phase_counts[(name, "B")],
            phase_counts[(name, "E")],
            len(spans.get(name, ())),
        )
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

    forward_spans = sorted(
        spans.get("forward-step", ()), key=lambda span: span.begin_index
    )
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
                expert_model_parallel_size=expert_model_parallel_size,
                expected_layers=main_layers[pipeline_rank],
                rank=rank,
                microbatch=microbatch,
                region="decoder",
            )
        )
        failures.extend(
            _validate_moe_region(
                iteration,
                postprocess[0],
                expert_model_parallel_size=expert_model_parallel_size,
                expected_layers=mtp_layers[pipeline_rank],
                rank=rank,
                microbatch=microbatch,
                region="decoder-postprocess",
            )
        )
        output_layers = _contained(spans.get("output_layer", ()), postprocess[0])
        losses = _contained(spans.get("loss", ()), postprocess[0])
        expected_postprocess = (
            1 if pipeline_rank == pipeline_model_parallel_size - 1 else 0
        )
        if (
            len(output_layers) != expected_postprocess
            or len(losses) != expected_postprocess
        ):
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
    forward_spans = sorted(
        spans.get("forward-step", ()), key=lambda span: span.begin_index
    )
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
    pipeline_model_parallel_size: int,
    microbatches_per_iteration: int,
    data_parallel_size: int,
    expert_model_parallel_size: int,
    main_layers: Mapping[int, Sequence[int]],
    mtp_layers: Mapping[int, Sequence[int]],
) -> list[Failure]:
    """Require routed assignments to be conserved within every reviewed EP group."""

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
    expert_data_replicas = data_parallel_size // expert_model_parallel_size
    for pipeline_rank in range(pipeline_model_parallel_size):
        expected_layers = tuple(main_layers[pipeline_rank]) + tuple(
            mtp_layers[pipeline_rank]
        )
        stage_base = pipeline_rank * data_parallel_size
        for expert_data_rank in range(expert_data_replicas):
            group_base = stage_base + expert_data_rank * expert_model_parallel_size
            group_ranks = tuple(
                range(group_base, group_base + expert_model_parallel_size)
            )
            for iteration in _ITERATIONS:
                for microbatch in range(microbatches_per_iteration):
                    for layer in expected_layers:
                        group_values = [
                            workloads.get((rank, iteration), {}).get(
                                (microbatch, layer)
                            )
                            for rank in group_ranks
                        ]
                        if any(value is None for value in group_values):
                            # The per-rank structural contract reports the missing or
                            # malformed call; conservation only evaluates complete groups.
                            continue
                        router_total = sum(
                            value[0] for value in group_values if value is not None
                        )
                        experts_total = sum(
                            value[1] for value in group_values if value is not None
                        )
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


def _d2_failure(
    code: str,
    message: str,
    *,
    rank: int,
    iteration: int | None,
) -> Failure:
    evidence = f"rank={rank}"
    if iteration is not None:
        evidence += f" iteration={iteration}"
    return Failure(f"trace.deepseek_d2.{code}", message, evidence)


def _pair_dp_scopes(
    iteration: Iteration,
    *,
    rank: int,
) -> tuple[Mapping[str, Sequence[_Span]], list[Failure]]:
    open_scopes: list[tuple[int, Event]] = []
    spans: dict[str, list[_Span]] = defaultdict(list)
    failures: list[Failure] = []
    iteration_id = (
        None if iteration.iteration_id is None else int(iteration.iteration_id)
    )

    for index, event in enumerate(iteration.events):
        if event.name not in _DP_LIFECYCLE_EVENTS:
            continue
        if event.ph == "B":
            if open_scopes:
                active = open_scopes[-1][1]
                failures.append(
                    _d2_failure(
                        "dp_group_nesting",
                        f"event {event.name!r} begins while DP lifecycle scope "
                        f"{active.name!r} is active; DP route and completion "
                        "scopes must not overlap or nest",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
            open_scopes.append((index, event))
            continue
        if event.ph != "E" or not open_scopes:
            failures.append(
                _d2_failure(
                    "dp_group_nesting",
                    f"event {event.name!r} has unmatched phase {event.ph!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        if open_scopes[-1][1].name != event.name:
            failures.append(
                _d2_failure(
                    "dp_group_nesting",
                    f"event {event.name!r} closes while DP lifecycle scope "
                    f"{open_scopes[-1][1].name!r} is active",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        begin_index, begin = open_scopes.pop()
        spans[event.name].append(_Span(begin_index, index, begin, event))

    for _index, event in open_scopes:
        failures.append(
            _d2_failure(
                "dp_group_nesting",
                f"event {event.name!r} has no matching end",
                rank=rank,
                iteration=iteration_id,
            )
        )
    return {name: tuple(items) for name, items in spans.items()}, failures


def _validate_distopt_groups(
    by_rank: Mapping[int, tuple[Rank, Sequence[Iteration]]],
    *,
    pipeline_model_parallel_size: int,
    data_parallel_size: int,
    expert_model_parallel_size: int,
    contract_name: str,
) -> list[Failure]:
    """Require reviewed DistOpt model-DP and expert-DP groups."""

    failures: list[Failure] = []
    expected_roles = frozenset(("model-dp", "expert-dp"))
    for rank, (_shard, iterations) in by_rank.items():
        pipeline_rank = rank // data_parallel_size
        if pipeline_rank >= pipeline_model_parallel_size:
            continue
        stage_base = pipeline_rank * data_parallel_size
        stage_rank = rank - stage_base
        expert_rank = stage_rank % expert_model_parallel_size
        expert_data_parallel_size = (
            data_parallel_size // expert_model_parallel_size
        )
        expected_groups = {
            "model-dp": tuple(range(stage_base, stage_base + data_parallel_size)),
            "expert-dp": tuple(
                stage_base + expert_rank + replica * expert_model_parallel_size
                for replica in range(expert_data_parallel_size)
            ),
        }
        observed_roles: dict[str, set[str]] = {name: set() for name in _DP_GROUP_ROUTES}
        dispatch_ends: dict[str, tuple[int, int]] = {}
        duplicate_operation_ids: set[str] = set()
        completions: list[tuple[str, tuple[int, int], int | None, str]] = []
        for iteration_order, iteration in enumerate(iterations):
            iteration_id = (
                None if iteration.iteration_id is None else int(iteration.iteration_id)
            )
            iteration_roles: dict[str, set[str]] = {
                name: set() for name in _DP_GROUP_ROUTES
            }
            spans, pairing_failures = _pair_dp_scopes(iteration, rank=rank)
            failures.extend(pairing_failures)
            for name, expected_op in _DP_GROUP_ROUTES.items():
                for span in spans.get(name, ()):
                    operation_id = span.begin.attrs.get("operation_id")
                    if isinstance(operation_id, str):
                        if operation_id in dispatch_ends:
                            duplicate_operation_ids.add(operation_id)
                        else:
                            dispatch_ends[operation_id] = (
                                iteration_order,
                                span.end_index,
                            )
                    group_size = span.begin.attrs.get("group_size")
                    peers = span.end.attrs.get("group")
                    matched_role = None
                    for role, group in expected_groups.items():
                        expected_peers = [peer for peer in group if peer != rank]
                        if group_size == len(group) and peers == expected_peers:
                            matched_role = role
                            break
                    if matched_role is None:
                        failures.append(
                            _d2_failure(
                                "dp_group",
                                f"event {name!r} has group_size={group_size!r} and "
                                f"peer group={peers!r}; expected model-DP"
                                f"{data_parallel_size} or expert-DP"
                                f"{expert_data_parallel_size} membership",
                                rank=rank,
                                iteration=iteration_id,
                            )
                        )
                        continue
                    observed_roles[name].add(matched_role)
                    iteration_roles[name].add(matched_role)
                    for field_name, expected in (
                        ("group_role", "intra_optimizer_instance"),
                        ("op", expected_op),
                    ):
                        if span.begin.attrs.get(field_name) != expected:
                            failures.append(
                                _d2_failure(
                                    "dp_field",
                                    f"event {name!r} has {field_name}="
                                    f"{span.begin.attrs.get(field_name)!r}, expected "
                                    f"{expected!r}",
                                    rank=rank,
                                    iteration=iteration_id,
                                )
                            )
                    for field_name in ("data_bytes", "n_buckets"):
                        value = span.begin.attrs.get(field_name)
                        if (
                            not isinstance(value, int)
                            or isinstance(value, bool)
                            or value <= 0
                        ):
                            failures.append(
                                _d2_failure(
                                    "dp_field",
                                    f"event {name!r} has invalid "
                                    f"{field_name}={value!r}",
                                    rank=rank,
                                    iteration=iteration_id,
                                )
                            )
            for name in _DP_COMPLETION_EVENTS:
                for span in spans.get(name, ()):
                    operation_ids = span.begin.attrs.get("operation_ids")
                    if not isinstance(operation_ids, list):
                        continue
                    completions.extend(
                        (
                            operation_id,
                            (iteration_order, span.begin_index),
                            iteration_id,
                            span.begin.name,
                        )
                        for operation_id in operation_ids
                        if isinstance(operation_id, str)
                    )
            if iteration_roles["dp-reduce-scatter"] != expected_roles:
                failures.append(
                    _d2_failure(
                        "dp_group_count",
                        "event 'dp-reduce-scatter' must cover model-DP"
                        f"{data_parallel_size} and expert-DP"
                        f"{expert_data_parallel_size} in each iteration",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
        for (
            operation_id,
            completion_position,
            iteration_id,
            completion_name,
        ) in completions:
            dispatch_end = dispatch_ends.get(operation_id)
            if dispatch_end is None or operation_id in duplicate_operation_ids:
                continue
            if completion_position <= dispatch_end:
                failures.append(
                    _d2_failure(
                        "dp_completion_order",
                        f"event {completion_name!r} for operation_id="
                        f"{operation_id!r} begins at {completion_position}, before "
                        f"the dispatch span ends at {dispatch_end}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
        for name, roles in observed_roles.items():
            if name == "dp-reduce-scatter":
                continue
            if roles != expected_roles:
                failures.append(
                    _d2_failure(
                        "dp_group_count",
                        f"event {name!r} must cover model-DP{data_parallel_size} "
                        f"and expert-DP{expert_data_parallel_size} "
                        "across the two-iteration window",
                        rank=rank,
                        iteration=None,
                    )
                )
    if contract_name == "deepseek_d2":
        return failures
    return [
        Failure(
            failure.code.replace(
                "trace.deepseek_d2.", f"trace.{contract_name}.", 1
            ),
            failure.message,
            failure.evidence,
        )
        for failure in failures
    ]


def _validate_d2_dp_groups(
    by_rank: Mapping[int, tuple[Rank, Sequence[Iteration]]],
) -> list[Failure]:
    return _validate_distopt_groups(
        by_rank,
        pipeline_model_parallel_size=_PIPELINE_MODEL_PARALLEL_SIZE,
        data_parallel_size=D2_DATA_PARALLEL_SIZE,
        expert_model_parallel_size=D2_EXPERT_MODEL_PARALLEL_SIZE,
        contract_name="deepseek_d2",
    )


def _validate_etp1_metadata(
    by_rank: Mapping[int, tuple[Rank, Sequence[Iteration]]],
    *,
    pipeline_model_parallel_size: int,
    data_parallel_size: int,
    expert_model_parallel_size: int,
    contract_name: str,
) -> list[Failure]:
    """Require ETP1 metadata gathers and reject dispatcher TP work."""

    failures: list[Failure] = []
    for rank, (_shard, iterations) in by_rank.items():
        pipeline_rank = rank // data_parallel_size
        if pipeline_rank >= pipeline_model_parallel_size:
            continue
        stage_base = pipeline_rank * data_parallel_size
        stage_rank = rank - stage_base
        expert_data_rank = stage_rank // expert_model_parallel_size
        group_base = (
            stage_base + expert_data_rank * expert_model_parallel_size
        )
        metadata_group = tuple(
            range(group_base, group_base + expert_model_parallel_size)
        )
        metadata_peers = [peer for peer in metadata_group if peer != rank]
        for iteration in iterations:
            iteration_id = (
                None if iteration.iteration_id is None else int(iteration.iteration_id)
            )
            events = iteration.events
            starts = [
                index
                for index, event in enumerate(events)
                if event.name == "moe-shared-expert" and event.ph == "B"
            ]
            ends = [
                index
                for index, event in enumerate(events)
                if event.name == "moe-combine" and event.ph == "E"
            ]
            consumed_collectives: set[int] = set()
            for start, end in zip(starts, ends):
                collective_rows = [
                    (index, event)
                    for index, event in enumerate(events[start : end + 1], start)
                    if event.name in _ETP_TP_COLLECTIVES
                ]
                observed_sequence = tuple(
                    (event.name, event.ph) for _index, event in collective_rows
                )
                if observed_sequence != (
                    ("tp-all-gather-first", "B"),
                    ("tp-all-gather-first", "E"),
                ):
                    failures.append(
                        _d2_failure(
                            "etp_collective",
                            "each ETP1 MoE call must contain exactly one TP×EP "
                            "metadata AllGather and no dispatcher TP collective",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
                    continue
                (begin_index, begin), (end_index, end_event) = collective_rows
                consumed_collectives.update((begin_index, end_index))
                router_end = next(
                    (
                        index
                        for index in range(start, end + 1)
                        if events[index].name == "moe-router"
                        and events[index].ph == "E"
                    ),
                    None,
                )
                dispatch_begin = next(
                    (
                        index
                        for index in range(start, end + 1)
                        if events[index].name == "moe-dispatch"
                        and events[index].ph == "B"
                    ),
                    None,
                )
                if (
                    router_end is None
                    or dispatch_begin is None
                    or not (router_end < begin_index < end_index < dispatch_begin)
                ):
                    failures.append(
                        _d2_failure(
                            "metadata_collective",
                            "TP×EP metadata AllGather must remain between Router "
                            "and Dispatch",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
                for field, expected in (
                    ("op", "all-gather"),
                    ("dim", "first"),
                    ("group_size", expert_model_parallel_size),
                ):
                    if begin.attrs.get(field) != expected:
                        failures.append(
                            _d2_failure(
                                "metadata_collective",
                                f"metadata AllGather field {field!r}="
                                f"{begin.attrs.get(field)!r}, expected {expected!r}",
                                rank=rank,
                                iteration=iteration_id,
                            )
                        )
                data_bytes = begin.attrs.get("data_bytes")
                if (
                    not isinstance(data_bytes, int)
                    or isinstance(data_bytes, bool)
                    or data_bytes <= 0
                ):
                    failures.append(
                        _d2_failure(
                            "metadata_collective",
                            f"metadata AllGather has invalid data_bytes={data_bytes!r}",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
                if "split_sizes" in begin.attrs:
                    failures.append(
                        _d2_failure(
                            "etp_collective",
                            "ETP1 metadata AllGather must not carry dispatcher "
                            "split_sizes",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
                if end_event.attrs.get("group") != metadata_peers:
                    failures.append(
                        _d2_failure(
                            "metadata_collective",
                            f"metadata AllGather peer group="
                            f"{end_event.attrs.get('group')!r}, expected "
                            f"{metadata_peers!r}",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
            unconsumed = sorted(
                {
                    event.name
                    for index, event in enumerate(events)
                    if event.name in _ETP_TP_COLLECTIVES
                    and index not in consumed_collectives
                }
            )
            if unconsumed:
                failures.append(
                    _d2_failure(
                        "etp_collective",
                        f"ETP1 trace contains unexpected TP collectives {unconsumed!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
    if contract_name == "deepseek_d2":
        return failures
    return [
        Failure(
            failure.code.replace(
                "trace.deepseek_d2.", f"trace.{contract_name}.", 1
            ),
            failure.message,
            failure.evidence,
        )
        for failure in failures
    ]


def _validate_d2_etp1(
    by_rank: Mapping[int, tuple[Rank, Sequence[Iteration]]],
) -> list[Failure]:
    return _validate_etp1_metadata(
        by_rank,
        pipeline_model_parallel_size=_PIPELINE_MODEL_PARALLEL_SIZE,
        data_parallel_size=D2_DATA_PARALLEL_SIZE,
        expert_model_parallel_size=D2_EXPERT_MODEL_PARALLEL_SIZE,
        contract_name="deepseek_d2",
    )


def _validate_deepseek_trace(
    trace_root: Path,
    *,
    contract_name: str,
    pipeline_model_parallel_size: int,
    microbatches_per_iteration: int,
    data_parallel_size: int,
    expert_model_parallel_size: int,
    main_layers: Mapping[int, Sequence[int]],
    mtp_layers: Mapping[int, Sequence[int]],
) -> tuple[Failure, ...]:
    if (
        pipeline_model_parallel_size,
        data_parallel_size,
        expert_model_parallel_size,
    ) not in _REVIEWED_TOPOLOGIES:
        raise ValueError(
            "pipeline/data/expert parallel sizes do not match a reviewed "
            "D0/D2/D3 topology"
        )
    by_rank = _load_iterations(trace_root)
    failures: list[Failure] = []
    expected_ranks = tuple(range(pipeline_model_parallel_size * data_parallel_size))
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != expected_ranks:
        failures.append(
            Failure(
                f"trace.{contract_name}.ranks",
                f"expected ranks {expected_ranks}, observed {observed_ranks}",
                contract_name.replace("_", "-"),
            )
        )

    for global_rank in expected_ranks:
        loaded = by_rank.get(global_rank)
        if loaded is None:
            continue
        shard_rank, iterations = loaded
        # D0/D2 rely on Megatron's target-side default tp-cp-ep-dp-pp rank order.
        # With TP=CP=1, PP is the slowest-varying axis and each PP stage owns
        # data_parallel_size consecutive data-parallel ranks.
        pipeline_rank = global_rank // data_parallel_size
        data_rank = global_rank % data_parallel_size
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
                    pipeline_model_parallel_size=pipeline_model_parallel_size,
                    microbatches_per_iteration=microbatches_per_iteration,
                    expert_model_parallel_size=expert_model_parallel_size,
                    main_layers=main_layers,
                    mtp_layers=mtp_layers,
                )
            )
    failures.extend(
        _validate_ep_workload_conservation(
            by_rank,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            microbatches_per_iteration=microbatches_per_iteration,
            data_parallel_size=data_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            main_layers=main_layers,
            mtp_layers=mtp_layers,
        )
    )
    if contract_name != "deepseek_d0":
        failures = [
            Failure(
                failure.code.replace(
                    "trace.deepseek_d0.", f"trace.{contract_name}.", 1
                ),
                failure.message,
                failure.evidence,
            )
            for failure in failures
        ]
    return tuple(failures)


def validate_deepseek_d0_trace(
    trace_root: Path,
    *,
    microbatches_per_iteration: int = DEFAULT_MICROBATCHES_PER_ITERATION,
    data_parallel_size: int = DEFAULT_DATA_PARALLEL_SIZE,
) -> tuple[Failure, ...]:
    """Validate the reviewed D0 PP2/EP4 MoE, shared-expert, and MTP trace."""

    if microbatches_per_iteration < 1:
        raise ValueError("microbatches_per_iteration must be positive")
    if data_parallel_size not in _SUPPORTED_DATA_PARALLEL_SIZES:
        raise ValueError("data_parallel_size must be 4 or 8 for a reviewed D0 profile")
    return _validate_deepseek_trace(
        trace_root,
        contract_name="deepseek_d0",
        pipeline_model_parallel_size=_PIPELINE_MODEL_PARALLEL_SIZE,
        microbatches_per_iteration=microbatches_per_iteration,
        data_parallel_size=data_parallel_size,
        expert_model_parallel_size=_D0_EXPERT_MODEL_PARALLEL_SIZE,
        main_layers=_MAIN_LAYERS,
        mtp_layers=_MTP_LAYERS,
    )


def validate_deepseek_d2_trace(
    trace_root: Path,
    *,
    microbatches_per_iteration: int = DEFAULT_MICROBATCHES_PER_ITERATION,
) -> tuple[Failure, ...]:
    """Validate D2 PP2/DP8/EP8/ETP1/expert-DP1 and DistOpt lifecycle."""

    if microbatches_per_iteration < 1:
        raise ValueError("microbatches_per_iteration must be positive")
    by_rank = _load_iterations(trace_root)
    return (
        *_validate_deepseek_trace(
            trace_root,
            contract_name="deepseek_d2",
            pipeline_model_parallel_size=_PIPELINE_MODEL_PARALLEL_SIZE,
            microbatches_per_iteration=microbatches_per_iteration,
            data_parallel_size=D2_DATA_PARALLEL_SIZE,
            expert_model_parallel_size=D2_EXPERT_MODEL_PARALLEL_SIZE,
            main_layers=_MAIN_LAYERS,
            mtp_layers=_MTP_LAYERS,
        ),
        *_validate_d2_etp1(by_rank),
        *_validate_d2_dp_groups(by_rank),
        *dp_probe_contract.validate_dp_distopt_overlap(trace_root),
    )


def validate_deepseek_d3_trace(
    trace_root: Path,
    *,
    microbatches_per_iteration: int = 1,
) -> tuple[Failure, ...]:
    """Validate D3 PP1/DP8/EP4/ETP1/expert-DP2 and DistOpt lifecycle."""

    if microbatches_per_iteration < 1:
        raise ValueError("microbatches_per_iteration must be positive")
    by_rank = _load_iterations(trace_root)
    return (
        *_validate_deepseek_trace(
            trace_root,
            contract_name="deepseek_d3",
            pipeline_model_parallel_size=_D3_PIPELINE_MODEL_PARALLEL_SIZE,
            microbatches_per_iteration=microbatches_per_iteration,
            data_parallel_size=D3_DATA_PARALLEL_SIZE,
            expert_model_parallel_size=D3_EXPERT_MODEL_PARALLEL_SIZE,
            main_layers=_D3_MAIN_LAYERS,
            mtp_layers=_D3_MTP_LAYERS,
        ),
        *_validate_etp1_metadata(
            by_rank,
            pipeline_model_parallel_size=_D3_PIPELINE_MODEL_PARALLEL_SIZE,
            data_parallel_size=D3_DATA_PARALLEL_SIZE,
            expert_model_parallel_size=D3_EXPERT_MODEL_PARALLEL_SIZE,
            contract_name="deepseek_d3",
        ),
        *_validate_distopt_groups(
            by_rank,
            pipeline_model_parallel_size=_D3_PIPELINE_MODEL_PARALLEL_SIZE,
            data_parallel_size=D3_DATA_PARALLEL_SIZE,
            expert_model_parallel_size=D3_EXPERT_MODEL_PARALLEL_SIZE,
            contract_name="deepseek_d3",
        ),
        *dp_probe_contract.validate_dp_distopt_overlap(trace_root),
    )


__all__ = [
    "D0_RANK_ORDER",
    "D2_DATA_PARALLEL_SIZE",
    "D2_EXPERT_MODEL_PARALLEL_SIZE",
    "D3_DATA_PARALLEL_SIZE",
    "D3_EXPERT_MODEL_PARALLEL_SIZE",
    "DEFAULT_DATA_PARALLEL_SIZE",
    "DEFAULT_MICROBATCHES_PER_ITERATION",
    "validate_deepseek_d0_trace",
    "validate_deepseek_d2_trace",
    "validate_deepseek_d3_trace",
]
