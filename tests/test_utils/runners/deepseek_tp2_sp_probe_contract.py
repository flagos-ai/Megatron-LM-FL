# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact Trace contract for the single-node DeepSeek TP2/SP L3 profile."""

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
from tests.test_utils.runners import tp_probe_contract
from tests.test_utils.runners.megalens_run_manifest import Failure

RANK_ORDER = "tp-cp-ep-dp-pp"
WORLD_SIZE = 8
TENSOR_MODEL_PARALLEL_SIZE = 2
PIPELINE_MODEL_PARALLEL_SIZE = 2
MODEL_DATA_PARALLEL_SIZE = 2
EXPERT_MODEL_PARALLEL_SIZE = 4
EXPERT_TENSOR_PARALLEL_SIZE = 1
EXPERT_DATA_PARALLEL_SIZE = 1
MICROBATCHES_PER_ITERATION = 2

_ITERATIONS = (1, 2)
_STAGE_SIZE = TENSOR_MODEL_PARALLEL_SIZE * MODEL_DATA_PARALLEL_SIZE
_TOKENS_PER_ROUTER = 4096 // TENSOR_MODEL_PARALLEL_SIZE
_ROUTED_TOKENS_PER_ROUTER = _TOKENS_PER_ROUTER * 6
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
_MAIN_LAYERS = {0: tuple(range(2, 14)), 1: tuple(range(14, 28))}
_MTP_LAYERS = {0: (), 1: (1,)}
_ROUTER_HANDOFF_FIELDS = (
    "dropped_tokens",
    "drop_rate",
    "expert_cv",
    "top1_expert_share",
    "aux_loss",
    "z_loss",
)
_TP_COLLECTIVES = {
    "tp-all-gather-first": {"op": "all-gather", "dim": "first"},
    "tp-all-gather-last": {"op": "all-gather", "dim": "last"},
    "tp-reduce-scatter": {"op": "reduce-scatter", "dim": "first"},
    "tp-reduce-scatter-last": {"op": "reduce-scatter", "dim": "last"},
}
_REQUIRED_MODEL_TP_COLLECTIVES = frozenset(
    ("tp-all-gather-first", "tp-reduce-scatter")
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
    return Failure(f"trace.deepseek_tp2_sp.{code}", message, evidence)


def _load_iterations(
    trace_root: Path,
) -> Mapping[int, tuple[Rank, Sequence[Iteration]]]:
    result: dict[int, tuple[Rank, Sequence[Iteration]]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None or rank.global_rank in result:
            raise ValueError("DeepSeek TP2/SP requires one shard per global rank")
        result[rank.global_rank] = (rank, tuple(read_benchmark_file(rank, content)))
    return result


def _pair_scopes(
    iteration: Iteration,
    names: Iterable[str],
    *,
    rank: int,
    code: str,
) -> tuple[Mapping[str, Sequence[_Span]], list[Failure]]:
    selected = frozenset(names)
    pending: list[tuple[int, Event]] = []
    spans: dict[str, list[_Span]] = defaultdict(list)
    failures: list[Failure] = []
    iteration_id = int(iteration.iteration_id)
    for index, event in enumerate(iteration.events):
        if event.name not in selected:
            continue
        if event.ph == "B":
            pending.append((index, event))
            continue
        if event.ph != "E":
            failures.append(
                _failure(
                    code,
                    f"event {event.name!r} uses phase {event.ph!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        if not pending or pending[-1][1].name != event.name:
            active = pending[-1][1].name if pending else None
            failures.append(
                _failure(
                    code,
                    f"event {event.name!r} closes while {active!r} is active",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        begin_index, begin = pending.pop()
        spans[event.name].append(_Span(begin_index, index, begin, event))
    for _index, event in pending:
        failures.append(
            _failure(
                code,
                f"event {event.name!r} has no matching end",
                rank=rank,
                iteration=iteration_id,
            )
        )
    return {name: tuple(items) for name, items in spans.items()}, failures


def _inside(child: _Span, parent: _Span) -> bool:
    return parent.begin_index < child.begin_index and child.end_index < parent.end_index


def _contained(spans: Iterable[_Span], parent: _Span) -> list[_Span]:
    return sorted(
        (span for span in spans if _inside(span, parent)),
        key=lambda span: span.begin_index,
    )


def _field(
    failures: list[Failure],
    event: Event,
    name: str,
    expected: object,
    *,
    rank: int,
    iteration: int,
    microbatch: int,
    region: str,
    layer: int,
) -> None:
    observed = event.attrs.get(name)
    if observed != expected:
        failures.append(
            _failure(
                "field",
                f"layer={layer} {event.name} field {name!r}={observed!r}, "
                f"expected {expected!r}",
                rank=rank,
                iteration=iteration,
                microbatch=microbatch,
                region=region,
            )
        )


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
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
        _field(
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
        _field(
            failures,
            event,
            "ep_size",
            EXPERT_MODEL_PARALLEL_SIZE,
            rank=rank,
            iteration=iteration,
            microbatch=microbatch,
            region=region,
            layer=layer,
        )
    for event in (router, dispatch, experts, combine):
        for name, expected in (
            ("num_experts", 64),
            ("num_local_experts", 16),
        ):
            _field(
                failures,
                event,
                name,
                expected,
                rank=rank,
                iteration=iteration,
                microbatch=microbatch,
                region=region,
                layer=layer,
            )

    for event in (router, dispatch):
        _field(
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
    _field(
        failures,
        router,
        "num_tokens",
        _TOKENS_PER_ROUTER,
        rank=rank,
        iteration=iteration,
        microbatch=microbatch,
        region=region,
        layer=layer,
    )
    _field(
        failures,
        router,
        "routed_tokens",
        _ROUTED_TOKENS_PER_ROUTER,
        rank=rank,
        iteration=iteration,
        microbatch=microbatch,
        region=region,
        layer=layer,
    )
    _field(
        failures,
        dispatch,
        "num_tokens",
        _ROUTED_TOKENS_PER_ROUTER,
        rank=rank,
        iteration=iteration,
        microbatch=microbatch,
        region=region,
        layer=layer,
    )
    for event in (dispatch, combine):
        _field(
            failures,
            event,
            "dispatcher",
            "alltoall",
            rank=rank,
            iteration=iteration,
            microbatch=microbatch,
            region=region,
            layer=layer,
        )
    _field(
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
    for name in _ROUTER_HANDOFF_FIELDS:
        if router.attrs.get(name) != dispatch.attrs.get(name):
            failures.append(
                _failure(
                    "router_handoff",
                    f"layer={layer} field {name!r} differs between Router and "
                    f"Dispatch: {router.attrs.get(name)!r} != "
                    f"{dispatch.attrs.get(name)!r}",
                    rank=rank,
                    iteration=iteration,
                    microbatch=microbatch,
                    region=region,
                )
            )
    for name, expected in (("dropped_tokens", 0), ("drop_rate", 0.0), ("z_loss", None)):
        _field(
            failures,
            router,
            name,
            expected,
            rank=rank,
            iteration=iteration,
            microbatch=microbatch,
            region=region,
            layer=layer,
        )
    for name in ("expert_cv", "top1_expert_share", "routing_entropy", "aux_loss"):
        if not _finite(router.attrs.get(name)):
            failures.append(
                _failure(
                    "router_metric",
                    f"layer={layer} Router field {name!r}="
                    f"{router.attrs.get(name)!r}, expected a finite number",
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
    valid_routed = (
        isinstance(routed_tokens, int)
        and not isinstance(routed_tokens, bool)
        and routed_tokens >= 0
    )
    expert_total = sum(tokens_per_expert) if valid_counts else None
    if not valid_counts or not valid_routed or expert_total != routed_tokens:
        failures.append(
            _failure(
                "expert_workload",
                f"layer={layer} tokens_per_expert must contain 16 non-negative "
                "counts whose sum equals routed_tokens",
                rank=rank,
                iteration=iteration,
                microbatch=microbatch,
                region=region,
            )
        )
    combine_tokens = combine.attrs.get("num_tokens")
    if (
        not isinstance(combine_tokens, int)
        or isinstance(combine_tokens, bool)
        or combine_tokens < 0
        or not valid_routed
        or combine_tokens != routed_tokens
        or combine_tokens != expert_total
    ):
        failures.append(
            _failure(
                "combine_workload",
                f"layer={layer} Combine num_tokens={combine_tokens!r} must equal "
                f"Experts routed_tokens={routed_tokens!r} and local count sum="
                f"{expert_total!r}",
                rank=rank,
                iteration=iteration,
                microbatch=microbatch,
                region=region,
            )
        )

    for event in (experts,):
        for name in ("expert_cv", "top1_expert_share", "expert_max_over_mean"):
            if not _finite(event.attrs.get(name)):
                failures.append(
                    _failure(
                        "expert_metric",
                        f"layer={layer} Experts field {name!r}="
                        f"{event.attrs.get(name)!r}, expected a finite number",
                        rank=rank,
                        iteration=iteration,
                        microbatch=microbatch,
                        region=region,
                    )
                )

    for name in ("ep-alltoall-dispatch", "ep-alltoall-combine"):
        collective = end_events[name]
        for field_name, expected in (
            ("comm_type", "ep-alltoall"),
            ("dispatcher", "alltoall"),
            ("group_size", EXPERT_MODEL_PARALLEL_SIZE),
            ("ep_size", EXPERT_MODEL_PARALLEL_SIZE),
            ("tp_size", EXPERT_TENSOR_PARALLEL_SIZE),
        ):
            _field(
                failures,
                collective,
                field_name,
                expected,
                rank=rank,
                iteration=iteration,
                microbatch=microbatch,
                region=region,
                layer=layer,
            )
        data_bytes = collective.attrs.get("data_bytes")
        valid_bytes = isinstance(data_bytes, int) and not isinstance(data_bytes, bool)
        if name == "ep-alltoall-combine" and routed_tokens == 0:
            valid_bytes = valid_bytes and data_bytes == 0
        else:
            valid_bytes = valid_bytes and data_bytes > 0
        if not valid_bytes:
            failures.append(
                _failure(
                    "collective_bytes",
                    f"layer={layer} {name} has invalid data_bytes={data_bytes!r}",
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
    records = [
        event
        for event in iteration.events[span.begin_index + 1 : span.end_index]
        if event.name in _MOE_NAMES
    ]
    expected_sequence = _MOE_CALL_SEQUENCE * len(expected_layers)
    observed_sequence = tuple((event.name, event.ph) for event in records)
    iteration_id = int(iteration.iteration_id)
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


def _validate_model_iteration(
    iteration: Iteration,
    *,
    rank: int,
    pipeline_rank: int,
) -> list[Failure]:
    iteration_id = int(iteration.iteration_id)
    spans, failures = _pair_scopes(
        iteration,
        _MODEL_SCOPE_NAMES,
        rank=rank,
        code="model_nesting",
    )
    counts = Counter(
        (event.name, event.ph)
        for event in iteration.events
        if event.name in _MODEL_SCOPE_NAMES
    )
    expected_counts = {
        "forward-step": MICROBATCHES_PER_ITERATION,
        "decoder": MICROBATCHES_PER_ITERATION,
        "decoder-postprocess": MICROBATCHES_PER_ITERATION,
        "output_layer": MICROBATCHES_PER_ITERATION if pipeline_rank == 1 else 0,
        "loss": MICROBATCHES_PER_ITERATION if pipeline_rank == 1 else 0,
    }
    for name, expected in expected_counts.items():
        observed = (counts[(name, "B")], counts[(name, "E")], len(spans.get(name, ())))
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

    forwards = sorted(spans.get("forward-step", ()), key=lambda span: span.begin_index)
    for microbatch, forward in enumerate(forwards):
        decoder = _contained(spans.get("decoder", ()), forward)
        postprocess = _contained(spans.get("decoder-postprocess", ()), forward)
        if len(decoder) != 1 or len(postprocess) != 1:
            failures.append(
                _failure(
                    "forward_tree",
                    f"forward-step contains {len(decoder)} decoder and "
                    f"{len(postprocess)} decoder-postprocess scopes",
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
        output = _contained(spans.get("output_layer", ()), postprocess[0])
        loss = _contained(spans.get("loss", ()), postprocess[0])
        expected = 1 if pipeline_rank == 1 else 0
        if len(output) != expected or len(loss) != expected:
            failures.append(
                _failure(
                    "postprocess_tree",
                    f"decoder-postprocess contains {len(output)} output_layer and "
                    f"{len(loss)} loss scopes, expected {expected} each",
                    rank=rank,
                    iteration=iteration_id,
                    microbatch=microbatch,
                    region="decoder-postprocess",
                )
            )
    return failures


def _validate_tp_iteration(iteration: Iteration, *, rank: int) -> list[Failure]:
    iteration_id = int(iteration.iteration_id)
    spans, failures = _pair_scopes(
        iteration,
        _TP_COLLECTIVES,
        rank=rank,
        code="tp_collective_nesting",
    )
    peer = rank ^ 1
    pipeline_rank = rank // _STAGE_SIZE
    required_model_tp_collectives = _REQUIRED_MODEL_TP_COLLECTIVES
    if pipeline_rank == PIPELINE_MODEL_PARALLEL_SIZE - 1:
        required_model_tp_collectives |= {"tp-all-gather-last"}
    ep_group = range(
        pipeline_rank * _STAGE_SIZE,
        (pipeline_rank + 1) * _STAGE_SIZE,
    )
    ep_peers = [ep_rank for ep_rank in ep_group if ep_rank != rank]
    model_tp_observed: set[str] = set()
    ep_metadata_gather_observed = False
    for name, expected_fields in _TP_COLLECTIVES.items():
        for span in spans.get(name, ()):
            for field_name, expected in expected_fields.items():
                if span.begin.attrs.get(field_name) != expected:
                    failures.append(
                        _failure(
                            "tp_collective_field",
                            f"event {name!r} has {field_name}="
                            f"{span.begin.attrs.get(field_name)!r}, expected {expected!r}",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
            data_bytes = span.begin.attrs.get("data_bytes")
            if (
                not isinstance(data_bytes, int)
                or isinstance(data_bytes, bool)
                or data_bytes <= 0
            ):
                failures.append(
                    _failure(
                        "tp_collective_field",
                        f"event {name!r} has invalid data_bytes={data_bytes!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
            group_size = span.begin.attrs.get("group_size")
            group = span.end.attrs.get("group")
            if group_size == TENSOR_MODEL_PARALLEL_SIZE and group == [peer]:
                model_tp_observed.add(name)
            elif (
                name == "tp-all-gather-first"
                and group_size == EXPERT_MODEL_PARALLEL_SIZE
                and group == ep_peers
            ):
                ep_metadata_gather_observed = True
            else:
                failures.append(
                    _failure(
                        "tp_collective_group",
                        f"event {name!r} has group_size={group_size!r} and peer "
                        f"group={group!r}; expected model TP2 {[peer]!r} or the "
                        f"EP4 metadata gather {ep_peers!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
    for required in sorted(required_model_tp_collectives - model_tp_observed):
        failures.append(
            _failure(
                "tp_collective_count",
                f"event {required!r} has no complete model TP2 span",
                rank=rank,
                iteration=iteration_id,
            )
        )
    if spans.get("tp-reduce-scatter-last"):
        failures.append(
            _failure(
                "tp_collective_count",
                "tp-reduce-scatter-last must be absent from the MLA/MTP profile",
                rank=rank,
                iteration=iteration_id,
            )
        )
    if not ep_metadata_gather_observed:
        failures.append(
            _failure(
                "tp_collective_count",
                "tp-all-gather-first has no complete EP4 metadata span",
                rank=rank,
                iteration=iteration_id,
            )
        )
    linear_failures, _operation_ids = tp_probe_contract._validate_linear_lifecycle(
        iteration,
        rank=rank,
        expected_routes=tp_probe_contract._SP_LINEAR_ROUTES,
        tensor_parallel_size=TENSOR_MODEL_PARALLEL_SIZE,
    )
    failures.extend(linear_failures)
    sync_failures, _sp_bytes, _embedding_bytes = (
        tp_probe_contract._validate_final_grad_sync(
            iteration,
            rank=rank,
            schedule="non-interleaved-1f1b",
            expect_sp_layernorm=True,
            tp_peers=(peer,),
            embedding_peer=None,
        )
    )
    failures.extend(sync_failures)
    return failures


def _collect_workloads(
    iteration: Iteration,
    *,
    rank: int,
) -> Mapping[tuple[int, int], tuple[int, int]]:
    spans, failures = _pair_scopes(
        iteration,
        _MODEL_SCOPE_NAMES,
        rank=rank,
        code="model_nesting",
    )
    if failures:
        return {}
    workloads: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    duplicates: set[tuple[int, int]] = set()
    forwards = sorted(spans.get("forward-step", ()), key=lambda span: span.begin_index)
    for microbatch, forward in enumerate(forwards):
        for event in iteration.events[forward.begin_index + 1 : forward.end_index]:
            if event.ph != "E" or event.name not in ("moe-router", "moe-experts"):
                continue
            layer = event.attrs.get("layer")
            value = event.attrs.get("routed_tokens")
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
                duplicates.add(key)
            else:
                workloads[key][event.name] = value
    return {
        key: (values["moe-router"], values["moe-experts"])
        for key, values in workloads.items()
        if key not in duplicates
        and "moe-router" in values
        and "moe-experts" in values
    }


def _validate_ep_conservation(
    by_rank: Mapping[int, tuple[Rank, Sequence[Iteration]]],
) -> list[Failure]:
    workloads: dict[tuple[int, int], Mapping[tuple[int, int], tuple[int, int]]] = {}
    for rank, (_shard, iterations) in by_rank.items():
        for iteration in iterations:
            workloads[(rank, int(iteration.iteration_id))] = _collect_workloads(
                iteration,
                rank=rank,
            )
    failures: list[Failure] = []
    for pipeline_rank in range(PIPELINE_MODEL_PARALLEL_SIZE):
        group = tuple(
            range(pipeline_rank * _STAGE_SIZE, (pipeline_rank + 1) * _STAGE_SIZE)
        )
        expected_layers = _MAIN_LAYERS[pipeline_rank] + _MTP_LAYERS[pipeline_rank]
        for iteration in _ITERATIONS:
            for microbatch in range(MICROBATCHES_PER_ITERATION):
                for layer in expected_layers:
                    values = [
                        workloads.get((rank, iteration), {}).get((microbatch, layer))
                        for rank in group
                    ]
                    if any(value is None for value in values):
                        continue
                    router_total = sum(value[0] for value in values if value is not None)
                    experts_total = sum(value[1] for value in values if value is not None)
                    if router_total != experts_total:
                        failures.append(
                            _failure(
                                "ep_conservation",
                                f"EP group {group} layer={layer} Router total="
                                f"{router_total}, Experts total={experts_total}",
                                rank=group[0],
                                iteration=iteration,
                                microbatch=microbatch,
                            )
                        )
    return failures


def validate_deepseek_tp2_sp_trace(trace_root: Path) -> tuple[Failure, ...]:
    """Validate the fixed TP2/PP2/DP2/EP4/ETP1 DeepSeek L3 trace."""

    by_rank = _load_iterations(trace_root)
    failures: list[Failure] = []
    expected_ranks = tuple(range(WORLD_SIZE))
    if tuple(sorted(by_rank)) != expected_ranks:
        failures.append(
            Failure(
                "trace.deepseek_tp2_sp.ranks",
                f"expected ranks {expected_ranks}, observed {tuple(sorted(by_rank))}",
                "deepseek-tp2-sp-mock",
            )
        )
    for global_rank in expected_ranks:
        loaded = by_rank.get(global_rank)
        if loaded is None:
            continue
        shard_rank, iterations = loaded
        pipeline_rank = global_rank // _STAGE_SIZE
        stage_rank = global_rank % _STAGE_SIZE
        tensor_rank = stage_rank % TENSOR_MODEL_PARALLEL_SIZE
        data_rank = stage_rank // TENSOR_MODEL_PARALLEL_SIZE
        observed_coordinates = (
            shard_rank.data,
            shard_rank.pipeline,
            shard_rank.tensor,
        )
        expected_coordinates = (data_rank, pipeline_rank, tensor_rank)
        if observed_coordinates != expected_coordinates:
            failures.append(
                _failure(
                    "coordinates",
                    f"shard coordinates={observed_coordinates}, expected "
                    f"{expected_coordinates}",
                    rank=global_rank,
                    iteration=None,
                )
            )
        iteration_ids = tuple(iteration.iteration_id for iteration in iterations)
        if iteration_ids != _ITERATIONS:
            failures.append(
                _failure(
                    "iterations",
                    f"expected iteration IDs {_ITERATIONS}, observed {iteration_ids}",
                    rank=global_rank,
                    iteration=None,
                )
            )
        for iteration in iterations:
            failures.extend(
                _validate_model_iteration(
                    iteration,
                    rank=global_rank,
                    pipeline_rank=pipeline_rank,
                )
            )
            failures.extend(_validate_tp_iteration(iteration, rank=global_rank))
    failures.extend(_validate_ep_conservation(by_rank))
    return tuple(failures)


__all__ = [
    "EXPERT_DATA_PARALLEL_SIZE",
    "EXPERT_MODEL_PARALLEL_SIZE",
    "EXPERT_TENSOR_PARALLEL_SIZE",
    "MICROBATCHES_PER_ITERATION",
    "MODEL_DATA_PARALLEL_SIZE",
    "PIPELINE_MODEL_PARALLEL_SIZE",
    "RANK_ORDER",
    "TENSOR_MODEL_PARALLEL_SIZE",
    "WORLD_SIZE",
    "validate_deepseek_tp2_sp_trace",
]
