# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact Trace contracts for the reviewed DeepSeek TP2/SP L3 profiles."""

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
from tests.test_utils.runners import tp_probe_contract
from tests.test_utils.runners.megalens_run_manifest import Failure

RANK_ORDER = "tp-cp-ep-dp-pp"
TENSOR_MODEL_PARALLEL_SIZE = 2
PIPELINE_MODEL_PARALLEL_SIZE = 2
EXPERT_MODEL_PARALLEL_SIZE = 4
EXPERT_TENSOR_PARALLEL_SIZE = 1


@dataclass(frozen=True)
class _DeepSeekTopology:
    profile_name: str
    world_size: int
    model_data_parallel_size: int
    expert_tensor_parallel_size: int
    expert_data_parallel_size: int
    microbatches_per_iteration: int

    def __post_init__(self) -> None:
        stage_size = self.world_size // PIPELINE_MODEL_PARALLEL_SIZE
        if stage_size != TENSOR_MODEL_PARALLEL_SIZE * self.model_data_parallel_size:
            raise ValueError("world size does not match TP2/PP2/model-DP topology")
        if self.expert_tensor_parallel_size < 1:
            raise ValueError("expert tensor parallel size must be positive")
        if stage_size != (
            self.expert_tensor_parallel_size
            * EXPERT_MODEL_PARALLEL_SIZE
            * self.expert_data_parallel_size
        ):
            raise ValueError("world size does not match PP2/EP4/ETP/expert-DP topology")
        if self.microbatches_per_iteration < 1:
            raise ValueError("microbatches per iteration must be positive")

    @property
    def stage_size(self) -> int:
        return TENSOR_MODEL_PARALLEL_SIZE * self.model_data_parallel_size

    def pipeline_rank(self, rank: int) -> int:
        return rank // self.stage_size

    @property
    def tp_ep_size(self) -> int:
        return self.expert_tensor_parallel_size * EXPERT_MODEL_PARALLEL_SIZE

    def tp_ep_group(self, rank: int) -> tuple[int, ...]:
        stage_base = self.pipeline_rank(rank) * self.stage_size
        stage_rank = rank - stage_base
        replica = stage_rank // self.tp_ep_size
        group_base = stage_base + replica * self.tp_ep_size
        return tuple(range(group_base, group_base + self.tp_ep_size))

    def expert_tensor_parallel_group(self, rank: int) -> tuple[int, ...]:
        group = self.tp_ep_group(rank)
        group_rank = rank - group[0]
        group_base = group[0] + (
            group_rank // self.expert_tensor_parallel_size
        ) * self.expert_tensor_parallel_size
        return tuple(
            range(group_base, group_base + self.expert_tensor_parallel_size)
        )

    def expert_model_parallel_group(self, rank: int) -> tuple[int, ...]:
        group = self.tp_ep_group(rank)
        expert_tensor_rank = (rank - group[0]) % self.expert_tensor_parallel_size
        return tuple(
            group[0]
            + expert_tensor_rank
            + expert_rank * self.expert_tensor_parallel_size
            for expert_rank in range(EXPERT_MODEL_PARALLEL_SIZE)
        )

    def model_data_parallel_group(self, rank: int) -> tuple[int, ...]:
        stage_base = self.pipeline_rank(rank) * self.stage_size
        tensor_rank = (rank - stage_base) % TENSOR_MODEL_PARALLEL_SIZE
        return tuple(
            stage_base + tensor_rank + data_rank * TENSOR_MODEL_PARALLEL_SIZE
            for data_rank in range(self.model_data_parallel_size)
        )

    def expert_data_parallel_group(self, rank: int) -> tuple[int, ...]:
        stage_base = self.pipeline_rank(rank) * self.stage_size
        expert_parallel_rank = (rank - stage_base) % self.tp_ep_size
        return tuple(
            stage_base + expert_parallel_rank + data_rank * self.tp_ep_size
            for data_rank in range(self.expert_data_parallel_size)
        )

    def pipeline_peer(self, rank: int) -> int:
        if self.pipeline_rank(rank) == 0:
            return rank + self.stage_size
        return rank - self.stage_size


_SINGLE_NODE_TOPOLOGY = _DeepSeekTopology(
    profile_name="deepseek-tp2-sp-mock",
    world_size=8,
    model_data_parallel_size=2,
    expert_tensor_parallel_size=1,
    expert_data_parallel_size=1,
    microbatches_per_iteration=2,
)
_D1_ETP1_TOPOLOGY = _DeepSeekTopology(
    profile_name="deepseek-d1-tp2-sp-etp1-mock",
    world_size=16,
    model_data_parallel_size=4,
    expert_tensor_parallel_size=1,
    expert_data_parallel_size=2,
    microbatches_per_iteration=1,
)
_D1_ETP2_TOPOLOGY = _DeepSeekTopology(
    profile_name="deepseek-d1-tp2-sp-etp2-mock",
    world_size=16,
    model_data_parallel_size=4,
    expert_tensor_parallel_size=2,
    expert_data_parallel_size=1,
    microbatches_per_iteration=1,
)

WORLD_SIZE = _SINGLE_NODE_TOPOLOGY.world_size
MODEL_DATA_PARALLEL_SIZE = _SINGLE_NODE_TOPOLOGY.model_data_parallel_size
EXPERT_DATA_PARALLEL_SIZE = _SINGLE_NODE_TOPOLOGY.expert_data_parallel_size
MICROBATCHES_PER_ITERATION = _SINGLE_NODE_TOPOLOGY.microbatches_per_iteration
D1_WORLD_SIZE = _D1_ETP1_TOPOLOGY.world_size
D1_MODEL_DATA_PARALLEL_SIZE = _D1_ETP1_TOPOLOGY.model_data_parallel_size
D1_EXPERT_DATA_PARALLEL_SIZE = _D1_ETP1_TOPOLOGY.expert_data_parallel_size
D1_MICROBATCHES_PER_ITERATION = _D1_ETP1_TOPOLOGY.microbatches_per_iteration
D1_ETP2_WORLD_SIZE = _D1_ETP2_TOPOLOGY.world_size
D1_ETP2_MODEL_DATA_PARALLEL_SIZE = _D1_ETP2_TOPOLOGY.model_data_parallel_size
D1_ETP2_EXPERT_DATA_PARALLEL_SIZE = (
    _D1_ETP2_TOPOLOGY.expert_data_parallel_size
)
D1_ETP2_MICROBATCHES_PER_ITERATION = (
    _D1_ETP2_TOPOLOGY.microbatches_per_iteration
)

_ITERATIONS = (1, 2)
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
_DP_GROUP_ROUTES = {
    "dp-reduce-scatter": "reduce_scatter",
    "dp-param-all-gather": "all_gather",
}


@dataclass(frozen=True)
class _Span:
    begin_index: int
    end_index: int
    begin: Event
    end: Event


@dataclass(frozen=True)
class _Workload:
    router_routed_tokens: int
    experts_routed_tokens: int
    combine_tokens: int
    tokens_per_expert: tuple[int, ...]


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
    topology: _DeepSeekTopology,
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
    valid_combine = (
        isinstance(combine_tokens, int)
        and not isinstance(combine_tokens, bool)
        and combine_tokens >= 0
    )
    if topology.expert_tensor_parallel_size == 1:
        valid_combine = (
            valid_combine
            and valid_routed
            and combine_tokens == routed_tokens
            and combine_tokens == expert_total
        )
    if not valid_combine:
        expectation = (
            f"equal Experts routed_tokens={routed_tokens!r} and local count "
            f"sum={expert_total!r}"
            if topology.expert_tensor_parallel_size == 1
            else "be a non-negative integer; ETP2 conservation is checked across ranks"
        )
        failures.append(
            _failure(
                "combine_workload",
                f"layer={layer} Combine num_tokens={combine_tokens!r} must {expectation}",
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
            ("tp_size", topology.expert_tensor_parallel_size),
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
    topology: _DeepSeekTopology,
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
                topology=topology,
            )
        )
    return failures


def _validate_model_iteration(
    iteration: Iteration,
    *,
    rank: int,
    pipeline_rank: int,
    topology: _DeepSeekTopology,
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
        "forward-step": topology.microbatches_per_iteration,
        "decoder": topology.microbatches_per_iteration,
        "decoder-postprocess": topology.microbatches_per_iteration,
        "output_layer": topology.microbatches_per_iteration if pipeline_rank == 1 else 0,
        "loss": topology.microbatches_per_iteration if pipeline_rank == 1 else 0,
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
                topology=topology,
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
                topology=topology,
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


def _valid_split_sizes(value: object, expected_size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected_size
        and all(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0
            for size in value
        )
    )


def _validate_moe_tp_positions(
    iteration: Iteration,
    spans: Mapping[str, Sequence[_Span]],
    *,
    rank: int,
    topology: _DeepSeekTopology,
) -> list[Failure]:
    """Check source-stable forward dispatcher collectives at their MoE boundaries."""

    pipeline_rank = topology.pipeline_rank(rank)
    expected_calls = topology.microbatches_per_iteration * (
        len(_MAIN_LAYERS[pipeline_rank]) + len(_MTP_LAYERS[pipeline_rank])
    )
    records = [
        (index, event)
        for index, event in enumerate(iteration.events)
        if event.name in _MOE_NAMES
    ]
    records_per_call = len(_MOE_CALL_SEQUENCE)
    expected_sequence = _MOE_CALL_SEQUENCE * expected_calls
    if tuple((event.name, event.ph) for _index, event in records) != expected_sequence:
        return []

    iteration_id = int(iteration.iteration_id)
    tp_ep_peers = [
        peer for peer in topology.tp_ep_group(rank) if peer != rank
    ]
    expert_tp_peers = [
        peer
        for peer in topology.expert_tensor_parallel_group(rank)
        if peer != rank
    ]
    expert_tensor_rank = rank - topology.expert_tensor_parallel_group(rank)[0]
    failures: list[Failure] = []
    for occurrence in range(expected_calls):
        call = records[
            occurrence * records_per_call : (occurrence + 1) * records_per_call
        ]
        positions = {
            (event.name, event.ph): index for index, event in call
        }
        call_events = {
            (event.name, event.ph): event for _index, event in call
        }
        layer = next(
            event.attrs.get("layer")
            for _index, event in call
            if event.name == "moe-router" and event.ph == "E"
        )
        router_end = positions[("moe-router", "E")]
        dispatch_begin = positions[("moe-dispatch", "B")]
        dispatch_end = positions[("moe-dispatch", "E")]
        experts_begin = positions[("moe-experts", "B")]
        experts_end = positions[("moe-experts", "E")]
        combine_begin = positions[("moe-combine", "B")]

        metadata = [
            span
            for span in spans.get("tp-all-gather-first", ())
            if router_end < span.begin_index
            and span.end_index < dispatch_begin
            and "split_sizes" not in span.begin.attrs
            and span.begin.attrs.get("group_size") == topology.tp_ep_size
            and span.end.attrs.get("group") == tp_ep_peers
        ]
        if len(metadata) != 1:
            failures.append(
                _failure(
                    "metadata_tp_order",
                    f"MoE call {occurrence} layer={layer!r} has {len(metadata)} "
                    "metadata gathers between Router and Dispatch, expected 1",
                    rank=rank,
                    iteration=iteration_id,
                )
            )

        if topology.expert_tensor_parallel_size == 1:
            continue
        forward_gathers = [
            span
            for span in spans.get("tp-all-gather-first", ())
            if dispatch_end < span.begin_index
            and span.end_index < experts_begin
            and span.begin.attrs.get("group_size")
            == topology.expert_tensor_parallel_size
            and span.end.attrs.get("group") == expert_tp_peers
            and _valid_split_sizes(
                span.begin.attrs.get("split_sizes"),
                topology.expert_tensor_parallel_size,
            )
        ]
        forward_reduce_scatters = [
            span
            for span in spans.get("tp-reduce-scatter", ())
            if experts_end < span.begin_index
            and span.end_index < combine_begin
            and span.begin.attrs.get("group_size")
            == topology.expert_tensor_parallel_size
            and span.end.attrs.get("group") == expert_tp_peers
            and _valid_split_sizes(
                span.begin.attrs.get("split_sizes"),
                topology.expert_tensor_parallel_size,
            )
        ]
        if len(forward_gathers) != 2 or len(forward_reduce_scatters) != 1:
            failures.append(
                _failure(
                    "dispatcher_tp_order",
                    f"MoE call {occurrence} layer={layer!r} has forward ETP "
                    f"AG/RS={len(forward_gathers)}/{len(forward_reduce_scatters)}, "
                    "expected 2/1 at the Dispatch/Experts/Combine boundaries",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        split_sizes = [
            span.begin.attrs["split_sizes"]
            for span in (*forward_gathers, *forward_reduce_scatters)
        ]
        reference_split_sizes = split_sizes[0]
        if any(value != reference_split_sizes for value in split_sizes[1:]):
            failures.append(
                _failure(
                    "dispatcher_tp_split",
                    f"MoE call {occurrence} layer={layer!r} has inconsistent "
                    f"forward split_sizes={split_sizes!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            continue
        experts_routed = call_events[("moe-experts", "E")].attrs.get(
            "routed_tokens"
        )
        combine_tokens = call_events[("moe-combine", "E")].attrs.get(
            "num_tokens"
        )
        if (
            sum(reference_split_sizes) != experts_routed
            or reference_split_sizes[expert_tensor_rank] != combine_tokens
        ):
            failures.append(
                _failure(
                    "dispatcher_tp_split",
                    f"MoE call {occurrence} layer={layer!r} split_sizes="
                    f"{reference_split_sizes!r} must sum to Experts routed_tokens="
                    f"{experts_routed!r} and select Combine num_tokens="
                    f"{combine_tokens!r} at ETP rank {expert_tensor_rank}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    return failures


def _validate_tp_iteration(
    iteration: Iteration,
    *,
    rank: int,
    topology: _DeepSeekTopology,
) -> list[Failure]:
    iteration_id = int(iteration.iteration_id)
    spans, failures = _pair_scopes(
        iteration,
        _TP_COLLECTIVES,
        rank=rank,
        code="tp_collective_nesting",
    )
    peer = rank ^ 1
    pipeline_rank = topology.pipeline_rank(rank)
    required_model_tp_collectives = _REQUIRED_MODEL_TP_COLLECTIVES
    if pipeline_rank == PIPELINE_MODEL_PARALLEL_SIZE - 1:
        required_model_tp_collectives |= {"tp-all-gather-last"}
    tp_ep_group = topology.tp_ep_group(rank)
    tp_ep_peers = [group_rank for group_rank in tp_ep_group if group_rank != rank]
    expert_tp_group = topology.expert_tensor_parallel_group(rank)
    expert_tp_peers = [
        group_rank for group_rank in expert_tp_group if group_rank != rank
    ]
    model_tp_observed: set[str] = set()
    ep_metadata_gather_count = 0
    dispatcher_tp_counts: Counter[str] = Counter()
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
            has_split_sizes = "split_sizes" in span.begin.attrs
            if topology.expert_tensor_parallel_size > 1 and has_split_sizes:
                split_sizes = span.begin.attrs.get("split_sizes")
                valid_split_sizes = _valid_split_sizes(
                    split_sizes, topology.expert_tensor_parallel_size
                )
                if not valid_split_sizes:
                    failures.append(
                        _failure(
                            "dispatcher_tp_field",
                            f"event {name!r} has split_sizes={split_sizes!r}; "
                            f"expected {topology.expert_tensor_parallel_size} "
                            "non-negative integer sizes",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
                valid_dispatcher_group = (
                    name in ("tp-all-gather-first", "tp-reduce-scatter")
                    and group_size == topology.expert_tensor_parallel_size
                    and group == expert_tp_peers
                )
                if not valid_dispatcher_group:
                    failures.append(
                        _failure(
                            "dispatcher_tp_group",
                            f"event {name!r} with split_sizes has "
                            f"group_size={group_size!r} and peer group={group!r}; "
                            f"expected dispatcher ETP"
                            f"{topology.expert_tensor_parallel_size} "
                            f"{expert_tp_peers!r}",
                            rank=rank,
                            iteration=iteration_id,
                        )
                    )
                elif valid_split_sizes:
                    dispatcher_tp_counts[name] += 1
            elif group_size == TENSOR_MODEL_PARALLEL_SIZE and group == [peer]:
                model_tp_observed.add(name)
            elif (
                name == "tp-all-gather-first"
                and group_size == topology.tp_ep_size
                and group == tp_ep_peers
            ):
                ep_metadata_gather_count += 1
            else:
                failures.append(
                    _failure(
                        "tp_collective_group",
                        f"event {name!r} has group_size={group_size!r} and peer "
                        f"group={group!r}; expected model TP2 {[peer]!r} or the "
                        f"ETP{topology.expert_tensor_parallel_size}xEP4 metadata "
                        f"gather {tp_ep_peers!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
    failures.extend(
        _validate_moe_tp_positions(
            iteration,
            spans,
            rank=rank,
            topology=topology,
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
    moe_calls = topology.microbatches_per_iteration * (
        len(_MAIN_LAYERS[pipeline_rank]) + len(_MTP_LAYERS[pipeline_rank])
    )
    if ep_metadata_gather_count != moe_calls:
        failures.append(
            _failure(
                "tp_collective_count",
                "tp-all-gather-first has "
                f"{ep_metadata_gather_count} complete "
                f"ETP{topology.expert_tensor_parallel_size}xEP4 metadata spans, "
                f"expected {moe_calls}",
                rank=rank,
                iteration=iteration_id,
            )
        )
    if topology.expert_tensor_parallel_size > 1:
        expected_dispatcher_count = 3 * moe_calls
        for name in ("tp-all-gather-first", "tp-reduce-scatter"):
            observed_count = dispatcher_tp_counts[name]
            if observed_count != expected_dispatcher_count:
                failures.append(
                    _failure(
                        "dispatcher_tp_count",
                        f"event {name!r} has {observed_count} complete ETP2 spans "
                        f"with split_sizes, expected {expected_dispatcher_count} "
                        f"for {moe_calls} MoE calls",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
    allreduce_spans, allreduce_failures = _pair_scopes(
        iteration,
        ("tp-allreduce",),
        rank=rank,
        code="tp_allreduce_nesting",
    )
    failures.extend(allreduce_failures)
    expected_allreduce_count = topology.microbatches_per_iteration * (
        len(_MAIN_LAYERS[pipeline_rank]) + len(_MTP_LAYERS[pipeline_rank])
    )
    observed_allreduce_spans = allreduce_spans.get("tp-allreduce", ())
    if len(observed_allreduce_spans) != expected_allreduce_count:
        failures.append(
            _failure(
                "tp_allreduce_count",
                f"tp-allreduce has {len(observed_allreduce_spans)} complete spans, "
                f"expected {expected_allreduce_count}",
                rank=rank,
                iteration=iteration_id,
            )
        )
    for span in observed_allreduce_spans:
        for field_name, expected in (
            ("op", "all_reduce"),
            ("data_bytes", 512),
            ("group_size", TENSOR_MODEL_PARALLEL_SIZE),
            ("timing_phase", "collective_call"),
            ("payload_role", "inplace_input_output"),
        ):
            if span.begin.attrs.get(field_name) != expected:
                failures.append(
                    _failure(
                        "tp_allreduce_field",
                        f"tp-allreduce has {field_name}="
                        f"{span.begin.attrs.get(field_name)!r}, expected {expected!r}",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
        if span.end.attrs.get("group") != [peer]:
            failures.append(
                _failure(
                    "tp_allreduce_group",
                    f"tp-allreduce has peer group={span.end.attrs.get('group')!r}, "
                    f"expected {[peer]!r}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    if pipeline_rank == PIPELINE_MODEL_PARALLEL_SIZE - 1:
        linear_failures, _operation_ids = (
            tp_probe_contract._validate_linear_lifecycle(
                iteration,
                rank=rank,
                expected_routes=tp_probe_contract._SP_LINEAR_ROUTES,
                tensor_parallel_size=TENSOR_MODEL_PARALLEL_SIZE,
            )
        )
        failures.extend(linear_failures)
        linear_route_counts = Counter(
            event.attrs.get("collective_op")
            for event in iteration.events
            if event.name == "tp-linear-async-launch" and event.ph == "B"
        )
        launches_per_route = 2 * topology.microbatches_per_iteration
        expected_linear_route_counts = Counter(
            {
                "all-gather": launches_per_route,
                "reduce-scatter": launches_per_route,
            }
        )
        if linear_route_counts != expected_linear_route_counts:
            failures.append(
                _failure(
                    "tp_linear_count",
                    f"local Linear launch routes are {dict(linear_route_counts)}, "
                    f"expected {dict(expected_linear_route_counts)}",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    else:
        unexpected_linear_events = [
            event
            for event in iteration.events
            if event.name
            in ("tp-linear-async-launch", "tp-linear-async-complete")
        ]
        if unexpected_linear_events:
            failures.append(
                _failure(
                    "tp_linear_stage",
                    "local Linear lifecycle events must be absent before the MTP stage",
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    sync_failures, _sp_bytes, _embedding_bytes = (
        tp_probe_contract._validate_final_grad_sync(
            iteration,
            rank=rank,
            schedule="non-interleaved-1f1b",
            expect_sp_layernorm=True,
            tp_peers=(peer,),
            embedding_peer=topology.pipeline_peer(rank),
        )
    )
    failures.extend(sync_failures)
    return failures


def _collect_workloads(
    iteration: Iteration,
    *,
    rank: int,
) -> Mapping[tuple[int, int], _Workload]:
    spans, failures = _pair_scopes(
        iteration,
        _MODEL_SCOPE_NAMES,
        rank=rank,
        code="model_nesting",
    )
    if failures:
        return {}
    workloads: dict[tuple[int, int], dict[str, object]] = defaultdict(dict)
    duplicates: set[tuple[int, int]] = set()
    forwards = sorted(spans.get("forward-step", ()), key=lambda span: span.begin_index)
    for microbatch, forward in enumerate(forwards):
        for event in iteration.events[forward.begin_index + 1 : forward.end_index]:
            if event.ph != "E" or event.name not in (
                "moe-router",
                "moe-experts",
                "moe-combine",
            ):
                continue
            layer = event.attrs.get("layer")
            if not isinstance(layer, int) or isinstance(layer, bool):
                continue
            if event.name == "moe-combine":
                value: object = event.attrs.get("num_tokens")
            elif event.name == "moe-experts":
                routed_tokens = event.attrs.get("routed_tokens")
                tokens_per_expert = event.attrs.get("tokens_per_expert")
                if not (
                    isinstance(routed_tokens, int)
                    and not isinstance(routed_tokens, bool)
                    and routed_tokens >= 0
                    and isinstance(tokens_per_expert, list)
                    and len(tokens_per_expert) == 16
                    and all(
                        isinstance(count, int)
                        and not isinstance(count, bool)
                        and count >= 0
                        for count in tokens_per_expert
                    )
                ):
                    continue
                value = (routed_tokens, tuple(tokens_per_expert))
            else:
                value = event.attrs.get("routed_tokens")
            if (
                event.name != "moe-experts"
                and (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                )
            ):
                continue
            key = (microbatch, layer)
            if event.name in workloads[key]:
                duplicates.add(key)
            else:
                workloads[key][event.name] = value
    result: dict[tuple[int, int], _Workload] = {}
    for key, values in workloads.items():
        if key in duplicates or not {
            "moe-router",
            "moe-experts",
            "moe-combine",
        } <= values.keys():
            continue
        router_routed = values["moe-router"]
        experts = values["moe-experts"]
        combine_tokens = values["moe-combine"]
        if not (
            isinstance(router_routed, int)
            and not isinstance(router_routed, bool)
            and isinstance(experts, tuple)
            and len(experts) == 2
            and isinstance(experts[0], int)
            and isinstance(experts[1], tuple)
            and isinstance(combine_tokens, int)
            and not isinstance(combine_tokens, bool)
        ):
            continue
        result[key] = _Workload(
            router_routed_tokens=router_routed,
            experts_routed_tokens=experts[0],
            combine_tokens=combine_tokens,
            tokens_per_expert=experts[1],
        )
    return result


def _validate_ep_conservation(
    by_rank: Mapping[int, tuple[Rank, Sequence[Iteration]]],
    *,
    topology: _DeepSeekTopology,
) -> list[Failure]:
    workloads: dict[tuple[int, int], Mapping[tuple[int, int], _Workload]] = {}
    for rank, (_shard, iterations) in by_rank.items():
        for iteration in iterations:
            workloads[(rank, int(iteration.iteration_id))] = _collect_workloads(
                iteration,
                rank=rank,
            )
    failures: list[Failure] = []
    for pipeline_rank in range(PIPELINE_MODEL_PARALLEL_SIZE):
        expected_layers = _MAIN_LAYERS[pipeline_rank] + _MTP_LAYERS[pipeline_rank]
        stage_base = pipeline_rank * topology.stage_size
        for expert_data_rank in range(topology.expert_data_parallel_size):
            group_base = stage_base + expert_data_rank * topology.tp_ep_size
            group = tuple(range(group_base, group_base + topology.tp_ep_size))
            for iteration in _ITERATIONS:
                for microbatch in range(topology.microbatches_per_iteration):
                    for layer in expected_layers:
                        by_group_rank = {
                            rank: workloads.get((rank, iteration), {}).get(
                                (microbatch, layer)
                            )
                            for rank in group
                        }
                        if any(value is None for value in by_group_rank.values()):
                            continue
                        complete = {
                            rank: value
                            for rank, value in by_group_rank.items()
                            if value is not None
                        }
                        router_total = sum(
                            value.router_routed_tokens for value in complete.values()
                        )
                        for expert_tensor_rank in range(
                            topology.expert_tensor_parallel_size
                        ):
                            ep_slice = topology.expert_model_parallel_group(
                                group_base + expert_tensor_rank
                            )
                            experts_total = sum(
                                complete[rank].experts_routed_tokens
                                for rank in ep_slice
                            )
                            if router_total != experts_total:
                                failures.append(
                                    _failure(
                                        "ep_conservation",
                                        f"ETP{topology.expert_tensor_parallel_size}xEP4 "
                                        f"group {group} ETP slice {ep_slice} layer={layer} "
                                        f"Router total={router_total}, Experts total="
                                        f"{experts_total}",
                                        rank=ep_slice[0],
                                        iteration=iteration,
                                        microbatch=microbatch,
                                    )
                                )
                        if topology.expert_tensor_parallel_size == 1:
                            continue
                        combine_total = sum(
                            value.combine_tokens for value in complete.values()
                        )
                        if combine_total != router_total:
                            failures.append(
                                _failure(
                                    "combine_conservation",
                                    f"ETP2xEP4 group {group} layer={layer} Combine "
                                    f"total={combine_total}, Router total={router_total}",
                                    rank=group[0],
                                    iteration=iteration,
                                    microbatch=microbatch,
                                )
                            )
                        for expert_rank in range(EXPERT_MODEL_PARALLEL_SIZE):
                            etp_group = tuple(
                                group_base
                                + expert_rank * topology.expert_tensor_parallel_size
                                + expert_tensor_rank
                                for expert_tensor_rank in range(
                                    topology.expert_tensor_parallel_size
                                )
                            )
                            reference = complete[etp_group[0]]
                            mismatched = [
                                rank
                                for rank in etp_group[1:]
                                if complete[rank].tokens_per_expert
                                != reference.tokens_per_expert
                            ]
                            if mismatched:
                                failures.append(
                                    _failure(
                                        "etp_workload",
                                        f"ETP group {etp_group} layer={layer} has "
                                        "different tokens_per_expert values",
                                        rank=mismatched[0],
                                        iteration=iteration,
                                        microbatch=microbatch,
                                    )
                                )
                            etp_combine_total = sum(
                                complete[rank].combine_tokens for rank in etp_group
                            )
                            if etp_combine_total != reference.experts_routed_tokens:
                                failures.append(
                                    _failure(
                                        "etp_combine_conservation",
                                        f"ETP group {etp_group} layer={layer} Combine "
                                        f"total={etp_combine_total}, Experts routed="
                                        f"{reference.experts_routed_tokens}",
                                        rank=etp_group[0],
                                        iteration=iteration,
                                        microbatch=microbatch,
                                    )
                                )
    return failures


def _validate_d1_dp_groups(
    by_rank: Mapping[int, tuple[Rank, Sequence[Iteration]]],
    *,
    topology: _DeepSeekTopology,
) -> list[Failure]:
    """Require D1 DistOpt events to expose model-DP and expert-DP groups."""

    failures: list[Failure] = []
    expected_roles = frozenset(("model-dp", "expert-dp"))
    for rank, (_shard, iterations) in by_rank.items():
        expected_groups = {
            "model-dp": topology.model_data_parallel_group(rank),
            "expert-dp": topology.expert_data_parallel_group(rank),
        }
        observed_roles: dict[str, set[str]] = {
            name: set() for name in _DP_GROUP_ROUTES
        }
        for iteration in iterations:
            iteration_id = int(iteration.iteration_id)
            iteration_roles: dict[str, set[str]] = {
                name: set() for name in _DP_GROUP_ROUTES
            }
            spans, pairing_failures = _pair_scopes(
                iteration,
                _DP_GROUP_ROUTES,
                rank=rank,
                code="dp_group_nesting",
            )
            failures.extend(pairing_failures)
            for name, expected_op in _DP_GROUP_ROUTES.items():
                for span in spans.get(name, ()):
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
                            _failure(
                                "dp_group",
                                f"event {name!r} has group_size={group_size!r} and "
                                f"peer group={peers!r}; expected model-DP"
                                f"{topology.model_data_parallel_size} or expert-DP"
                                f"{topology.expert_data_parallel_size} membership",
                                rank=rank,
                                iteration=iteration_id,
                            )
                        )
                        continue
                    observed_roles[name].add(matched_role)
                    iteration_roles[name].add(matched_role)
                    expected_fields = {
                        "group_role": "intra_optimizer_instance",
                        "op": expected_op,
                    }
                    for field_name, expected in expected_fields.items():
                        if span.begin.attrs.get(field_name) != expected:
                            failures.append(
                                _failure(
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
                                _failure(
                                    "dp_field",
                                    f"event {name!r} has invalid "
                                    f"{field_name}={value!r}",
                                    rank=rank,
                                    iteration=iteration_id,
                                )
                            )
            reduce_scatter_roles = iteration_roles["dp-reduce-scatter"]
            if reduce_scatter_roles != expected_roles:
                failures.append(
                    _failure(
                        "dp_group_count",
                        "event 'dp-reduce-scatter' covers "
                        f"{sorted(reduce_scatter_roles)!r}, expected "
                        f"{sorted(expected_roles)!r} in this iteration",
                        rank=rank,
                        iteration=iteration_id,
                    )
                )
        for name, roles in observed_roles.items():
            if name == "dp-reduce-scatter":
                continue
            if roles != expected_roles:
                failures.append(
                    _failure(
                        "dp_group_count",
                        f"event {name!r} covers {sorted(roles)!r}, expected "
                        f"{sorted(expected_roles)!r} across the two-iteration window",
                        rank=rank,
                        iteration=None,
                    )
                )
    return failures


def _validate_trace(
    trace_root: Path,
    *,
    topology: _DeepSeekTopology,
    require_d1_dp_groups: bool = False,
) -> tuple[Failure, ...]:
    by_rank = _load_iterations(trace_root)
    failures: list[Failure] = []
    expected_ranks = tuple(range(topology.world_size))
    if tuple(sorted(by_rank)) != expected_ranks:
        failures.append(
            Failure(
                "trace.deepseek_tp2_sp.ranks",
                f"expected ranks {expected_ranks}, observed {tuple(sorted(by_rank))}",
                topology.profile_name,
            )
        )
    for global_rank in expected_ranks:
        loaded = by_rank.get(global_rank)
        if loaded is None:
            continue
        shard_rank, iterations = loaded
        pipeline_rank = topology.pipeline_rank(global_rank)
        stage_rank = global_rank % topology.stage_size
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
                    topology=topology,
                )
            )
            failures.extend(
                _validate_tp_iteration(
                    iteration,
                    rank=global_rank,
                    topology=topology,
                )
            )
    failures.extend(_validate_ep_conservation(by_rank, topology=topology))
    if require_d1_dp_groups:
        failures.extend(_validate_d1_dp_groups(by_rank, topology=topology))
    return tuple(failures)


def validate_deepseek_tp2_sp_trace(trace_root: Path) -> tuple[Failure, ...]:
    """Validate the fixed TP2/PP2/DP2/EP4/ETP1 DeepSeek L3 trace."""

    return _validate_trace(trace_root, topology=_SINGLE_NODE_TOPOLOGY)


def validate_deepseek_d1_tp2_sp_etp1_trace(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate the D1 TP2/PP2/DP4/EP4/ETP1/expert-DP2 trace."""

    return (
        *_validate_trace(
            trace_root,
            topology=_D1_ETP1_TOPOLOGY,
            require_d1_dp_groups=True,
        ),
        *dp_probe_contract.validate_dp_distopt_overlap(trace_root),
    )


def validate_deepseek_d1_tp2_sp_etp2_trace(
    trace_root: Path,
) -> tuple[Failure, ...]:
    """Validate the D1 TP2/PP2/DP4/EP4/ETP2/expert-DP1 trace."""

    return (
        *_validate_trace(
            trace_root,
            topology=_D1_ETP2_TOPOLOGY,
            require_d1_dp_groups=True,
        ),
        *dp_probe_contract.validate_dp_distopt_overlap(trace_root),
    )


__all__ = [
    "D1_ETP2_EXPERT_DATA_PARALLEL_SIZE",
    "D1_ETP2_MICROBATCHES_PER_ITERATION",
    "D1_ETP2_MODEL_DATA_PARALLEL_SIZE",
    "D1_ETP2_WORLD_SIZE",
    "D1_EXPERT_DATA_PARALLEL_SIZE",
    "D1_MICROBATCHES_PER_ITERATION",
    "D1_MODEL_DATA_PARALLEL_SIZE",
    "D1_WORLD_SIZE",
    "EXPERT_DATA_PARALLEL_SIZE",
    "EXPERT_MODEL_PARALLEL_SIZE",
    "EXPERT_TENSOR_PARALLEL_SIZE",
    "MICROBATCHES_PER_ITERATION",
    "MODEL_DATA_PARALLEL_SIZE",
    "PIPELINE_MODEL_PARALLEL_SIZE",
    "RANK_ORDER",
    "TENSOR_MODEL_PARALLEL_SIZE",
    "WORLD_SIZE",
    "validate_deepseek_d1_tp2_sp_etp1_trace",
    "validate_deepseek_d1_tp2_sp_etp2_trace",
    "validate_deepseek_tp2_sp_trace",
]
