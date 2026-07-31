# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime contracts for the controlled two-rank MultiModule Bridge profile."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from megatron.megalens.trace_aggregate import (
    Event,
    Iteration,
    collect_benchmark_files,
    read_benchmark_file,
)
from tests.test_utils.runners.megalens_run_manifest import Failure

_DIRECTIONAL_EVENTS = frozenset(
    (
        "bridge-send-forward",
        "bridge-recv-forward",
        "bridge-send-backward",
        "bridge-recv-backward",
    )
)
_RANK_DIRECTIONS = {
    0: ("bridge-send-forward", "bridge-recv-backward"),
    1: ("bridge-recv-forward", "bridge-send-backward"),
}
_DIRECTION_TO_EVENT = {
    ("send", "forward"): "bridge-send-forward",
    ("recv", "forward"): "bridge-recv-forward",
    ("send", "backward"): "bridge-send-backward",
    ("recv", "backward"): "bridge-recv-backward",
}
_SHAPE_BYTES = 3 * 8
_PAYLOAD_BYTES = 64 * 2 * 512 * 2


@dataclass(frozen=True)
class _BridgeRankPlan:
    role: str
    peers: tuple[int, ...]
    payload_bytes: int


@dataclass(frozen=True)
class _BridgeBroadcastPlan:
    pipeline_direction: str
    grid_side: str
    collective_role: str
    source_rank: int
    payload_bytes: int


@dataclass(frozen=True)
class _AsymmetricBridgeTopology:
    name: str
    loss_ranks: frozenset[int]
    bridge_ranks: Mapping[int, _BridgeRankPlan]
    broadcast_ranks: Mapping[int, _BridgeBroadcastPlan]


_FANIN_TOPOLOGY = _AsymmetricBridgeTopology(
    name="multimodule-bridge8-fanin",
    loss_ranks=frozenset((6, 7)),
    bridge_ranks={
        1: _BridgeRankPlan("sender", (4,), _PAYLOAD_BYTES * 2),
        3: _BridgeRankPlan("sender", (4,), _PAYLOAD_BYTES * 2),
        4: _BridgeRankPlan("receiver", (1, 3), _PAYLOAD_BYTES * 2),
    },
    broadcast_ranks={
        1: _BridgeBroadcastPlan("backward", "src", "source", 1, _PAYLOAD_BYTES * 2),
        3: _BridgeBroadcastPlan("backward", "src", "source", 3, _PAYLOAD_BYTES * 2),
        4: _BridgeBroadcastPlan("forward", "dest", "source", 4, _PAYLOAD_BYTES * 4),
        5: _BridgeBroadcastPlan("forward", "dest", "participant", 4, _PAYLOAD_BYTES * 4),
    },
)

_FANOUT_TOPOLOGY = _AsymmetricBridgeTopology(
    name="multimodule-bridge8-fanout",
    loss_ranks=frozenset((5, 7)),
    bridge_ranks={
        3: _BridgeRankPlan("sender", (4, 6), _PAYLOAD_BYTES),
        4: _BridgeRankPlan("receiver", (3,), _PAYLOAD_BYTES),
        6: _BridgeRankPlan("receiver", (3,), _PAYLOAD_BYTES),
    },
    broadcast_ranks={
        2: _BridgeBroadcastPlan("backward", "src", "participant", 3, _PAYLOAD_BYTES * 2),
        3: _BridgeBroadcastPlan("backward", "src", "source", 3, _PAYLOAD_BYTES * 2),
        4: _BridgeBroadcastPlan("forward", "dest", "source", 4, _PAYLOAD_BYTES),
        6: _BridgeBroadcastPlan("forward", "dest", "source", 6, _PAYLOAD_BYTES),
    },
)


def _failure(code: str, message: str, evidence: str) -> Failure:
    return Failure(code, message, evidence)


def validate_multimodule_bridge_run(
    run_root: Path,
    trace_enabled: bool,
) -> tuple[Failure, ...]:
    """Validate the rank-local schedule result written for trace-off and trace-on."""

    failures: list[Failure] = []
    for rank, role in ((0, "encoder"), (1, "llm")):
        path = run_root / f"training-result-rank-{rank}.json"
        if not path.is_file():
            failures.append(
                _failure(
                    "run.bridge.result_missing",
                    f"rank {rank} did not write a training result",
                    str(path),
                )
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "backward_completed": True,
            "completed": True,
            "global_rank": rank,
            "gradient_finite": True,
            "loss_finite": True,
            "module_role": role,
            "optimizer_step": "not_run",
            "world_size": 2,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                failures.append(
                    _failure(
                        "run.bridge.result",
                        f"rank {rank} has {field}={payload.get(field)!r}, expected {value!r}",
                        str(path),
                    )
                )
        if payload.get("trace_enabled") is not trace_enabled:
            failures.append(
                _failure(
                    "run.bridge.trace_mode",
                    (
                        f"rank {rank} recorded trace_enabled="
                        f"{payload.get('trace_enabled')!r}, expected {trace_enabled}"
                    ),
                    str(path),
                )
            )
        gradient_count = payload.get("gradient_count")
        gradient_norm = payload.get("gradient_norm")
        if (
            not isinstance(gradient_count, int)
            or gradient_count <= 0
            or not isinstance(gradient_norm, (int, float))
            or not math.isfinite(gradient_norm)
            or gradient_norm <= 0
        ):
            failures.append(
                _failure(
                    "run.bridge.gradient",
                    f"rank {rank} has invalid gradient terminal state",
                    str(path),
                )
            )
        loss_count = payload.get("loss_count")
        expected_loss = loss_count == (0 if rank == 0 else 4)
        if not expected_loss:
            failures.append(
                _failure(
                    "run.bridge.loss",
                    f"rank {rank} has loss_count={loss_count!r}",
                    str(path),
                )
            )
    return tuple(failures)


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    by_rank: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None:
            raise ValueError(f"trace shard {rank} has no global rank")
        by_rank[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return by_rank


def _events(iteration: Iteration, name: str, phase: str = "B") -> list[Event]:
    return [
        event
        for event in iteration.events
        if event.name == name and event.ph == phase
    ]


def _validate_scope_completions(
    iteration: Iteration,
    *,
    rank: int,
    event_names: Sequence[str],
) -> list[Failure]:
    iteration_id = int(iteration.iteration_id)
    failures: list[Failure] = []
    for event_name in event_names:
        begins = _events(iteration, event_name, "B")
        ends = _events(iteration, event_name, "E")
        if len(begins) != len(ends):
            failures.append(
                _failure(
                    "trace.bridge.scope_pairing",
                    (
                        f"{event_name} has {len(begins)} begin records and "
                        f"{len(ends)} end records"
                    ),
                    f"rank={rank} iteration={iteration_id}",
                )
            )
        scope_open = False
        for event in (
            event for event in iteration.events if event.name == event_name
        ):
            if event.ph == "B":
                if scope_open:
                    failures.append(
                        _failure(
                            "trace.bridge.scope_pairing",
                            f"{event_name} contains overlapping begin records",
                            f"rank={rank} iteration={iteration_id}",
                        )
                    )
                scope_open = True
                continue
            if event.ph != "E":
                continue
            if not scope_open:
                failures.append(
                    _failure(
                        "trace.bridge.scope_pairing",
                        f"{event_name} has an end record without an open scope",
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
            scope_open = False
            if (
                event.attrs.get("completed") is not True
                or event.attrs.get("error_type") is not None
            ):
                failures.append(
                    _failure(
                        "trace.bridge.scope_completion",
                        f"{event_name} did not complete successfully",
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
        if scope_open:
            failures.append(
                _failure(
                    "trace.bridge.scope_pairing",
                    f"{event_name} has an unclosed begin record",
                    f"rank={rank} iteration={iteration_id}",
                )
            )
    return failures


def _check_semantic_role(
    attrs: Mapping[str, Any],
    *,
    rank: int,
    iteration: int,
    event_name: str,
) -> list[Failure]:
    message_kind = attrs.get("message_kind")
    pipeline_direction = attrs.get("pipeline_direction")
    semantic_role = "activation" if pipeline_direction == "forward" else "gradient"
    failures = []
    if message_kind == "shape":
        valid = attrs.get("shape_of") == semantic_role and "payload_role" not in attrs
    elif message_kind == "payload":
        valid = attrs.get("payload_role") == semantic_role and "shape_of" not in attrs
    else:
        valid = False
    if not valid:
        failures.append(
            _failure(
                "trace.bridge.message_kind",
                f"{event_name} has inconsistent shape/payload fields",
                f"rank={rank} iteration={iteration}",
            )
        )
    return failures


def _validate_iteration(
    iteration: Iteration,
    *,
    rank: int,
) -> tuple[Failure, ...]:
    iteration_id = int(iteration.iteration_id)
    failures: list[Failure] = []
    expected_peer = 1 - rank
    directional = [
        event
        for event in iteration.events
        if event.name in _DIRECTIONAL_EVENTS and event.ph == "B"
    ]
    direction_counts = Counter(event.name for event in directional)
    if set(direction_counts) != set(_RANK_DIRECTIONS[rank]):
        failures.append(
            _failure(
                "trace.bridge.direction",
                (
                    f"rank {rank} has Bridge direction counts "
                    f"{dict(sorted(direction_counts.items()))}"
                ),
                f"iteration={iteration_id}",
            )
        )
    failures.extend(
        _validate_scope_completions(
            iteration,
            rank=rank,
            event_names=tuple(_DIRECTIONAL_EVENTS),
        )
    )

    variants = {
        event_name: Counter() for event_name in _RANK_DIRECTIONS[rank]
    }
    for event in directional:
        attrs = event.attrs
        expected_event_name = _DIRECTION_TO_EVENT.get(
            (attrs.get("direction"), attrs.get("pipeline_direction"))
        )
        if (
            event.name != expected_event_name
            or attrs.get("peer_rank") != expected_peer
            or attrs.get("src_module") != "encoder"
            or attrs.get("dest_module") != "llm"
            or attrs.get("backend") != "nccl"
        ):
            failures.append(
                _failure(
                    "trace.bridge.route",
                    f"{event.name} has an invalid peer, module, or direction route",
                    f"rank={rank} iteration={iteration_id}",
                )
            )
        failures.extend(
            _check_semantic_role(
                attrs,
                rank=rank,
                iteration=iteration_id,
                event_name=event.name,
            )
        )
        transport = attrs.get("transport_api")
        message_kind = attrs.get("message_kind")
        expected_bytes = (
            _SHAPE_BYTES if message_kind == "shape" else _PAYLOAD_BYTES
        )
        if attrs.get("data_bytes") != expected_bytes:
            failures.append(
                _failure(
                    "trace.bridge.data_bytes",
                    (
                        f"{event.name} has data_bytes={attrs.get('data_bytes')!r}, "
                        f"expected {expected_bytes}"
                    ),
                    f"rank={rank} iteration={iteration_id}",
                )
            )
        if transport == "send_recv" and message_kind == "payload":
            if attrs.get("completion_kind") != "inline_api_return":
                failures.append(
                    _failure(
                        "trace.bridge.completion",
                        f"{event.name} blocking completion is not inline API return",
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
            if event.name in variants:
                variants[event.name]["blocking-payload"] += 1
        elif transport == "batch_isend_irecv" and message_kind in {"shape", "payload"}:
            if attrs.get("completion_kind") != "work_wait":
                failures.append(
                    _failure(
                        "trace.bridge.completion",
                        f"{event.name} batch completion is not Work.wait",
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
            if event.name in variants:
                variants[event.name][f"batch-{message_kind}"] += 1
        else:
            failures.append(
                _failure(
                    "trace.bridge.transport",
                    f"{event.name} has unsupported transport/message variant",
                    f"rank={rank} iteration={iteration_id}",
                )
            )

    launched: dict[str, tuple[str, str, str]] = {}
    launch_events = _events(iteration, "bridge-p2p-launch")
    launch_ends = _events(iteration, "bridge-p2p-launch", "E")
    if len(launch_events) != 8 or len(launch_ends) != 8:
        failures.append(
            _failure(
                "trace.bridge.launch_count",
                (
                    f"rank {rank} has {len(launch_events)} launch begins and "
                    f"{len(launch_ends)} launch ends"
                ),
                f"iteration={iteration_id}",
            )
        )
    launch_pairs: list[tuple[Event, Event]] = []
    open_launch = None
    for event in (
        event
        for event in iteration.events
        if event.name == "bridge-p2p-launch"
    ):
        if event.ph == "B":
            if open_launch is not None:
                failures.append(
                    _failure(
                        "trace.bridge.launch_scope",
                        "bridge-p2p-launch contains overlapping begin records",
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
            open_launch = event
        elif event.ph == "E":
            if open_launch is None:
                failures.append(
                    _failure(
                        "trace.bridge.launch_scope",
                        "bridge-p2p-launch has an unmatched end record",
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
            else:
                launch_pairs.append((open_launch, event))
                open_launch = None
    if open_launch is not None:
        failures.append(
            _failure(
                "trace.bridge.launch_scope",
                "bridge-p2p-launch has an unclosed begin record",
                f"rank={rank} iteration={iteration_id}",
            )
        )

    batch_ids: set[str] = set()
    launch_kinds: Counter[str] = Counter()
    expected_combined_order = {
        0: {
            "shape": ("bridge-send-forward", "bridge-recv-backward"),
            "payload": ("bridge-send-forward", "bridge-recv-backward"),
        },
        1: {
            "shape": ("bridge-recv-forward", "bridge-send-backward"),
            "payload": ("bridge-send-backward", "bridge-recv-forward"),
        },
    }
    observed_launch_sequence = []
    for launch in launch_events:
        attrs = launch.attrs
        operations = attrs.get("operations")
        batch_id = attrs.get("batch_id")
        message_kind = attrs.get("message_kind")
        if (
            attrs.get("transport_api") != "batch_isend_irecv"
            or attrs.get("completion_mode") != "internal_wait"
            or attrs.get("src_module") != "encoder"
            or attrs.get("dest_module") != "llm"
            or message_kind not in {"shape", "payload"}
            or not isinstance(batch_id, str)
            or not batch_id
            or batch_id in batch_ids
            or not isinstance(operations, list)
            or attrs.get("operation_count") != len(operations)
            or not operations
        ):
            failures.append(
                _failure(
                    "trace.bridge.launch",
                    "bridge-p2p-launch has an invalid operation list or route",
                    f"rank={rank} iteration={iteration_id}",
                )
            )
            continue
        batch_ids.add(batch_id)
        launch_kinds[message_kind] += 1
        operation_names = tuple(
            _DIRECTION_TO_EVENT.get(
                (operation.get("direction"), operation.get("pipeline_direction"))
            )
            if isinstance(operation, Mapping)
            else None
            for operation in operations
        )
        observed_launch_sequence.append((message_kind, operation_names))
        if len(operations) == 2:
            if operation_names != expected_combined_order[rank][message_kind]:
                failures.append(
                    _failure(
                        "trace.bridge.operation_order",
                        (
                            f"{batch_id} has operation order {operation_names}, "
                            f"expected {expected_combined_order[rank][message_kind]}"
                        ),
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
        elif len(operations) != 1:
            failures.append(
                _failure(
                    "trace.bridge.operation_count",
                    f"{batch_id} has {len(operations)} operations",
                    f"rank={rank} iteration={iteration_id}",
                )
            )
        for index, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                failures.append(
                    _failure(
                        "trace.bridge.operation",
                        f"{batch_id} contains a non-mapping operation",
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
                continue
            operation_id = operation.get("operation_id")
            event_name = _DIRECTION_TO_EVENT.get(
                (operation.get("direction"), operation.get("pipeline_direction"))
            )
            expected_bytes = (
                _SHAPE_BYTES
                if operation.get("message_kind") == "shape"
                else _PAYLOAD_BYTES
            )
            if (
                operation_id != f"{batch_id}:{index}"
                or operation_id in launched
                or operation.get("request_id") != operation_id
                or event_name not in _RANK_DIRECTIONS[rank]
                or operation.get("message_kind") != message_kind
                or operation.get("peer_rank") != expected_peer
                or operation.get("backend") != "nccl"
                or operation.get("transport_api") != "batch_isend_irecv"
                or operation.get("completion_mode") != "internal_wait"
                or operation.get("data_bytes") != expected_bytes
            ):
                failures.append(
                    _failure(
                        "trace.bridge.operation",
                        f"bridge-p2p-launch contains an invalid operation {operation!r}",
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
                continue
            failures.extend(
                _check_semantic_role(
                    operation,
                    rank=rank,
                    iteration=iteration_id,
                    event_name="bridge-p2p-launch operation",
                )
            )
            launched[operation_id] = (
                batch_id,
                event_name,
                str(operation.get("message_kind")),
            )

    if launch_kinds != Counter({"shape": 5, "payload": 3}):
        failures.append(
            _failure(
                "trace.bridge.launch_kinds",
                f"rank {rank} has launch kinds {dict(launch_kinds)}",
                f"iteration={iteration_id}",
            )
        )
    combined_orders = expected_combined_order[rank]
    expected_launch_sequence = [
        (
            "shape",
            (
                "bridge-send-forward"
                if rank == 0
                else "bridge-recv-forward",
            ),
        )
    ]
    for _ in range(3):
        expected_launch_sequence.extend(
            (
                ("shape", combined_orders["shape"]),
                ("payload", combined_orders["payload"]),
            )
        )
    expected_launch_sequence.append(
        (
            "shape",
            (
                "bridge-recv-backward"
                if rank == 0
                else "bridge-send-backward",
            ),
        )
    )
    if observed_launch_sequence != expected_launch_sequence:
        failures.append(
            _failure(
                "trace.bridge.launch_order",
                f"rank {rank} has an invalid 1F1B Bridge launch sequence",
                f"iteration={iteration_id}",
            )
        )

    completed: Counter[str] = Counter()
    observed_completion_order = []
    completion_begins: dict[str, list[Event]] = {}
    completion_ends: dict[str, list[Event]] = {}
    for event_name in _RANK_DIRECTIONS[rank]:
        begins = _events(iteration, event_name)
        ends = _events(iteration, event_name, "E")
        for begin, end in zip(begins, ends):
            operation_id = begin.attrs.get("operation_id")
            if (
                begin.attrs.get("transport_api") == "batch_isend_irecv"
                and isinstance(operation_id, str)
            ):
                completion_begins.setdefault(operation_id, []).append(begin)
                completion_ends.setdefault(operation_id, []).append(end)
    for event in directional:
        if event.attrs.get("transport_api") != "batch_isend_irecv":
            continue
        operation_id = event.attrs.get("operation_id")
        if isinstance(operation_id, str):
            observed_completion_order.append(operation_id)
        expected = launched.get(operation_id)
        observed = (
            str(event.attrs.get("batch_id")),
            event.name,
            str(event.attrs.get("message_kind")),
        )
        if (
            expected is None
            or expected != observed
            or event.attrs.get("request_id") != operation_id
        ):
            failures.append(
                _failure(
                    "trace.bridge.identity",
                    f"{event.name} cannot be paired with launch operation {operation_id!r}",
                    f"rank={rank} iteration={iteration_id}",
                )
            )
        if isinstance(operation_id, str):
            completed[operation_id] += 1

    launch_operation_ids = [
        [
            operation.get("operation_id")
            for operation in launch.attrs.get("operations", ())
            if isinstance(operation, Mapping)
            and isinstance(operation.get("operation_id"), str)
        ]
        for launch in launch_events
    ]
    expected_completed: Counter[str] = Counter()
    expected_completion_order = []
    for operation_ids in launch_operation_ids:
        completion_counts = [completed[operation_id] for operation_id in operation_ids]
        positional = bool(completion_counts) and all(
            count == 1 for count in completion_counts
        )
        coalesced = len(operation_ids) > 1 and all(
            count == 0 for count in completion_counts
        )
        if positional:
            expected_completed.update(operation_ids)
            expected_completion_order.extend(operation_ids)
        elif not coalesced:
            failures.append(
                _failure(
                    "trace.bridge.identity",
                    (
                        "Bridge launch has neither one completion per logical "
                        "operation nor an unpaired coalesced completion"
                    ),
                    f"rank={rank} iteration={iteration_id}",
                )
            )
    if completed != expected_completed:
        failures.append(
            _failure(
                "trace.bridge.identity",
                "Bridge Work completion IDs do not match the observed launch mode",
                f"rank={rank} iteration={iteration_id}",
            )
        )
    if observed_completion_order != expected_completion_order:
        failures.append(
            _failure(
                "trace.bridge.completion_order",
                f"rank {rank} Work.wait records do not follow launch position order",
                f"iteration={iteration_id}",
            )
        )

    expected_variants = {
        event_name: Counter({"blocking-payload": 1})
        for event_name in _RANK_DIRECTIONS[rank]
    }
    for operation_id in expected_completed:
        operation = launched.get(operation_id)
        if operation is None:
            continue
        _batch_id, event_name, message_kind = operation
        expected_variants[event_name][f"batch-{message_kind}"] += 1
    for name, observed in variants.items():
        if observed != expected_variants[name]:
            failures.append(
                _failure(
                    "trace.bridge.variants",
                    (
                        f"{name} has variants {dict(observed)}, "
                        f"expected {dict(expected_variants[name])}"
                    ),
                    f"rank={rank} iteration={iteration_id}",
                )
            )

    for index, (launch, launch_end) in enumerate(launch_pairs):
        operations = launch.attrs.get("operations")
        operation_ids = (
            [
                operation.get("operation_id")
                for operation in operations
                if isinstance(operation, Mapping)
            ]
            if isinstance(operations, list)
            else []
        )
        operation_begins = [
            begin
            for operation_id in operation_ids
            for begin in completion_begins.get(operation_id, ())
        ]
        operation_ends = [
            end
            for operation_id in operation_ids
            for end in completion_ends.get(operation_id, ())
        ]
        next_launch = (
            launch_pairs[index + 1][0]
            if index + 1 < len(launch_pairs)
            else None
        )
        if any(begin.rel_ts <= launch_end.rel_ts for begin in operation_begins):
            failures.append(
                _failure(
                    "trace.bridge.launch_scope",
                    "a Work.wait began before its launch scope ended",
                    f"rank={rank} iteration={iteration_id}",
                )
            )
        if next_launch is not None and any(
            end.rel_ts >= next_launch.rel_ts for end in operation_ends
        ):
            failures.append(
                _failure(
                    "trace.bridge.launch_scope",
                    "the next launch began before the previous Work.wait scopes ended",
                    f"rank={rank} iteration={iteration_id}",
                )
            )

    broadcasts = _events(iteration, "bridge-grid-broadcast")
    failures.extend(
        _validate_scope_completions(
            iteration,
            rank=rank,
            event_names=("bridge-grid-broadcast",),
        )
    )
    direction = "backward" if rank == 0 else "forward"
    grid_side = "src" if rank == 0 else "dest"
    if len(broadcasts) != 8:
        failures.append(
            _failure(
                "trace.bridge.broadcast_count",
                f"rank {rank} has {len(broadcasts)} broadcast begin records",
                f"iteration={iteration_id}",
            )
        )
    broadcast_kinds = [event.attrs.get("message_kind") for event in broadcasts]
    if broadcast_kinds != ["shape", "payload"] * 4:
        failures.append(
            _failure(
                "trace.bridge.broadcast_order",
                f"rank {rank} has broadcast order {broadcast_kinds}",
                f"iteration={iteration_id}",
            )
        )
    for event in broadcasts:
        attrs = event.attrs
        failures.extend(
            _check_semantic_role(
                attrs,
                rank=rank,
                iteration=iteration_id,
                event_name=event.name,
            )
        )
        expected_bytes = (
            _SHAPE_BYTES
            if attrs.get("message_kind") == "shape"
            else _PAYLOAD_BYTES
        )
        if (
            attrs.get("pipeline_direction") != direction
            or attrs.get("grid_side") != grid_side
            or attrs.get("collective_role") != "source"
            or attrs.get("source_rank") != rank
            or attrs.get("src_module") != "encoder"
            or attrs.get("dest_module") != "llm"
            or attrs.get("transport_api") != "broadcast"
            or attrs.get("backend") != "nccl"
            or attrs.get("completion_kind") != "inline_api_return"
            or attrs.get("data_bytes") != expected_bytes
        ):
            failures.append(
                _failure(
                    "trace.bridge.broadcast",
                    f"rank {rank} has an invalid broadcast route or payload",
                    f"iteration={iteration_id}",
                )
            )
    return tuple(failures)


def validate_multimodule_bridge_trace(trace_root: Path) -> tuple[Failure, ...]:
    """Validate blocking, batch Work, shape/payload, and grid broadcast evidence."""

    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    if tuple(sorted(by_rank)) != (0, 1):
        failures.append(
            _failure(
                "trace.bridge.ranks",
                f"Bridge profile expects ranks [0, 1], observed {sorted(by_rank)}",
                "multimodule-bridge2",
            )
        )
    for rank in (0, 1):
        iterations = by_rank.get(rank, ())
        if tuple(iteration.iteration_id for iteration in iterations) != (1,):
            failures.append(
                _failure(
                    "trace.bridge.iterations",
                    f"rank {rank} does not contain exactly iteration 1",
                    f"rank={rank}",
                )
            )
        for iteration in iterations:
            failures.extend(_validate_iteration(iteration, rank=rank))
    return tuple(failures)


def _validate_asymmetric_bridge_run(
    run_root: Path,
    trace_enabled: bool,
    topology: _AsymmetricBridgeTopology,
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    for rank in range(8):
        path = run_root / f"training-result-rank-{rank}.json"
        if not path.is_file():
            failures.append(
                _failure(
                    "run.bridge.result_missing",
                    f"rank {rank} did not write a training result",
                    str(path),
                )
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "backward_completed": True,
            "completed": True,
            "global_rank": rank,
            "gradient_finite": True,
            "loss_finite": True,
            "module_role": "encoder" if rank < 4 else "llm",
            "optimizer_step": "not_run",
            "trace_enabled": trace_enabled,
            "world_size": 8,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                failures.append(
                    _failure(
                        "run.bridge.result",
                        (
                            f"rank {rank} has {field}={payload.get(field)!r}, "
                            f"expected {value!r}"
                        ),
                        str(path),
                    )
                )
        gradient_count = payload.get("gradient_count")
        gradient_norm = payload.get("gradient_norm")
        if (
            not isinstance(gradient_count, int)
            or gradient_count <= 0
            or not isinstance(gradient_norm, (int, float))
            or not math.isfinite(gradient_norm)
            or gradient_norm <= 0
        ):
            failures.append(
                _failure(
                    "run.bridge.gradient",
                    f"rank {rank} has invalid gradient terminal state",
                    str(path),
                )
            )
        expected_loss_count = 4 if rank in topology.loss_ranks else 0
        if payload.get("loss_count") != expected_loss_count:
            failures.append(
                _failure(
                    "run.bridge.loss",
                    (
                        f"rank {rank} has loss_count={payload.get('loss_count')!r}, "
                        f"expected {expected_loss_count}"
                    ),
                    str(path),
                )
            )
    return tuple(failures)


def _expected_asymmetric_launches(
    plan: _BridgeRankPlan,
) -> list[tuple[str, tuple[tuple[str, int], ...]]]:
    if plan.role == "sender":
        forward = "bridge-send-forward"
        backward = "bridge-recv-backward"
        payload_pair = (forward, backward)
    else:
        forward = "bridge-recv-forward"
        backward = "bridge-send-backward"
        payload_pair = (backward, forward)

    forward_operations = tuple((forward, peer) for peer in plan.peers)
    backward_operations = tuple((backward, peer) for peer in plan.peers)
    payload_operations = tuple(
        (event_name, peer)
        for peer in plan.peers
        for event_name in payload_pair
    )
    launches = [
        ("shape", forward_operations),
        ("shape", forward_operations),
    ]
    for _ in range(2):
        launches.extend(
            (
                ("shape", forward_operations + backward_operations),
                ("payload", payload_operations),
            )
        )
    launches.extend(
        (
            ("shape", backward_operations),
            ("shape", backward_operations),
        )
    )
    return launches


def _expected_blocking_routes(
    plan: _BridgeRankPlan,
) -> Counter[tuple[str, int, int]]:
    if plan.role == "sender":
        names = ("bridge-send-forward", "bridge-recv-backward")
    else:
        names = ("bridge-recv-forward", "bridge-send-backward")
    return Counter(
        (event_name, peer, plan.payload_bytes)
        for _ in range(2)
        for event_name in names
        for peer in plan.peers
    )


def _validate_asymmetric_launches(
    iteration: Iteration,
    *,
    rank: int,
    topology: _AsymmetricBridgeTopology,
) -> tuple[
    list[Failure],
    dict[str, tuple[str, str, int, int, str]],
    list[list[str]],
    list[tuple[int | float, int | float | None]],
]:
    iteration_id = int(iteration.iteration_id)
    failures: list[Failure] = []
    plan = topology.bridge_ranks.get(rank)
    expected_launches = _expected_asymmetric_launches(plan) if plan else []
    begins = _events(iteration, "bridge-p2p-launch")
    ends = _events(iteration, "bridge-p2p-launch", "E")
    if len(begins) != len(expected_launches) or len(ends) != len(expected_launches):
        failures.append(
            _failure(
                "trace.bridge.launch_count",
                (
                    f"rank {rank} has {len(begins)} launch begins and "
                    f"{len(ends)} launch ends, expected {len(expected_launches)}"
                ),
                f"{topology.name} iteration={iteration_id}",
            )
        )
    launch_phases = [
        event.ph
        for event in iteration.events
        if event.name == "bridge-p2p-launch"
    ]
    if launch_phases != ["B", "E"] * len(expected_launches):
        failures.append(
            _failure(
                "trace.bridge.launch_scope",
                f"rank {rank} has an invalid launch B/E sequence",
                f"{topology.name} iteration={iteration_id}",
            )
        )

    launched: dict[str, tuple[str, str, int, int, str]] = {}
    launch_operation_ids: list[list[str]] = []
    batch_ids: set[str] = set()
    for launch, (message_kind, expected_operations) in zip(
        begins, expected_launches
    ):
        attrs = launch.attrs
        batch_id = attrs.get("batch_id")
        operations = attrs.get("operations")
        if (
            attrs.get("transport_api") != "batch_isend_irecv"
            or attrs.get("completion_mode") != "internal_wait"
            or attrs.get("src_module") != "encoder"
            or attrs.get("dest_module") != "llm"
            or attrs.get("message_kind") != message_kind
            or not isinstance(batch_id, str)
            or not batch_id
            or batch_id in batch_ids
            or not isinstance(operations, list)
            or attrs.get("operation_count") != len(operations)
            or len(operations) != len(expected_operations)
        ):
            failures.append(
                _failure(
                    "trace.bridge.launch",
                    "bridge-p2p-launch has an invalid operation list or route",
                    f"rank={rank} iteration={iteration_id}",
                )
            )
            launch_operation_ids.append([])
            continue
        batch_ids.add(batch_id)
        operation_ids: list[str] = []
        for index, (operation, (event_name, peer_rank)) in enumerate(
            zip(operations, expected_operations)
        ):
            if not isinstance(operation, Mapping):
                failures.append(
                    _failure(
                        "trace.bridge.operation",
                        f"{batch_id} contains a non-mapping operation",
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
                continue
            direction, pipeline_direction = {
                "bridge-send-forward": ("send", "forward"),
                "bridge-recv-forward": ("recv", "forward"),
                "bridge-send-backward": ("send", "backward"),
                "bridge-recv-backward": ("recv", "backward"),
            }[event_name]
            operation_id = operation.get("operation_id")
            expected_bytes = (
                _SHAPE_BYTES if message_kind == "shape" else plan.payload_bytes
            )
            if (
                operation_id != f"{batch_id}:{index}"
                or operation_id in launched
                or operation.get("request_id") != operation_id
                or operation.get("backend") != "nccl"
                or operation.get("transport_api") != "batch_isend_irecv"
                or operation.get("completion_mode") != "internal_wait"
                or operation.get("direction") != direction
                or operation.get("pipeline_direction") != pipeline_direction
                or operation.get("peer_rank") != peer_rank
                or operation.get("message_kind") != message_kind
                or operation.get("data_bytes") != expected_bytes
            ):
                failures.append(
                    _failure(
                        "trace.bridge.operation",
                        f"{batch_id} contains an invalid operation {operation!r}",
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
                continue
            failures.extend(
                _check_semantic_role(
                    operation,
                    rank=rank,
                    iteration=iteration_id,
                    event_name="bridge-p2p-launch operation",
                )
            )
            launched[operation_id] = (
                batch_id,
                event_name,
                peer_rank,
                expected_bytes,
                message_kind,
            )
            operation_ids.append(operation_id)
        launch_operation_ids.append(operation_ids)
    launch_windows = [
        (
            ends[index].rel_ts,
            begins[index + 1].rel_ts if index + 1 < len(begins) else None,
        )
        for index in range(min(len(begins), len(ends)))
    ]
    return failures, launched, launch_operation_ids, launch_windows


def _validate_asymmetric_directions(
    iteration: Iteration,
    *,
    rank: int,
    topology: _AsymmetricBridgeTopology,
    launched: Mapping[str, tuple[str, str, int, int, str]],
    launch_operation_ids: Sequence[Sequence[str]],
    launch_windows: Sequence[tuple[int | float, int | float | None]],
) -> list[Failure]:
    iteration_id = int(iteration.iteration_id)
    failures: list[Failure] = []
    failures.extend(
        _validate_scope_completions(
            iteration,
            rank=rank,
            event_names=tuple(_DIRECTIONAL_EVENTS),
        )
    )
    plan = topology.bridge_ranks.get(rank)
    expected_blocking = (
        _expected_blocking_routes(plan) if plan else Counter()
    )
    observed_blocking: Counter[tuple[str, int, int]] = Counter()
    completion_counts: Counter[str] = Counter()
    completion_order: list[str] = []
    completion_scopes: dict[str, tuple[int | float, int | float]] = {}
    for event_name in _DIRECTIONAL_EVENTS:
        begins = _events(iteration, event_name)
        ends = _events(iteration, event_name, "E")
        for begin, end in zip(begins, ends):
            operation_id = begin.attrs.get("operation_id")
            if (
                begin.attrs.get("transport_api") == "batch_isend_irecv"
                and isinstance(operation_id, str)
            ):
                completion_scopes[operation_id] = (begin.rel_ts, end.rel_ts)

    for event in (
        event
        for event in iteration.events
        if event.name in _DIRECTIONAL_EVENTS and event.ph == "B"
    ):
        attrs = event.attrs
        expected_event_name = _DIRECTION_TO_EVENT.get(
            (attrs.get("direction"), attrs.get("pipeline_direction"))
        )
        if (
            event.name != expected_event_name
            or attrs.get("src_module") != "encoder"
            or attrs.get("dest_module") != "llm"
            or attrs.get("backend") != "nccl"
        ):
            failures.append(
                _failure(
                    "trace.bridge.route",
                    f"{event.name} has an invalid module or direction route",
                    f"rank={rank} iteration={iteration_id}",
                )
            )
        failures.extend(
            _check_semantic_role(
                attrs,
                rank=rank,
                iteration=iteration_id,
                event_name=event.name,
            )
        )
        transport = attrs.get("transport_api")
        if transport == "send_recv":
            route = (
                event.name,
                attrs.get("peer_rank"),
                attrs.get("data_bytes"),
            )
            observed_blocking[route] += 1
            if (
                attrs.get("message_kind") != "payload"
                or attrs.get("completion_kind") != "inline_api_return"
            ):
                failures.append(
                    _failure(
                        "trace.bridge.completion",
                        f"{event.name} blocking completion is invalid",
                        f"rank={rank} iteration={iteration_id}",
                    )
                )
            continue
        if transport != "batch_isend_irecv":
            failures.append(
                _failure(
                    "trace.bridge.transport",
                    f"{event.name} has unsupported transport {transport!r}",
                    f"rank={rank} iteration={iteration_id}",
                )
            )
            continue
        operation_id = attrs.get("operation_id")
        expected = launched.get(operation_id)
        observed = (
            attrs.get("batch_id"),
            event.name,
            attrs.get("peer_rank"),
            attrs.get("data_bytes"),
            attrs.get("message_kind"),
        )
        if (
            expected is None
            or expected != observed
            or attrs.get("request_id") != operation_id
            or attrs.get("completion_kind") != "work_wait"
        ):
            failures.append(
                _failure(
                    "trace.bridge.identity",
                    f"{event.name} cannot be paired with launch operation {operation_id!r}",
                    f"rank={rank} iteration={iteration_id}",
                )
            )
        if isinstance(operation_id, str):
            completion_counts[operation_id] += 1
            completion_order.append(operation_id)

    if observed_blocking != expected_blocking:
        failures.append(
            _failure(
                "trace.bridge.blocking_route",
                (
                    f"rank {rank} has blocking routes "
                    f"{dict(observed_blocking)}, expected {dict(expected_blocking)}"
                ),
                f"{topology.name} iteration={iteration_id}",
            )
        )

    expected_completion_order: list[str] = []
    for index, operation_ids in enumerate(launch_operation_ids):
        counts = [completion_counts[operation_id] for operation_id in operation_ids]
        if len(operation_ids) == 1:
            valid = counts == [1]
        else:
            valid = bool(operation_ids) and (
                all(count == 1 for count in counts)
                or all(count == 0 for count in counts)
            )
        if not valid:
            failures.append(
                _failure(
                    "trace.bridge.identity",
                    "Bridge launch has an invalid positional/coalesced completion set",
                    f"rank={rank} iteration={iteration_id}",
                )
            )
        if counts and all(count == 1 for count in counts):
            expected_completion_order.extend(operation_ids)
            if index < len(launch_windows):
                launch_end, next_launch_begin = launch_windows[index]
                for operation_id in operation_ids:
                    completion_scope = completion_scopes.get(operation_id)
                    if completion_scope is None:
                        continue
                    completion_begin, completion_end = completion_scope
                    if completion_begin <= launch_end or (
                        next_launch_begin is not None
                        and completion_end >= next_launch_begin
                    ):
                        failures.append(
                            _failure(
                                "trace.bridge.launch_scope",
                                (
                                    "Bridge Work.wait is outside the interval "
                                    "after its launch and before the next launch"
                                ),
                                f"rank={rank} iteration={iteration_id}",
                            )
                        )
    if completion_order != expected_completion_order:
        failures.append(
            _failure(
                "trace.bridge.completion_order",
                f"rank {rank} Work.wait records do not follow launch position order",
                f"iteration={iteration_id}",
            )
        )
    return failures


def _validate_asymmetric_broadcasts(
    iteration: Iteration,
    *,
    rank: int,
    topology: _AsymmetricBridgeTopology,
) -> list[Failure]:
    iteration_id = int(iteration.iteration_id)
    failures = _validate_scope_completions(
        iteration,
        rank=rank,
        event_names=("bridge-grid-broadcast",),
    )
    plan = topology.broadcast_ranks.get(rank)
    broadcasts = _events(iteration, "bridge-grid-broadcast")
    expected_count = 8 if plan else 0
    if len(broadcasts) != expected_count:
        failures.append(
            _failure(
                "trace.bridge.broadcast_count",
                (
                    f"rank {rank} has {len(broadcasts)} broadcast begin records, "
                    f"expected {expected_count}"
                ),
                f"{topology.name} iteration={iteration_id}",
            )
        )
    if plan is None:
        return failures
    kinds = [event.attrs.get("message_kind") for event in broadcasts]
    if kinds != ["shape", "payload"] * 4:
        failures.append(
            _failure(
                "trace.bridge.broadcast_order",
                f"rank {rank} has broadcast order {kinds}",
                f"iteration={iteration_id}",
            )
        )
    for event in broadcasts:
        attrs = event.attrs
        failures.extend(
            _check_semantic_role(
                attrs,
                rank=rank,
                iteration=iteration_id,
                event_name=event.name,
            )
        )
        expected_bytes = (
            _SHAPE_BYTES
            if attrs.get("message_kind") == "shape"
            else plan.payload_bytes
        )
        if (
            attrs.get("pipeline_direction") != plan.pipeline_direction
            or attrs.get("grid_side") != plan.grid_side
            or attrs.get("collective_role") != plan.collective_role
            or attrs.get("source_rank") != plan.source_rank
            or attrs.get("src_module") != "encoder"
            or attrs.get("dest_module") != "llm"
            or attrs.get("transport_api") != "broadcast"
            or attrs.get("backend") != "nccl"
            or attrs.get("completion_kind") != "inline_api_return"
            or attrs.get("data_bytes") != expected_bytes
        ):
            failures.append(
                _failure(
                    "trace.bridge.broadcast",
                    f"rank {rank} has an invalid broadcast route or payload",
                    f"iteration={iteration_id}",
                )
            )
    return failures


def _expected_asymmetric_major_sequence(
    bridge_plan: _BridgeRankPlan | None,
    broadcast_plan: _BridgeBroadcastPlan | None,
) -> list[tuple[str, str, int | None]]:
    sequence: list[tuple[str, str, int | None]] = []

    def launch(message_kind: str) -> None:
        sequence.append(("bridge-p2p-launch", message_kind, None))

    def blocking(event_name: str) -> None:
        assert bridge_plan is not None
        sequence.extend(
            (event_name, "payload", peer) for peer in bridge_plan.peers
        )

    def broadcast_pair() -> None:
        sequence.extend(
            (
                ("bridge-grid-broadcast", "shape", None),
                ("bridge-grid-broadcast", "payload", None),
            )
        )

    if bridge_plan is None:
        if broadcast_plan is not None:
            for _ in range(4):
                broadcast_pair()
        return sequence

    if bridge_plan.role == "sender":
        for _ in range(2):
            launch("shape")
            blocking("bridge-send-forward")
        for _ in range(2):
            launch("shape")
            launch("payload")
            broadcast_pair()
        for _ in range(2):
            launch("shape")
            blocking("bridge-recv-backward")
            broadcast_pair()
    else:
        for _ in range(2):
            launch("shape")
            blocking("bridge-recv-forward")
            broadcast_pair()
        for _ in range(2):
            launch("shape")
            launch("payload")
            broadcast_pair()
        for _ in range(2):
            launch("shape")
            blocking("bridge-send-backward")
    return sequence


def _validate_asymmetric_major_sequence(
    iteration: Iteration,
    *,
    rank: int,
    topology: _AsymmetricBridgeTopology,
) -> list[Failure]:
    observed: list[tuple[str, str, int | None]] = []
    for event in iteration.events:
        if event.ph != "B":
            continue
        if event.name == "bridge-p2p-launch":
            observed.append(
                (event.name, str(event.attrs.get("message_kind")), None)
            )
        elif (
            event.name in _DIRECTIONAL_EVENTS
            and event.attrs.get("transport_api") == "send_recv"
        ):
            observed.append(
                (
                    event.name,
                    str(event.attrs.get("message_kind")),
                    event.attrs.get("peer_rank"),
                )
            )
        elif event.name == "bridge-grid-broadcast":
            observed.append(
                (event.name, str(event.attrs.get("message_kind")), None)
            )
    expected = _expected_asymmetric_major_sequence(
        topology.bridge_ranks.get(rank),
        topology.broadcast_ranks.get(rank),
    )
    if observed == expected:
        return []
    return [
        _failure(
            "trace.bridge.schedule_order",
            f"rank {rank} has an invalid PP2 Bridge lifecycle order",
            f"{topology.name} iteration={iteration.iteration_id}",
        )
    ]


def _validate_asymmetric_bridge_trace(
    trace_root: Path,
    topology: _AsymmetricBridgeTopology,
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    by_rank = _load_iterations(trace_root)
    if tuple(sorted(by_rank)) != tuple(range(8)):
        failures.append(
            _failure(
                "trace.bridge.ranks",
                f"Bridge profile expects ranks 0..7, observed {sorted(by_rank)}",
                topology.name,
            )
        )
    for rank in range(8):
        iterations = by_rank.get(rank, ())
        if tuple(iteration.iteration_id for iteration in iterations) != (1,):
            failures.append(
                _failure(
                    "trace.bridge.iterations",
                    f"rank {rank} does not contain exactly iteration 1",
                    f"{topology.name} rank={rank}",
                )
            )
        for iteration in iterations:
            (
                launch_failures,
                launched,
                launch_operation_ids,
                launch_windows,
            ) = (
                _validate_asymmetric_launches(
                    iteration,
                    rank=rank,
                    topology=topology,
                )
            )
            failures.extend(launch_failures)
            failures.extend(
                _validate_asymmetric_directions(
                    iteration,
                    rank=rank,
                    topology=topology,
                    launched=launched,
                    launch_operation_ids=launch_operation_ids,
                    launch_windows=launch_windows,
                )
            )
            failures.extend(
                _validate_asymmetric_broadcasts(
                    iteration,
                    rank=rank,
                    topology=topology,
                )
            )
            failures.extend(
                _validate_asymmetric_major_sequence(
                    iteration,
                    rank=rank,
                    topology=topology,
                )
            )
    return tuple(failures)


def validate_multimodule_bridge_fanin_run(
    run_root: Path,
    trace_enabled: bool,
) -> tuple[Failure, ...]:
    return _validate_asymmetric_bridge_run(
        run_root,
        trace_enabled,
        _FANIN_TOPOLOGY,
    )


def validate_multimodule_bridge_fanout_run(
    run_root: Path,
    trace_enabled: bool,
) -> tuple[Failure, ...]:
    return _validate_asymmetric_bridge_run(
        run_root,
        trace_enabled,
        _FANOUT_TOPOLOGY,
    )


def validate_multimodule_bridge_fanin_trace(
    trace_root: Path,
) -> tuple[Failure, ...]:
    return _validate_asymmetric_bridge_trace(trace_root, _FANIN_TOPOLOGY)


def validate_multimodule_bridge_fanout_trace(
    trace_root: Path,
) -> tuple[Failure, ...]:
    return _validate_asymmetric_bridge_trace(trace_root, _FANOUT_TOPOLOGY)
