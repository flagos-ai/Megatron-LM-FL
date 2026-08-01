# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Trace contract for the controlled PP2/DP2/EP2 DualPipeV profile."""

from __future__ import annotations

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

_RANKS = (0, 1, 2, 3)
_ITERATIONS = (1, 2)
_SCHEDULE_EVENTS = frozenset(("forward-step", "backward-step", "combined-forward-backward-step"))
_DIRECTION_EVENTS = frozenset(("send-forward", "recv-forward", "send-backward", "recv-backward"))
_A2A_LAUNCH = "ep-alltoall-async-launch"
_A2A_COMPLETE = "ep-alltoall-async-complete"

_FORWARD_PAYLOADS = frozenset(
    (
        ("forward", "dispatch", "token_hidden_states"),
        ("forward", "dispatch", "routing_probabilities"),
        ("forward", "combine", "expert_output"),
    )
)
_BACKWARD_PAYLOADS = frozenset(
    (
        ("backward", "combine", "expert_output_gradient"),
        ("backward", "dispatch", "token_hidden_states_gradient"),
        ("backward", "dispatch", "routing_probabilities_gradient"),
    )
)
_TERMINAL_SITES = {
    ("forward", "dispatch", "token_hidden_states"): ("forward_dispatch_hidden_ready"),
    ("forward", "dispatch", "routing_probabilities"): ("forward_dispatch_probabilities_ready"),
    ("forward", "combine", "expert_output"): "forward_combine_output_ready",
    ("backward", "combine", "expert_output_gradient"): ("backward_combine_gradient_ready"),
    ("backward", "dispatch", "token_hidden_states_gradient"): (
        "backward_dispatch_hidden_gradient_ready"
    ),
    ("backward", "dispatch", "routing_probabilities_gradient"): (
        "backward_dispatch_probability_gradient_ready"
    ),
}
_DEPENDENCY_SITE = "comm_stream_dependency_before_forward_dispatch_probabilities"
_A2A_IDENTITY_FIELDS = (
    "operation_id",
    "request_id",
    "layer",
    "comm_type",
    "dispatcher",
    "data_bytes",
    "group_size",
    "ep_size",
    "tp_size",
    "execution_route",
    "pass_direction",
    "logical_phase",
    "payload_role",
    "transport_api",
)

# (event, forward identity, backward identity, schedule phase)
_SCHEDULES = {
    0: (
        ("forward-step", (0, 0), None, "warmup"),
        ("forward-step", (1, 0), None, "warmup"),
        ("forward-step", (2, 0), None, "warmup"),
        ("forward-step", (4, 1), None, "warmup"),
        ("backward-step", None, (4, 1), "steady"),
        ("forward-step", (5, 1), None, "steady"),
        ("combined-forward-backward-step", (3, 0), (5, 1), "steady"),
        ("combined-forward-backward-step", (6, 1), (0, 0), "steady"),
        ("backward-step", None, (6, 1), "steady"),
        ("combined-forward-backward-step", (7, 1), (1, 0), "steady"),
        ("backward-step", None, (7, 1), "steady"),
        ("backward-step", None, (2, 0), "cooldown"),
        ("backward-step", None, (3, 0), "cooldown"),
    ),
    1: (
        ("forward-step", (0, 0), None, "warmup"),
        ("forward-step", (4, 1), None, "warmup"),
        ("forward-step", (1, 0), None, "warmup"),
        ("forward-step", (5, 1), None, "warmup"),
        ("forward-step", (2, 0), None, "steady"),
        ("backward-step", None, (4, 1), "steady"),
        ("combined-forward-backward-step", (6, 1), (0, 0), "steady"),
        ("combined-forward-backward-step", (3, 0), (5, 1), "steady"),
        ("combined-forward-backward-step", (7, 1), (1, 0), "steady"),
        ("backward-step", None, (6, 1), "steady"),
        ("backward-step", None, (2, 0), "steady"),
        ("backward-step", None, (7, 1), "cooldown"),
        ("backward-step", None, (3, 0), "cooldown"),
    ),
}

_P2P_GROUPS = {
    0: (
        (("send-forward",), "internal_wait"),
        (("send-forward",), "external_wait"),
        (("recv-backward",), "internal_wait"),
        (("send-forward",), "external_wait"),
        (("send-forward",), "internal_wait"),
        (("recv-backward",), "internal_wait"),
        (("send-forward", "recv-backward"), "external_wait"),
        (("send-forward", "recv-backward"), "external_wait"),
        (("recv-backward",), "external_wait"),
        (("send-forward", "recv-backward"), "internal_wait"),
        (("send-forward", "recv-backward"), "internal_wait"),
        (("recv-backward",), "external_wait"),
    ),
    1: (
        (("recv-forward",), "internal_wait"),
        (("recv-forward",), "external_wait"),
        (("send-backward",), "external_wait"),
        (("recv-forward",), "external_wait"),
        (("recv-forward",), "external_wait"),
        (("send-backward",), "external_wait"),
        (("send-backward", "recv-forward"), "external_wait"),
        (("send-backward", "recv-forward"), "external_wait"),
        (("send-backward",), "external_wait"),
        (("send-backward", "recv-forward"), "external_wait"),
        (("send-backward", "recv-forward"), "internal_wait"),
        (("send-backward",), "external_wait"),
    ),
}


@dataclass(frozen=True)
class _Scope:
    name: str
    begin: int
    end: int
    event: Event
    forward: tuple[int, int] | None
    backward: tuple[int, int] | None


def _failure(code: str, message: str, rank: int, iteration: int) -> Failure:
    return Failure(f"trace.dualpipev.{code}", message, f"rank={rank} iteration={iteration}")


def _load_iterations(trace_root: Path) -> Mapping[int, Sequence[Iteration]]:
    result: dict[int, Sequence[Iteration]] = {}
    for rank, content in collect_benchmark_files(trace_root):
        if rank.global_rank is None or rank.global_rank in result:
            raise ValueError("DualPipeV contract requires one shard per global rank")
        result[rank.global_rank] = tuple(read_benchmark_file(rank, content))
    return result


def _events(iteration: Iteration, name: str, phase: str) -> list[Event]:
    return [event for event in iteration.events if event.name == name and event.ph == phase]


def _expect_fields(
    event: Event, expected: Mapping[str, Any], *, rank: int, iteration: int
) -> list[Failure]:
    return [
        _failure(
            "field",
            f"{event.name!r} has {field}={event.attrs.get(field, '<missing>')!r}, "
            f"expected {value!r}",
            rank,
            iteration,
        )
        for field, value in expected.items()
        if event.attrs.get(field, "<missing>") != value
    ]


def _identity(attrs: Mapping[str, Any], prefix: str = "") -> tuple[int, int] | None:
    microbatch = attrs.get(f"{prefix}microbatch")
    stage_key = "dualpipev_stage" if prefix == "current_" else f"{prefix}dualpipev_stage"
    stage = attrs.get(stage_key)
    if (
        not isinstance(microbatch, int)
        or isinstance(microbatch, bool)
        or not isinstance(stage, int)
        or isinstance(stage, bool)
    ):
        return None
    return microbatch, stage


def _operation_id(identity: tuple[int, int]) -> str:
    microbatch, stage = identity
    return f"pp:microbatch={microbatch}:dualpipev_stage={stage}"


def _schedule_scopes(
    iteration: Iteration, rank: int, iteration_id: int
) -> tuple[list[_Scope], list[Failure]]:
    scopes: list[_Scope] = []
    failures: list[Failure] = []
    active: tuple[int, Event] | None = None
    for position, event in enumerate(iteration.events):
        if event.name not in _SCHEDULE_EVENTS:
            continue
        if event.ph == "B":
            if active is not None:
                failures.append(
                    _failure("schedule_pairing", "schedule scopes overlap", rank, iteration_id)
                )
            active = (position, event)
            continue
        if event.ph != "E" or active is None or active[1].name != event.name:
            failures.append(
                _failure(
                    "schedule_pairing",
                    f"unmatched schedule boundary {event.name!r}/{event.ph!r}",
                    rank,
                    iteration_id,
                )
            )
            continue
        begin, begin_event = active
        attrs = begin_event.attrs
        if begin_event.name == "forward-step":
            forward = _identity(attrs, "current_")
            backward = None
        elif begin_event.name == "backward-step":
            forward = None
            backward = _identity(attrs, "current_")
        else:
            forward = _identity(attrs, "forward_")
            backward = _identity(attrs, "backward_")
        scopes.append(_Scope(begin_event.name, begin, position, begin_event, forward, backward))
        active = None
    if active is not None:
        failures.append(
            _failure("schedule_pairing", "schedule begin has no end", rank, iteration_id)
        )
    return scopes, failures


def _validate_schedule(
    iteration: Iteration, scopes: Sequence[_Scope], *, rank: int, pp_rank: int, iteration_id: int
) -> list[Failure]:
    failures: list[Failure] = []
    observed = tuple(
        (scope.name, scope.forward, scope.backward, scope.event.attrs.get("schedule_phase"))
        for scope in scopes
    )
    expected = _SCHEDULES[pp_rank]
    if observed != expected:
        failures.append(
            _failure("schedule", f"schedule differs: observed={observed!r}", rank, iteration_id)
        )

    forward = [identity for scope in scopes if (identity := scope.forward) is not None]
    backward = [identity for scope in scopes if (identity := scope.backward) is not None]
    expected_identities = {(microbatch, 0) for microbatch in range(4)} | {
        (microbatch, 1) for microbatch in range(4, 8)
    }
    for direction, identities in (("forward", forward), ("backward", backward)):
        if len(identities) != 8 or set(identities) != expected_identities:
            failures.append(
                _failure(
                    "coverage", f"{direction} identities differ: {identities!r}", rank, iteration_id
                )
            )

    for scope in scopes:
        attrs = scope.event.attrs
        failures.extend(
            _expect_fields(
                scope.event,
                {
                    "schedule": "dualpipev",
                    "schedule_phase": attrs.get("schedule_phase"),
                    "uses_model_graph": True,
                    "timing_phase": "framework_phase",
                },
                rank=rank,
                iteration=iteration_id,
            )
        )
        if attrs.get("vp_stage", None) is not None:
            failures.append(_failure("field", "vp_stage must be None", rank, iteration_id))
        if scope.name == "combined-forward-backward-step":
            if scope.forward is None or scope.backward is None:
                failures.append(
                    _failure(
                        "identity",
                        "combined scope lacks a forward or backward identity",
                        rank,
                        iteration_id,
                    )
                )
                continue
            forward_id = _operation_id(scope.forward)
            backward_id = _operation_id(scope.backward)
            failures.extend(
                _expect_fields(
                    scope.event,
                    {
                        "forward_operation_id": forward_id,
                        "backward_operation_id": backward_id,
                        "operation_id": (
                            f"pp-combined:forward={forward_id}:backward={backward_id}"
                        ),
                        "execution_mode": "combined",
                        "overlap_active": True,
                    },
                    rank=rank,
                    iteration=iteration_id,
                )
            )
        else:
            identity = scope.forward or scope.backward
            if identity is None:
                failures.append(
                    _failure(
                        "identity",
                        f"{scope.name!r} lacks a microbatch/stage identity",
                        rank,
                        iteration_id,
                    )
                )
                continue
            failures.extend(
                _expect_fields(
                    scope.event,
                    {"operation_id": _operation_id(identity)},
                    rank=rank,
                    iteration=iteration_id,
                )
            )
    return failures


def _payload_key(event: Event) -> tuple[Any, Any, Any]:
    return (
        event.attrs.get("pass_direction"),
        event.attrs.get("logical_phase"),
        event.attrs.get("payload_role"),
    )


def _validate_a2a_scope(
    iteration: Iteration, scope: _Scope, *, rank: int, pp_rank: int, iteration_id: int
) -> list[Failure]:
    failures: list[Failure] = []
    rows = list(enumerate(iteration.events[scope.begin + 1 : scope.end], scope.begin + 1))
    launches = [
        (position, event)
        for position, event in rows
        if event.name == _A2A_LAUNCH and event.ph == "B"
    ]
    completions = [
        (position, event)
        for position, event in rows
        if event.name == _A2A_COMPLETE and event.ph == "B"
    ]
    expected_payloads = set()
    if scope.forward is not None:
        expected_payloads.update(_FORWARD_PAYLOADS)
    if scope.backward is not None:
        expected_payloads.update(_BACKWARD_PAYLOADS)
    observed_payloads = [_payload_key(event) for _, event in launches]
    if len(launches) != len(expected_payloads) or set(observed_payloads) != expected_payloads:
        failures.append(
            _failure(
                "a2a_payloads",
                f"{scope.name!r} launch payloads differ: {observed_payloads!r}",
                rank,
                iteration_id,
            )
        )

    launch_by_id: dict[str, tuple[int, Event]] = {}
    for position, launch in launches:
        operation_id = launch.attrs.get("operation_id")
        stage = (
            scope.forward[1]
            if launch.attrs.get("pass_direction") == "forward"
            else (scope.backward[1] if scope.backward is not None else None)
        )
        expected_layer = {0: (1, 4), 1: (2, 3)}[pp_rank][int(stage)] if stage in (0, 1) else None
        failures.extend(
            _expect_fields(
                launch,
                {
                    "request_id": operation_id,
                    "operation_id_scope": "rank_local",
                    "layer": expected_layer,
                    "comm_type": "ep-alltoall",
                    "dispatcher": "alltoall",
                    "group_size": 2,
                    "ep_size": 2,
                    "tp_size": 1,
                    "execution_route": "dualpipev_fb_overlap",
                    "transport_api": "all_to_all_single",
                    "async_op": True,
                    "completion_included": False,
                    "stage": "ep_request_launch",
                    "timing_phase": "async_dispatch",
                },
                rank=rank,
                iteration=iteration_id,
            )
        )
        if not isinstance(launch.attrs.get("data_bytes"), int) or launch.attrs["data_bytes"] <= 0:
            failures.append(
                _failure("a2a_bytes", "A2A data_bytes must be positive", rank, iteration_id)
            )
        if not isinstance(operation_id, str) or operation_id in launch_by_id:
            failures.append(
                _failure(
                    "a2a_operation",
                    f"invalid A2A operation_id={operation_id!r}",
                    rank,
                    iteration_id,
                )
            )
        else:
            launch_by_id[operation_id] = (position, launch)

    completions_by_id: dict[str, list[tuple[int, Event]]] = {}
    for position, completion in completions:
        operation_id = completion.attrs.get("operation_id")
        if isinstance(operation_id, str):
            completions_by_id.setdefault(operation_id, []).append((position, completion))
        launch = launch_by_id.get(str(operation_id))
        if launch is None:
            failures.append(
                _failure(
                    "a2a_completion",
                    f"completion has no launch: {operation_id!r}",
                    rank,
                    iteration_id,
                )
            )
            continue
        _, launch_event = launch
        for field in _A2A_IDENTITY_FIELDS:
            if completion.attrs.get(field, "<missing>") != launch_event.attrs.get(
                field, "<missing>"
            ):
                failures.append(
                    _failure(
                        "a2a_completion_metadata",
                        f"completion differs from launch for {field!r}",
                        rank,
                        iteration_id,
                    )
                )
        failures.extend(
            _expect_fields(
                completion,
                {
                    "completion_guarantee": "current_stream_after_wait",
                    "completion_included": True,
                    "completion_kind": "work_wait",
                    "duration_attribution": "per_request",
                    "host_blocking_guaranteed": False,
                    "stage": "ep_request_completion",
                    "timing_phase": "stream_dependency",
                },
                rank=rank,
                iteration=iteration_id,
            )
        )

    dependency_operation: str | None = None
    for operation_id, (launch_position, launch) in launch_by_id.items():
        operation_completions = completions_by_id.get(operation_id, [])
        terminal = [
            item for item in operation_completions if item[1].attrs.get("wait_role") == "terminal"
        ]
        dependency = [
            item for item in operation_completions if item[1].attrs.get("wait_role") == "dependency"
        ]
        key = _payload_key(launch)
        if len(terminal) != 1:
            failures.append(
                _failure(
                    "a2a_terminal",
                    f"{operation_id!r} has {len(terminal)} terminal waits",
                    rank,
                    iteration_id,
                )
            )
        else:
            terminal_position, terminal_event = terminal[0]
            failures.extend(
                _expect_fields(
                    terminal_event,
                    {"terminal": True, "completion_site": _TERMINAL_SITES.get(key)},
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            if not launch_position < terminal_position:
                failures.append(
                    _failure("a2a_order", "terminal wait precedes launch", rank, iteration_id)
                )

        expects_dependency = scope.name == "combined-forward-backward-step" and key == (
            "backward",
            "combine",
            "expert_output_gradient",
        )
        if len(dependency) != int(expects_dependency):
            failures.append(
                _failure(
                    "a2a_dependency",
                    f"{operation_id!r} has {len(dependency)} dependency waits",
                    rank,
                    iteration_id,
                )
            )
        if dependency:
            dependency_operation = operation_id
            dependency_position, dependency_event = dependency[0]
            failures.extend(
                _expect_fields(
                    dependency_event,
                    {"terminal": False, "completion_site": _DEPENDENCY_SITE},
                    rank=rank,
                    iteration=iteration_id,
                )
            )
            routing_positions = [
                position
                for position, candidate in launches
                if _payload_key(candidate) == ("forward", "dispatch", "routing_probabilities")
            ]
            terminal_position = terminal[0][0] if terminal else -1
            if len(routing_positions) != 1 or not (
                launch_position < dependency_position < routing_positions[0] < terminal_position
            ):
                failures.append(
                    _failure(
                        "a2a_dependency_order",
                        "combined dependency/routing/terminal order differs",
                        rank,
                        iteration_id,
                    )
                )
    if scope.name == "combined-forward-backward-step" and dependency_operation is None:
        failures.append(
            _failure("a2a_dependency", "combined scope has no dependency wait", rank, iteration_id)
        )

    launch_ends = sum(event.name == _A2A_LAUNCH and event.ph == "E" for _, event in rows)
    completion_ends = [
        event for _, event in rows if event.name == _A2A_COMPLETE and event.ph == "E"
    ]
    if launch_ends != len(launches) or len(completion_ends) != len(completions):
        failures.append(_failure("a2a_pairing", "A2A B/E counts differ", rank, iteration_id))
    if any(
        event.attrs.get("completed") is not True or event.attrs.get("error_type") is not None
        for event in completion_ends
    ):
        failures.append(_failure("a2a_completion", "A2A completion failed", rank, iteration_id))
    return failures


def _operation_event_name(operation: Mapping[str, Any]) -> str:
    return f"{operation.get('direction')}-{operation.get('pipeline_direction')}"


def _validate_p2p(
    iteration: Iteration, *, rank: int, pp_rank: int, iteration_id: int
) -> list[Failure]:
    failures: list[Failure] = []
    launches = _events(iteration, "p2p-launch", "B")
    expected_groups = _P2P_GROUPS[pp_rank]
    observed_groups: list[tuple[tuple[str, ...], Any]] = []
    launched: dict[str, tuple[str, str, str, int, int]] = {}
    for launch in launches:
        operations = launch.attrs.get("operations")
        mode = launch.attrs.get("completion_mode")
        if not isinstance(operations, list):
            failures.append(
                _failure("p2p_launch", "launch operations is invalid", rank, iteration_id)
            )
            continue
        names = tuple(
            _operation_event_name(operation)
            for operation in operations
            if isinstance(operation, Mapping)
        )
        observed_groups.append((names, mode))
        failures.extend(
            _expect_fields(
                launch,
                {
                    "transport_api": "isend_irecv",
                    "request_pairing": "key",
                    "completion_included": False,
                    "operation_count": len(operations),
                },
                rank=rank,
                iteration=iteration_id,
            )
        )
        batch_id = launch.attrs.get("batch_id")
        for operation in operations:
            if not isinstance(operation, Mapping):
                continue
            operation_id = operation.get("operation_id")
            event_name = _operation_event_name(operation)
            peer = rank + 2 if pp_rank == 0 else rank - 2
            valid = (
                isinstance(operation_id, str)
                and operation.get("request_id") == operation_id
                and event_name in _DIRECTION_EVENTS
                and operation.get("peer_rank") == peer
                and isinstance(operation.get("data_bytes"), int)
                and operation["data_bytes"] > 0
                and operation.get("microbatch") is None
                and operation.get("transport_api") == "isend_irecv"
                and operation.get("completion_mode") == mode
                and isinstance(batch_id, str)
            )
            if not valid or operation_id in launched:
                failures.append(
                    _failure(
                        "p2p_operation", f"invalid P2P operation {operation!r}", rank, iteration_id
                    )
                )
            else:
                launched[operation_id] = (
                    event_name,
                    str(batch_id),
                    str(mode),
                    peer,
                    int(operation["data_bytes"]),
                )
    if tuple(observed_groups) != expected_groups:
        failures.append(
            _failure("p2p_groups", f"P2P groups differ: {observed_groups!r}", rank, iteration_id)
        )
    if len(launches) != 12 or len(launched) != 16:
        failures.append(
            _failure(
                "p2p_count",
                f"P2P launch/operation count is {len(launches)}/{len(launched)}",
                rank,
                iteration_id,
            )
        )
    if len(_events(iteration, "p2p-launch", "E")) != len(launches):
        failures.append(_failure("p2p_pairing", "P2P launch B/E counts differ", rank, iteration_id))

    completion_begins = [
        event for event in iteration.events if event.name in _DIRECTION_EVENTS and event.ph == "B"
    ]
    completed: set[str] = set()
    for completion in completion_begins:
        operation_id = completion.attrs.get("operation_id")
        launch = launched.get(str(operation_id))
        if launch is None or operation_id in completed:
            failures.append(
                _failure(
                    "p2p_completion",
                    f"unmatched P2P completion {operation_id!r}",
                    rank,
                    iteration_id,
                )
            )
            continue
        event_name, batch_id, mode, peer, data_bytes = launch
        expected_site = (
            "communicate_internal_wait" if mode == "internal_wait" else "exposed_request_wait"
        )
        failures.extend(
            _expect_fields(
                completion,
                {
                    "request_id": operation_id,
                    "batch_id": batch_id,
                    "peer_rank": peer,
                    "data_bytes": data_bytes,
                    "transport_api": "isend_irecv",
                    "completion_mode": mode,
                    "completion_kind": "work_wait",
                    "completion_included": True,
                    "completion_site": expected_site,
                    "request_pairing": "key",
                },
                rank=rank,
                iteration=iteration_id,
            )
        )
        if completion.name != event_name:
            failures.append(
                _failure(
                    "p2p_direction",
                    f"completion name differs for {operation_id!r}",
                    rank,
                    iteration_id,
                )
            )
        completed.add(str(operation_id))
    if completed != set(launched):
        failures.append(
            _failure("p2p_completion", "P2P operations lack unique completions", rank, iteration_id)
        )
    completion_ends = [
        event for event in iteration.events if event.name in _DIRECTION_EVENTS and event.ph == "E"
    ]
    if len(completion_begins) != 16 or len(completion_ends) != 16:
        failures.append(
            _failure("p2p_pairing", "direction completion B/E counts differ", rank, iteration_id)
        )
    if any(
        event.attrs.get("completed") is not True or event.attrs.get("error_type") is not None
        for event in completion_ends
    ):
        failures.append(
            _failure("p2p_completion", "a P2P Work wait did not complete", rank, iteration_id)
        )
    if _events(iteration, "p2p-batch-complete", "B") or _events(
        iteration, "p2p-batch-device-sync", "B"
    ):
        failures.append(
            _failure("p2p_route", "unexpected batch completion or device sync", rank, iteration_id)
        )
    return failures


def _validate_grad_sync(iteration: Iteration, *, rank: int, iteration_id: int) -> list[Failure]:
    failures: list[Failure] = []
    selected = [
        event
        for event in iteration.events
        if event.name in {"grad-sync", "all-grads-sync", "dp-allreduce"}
    ]
    expected_boundaries = [
        ("grad-sync", "B"),
        ("all-grads-sync", "B"),
        *(
            boundary
            for _ in range(4)
            for boundary in (("dp-allreduce", "B"), ("dp-allreduce", "E"))
        ),
        ("all-grads-sync", "E"),
        ("grad-sync", "E"),
    ]
    observed = [(event.name, event.ph) for event in selected]
    if observed != expected_boundaries:
        failures.append(
            _failure(
                "grad_sync_order", f"gradient sync order differs: {observed!r}", rank, iteration_id
            )
        )
        return failures
    last_schedule_end = max(
        position
        for position, event in enumerate(iteration.events)
        if event.name in _SCHEDULE_EVENTS and event.ph == "E"
    )
    grad_sync_begin = next(
        position
        for position, event in enumerate(iteration.events)
        if event.name == "grad-sync" and event.ph == "B"
    )
    if grad_sync_begin <= last_schedule_end:
        failures.append(
            _failure(
                "grad_sync_order",
                "gradient sync begins before the final backward scope ends",
                rank,
                iteration_id,
            )
        )
    grad_sync = selected[0]
    failures.extend(
        _expect_fields(
            grad_sync,
            {"schedule": "dualpipev", "timing_phase": "framework_phase"},
            rank=rank,
            iteration=iteration_id,
        )
    )
    allreduces = [event for event in selected if event.name == "dp-allreduce" and event.ph == "B"]
    if tuple(event.attrs.get("group_size") for event in allreduces) != (2, 1, 2, 1):
        failures.append(
            _failure("grad_sync_groups", "DP bucket group sizes differ", rank, iteration_id)
        )
    operation_ids: set[str] = set()
    for event in allreduces:
        failures.extend(
            _expect_fields(
                event,
                {
                    "api_async_op": False,
                    "async_op": False,
                    "completion_included": False,
                    "group_role": "data_parallel",
                    "op": "all_reduce",
                    "operation_id_scope": "rank_local",
                    "overlap_enabled": False,
                    "payload_role": "gradient_bucket",
                    "stage": "main_bucket_allreduce",
                    "timing_phase": "collective_call",
                },
                rank=rank,
                iteration=iteration_id,
            )
        )
        operation_id = event.attrs.get("operation_id")
        if (
            not isinstance(operation_id, str)
            or operation_id in operation_ids
            or not isinstance(event.attrs.get("data_bytes"), int)
            or event.attrs["data_bytes"] <= 0
        ):
            failures.append(
                _failure("grad_sync_bucket", "invalid DP gradient bucket", rank, iteration_id)
            )
        else:
            operation_ids.add(operation_id)
    return failures


def _validate_iteration(iteration: Iteration, *, rank: int, pp_rank: int) -> list[Failure]:
    if iteration.iteration_id is None:
        return [_failure("iteration", "events are outside a numbered iteration", rank, -1)]
    iteration_id = int(iteration.iteration_id)
    scopes, failures = _schedule_scopes(iteration, rank, iteration_id)
    failures.extend(
        _validate_schedule(iteration, scopes, rank=rank, pp_rank=pp_rank, iteration_id=iteration_id)
    )
    for scope in scopes:
        failures.extend(
            _validate_a2a_scope(
                iteration, scope, rank=rank, pp_rank=pp_rank, iteration_id=iteration_id
            )
        )
    if sum(len(_events(iteration, name, "B")) for name in _SCHEDULE_EVENTS) == 13:
        a2a_launches = _events(iteration, _A2A_LAUNCH, "B")
        if len(a2a_launches) != 48:
            failures.append(_failure("a2a_count", "expected 48 A2A launches", rank, iteration_id))
        operation_ids = [event.attrs.get("operation_id") for event in a2a_launches]
        if len(set(operation_ids)) != len(operation_ids):
            failures.append(
                _failure(
                    "a2a_operation",
                    "A2A operation IDs are reused within an iteration",
                    rank,
                    iteration_id,
                )
            )
        if len(_events(iteration, _A2A_COMPLETE, "B")) != 51:
            failures.append(
                _failure("a2a_count", "expected 51 A2A completions", rank, iteration_id)
            )
    failures.extend(_validate_p2p(iteration, rank=rank, pp_rank=pp_rank, iteration_id=iteration_id))
    failures.extend(_validate_grad_sync(iteration, rank=rank, iteration_id=iteration_id))
    return failures


def validate_dualpipev_route(trace_root: Path) -> tuple[Failure, ...]:
    """Validate the controlled DualPipeV schedule and native Work lifecycles."""

    by_rank = _load_iterations(trace_root)
    failures: list[Failure] = []
    observed_ranks = tuple(sorted(by_rank))
    if observed_ranks != _RANKS:
        return (
            Failure(
                "trace.dualpipev.ranks",
                f"expected ranks {_RANKS}, observed {observed_ranks}",
                "pp2-dp2-ep2-dualpipev",
            ),
        )
    for rank, iterations in sorted(by_rank.items()):
        observed_iterations = tuple(
            int(iteration.iteration_id)
            for iteration in iterations
            if iteration.iteration_id is not None
        )
        if observed_iterations != _ITERATIONS:
            failures.append(
                Failure(
                    "trace.dualpipev.iterations",
                    f"expected iterations {_ITERATIONS}, observed {observed_iterations}",
                    f"rank={rank}",
                )
            )
        pp_rank = rank // 2
        for iteration in iterations:
            failures.extend(_validate_iteration(iteration, rank=rank, pp_rank=pp_rank))
    return tuple(failures)
