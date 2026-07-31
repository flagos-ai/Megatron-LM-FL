# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.test_utils.runners import bridge_probe_contract
from tests.test_utils.runners import megalens_run_manifest as manifest
from tests.test_utils.runners import run_flagscale_megalens as gate

_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "flagscale_single_node_multimodule_bridge_smoke.yaml"
)
_ASYMMETRIC_FIXTURES = {
    "multimodule-bridge8-fanin": (
        Path(__file__).parent
        / "fixtures"
        / "flagscale_single_node_multimodule_bridge_fanin.yaml"
    ),
    "multimodule-bridge8-fanout": (
        Path(__file__).parent
        / "fixtures"
        / "flagscale_single_node_multimodule_bridge_fanout.yaml"
    ),
}


def _semantic_fields(message_kind: str, pipeline_direction: str) -> dict[str, str]:
    role = "activation" if pipeline_direction == "forward" else "gradient"
    if message_kind == "shape":
        return {"shape_of": role}
    return {"payload_role": role}


def _write_trace(
    trace_root: Path,
    rank: int,
    *,
    wrong_request_id: bool = False,
    mixed_shape_fields: bool = False,
    failed_completion: bool = False,
    duplicate_completion: bool = False,
    delayed_launch_end: bool = False,
    coalesced_combined: bool = False,
) -> None:
    trace_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    timestamp = 0
    source_directions = (("send", "forward"), ("recv", "backward"))
    destination_shape_directions = (
        ("recv", "forward"),
        ("send", "backward"),
    )
    destination_payload_directions = (
        ("send", "backward"),
        ("recv", "forward"),
    )
    batch_sequence = 0
    wrong_request_used = False
    mixed_shape_used = False
    failed_completion_used = False
    duplicate_completion_used = False
    delayed_launch_end_used = False

    def event(name: str, phase: str, **attrs: object) -> dict[str, object]:
        nonlocal timestamp
        timestamp += 1
        row = {
            "name": name,
            "ph": phase,
            "rel_ts": timestamp,
            "dev": rank,
            "g_rk": rank,
            "dp_rk": 0,
            "pp_rk": 0,
            "tp_rk": 0,
            **attrs,
        }
        rows.append(row)
        return row

    def event_name(direction: str, pipeline_direction: str) -> str:
        return f"bridge-{direction}-{pipeline_direction}"

    def finish_observed_scope(name: str) -> None:
        nonlocal failed_completion_used
        completed = True
        error_type = None
        if failed_completion and not failed_completion_used:
            completed = False
            error_type = "RuntimeError"
            failed_completion_used = True
        event(name, "E", completed=completed, error_type=error_type)

    def add_blocking(direction: str, pipeline_direction: str) -> None:
        name = event_name(direction, pipeline_direction)
        event(
            name,
            "B",
            backend="nccl",
            communicator_kind="bridge",
            completion_kind="inline_api_return",
            data_bytes=131072,
            direction=direction,
            message_kind="payload",
            peer_rank=1 - rank,
            pipeline_direction=pipeline_direction,
            src_module="encoder",
            dest_module="llm",
            transport_api="send_recv",
            **_semantic_fields("payload", pipeline_direction),
        )
        finish_observed_scope(name)

    def add_launch(
        message_kind: str,
        directions: tuple[tuple[str, str], ...],
    ) -> None:
        nonlocal batch_sequence
        nonlocal duplicate_completion_used
        nonlocal delayed_launch_end_used
        nonlocal mixed_shape_used
        nonlocal wrong_request_used

        batch_sequence += 1
        batch_id = f"bridge-p2p:{rank}:{batch_sequence}"
        operations = []
        for index, (direction, pipeline_direction) in enumerate(directions):
            operation_id = f"{batch_id}:{index}"
            operation = {
                "backend": "nccl",
                "comm_type": "p2p",
                "completion_mode": "internal_wait",
                "data_bytes": 24 if message_kind == "shape" else 131072,
                "direction": direction,
                "message_kind": message_kind,
                "operation_id": operation_id,
                "peer_rank": 1 - rank,
                "pipeline_direction": pipeline_direction,
                "request_id": operation_id,
                "transport_api": "batch_isend_irecv",
                **_semantic_fields(message_kind, pipeline_direction),
            }
            if (
                mixed_shape_fields
                and message_kind == "shape"
                and not mixed_shape_used
            ):
                operation["payload_role"] = operation["shape_of"]
                mixed_shape_used = True
            operations.append(operation)
        event(
            "bridge-p2p-launch",
            "B",
            batch_id=batch_id,
            communicator_kind="bridge",
            completion_mode="internal_wait",
            dest_module="llm",
            message_kind=message_kind,
            operation_count=len(operations),
            operations=operations,
            src_module="encoder",
            transport_api="batch_isend_irecv",
        )
        delay_launch_end = delayed_launch_end and not delayed_launch_end_used
        if not delay_launch_end:
            event("bridge-p2p-launch", "E")
        if coalesced_combined and len(operations) > 1:
            if delay_launch_end:
                event("bridge-p2p-launch", "E")
                delayed_launch_end_used = True
            return
        for operation in operations:
            direction = str(operation["direction"])
            pipeline_direction = str(operation["pipeline_direction"])
            request_id = operation["operation_id"]
            if (
                wrong_request_id
                and message_kind == "payload"
                and not wrong_request_used
            ):
                request_id = "wrong"
                wrong_request_used = True
            attrs = {
                "batch_id": batch_id,
                "communicator_kind": "bridge",
                "completion_kind": "work_wait",
                "data_bytes": operation["data_bytes"],
                "direction": direction,
                "message_kind": message_kind,
                "operation_id": operation["operation_id"],
                "peer_rank": 1 - rank,
                "pipeline_direction": pipeline_direction,
                "request_id": request_id,
                "src_module": "encoder",
                "dest_module": "llm",
                "transport_api": "batch_isend_irecv",
                **_semantic_fields(message_kind, pipeline_direction),
            }
            event(
                event_name(direction, pipeline_direction),
                "B",
                backend="nccl",
                **attrs,
            )
            finish_observed_scope(event_name(direction, pipeline_direction))
            if duplicate_completion and not duplicate_completion_used:
                event(
                    event_name(direction, pipeline_direction),
                    "B",
                    backend="nccl",
                    **attrs,
                )
                finish_observed_scope(event_name(direction, pipeline_direction))
                duplicate_completion_used = True
        if delay_launch_end:
            event("bridge-p2p-launch", "E")
            delayed_launch_end_used = True

    def add_broadcast_pair() -> None:
        pipeline_direction = "backward" if rank == 0 else "forward"
        grid_side = "src" if rank == 0 else "dest"
        for message_kind in ("shape", "payload"):
            event(
                "bridge-grid-broadcast",
                "B",
                backend="nccl",
                collective_role="source",
                communicator_kind="bridge",
                completion_kind="inline_api_return",
                data_bytes=24 if message_kind == "shape" else 131072,
                dest_module="llm",
                grid_side=grid_side,
                message_kind=message_kind,
                pipeline_direction=pipeline_direction,
                source_rank=rank,
                src_module="encoder",
                transport_api="broadcast",
                **_semantic_fields(message_kind, pipeline_direction),
            )
            finish_observed_scope("bridge-grid-broadcast")

    rows.append({"name": "iteration", "ph": "B", "iteration": 1, "pad_before": 0})
    if rank == 0:
        add_launch("shape", (source_directions[0],))
        add_blocking(*source_directions[0])
        for _ in range(3):
            add_launch("shape", source_directions)
            add_launch("payload", source_directions)
            add_broadcast_pair()
        add_launch("shape", (source_directions[1],))
        add_blocking(*source_directions[1])
        add_broadcast_pair()
    else:
        add_launch("shape", (destination_shape_directions[0],))
        add_blocking(*destination_shape_directions[0])
        add_broadcast_pair()
        for _ in range(3):
            add_launch("shape", destination_shape_directions)
            add_launch("payload", destination_payload_directions)
            add_broadcast_pair()
        add_launch("shape", (destination_shape_directions[1],))
        add_blocking(*destination_shape_directions[1])

    rows.append(
        {
            "name": "iteration",
            "ph": "E",
            "iteration": 1,
            "duration_wall": timestamp,
        }
    )
    path = (
        trace_root
        / f"benchmark-global-{rank}-data-0-pipeline-0-tensor-0.json"
    )
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_run_results(
    run_root: Path,
    *,
    gradient_norm: float = 1.0,
    trace_enabled: bool = True,
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    for rank, role in ((0, "encoder"), (1, "llm")):
        payload = {
            "backward_completed": True,
            "completed": True,
            "global_rank": rank,
            "gradient_count": 4,
            "gradient_finite": True,
            "gradient_norm": gradient_norm,
            "loss_count": 0 if rank == 0 else 4,
            "loss_finite": True,
            "module_role": role,
            "optimizer_step": "not_run",
            "trace_enabled": trace_enabled,
            "world_size": 2,
        }
        (run_root / f"training-result-rank-{rank}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


def _write_asymmetric_run_results(
    run_root: Path,
    *,
    topology,
    trace_enabled: bool = True,
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    for rank in range(8):
        payload = {
            "backward_completed": True,
            "completed": True,
            "global_rank": rank,
            "gradient_count": 4,
            "gradient_finite": True,
            "gradient_norm": 1.0,
            "loss_count": 4 if rank in topology.loss_ranks else 0,
            "loss_finite": True,
            "module_role": "encoder" if rank < 4 else "llm",
            "optimizer_step": "not_run",
            "trace_enabled": trace_enabled,
            "world_size": 8,
        }
        (run_root / f"training-result-rank-{rank}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


def _write_asymmetric_trace(
    trace_root: Path,
    *,
    topology,
    wrong_peer: bool = False,
    coalesced_multi_operation: bool = False,
    delayed_launch_end: bool = False,
) -> None:
    trace_root.mkdir(parents=True, exist_ok=True)
    event_routes = {
        event_name: route
        for route, event_name in bridge_probe_contract._DIRECTION_TO_EVENT.items()
    }
    wrong_peer_used = False

    for rank in range(8):
        rows: list[dict[str, object]] = [
            {"name": "iteration", "ph": "B", "iteration": 1, "pad_before": 0}
        ]
        timestamp = 0

        def event(name: str, phase: str, **attrs: object) -> None:
            nonlocal timestamp
            timestamp += 1
            rows.append(
                {
                    "name": name,
                    "ph": phase,
                    "rel_ts": timestamp,
                    "dev": rank,
                    "g_rk": rank,
                    "dp_rk": 0,
                    "pp_rk": 0,
                    "tp_rk": 0,
                    **attrs,
                }
            )

        def finish(name: str) -> None:
            event(name, "E", completed=True, error_type=None)

        broadcast_plan = topology.broadcast_ranks.get(rank)
        rank_plan = topology.bridge_ranks.get(rank)
        delayed_launch_used = False

        def add_launch(
            launch_index: int,
            message_kind: str,
            expected_operations,
        ) -> None:
            nonlocal delayed_launch_used
            nonlocal wrong_peer_used

            batch_id = f"bridge-p2p:{rank}:{launch_index}"
            operations = []
            for operation_index, (event_name, peer_rank) in enumerate(
                expected_operations
            ):
                direction, pipeline_direction = event_routes[event_name]
                operation_id = f"{batch_id}:{operation_index}"
                operation_peer = peer_rank
                if wrong_peer and not wrong_peer_used:
                    operation_peer = 7
                    wrong_peer_used = True
                operations.append(
                    {
                        "backend": "nccl",
                        "comm_type": "p2p",
                        "completion_mode": "internal_wait",
                        "data_bytes": (
                            bridge_probe_contract._SHAPE_BYTES
                            if message_kind == "shape"
                            else rank_plan.payload_bytes
                        ),
                        "direction": direction,
                        "message_kind": message_kind,
                        "operation_id": operation_id,
                        "peer_rank": operation_peer,
                        "pipeline_direction": pipeline_direction,
                        "request_id": operation_id,
                        "transport_api": "batch_isend_irecv",
                        **_semantic_fields(
                            message_kind,
                            pipeline_direction,
                        ),
                    }
                )
            event(
                "bridge-p2p-launch",
                "B",
                batch_id=batch_id,
                communicator_kind="bridge",
                completion_mode="internal_wait",
                dest_module="llm",
                message_kind=message_kind,
                operation_count=len(operations),
                operations=operations,
                src_module="encoder",
                transport_api="batch_isend_irecv",
            )
            delay_end = delayed_launch_end and not delayed_launch_used
            if not delay_end:
                event("bridge-p2p-launch", "E")
            if not (
                coalesced_multi_operation and len(operations) > 1
            ):
                for operation, (event_name, _peer_rank) in zip(
                    operations, expected_operations
                ):
                    event(
                        event_name,
                        "B",
                        backend="nccl",
                        batch_id=batch_id,
                        communicator_kind="bridge",
                        completion_kind="work_wait",
                        data_bytes=operation["data_bytes"],
                        direction=operation["direction"],
                        message_kind=message_kind,
                        operation_id=operation["operation_id"],
                        peer_rank=operation["peer_rank"],
                        pipeline_direction=operation["pipeline_direction"],
                        request_id=operation["operation_id"],
                        src_module="encoder",
                        dest_module="llm",
                        transport_api="batch_isend_irecv",
                        **_semantic_fields(
                            message_kind,
                            str(operation["pipeline_direction"]),
                        ),
                    )
                    finish(event_name)
            if delay_end:
                event("bridge-p2p-launch", "E")
                delayed_launch_used = True

        def add_blocking(event_name: str) -> None:
            direction, pipeline_direction = event_routes[event_name]
            for peer_rank in rank_plan.peers:
                event(
                    event_name,
                    "B",
                    backend="nccl",
                    communicator_kind="bridge",
                    completion_kind="inline_api_return",
                    data_bytes=rank_plan.payload_bytes,
                    direction=direction,
                    message_kind="payload",
                    peer_rank=peer_rank,
                    pipeline_direction=pipeline_direction,
                    src_module="encoder",
                    dest_module="llm",
                    transport_api="send_recv",
                    **_semantic_fields("payload", pipeline_direction),
                )
                finish(event_name)

        def add_broadcast_pair() -> None:
            for message_kind in ("shape", "payload"):
                event(
                    "bridge-grid-broadcast",
                    "B",
                    backend="nccl",
                    collective_role=broadcast_plan.collective_role,
                    communicator_kind="bridge",
                    completion_kind="inline_api_return",
                    data_bytes=(
                        bridge_probe_contract._SHAPE_BYTES
                        if message_kind == "shape"
                        else broadcast_plan.payload_bytes
                    ),
                    dest_module="llm",
                    grid_side=broadcast_plan.grid_side,
                    message_kind=message_kind,
                    pipeline_direction=broadcast_plan.pipeline_direction,
                    source_rank=broadcast_plan.source_rank,
                    src_module="encoder",
                    transport_api="broadcast",
                    **_semantic_fields(
                        message_kind,
                        broadcast_plan.pipeline_direction,
                    ),
                )
                finish("bridge-grid-broadcast")

        if rank_plan is not None:
            launches = iter(
                enumerate(
                    bridge_probe_contract._expected_asymmetric_launches(
                        rank_plan
                    ),
                    start=1,
                )
            )

            def next_launch() -> None:
                launch_index, (message_kind, operations) = next(launches)
                add_launch(launch_index, message_kind, operations)

            if rank_plan.role == "sender":
                for _ in range(2):
                    next_launch()
                    add_blocking("bridge-send-forward")
                for _ in range(2):
                    next_launch()
                    next_launch()
                    add_broadcast_pair()
                for _ in range(2):
                    next_launch()
                    add_blocking("bridge-recv-backward")
                    add_broadcast_pair()
            else:
                for _ in range(2):
                    next_launch()
                    add_blocking("bridge-recv-forward")
                    add_broadcast_pair()
                for _ in range(2):
                    next_launch()
                    next_launch()
                    add_broadcast_pair()
                for _ in range(2):
                    next_launch()
                    add_blocking("bridge-send-backward")
        elif broadcast_plan is not None:
            for _ in range(4):
                add_broadcast_pair()

        rows.append(
            {
                "name": "iteration",
                "ph": "E",
                "iteration": 1,
                "duration_wall": timestamp,
            }
        )
        path = (
            trace_root
            / f"benchmark-global-{rank}-data-0-pipeline-0-tensor-0.json"
        )
        path.write_text(json.dumps(rows), encoding="utf-8")


def test_multimodule_bridge_profile_uses_flagscale_native_entry() -> None:
    config = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))

    assert config["experiment"]["task"] == {
        "type": "train",
        "backend": "native",
        "entrypoint": (
            "/workspace/Megatron-LM-FL/"
            "tests/test_utils/runners/run_multimodule_bridge.py"
        ),
    }
    assert config["experiment"]["runner"]["nproc_per_node"] == 2
    assert config["train"]["bridge"] == {
        "hidden_size": 512,
        "seq_length": 64,
        "micro_batch_size": 2,
        "num_microbatches": 4,
    }
    assert gate._CONFIG_PROFILES[_FIXTURE.stem] == "multimodule-bridge2"


@pytest.mark.parametrize(
    ("profile_name", "encoder_grid", "llm_grid"),
    (
        (
            "multimodule-bridge8-fanin",
            {"tp": 1, "pp": 2, "dp": 2, "grid_offset": 0},
            {"tp": 2, "pp": 2, "dp": 1, "grid_offset": 4},
        ),
        (
            "multimodule-bridge8-fanout",
            {"tp": 2, "pp": 2, "dp": 1, "grid_offset": 0},
            {"tp": 1, "pp": 2, "dp": 2, "grid_offset": 4},
        ),
    ),
)
def test_asymmetric_bridge_profiles_use_eight_rank_native_entries(
    profile_name: str,
    encoder_grid: dict[str, int],
    llm_grid: dict[str, int],
) -> None:
    fixture = _ASYMMETRIC_FIXTURES[profile_name]
    config = yaml.safe_load(fixture.read_text(encoding="utf-8"))

    assert config["experiment"]["task"] == {
        "type": "train",
        "backend": "native",
        "entrypoint": (
            "/workspace/Megatron-LM-FL/"
            "tests/test_utils/runners/run_multimodule_bridge.py"
        ),
    }
    assert config["experiment"]["runner"]["nproc_per_node"] == 8
    assert config["train"]["bridge"] == {
        "hidden_size": 512,
        "seq_length": 64,
        "micro_batch_size": 4,
        "num_microbatches": 4,
        "encoder_grid": encoder_grid,
        "llm_grid": llm_grid,
    }
    assert gate._CONFIG_PROFILES[fixture.stem] == profile_name
    assert gate.PROFILES[profile_name].rank_count == 8


def test_asymmetric_bridge_launch_plans_lock_pp2_lifecycle() -> None:
    sender = bridge_probe_contract._BridgeRankPlan(
        "sender",
        (4, 6),
        131072,
    )
    receiver = bridge_probe_contract._BridgeRankPlan(
        "receiver",
        (1, 3),
        262144,
    )

    assert bridge_probe_contract._expected_asymmetric_launches(sender) == [
        (
            "shape",
            (("bridge-send-forward", 4), ("bridge-send-forward", 6)),
        ),
        (
            "shape",
            (("bridge-send-forward", 4), ("bridge-send-forward", 6)),
        ),
        (
            "shape",
            (
                ("bridge-send-forward", 4),
                ("bridge-send-forward", 6),
                ("bridge-recv-backward", 4),
                ("bridge-recv-backward", 6),
            ),
        ),
        (
            "payload",
            (
                ("bridge-send-forward", 4),
                ("bridge-recv-backward", 4),
                ("bridge-send-forward", 6),
                ("bridge-recv-backward", 6),
            ),
        ),
        (
            "shape",
            (
                ("bridge-send-forward", 4),
                ("bridge-send-forward", 6),
                ("bridge-recv-backward", 4),
                ("bridge-recv-backward", 6),
            ),
        ),
        (
            "payload",
            (
                ("bridge-send-forward", 4),
                ("bridge-recv-backward", 4),
                ("bridge-send-forward", 6),
                ("bridge-recv-backward", 6),
            ),
        ),
        (
            "shape",
            (("bridge-recv-backward", 4), ("bridge-recv-backward", 6)),
        ),
        (
            "shape",
            (("bridge-recv-backward", 4), ("bridge-recv-backward", 6)),
        ),
    ]
    assert bridge_probe_contract._expected_asymmetric_launches(receiver) == [
        (
            "shape",
            (("bridge-recv-forward", 1), ("bridge-recv-forward", 3)),
        ),
        (
            "shape",
            (("bridge-recv-forward", 1), ("bridge-recv-forward", 3)),
        ),
        (
            "shape",
            (
                ("bridge-recv-forward", 1),
                ("bridge-recv-forward", 3),
                ("bridge-send-backward", 1),
                ("bridge-send-backward", 3),
            ),
        ),
        (
            "payload",
            (
                ("bridge-send-backward", 1),
                ("bridge-recv-forward", 1),
                ("bridge-send-backward", 3),
                ("bridge-recv-forward", 3),
            ),
        ),
        (
            "shape",
            (
                ("bridge-recv-forward", 1),
                ("bridge-recv-forward", 3),
                ("bridge-send-backward", 1),
                ("bridge-send-backward", 3),
            ),
        ),
        (
            "payload",
            (
                ("bridge-send-backward", 1),
                ("bridge-recv-forward", 1),
                ("bridge-send-backward", 3),
                ("bridge-recv-forward", 3),
            ),
        ),
        (
            "shape",
            (("bridge-send-backward", 1), ("bridge-send-backward", 3)),
        ),
        (
            "shape",
            (("bridge-send-backward", 1), ("bridge-send-backward", 3)),
        ),
    ]


@pytest.mark.parametrize(
    ("profile_name", "topology"),
    (
        (
            "multimodule-bridge8-fanin",
            bridge_probe_contract._FANIN_TOPOLOGY,
        ),
        (
            "multimodule-bridge8-fanout",
            bridge_probe_contract._FANOUT_TOPOLOGY,
        ),
    ),
)
def test_asymmetric_bridge_trace_and_run_contracts_accept_valid_evidence(
    tmp_path: Path,
    profile_name: str,
    topology,
) -> None:
    _write_asymmetric_trace(tmp_path / "traces", topology=topology)
    _write_asymmetric_run_results(tmp_path, topology=topology)

    profile = gate.PROFILES[profile_name]
    report = manifest.validate_trace(
        tmp_path / "traces",
        profile,
        trace_enabled=True,
    )
    report = manifest.validate_run_artifacts(
        tmp_path,
        profile,
        report,
        trace_enabled=True,
    )

    assert report.passed


@pytest.mark.parametrize(
    "topology",
    (
        bridge_probe_contract._FANIN_TOPOLOGY,
        bridge_probe_contract._FANOUT_TOPOLOGY,
    ),
)
def test_asymmetric_bridge_trace_rejects_wrong_peer(
    tmp_path: Path,
    topology,
) -> None:
    _write_asymmetric_trace(tmp_path, topology=topology, wrong_peer=True)

    failures = bridge_probe_contract._validate_asymmetric_bridge_trace(
        tmp_path,
        topology,
    )

    assert "trace.bridge.operation" in {
        failure.code for failure in failures
    }


@pytest.mark.parametrize(
    "topology",
    (
        bridge_probe_contract._FANIN_TOPOLOGY,
        bridge_probe_contract._FANOUT_TOPOLOGY,
    ),
)
def test_asymmetric_bridge_trace_accepts_coalesced_multi_operation_work(
    tmp_path: Path,
    topology,
) -> None:
    _write_asymmetric_trace(
        tmp_path,
        topology=topology,
        coalesced_multi_operation=True,
    )

    failures = bridge_probe_contract._validate_asymmetric_bridge_trace(
        tmp_path,
        topology,
    )

    assert not failures


def test_asymmetric_bridge_trace_rejects_wait_before_launch_end(
    tmp_path: Path,
) -> None:
    _write_asymmetric_trace(
        tmp_path,
        topology=bridge_probe_contract._FANIN_TOPOLOGY,
        delayed_launch_end=True,
    )

    failures = bridge_probe_contract.validate_multimodule_bridge_fanin_trace(
        tmp_path
    )

    assert "trace.bridge.launch_scope" in {
        failure.code for failure in failures
    }


def test_multimodule_bridge_trace_and_run_contracts_accept_valid_evidence(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_trace(tmp_path / "traces", rank)
    _write_run_results(tmp_path)

    profile = gate.PROFILES["multimodule-bridge2"]
    report = manifest.validate_trace(
        tmp_path / "traces",
        profile,
        trace_enabled=True,
    )
    report = manifest.validate_run_artifacts(
        tmp_path,
        profile,
        report,
        trace_enabled=True,
    )

    assert report.passed


def test_multimodule_bridge_trace_rejects_wrong_work_identity(
    tmp_path: Path,
) -> None:
    _write_trace(tmp_path, 0, wrong_request_id=True)
    _write_trace(tmp_path, 1)

    failures = bridge_probe_contract.validate_multimodule_bridge_trace(tmp_path)

    assert "trace.bridge.identity" in {failure.code for failure in failures}


def test_multimodule_bridge_trace_accepts_non_positional_coalesced_work(
    tmp_path: Path,
) -> None:
    _write_trace(tmp_path, 0, coalesced_combined=True)
    _write_trace(tmp_path, 1, coalesced_combined=True)

    failures = bridge_probe_contract.validate_multimodule_bridge_trace(tmp_path)

    assert not failures


def test_multimodule_bridge_trace_rejects_mixed_shape_fields(
    tmp_path: Path,
) -> None:
    _write_trace(tmp_path, 0, mixed_shape_fields=True)
    _write_trace(tmp_path, 1)

    failures = bridge_probe_contract.validate_multimodule_bridge_trace(tmp_path)

    assert "trace.bridge.message_kind" in {
        failure.code for failure in failures
    }


def test_multimodule_bridge_trace_rejects_failed_completion(
    tmp_path: Path,
) -> None:
    _write_trace(tmp_path, 0, failed_completion=True)
    _write_trace(tmp_path, 1)

    failures = bridge_probe_contract.validate_multimodule_bridge_trace(tmp_path)

    assert "trace.bridge.scope_completion" in {
        failure.code for failure in failures
    }


def test_multimodule_bridge_trace_rejects_duplicate_work_completion(
    tmp_path: Path,
) -> None:
    _write_trace(tmp_path, 0, duplicate_completion=True)
    _write_trace(tmp_path, 1)

    failures = bridge_probe_contract.validate_multimodule_bridge_trace(tmp_path)

    assert "trace.bridge.identity" in {failure.code for failure in failures}


def test_multimodule_bridge_trace_rejects_launch_scope_ending_after_wait(
    tmp_path: Path,
) -> None:
    _write_trace(tmp_path, 0, delayed_launch_end=True)
    _write_trace(tmp_path, 1)

    failures = bridge_probe_contract.validate_multimodule_bridge_trace(tmp_path)

    assert "trace.bridge.launch_scope" in {
        failure.code for failure in failures
    }


def test_multimodule_bridge_run_contract_applies_to_trace_off(
    tmp_path: Path,
) -> None:
    _write_run_results(
        tmp_path,
        gradient_norm=0.0,
        trace_enabled=False,
    )
    profile = gate.PROFILES["multimodule-bridge2"]
    report = manifest.validate_trace(
        tmp_path / "traces",
        profile,
        trace_enabled=False,
    )

    report = manifest.validate_run_artifacts(
        tmp_path,
        profile,
        report,
        trace_enabled=False,
    )

    assert not report.passed
    assert {failure.code for failure in report.failures} == {
        "run.bridge.gradient"
    }


def test_multimodule_bridge_run_contract_rejects_trace_mode_mismatch(
    tmp_path: Path,
) -> None:
    _write_run_results(tmp_path, trace_enabled=True)
    profile = gate.PROFILES["multimodule-bridge2"]
    report = manifest.validate_trace(
        tmp_path / "traces",
        profile,
        trace_enabled=False,
    )

    report = manifest.validate_run_artifacts(
        tmp_path,
        profile,
        report,
        trace_enabled=False,
    )

    assert not report.passed
    assert {failure.code for failure in report.failures} == {
        "run.bridge.trace_mode"
    }
