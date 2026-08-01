# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate the PP2 individual P2P API order with real two-rank NCCL Works."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Sequence

import torch
import torch.distributed as dist

from megatron.core.pipeline_parallel import p2p_communication


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _expected_api_calls(rank: int) -> tuple[tuple[str, str, str], ...]:
    if rank == 0:
        return (("send_next", "isend", "primary"), ("recv_next", "irecv", "world"))
    if rank == 1:
        return (("recv_prev", "irecv", "primary"), ("send_prev", "isend", "world"))
    raise ValueError("the PP2 order check requires rank 0 or rank 1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-elements", type=_positive_int, default=8)
    return parser


def _communicator_name(group: dist.ProcessGroup, *, primary_group: dist.ProcessGroup) -> str:
    if group is primary_group:
        return "primary"
    if group is dist.group.WORLD:
        return "world"
    raise AssertionError("_p2p_ops selected an unexpected process group")


def _run_order_check(
    *, rank: int, local_rank: int, tensor_elements: int, primary_group: dist.ProcessGroup
) -> dict[str, Any]:
    device = torch.device("cuda", local_rank)
    peer_rank = 1 - rank
    forward_payload = torch.arange(tensor_elements, dtype=torch.float32, device=device).add_(100)
    backward_payload = torch.arange(tensor_elements, dtype=torch.float32, device=device).add_(200)

    if rank == 0:
        tensor_send_prev = None
        tensor_recv_prev = None
        tensor_send_next = forward_payload
        tensor_recv_next = torch.empty_like(backward_payload)
        tensors_by_operation = {
            id(tensor_send_next): "send_next",
            id(tensor_recv_next): "recv_next",
        }
        received = tensor_recv_next
        expected_payload = backward_payload
    else:
        tensor_send_prev = backward_payload
        tensor_recv_prev = torch.empty_like(forward_payload)
        tensor_send_next = None
        tensor_recv_next = None
        tensors_by_operation = {
            id(tensor_recv_prev): "recv_prev",
            id(tensor_send_prev): "send_prev",
        }
        received = tensor_recv_prev
        expected_payload = forward_payload

    calls: list[dict[str, Any]] = []
    launched_work: dict[str, dist.Work] = {}
    real_isend = dist.isend
    real_irecv = dist.irecv

    def record_isend(*, tensor, dst, group):
        operation = tensors_by_operation[id(tensor)]
        calls.append(
            {
                "index": len(calls),
                "operation": operation,
                "api": "isend",
                "peer_rank": dst,
                "communicator": _communicator_name(group, primary_group=primary_group),
            }
        )
        request = real_isend(tensor=tensor, dst=dst, group=group)
        launched_work[operation] = request
        return request

    def record_irecv(*, tensor, src, group):
        operation = tensors_by_operation[id(tensor)]
        calls.append(
            {
                "index": len(calls),
                "operation": operation,
                "api": "irecv",
                "peer_rank": src,
                "communicator": _communicator_name(group, primary_group=primary_group),
            }
        )
        request = real_irecv(tensor=tensor, src=src, group=group)
        launched_work[operation] = request
        return request

    try:
        dist.isend = record_isend
        dist.irecv = record_irecv
        requests = p2p_communication._p2p_ops(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=primary_group,
            prev_pipeline_rank=peer_rank,
            next_pipeline_rank=peer_rank,
        )
    finally:
        dist.isend = real_isend
        dist.irecv = real_irecv

    expected_calls = _expected_api_calls(rank)
    observed_calls = tuple((call["operation"], call["api"], call["communicator"]) for call in calls)
    if observed_calls != expected_calls:
        raise AssertionError(
            f"rank {rank} submitted P2P APIs in order {observed_calls!r}; "
            f"expected {expected_calls!r}"
        )
    if any(call["peer_rank"] != peer_rank for call in calls):
        raise AssertionError(f"rank {rank} submitted a P2P API to the wrong peer")

    expected_request_keys = tuple(call[0] for call in expected_calls)
    if tuple(requests) != expected_request_keys:
        raise AssertionError(
            f"rank {rank} returned request keys {tuple(requests)!r}; "
            f"expected {expected_request_keys!r}"
        )

    work_types = []
    for operation, request in requests.items():
        if not isinstance(request, dist.Work):
            raise AssertionError(f"{operation} returned a non-native Work")
        if request is not launched_work[operation]:
            raise AssertionError(f"{operation} replaced the native Work identity")
        result = request.wait()
        if result is False:
            raise AssertionError(f"{operation} Work.wait() reported failure")
        work_types.append(type(request).__name__)

    if not torch.equal(received, expected_payload):
        raise AssertionError(f"rank {rank} received an unexpected P2P payload")
    if primary_group is dist.group.WORLD:
        raise AssertionError("the primary PP2 group did not create a distinct communicator")
    if dist.get_backend(primary_group) != "nccl" or dist.get_backend() != "nccl":
        raise AssertionError("the P2P order check requires NCCL communicators")

    return {
        "schema_version": "1.0.0",
        "status": "passed",
        "rank": rank,
        "world_size": dist.get_world_size(),
        "backend": "nccl",
        "target_symbol": "megatron.core.pipeline_parallel.p2p_communication._p2p_ops",
        "order_scope": "rank_local_python_api_invocation",
        "communicators": {"primary": "dist.new_group([0, 1])", "alternate": "dist.group.WORLD"},
        "primary_group_distinct_from_world": True,
        "api_calls": calls,
        "request_keys": list(requests),
        "work_identity_preserved": True,
        "work_types": work_types,
        "payload_elements": tensor_elements,
        "payload_sum": float(received.sum().item()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    primary_group: dist.ProcessGroup | None = None
    try:
        if dist.get_world_size() != 2:
            raise ValueError("run_p2p_order_nccl.py requires exactly two ranks")
        primary_group = dist.new_group(ranks=[0, 1], backend="nccl")
        dist.barrier()
        evidence = _run_order_check(
            rank=dist.get_rank(),
            local_rank=local_rank,
            tensor_elements=args.tensor_elements,
            primary_group=primary_group,
        )
        dist.barrier()
        print(json.dumps(evidence, sort_keys=True), flush=True)
    finally:
        if primary_group is not None:
            dist.destroy_process_group(primary_group)
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
