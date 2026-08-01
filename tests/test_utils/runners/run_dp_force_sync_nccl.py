# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise the DP force-sync Probe with a real two-rank NCCL Work."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
from megatron.megalens.runtime import MegaLensRuntime


def _runtime_args(trace_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        trace=True,
        trace_mode=1,
        trace_dir=str(trace_dir),
        trace_interval=1,
        continuous_trace_iterations=1,
        trace_granularity="base",
        trace_cupti_kernels="off",
        trace_gather_to_rank0=False,
        hardware_monitor=False,
        sentinel_hw_sample_ms=100.0,
        sentinel_flush_interval=100,
        cuda_graph_impl="none",
        profile=False,
        use_pytorch_profiler=False,
    )


def _bucket_group(process_group: dist.ProcessGroup) -> tuple[object, torch.Tensor]:
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    param_data = torch.zeros(8, dtype=torch.float32, device="cuda")
    param_data.view(world_size, -1)[rank].fill_(rank + 1)
    bucket = SimpleNamespace(param_data=param_data)

    group = _ParamAndGradBucketGroup.__new__(_ParamAndGradBucketGroup)
    group.buckets = [bucket]
    group.cached_param_buffer_shard_list = [None]
    group.ddp_config = SimpleNamespace(
        use_distributed_optimizer=True,
        overlap_param_gather=True,
    )
    group.intra_distributed_optimizer_instance_group = process_group
    group.intra_distributed_optimizer_instance_size = world_size
    group.intra_distributed_optimizer_instance_rank = rank
    group.param_gather_handle = None
    group.param_gather_dispatched = False
    group._param_gather_trace_operation_id = None
    return group, param_data


def _validate_trace(trace_dir: Path, rank: int) -> dict[str, object]:
    shards = tuple(trace_dir.glob(f"benchmark-global-{rank}-*.json"))
    if len(shards) != 1:
        raise AssertionError(f"rank {rank} produced {len(shards)} trace shards")
    rows = json.loads(shards[0].read_text(encoding="utf-8"))
    lifecycle = [
        row
        for row in rows
        if row.get("name") in {"dp-param-all-gather", "dp-param-sync-complete"}
    ]
    sequence = [(row.get("name"), row.get("ph")) for row in lifecycle]
    expected_sequence = [
        ("dp-param-all-gather", "B"),
        ("dp-param-all-gather", "E"),
        ("dp-param-sync-complete", "B"),
        ("dp-param-sync-complete", "E"),
    ]
    if sequence != expected_sequence:
        raise AssertionError(f"rank {rank} has lifecycle {sequence!r}")

    launch, launch_end, completion, completion_end = lifecycle
    operation_id = launch.get("operation_id")
    if not isinstance(operation_id, str):
        raise AssertionError("parameter AllGather has no operation identity")
    expected_group = [peer for peer in range(dist.get_world_size()) if peer != rank]
    expected_fields = {
        "api_async_op": True,
        "async_op": True,
        "group_size": 2,
        "timing_phase": "async_dispatch",
    }
    if any(launch.get(key) != value for key, value in expected_fields.items()):
        raise AssertionError(f"rank {rank} has invalid launch fields")
    if launch_end.get("group") != expected_group:
        raise AssertionError(f"rank {rank} recorded group={launch_end.get('group')!r}")
    if completion.get("operation_id") != operation_id or completion.get(
        "completion_site"
    ) != "force_sync":
        raise AssertionError(f"rank {rank} did not correlate the force-sync wait")
    if completion_end.get("completed") is not True:
        raise AssertionError(f"rank {rank} did not complete the force-sync wait")
    return {
        "rank": rank,
        "operation_id": operation_id,
        "group": expected_group,
        "records": len(lifecycle),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    runtime: MegaLensRuntime | None = None
    try:
        parallel_state.initialize_model_parallel(create_gloo_process_groups=False)
        process_group = parallel_state.get_data_parallel_group()
        runtime = MegaLensRuntime(_runtime_args(args.trace_dir))
        bucket_group, param_data = _bucket_group(process_group)
        with runtime.iteration(1):
            bucket_group.start_param_sync()
            if bucket_group.param_gather_handle is None:
                raise AssertionError("asynchronous parameter AllGather returned no Work")
            bucket_group.start_param_sync(force_sync=True)
            if bucket_group.param_gather_handle is not None:
                raise AssertionError("force-sync retained the completed Work")
        expected = torch.cat(
            [
                torch.full((4,), rank + 1, dtype=torch.float32, device="cuda")
                for rank in range(dist.get_world_size(process_group))
            ]
        )
        if not torch.equal(param_data, expected):
            raise AssertionError("NCCL parameter AllGather produced an unexpected payload")
        runtime.shutdown(graceful=True)
        runtime = None
        dist.barrier()
        evidence = _validate_trace(args.trace_dir, dist.get_rank())
        print(json.dumps(evidence, sort_keys=True), flush=True)
    finally:
        if runtime is not None:
            runtime.shutdown(graceful=False)
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
