# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

from tests.test_utils.runners import tp_probe_contract


def _write_collective_trace(
    trace_root: Path,
    *,
    rank: int,
    pipeline_rank: int = 0,
    tensor_rank: int | None = None,
    tp_peer_rank: int | None = None,
    embedding_peer_rank: int | None = None,
    grad_sync_schedule: str = "no-pipelining",
    omit_nested_reduce_scatter: bool = False,
    cross_all_gather_scopes: bool = False,
    include_first_all_gather: bool = True,
    include_linear_lifecycle: bool = False,
    include_linear_allreduce: bool = False,
    mismatched_linear_completion: bool = False,
    include_final_grad_sync: bool = False,
    include_sp_sync: bool = True,
    include_embedding_sync: bool = False,
    nest_sp_in_all_grads: bool = False,
    include_te_model_scopes: bool = False,
    reverse_te_model_scopes: bool = False,
) -> None:
    rows: list[dict[str, object]] = []
    timestamp = 0
    tensor_rank = rank if tensor_rank is None else tensor_rank
    tp_peer_rank = 1 - rank if tp_peer_rank is None else tp_peer_rank
    embedding_peer_rank = (
        1 - rank if embedding_peer_rank is None else embedding_peer_rank
    )

    def event(name: str, phase: str, **attrs: object) -> None:
        nonlocal timestamp
        timestamp += 1
        rows.append(
            {
                "name": name,
                "ph": phase,
                "rel_ts": timestamp,
                "g_rk": rank,
                "dp_rk": 0,
                "pp_rk": pipeline_rank,
                "tp_rk": tensor_rank,
                **attrs,
            }
        )

    def collective(name: str, *, op: str, dim: str) -> None:
        event(name, "B", op=op, dim=dim, data_bytes=32768, group_size=2)
        event(name, "E", group=[tp_peer_rank])

    def linear_lifecycle(
        *,
        operation_id: str,
        collective_op: str,
        launch_site: str,
        payload_role: str,
        completion_site: str,
        wait_role: str,
        dim: str | None = "first",
        mismatch_completion: bool = False,
    ) -> None:
        route = {
            "operation_id": operation_id,
            "operation_id_scope": "rank_local",
            "execution_route": "local_linear_direct_async",
            "collective_op": collective_op,
            "data_bytes": 32768,
            "group_size": 2,
            "launch_site": launch_site,
            "pass_direction": "backward",
            "payload_role": payload_role,
        }
        if dim is not None:
            route["dim"] = dim
        event(
            "tp-linear-async-launch",
            "B",
            **route,
            async_op=True,
            completion_included=False,
            timing_phase="launch_attempt",
        )
        event(
            "tp-linear-async-launch",
            "E",
            api_returned=True,
            error_type=None,
        )
        if mismatch_completion:
            route["operation_id"] = f"{operation_id}:unknown"
        event(
            "tp-linear-async-complete",
            "B",
            **route,
            completion_guarantee="current_stream_after_wait",
            completion_included=True,
            completion_kind="work_wait",
            completion_site=completion_site,
            duration_attribution="per_request",
            global_device_completion_guaranteed=False,
            host_blocking_guaranteed=False,
            launch_observed=True,
            op="wait",
            terminal=True,
            timing_phase="stream_dependency",
            wait_role=wait_role,
        )
        event(
            "tp-linear-async-complete",
            "E",
            completed=True,
            error_type=None,
        )

    def sp_layernorm_sync() -> None:
        event(
            "sp-layernorm-allreduce",
            "B",
            data_bytes=512,
            group_size=2,
            reduce_op="SUM",
            grad_bucket="sum",
        )
        event("sp-layernorm-allreduce", "E", group=[tp_peer_rank])

    for iteration in (1, 2):
        rows.append(
            {
                "name": "iteration",
                "ph": "B",
                "pad_before": 0,
                "iteration": iteration,
            }
        )
        if include_te_model_scopes:
            for _layer in (1, 2):
                event("transformer_layer", "B")
                model_scopes = (
                    ("MLP.forward", "attention")
                    if reverse_te_model_scopes
                    else ("attention", "MLP.forward")
                )
                for name in model_scopes:
                    event(name, "B")
                    event(name, "E")
                event("transformer_layer", "E")
        if cross_all_gather_scopes:
            event(
                "tp-all-gather-first",
                "B",
                op="all-gather",
                dim="first",
                data_bytes=32768,
                group_size=2,
            )
            event(
                "tp-all-gather-last",
                "B",
                op="all-gather",
                dim="last",
                data_bytes=32768,
                group_size=2,
            )
            event("tp-all-gather-first", "E", group=[tp_peer_rank])
            event("tp-all-gather-last", "E", group=[tp_peer_rank])
        else:
            if include_first_all_gather:
                collective("tp-all-gather-first", op="all-gather", dim="first")
                collective("tp-all-gather-first", op="all-gather", dim="first")
            collective("tp-all-gather-last", op="all-gather", dim="last")
        event(
            "tp-reduce-scatter-last",
            "B",
            op="reduce-scatter",
            dim="last",
            data_bytes=32768,
            group_size=2,
        )
        if not omit_nested_reduce_scatter:
            collective("tp-reduce-scatter", op="reduce-scatter", dim="first")
        event("tp-reduce-scatter-last", "E", group=[tp_peer_rank])
        if include_linear_lifecycle:
            linear_lifecycle(
                operation_id=f"tp-linear:{rank}:{iteration}:all-gather",
                collective_op="all-gather",
                launch_site="linear_backward_wgrad_input_all_gather",
                payload_role="weight_gradient_input",
                completion_site="linear_backward_wgrad_input_ready",
                wait_role="dependency",
            )
            linear_lifecycle(
                operation_id=f"tp-linear:{rank}:{iteration}:reduce-scatter",
                collective_op="reduce-scatter",
                launch_site="linear_backward_dgrad_reduce_scatter",
                payload_role="input_gradient",
                completion_site="linear_backward_dgrad_reduce_scatter_return",
                wait_role="return",
                mismatch_completion=mismatched_linear_completion
                and rank == 0
                and iteration == 1,
            )
        if include_linear_allreduce:
            linear_lifecycle(
                operation_id=f"tp-linear:{rank}:{iteration}:all-reduce",
                collective_op="all-reduce",
                launch_site="linear_backward_dgrad_all_reduce",
                payload_role="input_gradient",
                completion_site="linear_backward_dgrad_all_reduce_return",
                wait_role="return",
                dim=None,
            )
        if include_final_grad_sync:
            event(
                "grad-sync",
                "B",
                schedule=grad_sync_schedule,
                timing_phase="framework_phase",
            )
            event("all-grads-sync", "B")
            if include_sp_sync and nest_sp_in_all_grads:
                sp_layernorm_sync()
            event("all-grads-sync", "E")
            if include_sp_sync and not nest_sp_in_all_grads:
                sp_layernorm_sync()
            if include_embedding_sync:
                event(
                    "embedding-grads-allreduce",
                    "B",
                    data_bytes=2048,
                    group_size=2,
                    embedding_kind="word",
                )
                event(
                    "embedding-grads-allreduce",
                    "E",
                    group=[embedding_peer_rank],
                )
            event("grad-sync", "E")
        rows.append(
            {
                "name": "iteration",
                "ph": "E",
                "iteration": iteration,
                "duration_wall": timestamp,
            }
        )

    trace_root.mkdir(parents=True, exist_ok=True)
    path = trace_root / (
        f"benchmark-global-{rank}-data-0-"
        f"pipeline-{pipeline_rank}-tensor-{tensor_rank}.json"
    )
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_tp2_collective_contract_accepts_first_last_hierarchy(tmp_path: Path) -> None:
    for rank in (0, 1):
        _write_collective_trace(tmp_path, rank=rank)

    assert tp_probe_contract.validate_tp2_gqa_collective_hierarchy(tmp_path) == ()


def test_tp2_collective_contract_requires_nested_physical_reduce_scatter(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            omit_nested_reduce_scatter=rank == 0,
        )

    failures = tp_probe_contract.validate_tp2_gqa_collective_hierarchy(tmp_path)

    assert "trace.tp.collective_hierarchy" in {
        failure.code for failure in failures
    }


def test_tp2_collective_contract_rejects_crossed_scopes(tmp_path: Path) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            cross_all_gather_scopes=rank == 0,
        )

    failures = tp_probe_contract.validate_tp2_gqa_collective_hierarchy(tmp_path)

    assert "trace.tp.nesting" in {failure.code for failure in failures}


def test_tp2_no_sp_collective_contract_accepts_gqa_last_dimension(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_first_all_gather=False,
        )

    assert (
        tp_probe_contract.validate_tp2_gqa_no_sp_collective_hierarchy(tmp_path)
        == ()
    )


def test_tp2_no_sp_collective_contract_rejects_first_all_gather(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(tmp_path, rank=rank)

    failures = tp_probe_contract.validate_tp2_gqa_no_sp_collective_hierarchy(
        tmp_path
    )

    assert "trace.tp.collective_count" in {
        failure.code for failure in failures
    }


def test_tp2_sp_linear_contract_accepts_all_gather_and_reduce_scatter(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_linear_lifecycle=True,
        )

    assert tp_probe_contract.validate_tp2_sp_linear_lifecycle(tmp_path) == ()


def test_tp2_sp_linear_contract_rejects_unknown_completion_identity(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_linear_lifecycle=True,
            mismatched_linear_completion=True,
        )

    failures = tp_probe_contract.validate_tp2_sp_linear_lifecycle(tmp_path)

    assert "trace.tp_linear.operation_id" in {
        failure.code for failure in failures
    }


def test_tp2_sp_linear_contract_rejects_allreduce_route(tmp_path: Path) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_linear_lifecycle=True,
            include_linear_allreduce=True,
        )

    failures = tp_probe_contract.validate_tp2_sp_linear_lifecycle(tmp_path)

    assert "trace.tp_linear.route" in {failure.code for failure in failures}


def test_tp2_sp_te_linear_contract_accepts_the_mcore_visible_boundary(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_linear_lifecycle=True,
            include_final_grad_sync=True,
            include_te_model_scopes=True,
        )

    assert tp_probe_contract.validate_tp2_sp_te_linear_profile(tmp_path) == ()


def test_tp2_sp_te_linear_contract_requires_the_outer_te_model_scopes(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_linear_lifecycle=True,
            include_final_grad_sync=True,
        )

    failures = tp_probe_contract.validate_tp2_sp_te_linear_profile(tmp_path)

    assert "trace.tp_te.scope_count" in {failure.code for failure in failures}


def test_tp2_sp_te_linear_contract_requires_attention_before_mlp(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_linear_lifecycle=True,
            include_final_grad_sync=True,
            include_te_model_scopes=True,
            reverse_te_model_scopes=True,
        )

    failures = tp_probe_contract.validate_tp2_sp_te_linear_profile(tmp_path)

    assert "trace.tp_te.scope_hierarchy" in {
        failure.code for failure in failures
    }


def test_tp2_no_sp_linear_contract_accepts_allreduce(tmp_path: Path) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_linear_allreduce=True,
        )

    assert tp_probe_contract.validate_tp2_local_allreduce_lifecycle(tmp_path) == ()


def test_tp2_no_sp_linear_contract_rejects_sp_routes(tmp_path: Path) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_linear_lifecycle=True,
        )

    failures = tp_probe_contract.validate_tp2_local_allreduce_lifecycle(tmp_path)

    assert "trace.tp_linear.route" in {failure.code for failure in failures}


def test_tp2_sp_final_sync_contract_accepts_layernorm_sibling(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_final_grad_sync=True,
        )

    assert tp_probe_contract.validate_tp2_sp_final_grad_sync(tmp_path) == ()


def test_tp2_sp_final_sync_contract_rejects_embedding_collective(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_final_grad_sync=True,
            include_embedding_sync=True,
        )

    failures = tp_probe_contract.validate_tp2_sp_final_grad_sync(tmp_path)

    assert "trace.tp.final_sync_count" in {
        failure.code for failure in failures
    }


def test_tp2_sp_final_sync_contract_rejects_layernorm_inside_all_grads(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_final_grad_sync=True,
            nest_sp_in_all_grads=True,
        )

    failures = tp_probe_contract.validate_tp2_sp_final_grad_sync(tmp_path)

    assert "trace.tp.final_sync_parent" in {
        failure.code for failure in failures
    }


def test_tp2_no_sp_final_sync_contract_accepts_dp_finish_only(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_final_grad_sync=True,
            include_sp_sync=False,
        )

    assert tp_probe_contract.validate_tp2_no_sp_final_grad_sync(tmp_path) == ()


def test_tp2_no_sp_final_sync_contract_rejects_layernorm_collective(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_final_grad_sync=True,
        )

    failures = tp_probe_contract.validate_tp2_no_sp_final_grad_sync(tmp_path)

    assert "trace.tp.final_sync_count" in {
        failure.code for failure in failures
    }


def test_tp2_pp2_embedding_final_sync_contract_accepts_two_parallel_axes(
    tmp_path: Path,
) -> None:
    for rank in range(4):
        pipeline_rank, tensor_rank = divmod(rank, 2)
        _write_collective_trace(
            tmp_path,
            rank=rank,
            pipeline_rank=pipeline_rank,
            tensor_rank=tensor_rank,
            tp_peer_rank=rank ^ 1,
            embedding_peer_rank=rank ^ 2,
            grad_sync_schedule="non-interleaved-1f1b",
            include_final_grad_sync=True,
            include_embedding_sync=True,
        )

    assert (
        tp_probe_contract.validate_tp2_pp2_embedding_final_grad_sync(tmp_path)
        == ()
    )


def test_tp2_pp2_embedding_final_sync_contract_rejects_wrong_pp_peer(
    tmp_path: Path,
) -> None:
    for rank in range(4):
        pipeline_rank, tensor_rank = divmod(rank, 2)
        _write_collective_trace(
            tmp_path,
            rank=rank,
            pipeline_rank=pipeline_rank,
            tensor_rank=tensor_rank,
            tp_peer_rank=rank ^ 1,
            embedding_peer_rank=3 if rank == 0 else rank ^ 2,
            grad_sync_schedule="non-interleaved-1f1b",
            include_final_grad_sync=True,
            include_embedding_sync=True,
        )

    failures = tp_probe_contract.validate_tp2_pp2_embedding_final_grad_sync(
        tmp_path
    )

    assert "trace.tp.final_sync_field" in {
        failure.code for failure in failures
    }


def test_tp2_pp2_embedding_final_sync_contract_requires_each_rank(
    tmp_path: Path,
) -> None:
    for rank in range(4):
        pipeline_rank, tensor_rank = divmod(rank, 2)
        _write_collective_trace(
            tmp_path,
            rank=rank,
            pipeline_rank=pipeline_rank,
            tensor_rank=tensor_rank,
            tp_peer_rank=rank ^ 1,
            embedding_peer_rank=rank ^ 2,
            grad_sync_schedule="non-interleaved-1f1b",
            include_final_grad_sync=True,
            include_embedding_sync=rank != 0,
        )

    failures = tp_probe_contract.validate_tp2_pp2_embedding_final_grad_sync(
        tmp_path
    )

    assert "trace.tp.final_sync_count" in {
        failure.code for failure in failures
    }


def test_tp2_local_allreduce_profile_combines_all_three_boundaries(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_first_all_gather=False,
            include_linear_allreduce=True,
            include_final_grad_sync=True,
            include_sp_sync=False,
        )

    assert tp_probe_contract.validate_tp2_local_allreduce_profile(tmp_path) == ()


def test_tp2_sp_profile_contract_combines_all_three_boundaries(
    tmp_path: Path,
) -> None:
    for rank in (0, 1):
        _write_collective_trace(
            tmp_path,
            rank=rank,
            include_linear_lifecycle=True,
            include_final_grad_sync=True,
        )

    assert tp_probe_contract.validate_tp2_sp_profile(tmp_path) == ()
