# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.test_utils.runners import check_ring_exchange
from tests.test_utils.runners import dp_probe_contract
from tests.test_utils.runners import dualpipev_probe_contract
from tests.test_utils.runners import gpt_probe_contract
from tests.test_utils.runners import megalens_run_manifest as manifest
from tests.test_utils.runners import moe_capacity_probe_contract
from tests.test_utils.runners import moe_flex_deepep_probe_contract
from tests.test_utils.runners import moe_flex_hybridep_probe_contract
from tests.test_utils.runners import moe_recompute_fp8_probe_contract
from tests.test_utils.runners import moe_shared_expert_overlap_probe_contract
from tests.test_utils.runners import moe_shared_expert_probe_contract
from tests.test_utils.runners import p2p_probe_contract
from tests.test_utils.runners import run_flagscale_megalens as gate
from tests.test_utils.runners import tp_probe_contract
from tests.test_utils.runners import training_run_contract

_FIXTURES = Path(__file__).parent / "fixtures"
_CONFIG_PROFILE_CASES = {
    "flagscale_single_node_smoke.yaml": "pp1",
    "flagscale_single_node_gpt_eager_full_smoke.yaml": "gpt-eager-full",
    "flagscale_single_node_cp2_te_smoke.yaml": "cp2-te",
    "flagscale_single_node_cp4_te_smoke.yaml": "cp4-te",
    "flagscale_single_node_tp2_sp_local_smoke.yaml": "tp2-sp-local",
    "flagscale_single_node_tp2_sp_te_linear_smoke.yaml": "tp2-sp-te-linear",
    "flagscale_single_node_tp2_sp_te_userbuffer_smoke.yaml": (
        "tp2-sp-te-userbuffer"
    ),
    "flagscale_single_node_tp2_sp_te_op_fuser_smoke.yaml": (
        "tp2-sp-te-op-fuser"
    ),
    "flagscale_single_node_qwen3_enron_tp2_sp.yaml": (
        "qwen3-enron-tp2-sp"
    ),
    "flagscale_single_node_qwen3_enron_tp4_sp.yaml": (
        "qwen3-enron-tp4-sp"
    ),
    "flagscale_single_node_qwen3_enron_tp4_local_no_sp.yaml": (
        "qwen3-enron-tp4-local-no-sp"
    ),
    "flagscale_single_node_qwen3_enron_tp8_sp.yaml": (
        "qwen3-enron-tp8-sp"
    ),
    "flagscale_single_node_qwen3_enron_cp2.yaml": "qwen3-enron-cp2",
    "flagscale_single_node_qwen3_enron_cp4.yaml": "qwen3-enron-cp4",
    "flagscale_single_node_deepseek_tp2_sp_mock.yaml": (
        "deepseek-tp2-sp-mock"
    ),
    "flagscale_single_node_tp2_local_allreduce_smoke.yaml": (
        "tp2-local-allreduce"
    ),
    "flagscale_single_node_tp2_pp2_embedding_smoke.yaml": "tp2-pp2-embedding",
    "flagscale_single_node_pp2_smoke.yaml": "pp2",
    "flagscale_single_node_pp2_batched_steady_smoke.yaml": (
        "pp2-batched-steady"
    ),
    "flagscale_single_node_pp2_unbatched_smoke.yaml": "pp2-unbatched",
    "flagscale_single_node_pp2_unbatched_warmup_flush_smoke.yaml": (
        "pp2-unbatched-warmup-flush"
    ),
    "flagscale_single_node_pp2_overlap_timeline_smoke.yaml": (
        "pp2-overlap-timeline"
    ),
    "flagscale_single_node_ep2_smoke.yaml": "ep2-alltoall",
    "flagscale_single_node_ep2_recompute_smoke.yaml": "ep2-recompute",
    "flagscale_single_node_ep2_fp8_smoke.yaml": "ep2-fp8",
    "flagscale_single_node_ep2_fp8_recompute_smoke.yaml": (
        "ep2-fp8-recompute"
    ),
    "flagscale_single_node_ep2_capacity_drop_smoke.yaml": (
        "ep2-alltoall-capacity-drop"
    ),
    "flagscale_single_node_ep2_shared_expert_smoke.yaml": (
        "ep2-alltoall-shared-expert"
    ),
    "flagscale_single_node_ep2_shared_expert_overlap_smoke.yaml": (
        "ep2-alltoall-shared-expert-overlap"
    ),
    "flagscale_single_node_tp2_ep4_flex_deepep_smoke.yaml": (
        "tp2-ep4-flex-deepep"
    ),
    "flagscale_single_node_tp2_ep4_flex_hybridep_smoke.yaml": (
        "tp2-ep4-flex-hybridep"
    ),
    "flagscale_single_node_ep2_fine_grained_smoke.yaml": "ep2-fine-grained",
    "flagscale_single_node_pp2_dp2_ep2_dualpipev_smoke.yaml": (
        "pp2-dp2-ep2-dualpipev"
    ),
    "flagscale_single_node_dp2_standard_smoke.yaml": "dp2-standard-ddp",
    "flagscale_single_node_dp2_standard_overlap_smoke.yaml": (
        "dp2-standard-ddp-overlap"
    ),
    "flagscale_single_node_dp2_distopt_smoke.yaml": "dp2-distopt",
    "flagscale_single_node_dp2_distopt_overlap_smoke.yaml": (
        "dp2-distopt-overlap"
    ),
    "flagscale_single_node_pp2_dp2_distopt_force_sync_smoke.yaml": (
        "pp2-dp2-distopt-force-sync"
    ),
    "flagscale_single_node_dp2_layerwise_overlap_smoke.yaml": (
        "dp2-layerwise-overlap"
    ),
    "flagscale_single_node_dp4_distopt_multi_instance_overlap_smoke.yaml": (
        "dp4-distopt-multi-instance-overlap"
    ),
    "flagscale_single_node_dp8_standard_smoke.yaml": "dp8-standard-ddp",
    "flagscale_single_node_dp8_standard_overlap_smoke.yaml": (
        "dp8-standard-ddp-overlap"
    ),
    "flagscale_single_node_dp8_distopt_smoke.yaml": "dp8-distopt",
    "flagscale_single_node_dp8_distopt_overlap_smoke.yaml": (
        "dp8-distopt-overlap"
    ),
    "flagscale_single_node_te_cuda_graph_attn_smoke.yaml": ("te-attn-cuda-graph"),
    "flagscale_single_node_te_cuda_graph_attn_mlp_recompute_smoke.yaml": (
        "te-attn-mlp-recompute-cuda-graph"
    ),
    "flagscale_single_node_te_cuda_graph_full_smoke.yaml": "te-full-cuda-graph",
    "flagscale_single_node_local_cuda_graph_full_smoke.yaml": (
        "local-layerwise-full-cuda-graph"
    ),
    "flagscale_single_node_local_cuda_graph_cupti_smoke.yaml": (
        "local-layerwise-cuda-kernels"
    ),
    "flagscale_single_node_cuda_graph_full_iteration_smoke.yaml": (
        "local-full-iteration-cuda-graph"
    ),
    "flagscale_single_node_te_cuda_graph_moe_router_smoke.yaml": (
        "te-moe-router-cuda-graph"
    ),
    "flagscale_single_node_bert_smoke.yaml": "bert-encoder",
    "flagscale_single_node_multimodule_bridge_smoke.yaml": (
        "multimodule-bridge2"
    ),
    "flagscale_single_node_multimodule_bridge_fanin.yaml": (
        "multimodule-bridge8-fanin"
    ),
    "flagscale_single_node_multimodule_bridge_fanout.yaml": (
        "multimodule-bridge8-fanout"
    ),
}
def _write_rank_trace(run_dir: Path, rank: int, event: str = "forward") -> None:
    trace_root = run_dir / "traces"
    trace_root.mkdir(exist_ok=True)
    fields = {
        "iteration": 2,
        "g_rk": rank,
        "dp_rk": rank,
        "pp_rk": 0,
        "tp_rk": 0,
    }
    path = trace_root / f"benchmark-global-{rank}-data-{rank}-pipeline-0-tensor-0.json"
    path.write_text(
        json.dumps(
            [
                {"name": event, "ph": "B", **fields},
                {"name": event, "ph": "E", **fields},
            ]
        ),
        encoding="utf-8",
    )


def _write_terminal_checkpoint(run_dir: Path, *, iteration: int = 2) -> None:
    checkpoint_root = run_dir / "checkpoints"
    iteration_root = checkpoint_root / f"iter_{iteration:07d}"
    iteration_root.mkdir(parents=True)
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
        str(iteration), encoding="utf-8"
    )
    (iteration_root / "common.pt").write_bytes(b"checkpoint")


def _write_flagcx_worker_logs(
    run_dir: Path,
    *,
    ranks: tuple[int, int] = (0, 1),
    fatal_marker: str | None = None,
) -> None:
    log_root = run_dir / "logs"
    log_root.mkdir()
    for host_rank, dp_rank in enumerate(ranks):
        suffix = f"\n{fatal_marker}" if fatal_marker and host_rank == 1 else ""
        (log_root / f"host_{host_rank}.output").write_text(
            "[default0]:host:0:0 [0] FLAGCX INFO Bootstrap : Using "
            f"bond0.2208:10.6.208.{99 + 19 * host_rank}<0>\n"
            "[default0]:host:0:0 [0] FLAGCX INFO rank "
            f"{dp_rank} nranks 2 - DONE{suffix}\n",
            encoding="utf-8",
        )


def _write_legacy_pp2_terminal_checkpoint(run_dir: Path) -> None:
    checkpoint_root = run_dir / "checkpoints"
    iteration_root = checkpoint_root / "iter_0000002"
    (checkpoint_root / "latest_checkpointed_iteration.txt").parent.mkdir(
        parents=True, exist_ok=True
    )
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
        "2", encoding="utf-8"
    )
    for pipeline_rank in range(2):
        rank_root = iteration_root / f"mp_rank_00_{pipeline_rank:03d}"
        rank_root.mkdir(parents=True)
        (rank_root / "model_optim_rng.pt").write_bytes(b"model")
        (rank_root / "distrib_optim.pt").write_bytes(b"optimizer")


def _write_gpt_phase_trace(
    trace_root: Path,
    *,
    rank: int,
    pipeline_rank: int,
    include_postprocess: bool,
    tensor_rank: int = 0,
    eager_layers: int = 0,
    external_mlp_layers: int = 0,
    recompute_mlp_layers: int = 0,
    include_optimizer: bool = False,
    include_backward: bool = False,
    eager_iterations: frozenset[int] | None = None,
    model_iterations: frozenset[int] | None = None,
    iteration_ids: tuple[int, ...] = (1, 2),
    microbatches: int = 1,
    include_schedule_finalize: bool = False,
    include_optimizer_postprocess: bool = True,
    p2p_route: str | None = None,
    ring_directional_wait: bool = False,
    kernel_iterations: frozenset[int] | None = None,
) -> None:
    trace_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
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
                "pp_rk": pipeline_rank,
                "tp_rk": tensor_rank,
                **attrs,
            }
        )

    def p2p_events(iteration: int) -> None:
        if p2p_route is None:
            return
        if p2p_route not in {"batch", "batch-steady", "unbatched", "ring"}:
            raise ValueError(f"unknown P2P route: {p2p_route}")

        if p2p_route in {"batch", "batch-steady"}:
            transport_api = "batch_isend_irecv"
            launch_pairing = "backend_dependent"
            completion_pairing = "position"
            launch_groups = (
                ("internal_wait", (("send-forward", "send_next"),)),
                ("internal_wait", (("recv-backward", "recv_next"),)),
            )
            if rank == 1:
                launch_groups = (
                    ("internal_wait", (("recv-forward", "recv_prev"),)),
                    ("internal_wait", (("send-backward", "send_prev"),)),
                )
            if p2p_route == "batch-steady":
                launch_groups = (
                    ("internal_wait", (("send-forward", "send_next"),)),
                    (
                        "internal_wait",
                        (
                            ("send-forward", "send_next"),
                            ("recv-backward", "recv_next"),
                        ),
                    ),
                    ("internal_wait", (("recv-backward", "recv_next"),)),
                )
                if rank == 1:
                    launch_groups = (
                        ("internal_wait", (("recv-forward", "recv_prev"),)),
                        (
                            "internal_wait",
                            (
                                ("send-backward", "send_prev"),
                                ("recv-forward", "recv_prev"),
                            ),
                        ),
                        ("internal_wait", (("send-backward", "send_prev"),)),
                    )
        elif p2p_route == "unbatched":
            transport_api = "isend_irecv"
            launch_pairing = "key"
            completion_pairing = "key"
            backward_mode = "internal_wait" if rank == 0 else "external_wait"
            launch_groups = (
                (
                    "internal_wait",
                    (
                        ("recv-forward", "recv_prev"),
                        ("send-forward", "send_next"),
                    ),
                ),
                (
                    backward_mode,
                    (
                        ("send-backward", "send_prev"),
                        ("recv-backward", "recv_next"),
                    ),
                ),
            )
        else:
            transport_api = "ring_exchange"
            launch_pairing = "none"
            completion_pairing = "none"
            launch_groups = (
                ("inline", (("send-forward", "send_next"),)),
                ("inline", (("recv-backward", "recv_next"),)),
            )
            if rank == 1:
                launch_groups = (
                    ("inline", (("recv-forward", "recv_prev"),)),
                    ("inline", (("send-backward", "send_prev"),)),
                )

        for index, (completion_mode, directional_events) in enumerate(launch_groups):
            batch_id = f"p2p:{iteration}:{index}"
            operations = []
            for event_name, key in directional_events:
                direction, pipeline_direction = event_name.split("-", 1)
                operation_id = f"{batch_id}:{key}"
                operations.append(
                    {
                        "operation_id": operation_id,
                        "request_id": (
                            None if p2p_route == "ring" else operation_id
                        ),
                        "direction": direction,
                        "pipeline_direction": pipeline_direction,
                        "peer_rank": 1 - rank,
                        "data_bytes": 32768,
                        "microbatch": None,
                        "comm_type": "p2p",
                        "backend": "nccl",
                        "transport_api": transport_api,
                        "completion_mode": completion_mode,
                    }
                )
            launch_fields = {
                "batch_id": batch_id,
                "comm_type": "p2p-launch",
                "backend": "nccl",
                "backends": ["nccl"],
                "backend_complete": True,
                "transport_api": transport_api,
                "request_pairing": launch_pairing,
                "completion_mode": completion_mode,
                "completion_included": False,
                "operation_count": len(operations),
                "operations": operations,
            }
            if p2p_route == "ring":
                launch_fields.update(
                    {
                        "timing_phase": "inline_api_call",
                        "api_return_included": True,
                        "completion_guarantee": "api_return_observed",
                        "completion_kind": "inline_api_return",
                        "device_completion_guaranteed": False,
                        "duration_attribution": "shared_nonexclusive",
                        "host_blocking_guaranteed": False,
                        "operation_ids": [
                            operation["operation_id"] for operation in operations
                        ],
                        "operation_id_scope": "rank_local",
                        "physical_request_count": 0,
                        "stage": "p2p_inline_api",
                    }
                )
            event("p2p-launch", "B", **launch_fields)
            event(
                "p2p-launch",
                "E",
                **({"completed": True} if p2p_route == "ring" else {}),
            )
            if p2p_route == "ring":
                if ring_directional_wait:
                    operation = operations[0]
                    event(
                        str(directional_events[0][0]),
                        "B",
                        **operation,
                        completion_kind="work_wait",
                    )
                    event(str(directional_events[0][0]), "E", completed=True)
                operation_ids = [
                    operation["operation_id"] for operation in operations
                ]
                event(
                    "p2p-batch-device-sync",
                    "B",
                    batch_id=batch_id,
                    comm_type="p2p",
                    backend="nccl",
                    backends=["nccl"],
                    backend_complete=True,
                    transport_api="ring_exchange",
                    request_pairing="none",
                    completion_site="batch_p2p_sync_workaround",
                    completion_included=True,
                    completion_kind="device_synchronize",
                    operation_count=len(operations),
                    operation_ids=operation_ids,
                    operations=operations,
                    physical_request_count=0,
                )
                event(
                    "p2p-batch-device-sync",
                    "E",
                    completed=True,
                    device_completion_guaranteed=True,
                    error_type=None,
                    host_blocking_guaranteed=True,
                )
                continue
            completion_site = (
                "communicate_internal_wait"
                if completion_mode == "internal_wait"
                else "exposed_request_wait"
            )
            operation_ids = [
                operation["operation_id"] for operation in operations
            ]
            if (
                p2p_route in {"batch", "batch-steady"}
                and len(operations) > 1
            ):
                event(
                    "p2p-batch-complete",
                    "B",
                    batch_id=batch_id,
                    comm_type="p2p",
                    backend="nccl",
                    backends=["nccl"],
                    backend_complete=True,
                    completion_guarantee="current_stream_after_wait",
                    completion_included=True,
                    completion_kind="aggregate_work_wait",
                    completion_mode=completion_mode,
                    duration_attribution="shared_nonexclusive",
                    host_blocking_guaranteed=False,
                    op="wait",
                    operation_count=len(operations),
                    operation_ids=operation_ids,
                    operation_id_scope="rank_local",
                    operations=operations,
                    physical_request_count=1,
                    request_id=f"{batch_id}:aggregate",
                    request_pairing="aggregate",
                    stage="batch_p2p_completion",
                    timing_phase="stream_dependency",
                    transport_api=transport_api,
                )
                event(
                    "p2p-batch-complete",
                    "E",
                    completed=True,
                    error_type=None,
                )
            else:
                for operation in operations:
                    event_name = (
                        f"{operation['direction']}-"
                        f"{operation['pipeline_direction']}"
                    )
                    operation_id = operation["operation_id"]
                    event(
                        event_name,
                        "B",
                        **operation,
                        batch_id=batch_id,
                        request_pairing=completion_pairing,
                        completion_site=completion_site,
                        completion_included=True,
                        completion_kind="work_wait",
                        operation_count=1,
                        operation_ids=[operation_id],
                    )
                    event(event_name, "E", completed=True, error_type=None)
            if p2p_route in {"batch", "batch-steady"}:
                event(
                    "p2p-batch-device-sync",
                    "B",
                    batch_id=batch_id,
                    comm_type="p2p",
                    backend="nccl",
                    backends=["nccl"],
                    backend_complete=True,
                    transport_api=transport_api,
                    request_pairing=(
                        "aggregate" if len(operations) > 1 else "position"
                    ),
                    completion_site="batch_p2p_sync_workaround",
                    completion_included=True,
                    completion_kind="device_synchronize",
                    operation_count=len(operations),
                    operation_ids=operation_ids,
                    operations=operations,
                    physical_request_count=1,
                )
                event(
                    "p2p-batch-device-sync",
                    "E",
                    completed=True,
                    device_completion_guaranteed=True,
                    error_type=None,
                    host_blocking_guaranteed=True,
                )

    for iteration in iteration_ids:
        rows.append(
            {
                "name": "iteration",
                "ph": "B",
                "pad_before": 0,
                "iteration": iteration,
            }
        )
        if model_iterations is None or iteration in model_iterations:
            layer_count = (
                eager_layers
                if eager_iterations is None or iteration in eager_iterations
                else 0
            )
            for microbatch in range(microbatches):
                event(
                    "forward-step",
                    "B",
                    current_microbatch=microbatch,
                )
                event("decoder", "B")
                for _ in range(layer_count):
                    event("transformer_layer", "B")
                    event("_forward_attention", "B")
                    event("attention", "B")
                    event("attention", "E")
                    event("_forward_attention", "E")
                    event("_forward_mlp", "B")
                    event("MLP.forward", "B")
                    event("MLP.forward", "E")
                    event("_forward_mlp", "E")
                    event("transformer_layer", "E")
                for _ in range(external_mlp_layers):
                    event("MLP.forward", "B")
                    event("MLP.forward", "E")
                event("decoder", "E")
                event("decoder-postprocess", "B")
                if include_postprocess:
                    event("output_layer", "B")
                    event("output_layer", "E")
                    event("loss", "B")
                    event("loss", "E")
                event("decoder-postprocess", "E")
                if include_schedule_finalize:
                    event(
                        "forward-step-calc-loss",
                        "B",
                        current_microbatch=microbatch,
                    )
                    event("forward-step-calc-loss", "E")
                event("forward-step", "E")
                if include_backward:
                    event(
                        "backward-step",
                        "B",
                        current_microbatch=microbatch,
                    )
                    for _ in range(recompute_mlp_layers):
                        event("MLP.forward", "B")
                        event("MLP.forward", "E")
                    event("backward-step", "E")
            if include_schedule_finalize and include_backward:
                event("grad-sync", "B")
                event("all-grads-sync", "B")
                event("all-grads-sync", "E")
                event("grad-sync", "E")
        p2p_events(iteration)
        if include_optimizer:
            event("optimizer", "B")
            event("optimizer-step", "B")
            event("optimizer-step", "E")
            event("optimizer", "E")
            if include_optimizer_postprocess:
                event("optimizer-postprocess", "B")
                event("optimizer-postprocess", "E")
        rows.append(
            {
                "name": "iteration",
                "ph": "E",
                "iteration": iteration,
                "duration_wall": timestamp,
            }
        )

    for iteration in sorted(kernel_iterations or ()):
        start_us = iteration * 100
        end_us = start_us + 20
        iter_rel_start_us = 50
        iter_rel_end_us = iter_rel_start_us + 20
        rows.append(
            {
                "record_type": "cuda_kernel",
                "name": "transformer_engine::model_gemm",
                "ph": "X",
                "start_us": start_us,
                "end_us": end_us,
                "wall_start_us": 10_000 + start_us,
                "wall_end_us": 10_000 + end_us,
                "iter_rel_start_us": iter_rel_start_us,
                "iter_rel_end_us": iter_rel_end_us,
                "duration_us": end_us - start_us,
                "device": rank,
                "iteration": iteration,
                "g_rk": rank,
                "dp_rk": 0,
                "pp_rk": pipeline_rank,
                "tp_rk": tensor_rank,
            }
        )

    path = (
        trace_root
        / f"benchmark-global-{rank}-data-0-pipeline-{pipeline_rank}-"
        f"tensor-{tensor_rank}.json"
    )
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_pp2_unbatched_warmup_flush_trace(
    trace_root: Path,
    *,
    rank: int,
    kernel_timeline: bool = False,
) -> None:
    # Reuse the locked profile table so the fixture focuses on trace encoding.
    launch_specs = p2p_probe_contract._WARMUP_FLUSH_LAUNCHES
    timelines = p2p_probe_contract._WARMUP_FLUSH_TIMELINE
    compute_specs = p2p_probe_contract._WARMUP_FLUSH_COMPUTE
    request_keys = {
        "send-forward": "send_next",
        "recv-forward": "recv_prev",
        "send-backward": "send_prev",
        "recv-backward": "recv_next",
    }
    trace_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
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
                "pp_rk": rank,
                "tp_rk": 0,
                **attrs,
            }
        )

    for iteration in (1, 2):
        rows.append(
            {
                "name": "iteration",
                "ph": "B",
                "pad_before": 0,
                "iteration": iteration,
            }
        )
        for name, microbatch, vp_stage in compute_specs[rank]:
            event(
                name,
                "B",
                current_microbatch=microbatch,
                vp_stage=vp_stage,
                is_first_microbatch=microbatch == 0,
                is_last_stage=rank == 1 and vp_stage == 1,
                timing_phase="framework_phase",
            )
            event(
                name,
                "E",
                operation_id=f"pp:microbatch={microbatch}:vp={vp_stage}",
            )

        launch_records: list[tuple[dict[str, object], ...]] = []
        for launch_index, (operation_names, completion_mode) in enumerate(
            launch_specs[rank]
        ):
            batch_id = f"p2p:{rank}:{iteration}:{launch_index}"
            operations = []
            for event_name in operation_names:
                direction, pipeline_direction = event_name.split("-", 1)
                request_key = request_keys[event_name]
                operation_id = f"{batch_id}:{request_key}"
                operations.append(
                    {
                        "operation_id": operation_id,
                        "request_id": operation_id,
                        "direction": direction,
                        "pipeline_direction": pipeline_direction,
                        "peer_rank": 1 - rank,
                        "data_bytes": 32768,
                        "microbatch": None,
                        "comm_type": "p2p",
                        "backend": "nccl",
                        "transport_api": "isend_irecv",
                        "completion_mode": completion_mode,
                    }
                )
            launch_records.append(tuple(operations))

        for action, launch_index, event_name in timelines[rank]:
            operations = launch_records[launch_index]
            completion_mode = launch_specs[rank][launch_index][1]
            batch_id = str(operations[0]["operation_id"]).rsplit(":", 1)[0]
            if action == "launch":
                event(
                    "p2p-launch",
                    "B",
                    batch_id=batch_id,
                    comm_type="p2p-launch",
                    timing_phase="launch",
                    backend="nccl",
                    backends=["nccl"],
                    backend_complete=True,
                    transport_api="isend_irecv",
                    request_pairing="key",
                    completion_mode=completion_mode,
                    completion_included=False,
                    operation_count=len(operations),
                    operations=list(operations),
                )
                event("p2p-launch", "E")
                continue

            operation = next(
                operation
                for operation in operations
                if (
                    f"{operation['direction']}-"
                    f"{operation['pipeline_direction']}"
                )
                == event_name
            )
            operation_id = str(operation["operation_id"])
            completion_site = (
                "communicate_internal_wait"
                if completion_mode == "internal_wait"
                else "exposed_request_wait"
            )
            event(
                str(event_name),
                "B",
                **operation,
                batch_id=batch_id,
                completion_guarantee="current_stream_after_wait",
                completion_included=True,
                completion_kind="work_wait",
                completion_site=completion_site,
                duration_attribution="per_request",
                host_blocking_guaranteed=False,
                op="wait",
                operation_count=1,
                operation_ids=[operation_id],
                operation_id_scope="rank_local",
                request_pairing="key",
                stage="p2p_request_completion",
                timeout_supplied=False,
                timing_phase="stream_dependency",
            )
            event(str(event_name), "E", completed=True, error_type=None)

        rows.append(
            {
                "name": "iteration",
                "ph": "E",
                "iteration": iteration,
                "duration_wall": timestamp,
            }
        )

    if kernel_timeline:
        p2p_start, p2p_end = 100, 200
        compute_start, compute_end = ((150, 160) if rank == 1 else (250, 260))
        for name, start, end in (
            (
                "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)",
                p2p_start,
                p2p_end,
            ),
            ("transformer_engine::model_gemm", compute_start, compute_end),
        ):
            rows.append(
                {
                    "record_type": "cuda_kernel",
                    "name": name,
                    "ph": "X",
                    "start_us": start,
                    "end_us": end,
                    "wall_start_us": 1000 + start,
                    "wall_end_us": 1000 + end,
                    "iter_rel_start_us": start,
                    "iter_rel_end_us": end,
                    "duration_us": end - start,
                    "device": rank,
                    "iteration": 1,
                    "g_rk": rank,
                    "dp_rk": 0,
                    "pp_rk": rank,
                    "tp_rk": 0,
                }
            )

    path = (
        trace_root
        / f"benchmark-global-{rank}-data-0-pipeline-{rank}-tensor-0.json"
    )
    path.write_text(json.dumps(rows), encoding="utf-8")


def _invoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "trace-on",
    image: str = "example/flagscale:dev",
    child_returncode: int = 0,
    config: str = "flagscale_single_node_smoke.yaml",
    write_checkpoint: bool = True,
) -> tuple[int, Path, tuple[str, ...]]:
    run_dir = tmp_path / "run"
    observed_command: tuple[str, ...] = ()

    def fake_run_foreground(
        argv: tuple[str, ...],
        *,
        env: dict[str, str],
        launcher_log: Path,
        **_: object,
    ) -> gate.ExecutionResult:
        nonlocal observed_command
        observed_command = tuple(argv)
        launcher_log.write_text("official FlagScale entrypoint\n", encoding="utf-8")
        if child_returncode == 0:
            run_root = Path(env["MEGALENS_GATE_RUN_DIR"])
            if write_checkpoint:
                _write_terminal_checkpoint(run_root)
            if mode == "trace-on":
                _write_rank_trace(run_root, 0)
        return gate.ExecutionResult(child_returncode)

    monkeypatch.setattr(gate, "run_foreground", fake_run_foreground)
    result = gate.main(
        (
            "--run-dir",
            str(run_dir),
            "--input-config",
            str(_FIXTURES / config),
            "--mode",
            mode,
            "--image",
            image,
            "--controller-revision",
            "legacy-value-is-accepted",
        )
    )
    return result, run_dir, observed_command


@pytest.mark.parametrize(
    ("config", "profile"),
    tuple(_CONFIG_PROFILE_CASES.items()),
)
def test_existing_yaml_profiles_remain_selectable(config: str, profile: str) -> None:
    args = gate._parser().parse_args(
        (
            "--run-dir",
            "unused",
            "--input-config",
            str(_FIXTURES / config),
            "--mode",
            "trace-on",
            "--image",
            "example/flagscale:dev",
        )
    )

    assert gate._profile_from_arguments(args).name == profile


@pytest.mark.parametrize(
    "config_name",
    (
        "flagscale_dual_node_dp2_standard_flagcx.yaml",
        "flagscale_dual_node_dp2_distopt_flagcx.yaml",
        "flagscale_dual_node_qwen3_enron_cp2_dp4.yaml",
        "flagscale_dual_node_qwen3_enron_cp2_dp8.yaml",
    ),
)
def test_dual_node_config_is_reserved_for_offline_validation(
    config_name: str,
) -> None:
    args = gate._parser().parse_args(
        (
            "--run-dir",
            "unused",
            "--input-config",
            str(_FIXTURES / config_name),
            "--mode",
            "trace-on",
            "--image",
            "example/flagscale:dev",
        )
    )

    with pytest.raises(ValueError, match="manual dual-node orchestration"):
        gate._profile_from_arguments(args)


@pytest.mark.parametrize(
    ("config_stem", "profile_name", "event_names"),
    (
        (
            "flagscale_dual_node_dp2_standard_flagcx",
            "flagcx-dp2-standard",
            {"dp-allreduce"},
        ),
        (
            "flagscale_dual_node_dp2_distopt_flagcx",
            "flagcx-dp2-distopt",
            {"dp-reduce-scatter", "dp-param-all-gather"},
        ),
    ),
)
def test_dual_node_flagcx_offline_profiles_keep_the_dp_event_contract(
    config_stem: str, profile_name: str, event_names: set[str]
) -> None:
    profile = gate._OFFLINE_CONFIG_PROFILES[config_stem]

    assert profile.name == profile_name
    assert profile.rank_count == 2
    assert {requirement.name for requirement in profile.events} == event_names
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_flagcx_checkpoint
    )


def test_gpt_eager_profile_disables_persistent_layernorm() -> None:
    payload = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_gpt_eager_full_smoke.yaml"
        ).read_text(encoding="utf-8")
    )

    assert payload["train"]["model"]["transformer_impl"] == "local"
    assert payload["train"]["model"]["no_persist_layer_norm"] is True


def test_te_attention_mlp_recompute_profile_only_adds_recompute() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_te_cuda_graph_attn_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    recompute = yaml.safe_load(
        (
            _FIXTURES
            / "flagscale_single_node_te_cuda_graph_attn_mlp_recompute_smoke.yaml"
        ).read_text(encoding="utf-8")
    )

    baseline["experiment"]["exp_name"] = recompute["experiment"]["exp_name"]
    baseline["train"]["model"].update(
        {
            "recompute_granularity": "selective",
            "recompute_modules": ["mlp"],
        }
    )

    assert recompute == baseline
    assert recompute["train"]["system"]["trace_cupti_kernels"] == "off"

    profile = gate.PROFILES["te-attn-mlp-recompute-cuda-graph"]
    assert profile.rank_count == 1
    assert profile.contract is (
        gpt_probe_contract.validate_te_attention_cuda_graph_mlp_recompute_phases
    )
    assert profile.run_contract is (
        training_run_contract.validate_two_iteration_te_attention_mlp_recompute_checkpoint
    )


def test_te_full_cuda_graph_profile_only_changes_the_graph_lifecycle() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_te_cuda_graph_attn_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    full = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_te_cuda_graph_full_smoke.yaml"
        ).read_text(encoding="utf-8")
    )

    baseline["experiment"]["exp_name"] = full["experiment"]["exp_name"]
    baseline["train"]["model"].pop("cuda_graph_scope")
    baseline["train"]["model"]["cuda_graph_warmup_steps"] = 1

    assert full == baseline
    assert "cuda_graph_scope" not in full["train"]["model"]

    profile = gate.PROFILES["te-full-cuda-graph"]
    assert profile.rank_count == 1
    assert (
        profile.contract
        is gpt_probe_contract.validate_te_full_cuda_graph_phases
    )
    assert profile.run_contract is (
        training_run_contract.validate_two_iteration_te_full_cuda_graph_checkpoint
    )


def test_local_layerwise_full_cuda_graph_profile_only_changes_the_graph_owner() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_te_cuda_graph_full_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    local = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_local_cuda_graph_full_smoke.yaml"
        ).read_text(encoding="utf-8")
    )

    baseline["experiment"]["exp_name"] = local["experiment"]["exp_name"]
    baseline["train"]["model"]["cuda_graph_impl"] = "local"

    assert local == baseline
    assert "cuda_graph_scope" not in local["train"]["model"]

    profile = gate.PROFILES["local-layerwise-full-cuda-graph"]
    assert profile.rank_count == 1
    assert profile.contract is (
        gpt_probe_contract.validate_local_layerwise_full_cuda_graph_phases
    )
    assert profile.run_contract is (
        training_run_contract.validate_two_iteration_local_layerwise_full_cuda_graph_checkpoint
    )


def test_local_cuda_graph_kernel_profile_only_enables_cupti() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_local_cuda_graph_full_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    cupti = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_local_cuda_graph_cupti_smoke.yaml"
        ).read_text(encoding="utf-8")
    )

    baseline["experiment"]["exp_name"] = cupti["experiment"]["exp_name"]
    baseline["train"]["system"]["trace_cupti_kernels"] = "on"

    assert cupti == baseline
    profile = gate.PROFILES["local-layerwise-cuda-kernels"]
    assert profile.rank_count == 1
    assert profile.events == gate.PROFILES["local-layerwise-full-cuda-graph"].events
    assert profile.contract is (
        gpt_probe_contract.validate_local_layerwise_cuda_graph_kernel_phases
    )
    assert profile.run_contract is (
        training_run_contract.validate_two_iteration_local_layerwise_cuda_graph_kernel_checkpoint
    )


def test_local_full_iteration_cuda_graph_profile_only_changes_its_owner_and_schedule() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_te_cuda_graph_full_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    full_iteration = yaml.safe_load(
        (
            _FIXTURES
            / "flagscale_single_node_cuda_graph_full_iteration_smoke.yaml"
        ).read_text(encoding="utf-8")
    )

    baseline["experiment"]["exp_name"] = full_iteration["experiment"][
        "exp_name"
    ]
    baseline["experiment"]["save_steps"] = 3
    baseline["train"]["system"]["checkpoint"].update(
        {"save_interval": 3, "load": None}
    )
    baseline["train"]["model"].update(
        {
            "cuda_graph_impl": "local",
            "cuda_graph_scope": ["full_iteration"],
            "optimizer_cuda_graph": False,
            "no_check_for_nan_in_loss_and_grad": True,
            "global_batch_size": 2,
            "train_iters": 3,
        }
    )
    baseline["train"]["model"]["optimizer"]["lr_scheduler"][
        "lr_decay_iters"
    ] = 3

    assert full_iteration == baseline
    profile = gate.PROFILES["local-full-iteration-cuda-graph"]
    assert profile.rank_count == 1
    assert profile.contract is (
        gpt_probe_contract.validate_local_full_iteration_cuda_graph_phases
    )
    assert profile.run_contract is (
        training_run_contract.validate_three_iteration_local_full_cuda_graph_checkpoint
    )


def test_ep2_capacity_drop_profile_only_enables_unpadded_token_dropping() -> None:
    baseline = yaml.safe_load(
        (_FIXTURES / "flagscale_single_node_ep2_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    capacity_drop = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_ep2_capacity_drop_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = capacity_drop["experiment"]["exp_name"]
    baseline["train"]["model"].update(
        {
            "moe_expert_capacity_factor": 0.5,
            "moe_token_drop_policy": "probs",
            "moe_pad_expert_input_to_capacity": False,
        }
    )

    assert capacity_drop == baseline
    profile = gate.PROFILES["ep2-alltoall-capacity-drop"]
    assert profile.rank_count == 2
    assert profile.contract is moe_capacity_probe_contract.validate_ep2_capacity_drop
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_checkpoint
    )
    requirements = {requirement.name: requirement for requirement in profile.events}
    assert {
        "moe-router",
        "moe-dispatch",
        "moe-experts",
        "moe-combine",
        "ep-alltoall-dispatch",
        "ep-alltoall-combine",
    } == set(requirements)
    assert {
        "num_tokens",
        "routed_tokens",
        "dropped_tokens",
        "drop_rate",
    } <= set(requirements["moe-router"].fields)
    assert "capacity_factor" in requirements["moe-dispatch"].fields


@pytest.mark.parametrize(
    ("config_name", "profile_name", "recompute", "fp8", "contract"),
    (
        (
            "flagscale_single_node_ep2_recompute_smoke.yaml",
            "ep2-recompute",
            True,
            False,
            moe_recompute_fp8_probe_contract.validate_ep2_recompute,
        ),
        (
            "flagscale_single_node_ep2_fp8_smoke.yaml",
            "ep2-fp8",
            False,
            True,
            moe_recompute_fp8_probe_contract.validate_ep2_fp8,
        ),
        (
            "flagscale_single_node_ep2_fp8_recompute_smoke.yaml",
            "ep2-fp8-recompute",
            True,
            True,
            moe_recompute_fp8_probe_contract.validate_ep2_fp8_recompute,
        ),
    ),
)
def test_ep2_recompute_fp8_profiles_only_enable_the_selected_route(
    config_name: str,
    profile_name: str,
    recompute: bool,
    fp8: bool,
    contract,
) -> None:
    baseline = yaml.safe_load(
        (_FIXTURES / "flagscale_single_node_ep2_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    selected = yaml.safe_load((_FIXTURES / config_name).read_text(encoding="utf-8"))
    baseline["experiment"]["exp_name"] = selected["experiment"]["exp_name"]
    if recompute:
        baseline["train"]["system"].update(
            {
                "recompute_granularity": "selective",
                "recompute_modules": ["moe"],
            }
        )
    if fp8:
        baseline["train"]["system"]["precision"].update(
            {"fp8_format": "hybrid", "fp8_recipe": "delayed"}
        )

    assert selected == baseline

    # FlagScale runner_train merges these three groups and recursively flattens
    # leaf names, replacing underscores with dashes.
    flattened: list[str] = []

    def flatten(values: dict[str, object]) -> None:
        for key, value in values.items():
            option = f"--{key.replace('_', '-')}"
            if isinstance(value, dict):
                flatten(value)
            elif isinstance(value, list):
                flattened.extend((option, *(str(item) for item in value)))
            elif isinstance(value, bool):
                if value:
                    flattened.append(option)
            else:
                flattened.extend((option, str(value)))

    for group in ("system", "model", "data"):
        flatten(selected["train"][group])

    if recompute:
        assert "--recompute-granularity" in flattened
        assert flattened[flattened.index("--recompute-granularity") + 1] == "selective"
        assert "--recompute-modules" in flattened
        assert flattened[flattened.index("--recompute-modules") + 1] == "moe"
    else:
        assert "--recompute-granularity" not in flattened
        assert "--recompute-modules" not in flattened
    if fp8:
        assert "--fp8-format" in flattened
        assert flattened[flattened.index("--fp8-format") + 1] == "hybrid"
        assert "--fp8-recipe" in flattened
        assert flattened[flattened.index("--fp8-recipe") + 1] == "delayed"
    else:
        assert "--fp8-format" not in flattened
        assert "--fp8-recipe" not in flattened
    assert "--fp8" not in flattened

    profile = gate.PROFILES[profile_name]
    assert profile.rank_count == 2
    assert profile.contract is contract
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_checkpoint
    )
    assert gate._ep_dispatcher(profile, None) == "alltoall"
    assert {
        "moe-router",
        "moe-dispatch",
        "moe-experts",
        "moe-combine",
        "ep-alltoall-dispatch",
        "ep-alltoall-combine",
    } == {requirement.name for requirement in profile.events}


def test_ep2_shared_expert_profile_only_enables_nonoverlap_shared_compute() -> None:
    baseline = yaml.safe_load(
        (_FIXTURES / "flagscale_single_node_ep2_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    shared_expert = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_ep2_shared_expert_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = shared_expert["experiment"]["exp_name"]
    baseline["train"]["model"].update(
        {
            "moe_shared_expert_intermediate_size": 256,
            "moe_shared_expert_overlap": False,
        }
    )

    assert shared_expert == baseline
    profile = gate.PROFILES["ep2-alltoall-shared-expert"]
    assert profile.rank_count == 2
    assert (
        profile.contract
        is moe_shared_expert_probe_contract.validate_ep2_shared_expert
    )
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_checkpoint
    )
    requirements = {requirement.name: requirement for requirement in profile.events}
    assert {
        "moe-router",
        "moe-dispatch",
        "moe-experts",
        "moe-shared-expert",
        "moe-combine",
        "ep-alltoall-dispatch",
        "ep-alltoall-combine",
    } == set(requirements)
    assert {"layer", "ep_size"} <= set(requirements["moe-shared-expert"].fields)


def test_ep2_shared_expert_overlap_profile_only_enables_overlap() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_ep2_shared_expert_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    overlap = yaml.safe_load(
        (
            _FIXTURES
            / "flagscale_single_node_ep2_shared_expert_overlap_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = overlap["experiment"]["exp_name"]
    baseline["train"]["model"]["moe_shared_expert_overlap"] = True

    assert overlap == baseline
    profile = gate.PROFILES["ep2-alltoall-shared-expert-overlap"]
    assert profile.rank_count == 2
    assert (
        profile.contract
        is moe_shared_expert_overlap_probe_contract.validate_ep2_shared_expert_overlap
    )
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_checkpoint
    )
    requirements = {requirement.name: requirement for requirement in profile.events}
    assert {
        "moe-router",
        "moe-dispatch",
        "moe-experts",
        "moe-shared-expert",
        "moe-combine",
        "ep-alltoall-dispatch",
        "ep-alltoall-combine",
    } == set(requirements)
    assert {"layer", "ep_size", "stage"} <= set(
        requirements["moe-shared-expert"].fields
    )


def test_tp2_ep4_flex_deepep_profile_selects_the_fused_source_route() -> None:
    baseline = yaml.safe_load(
        (_FIXTURES / "flagscale_single_node_ep2_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    flex_deepep = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_ep4_flex_deepep_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = flex_deepep["experiment"]["exp_name"]
    baseline["experiment"]["runner"]["nproc_per_node"] = 8
    baseline["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    baseline["train"]["system"].update(
        {
            "tensor_model_parallel_size": 2,
            "expert_model_parallel_size": 4,
            "sequence_parallel": True,
        }
    )
    baseline["train"]["model"].update(
        {
            "num_experts": 8,
            "moe_router_dtype": "fp32",
            "moe_flex_dispatcher_backend": "deepep",
            "global_batch_size": 4,
        }
    )

    assert flex_deepep == baseline
    profile = gate.PROFILES["tp2-ep4-flex-deepep"]
    assert profile.rank_count == 8
    assert (
        profile.contract
        is moe_flex_deepep_probe_contract.validate_tp2_ep4_flex_deepep
    )
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_checkpoint
    )
    requirements = {requirement.name: requirement for requirement in profile.events}
    assert {
        "moe-router",
        "moe-dispatch",
        "moe-experts",
        "moe-combine",
        "ep-alltoall-dispatch",
        "ep-alltoall-combine",
        "tp-all-gather-first",
        "tp-reduce-scatter",
        "tp-allreduce",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
    } == set(requirements)
    for name in ("ep-alltoall-dispatch", "ep-alltoall-combine"):
        assert requirements[name].phase == "E"
        assert {
            "comm_type",
            "dispatcher",
            "data_bytes",
            "group_size",
            "ep_size",
            "tp_size",
        } <= set(requirements[name].fields)
    for name in ("tp-all-gather-first", "tp-reduce-scatter", "tp-allreduce"):
        assert {"data_bytes", "group_size"} <= set(requirements[name].fields)
    for name in ("tp-linear-async-launch", "tp-linear-async-complete"):
        assert {"operation_id", "collective_op"} <= set(
            requirements[name].fields
        )
    assert gate._ep_dispatcher(profile, None) == "flex"
    assert flex_deepep["train"]["system"]["tensor_model_parallel_size"] == 2
    assert flex_deepep["train"]["system"]["expert_tensor_parallel_size"] == 1
    data_parallel_size = 8 // flex_deepep["train"]["system"][
        "tensor_model_parallel_size"
    ]
    assert flex_deepep["train"]["model"]["global_batch_size"] == (
        flex_deepep["train"]["model"]["micro_batch_size"] * data_parallel_size
    )


def test_tp2_ep4_flex_hybridep_profile_only_changes_backend_and_jit_environment() -> None:
    flex_deepep = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_ep4_flex_deepep_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    flex_hybridep = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_ep4_flex_hybridep_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    flex_deepep["experiment"]["exp_name"] = flex_hybridep["experiment"]["exp_name"]
    flex_deepep["experiment"]["cmds"]["before_start"] = flex_hybridep["experiment"][
        "cmds"
    ]["before_start"]
    flex_deepep["train"]["model"]["moe_flex_dispatcher_backend"] = "hybridep"

    assert flex_hybridep == flex_deepep
    assert flex_hybridep["experiment"]["cmds"]["before_start"].endswith(
        "source /workspace/Megatron-LM-FL/docker/common/hybridep-cu128-env.sh"
    )
    assert flex_hybridep["train"]["model"]["moe_router_dtype"] == "fp32"
    assert "moe_enable_deepep" not in flex_hybridep["train"]["model"]

    profile = gate.PROFILES["tp2-ep4-flex-hybridep"]
    assert profile.rank_count == 8
    assert (
        profile.contract
        is moe_flex_hybridep_probe_contract.validate_tp2_ep4_flex_hybridep
    )
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_checkpoint
    )
    hybrid_events = {requirement.name for requirement in profile.events}
    deepep_events = {
        requirement.name
        for requirement in gate.PROFILES["tp2-ep4-flex-deepep"].events
    }
    assert deepep_events - hybrid_events == {
        "tp-all-gather-first",
        "tp-reduce-scatter",
        "tp-allreduce",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
    }
    assert gate._ep_dispatcher(profile, None) == "flex"


def test_dualpipev_profile_is_the_minimal_supported_ep2_route() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_ep2_fine_grained_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    dualpipev = yaml.safe_load(
        (
            _FIXTURES
            / "flagscale_single_node_pp2_dp2_ep2_dualpipev_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = dualpipev["experiment"]["exp_name"]
    system = baseline["train"]["system"]
    system.pop("num_layers_per_virtual_pipeline_stage")
    system.pop("microbatch_group_size_per_virtual_pipeline_stage")
    system.update(
        {
            "use_dualpipev": True,
            "moe_fb_overlap": True,
            "delay_wgrad_compute": True,
        }
    )
    model = baseline["train"]["model"]
    model.update(
        {
            "moe_token_dispatcher_type": "alltoall",
            "moe_grouped_gemm": True,
            "moe_shared_expert_intermediate_size": 256,
            "moe_shared_expert_overlap": False,
        }
    )

    assert dualpipev == baseline
    assert dualpipev["experiment"]["runner"]["nproc_per_node"] == 4
    assert system["pipeline_model_parallel_size"] == 2
    assert system["expert_model_parallel_size"] == 2
    assert model["num_layers"] == 4
    assert model["micro_batch_size"] == 1
    assert model["global_batch_size"] == 8

    profile = gate.PROFILES["pp2-dp2-ep2-dualpipev"]
    assert profile.rank_count == 4
    assert {requirement.name for requirement in profile.events} == {
        "forward-step",
        "backward-step",
        "combined-forward-backward-step",
        "ep-alltoall-async-launch",
        "ep-alltoall-async-complete",
        "moe-router",
        "p2p-launch",
        "send-forward",
        "recv-forward",
        "send-backward",
        "recv-backward",
        "grad-sync",
        "all-grads-sync",
        "dp-allreduce",
    }
    assert {
        "moe-dispatch",
        "moe-experts",
        "moe-combine",
        "ep-alltoall-dispatch",
        "ep-alltoall-combine",
    }.isdisjoint(requirement.name for requirement in profile.events)
    assert (
        profile.contract is dualpipev_probe_contract.validate_dualpipev_route
    )
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_checkpoint
    )


def test_training_contract_requires_the_terminal_checkpoint(
    tmp_path: Path,
) -> None:
    failures = training_run_contract.validate_two_iteration_checkpoint(
        tmp_path, False
    )

    assert {failure.code for failure in failures} == {
        "run.training.tracker",
        "run.training.checkpoint",
    }

    _write_terminal_checkpoint(tmp_path)

    assert (
        training_run_contract.validate_two_iteration_checkpoint(tmp_path, False)
        == ()
    )


def test_transformer_engine_training_contract_requires_the_parsed_model_route(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ local\n",
        encoding="utf-8",
    )

    failures = (
        training_run_contract.validate_two_iteration_transformer_engine_checkpoint(
            tmp_path, False
        )
    )
    assert {failure.code for failure in failures} == {
        "run.training.transformer_impl"
    }

    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ "
        "transformer_engine\n",
        encoding="utf-8",
    )
    assert (
        training_run_contract.validate_two_iteration_transformer_engine_checkpoint(
            tmp_path, False
        )
        == ()
    )


def test_te_full_cuda_graph_training_contract_requires_capture_and_replay(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    valid_log = "\n".join(
        (
            "[default0]:  transformer_impl ................ transformer_engine",
            "[default0]:  cuda_graph_impl ................. transformer_engine",
            "[default0]:  cuda_graph_scope ................ []",
            "[default0]:  cuda_graph_warmup_steps ......... 1",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Rank 0: 2 graphable layers.",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Start CUDA Graphs capture...",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Time spent in CUDA Graphs capture on rank 0: 1.25s",
            "[default0]: iteration 1/ 2 | lm loss: 1.0 |",
            "[default0]: iteration 2/ 2 | lm loss: 0.9 |",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Rank 0: 2 graphs deleted with explicit reset, "
            "0 graphs deleted without explicit reset.",
        )
    )
    launcher_log.write_text(valid_log, encoding="utf-8")

    assert (
        training_run_contract.validate_two_iteration_te_full_cuda_graph_checkpoint(
            tmp_path, True
        )
        == ()
    )

    launcher_log.write_text(
        valid_log.replace(
            "cuda_graph_scope ................ []",
            "cuda_graph_scope ................ [<CudaGraphScope.attn: 2>]",
        ),
        encoding="utf-8",
    )
    failures = (
        training_run_contract.validate_two_iteration_te_full_cuda_graph_checkpoint(
            tmp_path, True
        )
    )
    assert [failure.code for failure in failures] == [
        "run.training.cuda_graph_scope"
    ]


def test_te_attention_mlp_recompute_training_contract(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    valid_log = "\n".join(
        (
            "[default0]:  transformer_impl ................ transformer_engine",
            "[default0]:  cuda_graph_impl ................. transformer_engine",
            "[default0]:  cuda_graph_scope ................ "
            "[<CudaGraphScope.attn: 2>]",
            "[default0]:  cuda_graph_warmup_steps ......... 0",
            "[default0]:  recompute_granularity ........... selective",
            "[default0]:  recompute_modules ............... ['mlp']",
            "[default0]:  trace_cupti_kernels ............. off",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Rank 0: 2 graphable layers.",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Start CUDA Graphs capture...",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Time spent in CUDA Graphs capture on rank 0: 1.25s",
            "[default0]: iteration 1/ 2 | lm loss: 1.0 |",
            "[default0]: iteration 2/ 2 | lm loss: 0.9 |",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Rank 0: 0 graphs deleted with explicit reset, "
            "2 graphs deleted without explicit reset.",
        )
    )
    launcher_log.write_text(valid_log, encoding="utf-8")

    assert (
        training_run_contract.validate_two_iteration_te_attention_mlp_recompute_checkpoint(
            tmp_path, True
        )
        == ()
    )

    launcher_log.write_text(
        valid_log.replace(
            "recompute_modules ............... ['mlp']",
            "recompute_modules ............... ['core_attn']",
        ),
        encoding="utf-8",
    )
    failures = (
        training_run_contract.validate_two_iteration_te_attention_mlp_recompute_checkpoint(
            tmp_path, True
        )
    )
    assert [failure.code for failure in failures] == [
        "run.training.recompute_modules"
    ]

    launcher_log.write_text(
        valid_log.replace("Start CUDA Graphs capture...", "CUDA Graph capture omitted"),
        encoding="utf-8",
    )
    failures = (
        training_run_contract.validate_two_iteration_te_attention_mlp_recompute_checkpoint(
            tmp_path, True
        )
    )
    assert [failure.code for failure in failures] == [
        "run.training.cuda_graph_capture_start"
    ]


def test_local_layerwise_full_training_contract_requires_four_graphs(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    valid_log = "\n".join(
        (
            "[default0]:  transformer_impl ................ transformer_engine",
            "[default0]:  cuda_graph_impl ................. local",
            "[default0]:  cuda_graph_scope ................ []",
            "[default0]:  cuda_graph_warmup_steps ......... 1",
            "[default0]:  optimizer_cuda_graph ............ False",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Creating 4 CUDA graphs",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "> built 4 cuda graph(s) in 0.25 sec, with total memory usage: "
            "allocated 1.0 mb, reserved 2.0 mb.",
            "[default0]: iteration 1/ 2 | lm loss: 1.0 |",
            "[default0]: iteration 2/ 2 | lm loss: 0.9 |",
        )
    )
    launcher_log.write_text(valid_log, encoding="utf-8")

    assert (
        training_run_contract.validate_two_iteration_local_layerwise_full_cuda_graph_checkpoint(
            tmp_path, True
        )
        == ()
    )

    launcher_log.write_text(
        valid_log.replace("> built 4 cuda graph(s)", "> built 2 cuda graph(s)"),
        encoding="utf-8",
    )
    failures = (
        training_run_contract.validate_two_iteration_local_layerwise_full_cuda_graph_checkpoint(
            tmp_path, True
        )
    )
    assert [failure.code for failure in failures] == [
        "run.training.cuda_graph_capture_done"
    ]


def test_local_cuda_graph_kernel_training_contract_requires_each_iteration(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    valid_log = "\n".join(
        (
            "[default0]:  transformer_impl ................ transformer_engine",
            "[default0]:  cuda_graph_impl ................. local",
            "[default0]:  cuda_graph_scope ................ []",
            "[default0]:  cuda_graph_warmup_steps ......... 1",
            "[default0]:  optimizer_cuda_graph ............ False",
            "[default0]:  trace_cupti_kernels ............. on",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "Creating 4 CUDA graphs",
            "[default0]:INFO:megatron.core.transformer.cuda_graphs:"
            "> built 4 cuda graph(s) in 0.25 sec, with total memory usage: "
            "allocated 1.0 mb, reserved 2.0 mb.",
            "[default0]: iteration 1/ 2 | lm loss: 1.0 |",
            "[default0]:[trace] extracted 120 cuda kernel events at iter 1",
            "[default0]: iteration 2/ 2 | lm loss: 0.9 |",
            "[default0]:[trace] extracted 80 cuda kernel events at iter 2",
        )
    )
    launcher_log.write_text(valid_log, encoding="utf-8")

    assert (
        training_run_contract.validate_two_iteration_local_layerwise_cuda_graph_kernel_checkpoint(
            tmp_path, True
        )
        == ()
    )

    launcher_log.write_text(
        valid_log.replace(
            "[default0]:[trace] extracted 80 cuda kernel events at iter 2",
            "",
        ),
        encoding="utf-8",
    )
    failures = (
        training_run_contract.validate_two_iteration_local_layerwise_cuda_graph_kernel_checkpoint(
            tmp_path, True
        )
    )
    assert [failure.code for failure in failures] == [
        "run.training.cuda_kernel_capture"
    ]

    launcher_log.write_text(
        "\n".join(
            line for line in valid_log.splitlines() if "[trace] extracted" not in line
        ),
        encoding="utf-8",
    )
    assert (
        training_run_contract.validate_two_iteration_local_layerwise_cuda_graph_kernel_checkpoint(
            tmp_path, False
        )
        == ()
    )


def test_local_full_iteration_training_contract_requires_capture_and_replay(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path, iteration=3)
    launcher_log = tmp_path / "launcher.log"
    valid_log = "\n".join(
        (
            "[default0]:  transformer_impl ................ transformer_engine",
            "[default0]:  cuda_graph_impl ................. local",
            "[default0]:  cuda_graph_scope ................ "
            "[<CudaGraphScope.full_iteration: 9>]",
            "[default0]:  cuda_graph_warmup_steps ......... 1",
            "[default0]:  check_for_nan_in_loss_and_grad .. False",
            "[default0]:  optimizer_cuda_graph ............ False",
            "[default0]: iteration 1/ 3 | lm loss: 1.0 |",
            "[default0]: Capture CUDA graph for training!!!",
            "[default0]: CUDA graph capture done for training!!!",
            "[default0]: iteration 2/ 3 | lm loss: 0.9 |",
            "[default0]: iteration 3/ 3 | lm loss: 0.8 |",
        )
    )
    launcher_log.write_text(valid_log, encoding="utf-8")

    assert (
        training_run_contract.validate_three_iteration_local_full_cuda_graph_checkpoint(
            tmp_path, True
        )
        == ()
    )

    launcher_log.write_text(
        valid_log.replace(
            "[<CudaGraphScope.full_iteration: 9>]",
            "[<CudaGraphScope.attn: 2>]",
        ),
        encoding="utf-8",
    )
    failures = (
        training_run_contract.validate_three_iteration_local_full_cuda_graph_checkpoint(
            tmp_path, True
        )
    )
    assert [failure.code for failure in failures] == [
        "run.training.cuda_graph_scope"
    ]


def test_local_full_iteration_training_contract_rejects_missing_training_capture(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path, iteration=3)
    (tmp_path / "launcher.log").write_text(
        "\n".join(
            (
                "[default0]:  transformer_impl ................ transformer_engine",
                "[default0]:  cuda_graph_impl ................. local",
                "[default0]:  cuda_graph_scope ................ "
                "[<CudaGraphScope.full_iteration: 1>]",
                "[default0]:  cuda_graph_warmup_steps ......... 1",
                "[default0]:  check_for_nan_in_loss_and_grad .. False",
                "[default0]:  optimizer_cuda_graph ............ False",
                "[default0]: iteration 1/ 3 | lm loss: 1.0 |",
                "[default0]: CUDA graph capture done for training!!!",
                "[default0]: iteration 2/ 3 | lm loss: 0.9 |",
                "[default0]: iteration 3/ 3 | lm loss: 0.8 |",
            )
        ),
        encoding="utf-8",
    )

    failures = (
        training_run_contract.validate_three_iteration_local_full_cuda_graph_checkpoint(
            tmp_path, True
        )
    )
    assert [failure.code for failure in failures] == [
        "run.training.cuda_graph_capture_start"
    ]


def test_local_full_iteration_training_contract_rejects_optimizer_capture(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path, iteration=3)
    (tmp_path / "launcher.log").write_text(
        "\n".join(
            (
                "[default0]:  transformer_impl ................ transformer_engine",
                "[default0]:  cuda_graph_impl ................. local",
                "[default0]:  cuda_graph_scope ................ "
                "[<CudaGraphScope.full_iteration: 1>]",
                "[default0]:  cuda_graph_warmup_steps ......... 1",
                "[default0]:  check_for_nan_in_loss_and_grad .. False",
                "[default0]:  optimizer_cuda_graph ............ False",
                "[default0]: iteration 1/ 3 | lm loss: 1.0 |",
                "[default0]: Capture CUDA graph for training!!!",
                "[default0]: CUDA graph capture done for training!!!",
                "[default0]: Capture CUDA graph for optimizer!!!",
                "[default0]: Optimizer CUDA graph capture done!!!",
                "[default0]: iteration 2/ 3 | lm loss: 0.9 |",
                "[default0]: iteration 3/ 3 | lm loss: 0.8 |",
            )
        ),
        encoding="utf-8",
    )

    failures = (
        training_run_contract.validate_three_iteration_local_full_cuda_graph_checkpoint(
            tmp_path, True
        )
    )
    assert [failure.code for failure in failures] == [
        "run.training.optimizer_cuda_graph_capture"
    ]


def test_flagcx_training_contract_requires_the_parsed_backend(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    _write_flagcx_worker_logs(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "[default0]:  distributed_backend ............................ nccl\n",
        encoding="utf-8",
    )

    failures = training_run_contract.validate_two_iteration_flagcx_checkpoint(
        tmp_path, False
    )
    assert {failure.code for failure in failures} == {
        "run.training.distributed_backend"
    }

    launcher_log.write_text(
        "[default0]:  distributed_backend ............................ flagcx\n",
        encoding="utf-8",
    )
    assert (
        training_run_contract.validate_two_iteration_flagcx_checkpoint(
            tmp_path, False
        )
        == ()
    )


def test_flagcx_training_contract_requires_both_dp2_worker_ranks(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    _write_flagcx_worker_logs(tmp_path, ranks=(0, 0))
    (tmp_path / "launcher.log").write_text(
        "[default0]:  distributed_backend ............................ flagcx\n",
        encoding="utf-8",
    )

    failures = training_run_contract.validate_two_iteration_flagcx_checkpoint(
        tmp_path, False
    )
    assert {failure.code for failure in failures} == {
        "run.training.flagcx_dp2_group"
    }


def test_flagcx_training_contract_rejects_worker_failure_marker(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    _write_flagcx_worker_logs(tmp_path, fatal_marker="ChildFailedError")
    (tmp_path / "launcher.log").write_text(
        "[default0]:  distributed_backend ............................ flagcx\n",
        encoding="utf-8",
    )

    failures = training_run_contract.validate_two_iteration_flagcx_checkpoint(
        tmp_path, False
    )
    assert {failure.code for failure in failures} == {
        "run.training.flagcx_worker_failure"
    }


def test_qwen3_training_contract_requires_the_reviewed_model_and_topology(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    arguments = (
        ("transformer_impl", "transformer_engine"),
        *training_run_contract._QWEN3_TP2_SP_ARGUMENTS,
    )
    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "\n".join(
            f"[default0]:  {name} ................................ {value}"
            for name, value in arguments
        ),
        encoding="utf-8",
    )

    assert (
        training_run_contract.validate_two_iteration_qwen3_tp2_sp_checkpoint(
            tmp_path, True
        )
        == ()
    )

    launcher_log.write_text(
        "\n".join(
            f"[default0]:  {name} ................................ "
            f"{27 if name == 'num_layers' else value}"
            for name, value in arguments
        ),
        encoding="utf-8",
    )
    failures = (
        training_run_contract.validate_two_iteration_qwen3_tp2_sp_checkpoint(
            tmp_path, True
        )
    )
    assert "run.training.qwen3_argument" in {
        failure.code for failure in failures
    }


def test_qwen3_tp4_training_contract_requires_tensor_parallel_size_four(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    arguments = (
        ("transformer_impl", "transformer_engine"),
        *training_run_contract._QWEN3_TP4_SP_ARGUMENTS,
    )
    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "\n".join(
            f"[default0]:  {name} ................................ {value}"
            for name, value in arguments
        ),
        encoding="utf-8",
    )

    assert (
        training_run_contract.validate_two_iteration_qwen3_tp4_sp_checkpoint(
            tmp_path, True
        )
        == ()
    )

    launcher_log.write_text(
        launcher_log.read_text(encoding="utf-8").replace(
            "tensor_model_parallel_size ................................ 4",
            "tensor_model_parallel_size ................................ 2",
        ),
        encoding="utf-8",
    )
    failures = (
        training_run_contract.validate_two_iteration_qwen3_tp4_sp_checkpoint(
            tmp_path, True
        )
    )
    assert "run.training.qwen3_argument" in {
        failure.code for failure in failures
    }


def test_qwen3_tp4_local_no_sp_training_contract_requires_the_control_route(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "\n".join(
            f"[default0]:  {name} ................................ {value}"
            for name, value in (
                training_run_contract._QWEN3_TP4_LOCAL_NO_SP_ARGUMENTS
            )
        ),
        encoding="utf-8",
    )

    assert (
        training_run_contract.validate_two_iteration_qwen3_tp4_local_no_sp_checkpoint(
            tmp_path, True
        )
        == ()
    )

    launcher_log.write_text(
        launcher_log.read_text(encoding="utf-8").replace(
            "sequence_parallel ................................ False",
            "sequence_parallel ................................ True",
        ),
        encoding="utf-8",
    )
    failures = (
        training_run_contract.validate_two_iteration_qwen3_tp4_local_no_sp_checkpoint(
            tmp_path, True
        )
    )
    assert "run.training.qwen3_local_argument" in {
        failure.code for failure in failures
    }


def test_qwen3_tp8_training_contract_requires_tensor_parallel_size_eight(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    arguments = (
        ("transformer_impl", "transformer_engine"),
        *training_run_contract._QWEN3_TP8_SP_ARGUMENTS,
    )
    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "\n".join(
            f"[default0]:  {name} ................................ {value}"
            for name, value in arguments
        ),
        encoding="utf-8",
    )

    assert (
        training_run_contract.validate_two_iteration_qwen3_tp8_sp_checkpoint(
            tmp_path, True
        )
        == ()
    )

    launcher_log.write_text(
        launcher_log.read_text(encoding="utf-8").replace(
            "tensor_model_parallel_size ................................ 8",
            "tensor_model_parallel_size ................................ 4",
        ),
        encoding="utf-8",
    )
    failures = (
        training_run_contract.validate_two_iteration_qwen3_tp8_sp_checkpoint(
            tmp_path, True
        )
    )
    assert "run.training.qwen3_argument" in {
        failure.code for failure in failures
    }


def test_userbuffer_training_contract_requires_the_parsed_overlap_route(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ "
        "transformer_engine\n"
        "[default0]:  tp_comm_overlap ................................ False\n",
        encoding="utf-8",
    )

    failures = training_run_contract.validate_two_iteration_transformer_engine_userbuffer_checkpoint(
        tmp_path, False
    )
    assert {failure.code for failure in failures} == {
        "run.training.tp_comm_overlap"
    }

    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ "
        "transformer_engine\n"
        "[default0]:  tp_comm_overlap ................................ True\n",
        encoding="utf-8",
    )
    assert (
        training_run_contract.validate_two_iteration_transformer_engine_userbuffer_checkpoint(
            tmp_path, False
        )
        == ()
    )


def test_op_fuser_training_contract_requires_the_controlled_spec(
    tmp_path: Path,
) -> None:
    _write_terminal_checkpoint(tmp_path)
    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ "
        "transformer_engine\n"
        "[default0]:  spec ........................................... None\n",
        encoding="utf-8",
    )

    failures = training_run_contract.validate_two_iteration_transformer_engine_op_fuser_checkpoint(
        tmp_path, False
    )
    assert {failure.code for failure in failures} == {
        "run.training.te_op_fuser_spec"
    }

    launcher_log.write_text(
        "[default0]:  transformer_impl ................................ "
        "transformer_engine\n"
        "[default0]:  spec ........................................... "
        "['megalens_te_op_fuser_spec', 'te_op_fuser_spec']\n",
        encoding="utf-8",
    )
    assert (
        training_run_contract.validate_two_iteration_transformer_engine_op_fuser_checkpoint(
            tmp_path, False
        )
        == ()
    )


def test_legacy_pp2_training_contract_requires_both_pipeline_stages(
    tmp_path: Path,
) -> None:
    failures = training_run_contract.validate_two_iteration_legacy_pp2_checkpoint(
        tmp_path, False
    )

    assert {failure.code for failure in failures} == {
        "run.training.tracker",
        "run.training.checkpoint",
    }

    _write_legacy_pp2_terminal_checkpoint(tmp_path)

    assert (
        training_run_contract.validate_two_iteration_legacy_pp2_checkpoint(
            tmp_path, False
        )
        == ()
    )


def test_force_sync_run_contract_requires_all_weight_hash_callbacks(
    tmp_path: Path,
) -> None:
    _write_legacy_pp2_terminal_checkpoint(tmp_path)
    (tmp_path / "launcher.log").write_text(
        "\n".join(
            f">>> Weight hashes match after {iteration} iterations..."
            for iteration in (0, 1, 2)
        ),
        encoding="utf-8",
    )

    assert (
        training_run_contract.validate_two_iteration_legacy_pp2_force_sync(
            tmp_path, True
        )
        == ()
    )


def test_standard_training_profiles_require_the_terminal_checkpoint() -> None:
    profiles_with_specialized_run_contracts = {
        "multimodule-bridge2",
        "multimodule-bridge8-fanin",
        "multimodule-bridge8-fanout",
        "mimo-train2",
        "mimo-pretrain2",
        "mimo-pretrain-save2",
        "mimo-pretrain-resume2",
        "mimo-train8-fanin",
        "mimo-train8-fanout",
        "pp2-dp2-distopt-force-sync",
        "cp2-te",
        "cp4-te",
        "tp2-sp-te-linear",
        "tp2-sp-te-userbuffer",
        "tp2-sp-te-op-fuser",
        "te-attn-mlp-recompute-cuda-graph",
        "te-full-cuda-graph",
        "local-layerwise-full-cuda-graph",
        "local-layerwise-cuda-kernels",
        "local-full-iteration-cuda-graph",
            "qwen3-enron-tp2-sp",
            "qwen3-enron-tp4-sp",
            "qwen3-enron-tp4-local-no-sp",
            "qwen3-enron-tp8-sp",
        "qwen3-enron-cp2",
        "qwen3-enron-cp4",
        "deepseek-tp2-sp-mock",
    }

    for name, profile in gate.PROFILES.items():
        if name in profiles_with_specialized_run_contracts:
            assert profile.run_contract is not None
        else:
            assert (
                profile.run_contract
                is training_run_contract.validate_two_iteration_checkpoint
            )

    assert (
        gate.PROFILES["pp2-dp2-distopt-force-sync"].run_contract
        is training_run_contract.validate_two_iteration_legacy_pp2_force_sync
    )


def test_raw_framework_event_requirements_use_the_enclosing_iteration() -> None:
    required_fields = {
        field
        for profile in gate.PROFILES.values()
        for requirement in profile.events
        for field in requirement.fields
    }

    assert "iteration" not in required_fields
    assert {"g_rk", "dp_rk", "pp_rk", "tp_rk"} <= required_fields


def test_unbatched_pp2_profile_uses_vpp_without_unsupported_p2p_cli_keys() -> None:
    config = yaml.safe_load(
        (_FIXTURES / "flagscale_single_node_pp2_unbatched_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    system = config["train"]["system"]
    model = config["train"]["model"]

    assert system["pipeline_model_parallel_size"] == 2
    assert system["num_layers_per_virtual_pipeline_stage"] == 1
    assert system["microbatch_group_size_per_virtual_pipeline_stage"] == 2
    assert {
        "batch_p2p_comm",
        "batch_p2p_sync",
        "no_overlap_p2p_communication",
    }.isdisjoint(system)
    assert model["num_layers"] == 4
    assert model["micro_batch_size"] == 1
    assert model["global_batch_size"] == 2


def test_pp2_unbatched_warmup_flush_profile_only_enables_selected_route() -> None:
    baseline = yaml.safe_load(
        (_FIXTURES / "flagscale_single_node_pp2_unbatched_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    warmup_flush = yaml.safe_load(
        (
            _FIXTURES
            / "flagscale_single_node_pp2_unbatched_warmup_flush_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = warmup_flush["experiment"]["exp_name"]
    baseline["train"]["system"][
        "overlap_p2p_communication_warmup_flush"
    ] = True

    assert warmup_flush == baseline
    profile = gate.PROFILES["pp2-unbatched-warmup-flush"]
    assert profile.rank_count == 2
    assert {requirement.name for requirement in profile.events} == {
        "p2p-launch",
        "send-forward",
        "recv-forward",
        "send-backward",
        "recv-backward",
        "forward-step",
        "backward-step",
    }
    assert (
        profile.contract
        is p2p_probe_contract.validate_pp2_unbatched_warmup_flush_route
    )
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_checkpoint
    )


def test_pp2_overlap_timeline_profile_only_enables_cupti_kernel_capture() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES
            / "flagscale_single_node_pp2_unbatched_warmup_flush_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    timeline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_pp2_overlap_timeline_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = timeline["experiment"]["exp_name"]
    baseline["train"]["system"]["trace_cupti_kernels"] = "on"

    assert timeline == baseline
    profile = gate.PROFILES["pp2-overlap-timeline"]
    assert profile.rank_count == 2
    assert profile.events == gate.PROFILES["pp2-unbatched-warmup-flush"].events
    assert profile.contract is p2p_probe_contract.validate_pp2_overlap_timeline_route
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_checkpoint
    )


def test_pp2_batched_steady_profile_only_adds_a_second_microbatch() -> None:
    baseline = yaml.safe_load(
        (_FIXTURES / "flagscale_single_node_pp2_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    steady = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_pp2_batched_steady_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = steady["experiment"]["exp_name"]
    baseline["train"]["model"]["global_batch_size"] = 2

    assert steady == baseline
    profile = gate.PROFILES["pp2-batched-steady"]
    assert profile.rank_count == 2
    assert (
        profile.contract
        is p2p_probe_contract.validate_pp2_batched_steady_route
    )


@pytest.mark.parametrize(
    ("baseline_name", "overlap_name", "profile_name", "changes", "contract"),
    (
        (
            "flagscale_single_node_dp2_standard_smoke.yaml",
            "flagscale_single_node_dp2_standard_overlap_smoke.yaml",
            "dp2-standard-ddp-overlap",
            {"overlap_grad_reduce": True},
            dp_probe_contract.validate_dp_standard_overlap,
        ),
        (
            "flagscale_single_node_dp2_distopt_smoke.yaml",
            "flagscale_single_node_dp2_distopt_overlap_smoke.yaml",
            "dp2-distopt-overlap",
            {"overlap_grad_reduce": True, "overlap_param_gather": True},
            dp_probe_contract.validate_dp_distopt_overlap,
        ),
        (
            "flagscale_single_node_dp8_standard_smoke.yaml",
            "flagscale_single_node_dp8_standard_overlap_smoke.yaml",
            "dp8-standard-ddp-overlap",
            {"overlap_grad_reduce": True},
            dp_probe_contract.validate_dp_standard_overlap,
        ),
        (
            "flagscale_single_node_dp8_distopt_smoke.yaml",
            "flagscale_single_node_dp8_distopt_overlap_smoke.yaml",
            "dp8-distopt-overlap",
            {"overlap_grad_reduce": True, "overlap_param_gather": True},
            dp_probe_contract.validate_dp_distopt_overlap,
        ),
    ),
)
def test_dp_overlap_profiles_only_enable_the_selected_overlap_route(
    baseline_name: str,
    overlap_name: str,
    profile_name: str,
    changes: dict[str, bool],
    contract,
) -> None:
    baseline = yaml.safe_load(
        (_FIXTURES / baseline_name).read_text(encoding="utf-8")
    )
    overlap = yaml.safe_load(
        (_FIXTURES / overlap_name).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = overlap["experiment"]["exp_name"]
    baseline["train"]["system"].update(changes)

    assert overlap == baseline
    profile = gate.PROFILES[profile_name]
    assert profile.rank_count == overlap["experiment"]["runner"]["nproc_per_node"]
    assert profile.contract is contract


def test_dp4_multi_instance_profile_only_expands_the_dp_overlap_topology() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_dp2_distopt_overlap_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    multi_instance = yaml.safe_load(
        (
            _FIXTURES
            / "flagscale_single_node_dp4_distopt_multi_instance_overlap_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = multi_instance["experiment"]["exp_name"]
    baseline["experiment"]["runner"]["nproc_per_node"] = 4
    baseline["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    baseline["train"]["system"]["num_distributed_optimizer_instances"] = 2
    baseline["train"]["model"]["global_batch_size"] = 4

    assert multi_instance == baseline
    assert (
        gate.PROFILES["dp4-distopt-multi-instance-overlap"].contract
        is dp_probe_contract.validate_dp_multi_instance_distopt_overlap
    )

    args = gate._parser().parse_args(
        (
            "--run-dir",
            "unused",
            "--input-config",
            "unknown.yaml",
            "--mode",
            "trace-on",
            "--image",
            "example/flagscale:dev",
            "--topology",
            "dp4",
        )
    )
    assert gate._profile_from_arguments(args).name == (
        "dp4-distopt-multi-instance-overlap"
    )


def test_dp2_layerwise_profile_only_selects_dist_muon_parameter_overlap() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_dp2_standard_overlap_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    layerwise = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_dp2_layerwise_overlap_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = layerwise["experiment"]["exp_name"]
    baseline["train"]["system"]["overlap_param_gather"] = True
    baseline["train"]["model"]["optimizer"]["optimizer"] = "dist_muon"

    assert layerwise == baseline
    system = layerwise["train"]["system"]
    optimizer = layerwise["train"]["model"]["optimizer"]
    assert system["use_distributed_optimizer"] is False
    assert system["overlap_grad_reduce"] is True
    assert system["overlap_param_gather"] is True
    assert optimizer["optimizer"] == "dist_muon"
    assert "muon_tp_mode" not in optimizer
    assert "use_padded_layerwise_optimizer" not in system

    profile = gate.PROFILES["dp2-layerwise-overlap"]
    assert profile.rank_count == 2
    assert {requirement.name for requirement in profile.events} == {
        "dp-allreduce",
        "dp-param-all-gather",
        "dp-grad-sync-complete",
        "dp-param-sync-complete",
    }
    assert profile.contract is dp_probe_contract.validate_dp_layerwise_overlap


def test_force_sync_profile_selects_interleaved_pp2_dp2_legacy_checkpoint() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_pp2_unbatched_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    force_sync = yaml.safe_load(
        (
            _FIXTURES
            / "flagscale_single_node_pp2_dp2_distopt_force_sync_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = force_sync["experiment"]["exp_name"]
    baseline["experiment"]["ckpt_format"] = "torch"
    baseline["experiment"]["runner"]["nproc_per_node"] = 4
    baseline["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    system = baseline["train"]["system"]
    system.update(
        {
            "use_distributed_optimizer": True,
            "overlap_grad_reduce": True,
            "overlap_param_gather": True,
            "overlap_param_gather_with_optimizer_step": True,
            "num_distributed_optimizer_instances": 1,
            "check_weight_hash_across_dp_replicas_interval": 1,
        }
    )
    system["checkpoint"]["ckpt_format"] = "torch"
    baseline["train"]["model"]["global_batch_size"] = 4

    assert force_sync == baseline
    profile = gate.PROFILES["pp2-dp2-distopt-force-sync"]
    assert profile.rank_count == 4
    assert (
        profile.contract
        is dp_probe_contract.validate_dp_optimizer_step_force_sync_training
    )
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_legacy_pp2_force_sync
    )


def test_tp2_sp_profile_only_selects_the_local_tp_routes() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_gpt_eager_full_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    tp2_sp = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_sp_local_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = tp2_sp["experiment"]["exp_name"]
    baseline["experiment"]["runner"]["nproc_per_node"] = 2
    baseline["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1"
    baseline["train"]["system"]["tensor_model_parallel_size"] = 2
    baseline["train"]["system"]["sequence_parallel"] = True
    model = baseline["train"]["model"]
    model.pop("no_persist_layer_norm")
    model["no_gradient_accumulation_fusion"] = True
    model["group_query_attention"] = True
    model["num_query_groups"] = 1
    model["normalization"] = "LayerNorm"

    assert tp2_sp == baseline
    model = tp2_sp["train"]["model"]
    system = tp2_sp["train"]["system"]
    assert model["transformer_impl"] == "local"
    assert model["global_batch_size"] == 1
    assert model["untie_embeddings_and_output_weights"] is True
    assert system["pipeline_model_parallel_size"] == 1
    assert system["context_parallel_size"] == 1

    profile = gate.PROFILES["tp2-sp-local"]
    assert profile.rank_count == 2
    assert {requirement.name for requirement in profile.events} == {
        "tp-all-gather-first",
        "tp-all-gather-last",
        "tp-reduce-scatter",
        "tp-reduce-scatter-last",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
        "grad-sync",
        "all-grads-sync",
        "sp-layernorm-allreduce",
    }
    assert profile.contract is tp_probe_contract.validate_tp2_sp_profile

    args = gate._parser().parse_args(
        (
            "--run-dir",
            "unused",
            "--input-config",
            "unknown.yaml",
            "--mode",
            "trace-on",
            "--image",
            "example/flagscale:dev",
            "--topology",
            "tp2",
        )
    )
    assert gate._profile_from_arguments(args).name == "tp2-sp-local"


def test_tp2_sp_te_linear_profile_only_switches_the_transformer_implementation() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_sp_local_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    te_linear = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_sp_te_linear_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = te_linear["experiment"]["exp_name"]
    baseline["train"]["model"]["transformer_impl"] = "transformer_engine"

    assert te_linear == baseline

    profile = gate.PROFILES["tp2-sp-te-linear"]
    assert profile.rank_count == 2
    assert {requirement.name for requirement in profile.events} == {
        "transformer_layer",
        "attention",
        "MLP.forward",
        "tp-all-gather-first",
        "tp-all-gather-last",
        "tp-reduce-scatter",
        "tp-reduce-scatter-last",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
        "grad-sync",
        "all-grads-sync",
        "sp-layernorm-allreduce",
    }
    assert profile.contract is tp_probe_contract.validate_tp2_sp_te_linear_profile
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_transformer_engine_checkpoint
    )
    assert (
        gate._CONFIG_PROFILES["flagscale_single_node_tp2_sp_te_linear_smoke"]
        == "tp2-sp-te-linear"
    )


def test_tp2_sp_te_userbuffer_profile_only_enables_tp_overlap() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_sp_te_linear_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    userbuffer = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_sp_te_userbuffer_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = userbuffer["experiment"]["exp_name"]
    baseline["train"]["system"]["tp_comm_overlap"] = True

    assert userbuffer == baseline

    profile = gate.PROFILES["tp2-sp-te-userbuffer"]
    te_linear_profile = gate.PROFILES["tp2-sp-te-linear"]
    assert profile.rank_count == 2
    assert profile.events == te_linear_profile.events
    assert profile.contract is tp_probe_contract.validate_tp2_sp_te_linear_profile
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_transformer_engine_userbuffer_checkpoint
    )
    assert (
        gate._CONFIG_PROFILES[
            "flagscale_single_node_tp2_sp_te_userbuffer_smoke"
        ]
        == "tp2-sp-te-userbuffer"
    )


def test_tp2_sp_te_op_fuser_profile_only_selects_the_controlled_spec() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_sp_te_linear_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    op_fuser = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_sp_te_op_fuser_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = op_fuser["experiment"]["exp_name"]
    baseline["train"]["model"]["spec"] = [
        "megalens_te_op_fuser_spec",
        "te_op_fuser_spec",
    ]

    assert op_fuser == baseline

    profile = gate.PROFILES["tp2-sp-te-op-fuser"]
    assert profile.rank_count == 2
    assert {requirement.name for requirement in profile.events} == {
        "transformer_layer",
        "_forward_attention",
        "attention",
        "_forward_mlp",
        "tp-all-gather-first",
        "tp-all-gather-last",
        "tp-reduce-scatter",
        "tp-reduce-scatter-last",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
        "grad-sync",
        "all-grads-sync",
        "sp-layernorm-allreduce",
    }
    assert "MLP.forward" not in {
        requirement.name for requirement in profile.events
    }
    assert profile.contract is tp_probe_contract.validate_tp2_sp_te_op_fuser_profile
    assert (
        profile.run_contract
        is training_run_contract.validate_two_iteration_transformer_engine_op_fuser_checkpoint
    )
    assert (
        gate._CONFIG_PROFILES[
            "flagscale_single_node_tp2_sp_te_op_fuser_smoke"
        ]
        == "tp2-sp-te-op-fuser"
    )


def test_qwen3_enron_tp2_sp_profile_preserves_the_reviewed_l3_boundary() -> None:
    payload = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_qwen3_enron_tp2_sp.yaml"
        ).read_text(encoding="utf-8")
    )
    system = payload["train"]["system"]
    model = payload["train"]["model"]
    data = payload["train"]["data"]

    assert payload["action"] == "test"
    assert payload["experiment"]["runner"]["nproc_per_node"] == 2
    assert payload["experiment"]["ckpt_format"] == "torch_dist"
    assert system["tensor_model_parallel_size"] == 2
    assert system["pipeline_model_parallel_size"] == 1
    assert system["context_parallel_size"] == 1
    assert system["sequence_parallel"] is True
    assert system["distributed_backend"] == "nccl"
    assert system["use_distributed_optimizer"] is True
    assert system["overlap_grad_reduce"] is True
    assert system["overlap_param_gather"] is True
    assert model["transformer_impl"] == "transformer_engine"
    assert model["te_fl_prefer"] == "vendor"
    assert model["enable_flag_gems"] is False
    assert model["attention_backend"] == "flash"
    assert (
        model["num_layers"],
        model["hidden_size"],
        model["ffn_hidden_size"],
    ) == (28, 1024, 3072)
    assert (model["num_attention_heads"], model["num_query_groups"]) == (16, 8)
    assert (model["seq_length"], model["micro_batch_size"]) == (4096, 4)
    assert (model["global_batch_size"], model["train_iters"]) == (4, 2)
    assert model["init_method_std"] == 0.006
    assert "train_samples" not in model
    assert data["data_path"] == "${oc.env:MEGALENS_QWEN3_DATA_PATH}"
    assert data["tokenizer"]["tokenizer_path"] == (
        "${oc.env:MEGALENS_QWEN3_TOKENIZER_PATH}"
    )
    assert data["tokenizer"]["tokenizer_type"] == "QwenTokenizerFS"
    assert data["tokenizer"]["vocab_size"] == 151851
    assert "legacy_tokenizer" not in data["tokenizer"]
    assert "mock_data" not in data

    profile = gate.PROFILES["qwen3-enron-tp2-sp"]
    assert profile.rank_count == 2
    assert profile.run_contract is (
        training_run_contract.validate_two_iteration_qwen3_tp2_sp_checkpoint
    )
    assert gate._CONFIG_PROFILES["flagscale_single_node_qwen3_enron_tp2_sp"] == (
        "qwen3-enron-tp2-sp"
    )
    assert {requirement.name for requirement in profile.events} == {
        "forward-step",
        "decoder",
        "decoder-postprocess",
        "output_layer",
        "loss",
        "transformer_layer",
        "_forward_attention",
        "attention",
        "_forward_mlp",
        "MLP.forward",
        "tp-all-gather-first",
        "tp-reduce-scatter",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
        "grad-sync",
        "all-grads-sync",
        "sp-layernorm-allreduce",
    }


def test_qwen3_enron_tp4_sp_profile_only_scales_tensor_parallelism() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_qwen3_enron_tp2_sp.yaml"
        ).read_text(encoding="utf-8")
    )
    scaled = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_qwen3_enron_tp4_sp.yaml"
        ).read_text(encoding="utf-8")
    )

    assert scaled["experiment"]["runner"]["nproc_per_node"] == 4
    assert scaled["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert scaled["train"]["system"]["tensor_model_parallel_size"] == 4

    scaled["experiment"]["exp_name"] = baseline["experiment"]["exp_name"]
    scaled["experiment"]["runner"]["nproc_per_node"] = 2
    scaled["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1"
    scaled["train"]["system"]["tensor_model_parallel_size"] = 2
    assert scaled == baseline

    profile = gate.PROFILES["qwen3-enron-tp4-sp"]
    assert profile.rank_count == 4
    assert profile.events == gate.PROFILES["qwen3-enron-tp2-sp"].events
    assert profile.run_contract is (
        training_run_contract.validate_two_iteration_qwen3_tp4_sp_checkpoint
    )
    assert gate._CONFIG_PROFILES["flagscale_single_node_qwen3_enron_tp4_sp"] == (
        "qwen3-enron-tp4-sp"
    )


def test_qwen3_enron_tp4_local_no_sp_profile_only_changes_the_backend_route() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_qwen3_enron_tp4_sp.yaml"
        ).read_text(encoding="utf-8")
    )
    local = yaml.safe_load(
        (
            _FIXTURES
            / "flagscale_single_node_qwen3_enron_tp4_local_no_sp.yaml"
        ).read_text(encoding="utf-8")
    )

    assert local["experiment"]["exp_name"].endswith("tp4-local-no-sp")
    assert "NVTE_ALLOW_NONDETERMINISTIC_ALGO" not in local["experiment"]["envs"]
    assert local["train"]["system"]["sequence_parallel"] is False
    assert local["train"]["model"]["transformer_impl"] == "local"
    assert local["train"]["model"]["attention_backend"] == "auto"
    assert "spec" not in local["train"]["model"]
    assert local["train"]["model"]["no_persist_layer_norm"] is True
    assert local["train"]["model"]["no_gradient_accumulation_fusion"] is True
    assert "te_fl_prefer" not in local["train"]["model"]

    baseline["experiment"]["exp_name"] = local["experiment"]["exp_name"]
    baseline["experiment"]["envs"].pop("NVTE_ALLOW_NONDETERMINISTIC_ALGO")
    baseline["train"]["system"]["sequence_parallel"] = False
    baseline["train"]["model"].update(
        {
            "transformer_impl": "local",
            "attention_backend": "auto",
            "no_persist_layer_norm": True,
            "no_gradient_accumulation_fusion": True,
        }
    )
    baseline["train"]["model"].pop("te_fl_prefer")
    assert local == baseline

    profile = gate.PROFILES["qwen3-enron-tp4-local-no-sp"]
    assert profile.rank_count == 4
    assert profile.run_contract is (
        training_run_contract.validate_two_iteration_qwen3_tp4_local_no_sp_checkpoint
    )
    assert gate._CONFIG_PROFILES[
        "flagscale_single_node_qwen3_enron_tp4_local_no_sp"
    ] == "qwen3-enron-tp4-local-no-sp"
    assert {requirement.name for requirement in profile.events} == {
        "forward-step",
        "decoder",
        "decoder-postprocess",
        "output_layer",
        "loss",
        "transformer_layer",
        "_forward_attention",
        "attention",
        "_forward_mlp",
        "MLP.forward",
        "tp-allreduce",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
        "grad-sync",
        "all-grads-sync",
        "sp-layernorm-allreduce",
    }


def test_qwen3_enron_tp8_sp_profile_only_scales_tensor_parallelism() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_qwen3_enron_tp2_sp.yaml"
        ).read_text(encoding="utf-8")
    )
    scaled = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_qwen3_enron_tp8_sp.yaml"
        ).read_text(encoding="utf-8")
    )

    assert scaled["experiment"]["runner"]["nproc_per_node"] == 8
    assert (
        scaled["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"]
        == "0,1,2,3,4,5,6,7"
    )
    assert scaled["train"]["system"]["tensor_model_parallel_size"] == 8

    scaled["experiment"]["exp_name"] = baseline["experiment"]["exp_name"]
    scaled["experiment"]["runner"]["nproc_per_node"] = 2
    scaled["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1"
    scaled["train"]["system"]["tensor_model_parallel_size"] = 2
    assert scaled == baseline

    profile = gate.PROFILES["qwen3-enron-tp8-sp"]
    assert profile.rank_count == 8
    assert profile.events == gate.PROFILES["qwen3-enron-tp2-sp"].events
    assert profile.run_contract is (
        training_run_contract.validate_two_iteration_qwen3_tp8_sp_checkpoint
    )
    assert gate._CONFIG_PROFILES["flagscale_single_node_qwen3_enron_tp8_sp"] == (
        "qwen3-enron-tp8-sp"
    )


def test_tp2_local_allreduce_profile_only_disables_sequence_parallel() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_sp_local_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    allreduce = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_local_allreduce_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = allreduce["experiment"]["exp_name"]
    baseline["train"]["system"]["sequence_parallel"] = False

    assert allreduce == baseline

    profile = gate.PROFILES["tp2-local-allreduce"]
    assert profile.rank_count == 2
    assert {requirement.name for requirement in profile.events} == {
        "tp-allreduce",
        "tp-all-gather-last",
        "tp-reduce-scatter",
        "tp-reduce-scatter-last",
        "tp-linear-async-launch",
        "tp-linear-async-complete",
        "grad-sync",
        "all-grads-sync",
    }
    assert (
        profile.contract
        is tp_probe_contract.validate_tp2_local_allreduce_profile
    )


def test_tp2_pp2_embedding_profile_only_adds_pp_and_shared_weights() -> None:
    baseline = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_sp_local_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    embedding = yaml.safe_load(
        (
            _FIXTURES / "flagscale_single_node_tp2_pp2_embedding_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline["experiment"]["exp_name"] = embedding["experiment"]["exp_name"]
    baseline["experiment"]["runner"]["nproc_per_node"] = 4
    baseline["experiment"]["envs"]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    baseline["train"]["system"]["pipeline_model_parallel_size"] = 2
    baseline["train"]["model"]["untie_embeddings_and_output_weights"] = False

    assert embedding == baseline

    profile = gate.PROFILES["tp2-pp2-embedding"]
    assert profile.rank_count == 4
    assert {requirement.name for requirement in profile.events} == {
        "grad-sync",
        "all-grads-sync",
        "sp-layernorm-allreduce",
        "embedding-grads-allreduce",
    }
    assert (
        profile.contract
        is tp_probe_contract.validate_tp2_pp2_embedding_final_grad_sync
    )


def test_gpt_pp1_and_pp2_profiles_enforce_stage_specific_model_phases(
    tmp_path: Path,
) -> None:
    pp1_root = tmp_path / "pp1"
    _write_gpt_phase_trace(
        pp1_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
    )
    assert gate.PROFILES["gpt-eager-full"].contract(pp1_root) == ()

    pp2_root = tmp_path / "pp2"
    _write_gpt_phase_trace(
        pp2_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=False,
        include_optimizer=True,
    )
    _write_gpt_phase_trace(
        pp2_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
        include_optimizer=True,
    )
    assert gpt_probe_contract.validate_gpt_pp2_training_phases(pp2_root) == ()

    invalid_root = tmp_path / "invalid-pp2"
    _write_gpt_phase_trace(
        invalid_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        include_optimizer=True,
    )
    _write_gpt_phase_trace(
        invalid_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
        include_optimizer=True,
    )
    failures = gpt_probe_contract.validate_gpt_pp2_training_phases(invalid_root)
    assert failures
    assert {failure.code for failure in failures} == {"trace.gpt.count"}


def test_gpt_eager_profile_rejects_an_incomplete_layer_sequence(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "incomplete-eager"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=1,
    )

    failures = gate.PROFILES["gpt-eager-full"].contract(trace_root)

    assert [failure.code for failure in failures] == [
        "trace.gpt.eager_layers",
        "trace.gpt.eager_layers",
    ]
    assert [failure.evidence for failure in failures] == [
        "rank=0 iteration=1",
        "rank=0 iteration=2",
    ]


def test_te_full_cuda_graph_profile_separates_eager_and_replay_phases(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "te-full-cuda-graph"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1,)),
    )

    assert gate.PROFILES["te-full-cuda-graph"].contract(trace_root) == ()


def test_te_attention_mlp_recompute_profile_separates_calls(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "te-attn-mlp-recompute-cuda-graph"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        external_mlp_layers=2,
        recompute_mlp_layers=2,
        include_optimizer=True,
        include_backward=True,
    )

    assert (
        gate.PROFILES["te-attn-mlp-recompute-cuda-graph"].contract(trace_root)
        == ()
    )


def test_te_attention_mlp_recompute_profile_rejects_missing_recompute(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "te-attn-mlp-recompute-missing"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        external_mlp_layers=2,
        recompute_mlp_layers=1,
        include_optimizer=True,
        include_backward=True,
    )

    failures = gate.PROFILES["te-attn-mlp-recompute-cuda-graph"].contract(
        trace_root
    )

    assert [failure.code for failure in failures] == [
        "trace.gpt.cuda_graph_recompute_mlp",
        "trace.gpt.cuda_graph_recompute_mlp",
    ]


def test_te_attention_mlp_recompute_profile_rejects_captured_scopes(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "te-attn-mlp-recompute-inner"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=1,
        external_mlp_layers=1,
        recompute_mlp_layers=2,
        include_optimizer=True,
        include_backward=True,
    )

    failures = gate.PROFILES["te-attn-mlp-recompute-cuda-graph"].contract(
        trace_root
    )

    assert [failure.code for failure in failures] == [
        "trace.gpt.cuda_graph_attention_inner",
        "trace.gpt.cuda_graph_attention_inner",
    ]


def test_te_full_cuda_graph_profile_rejects_inner_replay_events(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "te-full-cuda-graph-inner-replay"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1, 2)),
    )

    failures = gate.PROFILES["te-full-cuda-graph"].contract(trace_root)

    assert [failure.code for failure in failures] == [
        "trace.gpt.cuda_graph_replay_inner"
    ]
    assert failures[0].evidence == "rank=0 iteration=2"


def test_local_layerwise_full_profile_separates_eager_and_replay_phases(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "local-layerwise-full-cuda-graph"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1,)),
    )

    assert (
        gate.PROFILES["local-layerwise-full-cuda-graph"].contract(trace_root)
        == ()
    )


def test_local_layerwise_full_profile_rejects_inner_replay_events(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "local-layerwise-full-cuda-graph-inner-replay"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1, 2)),
    )

    failures = gate.PROFILES["local-layerwise-full-cuda-graph"].contract(
        trace_root
    )

    assert [failure.code for failure in failures] == [
        "trace.gpt.cuda_graph_replay_inner"
    ]
    assert failures[0].evidence == "rank=0 iteration=2"


def test_local_cuda_graph_kernel_profile_captures_eager_and_replay_device_work(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "local-cuda-graph-kernels"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1,)),
        kernel_iterations=frozenset((1, 2)),
    )

    assert gate.PROFILES["local-layerwise-cuda-kernels"].contract(trace_root) == ()


def test_local_cuda_graph_kernel_profile_requires_replay_device_work(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "local-cuda-graph-missing-replay-kernels"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1,)),
        kernel_iterations=frozenset((1,)),
    )

    failures = gate.PROFILES["local-layerwise-cuda-kernels"].contract(trace_root)

    assert [failure.code for failure in failures] == [
        "trace.gpt.cuda_graph_kernel_capture"
    ]
    assert failures[0].evidence == "rank=0 iteration=2"


def test_local_cuda_graph_kernel_profile_requires_replay_model_compute(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "local-cuda-graph-no-replay-compute"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1,)),
        kernel_iterations=frozenset((1, 2)),
    )
    trace_path = next(trace_root.glob("*.json"))
    rows = json.loads(trace_path.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("record_type") == "cuda_kernel" and row.get("iteration") == 2:
            row["name"] = "ncclDevKernel_AllReduce"
    trace_path.write_text(json.dumps(rows), encoding="utf-8")

    failures = gate.PROFILES["local-layerwise-cuda-kernels"].contract(trace_root)

    assert [failure.code for failure in failures] == [
        "trace.gpt.cuda_graph_replay_compute_kernel"
    ]
    assert failures[0].evidence == "rank=0 iteration=2"


def test_local_cuda_graph_kernel_profile_rejects_invalid_rank_local_timeline(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "local-cuda-graph-invalid-kernel-timeline"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1,)),
        kernel_iterations=frozenset((1, 2)),
    )
    trace_path = next(trace_root.glob("*.json"))
    rows = json.loads(trace_path.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("record_type") == "cuda_kernel" and row.get("iteration") == 2:
            row["end_us"] = row["start_us"] - 1
    trace_path.write_text(json.dumps(rows), encoding="utf-8")

    failures = gate.PROFILES["local-layerwise-cuda-kernels"].contract(trace_root)

    assert [failure.code for failure in failures] == [
        "trace.gpt.cuda_graph_kernel_timeline"
    ]
    assert failures[0].evidence == "rank=0 iteration=2"


def test_local_full_iteration_profile_keeps_only_optimizer_outside_replay(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "local-full-iteration-cuda-graph"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1,)),
        model_iterations=frozenset((1,)),
        iteration_ids=(1, 2, 3),
        microbatches=2,
        include_schedule_finalize=True,
    )

    assert (
        gate.PROFILES["local-full-iteration-cuda-graph"].contract(trace_root)
        == ()
    )


def test_local_full_iteration_profile_rejects_captured_events_during_replay(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "local-full-iteration-cuda-graph-inner-replay"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=True,
        eager_layers=2,
        include_optimizer=True,
        include_backward=True,
        eager_iterations=frozenset((1,)),
        model_iterations=frozenset((1, 2)),
        iteration_ids=(1, 2, 3),
        microbatches=2,
        include_schedule_finalize=True,
    )

    failures = gate.PROFILES["local-full-iteration-cuda-graph"].contract(
        trace_root
    )

    assert [failure.code for failure in failures] == [
        "trace.gpt.full_iteration_replay_inner"
    ]
    assert failures[0].evidence == "rank=0 iteration=2"


def test_qwen3_tp2_profile_requires_28_eager_layers_on_both_ranks(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "qwen3-tp2"
    for rank in (0, 1):
        _write_gpt_phase_trace(
            trace_root,
            rank=rank,
            pipeline_rank=0,
            tensor_rank=rank,
            include_postprocess=True,
            eager_layers=28,
        )

    assert gpt_probe_contract.validate_qwen3_tp2_eager_phases(trace_root) == ()

    incomplete_root = tmp_path / "qwen3-tp2-incomplete"
    for rank in (0, 1):
        _write_gpt_phase_trace(
            incomplete_root,
            rank=rank,
            pipeline_rank=0,
            tensor_rank=rank,
            include_postprocess=True,
            eager_layers=27 if rank == 1 else 28,
        )

    failures = gpt_probe_contract.validate_qwen3_tp2_eager_phases(
        incomplete_root
    )
    assert {failure.code for failure in failures} == {"trace.gpt.eager_layers"}
    assert [failure.evidence for failure in failures] == [
        "rank=1 iteration=1",
        "rank=1 iteration=2",
    ]


def test_qwen3_tp4_profile_requires_28_eager_layers_on_all_ranks(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "qwen3-tp4"
    for rank in range(4):
        _write_gpt_phase_trace(
            trace_root,
            rank=rank,
            pipeline_rank=0,
            tensor_rank=rank,
            include_postprocess=True,
            eager_layers=28,
        )

    assert gpt_probe_contract.validate_qwen3_tp4_eager_phases(trace_root) == ()

    incomplete_root = tmp_path / "qwen3-tp4-incomplete"
    for rank in range(4):
        _write_gpt_phase_trace(
            incomplete_root,
            rank=rank,
            pipeline_rank=0,
            tensor_rank=rank,
            include_postprocess=True,
            eager_layers=27 if rank == 3 else 28,
        )

    failures = gpt_probe_contract.validate_qwen3_tp4_eager_phases(
        incomplete_root
    )
    assert {failure.code for failure in failures} == {"trace.gpt.eager_layers"}
    assert [failure.evidence for failure in failures] == [
        "rank=3 iteration=1",
        "rank=3 iteration=2",
    ]


def test_qwen3_tp8_profile_requires_28_eager_layers_on_all_ranks(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "qwen3-tp8"
    for rank in range(8):
        _write_gpt_phase_trace(
            trace_root,
            rank=rank,
            pipeline_rank=0,
            tensor_rank=rank,
            include_postprocess=True,
            eager_layers=28,
        )

    assert gpt_probe_contract.validate_qwen3_tp8_eager_phases(trace_root) == ()

    incomplete_root = tmp_path / "qwen3-tp8-incomplete"
    for rank in range(8):
        _write_gpt_phase_trace(
            incomplete_root,
            rank=rank,
            pipeline_rank=0,
            tensor_rank=rank,
            include_postprocess=True,
            eager_layers=27 if rank == 7 else 28,
        )

    failures = gpt_probe_contract.validate_qwen3_tp8_eager_phases(
        incomplete_root
    )
    assert {failure.code for failure in failures} == {"trace.gpt.eager_layers"}
    assert [failure.evidence for failure in failures] == [
        "rank=7 iteration=1",
        "rank=7 iteration=2",
    ]


def test_gpt_pp2_profile_rejects_missing_optimizer_postprocess(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "missing-optimizer-postprocess"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=False,
        include_optimizer=True,
        include_optimizer_postprocess=False,
    )
    _write_gpt_phase_trace(
        trace_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
        include_optimizer=True,
        include_optimizer_postprocess=False,
    )

    failures = gpt_probe_contract.validate_gpt_pp2_training_phases(trace_root)

    assert {failure.code for failure in failures} == {"trace.optimizer.sequence"}
    assert len(failures) == 4


def test_pp2_batched_profile_accepts_internal_wait_route(tmp_path: Path) -> None:
    trace_root = tmp_path / "pp2-batched"
    for rank in (0, 1):
        _write_gpt_phase_trace(
            trace_root,
            rank=rank,
            pipeline_rank=rank,
            include_postprocess=rank == 1,
            include_optimizer=True,
            p2p_route="batch",
        )

    assert gate.PROFILES["pp2"].contract(trace_root) == ()


def test_pp2_batched_steady_profile_accepts_steady_operation_groups(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "pp2-batched-steady"
    for rank in (0, 1):
        _write_gpt_phase_trace(
            trace_root,
            rank=rank,
            pipeline_rank=rank,
            include_postprocess=rank == 1,
            include_optimizer=True,
            p2p_route="batch-steady",
        )

    aggregate_completions = []
    for path in trace_root.glob("benchmark-*.json"):
        aggregate_completions.extend(
            row
            for row in json.loads(path.read_text(encoding="utf-8"))
            if row.get("name") == "p2p-batch-complete"
            and row.get("ph") == "B"
        )
    assert len(aggregate_completions) == 4
    assert all(
        completion["operation_count"] == 2
        and completion["request_pairing"] == "aggregate"
        and completion["physical_request_count"] == 1
        and "completion_site" not in completion
        for completion in aggregate_completions
    )
    assert gate.PROFILES["pp2-batched-steady"].contract(trace_root) == ()


def test_pp2_batched_steady_contract_rejects_reordered_aggregate_ids(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "pp2-batched-steady-reordered-aggregate"
    for rank in (0, 1):
        _write_gpt_phase_trace(
            trace_root,
            rank=rank,
            pipeline_rank=rank,
            include_postprocess=rank == 1,
            p2p_route="batch-steady",
        )

    rank_zero = next(trace_root.glob("*pipeline-0-tensor-0.json"))
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    completion = next(
        row
        for row in rows
        if row.get("name") == "p2p-batch-complete" and row.get("ph") == "B"
    )
    completion["operation_ids"] = list(reversed(completion["operation_ids"]))
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = p2p_probe_contract.validate_pp2_batched_steady_route(trace_root)

    assert "trace.p2p.identity" in {failure.code for failure in failures}


def test_pp2_batched_steady_contract_requires_completed_aggregate_end(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "pp2-batched-steady-incomplete-aggregate"
    for rank in (0, 1):
        _write_gpt_phase_trace(
            trace_root,
            rank=rank,
            pipeline_rank=rank,
            include_postprocess=rank == 1,
            p2p_route="batch-steady",
        )

    rank_zero = next(trace_root.glob("*pipeline-0-tensor-0.json"))
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    completion_end = next(
        row
        for row in rows
        if row.get("name") == "p2p-batch-complete" and row.get("ph") == "E"
    )
    completion_end["completed"] = False
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = p2p_probe_contract.validate_pp2_batched_steady_route(trace_root)

    assert "trace.p2p.field" in {failure.code for failure in failures}


def test_pp2_batched_steady_contract_rejects_single_microbatch_route(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "pp2-batched-no-steady"
    for rank in (0, 1):
        _write_gpt_phase_trace(
            trace_root,
            rank=rank,
            pipeline_rank=rank,
            include_postprocess=rank == 1,
            p2p_route="batch",
        )

    failures = p2p_probe_contract.validate_pp2_batched_steady_route(trace_root)

    assert "trace.p2p.batch_structure" in {
        failure.code for failure in failures
    }


def test_pp2_unbatched_profile_accepts_mixed_wait_route(tmp_path: Path) -> None:
    trace_root = tmp_path / "pp2-unbatched"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=False,
        p2p_route="unbatched",
    )
    _write_gpt_phase_trace(
        trace_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
        p2p_route="unbatched",
    )

    assert gate.PROFILES["pp2-unbatched"].contract(trace_root) == ()


def test_pp2_unbatched_warmup_flush_profile_accepts_exact_lifecycle(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "pp2-unbatched-warmup-flush"
    for rank in (0, 1):
        _write_pp2_unbatched_warmup_flush_trace(trace_root, rank=rank)

    assert gate.PROFILES["pp2-unbatched-warmup-flush"].contract(trace_root) == ()


def test_pp2_overlap_timeline_profile_accepts_device_kernel_interval_overlap(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "pp2-overlap-timeline"
    for rank in (0, 1):
        _write_pp2_unbatched_warmup_flush_trace(
            trace_root, rank=rank, kernel_timeline=True
        )

    assert gate.PROFILES["pp2-overlap-timeline"].contract(trace_root) == ()


def test_pp2_overlap_timeline_profile_rejects_disjoint_kernels(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "pp2-overlap-timeline-disjoint"
    for rank in (0, 1):
        _write_pp2_unbatched_warmup_flush_trace(
            trace_root, rank=rank, kernel_timeline=True
        )

    rank_one = next(trace_root.glob("*pipeline-1-tensor-0.json"))
    rows = json.loads(rank_one.read_text(encoding="utf-8"))
    model_kernel = next(
        row
        for row in rows
        if row.get("record_type") == "cuda_kernel"
        and row.get("name") == "transformer_engine::model_gemm"
    )
    model_kernel.update(
        {
            "start_us": 250,
            "end_us": 260,
            "wall_start_us": 1250,
            "wall_end_us": 1260,
            "iter_rel_start_us": 250,
            "iter_rel_end_us": 260,
        }
    )
    rank_one.write_text(json.dumps(rows), encoding="utf-8")

    failures = p2p_probe_contract.validate_pp2_overlap_timeline_route(trace_root)

    assert "trace.p2p.kernel_overlap" in {failure.code for failure in failures}


def test_pp2_overlap_timeline_profile_rejects_generic_triton_kernel(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "pp2-overlap-timeline-generic-triton"
    for rank in (0, 1):
        _write_pp2_unbatched_warmup_flush_trace(
            trace_root, rank=rank, kernel_timeline=True
        )

    rank_one = next(trace_root.glob("*pipeline-1-tensor-0.json"))
    rows = json.loads(rank_one.read_text(encoding="utf-8"))
    model_kernel = next(
        row
        for row in rows
        if row.get("record_type") == "cuda_kernel"
        and row.get("name") == "transformer_engine::model_gemm"
    )
    model_kernel["name"] = "triton_unrelated_kernel"
    rank_one.write_text(json.dumps(rows), encoding="utf-8")

    failures = p2p_probe_contract.validate_pp2_overlap_timeline_route(trace_root)

    assert "trace.p2p.kernel_overlap" in {failure.code for failure in failures}


def test_pp2_unbatched_warmup_flush_rejects_the_wrong_unwaited_send(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "pp2-unbatched-warmup-flush-wrong-gap"
    for rank in (0, 1):
        _write_pp2_unbatched_warmup_flush_trace(trace_root, rank=rank)

    rank_zero = next(trace_root.glob("*pipeline-0-tensor-0.json"))
    rows = json.loads(rank_zero.read_text(encoding="utf-8"))
    launches = [
        row
        for row in rows
        if row.get("name") == "p2p-launch" and row.get("ph") == "B"
    ]
    missing_launch = launches[3]
    replaced_launch = launches[9]
    missing_operation = missing_launch["operations"][0]
    replacement_index = next(
        index
        for index, row in enumerate(rows)
        if row.get("ph") == "B"
        and row.get("batch_id") == replaced_launch["batch_id"]
        and row.get("name") in {
            "send-forward",
            "recv-forward",
            "send-backward",
            "recv-backward",
        }
    )
    replacement = rows[replacement_index]
    replacement.update(missing_operation)
    replacement.update(
        {
            "name": "send-forward",
            "batch_id": missing_launch["batch_id"],
            "operation_ids": [missing_operation["operation_id"]],
        }
    )
    rows[replacement_index + 1]["name"] = "send-forward"
    rank_zero.write_text(json.dumps(rows), encoding="utf-8")

    failures = p2p_probe_contract.validate_pp2_unbatched_warmup_flush_route(
        trace_root
    )

    assert "trace.p2p.warmup_flush.completion_identity" in {
        failure.code for failure in failures
    }


def test_pp2_unbatched_warmup_flush_rejects_wrong_launch_mode(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "pp2-unbatched-warmup-flush-wrong-mode"
    for rank in (0, 1):
        _write_pp2_unbatched_warmup_flush_trace(trace_root, rank=rank)

    rank_one = next(trace_root.glob("*pipeline-1-tensor-0.json"))
    rows = json.loads(rank_one.read_text(encoding="utf-8"))
    launches = [
        row
        for row in rows
        if row.get("name") == "p2p-launch" and row.get("ph") == "B"
    ]
    changed_launch = launches[1]
    changed_launch["completion_mode"] = "internal_wait"
    changed_launch["operations"][0]["completion_mode"] = "internal_wait"
    completion = next(
        row
        for row in rows
        if row.get("ph") == "B"
        and row.get("operation_id")
        == changed_launch["operations"][0]["operation_id"]
        and row.get("name") == "recv-forward"
    )
    completion["completion_mode"] = "internal_wait"
    completion["completion_site"] = "communicate_internal_wait"
    rank_one.write_text(json.dumps(rows), encoding="utf-8")

    failures = p2p_probe_contract.validate_pp2_unbatched_warmup_flush_route(
        trace_root
    )

    assert "trace.p2p.warmup_flush.launch_structure" in {
        failure.code for failure in failures
    }


def test_pp2_unbatched_contract_rejects_batched_route(tmp_path: Path) -> None:
    trace_root = tmp_path / "wrong-pp2-route"
    _write_gpt_phase_trace(
        trace_root,
        rank=0,
        pipeline_rank=0,
        include_postprocess=False,
        p2p_route="batch",
    )
    _write_gpt_phase_trace(
        trace_root,
        rank=1,
        pipeline_rank=1,
        include_postprocess=True,
        p2p_route="batch",
    )

    failures = p2p_probe_contract.validate_pp2_unbatched_route(trace_root)

    codes = {failure.code for failure in failures}
    assert "trace.p2p.field" in codes
    assert "trace.p2p.sync_count" in codes
    assert "trace.p2p.external_wait" in codes


def test_pp2_ring_contract_accepts_inline_api_return(tmp_path: Path) -> None:
    trace_root = tmp_path / "pp2-ring"
    for rank in (0, 1):
        _write_gpt_phase_trace(
            trace_root,
            rank=rank,
            pipeline_rank=rank,
            include_postprocess=rank == 1,
            p2p_route="ring",
        )

    assert p2p_probe_contract.validate_pp2_ring_route(trace_root) == ()


def test_pp2_ring_contract_rejects_work_wait_events(tmp_path: Path) -> None:
    trace_root = tmp_path / "wrong-pp2-ring"
    for rank in (0, 1):
        _write_gpt_phase_trace(
            trace_root,
            rank=rank,
            pipeline_rank=rank,
            include_postprocess=rank == 1,
            p2p_route="ring",
            ring_directional_wait=rank == 0,
        )

    failures = p2p_probe_contract.validate_pp2_ring_route(trace_root)

    assert "trace.p2p.ring_completion" in {
        failure.code for failure in failures
    }


def test_ring_exchange_preflight_reports_selected_torch_build() -> None:
    fake_torch = SimpleNamespace(
        __file__=__file__,
        __version__="2.9.0+cu128",
        distributed=SimpleNamespace(ring_exchange=lambda **kwargs: None),
    )

    available, detail = check_ring_exchange.describe_capability(fake_torch)

    assert available is True
    assert detail == (
        f"ring_exchange=available torch=2.9.0+cu128 "
        f"torch_path={Path(__file__).resolve()}"
    )


def test_ring_exchange_preflight_rejects_a_standard_torch_build() -> None:
    fake_torch = SimpleNamespace(
        __file__=__file__,
        __version__="2.8.0+cpu",
        distributed=SimpleNamespace(),
    )

    available, detail = check_ring_exchange.describe_capability(fake_torch)

    assert available is False
    assert detail == (
        f"ring_exchange=unavailable torch=2.8.0+cpu "
        f"torch_path={Path(__file__).resolve()}"
    )


def test_runner_uses_requested_image_current_source_and_flagscale_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncode, run_dir, command = _invoke(tmp_path, monkeypatch)
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 0
    assert payload["status"] == "completed"
    assert payload["image"] == "example/flagscale:dev"
    assert payload["source"]["path"] == str(gate._REPOSITORY_ROOT.resolve())
    assert payload["source"]["head"] == gate._source_head(gate._REPOSITORY_ROOT)
    source_mount = f"{gate._REPOSITORY_ROOT.resolve()}:{gate.CONTAINER_SOURCE_ROOT}:ro"
    dataset_overlay = (
        f"{run_dir / 'build' / 'megatron-core-datasets'}:"
        f"{gate.CONTAINER_SOURCE_ROOT}/megatron/core/datasets"
    )
    assert source_mount in command
    assert dataset_overlay in command
    assert command.index(source_mount) < command.index("example/flagscale:dev")
    assert command.index(source_mount) < command.index(dataset_overlay)
    assert (run_dir / "build" / "megatron-core-datasets" / "Makefile").is_file()
    assert not tuple(
        (run_dir / "build" / "megatron-core-datasets").glob("helpers_cpp*.so")
    )
    assert any("conda activate flagscale-train" in argument for argument in command)
    assert ("flagscale", "run") == command[-5:-3]
    assert command[-1] == "--action=test"


def test_runner_records_config_returncode_log_and_trace_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncode, run_dir, command = _invoke(tmp_path, monkeypatch)
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 0
    assert payload["command"] == list(command)
    assert payload["config"]["path"].endswith("inputs/flagscale_single_node_smoke.yaml")
    assert len(payload["config"]["sha256"]) == 64
    assert payload["execution"]["returncode"] == 0
    assert Path(payload["execution"]["log"]).read_text(encoding="utf-8") == (
        "official FlagScale entrypoint\n"
    )
    assert payload["validation"]["trace"]["ranks"] == [0]
    assert len(payload["validation"]["trace"]["shards"]) == 1


def test_trace_off_succeeds_without_trace_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncode, run_dir, _ = _invoke(tmp_path, monkeypatch, mode="trace-off")
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 0
    assert payload["mode"] == "trace-off"
    assert payload["validation"]["trace"]["shards"] == []


def test_trace_off_rejects_a_missing_terminal_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncode, run_dir, _ = _invoke(
        tmp_path,
        monkeypatch,
        mode="trace-off",
        write_checkpoint=False,
    )
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 1
    assert payload["execution"]["returncode"] == 0
    assert payload["status"] == "failed"
    assert {failure["code"] for failure in payload["validation"]["failures"]} == {
        "run.training.tracker",
        "run.training.checkpoint",
    }


def test_child_failure_is_preserved_in_simple_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncode, run_dir, _ = _invoke(
        tmp_path,
        monkeypatch,
        child_returncode=23,
    )
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 23
    assert payload["status"] == "failed"
    assert payload["execution"]["returncode"] == 23
    assert payload["validation"]["passed"] is False


def test_profile_failure_changes_zero_child_exit_to_gate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncode, run_dir, _ = _invoke(
        tmp_path,
        monkeypatch,
        config="flagscale_single_node_pp2_smoke.yaml",
    )
    payload = json.loads((run_dir / manifest.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert returncode == 1
    assert payload["execution"]["returncode"] == 0
    assert {failure["code"] for failure in payload["validation"]["failures"]} >= {
        "trace.rank_count",
        "trace.event_missing",
    }


def test_docker_command_keeps_optional_flagscale_overlay_without_hash_lock(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "training.py"
    overlay.write_text("# compatibility overlay\n", encoding="utf-8")
    command = gate._docker_command(
        run_dir=tmp_path,
        config_name="smoke",
        mode="trace-on",
        image="example/flagscale:dev",
        source_root=gate._REPOSITORY_ROOT,
        rdzv_port=12345,
        ep_dispatcher=None,
        flagscale_training_overlay=overlay,
    )

    assert (
        f"{overlay.resolve()}:"
        "/workspace/FlagScale/flagscale/train/megatron/training/training.py:ro"
    ) in command
    assert not any("sha256:" in argument for argument in command)


def test_docker_command_mounts_checkpoint_input_read_only(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()

    command = gate._docker_command(
        run_dir=tmp_path,
        config_name="mimo-resume",
        mode="trace-on",
        image="example/flagscale:dev",
        source_root=gate._REPOSITORY_ROOT,
        rdzv_port=12345,
        ep_dispatcher=None,
        flagscale_training_overlay=None,
        checkpoint_load_root=checkpoint_root,
    )

    assert (
        f"{checkpoint_root.resolve()}:{gate.CONTAINER_CHECKPOINT_LOAD_ROOT}:ro"
    ) in command


def test_docker_command_mounts_qwen3_inputs_read_only(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    data_prefix = data_root / "enron_emails_demo_text_document_qwen"
    Path(f"{data_prefix}.bin").write_bytes(b"tokens")
    Path(f"{data_prefix}.idx").write_bytes(b"index")
    tokenizer_root = tmp_path / "qwentokenizer"
    tokenizer_root.mkdir()

    command = gate._docker_command(
        run_dir=tmp_path,
        config_name="qwen3-enron",
        mode="trace-on",
        image="example/flagscale:dev",
        source_root=gate._REPOSITORY_ROOT,
        rdzv_port=12345,
        ep_dispatcher=None,
        flagscale_training_overlay=None,
        qwen3_data_prefix=data_prefix,
        qwen3_tokenizer_root=tokenizer_root,
    )

    assert (
        f"{data_root.resolve()}:{gate.CONTAINER_QWEN3_DATA_ROOT}:ro"
    ) in command
    assert (
        f"{tokenizer_root.resolve()}:{gate.CONTAINER_QWEN3_TOKENIZER_ROOT}:ro"
    ) in command
    assert (
        "MEGALENS_QWEN3_DATA_PATH="
        f"{gate.CONTAINER_QWEN3_DATA_ROOT}/{data_prefix.name}"
    ) in command
    assert (
        f"MEGALENS_QWEN3_TOKENIZER_PATH={gate.CONTAINER_QWEN3_TOKENIZER_ROOT}"
        in command
    )
