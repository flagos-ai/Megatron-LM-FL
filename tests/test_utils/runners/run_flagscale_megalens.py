# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Run a FlagScale training profile and record compact MegaLens probe evidence."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tests.test_utils.runners import bridge_probe_contract  # noqa: E402
from tests.test_utils.runners import combined_1f1b_probe_contract  # noqa: E402
from tests.test_utils.runners import cp_probe_contract  # noqa: E402
from tests.test_utils.runners import deepseek_tp2_sp_probe_contract  # noqa: E402
from tests.test_utils.runners import dp_probe_contract  # noqa: E402
from tests.test_utils.runners import dualpipev_probe_contract  # noqa: E402
from tests.test_utils.runners import generate_bert_smoke_inputs  # noqa: E402
from tests.test_utils.runners import gpt_probe_contract  # noqa: E402
from tests.test_utils.runners import megalens_run_manifest as manifest  # noqa: E402
from tests.test_utils.runners import mimo_probe_contract  # noqa: E402
from tests.test_utils.runners import mimo_pretrain_probe_contract  # noqa: E402
from tests.test_utils.runners import moe_capacity_probe_contract  # noqa: E402
from tests.test_utils.runners import moe_flex_deepep_probe_contract  # noqa: E402
from tests.test_utils.runners import moe_flex_hybridep_probe_contract  # noqa: E402
from tests.test_utils.runners import moe_recompute_fp8_probe_contract  # noqa: E402
from tests.test_utils.runners import moe_shared_expert_overlap_probe_contract  # noqa: E402
from tests.test_utils.runners import moe_shared_expert_probe_contract  # noqa: E402
from tests.test_utils.runners import p2p_probe_contract  # noqa: E402
from tests.test_utils.runners import tp_probe_contract  # noqa: E402
from tests.test_utils.runners import training_run_contract  # noqa: E402

CONTAINER_SOURCE_ROOT = "/workspace/Megatron-LM-FL"
CONTAINER_RUN_ROOT = "/artifacts/run"
CONTAINER_CHECKPOINT_LOAD_ROOT = "/artifacts/load/checkpoints"
CONTAINER_QWEN3_DATA_ROOT = "/inputs/qwen3-data"
CONTAINER_QWEN3_TOKENIZER_ROOT = "/inputs/qwen3-tokenizer"
CONTAINER_DEEPSEEK_TOKENIZER_ROOT = "/inputs/deepseek-tokenizer"

_COMMON_FIELDS = ("g_rk", "dp_rk", "pp_rk", "tp_rk")


def _events(*names: str) -> tuple[manifest.EventRequirement, ...]:
    return tuple(
        manifest.EventRequirement(name=name, fields=_COMMON_FIELDS) for name in names
    )


def _contracts(
    *checks: Callable[[Path], Sequence[manifest.Failure]],
) -> Callable[[Path], tuple[manifest.Failure, ...]]:
    def validate(trace_root: Path) -> tuple[manifest.Failure, ...]:
        return tuple(
            failure
            for check in checks
            for failure in check(trace_root)
        )

    return validate


_PP2_EVENTS = _events(
    "forward-step",
    "decoder",
    "decoder-postprocess",
    "output_layer",
    "loss",
    "p2p-launch",
    "p2p-batch-device-sync",
    "send-forward",
    "recv-forward",
    "send-backward",
    "recv-backward",
    "optimizer",
    "optimizer-step",
    "optimizer-postprocess",
)
_PP2_UNBATCHED_EVENTS = _events(
    "p2p-launch",
    "send-forward",
    "recv-forward",
    "send-backward",
    "recv-backward",
)
_PP2_UNBATCHED_WARMUP_FLUSH_EVENTS = (
    *_PP2_UNBATCHED_EVENTS,
    *_events("forward-step", "backward-step"),
)
_EP_COMMON_EVENTS = (
    manifest.EventRequirement(
        "moe-router",
        (*_COMMON_FIELDS, "layer", "num_experts", "router_topk"),
        "E",
    ),
    manifest.EventRequirement(
        "moe-dispatch",
        (
            *_COMMON_FIELDS,
            "layer",
            "dispatcher",
            "num_tokens",
            "dropped_tokens",
            "drop_rate",
            "expert_cv",
            "top1_expert_share",
            "aux_loss",
            "z_loss",
        ),
        "E",
    ),
    *_events("moe-experts", "moe-combine"),
)
_EP_ROUTE_FIELDS = (*_COMMON_FIELDS, "comm_type", "dispatcher", "group_size")
_EP_FLEX_ROUTE_FIELDS = (*_EP_ROUTE_FIELDS, "data_bytes", "ep_size", "tp_size")
_DP_FIELDS = (
    *_COMMON_FIELDS,
    "op",
    "data_bytes",
    "group_size",
    "operation_id",
    "payload_role",
)
_CP_TE_EVENTS = (
    *_events(
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
        "grad-sync",
        "all-grads-sync",
    ),
    manifest.EventRequirement(
        "dp-allreduce",
        (*_DP_FIELDS, "group_role", "stage"),
        "B",
    ),
)
_TP_COLLECTIVE_FIELDS = (
    *_COMMON_FIELDS,
    "op",
    "dim",
    "data_bytes",
    "group_size",
)
_TP_ALLREDUCE_FIELDS = (
    *_COMMON_FIELDS,
    "op",
    "data_bytes",
    "group_size",
    "timing_phase",
    "payload_role",
)
_TP2_SP_EVENTS = (
    *(
        manifest.EventRequirement(
            name,
            _TP_COLLECTIVE_FIELDS,
            "B",
        )
        for name in (
            "tp-all-gather-first",
            "tp-all-gather-last",
            "tp-reduce-scatter",
            "tp-reduce-scatter-last",
        )
    ),
    manifest.EventRequirement(
        "tp-linear-async-launch",
        (*_COMMON_FIELDS, "operation_id", "collective_op", "launch_site"),
        "B",
    ),
    manifest.EventRequirement(
        "tp-linear-async-complete",
        (
            *_COMMON_FIELDS,
            "operation_id",
            "collective_op",
            "completion_kind",
        ),
        "B",
    ),
    manifest.EventRequirement(
        "grad-sync",
        (*_COMMON_FIELDS, "schedule", "timing_phase"),
        "B",
    ),
    manifest.EventRequirement("all-grads-sync", _COMMON_FIELDS, "B"),
    manifest.EventRequirement(
        "sp-layernorm-allreduce",
        (
            *_COMMON_FIELDS,
            "data_bytes",
            "group_size",
            "reduce_op",
            "grad_bucket",
        ),
        "B",
    ),
)
_TP2_SP_TE_LINEAR_EVENTS = (
    *_TP2_SP_EVENTS,
    *_events("transformer_layer", "attention", "MLP.forward"),
)
_TP2_SP_TE_OP_FUSER_EVENTS = (
    *_TP2_SP_EVENTS,
    *_events(
        "transformer_layer",
        "_forward_attention",
        "attention",
        "_forward_mlp",
    ),
)
_QWEN3_TP_SP_EVENTS = (
    *_events(
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
    ),
    *(
        requirement
        for requirement in _TP2_SP_EVENTS
        if requirement.name
        not in {"tp-all-gather-last", "tp-reduce-scatter-last"}
    ),
)
_QWEN3_TP4_LOCAL_NO_SP_EVENTS = (
    *_events(
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
    ),
    manifest.EventRequirement(
        "tp-allreduce",
        _TP_ALLREDUCE_FIELDS,
        "B",
    ),
    manifest.EventRequirement(
        "tp-linear-async-launch",
        (*_COMMON_FIELDS, "operation_id", "collective_op", "launch_site"),
        "B",
    ),
    manifest.EventRequirement(
        "tp-linear-async-complete",
        (
            *_COMMON_FIELDS,
            "operation_id",
            "collective_op",
            "completion_kind",
        ),
        "B",
    ),
    manifest.EventRequirement(
        "grad-sync",
        (*_COMMON_FIELDS, "schedule", "timing_phase"),
        "B",
    ),
    manifest.EventRequirement("all-grads-sync", _COMMON_FIELDS, "B"),
    manifest.EventRequirement(
        "sp-layernorm-allreduce",
        (
            *_COMMON_FIELDS,
            "data_bytes",
            "group_size",
            "reduce_op",
            "grad_bucket",
        ),
        "B",
    ),
)
_DEEPSEEK_TP2_SP_MODEL_EVENTS = (
    *_events(
        "forward-step",
        "decoder",
        "decoder-postprocess",
        "output_layer",
        "loss",
    ),
    *(
        requirement
        for requirement in _TP2_SP_EVENTS
        if requirement.name != "tp-reduce-scatter-last"
    ),
    manifest.EventRequirement("tp-allreduce", _TP_ALLREDUCE_FIELDS, "B"),
)
_TP2_EP4_MODEL_EVENTS = (
    *(
        requirement
        for requirement in _TP2_SP_EVENTS
        if requirement.name
        in {
            "tp-all-gather-first",
            "tp-reduce-scatter",
            "tp-linear-async-launch",
            "tp-linear-async-complete",
        }
    ),
    manifest.EventRequirement("tp-allreduce", _TP_ALLREDUCE_FIELDS, "B"),
)
_DUALPIPEV_PHASE_FIELDS = (
    *_COMMON_FIELDS,
    "current_microbatch",
    "dualpipev_stage",
    "operation_id",
    "schedule",
    "schedule_phase",
    "uses_model_graph",
)
_DUALPIPEV_COMBINED_FIELDS = (
    *_COMMON_FIELDS,
    "operation_id",
    "forward_operation_id",
    "backward_operation_id",
    "forward_microbatch",
    "backward_microbatch",
    "forward_dualpipev_stage",
    "backward_dualpipev_stage",
    "execution_mode",
    "overlap_active",
    "schedule",
    "schedule_phase",
)
_DUALPIPEV_A2A_FIELDS = (
    *_COMMON_FIELDS,
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
)
_DUALPIPEV_EVENTS = (
    manifest.EventRequirement("forward-step", _DUALPIPEV_PHASE_FIELDS, "B"),
    manifest.EventRequirement("backward-step", _DUALPIPEV_PHASE_FIELDS, "B"),
    manifest.EventRequirement(
        "combined-forward-backward-step",
        _DUALPIPEV_COMBINED_FIELDS,
        "B",
    ),
    manifest.EventRequirement(
        "ep-alltoall-async-launch",
        (*_DUALPIPEV_A2A_FIELDS, "async_op", "completion_included"),
        "B",
    ),
    manifest.EventRequirement(
        "ep-alltoall-async-complete",
        (
            *_DUALPIPEV_A2A_FIELDS,
            "completion_site",
            "terminal",
            "wait_role",
        ),
        "B",
    ),
    manifest.EventRequirement(
        "moe-router",
        (*_COMMON_FIELDS, "layer", "num_experts", "router_topk"),
        "E",
    ),
    *_events(
        "p2p-launch",
        "send-forward",
        "recv-forward",
        "send-backward",
        "recv-backward",
    ),
    manifest.EventRequirement(
        "grad-sync",
        (*_COMMON_FIELDS, "schedule", "timing_phase"),
        "B",
    ),
    manifest.EventRequirement("all-grads-sync", _COMMON_FIELDS, "B"),
    manifest.EventRequirement("dp-allreduce", _DP_FIELDS, "B"),
)
_BRIDGE_DIRECTION_FIELDS = (
    *_COMMON_FIELDS,
    "communicator_kind",
    "data_bytes",
    "direction",
    "message_kind",
    "peer_rank",
    "pipeline_direction",
    "src_module",
    "dest_module",
    "transport_api",
)
_BRIDGE_EVENTS = (
    manifest.EventRequirement(
        "bridge-p2p-launch",
        (
            *_COMMON_FIELDS,
            "batch_id",
            "message_kind",
            "operation_count",
            "operations",
            "src_module",
            "dest_module",
            "transport_api",
        ),
        "B",
    ),
    manifest.EventRequirement(
        "bridge-grid-broadcast",
        (
            *_COMMON_FIELDS,
            "collective_role",
            "grid_side",
            "message_kind",
            "pipeline_direction",
            "source_rank",
            "src_module",
            "dest_module",
            "transport_api",
        ),
        "B",
    ),
    *(
        manifest.EventRequirement(name, _BRIDGE_DIRECTION_FIELDS, "B")
        for name in (
            "bridge-send-forward",
            "bridge-recv-forward",
            "bridge-send-backward",
            "bridge-recv-backward",
        )
    ),
)
_MIMO_TERMINAL_EVENTS = (
    *_BRIDGE_EVENTS,
    manifest.EventRequirement("optimizer-step", _COMMON_FIELDS, "B"),
)
_MIMO_PRETRAIN_EVENTS = _events(
    "forward-step",
    "loss",
    "optimizer",
    "optimizer-step",
    "optimizer-postprocess",
)


def _ep_events(
    dispatcher: str,
    *,
    fine_grained: bool = False,
) -> tuple[manifest.EventRequirement, ...]:
    route_events = (
        manifest.EventRequirement(f"ep-{dispatcher}-dispatch", _EP_ROUTE_FIELDS, "E"),
        manifest.EventRequirement(f"ep-{dispatcher}-combine", _EP_ROUTE_FIELDS, "E"),
    )
    scheduler = (
        (
            manifest.EventRequirement(
                "combined-forward-backward-step",
                (*_COMMON_FIELDS, "execution_mode"),
                "B",
            ),
        )
        if fine_grained
        else ()
    )
    return (*_EP_COMMON_EVENTS, *route_events, *scheduler)


def _ep_capacity_drop_events() -> tuple[manifest.EventRequirement, ...]:
    base_events = _ep_events("alltoall")
    return (
        manifest.EventRequirement(
            "moe-router",
            (
                *_COMMON_FIELDS,
                "layer",
                "ep_size",
                "num_experts",
                "num_local_experts",
                "router_topk",
                "num_tokens",
                "routed_tokens",
                "dropped_tokens",
                "drop_rate",
                "expert_cv",
                "top1_expert_share",
                "routing_entropy",
                "aux_loss",
                "z_loss",
            ),
            "E",
        ),
        manifest.EventRequirement(
            "moe-dispatch",
            (
                *base_events[1].fields,
                "ep_size",
                "num_experts",
                "num_local_experts",
                "router_topk",
                "capacity_factor",
            ),
            "E",
        ),
        *base_events[2:],
    )


def _ep_shared_expert_events() -> tuple[manifest.EventRequirement, ...]:
    return (
        *_ep_events("alltoall"),
        manifest.EventRequirement(
            "moe-shared-expert",
            (*_COMMON_FIELDS, "layer", "ep_size"),
            "E",
        ),
    )


def _ep_shared_expert_overlap_events() -> tuple[manifest.EventRequirement, ...]:
    return (
        *_ep_events("alltoall"),
        manifest.EventRequirement(
            "moe-shared-expert",
            (*_COMMON_FIELDS, "layer", "ep_size", "stage"),
            "E",
        ),
    )


def _ep_flex_events() -> tuple[manifest.EventRequirement, ...]:
    return (
        *_EP_COMMON_EVENTS,
        manifest.EventRequirement("ep-alltoall-dispatch", _EP_FLEX_ROUTE_FIELDS, "E"),
        manifest.EventRequirement("ep-alltoall-combine", _EP_FLEX_ROUTE_FIELDS, "E"),
    )


def _dp_events(profile: str) -> tuple[manifest.EventRequirement, ...]:
    names = (
        ("dp-allreduce",)
        if profile == "standard-ddp"
        else ("dp-reduce-scatter", "dp-param-all-gather")
    )
    return tuple(manifest.EventRequirement(name, _DP_FIELDS, "B") for name in names)


_QWEN3_CP_EVENTS = (
    *_events(
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
        "grad-sync",
        "all-grads-sync",
    ),
    *_dp_events("distopt"),
    manifest.EventRequirement(
        "dp-grad-sync-complete",
        (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
        "B",
    ),
    manifest.EventRequirement(
        "dp-param-sync-complete",
        (*_COMMON_FIELDS, "operation_id", "completion_kind"),
        "B",
    ),
)


QWEN3_CP2_DP8_OFFLINE_PROFILE = manifest.TraceProfile(
    "qwen3-enron-cp2-dp8",
    16,
    _QWEN3_CP_EVENTS,
    cp_probe_contract.validate_qwen3_cp2_dp8_distopt_coexistence,
    training_run_contract.validate_two_iteration_qwen3_cp2_dp8_checkpoint,
)


# Profiles selectable by the loopback runner. Multi-node validation profiles
# remain separate and are applied only to already completed run artifacts.
PROFILES: Mapping[str, manifest.TraceProfile] = {
    "pp1": manifest.TraceProfile(
        "pp1",
        1,
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    "gpt-eager-full": manifest.TraceProfile(
        "gpt-eager-full",
        1,
        _events(
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
        ),
        gpt_probe_contract.validate_gpt_pp1_eager_phases,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "cp2-te": manifest.TraceProfile(
        "cp2-te",
        2,
        _CP_TE_EVENTS,
        cp_probe_contract.validate_cp2_te_coexistence,
        training_run_contract.validate_two_iteration_transformer_engine_cp2_checkpoint,
    ),
    "cp4-te": manifest.TraceProfile(
        "cp4-te",
        4,
        _CP_TE_EVENTS,
        cp_probe_contract.validate_cp4_te_coexistence,
        training_run_contract.validate_two_iteration_transformer_engine_cp4_checkpoint,
    ),
    "tp2-sp-local": manifest.TraceProfile(
        "tp2-sp-local",
        2,
        _TP2_SP_EVENTS,
        tp_probe_contract.validate_tp2_sp_profile,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "tp2-sp-te-linear": manifest.TraceProfile(
        "tp2-sp-te-linear",
        2,
        _TP2_SP_TE_LINEAR_EVENTS,
        tp_probe_contract.validate_tp2_sp_te_linear_profile,
        training_run_contract.validate_two_iteration_transformer_engine_checkpoint,
    ),
    "tp2-sp-te-userbuffer": manifest.TraceProfile(
        "tp2-sp-te-userbuffer",
        2,
        _TP2_SP_TE_LINEAR_EVENTS,
        tp_probe_contract.validate_tp2_sp_te_linear_profile,
        training_run_contract.validate_two_iteration_transformer_engine_userbuffer_checkpoint,
    ),
    "tp2-sp-te-op-fuser": manifest.TraceProfile(
        "tp2-sp-te-op-fuser",
        2,
        _TP2_SP_TE_OP_FUSER_EVENTS,
        tp_probe_contract.validate_tp2_sp_te_op_fuser_profile,
        training_run_contract.validate_two_iteration_transformer_engine_op_fuser_checkpoint,
    ),
    "qwen3-enron-tp2-sp": manifest.TraceProfile(
        "qwen3-enron-tp2-sp",
        2,
        _QWEN3_TP_SP_EVENTS,
        _contracts(
            gpt_probe_contract.validate_qwen3_tp2_eager_phases,
            tp_probe_contract.validate_qwen3_tp2_sp_profile,
        ),
        training_run_contract.validate_two_iteration_qwen3_tp2_sp_checkpoint,
    ),
    "qwen3-enron-tp4-sp": manifest.TraceProfile(
        "qwen3-enron-tp4-sp",
        4,
        _QWEN3_TP_SP_EVENTS,
        _contracts(
            gpt_probe_contract.validate_qwen3_tp4_eager_phases,
            tp_probe_contract.validate_qwen3_tp4_sp_profile,
        ),
        training_run_contract.validate_two_iteration_qwen3_tp4_sp_checkpoint,
    ),
    "qwen3-enron-tp4-local-no-sp": manifest.TraceProfile(
        "qwen3-enron-tp4-local-no-sp",
        4,
        _QWEN3_TP4_LOCAL_NO_SP_EVENTS,
        _contracts(
            gpt_probe_contract.validate_qwen3_tp4_eager_phases,
            tp_probe_contract.validate_qwen3_tp4_local_no_sp_profile,
        ),
        training_run_contract.validate_two_iteration_qwen3_tp4_local_no_sp_checkpoint,
    ),
    "qwen3-enron-tp8-sp": manifest.TraceProfile(
        "qwen3-enron-tp8-sp",
        8,
        _QWEN3_TP_SP_EVENTS,
        _contracts(
            gpt_probe_contract.validate_qwen3_tp8_eager_phases,
            tp_probe_contract.validate_qwen3_tp8_sp_profile,
        ),
        training_run_contract.validate_two_iteration_qwen3_tp8_sp_checkpoint,
    ),
    "qwen3-enron-cp2": manifest.TraceProfile(
        "qwen3-enron-cp2",
        2,
        _QWEN3_CP_EVENTS,
        cp_probe_contract.validate_qwen3_cp2_distopt_coexistence,
        training_run_contract.validate_two_iteration_qwen3_cp2_checkpoint,
    ),
    "qwen3-enron-cp4": manifest.TraceProfile(
        "qwen3-enron-cp4",
        4,
        _QWEN3_CP_EVENTS,
        cp_probe_contract.validate_qwen3_cp4_distopt_coexistence,
        training_run_contract.validate_two_iteration_qwen3_cp4_checkpoint,
    ),
    "deepseek-tp2-sp-mock": manifest.TraceProfile(
        "deepseek-tp2-sp-mock",
        8,
        (
            *_DEEPSEEK_TP2_SP_MODEL_EVENTS,
            *_ep_capacity_drop_events(),
            manifest.EventRequirement(
                "moe-shared-expert",
                (*_COMMON_FIELDS, "layer", "ep_size"),
                "E",
            ),
        ),
        deepseek_tp2_sp_probe_contract.validate_deepseek_tp2_sp_trace,
        training_run_contract.validate_two_iteration_deepseek_tp2_sp_checkpoint,
    ),
    "tp2-local-allreduce": manifest.TraceProfile(
        "tp2-local-allreduce",
        2,
        (
            manifest.EventRequirement(
                "tp-allreduce",
                _TP_ALLREDUCE_FIELDS,
                "B",
            ),
            *(
                manifest.EventRequirement(
                    name,
                    _TP_COLLECTIVE_FIELDS,
                    "B",
                )
                for name in (
                    "tp-all-gather-last",
                    "tp-reduce-scatter",
                    "tp-reduce-scatter-last",
                )
            ),
            manifest.EventRequirement(
                "tp-linear-async-launch",
                (*_COMMON_FIELDS, "operation_id", "collective_op", "launch_site"),
                "B",
            ),
            manifest.EventRequirement(
                "tp-linear-async-complete",
                (
                    *_COMMON_FIELDS,
                    "operation_id",
                    "collective_op",
                    "completion_kind",
                ),
                "B",
            ),
            manifest.EventRequirement(
                "grad-sync",
                (*_COMMON_FIELDS, "schedule", "timing_phase"),
                "B",
            ),
            manifest.EventRequirement("all-grads-sync", _COMMON_FIELDS, "B"),
        ),
        tp_probe_contract.validate_tp2_local_allreduce_profile,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "tp2-pp2-embedding": manifest.TraceProfile(
        "tp2-pp2-embedding",
        4,
        (
            manifest.EventRequirement(
                "grad-sync",
                (*_COMMON_FIELDS, "schedule", "timing_phase"),
                "B",
            ),
            manifest.EventRequirement("all-grads-sync", _COMMON_FIELDS, "B"),
            manifest.EventRequirement(
                "sp-layernorm-allreduce",
                (
                    *_COMMON_FIELDS,
                    "data_bytes",
                    "group_size",
                    "reduce_op",
                    "grad_bucket",
                ),
                "B",
            ),
            manifest.EventRequirement(
                "embedding-grads-allreduce",
                (
                    *_COMMON_FIELDS,
                    "data_bytes",
                    "group_size",
                    "embedding_kind",
                ),
                "B",
            ),
        ),
        tp_probe_contract.validate_tp2_pp2_embedding_final_grad_sync,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "pp2": manifest.TraceProfile(
        "pp2",
        2,
        _PP2_EVENTS,
        _contracts(
            gpt_probe_contract.validate_gpt_pp2_training_phases,
            p2p_probe_contract.validate_pp2_batched_route,
        ),
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "pp2-batched-steady": manifest.TraceProfile(
        "pp2-batched-steady",
        2,
        _PP2_EVENTS,
        p2p_probe_contract.validate_pp2_batched_steady_route,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "pp2-unbatched": manifest.TraceProfile(
        "pp2-unbatched",
        2,
        _PP2_UNBATCHED_EVENTS,
        p2p_probe_contract.validate_pp2_unbatched_route,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "pp2-unbatched-warmup-flush": manifest.TraceProfile(
        "pp2-unbatched-warmup-flush",
        2,
        _PP2_UNBATCHED_WARMUP_FLUSH_EVENTS,
        p2p_probe_contract.validate_pp2_unbatched_warmup_flush_route,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "pp2-overlap-timeline": manifest.TraceProfile(
        "pp2-overlap-timeline",
        2,
        _PP2_UNBATCHED_WARMUP_FLUSH_EVENTS,
        p2p_probe_contract.validate_pp2_overlap_timeline_route,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "ep2-alltoall": manifest.TraceProfile(
        "ep2-alltoall",
        2,
        _ep_events("alltoall"),
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    "ep2-recompute": manifest.TraceProfile(
        "ep2-recompute",
        2,
        _ep_capacity_drop_events(),
        moe_recompute_fp8_probe_contract.validate_ep2_recompute,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "ep2-fp8": manifest.TraceProfile(
        "ep2-fp8",
        2,
        _ep_capacity_drop_events(),
        moe_recompute_fp8_probe_contract.validate_ep2_fp8,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "ep2-fp8-recompute": manifest.TraceProfile(
        "ep2-fp8-recompute",
        2,
        _ep_capacity_drop_events(),
        moe_recompute_fp8_probe_contract.validate_ep2_fp8_recompute,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "ep2-alltoall-capacity-drop": manifest.TraceProfile(
        "ep2-alltoall-capacity-drop",
        2,
        _ep_capacity_drop_events(),
        moe_capacity_probe_contract.validate_ep2_capacity_drop,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "ep2-alltoall-shared-expert": manifest.TraceProfile(
        "ep2-alltoall-shared-expert",
        2,
        _ep_shared_expert_events(),
        moe_shared_expert_probe_contract.validate_ep2_shared_expert,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "ep2-alltoall-shared-expert-overlap": manifest.TraceProfile(
        "ep2-alltoall-shared-expert-overlap",
        2,
        _ep_shared_expert_overlap_events(),
        moe_shared_expert_overlap_probe_contract.validate_ep2_shared_expert_overlap,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "tp2-ep4-flex-deepep": manifest.TraceProfile(
        "tp2-ep4-flex-deepep",
        8,
        (*_ep_flex_events(), *_TP2_EP4_MODEL_EVENTS),
        moe_flex_deepep_probe_contract.validate_tp2_ep4_flex_deepep,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "tp2-ep4-flex-hybridep": manifest.TraceProfile(
        "tp2-ep4-flex-hybridep",
        8,
        _ep_flex_events(),
        moe_flex_hybridep_probe_contract.validate_tp2_ep4_flex_hybridep,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "ep2-allgather": manifest.TraceProfile(
        "ep2-allgather",
        2,
        _ep_events("allgather"),
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    "ep2-fine-grained": manifest.TraceProfile(
        "ep2-fine-grained",
        4,
        _ep_events("alltoall", fine_grained=True),
        combined_1f1b_probe_contract.validate_ep2_fine_grained_combined,
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    "pp2-dp2-ep2-dualpipev": manifest.TraceProfile(
        "pp2-dp2-ep2-dualpipev",
        4,
        _DUALPIPEV_EVENTS,
        dualpipev_probe_contract.validate_dualpipev_route,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "dp2-standard-ddp": manifest.TraceProfile(
        "dp2-standard-ddp",
        2,
        _dp_events("standard-ddp"),
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    "dp2-standard-ddp-overlap": manifest.TraceProfile(
        "dp2-standard-ddp-overlap",
        2,
        (
            *_dp_events("standard-ddp"),
            manifest.EventRequirement(
                "dp-grad-sync-complete",
                (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
                "B",
            ),
        ),
        dp_probe_contract.validate_dp_standard_overlap,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "dp2-distopt": manifest.TraceProfile(
        "dp2-distopt",
        2,
        _dp_events("distopt"),
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    "dp2-distopt-overlap": manifest.TraceProfile(
        "dp2-distopt-overlap",
        2,
        (
            *_dp_events("distopt"),
            manifest.EventRequirement(
                "dp-grad-sync-complete",
                (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
                "B",
            ),
            manifest.EventRequirement(
                "dp-param-sync-complete",
                (*_COMMON_FIELDS, "operation_id", "completion_kind"),
                "B",
            ),
        ),
        dp_probe_contract.validate_dp_distopt_overlap,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "pp2-dp2-distopt-force-sync": manifest.TraceProfile(
        "pp2-dp2-distopt-force-sync",
        4,
        (
            *_dp_events("distopt"),
            manifest.EventRequirement(
                "dp-grad-sync-complete",
                (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
                "B",
            ),
            manifest.EventRequirement(
                "dp-param-sync-complete",
                (*_COMMON_FIELDS, "operation_id", "completion_kind"),
                "B",
            ),
        ),
        dp_probe_contract.validate_dp_optimizer_step_force_sync_training,
        training_run_contract.validate_two_iteration_legacy_pp2_force_sync,
    ),
    "dp2-layerwise-overlap": manifest.TraceProfile(
        "dp2-layerwise-overlap",
        2,
        (
            *_dp_events("standard-ddp"),
            manifest.EventRequirement("dp-param-all-gather", _DP_FIELDS, "B"),
            manifest.EventRequirement(
                "dp-grad-sync-complete",
                (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
                "B",
            ),
            manifest.EventRequirement(
                "dp-param-sync-complete",
                (*_COMMON_FIELDS, "operation_id", "completion_kind"),
                "B",
            ),
        ),
        dp_probe_contract.validate_dp_layerwise_overlap,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "dp4-distopt-multi-instance-overlap": manifest.TraceProfile(
        "dp4-distopt-multi-instance-overlap",
        4,
        (
            *_dp_events("distopt"),
            manifest.EventRequirement("dp-allreduce", _DP_FIELDS, "B"),
            manifest.EventRequirement(
                "dp-grad-sync-complete",
                (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
                "B",
            ),
            manifest.EventRequirement(
                "dp-param-sync-complete",
                (*_COMMON_FIELDS, "operation_id", "completion_kind"),
                "B",
            ),
        ),
        dp_probe_contract.validate_dp_multi_instance_distopt_overlap,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "dp8-standard-ddp": manifest.TraceProfile(
        "dp8-standard-ddp",
        8,
        _dp_events("standard-ddp"),
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    "dp8-standard-ddp-overlap": manifest.TraceProfile(
        "dp8-standard-ddp-overlap",
        8,
        (
            *_dp_events("standard-ddp"),
            manifest.EventRequirement(
                "dp-grad-sync-complete",
                (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
                "B",
            ),
        ),
        dp_probe_contract.validate_dp_standard_overlap,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "dp8-distopt": manifest.TraceProfile(
        "dp8-distopt",
        8,
        _dp_events("distopt"),
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    "dp8-distopt-overlap": manifest.TraceProfile(
        "dp8-distopt-overlap",
        8,
        (
            *_dp_events("distopt"),
            manifest.EventRequirement(
                "dp-grad-sync-complete",
                (*_COMMON_FIELDS, "operation_ids", "completion_kind"),
                "B",
            ),
            manifest.EventRequirement(
                "dp-param-sync-complete",
                (*_COMMON_FIELDS, "operation_id", "completion_kind"),
                "B",
            ),
        ),
        dp_probe_contract.validate_dp_distopt_overlap,
        training_run_contract.validate_two_iteration_checkpoint,
    ),
    "te-attn-cuda-graph": manifest.TraceProfile(
        "te-attn-cuda-graph",
        1,
        _events("MLP.forward"),
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    "te-moe-router-cuda-graph": manifest.TraceProfile(
        "te-moe-router-cuda-graph",
        2,
        (
            *_EP_COMMON_EVENTS,
            manifest.EventRequirement("ep-alltoall-dispatch", _EP_ROUTE_FIELDS, "E"),
            manifest.EventRequirement("ep-alltoall-combine", _EP_ROUTE_FIELDS, "E"),
        ),
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    "bert-encoder": manifest.TraceProfile(
        "bert-encoder",
        1,
        _events("encoder"),
        run_contract=training_run_contract.validate_two_iteration_checkpoint,
    ),
    "multimodule-bridge2": manifest.TraceProfile(
        "multimodule-bridge2",
        2,
        _BRIDGE_EVENTS,
        bridge_probe_contract.validate_multimodule_bridge_trace,
        bridge_probe_contract.validate_multimodule_bridge_run,
    ),
    "multimodule-bridge8-fanin": manifest.TraceProfile(
        "multimodule-bridge8-fanin",
        8,
        _BRIDGE_EVENTS,
        bridge_probe_contract.validate_multimodule_bridge_fanin_trace,
        bridge_probe_contract.validate_multimodule_bridge_fanin_run,
    ),
    "multimodule-bridge8-fanout": manifest.TraceProfile(
        "multimodule-bridge8-fanout",
        8,
        _BRIDGE_EVENTS,
        bridge_probe_contract.validate_multimodule_bridge_fanout_trace,
        bridge_probe_contract.validate_multimodule_bridge_fanout_run,
    ),
    "mimo-train2": manifest.TraceProfile(
        "mimo-train2",
        2,
        _MIMO_TERMINAL_EVENTS,
        mimo_probe_contract.validate_mimo_training_trace,
        mimo_probe_contract.validate_mimo_training_run,
    ),
    "mimo-pretrain2": manifest.TraceProfile(
        "mimo-pretrain2",
        2,
        _MIMO_PRETRAIN_EVENTS,
        mimo_pretrain_probe_contract.validate_mimo_pretrain_trace,
        mimo_pretrain_probe_contract.validate_mimo_pretrain_run,
    ),
    "mimo-pretrain-save2": manifest.TraceProfile(
        "mimo-pretrain-save2",
        2,
        _MIMO_PRETRAIN_EVENTS,
        mimo_pretrain_probe_contract.validate_mimo_pretrain_save_trace,
        mimo_pretrain_probe_contract.validate_mimo_pretrain_save_run,
    ),
    "mimo-pretrain-resume2": manifest.TraceProfile(
        "mimo-pretrain-resume2",
        2,
        _MIMO_PRETRAIN_EVENTS,
        mimo_pretrain_probe_contract.validate_mimo_pretrain_resume_trace,
        mimo_pretrain_probe_contract.validate_mimo_pretrain_resume_run,
    ),
    "mimo-train8-fanin": manifest.TraceProfile(
        "mimo-train8-fanin",
        8,
        _MIMO_TERMINAL_EVENTS,
        mimo_probe_contract.validate_mimo_training_fanin_trace,
        mimo_probe_contract.validate_mimo_training_fanin_run,
    ),
    "mimo-train8-fanout": manifest.TraceProfile(
        "mimo-train8-fanout",
        8,
        _MIMO_TERMINAL_EVENTS,
        mimo_probe_contract.validate_mimo_training_fanout_trace,
        mimo_probe_contract.validate_mimo_training_fanout_run,
    ),
}

_CONFIG_PROFILES = {
    "flagscale_single_node_smoke": "pp1",
    "flagscale_single_node_gpt_eager_full_smoke": "gpt-eager-full",
    "flagscale_single_node_cp2_te_smoke": "cp2-te",
    "flagscale_single_node_cp4_te_smoke": "cp4-te",
    "flagscale_single_node_tp2_sp_local_smoke": "tp2-sp-local",
    "flagscale_single_node_tp2_sp_te_linear_smoke": "tp2-sp-te-linear",
    "flagscale_single_node_tp2_sp_te_userbuffer_smoke": "tp2-sp-te-userbuffer",
    "flagscale_single_node_tp2_sp_te_op_fuser_smoke": "tp2-sp-te-op-fuser",
    "flagscale_single_node_qwen3_enron_tp2_sp": "qwen3-enron-tp2-sp",
    "flagscale_single_node_qwen3_enron_tp4_sp": "qwen3-enron-tp4-sp",
    "flagscale_single_node_qwen3_enron_tp4_local_no_sp": (
        "qwen3-enron-tp4-local-no-sp"
    ),
    "flagscale_single_node_qwen3_enron_tp8_sp": "qwen3-enron-tp8-sp",
    "flagscale_single_node_qwen3_enron_cp2": "qwen3-enron-cp2",
    "flagscale_single_node_qwen3_enron_cp4": "qwen3-enron-cp4",
    "flagscale_single_node_deepseek_tp2_sp_mock": "deepseek-tp2-sp-mock",
    "flagscale_single_node_tp2_local_allreduce_smoke": "tp2-local-allreduce",
    "flagscale_single_node_tp2_pp2_embedding_smoke": "tp2-pp2-embedding",
    "flagscale_single_node_pp2_smoke": "pp2",
    "flagscale_single_node_pp2_batched_steady_smoke": "pp2-batched-steady",
    "flagscale_single_node_pp2_unbatched_smoke": "pp2-unbatched",
    "flagscale_single_node_pp2_unbatched_warmup_flush_smoke": (
        "pp2-unbatched-warmup-flush"
    ),
    "flagscale_single_node_pp2_overlap_timeline_smoke": "pp2-overlap-timeline",
    "flagscale_single_node_ep2_smoke": "ep2-alltoall",
    "flagscale_single_node_ep2_recompute_smoke": "ep2-recompute",
    "flagscale_single_node_ep2_fp8_smoke": "ep2-fp8",
    "flagscale_single_node_ep2_fp8_recompute_smoke": "ep2-fp8-recompute",
    "flagscale_single_node_ep2_capacity_drop_smoke": (
        "ep2-alltoall-capacity-drop"
    ),
    "flagscale_single_node_ep2_shared_expert_smoke": (
        "ep2-alltoall-shared-expert"
    ),
    "flagscale_single_node_ep2_shared_expert_overlap_smoke": (
        "ep2-alltoall-shared-expert-overlap"
    ),
    "flagscale_single_node_tp2_ep4_flex_deepep_smoke": (
        "tp2-ep4-flex-deepep"
    ),
    "flagscale_single_node_tp2_ep4_flex_hybridep_smoke": (
        "tp2-ep4-flex-hybridep"
    ),
    "flagscale_single_node_ep2_fine_grained_smoke": "ep2-fine-grained",
    "flagscale_single_node_pp2_dp2_ep2_dualpipev_smoke": (
        "pp2-dp2-ep2-dualpipev"
    ),
    "flagscale_single_node_dp2_standard_smoke": "dp2-standard-ddp",
    "flagscale_single_node_dp2_standard_overlap_smoke": (
        "dp2-standard-ddp-overlap"
    ),
    "flagscale_single_node_dp2_distopt_smoke": "dp2-distopt",
    "flagscale_single_node_dp2_distopt_overlap_smoke": "dp2-distopt-overlap",
    "flagscale_single_node_pp2_dp2_distopt_force_sync_smoke": (
        "pp2-dp2-distopt-force-sync"
    ),
    "flagscale_single_node_dp2_layerwise_overlap_smoke": "dp2-layerwise-overlap",
    "flagscale_single_node_dp4_distopt_multi_instance_overlap_smoke": (
        "dp4-distopt-multi-instance-overlap"
    ),
    "flagscale_single_node_dp8_standard_smoke": "dp8-standard-ddp",
    "flagscale_single_node_dp8_standard_overlap_smoke": (
        "dp8-standard-ddp-overlap"
    ),
    "flagscale_single_node_dp8_distopt_smoke": "dp8-distopt",
    "flagscale_single_node_dp8_distopt_overlap_smoke": "dp8-distopt-overlap",
    "flagscale_single_node_te_cuda_graph_attn_smoke": "te-attn-cuda-graph",
    "flagscale_single_node_te_cuda_graph_moe_router_smoke": (
        "te-moe-router-cuda-graph"
    ),
    "flagscale_single_node_bert_smoke": "bert-encoder",
    "flagscale_single_node_multimodule_bridge_smoke": "multimodule-bridge2",
    "flagscale_single_node_multimodule_bridge_fanin": (
        "multimodule-bridge8-fanin"
    ),
    "flagscale_single_node_multimodule_bridge_fanout": (
        "multimodule-bridge8-fanout"
    ),
    "flagscale_single_node_mimo_smoke": "mimo-train2",
    "flagscale_single_node_mimo_pretrain_smoke": "mimo-pretrain2",
    "flagscale_single_node_mimo_pretrain_save_smoke": "mimo-pretrain-save2",
    "flagscale_single_node_mimo_pretrain_resume_smoke": "mimo-pretrain-resume2",
    "flagscale_single_node_mimo_fanin": "mimo-train8-fanin",
    "flagscale_single_node_mimo_fanout": "mimo-train8-fanout",
}

_OFFLINE_CONFIG_PROFILES = {
    "flagscale_dual_node_qwen3_enron_cp2_dp8": QWEN3_CP2_DP8_OFFLINE_PROFILE,
}


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    timed_out: bool = False


def _source_head(source_root: Path) -> str:
    result = subprocess.run(
        ("git", "-C", str(source_root), "rev-parse", "HEAD"),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"cannot read source HEAD: {detail}")
    return result.stdout.strip()


def _reserve_loopback_port(requested: int) -> int:
    if requested:
        if not 1 <= requested <= 65535:
            raise ValueError("--rdzv-port must be between 1 and 65535")
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _profile_from_arguments(args: argparse.Namespace) -> manifest.TraceProfile:
    offline_profile = _OFFLINE_CONFIG_PROFILES.get(args.input_config.stem)
    if offline_profile is not None:
        raise ValueError(
            f"profile {offline_profile.name!r} requires manual dual-node "
            "orchestration and offline artifact validation"
        )
    if args.profile is not None:
        return PROFILES[args.profile]
    configured = _CONFIG_PROFILES.get(args.input_config.stem)
    if configured is not None:
        if configured == "ep2-alltoall" and args.ep_dispatcher == "allgather":
            configured = "ep2-allgather"
        return PROFILES[configured]
    if args.model_profile == "bert-encoder":
        return PROFILES["bert-encoder"]
    if args.cuda_graph_profile == "transformer-engine-attn":
        return PROFILES["te-attn-cuda-graph"]
    if args.cuda_graph_profile == "transformer-engine-moe-router":
        return PROFILES["te-moe-router-cuda-graph"]
    if args.topology == "tp2":
        return PROFILES["tp2-sp-local"]
    if args.topology == "pp2":
        return PROFILES["pp2"]
    if args.topology == "ep2":
        if args.ep_profile == "fine-grained":
            return PROFILES["ep2-fine-grained"]
        return PROFILES[f"ep2-{args.ep_dispatcher or 'alltoall'}"]
    if args.topology == "dp4":
        return PROFILES[
            f"dp4-{args.dp_profile or 'distopt-multi-instance-overlap'}"
        ]
    if args.topology in {"dp2", "dp8"}:
        return PROFILES[f"{args.topology}-{args.dp_profile or 'standard-ddp'}"]
    return PROFILES["pp1"]


def _ep_dispatcher(profile: manifest.TraceProfile, requested: str | None) -> str | None:
    if requested is not None:
        return requested
    if "alltoall" in profile.name or profile.name in {
        "ep2-fine-grained",
        "ep2-recompute",
        "ep2-fp8",
        "ep2-fp8-recompute",
        "te-moe-router-cuda-graph",
    }:
        return "alltoall"
    if "allgather" in profile.name:
        return "allgather"
    if profile.name in {"tp2-ep4-flex-deepep", "tp2-ep4-flex-hybridep"}:
        return "flex"
    return None


def _docker_command(
    *,
    run_dir: Path,
    config_name: str,
    mode: str,
    image: str,
    source_root: Path,
    rdzv_port: int,
    ep_dispatcher: str | None,
    flagscale_training_overlay: Path | None,
    dataset_helper_overlay: Path | None = None,
    checkpoint_load_root: Path | None = None,
    qwen3_data_prefix: Path | None = None,
    qwen3_tokenizer_root: Path | None = None,
    deepseek_tokenizer_root: Path | None = None,
) -> tuple[str, ...]:
    overlay = ()
    if flagscale_training_overlay is not None:
        overlay = (
            "--volume",
            f"{flagscale_training_overlay}:"
            "/workspace/FlagScale/flagscale/train/megatron/training/training.py:ro",
        )
    dataset_overlay = ()
    if dataset_helper_overlay is not None:
        dataset_overlay = (
            "--volume",
            f"{dataset_helper_overlay}:"
            f"{CONTAINER_SOURCE_ROOT}/megatron/core/datasets",
        )
    checkpoint_overlay = ()
    if checkpoint_load_root is not None:
        checkpoint_overlay = (
            "--volume",
            f"{checkpoint_load_root}:{CONTAINER_CHECKPOINT_LOAD_ROOT}:ro",
        )
    qwen3_inputs = ()
    if qwen3_data_prefix is not None and qwen3_tokenizer_root is not None:
        container_data_prefix = (
            f"{CONTAINER_QWEN3_DATA_ROOT}/{qwen3_data_prefix.name}"
        )
        qwen3_inputs = (
            "--volume",
            f"{qwen3_data_prefix.parent}:{CONTAINER_QWEN3_DATA_ROOT}:ro",
            "--volume",
            f"{qwen3_tokenizer_root}:{CONTAINER_QWEN3_TOKENIZER_ROOT}:ro",
            "--env",
            f"MEGALENS_QWEN3_DATA_PATH={container_data_prefix}",
            "--env",
            f"MEGALENS_QWEN3_TOKENIZER_PATH={CONTAINER_QWEN3_TOKENIZER_ROOT}",
        )
    deepseek_inputs = ()
    if deepseek_tokenizer_root is not None:
        deepseek_inputs = (
            "--volume",
            f"{deepseek_tokenizer_root}:{CONTAINER_DEEPSEEK_TOKENIZER_ROOT}:ro",
            "--env",
            "MEGALENS_DEEPSEEK_TOKENIZER_PATH="
            f"{CONTAINER_DEEPSEEK_TOKENIZER_ROOT}",
        )
    dispatcher = ()
    if ep_dispatcher is not None:
        dispatcher = ("--env", f"MEGALENS_GATE_EP_DISPATCHER={ep_dispatcher}")

    shell = (
        "set -euo pipefail; "
        "source /root/miniconda3/etc/profile.d/conda.sh; "
        "conda activate flagscale-train; "
        "export PYTHONPATH=/workspace/FlagScale:"
        "/workspace/FlagScale/flagscale/train:/workspace/Megatron-LM-FL:"
        "${PYTHONPATH:-}; "
        "cd /workspace/FlagScale; "
        'exec "$@"'
    )
    return (
        "docker",
        "run",
        "--rm",
        "--privileged",
        "--runtime=nvidia",
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size=64g",
        "--ulimit",
        "memlock=-1:-1",
        "--volume",
        f"{run_dir}:{CONTAINER_RUN_ROOT}",
        "--volume",
        f"{source_root}:{CONTAINER_SOURCE_ROOT}:ro",
        *dataset_overlay,
        *checkpoint_overlay,
        *qwen3_inputs,
        *deepseek_inputs,
        *overlay,
        "--env",
        f"MEGALENS_GATE_CONTAINER_RUN_DIR={CONTAINER_RUN_ROOT}",
        "--env",
        f"MEGALENS_GATE_RDZV_ENDPOINT=127.0.0.1:{rdzv_port}",
        "--env",
        f"MEGALENS_GATE_TRACE={'true' if mode == 'trace-on' else 'false'}",
        *dispatcher,
        image,
        "/bin/bash",
        "-lc",
        shell,
        "_",
        "flagscale",
        "run",
        f"--config-path={CONTAINER_RUN_ROOT}/inputs",
        f"--config-name={config_name}",
        "--action=test",
    )


def run_foreground(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    launcher_log: Path,
    timeout: float,
) -> ExecutionResult:
    try:
        with launcher_log.open("w", encoding="utf-8") as output:
            try:
                completed = subprocess.run(
                    tuple(argv),
                    cwd=cwd,
                    env=dict(env),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                output.write(
                    f"[FlagScale MegaLens] training timed out after {timeout} seconds\n"
                )
                return ExecutionResult(124, True)
            except FileNotFoundError as error:
                output.write(f"[FlagScale MegaLens] missing executable: {error}\n")
                return ExecutionResult(127)
            except OSError as error:
                output.write(f"[FlagScale MegaLens] launcher error: {error}\n")
                return ExecutionResult(126)
    except OSError:
        return ExecutionResult(126)
    return ExecutionResult(completed.returncode)


def _copy_inputs(
    source: Path,
    run_dir: Path,
    profile: manifest.TraceProfile,
) -> Path:
    inputs = run_dir / "inputs"
    inputs.mkdir()
    destination = inputs / source.name
    shutil.copyfile(source, destination)
    if profile.name == "bert-encoder":
        generate_bert_smoke_inputs.generate_inputs(inputs)
    return destination


def _prepare_dataset_helper_overlay(source_root: Path, run_dir: Path) -> Path:
    """Copy the dataset helper sources to a writable per-run build directory."""

    source = source_root / "megatron" / "core" / "datasets"
    destination = run_dir / "build" / "megatron-core-datasets"
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "helpers_cpp*.so"),
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--input-config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("trace-on", "trace-off"))
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--topology",
        choices=("pp1", "pp2", "tp2", "ep2", "dp2", "dp4", "dp8"),
        default=None,
    )
    parser.add_argument("--ep-dispatcher", choices=("alltoall", "allgather"))
    parser.add_argument("--ep-profile", choices=("standard", "fine-grained"))
    parser.add_argument(
        "--dp-profile",
        choices=(
            "standard-ddp",
            "standard-ddp-overlap",
            "distopt",
            "distopt-overlap",
            "layerwise-overlap",
            "distopt-multi-instance-overlap",
        ),
    )
    parser.add_argument(
        "--cuda-graph-profile",
        choices=("transformer-engine-attn", "transformer-engine-moe-router"),
    )
    parser.add_argument("--model-profile", choices=("bert-encoder",))
    parser.add_argument(
        "--megatron-source-root",
        type=Path,
        default=_REPOSITORY_ROOT,
        help="current source checkout mounted over the image source",
    )
    parser.add_argument("--flagscale-training-overlay", type=Path)
    parser.add_argument("--checkpoint-load-root", type=Path)
    parser.add_argument(
        "--qwen3-data-prefix",
        type=Path,
        help="host path prefix for the Qwen3 indexed .bin/.idx dataset",
    )
    parser.add_argument(
        "--qwen3-tokenizer-root",
        type=Path,
        help="host directory containing the Qwen tokenizer files",
    )
    parser.add_argument(
        "--deepseek-tokenizer-root",
        type=Path,
        help="host directory containing the DeepSeek Qwen tokenizer files",
    )
    parser.add_argument(
        "--controller-revision",
        help="accepted for compatibility; the manifest records the mounted source HEAD",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--rdzv-port", type=int, default=0)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    return parser


def _exit_code(execution: ExecutionResult, passed: bool) -> int:
    if execution.returncode == 0:
        return 0 if passed else 1
    if execution.returncode < 0:
        return min(255, 128 - execution.returncode)
    return min(255, execution.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if not args.input_config.is_file():
        parser.error("--input-config must be a YAML file")
    if args.input_config.suffix not in {".yaml", ".yml"}:
        parser.error("--input-config must be a YAML file")
    if not args.image.strip():
        parser.error("--image cannot be empty")

    source_root = args.megatron_source_root.resolve()
    if not source_root.is_dir():
        parser.error("--megatron-source-root must be a directory")
    checkpoint_load_root = None
    if args.checkpoint_load_root is not None:
        checkpoint_load_root = args.checkpoint_load_root.resolve()
        if not checkpoint_load_root.is_dir():
            parser.error("--checkpoint-load-root must be a directory")
    try:
        source_head = _source_head(source_root)
        rdzv_port = _reserve_loopback_port(args.rdzv_port)
    except ValueError as error:
        parser.error(str(error))

    try:
        profile = _profile_from_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    qwen3_data_prefix = None
    qwen3_tokenizer_root = None
    deepseek_tokenizer_root = None
    if profile.name in {
        "qwen3-enron-tp2-sp",
        "qwen3-enron-tp4-sp",
        "qwen3-enron-tp4-local-no-sp",
        "qwen3-enron-tp8-sp",
        "qwen3-enron-cp2",
        "qwen3-enron-cp4",
    }:
        if args.qwen3_data_prefix is None:
            parser.error("--qwen3-data-prefix is required for the Qwen3 profile")
        if args.qwen3_tokenizer_root is None:
            parser.error("--qwen3-tokenizer-root is required for the Qwen3 profile")
        qwen3_data_prefix = args.qwen3_data_prefix.resolve()
        missing_dataset_files = tuple(
            Path(f"{qwen3_data_prefix}{suffix}")
            for suffix in (".bin", ".idx")
            if not Path(f"{qwen3_data_prefix}{suffix}").is_file()
        )
        if missing_dataset_files:
            parser.error(
                "Qwen3 dataset prefix is missing indexed files: "
                + ", ".join(str(path) for path in missing_dataset_files)
            )
        qwen3_tokenizer_root = args.qwen3_tokenizer_root.resolve()
        if not qwen3_tokenizer_root.is_dir():
            parser.error("--qwen3-tokenizer-root must be a directory")
    if profile.name == "deepseek-tp2-sp-mock":
        if args.deepseek_tokenizer_root is None:
            parser.error(
                "--deepseek-tokenizer-root is required for the DeepSeek profile"
            )
        deepseek_tokenizer_root = args.deepseek_tokenizer_root.resolve()
        if not deepseek_tokenizer_root.is_dir():
            parser.error("--deepseek-tokenizer-root must be a directory")
    run_dir = args.run_dir.resolve()
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(
            f"[FlagScale MegaLens] run directory already exists: {run_dir}",
            file=sys.stderr,
        )
        return 2

    started_at = manifest.utc_now()
    copied_config = _copy_inputs(args.input_config.resolve(), run_dir, profile)
    dataset_helper_overlay = _prepare_dataset_helper_overlay(source_root, run_dir)
    launcher_log = run_dir / "launcher.log"
    dispatcher = _ep_dispatcher(profile, args.ep_dispatcher)
    overlay = (
        args.flagscale_training_overlay.resolve()
        if args.flagscale_training_overlay is not None
        else None
    )
    command = _docker_command(
        run_dir=run_dir,
        config_name=copied_config.stem,
        mode=args.mode,
        image=args.image,
        source_root=source_root,
        rdzv_port=rdzv_port,
        ep_dispatcher=dispatcher,
        flagscale_training_overlay=overlay,
        dataset_helper_overlay=dataset_helper_overlay,
        checkpoint_load_root=checkpoint_load_root,
        qwen3_data_prefix=qwen3_data_prefix,
        qwen3_tokenizer_root=qwen3_tokenizer_root,
        deepseek_tokenizer_root=deepseek_tokenizer_root,
    )
    environment = os.environ.copy()
    environment.update(
        MEGALENS_GATE_RUN_DIR=str(run_dir),
        MEGALENS_GATE_MODE=args.mode,
        MEGALENS_GATE_PROFILE=profile.name,
    )
    if dispatcher is not None:
        environment["MEGALENS_GATE_EP_DISPATCHER"] = dispatcher

    execution = run_foreground(
        command,
        cwd=args.cwd.resolve(),
        env=environment,
        launcher_log=launcher_log,
        timeout=args.timeout,
    )
    report = manifest.validate_trace(
        run_dir / "traces",
        profile,
        trace_enabled=args.mode == "trace-on",
    )
    report = manifest.validate_run_artifacts(
        run_dir,
        profile,
        report,
        trace_enabled=args.mode == "trace-on",
    )
    payload = manifest.build_manifest(
        run_id=run_dir.name,
        started_at=started_at,
        finished_at=manifest.utc_now(),
        profile=profile,
        mode=args.mode,
        command=command,
        config_path=copied_config,
        source_root=source_root,
        source_head=source_head,
        image=args.image,
        returncode=execution.returncode,
        timed_out=execution.timed_out,
        log_path=launcher_log,
        report=report,
    )
    manifest.write_manifest(run_dir, payload)
    exit_code = _exit_code(execution, report.passed)
    print(
        f"[FlagScale MegaLens] status={payload['status']} "
        f"profile={profile.name} rc={exit_code} manifest={run_dir / manifest.MANIFEST_NAME}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
