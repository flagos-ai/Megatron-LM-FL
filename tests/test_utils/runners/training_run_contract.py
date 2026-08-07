# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Terminal artifact contract for controlled two-iteration training profiles."""

from __future__ import annotations

import re
from pathlib import Path

from tests.test_utils.runners.megalens_run_manifest import Failure


def validate_two_iteration_checkpoint(
    run_root: Path, _trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require the terminal torch_dist checkpoint from a two-iteration run."""

    failures: list[Failure] = []
    checkpoint_root = run_root / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != "2":
        failures.append(
            Failure(
                "run.training.tracker",
                "training checkpoint tracker does not point to iteration 2",
                str(tracker),
            )
        )

    common_state = checkpoint_root / "iter_0000002" / "common.pt"
    if not common_state.is_file():
        failures.append(
            Failure(
                "run.training.checkpoint",
                "training iteration 2 torch_dist checkpoint has no common state",
                str(common_state),
            )
        )
    return tuple(failures)


_TRANSFORMER_ENGINE_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*transformer_impl\s+\.+\s+transformer_engine\s*$",
    re.MULTILINE,
)
_FLAGCX_BACKEND_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*distributed_backend\s+\.+\s+flagcx\s*$",
    re.MULTILINE,
)
_TP_COMM_OVERLAP_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*tp_comm_overlap\s+\.+\s+True\s*$",
    re.MULTILINE,
)
_TE_OP_FUSER_SPEC_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*spec\s+\.+\s+"
    r"\['megalens_te_op_fuser_spec',\s*'te_op_fuser_spec'\]\s*$",
    re.MULTILINE,
)


def validate_two_iteration_transformer_engine_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal state and the parsed Transformer Engine model selection."""

    failures = list(validate_two_iteration_checkpoint(run_root, trace_enabled))
    launcher_log = run_root / "launcher.log"
    try:
        log_text = launcher_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        failures.append(
            Failure(
                "run.training.log",
                f"cannot read the training launcher log: {error}",
                str(launcher_log),
            )
        )
        return tuple(failures)

    if _TRANSFORMER_ENGINE_ARGUMENT.search(log_text) is None:
        failures.append(
            Failure(
                "run.training.transformer_impl",
                "Megatron did not report transformer_impl=transformer_engine",
                str(launcher_log),
            )
        )
    return tuple(failures)


def validate_two_iteration_flagcx_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal state and the parsed FlagCX distributed backend."""

    failures = list(validate_two_iteration_checkpoint(run_root, trace_enabled))
    launcher_log = run_root / "launcher.log"
    try:
        log_text = launcher_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        failures.append(
            Failure(
                "run.training.log",
                f"cannot read the training launcher log: {error}",
                str(launcher_log),
            )
        )
        return tuple(failures)

    if _FLAGCX_BACKEND_ARGUMENT.search(log_text) is None:
        failures.append(
            Failure(
                "run.training.distributed_backend",
                "Megatron did not report distributed_backend=flagcx",
                str(launcher_log),
            )
        )
    return tuple(failures)


_QWEN3_TP_SP_COMMON_ARGUMENTS = (
    ("pipeline_model_parallel_size", "1"),
    ("context_parallel_size", "1"),
    ("sequence_parallel", "True"),
    ("num_layers", "28"),
    ("hidden_size", "1024"),
    ("ffn_hidden_size", "3072"),
    ("num_attention_heads", "16"),
    ("num_query_groups", "8"),
    ("seq_length", "4096"),
    ("micro_batch_size", "4"),
    ("global_batch_size", "4"),
    ("train_iters", "2"),
    ("tokenizer_type", "QwenTokenizerFS"),
    ("use_distributed_optimizer", "True"),
    ("overlap_grad_reduce", "True"),
    ("overlap_param_gather", "True"),
    ("te_fl_prefer", "vendor"),
    ("enable_flag_gems", "False"),
    ("distributed_backend", "nccl"),
    ("mock_data", "False"),
)


def _qwen3_tp_sp_arguments(tensor_parallel_size: int) -> tuple[tuple[str, str], ...]:
    return (
        ("tensor_model_parallel_size", str(tensor_parallel_size)),
        *_QWEN3_TP_SP_COMMON_ARGUMENTS,
    )


_QWEN3_TP2_SP_ARGUMENTS = _qwen3_tp_sp_arguments(2)
_QWEN3_TP4_SP_ARGUMENTS = _qwen3_tp_sp_arguments(4)
_QWEN3_TP8_SP_ARGUMENTS = _qwen3_tp_sp_arguments(8)


def _qwen3_cp_arguments(
    context_parallel_size: int,
) -> tuple[tuple[str, str], ...]:
    return (
        ("tensor_model_parallel_size", "1"),
        ("pipeline_model_parallel_size", "1"),
        ("context_parallel_size", str(context_parallel_size)),
        ("sequence_parallel", "False"),
        *(
            item
            for item in _QWEN3_TP_SP_COMMON_ARGUMENTS
            if item[0]
            not in {
                "pipeline_model_parallel_size",
                "context_parallel_size",
                "sequence_parallel",
            }
        ),
    )


_QWEN3_CP2_ARGUMENTS = _qwen3_cp_arguments(2)
_QWEN3_CP4_ARGUMENTS = _qwen3_cp_arguments(4)


def _qwen3_cp2_dp_arguments(
    data_parallel_size: int,
) -> tuple[tuple[str, str], ...]:
    world_size = 2 * data_parallel_size
    global_batch_size = 4 * data_parallel_size
    return (
        *(
            item
            for item in _QWEN3_CP2_ARGUMENTS
            if item[0] != "global_batch_size"
        ),
        ("global_batch_size", str(global_batch_size)),
        ("world_size", str(world_size)),
        ("data_parallel_size", str(data_parallel_size)),
        ("num_distributed_optimizer_instances", "1"),
    )


_QWEN3_CP2_DP4_ARGUMENTS = _qwen3_cp2_dp_arguments(4)
_QWEN3_CP2_DP8_ARGUMENTS = _qwen3_cp2_dp_arguments(8)


def _validate_two_iteration_qwen3_tp_sp_checkpoint(
    run_root: Path,
    trace_enabled: bool,
    *,
    arguments: tuple[tuple[str, str], ...],
) -> tuple[Failure, ...]:
    failures = list(
        validate_two_iteration_transformer_engine_checkpoint(
            run_root, trace_enabled
        )
    )
    launcher_log = run_root / "launcher.log"
    try:
        log_text = launcher_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return tuple(failures)

    for name, value in arguments:
        argument = re.compile(
            rf"^\[[^]]+\]:\s*{re.escape(name)}\s+\.+\s+"
            rf"{re.escape(value)}\s*$",
            re.MULTILINE,
        )
        if argument.search(log_text) is None:
            failures.append(
                Failure(
                    "run.training.qwen3_argument",
                    f"Megatron did not report {name}={value}",
                    str(launcher_log),
                )
            )
    return tuple(failures)


def validate_two_iteration_qwen3_tp2_sp_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal state and the parsed Qwen3 TP2/SP L3 configuration."""

    return _validate_two_iteration_qwen3_tp_sp_checkpoint(
        run_root,
        trace_enabled,
        arguments=_QWEN3_TP2_SP_ARGUMENTS,
    )


def validate_two_iteration_qwen3_tp4_sp_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal state and the parsed Qwen3 TP4/SP L3 configuration."""

    return _validate_two_iteration_qwen3_tp_sp_checkpoint(
        run_root,
        trace_enabled,
        arguments=_QWEN3_TP4_SP_ARGUMENTS,
    )


def validate_two_iteration_qwen3_cp2_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal state and the Qwen3 CP2/DP1 DistOpt configuration."""

    return _validate_two_iteration_qwen3_tp_sp_checkpoint(
        run_root,
        trace_enabled,
        arguments=_QWEN3_CP2_ARGUMENTS,
    )


def validate_two_iteration_qwen3_cp4_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal state and the Qwen3 CP4/DP1 DistOpt configuration."""

    return _validate_two_iteration_qwen3_tp_sp_checkpoint(
        run_root,
        trace_enabled,
        arguments=_QWEN3_CP4_ARGUMENTS,
    )


def _validate_two_iteration_qwen3_cp2_dp_checkpoint(
    run_root: Path,
    trace_enabled: bool,
    *,
    data_parallel_size: int,
) -> tuple[Failure, ...]:
    """Require terminal state for a two-node Qwen3 CP2/DP DistOpt run."""

    failures = list(
        _validate_two_iteration_qwen3_tp_sp_checkpoint(
            run_root,
            trace_enabled,
            arguments=_qwen3_cp2_dp_arguments(data_parallel_size),
        )
    )
    checkpoint_root = run_root / "checkpoints" / "iter_0000002"
    observed_shards = {path.name for path in checkpoint_root.glob("*.distcp")}
    world_size = 2 * data_parallel_size
    expected_shards = {f"__{rank}_0.distcp" for rank in range(world_size)}
    if observed_shards != expected_shards:
        failures.append(
            Failure(
                "run.training.checkpoint_shards",
                "Qwen3 CP2 terminal checkpoint does not cover global ranks "
                f"0 through {world_size - 1}; "
                f"missing={sorted(expected_shards - observed_shards)}, "
                f"unexpected={sorted(observed_shards - expected_shards)}",
                str(checkpoint_root),
            )
        )
    return tuple(failures)


def validate_two_iteration_qwen3_cp2_dp4_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal state and the reduced CP2/DP4 DistOpt configuration."""

    return _validate_two_iteration_qwen3_cp2_dp_checkpoint(
        run_root, trace_enabled, data_parallel_size=4
    )


def validate_two_iteration_qwen3_cp2_dp8_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal state and the Q3 CP2/DP8 DistOpt configuration."""

    return _validate_two_iteration_qwen3_cp2_dp_checkpoint(
        run_root, trace_enabled, data_parallel_size=8
    )


_QWEN3_TP4_LOCAL_NO_SP_ARGUMENTS = (
    ("tensor_model_parallel_size", "4"),
    ("pipeline_model_parallel_size", "1"),
    ("context_parallel_size", "1"),
    ("sequence_parallel", "False"),
    ("transformer_impl", "local"),
    ("attention_backend", "AttnBackend.auto"),
    ("no_persist_layer_norm", "True"),
    ("gradient_accumulation_fusion", "False"),
    ("qk_layernorm", "True"),
    ("normalization", "RMSNorm"),
    ("num_layers", "28"),
    ("hidden_size", "1024"),
    ("ffn_hidden_size", "3072"),
    ("num_attention_heads", "16"),
    ("num_query_groups", "8"),
    ("seq_length", "4096"),
    ("micro_batch_size", "4"),
    ("global_batch_size", "4"),
    ("train_iters", "2"),
    ("tokenizer_type", "QwenTokenizerFS"),
    ("use_distributed_optimizer", "True"),
    ("overlap_grad_reduce", "True"),
    ("overlap_param_gather", "True"),
    ("enable_flag_gems", "False"),
    ("distributed_backend", "nccl"),
    ("mock_data", "False"),
)


def validate_two_iteration_qwen3_tp4_local_no_sp_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal state and the parsed Qwen3 TP4 local/no-SP control."""

    failures = list(validate_two_iteration_checkpoint(run_root, trace_enabled))
    launcher_log = run_root / "launcher.log"
    try:
        log_text = launcher_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        failures.append(
            Failure(
                "run.training.log",
                f"cannot read the training launcher log: {error}",
                str(launcher_log),
            )
        )
        return tuple(failures)

    for name, value in _QWEN3_TP4_LOCAL_NO_SP_ARGUMENTS:
        argument = re.compile(
            rf"^\[[^]]+\]:\s*{re.escape(name)}\s+\.+\s+"
            rf"{re.escape(value)}\s*$",
            re.MULTILINE,
        )
        if argument.search(log_text) is None:
            failures.append(
                Failure(
                    "run.training.qwen3_local_argument",
                    f"Megatron did not report {name}={value}",
                    str(launcher_log),
                )
            )
    return tuple(failures)


def validate_two_iteration_qwen3_tp8_sp_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal state and the parsed Qwen3 TP8/SP L3 configuration."""

    return _validate_two_iteration_qwen3_tp_sp_checkpoint(
        run_root,
        trace_enabled,
        arguments=_QWEN3_TP8_SP_ARGUMENTS,
    )


_DEEPSEEK_TP2_SP_ARGUMENTS = (
    ("tensor_model_parallel_size", "2"),
    ("pipeline_model_parallel_size", "2"),
    ("decoder_first_pipeline_num_layers", "13"),
    ("context_parallel_size", "1"),
    ("expert_model_parallel_size", "4"),
    ("expert_tensor_parallel_size", "1"),
    ("sequence_parallel", "True"),
    ("num_layers", "27"),
    ("hidden_size", "2048"),
    ("ffn_hidden_size", "11264"),
    ("moe_ffn_hidden_size", "1408"),
    ("num_attention_heads", "16"),
    ("num_query_groups", "16"),
    ("multi_latent_attention", "True"),
    ("num_experts", "64"),
    ("moe_router_topk", "6"),
    ("moe_shared_expert_intermediate_size", "2816"),
    ("moe_token_dispatcher_type", "alltoall"),
    ("mtp_num_layers", "1"),
    ("seq_length", "4096"),
    ("micro_batch_size", "1"),
    ("global_batch_size", "4"),
    ("train_iters", "2"),
    ("tokenizer_type", "QwenTokenizerFS"),
    ("use_distributed_optimizer", "True"),
    ("overlap_grad_reduce", "True"),
    ("overlap_param_gather", "True"),
    ("te_fl_prefer", "vendor"),
    ("enable_flag_gems", "False"),
    ("distributed_backend", "nccl"),
    ("mock_data", "True"),
)


def validate_two_iteration_deepseek_tp2_sp_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require the terminal checkpoint and fixed DeepSeek TP2/SP arguments."""

    failures = list(
        validate_two_iteration_transformer_engine_checkpoint(
            run_root, trace_enabled
        )
    )
    launcher_log = run_root / "launcher.log"
    try:
        log_text = launcher_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return tuple(failures)

    for name, value in _DEEPSEEK_TP2_SP_ARGUMENTS:
        argument = re.compile(
            rf"^\[[^]]+\]:\s*{re.escape(name)}\s+\.+\s+"
            rf"{re.escape(value)}\s*$",
            re.MULTILINE,
        )
        if argument.search(log_text) is None:
            failures.append(
                Failure(
                    "run.training.deepseek_argument",
                    f"Megatron did not report {name}={value}",
                    str(launcher_log),
                )
            )
    return tuple(failures)


def _validate_two_iteration_transformer_engine_cp_checkpoint(
    run_root: Path, trace_enabled: bool, *, context_parallel_size: int
) -> tuple[Failure, ...]:
    failures = list(
        validate_two_iteration_transformer_engine_checkpoint(run_root, trace_enabled)
    )
    launcher_log = run_root / "launcher.log"
    try:
        log_text = launcher_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return tuple(failures)

    context_parallel_argument = re.compile(
        rf"^\[[^]]+\]:\s*context_parallel_size\s+\.+\s+"
        rf"{context_parallel_size}\s*$",
        re.MULTILINE,
    )
    if context_parallel_argument.search(log_text) is None:
        failures.append(
            Failure(
                "run.training.context_parallel_size",
                "Megatron did not report "
                f"context_parallel_size={context_parallel_size}",
                str(launcher_log),
            )
        )
    return tuple(failures)


def validate_two_iteration_transformer_engine_cp2_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal TE training with context parallel size two selected."""

    return _validate_two_iteration_transformer_engine_cp_checkpoint(
        run_root, trace_enabled, context_parallel_size=2
    )


def validate_two_iteration_transformer_engine_cp4_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal TE training with context parallel size four selected."""

    return _validate_two_iteration_transformer_engine_cp_checkpoint(
        run_root, trace_enabled, context_parallel_size=4
    )


def validate_two_iteration_transformer_engine_userbuffer_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal TE training with tensor-parallel UserBuffer enabled."""

    failures = list(
        validate_two_iteration_transformer_engine_checkpoint(run_root, trace_enabled)
    )
    launcher_log = run_root / "launcher.log"
    try:
        log_text = launcher_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return tuple(failures)

    if _TP_COMM_OVERLAP_ARGUMENT.search(log_text) is None:
        failures.append(
            Failure(
                "run.training.tp_comm_overlap",
                "Megatron did not report tp_comm_overlap=True",
                str(launcher_log),
            )
        )
    return tuple(failures)


def validate_two_iteration_transformer_engine_op_fuser_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal TE training with the controlled op-fuser spec selected."""

    failures = list(
        validate_two_iteration_transformer_engine_checkpoint(run_root, trace_enabled)
    )
    launcher_log = run_root / "launcher.log"
    try:
        log_text = launcher_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return tuple(failures)

    if _TE_OP_FUSER_SPEC_ARGUMENT.search(log_text) is None:
        failures.append(
            Failure(
                "run.training.te_op_fuser_spec",
                "Megatron did not report the controlled TE op-fuser spec",
                str(launcher_log),
            )
        )
    return tuple(failures)


def validate_two_iteration_legacy_pp2_checkpoint(
    run_root: Path, _trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require the terminal legacy checkpoint for a PP2 distributed optimizer run."""

    failures: list[Failure] = []
    checkpoint_root = run_root / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != "2":
        failures.append(
            Failure(
                "run.training.tracker",
                "training checkpoint tracker does not point to iteration 2",
                str(tracker),
            )
        )

    iteration_root = checkpoint_root / "iter_0000002"
    for pipeline_rank in range(2):
        rank_root = iteration_root / f"mp_rank_00_{pipeline_rank:03d}"
        for name in ("model_optim_rng.pt", "distrib_optim.pt"):
            artifact = rank_root / name
            if not artifact.is_file():
                failures.append(
                    Failure(
                        "run.training.checkpoint",
                        "training iteration 2 legacy checkpoint is incomplete",
                        str(artifact),
                    )
                )
    return tuple(failures)


def validate_two_iteration_legacy_pp2_force_sync(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require legacy terminal state and every configured weight-hash callback."""

    failures = list(
        validate_two_iteration_legacy_pp2_checkpoint(run_root, trace_enabled)
    )
    launcher_log = run_root / "launcher.log"
    try:
        log_text = launcher_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        failures.append(
            Failure(
                "run.training.log",
                f"cannot read the training launcher log: {error}",
                str(launcher_log),
            )
        )
        return tuple(failures)

    for iteration in (0, 1, 2):
        marker = f">>> Weight hashes match after {iteration} iterations..."
        if marker not in log_text:
            failures.append(
                Failure(
                    "run.training.force_sync",
                    "configured DP weight-hash callback did not complete",
                    marker,
                )
            )
    return tuple(failures)
