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


_QWEN3_TP2_SP_ARGUMENTS = (
    ("tensor_model_parallel_size", "2"),
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


def validate_two_iteration_qwen3_tp2_sp_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require terminal state and the parsed Qwen3 TP2/SP L3 configuration."""

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

    for name, value in _QWEN3_TP2_SP_ARGUMENTS:
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
