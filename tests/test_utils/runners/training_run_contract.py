# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Terminal artifact contracts for controlled training profiles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from tests.test_utils.runners.megalens_run_manifest import Failure


def _validate_iteration_checkpoint(
    run_root: Path, iteration: int
) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    checkpoint_root = run_root / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if (
        not tracker.is_file()
        or tracker.read_text(encoding="utf-8").strip() != str(iteration)
    ):
        failures.append(
            Failure(
                "run.training.tracker",
                f"training checkpoint tracker does not point to iteration {iteration}",
                str(tracker),
            )
        )

    common_state = checkpoint_root / f"iter_{iteration:07d}" / "common.pt"
    if not common_state.is_file():
        failures.append(
            Failure(
                "run.training.checkpoint",
                f"training iteration {iteration} torch_dist checkpoint has no common state",
                str(common_state),
            )
        )
    return tuple(failures)


def validate_two_iteration_checkpoint(
    run_root: Path, _trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require the terminal torch_dist checkpoint from a two-iteration run."""

    return _validate_iteration_checkpoint(run_root, 2)


def validate_three_iteration_checkpoint(
    run_root: Path, _trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require the terminal torch_dist checkpoint from a three-iteration run."""

    return _validate_iteration_checkpoint(run_root, 3)


_TRANSFORMER_ENGINE_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*transformer_impl\s+\.+\s+transformer_engine\s*$",
    re.MULTILINE,
)
_TRANSFORMER_ENGINE_CUDA_GRAPH_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*cuda_graph_impl\s+\.+\s+transformer_engine\s*$",
    re.MULTILINE,
)
_LOCAL_CUDA_GRAPH_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*cuda_graph_impl\s+\.+\s+local\s*$",
    re.MULTILINE,
)
_WHOLE_LAYER_CUDA_GRAPH_SCOPE_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*cuda_graph_scope\s+\.+\s+\[\]\s*$",
    re.MULTILINE,
)
_ATTENTION_CUDA_GRAPH_SCOPE_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*cuda_graph_scope\s+\.+\s+"
    r"\[<CudaGraphScope\.attn:\s*\d+>\]\s*$",
    re.MULTILINE,
)
_ONE_CUDA_GRAPH_WARMUP_STEP_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*cuda_graph_warmup_steps\s+\.+\s+1\s*$",
    re.MULTILINE,
)
_ZERO_CUDA_GRAPH_WARMUP_STEPS_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*cuda_graph_warmup_steps\s+\.+\s+0\s*$",
    re.MULTILINE,
)
_FULL_ITERATION_CUDA_GRAPH_SCOPE_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*cuda_graph_scope\s+\.+\s+"
    r"\[<CudaGraphScope\.full_iteration:\s*\d+>\]\s*$",
    re.MULTILINE,
)
_NAN_CHECK_DISABLED_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*check_for_nan_in_loss_and_grad\s+\.+\s+False\s*$",
    re.MULTILINE,
)
_OPTIMIZER_CUDA_GRAPH_DISABLED_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*optimizer_cuda_graph\s+\.+\s+False\s*$",
    re.MULTILINE,
)
_SELECTIVE_RECOMPUTE_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*recompute_granularity\s+\.+\s+selective\s*$",
    re.MULTILINE,
)
_MLP_RECOMPUTE_MODULE_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*recompute_modules\s+\.+\s+\['mlp'\]\s*$",
    re.MULTILINE,
)
_CUPTI_KERNEL_DISABLED_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*trace_cupti_kernels\s+\.+\s+off\s*$",
    re.MULTILINE,
)
_TWO_ITERATION_PROGRESS = re.compile(
    r"iteration\s+(?P<iteration>[12])/\s*2\s*\|"
)
_THREE_ITERATION_PROGRESS = re.compile(
    r"iteration\s+(?P<iteration>[123])/\s*3\s*\|"
)
_CUDA_GRAPH_DELETION = re.compile(
    r"Rank 0: (?P<explicit>\d+) graphs deleted with explicit reset, "
    r"(?P<implicit>\d+) graphs deleted without explicit reset\."
)
_LOCAL_CUDA_GRAPH_BUILD = re.compile(
    r"> built (?P<count>\d+) cuda graph\(s\) in "
)
_CUPTI_KERNEL_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*trace_cupti_kernels\s+\.+\s+on\s*$",
    re.MULTILINE,
)
_TRACE_INTERVAL_TWO_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*trace_interval\s+\.+\s+2\s*$",
    re.MULTILINE,
)
_CONTINUOUS_TRACE_TWO_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*continuous_trace_iterations\s+\.+\s+2\s*$",
    re.MULTILINE,
)
_CUDA_KERNEL_EXTRACTION = re.compile(
    r"\[trace\] extracted (?P<count>\d+) cuda kernel events at iter "
    r"(?P<iteration>[12])"
)
_FLAGCX_BACKEND_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*distributed_backend\s+\.+\s+flagcx\s*$",
    re.MULTILINE,
)
_FLAGCX_BOOTSTRAP = re.compile(
    r"FLAGCX INFO Bootstrap : Using (?P<interface>[^:]+):(?P<address>[^<\s]+)<"
)
_FLAGCX_RANK = re.compile(
    r"FLAGCX INFO rank (?P<rank>\d+) nranks (?P<nranks>\d+) - DONE"
)
_FLAGCX_FATAL_MARKER = re.compile(
    r"ChildFailedError|Traceback \(most recent call last\)|"
    r"FLAGCX (?:WARN|ERROR)|CUDA error"
)
_TRAINING_ITERATIONS = re.compile(
    r"setting training iterations to (?P<iterations>[1-9]\d*)"
)
_TRAINING_PROGRESS = re.compile(
    r"iteration\s+(?P<iteration>[1-9]\d*)/\s*(?P<total>[1-9]\d*)\s*\|"
)
_TRACE_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*trace\s+\.+\s+(?P<enabled>True|False)\s*$",
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


def validate_two_iteration_continuous_cuda_kernel_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require a two-iteration CUPTI window and terminal checkpoint."""

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

    for pattern, code, message in (
        (
            _TRACE_INTERVAL_TWO_ARGUMENT,
            "run.training.trace_interval",
            "Megatron did not report trace_interval=2",
        ),
        (
            _CONTINUOUS_TRACE_TWO_ARGUMENT,
            "run.training.continuous_trace_iterations",
            "Megatron did not report continuous_trace_iterations=2",
        ),
        (
            _CUPTI_KERNEL_ARGUMENT,
            "run.training.trace_cupti_kernels",
            "Megatron did not report trace_cupti_kernels=on",
        ),
    ):
        if pattern.search(log_text) is None:
            failures.append(Failure(code, message, str(launcher_log)))

    extractions = tuple(
        (int(match.group("iteration")), int(match.group("count")))
        for match in _CUDA_KERNEL_EXTRACTION.finditer(log_text)
    )
    if trace_enabled:
        if len(extractions) != 1 or extractions[0][0] != 2 or extractions[0][1] <= 0:
            failures.append(
                Failure(
                    "run.training.cuda_kernel_capture",
                    "continuous CUPTI run must extract one positive kernel batch "
                    f"at iteration 2; observed {list(extractions)}",
                    str(launcher_log),
                )
            )
    elif extractions:
        failures.append(
            Failure(
                "run.training.cuda_kernel_capture",
                "trace-off continuous CUPTI run unexpectedly extracted CUDA "
                f"kernels; observed {list(extractions)}",
                str(launcher_log),
            )
        )

    profiler_warnings = tuple(
        marker
        for marker in (
            "Warning: failed to start kernel profiler",
            "Warning: failed to stop kernel profiler",
        )
        if marker in log_text
    )
    if profiler_warnings:
        failures.append(
            Failure(
                "run.training.cuda_kernel_profiler",
                f"kernel profiler reported failures: {list(profiler_warnings)!r}",
                str(launcher_log),
            )
        )
    return tuple(failures)


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


def _validate_two_iteration_te_cuda_graph_checkpoint(
    run_root: Path,
    trace_enabled: bool,
    *,
    profile_label: str,
    scope_contract: tuple[re.Pattern[str], str, str],
    warmup_contract: tuple[re.Pattern[str], str, str],
    extra_argument_contracts: tuple[tuple[re.Pattern[str], str, str], ...] = (),
) -> tuple[Failure, ...]:
    """Require a completed two-layer TE capture followed by replay."""

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

    argument_contracts = (
        (
            _TRANSFORMER_ENGINE_ARGUMENT,
            "run.training.transformer_impl",
            "Megatron did not report transformer_impl=transformer_engine",
        ),
        (
            _TRANSFORMER_ENGINE_CUDA_GRAPH_ARGUMENT,
            "run.training.cuda_graph_impl",
            "Megatron did not report cuda_graph_impl=transformer_engine",
        ),
        scope_contract,
        warmup_contract,
        *extra_argument_contracts,
    )
    for pattern, code, message in argument_contracts:
        if pattern.search(log_text) is None:
            failures.append(Failure(code, message, str(launcher_log)))

    if "Rank 0: 2 graphable layers." not in log_text:
        failures.append(
            Failure(
                "run.training.cuda_graph_layers",
                "Transformer Engine did not discover both graphable layers",
                str(launcher_log),
            )
        )
    if "No graphable layers found" in log_text:
        failures.append(
            Failure(
                "run.training.cuda_graph_empty",
                "Transformer Engine reported no graphable layers",
                str(launcher_log),
            )
        )

    capture_start_count = log_text.count("Start CUDA Graphs capture...")
    if capture_start_count != 1:
        failures.append(
            Failure(
                "run.training.cuda_graph_capture_start",
                f"{profile_label} must start CUDA Graph capture exactly once; "
                f"observed {capture_start_count}",
                str(launcher_log),
            )
        )
    capture_done_count = log_text.count(
        "Time spent in CUDA Graphs capture on rank 0:"
    )
    if capture_done_count != 1:
        failures.append(
            Failure(
                "run.training.cuda_graph_capture_done",
                f"{profile_label} must finish CUDA Graph capture exactly once; "
                f"observed {capture_done_count}",
                str(launcher_log),
            )
        )

    deletion_matches = tuple(_CUDA_GRAPH_DELETION.finditer(log_text))
    deleted_graphs = (
        sum(
            int(match.group("explicit")) + int(match.group("implicit"))
            for match in deletion_matches
        )
        if deletion_matches
        else 0
    )
    if len(deletion_matches) != 1 or deleted_graphs != 2:
        failures.append(
            Failure(
                "run.training.cuda_graph_deletion",
                f"{profile_label} must delete exactly two captured layer graphs; "
                f"observed {deleted_graphs} across {len(deletion_matches)} log record(s)",
                str(launcher_log),
            )
        )

    completed_iterations = tuple(
        int(match.group("iteration"))
        for match in _TWO_ITERATION_PROGRESS.finditer(log_text)
    )
    if completed_iterations != (1, 2):
        failures.append(
            Failure(
                "run.training.iterations",
                f"{profile_label} must report iterations [1, 2]; "
                f"observed {list(completed_iterations)}",
                str(launcher_log),
            )
        )
    return tuple(failures)


def validate_two_iteration_te_full_cuda_graph_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require a completed TE whole-layer capture followed by replay."""

    return _validate_two_iteration_te_cuda_graph_checkpoint(
        run_root,
        trace_enabled,
        profile_label="TE whole-layer run",
        scope_contract=(
            _WHOLE_LAYER_CUDA_GRAPH_SCOPE_ARGUMENT,
            "run.training.cuda_graph_scope",
            "Megatron did not normalize the whole-layer CUDA Graph scope to []",
        ),
        warmup_contract=(
            _ONE_CUDA_GRAPH_WARMUP_STEP_ARGUMENT,
            "run.training.cuda_graph_warmup_steps",
            "Megatron did not report one CUDA Graph warmup step",
        ),
    )


def validate_two_iteration_te_attention_mlp_recompute_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require TE attention Graph replay with selective MLP recompute."""

    return _validate_two_iteration_te_cuda_graph_checkpoint(
        run_root,
        trace_enabled,
        profile_label="TE attention Graph with MLP recompute",
        scope_contract=(
            _ATTENTION_CUDA_GRAPH_SCOPE_ARGUMENT,
            "run.training.cuda_graph_scope",
            "Megatron did not report cuda_graph_scope=[attn]",
        ),
        warmup_contract=(
            _ZERO_CUDA_GRAPH_WARMUP_STEPS_ARGUMENT,
            "run.training.cuda_graph_warmup_steps",
            "Megatron did not report zero CUDA Graph warmup steps",
        ),
        extra_argument_contracts=(
            (
                _SELECTIVE_RECOMPUTE_ARGUMENT,
                "run.training.recompute_granularity",
                "Megatron did not report recompute_granularity=selective",
            ),
            (
                _MLP_RECOMPUTE_MODULE_ARGUMENT,
                "run.training.recompute_modules",
                "Megatron did not report recompute_modules=['mlp']",
            ),
            (
                _CUPTI_KERNEL_DISABLED_ARGUMENT,
                "run.training.trace_cupti_kernels",
                "Megatron did not report trace_cupti_kernels=off",
            ),
        ),
    )


def validate_two_iteration_local_layerwise_full_cuda_graph_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require local whole-layer graph creation followed by replay."""

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

    argument_contracts = (
        (
            _TRANSFORMER_ENGINE_ARGUMENT,
            "run.training.transformer_impl",
            "Megatron did not report transformer_impl=transformer_engine",
        ),
        (
            _LOCAL_CUDA_GRAPH_ARGUMENT,
            "run.training.cuda_graph_impl",
            "Megatron did not report cuda_graph_impl=local",
        ),
        (
            _WHOLE_LAYER_CUDA_GRAPH_SCOPE_ARGUMENT,
            "run.training.cuda_graph_scope",
            "Megatron did not normalize the whole-layer CUDA Graph scope to []",
        ),
        (
            _ONE_CUDA_GRAPH_WARMUP_STEP_ARGUMENT,
            "run.training.cuda_graph_warmup_steps",
            "Megatron did not report one CUDA Graph warmup step",
        ),
        (
            _OPTIMIZER_CUDA_GRAPH_DISABLED_ARGUMENT,
            "run.training.optimizer_cuda_graph",
            "Megatron did not report optimizer_cuda_graph=False",
        ),
    )
    for pattern, code, message in argument_contracts:
        if pattern.search(log_text) is None:
            failures.append(Failure(code, message, str(launcher_log)))

    capture_start_count = log_text.count("Creating 4 CUDA graphs")
    if capture_start_count != 1:
        failures.append(
            Failure(
                "run.training.cuda_graph_capture_start",
                "local whole-layer run must start creation of four layer graphs "
                f"exactly once; observed {capture_start_count}",
                str(launcher_log),
            )
        )

    built_counts = tuple(
        int(match.group("count"))
        for match in _LOCAL_CUDA_GRAPH_BUILD.finditer(log_text)
    )
    if built_counts != (4,):
        failures.append(
            Failure(
                "run.training.cuda_graph_capture_done",
                "local whole-layer run must finish exactly four layer graphs once; "
                f"observed {list(built_counts)}",
                str(launcher_log),
            )
        )

    full_iteration_markers = tuple(
        marker
        for marker in (
            "Capture CUDA graph for training!!!",
            "CUDA graph capture done for training!!!",
        )
        if marker in log_text
    )
    if full_iteration_markers:
        failures.append(
            Failure(
                "run.training.cuda_graph_owner",
                "local whole-layer run unexpectedly used the full-iteration wrapper; "
                f"observed {list(full_iteration_markers)!r}",
                str(launcher_log),
            )
        )

    completed_iterations = tuple(
        int(match.group("iteration"))
        for match in _TWO_ITERATION_PROGRESS.finditer(log_text)
    )
    if completed_iterations != (1, 2):
        failures.append(
            Failure(
                "run.training.iterations",
                "local whole-layer run must report iterations [1, 2]; "
                f"observed {list(completed_iterations)}",
                str(launcher_log),
            )
        )
    return tuple(failures)


def _validate_two_iteration_cuda_graph_kernel_checkpoint(
    run_root: Path,
    trace_enabled: bool,
    *,
    framework_contract: Callable[[Path, bool], tuple[Failure, ...]],
    profile_label: str,
) -> tuple[Failure, ...]:
    """Require Graph lifecycle and per-iteration CUDA kernel extraction."""

    failures = list(framework_contract(run_root, trace_enabled))
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

    if _CUPTI_KERNEL_ARGUMENT.search(log_text) is None:
        failures.append(
            Failure(
                "run.training.trace_cupti_kernels",
                "Megatron did not report trace_cupti_kernels=on",
                str(launcher_log),
            )
        )

    extractions = tuple(
        (int(match.group("iteration")), int(match.group("count")))
        for match in _CUDA_KERNEL_EXTRACTION.finditer(log_text)
    )
    if trace_enabled:
        if (
            tuple(iteration for iteration, _ in extractions) != (1, 2)
            or any(count <= 0 for _, count in extractions)
        ):
            failures.append(
                Failure(
                    "run.training.cuda_kernel_capture",
                    f"CUPTI-on {profile_label} run must extract a positive CUDA "
                    "kernel count exactly once for iterations [1, 2]; observed "
                    f"{list(extractions)}",
                    str(launcher_log),
                )
            )
    elif extractions:
        failures.append(
            Failure(
                "run.training.cuda_kernel_capture",
                f"trace-off {profile_label} run unexpectedly extracted CUDA "
                f"kernels; observed {list(extractions)}",
                str(launcher_log),
            )
        )

    return tuple(failures)


def validate_two_iteration_local_layerwise_cuda_graph_kernel_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require local Graph lifecycle and per-iteration CUDA kernel extraction."""

    return _validate_two_iteration_cuda_graph_kernel_checkpoint(
        run_root,
        trace_enabled,
        framework_contract=validate_two_iteration_local_layerwise_full_cuda_graph_checkpoint,
        profile_label="local whole-layer",
    )


def validate_two_iteration_te_full_cuda_graph_kernel_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require TE Graph lifecycle and per-iteration CUDA kernel extraction."""

    return _validate_two_iteration_cuda_graph_kernel_checkpoint(
        run_root,
        trace_enabled,
        framework_contract=validate_two_iteration_te_full_cuda_graph_checkpoint,
        profile_label="TE whole-layer",
    )


def validate_three_iteration_local_full_cuda_graph_checkpoint(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require local full-iteration capture followed by stable replay."""

    failures = list(validate_three_iteration_checkpoint(run_root, trace_enabled))
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

    argument_contracts = (
        (
            _TRANSFORMER_ENGINE_ARGUMENT,
            "run.training.transformer_impl",
            "Megatron did not report transformer_impl=transformer_engine",
        ),
        (
            _LOCAL_CUDA_GRAPH_ARGUMENT,
            "run.training.cuda_graph_impl",
            "Megatron did not report cuda_graph_impl=local",
        ),
        (
            _FULL_ITERATION_CUDA_GRAPH_SCOPE_ARGUMENT,
            "run.training.cuda_graph_scope",
            "Megatron did not report the local full_iteration Graph scope",
        ),
        (
            _ONE_CUDA_GRAPH_WARMUP_STEP_ARGUMENT,
            "run.training.cuda_graph_warmup_steps",
            "Megatron did not report one CUDA Graph warmup step",
        ),
        (
            _NAN_CHECK_DISABLED_ARGUMENT,
            "run.training.cuda_graph_nan_check",
            "Megatron did not disable the incompatible loss/gradient NaN check",
        ),
        (
            _OPTIMIZER_CUDA_GRAPH_DISABLED_ARGUMENT,
            "run.training.optimizer_cuda_graph",
            "Megatron did not report optimizer_cuda_graph=False",
        ),
    )
    for pattern, code, message in argument_contracts:
        if pattern.search(log_text) is None:
            failures.append(Failure(code, message, str(launcher_log)))

    capture_start_count = log_text.count("Capture CUDA graph for training!!!")
    if capture_start_count != 1:
        failures.append(
            Failure(
                "run.training.cuda_graph_capture_start",
                "local full-iteration run must start training Graph capture exactly once; "
                f"observed {capture_start_count}",
                str(launcher_log),
            )
        )
    capture_done_count = log_text.count(
        "CUDA graph capture done for training!!!"
    )
    if capture_done_count != 1:
        failures.append(
            Failure(
                "run.training.cuda_graph_capture_done",
                "local full-iteration run must finish training Graph capture exactly once; "
                f"observed {capture_done_count}",
                str(launcher_log),
            )
        )

    optimizer_capture_markers = tuple(
        marker
        for marker in (
            "Capture CUDA graph for optimizer!!!",
            "Optimizer CUDA graph capture done!!!",
        )
        if marker in log_text
    )
    if optimizer_capture_markers:
        failures.append(
            Failure(
                "run.training.optimizer_cuda_graph_capture",
                "local full-iteration run must keep optimizer capture disabled; "
                f"observed {list(optimizer_capture_markers)!r}",
                str(launcher_log),
            )
        )

    completed_iterations = tuple(
        int(match.group("iteration"))
        for match in _THREE_ITERATION_PROGRESS.finditer(log_text)
    )
    if completed_iterations != (1, 2, 3):
        failures.append(
            Failure(
                "run.training.iterations",
                "local full-iteration run must report iterations [1, 2, 3]; "
                f"observed {list(completed_iterations)}",
                str(launcher_log),
            )
        )
    return tuple(failures)


def _validate_flagcx_runtime(
    run_root: Path, *, expected_nranks: int
) -> tuple[list[Failure], str | None]:
    worker_logs = tuple(sorted((run_root / "logs").glob("host_*.output")))
    failures: list[Failure] = []
    if len(worker_logs) != 2:
        failures.append(
            Failure(
                "run.training.flagcx_worker_logs",
                "FlagCX validation requires exactly two host worker logs",
                str(run_root / "logs"),
            )
        )
        return failures, None

    worker_texts: list[str] = []
    for worker_log in worker_logs:
        try:
            worker_texts.append(worker_log.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as error:
            failures.append(
                Failure(
                    "run.training.flagcx_worker_log",
                    f"cannot read a FlagCX worker log: {error}",
                    str(worker_log),
                )
            )
    if len(worker_texts) != len(worker_logs):
        return failures, None

    worker_text = "\n".join(worker_texts)
    fatal_marker = _FLAGCX_FATAL_MARKER.search(worker_text)
    if fatal_marker is not None:
        failures.append(
            Failure(
                "run.training.flagcx_worker_failure",
                f"FlagCX worker logs contain {fatal_marker.group(0)!r}",
                str(run_root / "logs"),
            )
        )

    bootstraps = {
        (match.group("interface"), match.group("address"))
        for match in _FLAGCX_BOOTSTRAP.finditer(worker_text)
    }
    if len(bootstraps) != 2:
        failures.append(
            Failure(
                "run.training.flagcx_bootstrap",
                "FlagCX did not report two distinct host bootstrap addresses",
                str(run_root / "logs"),
            )
        )

    ranks = {
        int(match.group("rank"))
        for match in _FLAGCX_RANK.finditer(worker_text)
        if int(match.group("nranks")) == expected_nranks
    }
    if ranks != set(range(expected_nranks)):
        failures.append(
            Failure(
                f"run.training.flagcx_dp{expected_nranks}_group",
                f"FlagCX did not initialize ranks 0..{expected_nranks - 1} "
                f"of the {expected_nranks}-rank DP group",
                str(run_root / "logs"),
            )
        )
    return failures, worker_text


def validate_two_iteration_flagcx_checkpoint(
    run_root: Path, trace_enabled: bool, *, expected_nranks: int = 2
) -> tuple[Failure, ...]:
    """Require terminal state and both FlagCX worker logs."""

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
    runtime_failures, _ = _validate_flagcx_runtime(
        run_root, expected_nranks=expected_nranks
    )
    failures.extend(runtime_failures)
    return tuple(failures)


def validate_qwen3_v31_q0_artifacts(
    run_root: Path, trace_enabled: bool
) -> tuple[Failure, ...]:
    """Require one completed DP16 FlagCX run and its legacy checkpoint."""

    failures, worker_text = _validate_flagcx_runtime(run_root, expected_nranks=16)
    if worker_text is None:
        return tuple(failures)

    if _FLAGCX_BACKEND_ARGUMENT.search(worker_text) is None:
        failures.append(
            Failure(
                "run.training.distributed_backend",
                "Megatron did not report distributed_backend=flagcx",
                str(run_root / "logs"),
            )
        )
    trace_values = {match.group("enabled") for match in _TRACE_ARGUMENT.finditer(worker_text)}
    expected_trace = "True" if trace_enabled else "False"
    if trace_values != {expected_trace}:
        failures.append(
            Failure(
                "run.training.trace_mode",
                f"Megatron trace values are {sorted(trace_values)!r}, "
                f"expected only {expected_trace!r}",
                str(run_root / "logs"),
            )
        )

    planned = {
        int(match.group("iterations"))
        for match in _TRAINING_ITERATIONS.finditer(worker_text)
    }
    if len(planned) != 1:
        failures.append(
            Failure(
                "run.training.iterations",
                f"training logs report planned iteration counts {sorted(planned)!r}",
                str(run_root / "logs"),
            )
        )
        return tuple(failures)
    terminal_iteration = next(iter(planned))
    progress = {
        (int(match.group("iteration")), int(match.group("total")))
        for match in _TRAINING_PROGRESS.finditer(worker_text)
    }
    if (terminal_iteration, terminal_iteration) not in progress:
        failures.append(
            Failure(
                "run.training.iterations",
                f"training did not report terminal iteration {terminal_iteration}",
                str(run_root / "logs"),
            )
        )
    if "[after training is done]" not in worker_text:
        failures.append(
            Failure(
                "run.training.completion",
                "training logs do not report normal training completion",
                str(run_root / "logs"),
            )
        )

    checkpoint_root = run_root / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    try:
        tracked_iteration = int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError, ValueError):
        tracked_iteration = None
    if tracked_iteration != terminal_iteration:
        failures.append(
            Failure(
                "run.training.tracker",
                f"checkpoint tracker is {tracked_iteration!r}, "
                f"expected {terminal_iteration}",
                str(tracker),
            )
        )
    rank_root = (
        checkpoint_root
        / f"iter_{terminal_iteration:07d}"
        / "mp_rank_00"
    )
    for name in ("model_optim_rng.pt", "distrib_optim.pt"):
        checkpoint = rank_root / name
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            failures.append(
                Failure(
                    "run.training.checkpoint",
                    f"training iteration {terminal_iteration} has no non-empty {name}",
                    str(checkpoint),
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
