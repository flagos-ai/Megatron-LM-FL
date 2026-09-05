# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Terminal artifact contracts for controlled training profiles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from tests.test_utils.runners.megalens_run_manifest import Failure


def _validate_iteration_checkpoint(run_root: Path, iteration: int) -> tuple[Failure, ...]:
    failures: list[Failure] = []
    checkpoint_root = run_root / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != str(iteration):
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


def validate_two_iteration_checkpoint(run_root: Path, _trace_enabled: bool) -> tuple[Failure, ...]:
    """Require the terminal torch_dist checkpoint from a two-iteration run."""

    return _validate_iteration_checkpoint(run_root, 2)


_TRANSFORMER_ENGINE_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*transformer_impl\s+\.+\s+transformer_engine\s*$", re.MULTILINE
)

_TRANSFORMER_ENGINE_CUDA_GRAPH_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*cuda_graph_impl\s+\.+\s+transformer_engine\s*$", re.MULTILINE
)

_WHOLE_LAYER_CUDA_GRAPH_SCOPE_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*cuda_graph_scope\s+\.+\s+\[\]\s*$", re.MULTILINE
)

_ONE_CUDA_GRAPH_WARMUP_STEP_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*cuda_graph_warmup_steps\s+\.+\s+1\s*$", re.MULTILINE
)

_TWO_ITERATION_PROGRESS = re.compile(r"iteration\s+(?P<iteration>[12])/\s*2\s*\|")

_CUDA_GRAPH_DELETION = re.compile(
    r"Rank 0: (?P<explicit>\d+) graphs deleted with explicit reset, "
    r"(?P<implicit>\d+) graphs deleted without explicit reset\."
)

_CUPTI_KERNEL_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*trace_cupti_kernels\s+\.+\s+on\s*$", re.MULTILINE
)

_TRACE_INTERVAL_TWO_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*trace_interval\s+\.+\s+2\s*$", re.MULTILINE
)

_CONTINUOUS_TRACE_TWO_ARGUMENT = re.compile(
    r"^\[[^]]+\]:\s*continuous_trace_iterations\s+\.+\s+2\s*$", re.MULTILINE
)

_CUDA_KERNEL_EXTRACTION = re.compile(
    r"\[trace\] extracted (?P<count>\d+) cuda kernel events at iter " r"(?P<iteration>[12])"
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
    capture_done_count = log_text.count("Time spent in CUDA Graphs capture on rank 0:")
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
        int(match.group("iteration")) for match in _TWO_ITERATION_PROGRESS.finditer(log_text)
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
        if tuple(iteration for iteration, _ in extractions) != (1, 2) or any(
            count <= 0 for _, count in extractions
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


def _validate_two_iteration_transformer_engine_cp_checkpoint(
    run_root: Path, trace_enabled: bool, *, context_parallel_size: int
) -> tuple[Failure, ...]:
    failures = list(validate_two_iteration_transformer_engine_checkpoint(run_root, trace_enabled))
    launcher_log = run_root / "launcher.log"
    try:
        log_text = launcher_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return tuple(failures)

    context_parallel_argument = re.compile(
        rf"^\[[^]]+\]:\s*context_parallel_size\s+\.+\s+" rf"{context_parallel_size}\s*$",
        re.MULTILINE,
    )
    if context_parallel_argument.search(log_text) is None:
        failures.append(
            Failure(
                "run.training.context_parallel_size",
                "Megatron did not report " f"context_parallel_size={context_parallel_size}",
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
