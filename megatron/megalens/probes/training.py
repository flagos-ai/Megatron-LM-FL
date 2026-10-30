# Copyright (c) 2026, MegaLens Authors. All rights reserved.
"""Implicit training lifecycle calls without importing the tracing runtime."""

import functools
import logging
import warnings
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable, ParamSpec, TypeVar

import torch

_P = ParamSpec('_P')
_R = TypeVar('_R')


def owns_runtime(shutdown: Callable[..., None]) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Close the runtime after pretrain while preserving its result or failure."""

    def decorate(pretrain_func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(pretrain_func)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                result = pretrain_func(*args, **kwargs)
            except BaseException:
                try:
                    shutdown(graceful=False)
                except BaseException:
                    logging.getLogger(pretrain_func.__module__).exception(
                        "MegaLens non-graceful shutdown failed while handling a pretrain failure"
                    )
                raise
            shutdown(graceful=True)
            return result

        return wrapped

    return decorate


def iteration_scope(
    args: Any, iteration: int, get_runtime: Callable[[], Any], *, skip_iteration: bool
) -> AbstractContextManager[None]:
    """Prepare the existing iteration envelope and mode-1 WORLD barrier."""
    if not getattr(args, 'trace', False):
        return nullcontext()

    runtime = get_runtime()
    if skip_iteration:
        runtime.tracer.close_trace_window()
        return nullcontext()
    if not runtime.tracer.is_mode0():
        # Preserve source mode-1 alignment before entering the iteration scope.
        torch.distributed.barrier()
    return runtime.iteration(
        iteration + 1, enable_hw_monitor=getattr(args, 'hardware_monitor', False)
    )


def _add_megalens_args(parser):
    group = parser.add_argument_group(title='megalens')
    group.add_argument(
        '--trace', action='store_true', help='Enable rank-local MegaLens training traces.'
    )
    group.add_argument(
        '--trace-mode',
        type=int,
        default=1,
        choices=[0, 1],
        help='0 records the lightweight sentinel; 1 records framework scopes.',
    )
    group.add_argument(
        '--hardware-monitor',
        action='store_true',
        help='Sample hardware counters during active MegaLens iterations.',
    )
    group.add_argument('--trace-dir', type=str, default='trace_output')
    group.add_argument('--trace-interval', type=int, default=1000)
    group.add_argument('--continuous-trace-iterations', type=int, default=1)
    group.add_argument('--trace-granularity', choices=['base', 'full'], default='full')
    group.add_argument('--sentinel-hw-sample-ms', type=float, default=100.0)
    group.add_argument('--sentinel-flush-interval', type=int, default=100)
    group.add_argument(
        '--trace-gather-to-rank0',
        action='store_true',
        default=False,
        help='Use the legacy collective single-writer trace path.',
    )
    group.add_argument(
        '--trace-cupti-kernels',
        choices=['auto', 'on', 'off'],
        default='off',
        help=(
            'Optional CUDA kernel-level tracing via torch.profiler. "auto" enables '
            'kernel capture for mode-1 full traces, "on" forces it for mode-1 '
            'traces, and "off" disables it.'
        ),
    )
    return parser


def _validate_megalens_args(args):
    if args.hardware_monitor and not args.trace:
        raise ValueError('--hardware-monitor requires --trace')
    if not args.trace:
        return
    if args.trace_interval <= 0:
        raise ValueError('--trace-interval must be greater than zero')
    if not 0 < args.continuous_trace_iterations <= args.trace_interval:
        raise ValueError('--continuous-trace-iterations must be in [1, trace-interval]')
    if args.sentinel_hw_sample_ms <= 0:
        raise ValueError('--sentinel-hw-sample-ms must be greater than zero')
    if args.sentinel_flush_interval <= 0:
        raise ValueError('--sentinel-flush-interval must be greater than zero')
    kernel_capture_requested = args.trace_mode != 0 and args.trace_cupti_kernels != 'off'
    if args.cuda_graph_impl != 'none':
        warnings.warn(
            'MegaLens tracing with CUDA Graphs records eager and graph-external '
            'framework scopes; Python scopes inside managed capture/replay are '
            'unavailable.',
            RuntimeWarning,
            stacklevel=2,
        )
    if kernel_capture_requested and args.profile and args.use_pytorch_profiler:
        raise ValueError('MegaLens kernel capture cannot nest the Megatron PyTorch profiler')
