# Copyright (c) 2026, BAAI. All rights reserved.

"""Async checkpoint callers: request freeze and finalize registration, temporal worker join and abort, the persistent worker loop, completion polling, and process QoS setup."""

import logging
import os
import queue as queue_module
from typing import NamedTuple

import pytest
import torch
from torch import multiprocessing as mp

from megatron.core.dist_checkpointing.strategies.async_utils import (
    AsyncCaller,
    AsyncCallsQueue,
    AsyncRequest,
    PersistentAsyncCaller,
    TemporalAsyncCaller,
    _set_process_qos,
)
from tests.unit_tests.test_utilities import Utils


def _temporal_worker(release_queue):
    """Forked worker that blocks until the test releases it."""
    release_queue.get()


def _persistent_worker(release_queue):
    """Spawned worker that blocks until the test releases it."""
    release_queue.get()


class _DelegatingCaller(AsyncCaller):
    """Concrete caller that exercises the abstract base contracts."""

    def schedule_async_call(self, async_req):
        return super().schedule_async_call(async_req)

    def is_current_async_call_done(self, blocking=False, no_dist=False):
        return super().is_current_async_call_done(blocking, no_dist)

    def close(self, abort=False):
        return super().close(abort)

    def __del__(self):
        pass


class _LegacyRequest(NamedTuple):
    """Old-style request with fewer fields than the current AsyncRequest."""

    async_fn: object
    async_fn_args: tuple
    finalize_fns: list


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def shared_queue():
    """Manager-backed queue: picklable, so it can travel to the already running persistent worker."""
    # A plain mp.Queue may only be shared by inheritance; sending it through the persistent
    # caller's request queue fails in the feeder thread and the worker never sees the request.
    with mp.Manager() as manager:
        yield manager.Queue()


def test_set_process_qos_is_noop_without_priorities():
    nice_before = os.nice(0)

    _set_process_qos(cpu_priority=None, io_priority=None)

    assert os.nice(0) == nice_before


def test_set_process_qos_deprioritizes_cpu_priority():
    _set_process_qos(cpu_priority=10, io_priority=None)

    assert os.nice(0) == 10


def test_set_process_qos_skips_already_deprioritized_cpu():
    current_nice = os.nice(0)

    _set_process_qos(cpu_priority=current_nice, io_priority=None)

    assert os.nice(0) == current_nice


def test_set_process_qos_keeps_cpu_priority_with_valid_io_class():
    nice_before = os.nice(0)

    _set_process_qos(cpu_priority=None, io_priority=2)

    assert os.nice(0) == nice_before


def test_set_process_qos_logs_io_priority_failure(caplog, monkeypatch, tmp_path):
    # util-linux ionice exits 0 even for an unknown class, so make the binary unavailable instead.
    monkeypatch.setenv('PATH', str(tmp_path))
    with caplog.at_level(logging.WARNING):
        _set_process_qos(cpu_priority=None, io_priority=2)

    assert 'Failed to set I/O priority' in caplog.text


def test_async_request_add_finalize_fn_appends_in_order():
    request = AsyncRequest(None, (), [])
    first_finalize = lambda: None  # noqa: E731
    second_finalize = lambda: None  # noqa: E731

    request.add_finalize_fn(first_finalize)
    request.add_finalize_fn(second_finalize)

    assert request.finalize_fns == [first_finalize, second_finalize]


def test_async_request_add_finalize_fn_rejects_frozen_request():
    frozen_request = AsyncRequest(None, (), []).freeze()

    with pytest.raises(
        RuntimeError, match='Cannot add finalization functions to a frozen AsyncRequest'
    ):
        frozen_request.add_finalize_fn(lambda: None)


def test_async_caller_abstract_contracts_raise_not_implemented():
    caller = _DelegatingCaller()

    with pytest.raises(NotImplementedError, match='This should be implemented'):
        caller.schedule_async_call(None)
    with pytest.raises(NotImplementedError, match='This should be implemented'):
        caller.is_current_async_call_done(blocking=False, no_dist=False)
    with pytest.raises(NotImplementedError) as caught:
        caller.close()
    assert str(caught.value) == ''
    with pytest.raises(NotImplementedError, match='This should be implemented'):
        AsyncCaller.__del__(caller)


def test_temporal_caller_close_joins_completed_worker(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    release_queue = mp.Queue()
    caller = TemporalAsyncCaller()

    caller.schedule_async_call(AsyncRequest(_temporal_worker, (release_queue,), []))
    assert caller.process.is_alive()

    release_queue.put(None)
    caller.close()

    assert caller.process is None
    assert caller.start_time is None
    release_queue.close()
    release_queue.join_thread()


def test_temporal_caller_close_aborts_blocked_worker(model_parallel, caplog):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    release_queue = mp.Queue()
    caller = TemporalAsyncCaller()

    caller.schedule_async_call(AsyncRequest(_temporal_worker, (release_queue,), []))
    with caplog.at_level(logging.WARNING):
        caller.close(abort=True)

    assert caller.process is None
    if torch.distributed.get_rank() == 0:
        assert 'Temporal worker aborted' in caplog.text
    release_queue.close()


def test_persistent_async_loop_executes_requests_and_stops_on_done(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    main_queue = queue_module.Queue()
    preload_queue = queue_module.Queue()
    comp_queue = queue_module.Queue()
    seen_calls = []

    def fake_async_fn(first_arg, staged_arg, last_arg):
        seen_calls.append((first_arg, staged_arg, last_arg))

    main_queue.put(
        AsyncRequest(
            fake_async_fn,
            (0, 'placeholder', 'last_arg'),
            [],
            preload_fn=lambda: 'staged_payload',
            call_idx=7,
        )
    )
    main_queue.put(AsyncRequest(None, (), [], call_idx=8))
    main_queue.put('DONE')
    preload_queue.put(1)

    PersistentAsyncCaller.async_loop(
        torch.distributed.get_rank(),
        main_queue,
        preload_queue,
        comp_queue,
        log_level=logging.getLogger().level,
        cpu_priority=None,
        io_priority=None,
    )

    assert seen_calls == [(0, 'staged_payload', 'last_arg')]
    assert comp_queue.get_nowait() == 7
    assert comp_queue.get_nowait() == 8
    assert comp_queue.empty()


def test_persistent_caller_reports_nothing_pending_as_done(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    caller = PersistentAsyncCaller()

    assert caller.is_current_async_call_done(blocking=False).item() is True
    assert caller.cur_item is None


def test_persistent_caller_warmup_schedule_and_finalize(model_parallel, shared_queue):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    AsyncCallsQueue.warmup_persistent_caller(
        torch.distributed.get_rank(), mp_mode='spawn', cpu_priority=None, io_priority=None
    )
    assert PersistentAsyncCaller._persistent_process.is_alive()

    queue = AsyncCallsQueue(persistent=True)
    release_queue = shared_queue
    call_idx = queue.schedule_async_request(AsyncRequest(_persistent_worker, (release_queue,), []))
    assert call_idx == 0
    assert queue.get_num_unfinalized_calls() == 1

    assert queue.maybe_finalize_async_calls(blocking=False) == []

    release_queue.put(None)
    assert queue.maybe_finalize_async_calls(blocking=True) == [0]
    assert queue.get_num_unfinalized_calls() == 0

    queue.close()


def _busy_worker(total_iterations):
    """Spawned worker that spends a bounded amount of time computing."""
    total = 0
    for i in range(total_iterations):
        total += i * i
    return total


def test_persistent_caller_blocking_done_polls_while_worker_is_busy(model_parallel):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    queue = AsyncCallsQueue(persistent=True)
    call_idx = queue.schedule_async_request(AsyncRequest(_busy_worker, (10_000_000,), []))

    assert queue.maybe_finalize_async_calls(blocking=True) == [call_idx]
    assert queue.get_num_unfinalized_calls() == 0

    queue.close()


def test_schedule_async_request_rebuilds_legacy_request_fields(model_parallel, shared_queue):
    model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    queue = AsyncCallsQueue(persistent=True)
    release_queue = shared_queue
    legacy_request = _LegacyRequest(_persistent_worker, (release_queue,), [])

    call_idx = queue.schedule_async_request(legacy_request)

    scheduled_request = queue.async_calls[-1].async_request
    assert isinstance(scheduled_request, AsyncRequest)
    assert len(scheduled_request._fields) == len(AsyncRequest._fields)
    assert scheduled_request.async_fn is _persistent_worker
    assert scheduled_request.call_idx == call_idx

    release_queue.put(None)
    assert queue.maybe_finalize_async_calls(blocking=True) == [call_idx]
    queue.close()


def test_maybe_finalize_returns_empty_without_active_calls():
    queue = AsyncCallsQueue(persistent=False)

    assert queue.get_num_unfinalized_calls() == 0
    assert queue.maybe_finalize_async_calls(blocking=True) == []
