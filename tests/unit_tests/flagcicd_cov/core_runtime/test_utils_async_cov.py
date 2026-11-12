# Copyright (c) 2026, BAAI. All rights reserved.
"""Asyncio loop management, async exception tracing, NVTX ranges and decorators."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

import megatron.core.utils as utils
from megatron.core.utils import (
    deprecate_args,
    deprecate_inference_params,
    deprecated,
    get_asyncio_loop,
    nvtx_range_pop,
    nvtx_range_push,
    trace_async_exceptions,
)


@pytest.fixture
def loop():
    """Provide a fresh event loop that is closed after the test."""
    event_loop = asyncio.new_event_loop()
    yield event_loop
    event_loop.close()


@pytest.fixture
def uncached_module_loop(monkeypatch):
    """Reset the module loop cache to its initial value; close whatever loop gets created."""
    monkeypatch.setattr(utils, '_ASYNC_IO_LOOP', None)
    yield
    created = utils._ASYNC_IO_LOOP
    if created is not None:
        created.close()
        asyncio.set_event_loop(None)


def test_asyncio_loop_returns_the_callers_running_loop(loop):
    async def probe():
        return get_asyncio_loop()

    assert loop.run_until_complete(probe()) is loop


def test_asyncio_loop_returns_an_explicitly_passed_loop(loop):
    assert get_asyncio_loop(loop) is loop


def test_asyncio_loop_creates_and_caches_a_new_loop_when_none_is_running(uncached_module_loop):
    created = get_asyncio_loop()
    assert isinstance(created, asyncio.AbstractEventLoop)
    assert not created.is_running()
    assert get_asyncio_loop() is created
    assert utils._ASYNC_IO_LOOP is created


def test_trace_async_exceptions_returns_the_coroutine_result(loop):
    @trace_async_exceptions
    async def double(value):
        return value * 2

    assert loop.run_until_complete(double(21)) == 42


def test_trace_async_exceptions_exits_the_process_when_a_coroutine_raises(loop, caplog):
    @trace_async_exceptions
    async def failing():
        raise ValueError('boom')

    with pytest.raises(SystemExit) as caught:
        loop.run_until_complete(failing())
    assert caught.value.code == 1
    assert 'Exception in async function failing: boom' in caplog.text


def test_trace_async_exceptions_streams_the_async_generator_items(loop, caplog):
    @trace_async_exceptions(verbose=True)
    async def numbers(count):
        for value in range(count):
            yield value

    async def collect():
        return [item async for item in numbers(3)]

    with caplog.at_level(logging.INFO, logger='megatron.core.utils'):
        assert loop.run_until_complete(collect()) == [0, 1, 2]
    assert 'lifetime cnt: 1' in caplog.text


def test_trace_async_exceptions_exits_the_process_when_an_async_generator_raises(loop):
    @trace_async_exceptions
    async def failing():
        yield 1
        raise ValueError('broken stream')

    async def collect():
        return [item async for item in failing()]

    with pytest.raises(SystemExit) as caught:
        loop.run_until_complete(collect())
    assert caught.value.code == 1


def test_trace_async_exceptions_rejects_non_async_callables():
    with pytest.raises(TypeError, match='must be used on async functions or generators'):

        @trace_async_exceptions
        def plain():
            return 1


def test_trace_async_exceptions_reports_task_statistics_when_verbose(loop, caplog):
    @trace_async_exceptions(verbose=True)
    async def probe():
        return 'ok'

    with caplog.at_level(logging.INFO, logger='megatron.core.utils'):
        assert loop.run_until_complete(probe()) == 'ok'
    assert 'completed in' in caplog.text
    assert 'lifetime cnt: 1' in caplog.text


@pytest.fixture
def nvtx_enabled(monkeypatch):
    """Enable NVTX ranges with a recording platform so push/pop pairs can be checked."""
    pushed, popped = [], []

    def record_pop():
        popped.append(None)

    monkeypatch.setattr(utils, '_nvtx_enabled', True)
    monkeypatch.setattr(utils, '_nvtx_range_messages', [])
    monkeypatch.setattr(utils, '_nvtx_range_msg_pool', {})
    monkeypatch.setattr(
        utils, 'cur_platform', SimpleNamespace(range_push=pushed.append, range_pop=record_pop)
    )
    return SimpleNamespace(pushed=pushed, popped=popped)


def test_nvtx_ranges_are_pushed_and_popped_in_order(nvtx_enabled):
    nvtx_range_push('forward')
    nvtx_range_push('backward')
    nvtx_range_pop('backward')
    nvtx_range_pop('forward')
    assert nvtx_enabled.pushed == ['forward', 'backward']
    assert nvtx_enabled.popped == [None, None]
    assert utils._nvtx_range_messages == []


def test_nvtx_range_pop_rejects_a_mismatched_message(nvtx_enabled):
    nvtx_range_push('forward')
    with pytest.raises(
        ValueError, match='Attempted to pop NVTX range from stack with msg=backward'
    ):
        nvtx_range_pop('backward')
    assert nvtx_enabled.popped == []


def test_nvtx_range_pop_rejects_popping_an_empty_stack(nvtx_enabled):
    with pytest.raises(RuntimeError, match='Attempted to pop NVTX range from empty stack'):
        nvtx_range_pop('forward')


def test_deprecated_decorator_includes_the_reason_in_the_warning():
    @deprecated(
        version='1.2.0', removal_version='2.0.0', alternative='new_function', reason='superseded'
    )
    def old_function(value):
        return value + 1

    with pytest.warns(DeprecationWarning) as caught:
        assert old_function(1) == 2
    message = str(caught[0].message)
    assert 'old_function is deprecated since version 1.2.0.' in message
    assert 'Use new_function instead.' in message
    assert 'Will be removed in version 2.0.0.' in message
    assert 'Reason: superseded' in message


def test_deprecate_args_blocks_the_listed_keyword_arguments():
    @deprecate_args('old_arg', message="Argument '{name}' is gone.")
    def accept(**kwargs):
        return kwargs

    with pytest.raises(TypeError, match="Argument 'old_arg' is gone."):
        accept(old_arg=1)


def test_deprecate_args_passes_the_remaining_arguments_through():
    @deprecate_args('old_arg')
    def accept(**kwargs):
        return kwargs

    assert accept(kept=2) == {'kept': 2}


def test_inference_params_are_returned_with_a_rename_warning():
    params = object()
    with pytest.warns(UserWarning, match='`inference_params` renamed to `inference_context`'):
        assert deprecate_inference_params(None, params) is params


def test_inference_context_is_returned_unchanged():
    context = object()
    assert deprecate_inference_params(context, None) is context
    assert deprecate_inference_params(None, None) is None
