# Copyright (c) 2026, BAAI. All rights reserved.
"""StragglerDetector timing, cross-rank min/max reporting and the toggle controller."""

import itertools
import logging
from types import SimpleNamespace

import pytest
import torch

from megatron.core import utils
from megatron.core.utils import StragglerDetector, _ValueWithRank, socket


class ScriptedEvent:
    """Platform event whose timestamps come from a shared scripted clock (10 ms per tick)."""

    def __init__(self, clock):
        self.clock = clock
        self.time = None

    def record(self):
        self.time = next(self.clock) * 10.0

    def elapsed_time(self, other):
        return other.time - self.time


class RecordingSocket:
    def __init__(self, state, family, kind):
        self.calls = []
        state.sockets.append(self)

    def setsockopt(self, *args):
        pass

    def bind(self, address):
        self.calls.append(('bind', address))

    def listen(self, backlog):
        self.calls.append(('listen', backlog))


class RecordingThread:
    def __init__(self, state, target, args, name, daemon):
        self.record = [target.__name__, name, daemon, False]
        state.threads.append(self.record)

    def start(self):
        self.record[3] = True


class ScriptedConnection:
    def __init__(self):
        self.sent = []
        self.closed = False

    def recv(self, size):
        return b'GET / HTTP/1.0'

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        self.closed = True


class ScriptedListener:
    """Accepts one connection, then fails the way a closed socket does."""

    def __init__(self, connection):
        self.connection = connection
        self.accepts = 0

    def accept(self):
        self.accepts += 1
        if self.accepts == 1:
            return self.connection, None
        raise OSError('closed')


@pytest.fixture
def world(monkeypatch):
    """A fresh detector singleton on a scripted platform; collectives and sockets are recorded."""
    state = SimpleNamespace(
        available=True,
        backend='nccl',
        broadcasts=[],
        barriers=0,
        events_made=0,
        sockets=[],
        threads=[],
        socket_error=None,
    )
    clock = itertools.count(1)

    def make_event(enable_timing=False):
        state.events_made += 1
        return ScriptedEvent(clock)

    def make_socket(family, kind):
        if state.socket_error is not None:
            raise state.socket_error
        return RecordingSocket(state, family, kind)

    def gather_object(obj, object_gather_list=None, dst=0):
        # Rank r reports time + r, btime * (r + 1), temp + r and flops - r.
        if object_gather_list is not None:
            for rank in range(len(object_gather_list)):
                entry = dict(obj)
                entry.update(
                    rank=rank,
                    time=obj['time'] + rank,
                    btime=obj['btime'] * (rank + 1),
                    temp=obj['temp'] + rank,
                    flops=obj['flops'] - rank,
                )
                object_gather_list[rank] = entry

    def barrier(*args, **kwargs):
        state.barriers += 1

    perf_counter = itertools.count(0, 1_000_000)
    monkeypatch.setattr(
        utils,
        'cur_platform',
        SimpleNamespace(
            is_available=lambda: state.available,
            current_device=lambda: 'cpu',
            Event=make_event,
            temperature=lambda: 70,
            power_draw=lambda: 300,
            utilization=lambda: 90,
            clock_rate=lambda: 1400,
            synchronize=lambda: None,
        ),
    )
    monkeypatch.setattr(utils.time, 'perf_counter_ns', lambda: next(perf_counter))
    monkeypatch.setattr(torch.distributed, 'get_backend', lambda *args: state.backend)
    monkeypatch.setattr(
        torch.distributed,
        'broadcast',
        lambda tensor, src: state.broadcasts.append((tensor.item(), src)),
    )
    monkeypatch.setattr(torch.distributed, 'gather_object', gather_object)
    monkeypatch.setattr(torch.distributed, 'barrier', barrier)
    monkeypatch.setattr(
        utils,
        'socket',
        SimpleNamespace(
            socket=make_socket,
            AF_INET=socket.AF_INET,
            SOCK_STREAM=socket.SOCK_STREAM,
            SOL_SOCKET=socket.SOL_SOCKET,
            SO_REUSEADDR=socket.SO_REUSEADDR,
        ),
    )
    monkeypatch.setattr(
        utils,
        'threading',
        SimpleNamespace(
            Thread=lambda target, args, name, daemon: RecordingThread(
                state, target, args, name, daemon
            )
        ),
    )
    monkeypatch.delattr(StragglerDetector, '_instance', raising=False)
    monkeypatch.setattr(StragglerDetector, '_configured', False)
    state.new = StragglerDetector
    return state


@pytest.fixture
def messages(caplog):
    caplog.set_level(logging.INFO, logger=utils.logger.name)
    return lambda: [record.getMessage() for record in caplog.records]


def test_disabled_configuration_keeps_null_timers(world):
    detector = world.new()
    detector.configure(world=4, rank=1, prefill=3)
    assert detector.enabled is False
    assert detector.configured is True
    assert detector.start == detector.null_method
    assert detector.stop == detector.null_method
    assert detector.evt_q.qsize() == 3
    # NCCL takes the platform's current device as is.
    assert detector.dev == 'cpu'
    assert (detector.my_rank, detector.world_size, detector.mmcnt) == (1, 4, 1)
    with detector:
        pass
    assert detector.start_gemm_ev == []


def test_second_configure_is_ignored(world):
    detector = world.new()
    detector.configure(world=4, rank=1)
    detector.configure(world=9, rank=2)
    assert (detector.world_size, detector.my_rank) == (4, 1)


def test_disabled_detector_reports_nothing(world):
    detector = world.new()
    detector.configure(world=4, rank=1)
    assert detector.elapsed() == (0, 0, 0, 0, 0, 0)
    assert detector.reset() is None
    assert detector._min_max(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0) is None


def test_unavailable_platform_stays_off(world):
    world.available = False
    detector = world.new()
    detector.configure(world=4, rank=1, enabled=True)
    assert detector.enabled is False
    assert detector.world_size == 0
    assert detector.evt_q is None
    assert detector.configured is True


def test_gloo_backend_times_on_the_cpu(world):
    world.backend = 'gloo'
    detector = world.new()
    detector.configure(world=4, rank=1, mmcnt=0, prefill=2, enabled=True)
    assert detector.enabled is True
    # Any other backend falls back to an explicit CPU device object.
    assert isinstance(detector.dev, torch.device) and detector.dev.type == 'cpu'
    assert detector.mmcnt == 1
    assert detector.start == detector.start_method
    assert detector.stop == detector.stop_method


def test_kernel_and_data_intervals_are_timed_and_events_recycled(world):
    detector = world.new()
    detector.configure(world=4, rank=1, prefill=4, enabled=True)
    made = world.events_made
    with detector:
        pass
    with detector(bdata=True):
        pass
    with detector:
        pass
    assert world.events_made - made == 2
    assert detector.evt_q.qsize() == 0
    assert detector.start_gemm_tm == [0, 4_000_000]
    assert detector.stop_gemm_tm == [1_000_000, 5_000_000]
    assert detector.start_data_tm == [2_000_000]
    assert detector.stop_data_tm == [3_000_000]
    assert detector.elapsed() == (20.0, 10.0, 70, 300, 90, 1400)
    assert detector.evt_q.qsize() == 6
    assert detector.start_gemm_ev == []
    assert detector.start_data_tm == []
    assert detector.bdata is False


def test_stop_without_a_start_is_ignored(world):
    detector = world.new()
    detector.configure(world=4, rank=1, enabled=True)
    detector.stop_method()
    assert detector.stop_gemm_tm == []
    assert detector.stop_data_tm == []


def test_start_without_pooled_events_creates_them(world):
    detector = world.new()
    detector.configure(world=4, rank=1, prefill=0, enabled=True)
    made = world.events_made
    detector.start_method()
    assert world.events_made - made == 2
    assert len(detector.start_gemm_ev) == 1


def test_exception_in_the_context_is_logged_and_propagates(world, caplog):
    caplog.set_level(logging.WARNING, logger=utils.logger.name)
    detector = world.new()
    detector.configure(world=4, rank=1, enabled=True)
    with pytest.raises(RuntimeError, match='boom'):
        with detector:
            raise RuntimeError('boom')
    assert caplog.records[0].levelname == 'WARNING'
    assert caplog.records[0].getMessage().startswith('boom\n')
    assert detector.stop_gemm_tm == [1_000_000]


def test_report_without_flops_only_shares_the_collection_state(world):
    detector = world.new()
    detector.configure(world=4, rank=1, enabled=True)
    assert detector.report(total_flops=0.0, log_interval=1) is False
    assert world.broadcasts == [(False, 0)]


def test_other_ranks_contribute_without_logging(world, messages):
    detector = world.new()
    detector.configure(world=4, rank=1, prefill=4, enabled=True)
    with detector:
        pass
    assert detector.report(total_flops=8e12, log_interval=2) is False
    assert world.broadcasts == [(False, 0)]
    assert world.barriers == 1
    assert [message for message in messages() if 'MnRtt/Rnk' in message] == []


def test_rank_zero_logs_extremes_and_the_slowest_and_fastest_ranks(world, messages):
    detector = world.new()
    detector.configure(world=4, rank=0, mmcnt=2, port=1234, prefill=4, enabled=True)
    with detector:
        pass
    with detector(bdata=True):
        pass
    assert detector.report(total_flops=8e12, log_interval=2) is True
    assert world.barriers == 1
    logged = messages()
    summary = next(message for message in logged if 'MnRtt/Rnk' in message)
    assert '| MnRtt/Rnk: 5.00ms/0 | MxRtt/Rnk: 8.00ms/3 |' in summary
    assert '| MnTmp/Rnk: 70.00C/0 | MxTmp/Rnk: 73.00C/3 |' in summary
    assert '| MnDRtt/Rnk: 10.00ms/0 | MxDRtt/Rnk: 40.00ms/3 |' in summary
    assert summary.endswith('| MnEtpt/Rnk: 63.67TF/3 | MxEtpt/Rnk: 66.67TF/0')
    assert '^^^^ Bottom 2 Ranks with lowest  Etpt(TF): 63.67/3, 64.67/2,' in logged
    assert '^^^^ Top    2 Ranks with highest Etpt(TF): 65.67/1, 66.67/0,' in logged


def test_rank_zero_starts_the_toggle_controller(world):
    detector = world.new()
    detector.configure(world=4, rank=0, port=1234, enabled=True)
    assert detector.sock is world.sockets[0]
    assert world.sockets[0].calls == [('bind', ('0.0.0.0', 1234)), ('listen', 128)]
    assert world.threads == [['_handler', 'straggler', True, True]]


def test_controller_failure_is_logged_not_raised(world, messages):
    world.socket_error = OSError('port busy')
    detector = world.new()
    detector.configure(world=2, rank=0, enabled=True)
    assert detector.sock is None
    assert 'StragglerDetector cannot be controlled.. port busy' in messages()


def test_rank_zero_toggle_switches_collection_off_and_back_on(world, messages):
    detector = world.new()
    detector.configure(world=4, rank=0, enabled=True)
    detector.toggle = True
    detector._check_toggle()
    assert detector.enabled is False
    assert detector.start == detector.null_method
    assert detector.toggle is False
    assert world.broadcasts == [(True, 0)]
    assert 'Toggling StragglerDetector State OFF' in messages()
    detector.toggle = True
    detector._check_toggle()
    assert detector.enabled is True
    assert detector.start == detector.start_method
    assert 'Toggling StragglerDetector State ON' in messages()


def test_handler_acknowledges_a_request_and_stops_on_socket_error(world, messages):
    connection = ScriptedConnection()
    detector = world.new()
    detector._off = False
    detector.rank = 0
    detector.port = 77
    detector.sock = ScriptedListener(connection)
    assert detector._handler() is None
    assert detector.toggle is True
    assert connection.closed is True
    # Header separators are not pinned: utils.py:1739 builds them from a raw string (suspected
    # source bug, literal backslash pairs); only the status line, length and body are asserted.
    (response,) = connection.sent
    assert response.startswith(b'HTTP/1.0 200 OK')
    assert response.endswith(
        b'Content-length: 56\r\n\r\nWill turn StragglerDetector OFF at next logging interval'
    )
    logged = messages()
    assert 'Controller ready to recv commands on port 77. Current state ON' in logged
    assert 'Will turn StragglerDetector OFF at next logging interval' in logged
    assert 'Error in stragler handler.. closed' in logged


def test_handler_is_inactive_off_rank_zero(world):
    listener = ScriptedListener(ScriptedConnection())
    detector = world.new()
    detector.rank = 1
    detector.sock = listener
    assert detector._handler() is None
    assert listener.accepts == 0


def test_value_with_rank_compares_by_value_and_formats_with_unit():
    low = _ValueWithRank(1.5, 2, 'ms')
    high = _ValueWithRank(3.0, 1)
    assert low < high
    assert not low > high
    assert high > low
    assert low() == (1.5, 2, 'ms')
    assert str(low) == '1.50ms/2'
    assert str(high) == '3.00/1'
