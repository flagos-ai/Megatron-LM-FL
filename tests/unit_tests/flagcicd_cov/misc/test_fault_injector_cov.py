# Copyright (c) 2026, BAAI. All rights reserved.
"""Fault injector ext-availability fallbacks, rank-zero fault plan setup, and fault sampling."""

import enum
import importlib
import logging
import random
import sys
import types

import pytest
import torch

import megatron.core.fault_injector as fault_injector


class _FakeFault(enum.Enum):
    """Stand-in for nvidia_resiliency_ext.shared_utils.inject_fault.Fault."""

    HANG = 0
    CRASH = 1


@pytest.fixture
def resiliency_ext_present(monkeypatch):
    """Reload fault_injector against a stubbed nvidia_resiliency_ext import boundary."""
    inject_fault = types.ModuleType('nvidia_resiliency_ext.shared_utils.inject_fault')
    inject_fault.Fault = _FakeFault
    inject_fault.dispatched = []
    inject_fault.cleared = []
    inject_fault.clear_workload_exception = lambda: inject_fault.cleared.append(True)
    inject_fault.dispatch_fault_injection = lambda **kwargs: inject_fault.dispatched.append(kwargs)
    inject_fault.maybe_raise_workload_exception = lambda: None
    top = types.ModuleType('nvidia_resiliency_ext')
    top.__path__ = []
    mid = types.ModuleType('nvidia_resiliency_ext.shared_utils')
    mid.__path__ = []
    monkeypatch.setitem(sys.modules, 'nvidia_resiliency_ext', top)
    monkeypatch.setitem(sys.modules, 'nvidia_resiliency_ext.shared_utils', mid)
    monkeypatch.setitem(
        sys.modules, 'nvidia_resiliency_ext.shared_utils.inject_fault', inject_fault
    )
    importlib.reload(fault_injector)
    yield inject_fault
    monkeypatch.undo()
    importlib.reload(fault_injector)
    stale_names = (
        ('Fault', inject_fault.Fault),
        ('clear_workload_exception', inject_fault.clear_workload_exception),
        ('dispatch_fault_injection', inject_fault.dispatch_fault_injection),
        ('maybe_raise_workload_exception', inject_fault.maybe_raise_workload_exception),
    )
    for name, fake in stale_names:
        if fault_injector.__dict__.get(name) is fake:
            del fault_injector.__dict__[name]


@pytest.fixture
def resiliency_ext_absent(monkeypatch):
    """Reload fault_injector with the nvidia_resiliency_ext import blocked."""
    monkeypatch.setitem(sys.modules, 'nvidia_resiliency_ext', None)
    importlib.reload(fault_injector)
    yield
    monkeypatch.undo()
    importlib.reload(fault_injector)


@pytest.fixture
def rank_zero_fault_setup(resiliency_ext_present, monkeypatch):
    """Run setup_fault_injection on rank zero with a fixed seed so the module seeds its rng."""
    ext_stub = resiliency_ext_present
    records = types.SimpleNamespace(
        broadcasts=[], rank=0, dispatched=ext_stub.dispatched, cleared=ext_stub.cleared
    )

    def _broadcast(tensor, src=0):
        if records.broadcasts:
            tensor.copy_(records.broadcasts[-1])
        records.broadcasts.append(tensor.clone())

    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: records.rank)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 8)
    monkeypatch.setattr(torch.distributed, 'broadcast', _broadcast)
    monkeypatch.setattr(fault_injector.torch.cuda, 'current_device', lambda: 0)
    real_full = fault_injector.torch.full

    def _cpu_full(*args, **kwargs):
        device = kwargs.get('device')
        if isinstance(device, torch.device) and device.type == 'cuda':
            kwargs['device'] = torch.device('cpu')
        return real_full(*args, **kwargs)

    monkeypatch.setattr(fault_injector.torch, 'full', _cpu_full)
    config = fault_injector.FaultInjectorConfig(
        fault_injector_ranks='3',
        fault_injector_fault_types='hang,crash',
        fault_injector_fault_delay=5.0,
        fault_injector_seed=0,
    )
    fault_injector.setup_fault_injection(config)
    records.config = config
    yield records
    fault_injector.rng = None


def _expected_setup_fault():
    """Recompute the fault drawn by the module's seeded rng during the rank-zero setup."""
    return random.Random(0).choices([_FakeFault.HANG, _FakeFault.CRASH], [0.5, 0.5], k=1)[0]


def test_successful_ext_import_marks_the_module_available(resiliency_ext_present):
    assert fault_injector.has_nvidia_resiliency_ext is True
    assert fault_injector.Fault is _FakeFault


def test_workload_exception_fallback_raises_when_ext_is_missing(resiliency_ext_absent):
    with pytest.raises(
        ModuleNotFoundError, match="nvidia_resiliency_ext is required for fault injection"
    ):
        fault_injector.maybe_raise_workload_exception()


def test_fault_sampling_requires_the_ext_to_be_installed(resiliency_ext_absent):
    config = fault_injector.FaultInjectorConfig(fault_injector_fault_types='hang')

    with pytest.raises(
        ModuleNotFoundError, match="nvidia_resiliency_ext is required for fault injection"
    ):
        fault_injector.get_fault(config)


def test_get_fault_ranks_parses_a_single_explicit_rank(rank_zero_fault_setup):
    config = fault_injector.FaultInjectorConfig(fault_injector_ranks='7')

    assert fault_injector.get_fault_ranks(config) == [7]


def test_setup_fault_injection_broadcasts_the_rank_zero_fault_plan(rank_zero_fault_setup):
    expected_fault = _expected_setup_fault()
    plan = rank_zero_fault_setup.broadcasts[0]

    assert plan.dtype is torch.float64
    assert plan.shape == torch.Size([9])
    assert plan[3].item() == float(expected_fault.value)
    assert plan[8].item() == 5.0
    assert torch.isnan(plan[:3]).all()
    assert torch.isnan(plan[4:8]).all()
    assert rank_zero_fault_setup.cleared == [True]
    assert rank_zero_fault_setup.dispatched == []


def test_setup_fault_injection_dispatches_the_planned_fault_on_the_target_rank(
    rank_zero_fault_setup, caplog
):
    expected_fault = _expected_setup_fault()
    caplog.set_level(logging.WARNING, logger='megatron.core.fault_injector')
    rank_zero_fault_setup.rank = 3

    fault_injector.setup_fault_injection(rank_zero_fault_setup.config)

    assert rank_zero_fault_setup.dispatched == [
        {'fault': expected_fault, 'delay': 5.0, 'callback': None}
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any(f'Rank 3 will inject fault {expected_fault.name}' in message for message in messages)


@pytest.mark.parametrize(
    ('fault_type_name', 'expected_fault'),
    [('hang', _FakeFault.HANG), ('crash', _FakeFault.CRASH)],
    ids=['hang', 'crash'],
)
def test_get_fault_resolves_a_single_fault_type_name(
    rank_zero_fault_setup, fault_type_name, expected_fault
):
    config = fault_injector.FaultInjectorConfig(fault_injector_fault_types=fault_type_name)

    fault = fault_injector.get_fault(config)

    assert fault is expected_fault


def test_get_fault_weights_fault_types_uniformly_when_probabilities_are_not_configured(
    rank_zero_fault_setup,
):
    config = fault_injector.FaultInjectorConfig(fault_injector_fault_types='hang,crash')
    reference = random.Random(0)
    reference.choices([_FakeFault.HANG, _FakeFault.CRASH], [0.5, 0.5], k=1)
    expected = [
        reference.choices([_FakeFault.HANG, _FakeFault.CRASH], [0.5, 0.5], k=1)[0]
        for _ in range(16)
    ]

    sampled = [fault_injector.get_fault(config) for _ in range(16)]

    assert sampled == expected


def test_get_fault_accepts_a_lone_probability_for_a_single_fault_type(rank_zero_fault_setup):
    config = fault_injector.FaultInjectorConfig(
        fault_injector_fault_types='crash', fault_injector_fault_probabilities='0.25'
    )

    fault = fault_injector.get_fault(config)

    assert fault is _FakeFault.CRASH
