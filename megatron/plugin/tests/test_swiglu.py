# Copyright 2026 FlagOS Contributors
"""Ascend SwiGLU dispatch, autograd, dtype and tiled-width regressions."""

import builtins
import functools
import itertools
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def dispatch(monkeypatch):
    """Select NPU explicitly and restore all override registries and caches."""
    from megatron.core.fusions import fused_bias_swiglu as core
    from megatron.plugin import decorators
    from megatron.plugin.Ascend.fusions import fused_bias_swiglu as native

    snapshots = {
        name: {key: dict(value) for key, value in getattr(decorators, name).items()}
        for name in ('_plugin_registry', '_lazy_registry')
    }
    cached = dict(decorators._plugin_impl_cache)
    original = set(decorators._original_impl_cache)
    monkeypatch.setenv('MG_FL_PREFER', 'npu')
    decorators._plugin_impl_cache.clear()
    decorators._original_impl_cache.clear()

    @contextmanager
    def reference():
        saved = dict(decorators._plugin_impl_cache)
        saved_original = set(decorators._original_impl_cache)
        primitives = (
            core.swiglu.__wrapped__,
            core.swiglu_back.__wrapped__,
            core.weighted_swiglu_back.__wrapped__,
        )
        for func in primitives:
            decorators._plugin_impl_cache.pop(func, None)
            decorators._original_impl_cache.add(func)
        try:
            yield
        finally:
            decorators._plugin_impl_cache.clear()
            decorators._plugin_impl_cache.update(saved)
            decorators._original_impl_cache.clear()
            decorators._original_impl_cache.update(saved_original)

    try:
        yield SimpleNamespace(core=core, native=native, decorators=decorators, reference=reference)
    finally:
        for name, snapshot in snapshots.items():
            registry = getattr(decorators, name)
            registry.clear()
            registry.update(snapshot)
        decorators._plugin_impl_cache.clear()
        decorators._plugin_impl_cache.update(cached)
        decorators._original_impl_cache.clear()
        decorators._original_impl_cache.update(original)


@pytest.fixture
def npu_ops(monkeypatch):
    """Count real public operators in tests, binding each CI worker locally."""
    api = pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('NPU hardware is unavailable')
    for name in ('npu_swiglu', 'npu_swiglu_backward'):
        if not callable(getattr(api, name, None)):
            pytest.skip(f'Public operator {name} is unavailable')
    previous_device = torch.npu.current_device()
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', '0')) % torch.npu.device_count())
    counts = {'forward': 0, 'backward': 0}

    def counted(phase, fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            counts[phase] += 1
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(api, 'npu_swiglu', counted('forward', api.npu_swiglu))
    monkeypatch.setattr(api, 'npu_swiglu_backward', counted('backward', api.npu_swiglu_backward))
    try:
        yield SimpleNamespace(api=api, counts=counts, device=torch.device('npu'))
    finally:
        torch.npu.synchronize()
        torch.npu.set_device(previous_device)


def _sample(shape, dtype, device, seed=71):
    """Use a local CPU generator without changing global random state."""
    data = torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 0.5
    return data.to(device=device, dtype=dtype)


def _close(actual, expected, dtype, record_property, label):
    """Report actual errors separately from dtype-specific acceptance bounds."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    a, b = (actual.detach().float().cpu(), expected.detach().float().cpu())
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    maximum = (a - b).abs().max().item() if a.numel() else 0.0
    relative = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    rtol, atol, rel_limit = {
        torch.bfloat16: (0.03, 0.02, 0.01),
        torch.float16: (0.005, 0.003, 0.002),
        torch.float32: (2e-05, 2e-06, 1e-05),
        torch.float64: (1e-12, 1e-12, 1e-12),
    }[dtype]
    record_property(
        label,
        json.dumps(
            {
                'compute_dtype': str(dtype),
                'result_dtype': str(actual.dtype),
                'shape': list(actual.shape),
                'max_abs': maximum,
                'relative_l2': relative,
                'rtol': rtol,
                'atol': atol,
                'relative_l2_limit': rel_limit,
            }
        ),
    )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert relative <= rel_limit


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('rank', range(1, 9))
def test_native_primitives(dispatch, npu_ops, record_property, dtype, rank):
    """Cover the public 1D-8D contract with real Core forward/backward dispatch."""
    shape = (64,) if rank == 1 else (1,) * (rank - 2) + (3, 64)
    x = _sample(shape, dtype, npu_ops.device)
    g = _sample((*shape[:-1], shape[-1] // 2), dtype, npu_ops.device, seed=72)
    with dispatch.reference():
        with dispatch.reference():
            expected = dispatch.core.swiglu(x)
        with dispatch.reference():
            expected_grad = dispatch.core.swiglu_back(g, x)
    actual = dispatch.core.swiglu(x)
    actual_grad = dispatch.core.swiglu_back(g, x)
    _close(actual, expected, dtype, record_property, 'output')
    _close(actual_grad, expected_grad, dtype, record_property, 'input_grad')
    assert npu_ops.counts == {'forward': 1, 'backward': 1}
    cache = dispatch.decorators._plugin_impl_cache
    assert cache[dispatch.core.swiglu.__wrapped__] is dispatch.native.swiglu
    assert cache[dispatch.core.swiglu_back.__wrapped__] is dispatch.native.swiglu_back


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('row_multiple', [1, 3])
def test_native_alignment_boundary(dispatch, npu_ops, record_property, dtype, row_multiple):
    """Exercise minimum and odd-multiple 64-byte input rows on actual kernels."""
    width = 64 // torch.empty((), dtype=dtype).element_size() * row_multiple
    x = _sample((3, width), dtype, npu_ops.device)
    g = _sample((3, width // 2), dtype, npu_ops.device, seed=72)
    with dispatch.reference():
        with dispatch.reference():
            expected = dispatch.core.swiglu(x)
        with dispatch.reference():
            expected_grad = dispatch.core.swiglu_back(g, x)
    _close(dispatch.core.swiglu(x), expected, dtype, record_property, 'output')
    _close(dispatch.core.swiglu_back(g, x), expected_grad, dtype, record_property, 'input_grad')
    assert npu_ops.counts == {'forward': 1, 'backward': 1}


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('layout', ['sliced', 'expanded'])
def test_noncontiguous_native(dispatch, npu_ops, record_property, dtype, layout):
    """Preserve strided input and gradient values through layout adaptation."""
    if layout == 'sliced':
        x = _sample((3, 128), dtype, npu_ops.device)[:, ::2]
        g = _sample((3, 64), dtype, npu_ops.device, seed=72)[:, ::2]
    else:
        x = _sample((1, 64), dtype, npu_ops.device).expand(3, 64)
        g = _sample((1, 32), dtype, npu_ops.device, seed=72).expand(3, 32)
    assert not x.is_contiguous() and (not g.is_contiguous())
    x_before, g_before = (x.clone(), g.clone())
    with dispatch.reference():
        with dispatch.reference():
            expected = dispatch.core.swiglu(x)
        with dispatch.reference():
            expected_grad = dispatch.core.swiglu_back(g, x)
    _close(dispatch.core.swiglu(x), expected, dtype, record_property, 'output')
    _close(dispatch.core.swiglu_back(g, x), expected_grad, dtype, record_property, 'input_grad')
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    torch.testing.assert_close(g, g_before, rtol=0, atol=0)
    assert npu_ops.counts == {'forward': 1, 'backward': 1}


def _evaluate(core, x, g, kind, clamp=None, offload=False):
    """Execute the actual Core entry and return every differentiable input."""
    leaves = {'input_grad': x.detach().clone().requires_grad_()}
    inp = leaves['input_grad']
    if kind == 'direct':
        output = core.SwiGLUFunction.apply(inp, False, offload, clamp)
    elif kind in ('bias', 'bias_fp32'):
        bias_dtype = torch.float32 if kind == 'bias_fp32' else x.dtype
        leaves['bias_grad'] = _sample((x.shape[-1],), bias_dtype, x.device, 73).requires_grad_()
        output = core.bias_swiglu_impl(inp, leaves['bias_grad'], False, offload, clamp)
    elif kind.startswith('weighted'):
        weight_dtype = torch.float32 if kind == 'weighted_fp32' else x.dtype
        leaves['weights_grad'] = _sample((x.numel() // x.shape[-1], 1), weight_dtype, x.device, 74)
        leaves['weights_grad'].requires_grad_()
        output = core.weighted_bias_swiglu_impl(inp, None, leaves['weights_grad'], False, clamp)
    else:
        output = core.bias_swiglu_impl(inp, None, False, offload, clamp)
    output.backward(g.to(output.dtype))
    return (output.detach(), {label: leaf.grad.detach() for label, leaf in leaves.items()})


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('kind', ['plain', 'direct', 'bias', 'weighted', 'weighted_fp32'])
def test_core_autograd(dispatch, npu_ops, record_property, dtype, kind):
    """Cover SwiGLUFunction, bias reduction, and FP32 weighted type promotion."""
    x = _sample((4, 2, 64), dtype, npu_ops.device)
    g = _sample((4, 2, 32), dtype, npu_ops.device, 72)
    with dispatch.reference():
        with dispatch.reference():
            expected, expected_grads = _evaluate(dispatch.core, x, g, kind)
    actual, actual_grads = _evaluate(dispatch.core, x, g, kind)
    _close(actual, expected, dtype, record_property, 'output')
    for label, expected_grad in expected_grads.items():
        _close(actual_grads[label], expected_grad, dtype, record_property, label)
    forward = 1
    backward = int(not kind.startswith('weighted'))
    assert npu_ops.counts == {'forward': forward, 'backward': backward}


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('width', [65, 128, 512])
@pytest.mark.parametrize('strided', [False, True])
def test_weighted_backward(dispatch, npu_ops, record_property, launches, dtype, width, strided):
    from megatron.plugin.Ascend.fusions import weighted_swiglu_kernels as kernels

    factor = 2 if strided else 1
    x = _sample((7, width * 2 * factor), dtype, npu_ops.device)
    g = _sample((7, width * factor), dtype, npu_ops.device, 72)
    weights = _sample((7, 1), torch.float32, npu_ops.device, 73)
    if strided:
        x, g = (x[:, ::2], g[:, ::2])
    with dispatch.reference():
        expected = dispatch.core.weighted_swiglu_back(g, x, weights)
    actual = dispatch.core.weighted_swiglu_back(g, x, weights)
    assert len(launches) == 1
    for label, a, b in zip(('input_grad', 'weights_grad'), actual, expected):
        _close(a, b, dtype, record_property, label)
    cache = dispatch.decorators._plugin_impl_cache
    assert (
        cache[dispatch.core.weighted_swiglu_back.__wrapped__]
        is dispatch.native.weighted_swiglu_back
    )


def test_fp32_bias_promotion(dispatch, npu_ops, record_property):
    """Allow BF16 input plus FP32 bias without changing gradient dtypes."""
    x = _sample((4, 2, 64), torch.bfloat16, npu_ops.device)
    g = _sample((4, 2, 32), torch.float32, npu_ops.device, 72)
    with dispatch.reference():
        with dispatch.reference():
            expected, expected_grads = _evaluate(dispatch.core, x, g, 'bias_fp32')
    actual, actual_grads = _evaluate(dispatch.core, x, g, 'bias_fp32')
    _close(actual, expected, torch.float32, record_property, 'output')
    _close(
        actual_grads['input_grad'],
        expected_grads['input_grad'],
        torch.bfloat16,
        record_property,
        'input_grad',
    )
    _close(
        actual_grads['bias_grad'],
        expected_grads['bias_grad'],
        torch.float32,
        record_property,
        'bias_grad',
    )
    assert npu_ops.counts == {'forward': 1, 'backward': 1}


@pytest.mark.parametrize('kind', ['plain', 'bias', 'weighted_fp32'])
def test_clamp_stays_in_core(dispatch, npu_ops, record_property, kind):
    """Clamp branches preserve outputs and all gradients without native calls."""
    x = _sample((4, 2, 64), torch.bfloat16, npu_ops.device) * 4
    g = _sample((4, 2, 32), torch.bfloat16, npu_ops.device, 72)
    with dispatch.reference():
        with dispatch.reference():
            expected, expected_grads = _evaluate(dispatch.core, x, g, kind, clamp=0.75)
    actual, actual_grads = _evaluate(dispatch.core, x, g, kind, clamp=0.75)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for label, expected_grad in expected_grads.items():
        _close(actual_grads[label], expected_grad, torch.bfloat16, record_property, label)
    assert npu_ops.counts == {'forward': 0, 'backward': 0}


def test_target_bf16_shape(dispatch, npu_ops, record_property):
    """Check the training activation shape through the Core custom autograd."""
    x = _sample((16384, 8704), torch.bfloat16, npu_ops.device)
    g = _sample((16384, 4352), torch.bfloat16, npu_ops.device, 72)
    with dispatch.reference():
        with dispatch.reference():
            expected, expected_grads = _evaluate(dispatch.core, x, g, 'plain')
    actual, actual_grads = _evaluate(dispatch.core, x, g, 'plain')
    _close(actual, expected, torch.bfloat16, record_property, 'output')
    _close(
        actual_grads['input_grad'],
        expected_grads['input_grad'],
        torch.bfloat16,
        record_property,
        'input_grad',
    )
    assert npu_ops.counts == {'forward': 1, 'backward': 1}


@pytest.mark.parametrize('missing', ['npu_swiglu', 'npu_swiglu_backward'])
@pytest.mark.parametrize('absent', [True, False])
def test_missing_api_is_independent(dispatch, npu_ops, monkeypatch, missing, absent):
    """An absent or non-callable API falls back without disabling its sibling."""
    x = _sample((3, 64), torch.bfloat16, npu_ops.device)
    g = _sample((3, 32), torch.bfloat16, npu_ops.device, 72)
    with dispatch.reference():
        with dispatch.reference():
            expected = dispatch.core.swiglu(x)
        with dispatch.reference():
            expected_grad = dispatch.core.swiglu_back(g, x)
    if absent:
        monkeypatch.delattr(npu_ops.api, missing)
    else:
        monkeypatch.setattr(npu_ops.api, missing, None)
    actual = dispatch.core.swiglu(x)
    actual_grad = dispatch.core.swiglu_back(g, x)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.02)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0.03, atol=0.02)
    assert npu_ops.counts == {
        'forward': int(missing != 'npu_swiglu'),
        'backward': int(missing != 'npu_swiglu_backward'),
    }


@pytest.mark.parametrize('phase', ['forward', 'backward'])
def test_kernel_error_propagates(dispatch, npu_ops, monkeypatch, phase):
    """Mock only a kernel failure; it must propagate through the Core entry."""

    def fail(*args, **kwargs):
        raise RuntimeError('injected SwiGLU kernel failure')

    x = _sample((3, 64), torch.bfloat16, npu_ops.device)
    name = 'npu_swiglu' if phase == 'forward' else 'npu_swiglu_backward'
    monkeypatch.setattr(npu_ops.api, name, fail)
    with pytest.raises(RuntimeError, match='injected SwiGLU kernel failure'):
        if phase == 'forward':
            dispatch.core.swiglu(x)
        else:
            dispatch.core.swiglu_back(_sample((3, 32), x.dtype, x.device), x)


@pytest.mark.parametrize('case', ['empty', 'mixed_dtype', 'unaligned', 'broadcast_grad'])
def test_backward_fallback(dispatch, npu_ops, case):
    """Unsupported native contracts preserve the original broadcasting/dtype."""
    width = 18 if case == 'unaligned' else 64
    x = _sample((0 if case == 'empty' else 3, width), torch.bfloat16, npu_ops.device)
    g = _sample(
        (1 if case == 'broadcast_grad' else x.shape[0], width // 2),
        torch.float32 if case == 'mixed_dtype' else x.dtype,
        x.device,
    )
    expected = dispatch.core.swiglu_back.__wrapped__(g, x)
    torch.testing.assert_close(dispatch.core.swiglu_back(g, x), expected, rtol=0, atol=0)
    if case == 'empty':
        torch.testing.assert_close(dispatch.core.swiglu(x), dispatch.core.swiglu.__wrapped__(x))
    assert npu_ops.counts == {'forward': 0, 'backward': 0}


@pytest.mark.parametrize('case', ['float64', 'noncontiguous', 'empty', 'rank9', 'odd'])
def test_cpu_fallback(dispatch, case):
    """Non-NPU tensors preserve the default primitives, including unusual shapes."""
    shape = (0, 64) if case == 'empty' else (3, 3 if case == 'odd' else 64)
    if case == 'rank9':
        shape = (1,) * 8 + (64,)
    x = _sample(shape, torch.float64 if case == 'float64' else torch.float32, 'cpu')
    if case == 'noncontiguous':
        x = _sample((3, 128), x.dtype, x.device)[:, ::2]
    expected = dispatch.core.swiglu.__wrapped__(x)
    torch.testing.assert_close(dispatch.core.swiglu(x), expected, rtol=0, atol=0)
    g = torch.ones_like(expected)
    torch.testing.assert_close(
        dispatch.core.swiglu_back(g, x), dispatch.core.swiglu_back.__wrapped__(g, x), rtol=0, atol=0
    )


def test_vendor_fallback_cpu(dispatch, monkeypatch):
    """An explicit unmatched vendor keeps the original entry instead of NPU."""
    monkeypatch.setenv('MG_FL_PREFER', 'swiglu-test-missing-vendor')
    x = _sample((3, 64), torch.float32, 'cpu')
    torch.testing.assert_close(dispatch.core.swiglu(x), dispatch.core.swiglu.__wrapped__(x))
    assert dispatch.core.swiglu.__wrapped__ in dispatch.decorators._original_impl_cache


def test_saved_input_and_offload_cpu(dispatch):
    """Keep Core FP8 saved-input reconstruction and activation-offload tagging."""
    x = _sample((3, 64), torch.bfloat16, 'cpu').requires_grad_()
    g = _sample((3, 32), x.dtype, x.device, 72)
    saved = []

    def pack(tensor):
        saved.append((tensor.dtype, getattr(tensor, 'activation_offloading', False)))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = dispatch.core.SwiGLUFunction.apply(x, True, True, None)
        output.backward(g)
    assert saved == [(torch.float8_e4m3fn, True)]
    torch.testing.assert_close(output, dispatch.core.swiglu.__wrapped__(x), rtol=0, atol=0)
    restored = x.detach().to(torch.float8_e4m3fn).to(x.dtype)
    expected = dispatch.core.swiglu_back.__wrapped__(g, restored)
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)


def test_dispatch_fixture_restores_state(monkeypatch):
    """Exercise fixture teardown after a real lookup without leaking vendor state."""
    from megatron.plugin import decorators

    before_vendor = os.environ.get('MG_FL_PREFER')
    before_registry = {key: dict(value) for key, value in decorators._plugin_registry.items()}
    before_lazy = {key: dict(value) for key, value in decorators._lazy_registry.items()}
    before_cache = dict(decorators._plugin_impl_cache)
    before_original = set(decorators._original_impl_cache)
    with monkeypatch.context() as context:
        fixture = dispatch.__wrapped__(context)
        state = next(fixture)
        state.core.swiglu(_sample((3, 64), torch.float32, 'cpu'))
        with pytest.raises(StopIteration):
            next(fixture)
    assert os.environ.get('MG_FL_PREFER') == before_vendor
    assert decorators._plugin_registry == before_registry
    assert decorators._lazy_registry == before_lazy
    assert decorators._plugin_impl_cache == before_cache
    assert decorators._original_impl_cache == before_original


def test_npu_default_jit_policy(dispatch, npu_ops, monkeypatch):
    """Use the actual enabled NPU jit policy, without calling disable_jit_fuser."""
    from megatron.core import jit
    from megatron.plugin.platform import get_platform

    monkeypatch.delenv('TORCH_COMPILE_DISABLE', raising=False)
    monkeypatch.delenv('TORCHDYNAMO_DISABLE', raising=False)
    assert get_platform().device_name() == 'npu'
    assert jit.jit_fuser is jit.noop_decorator
    x = _sample((3, 64), torch.bfloat16, npu_ops.device).requires_grad_()
    dispatch.core.bias_swiglu_impl(x, None).backward(_sample((3, 32), x.dtype, x.device))
    assert npu_ops.counts == {'forward': 1, 'backward': 1}


def test_compiled_cpu_fallback(record_property):
    """Check real torch.compile graph capture with Core decorators in isolation."""
    code = "\nimport importlib,json,torch\n# Preserve PyTorch's real compiler before the NPU platform installs its no-op.\n# The child process isolates this compiler check from NPU tests and other suites.\nreal_compile=torch.compile\ncpu_cuda_available=torch.cuda.is_available\nfrom megatron.core import jit\n# Keep CPU compiler capability detection independent of transfer_to_npu.\ntorch.cuda.is_available=cpu_cuda_available\ngraphs=[]\ndef backend(graph,example_inputs):\n graphs.append(graph)\n return graph.forward\n# Pass the function directly: compiler decorator mode looks up the replaced\n# torch.compile again when applying its returned decorator.\njit.jit_fuser=lambda fn:real_compile(fn,backend=backend)\nfrom megatron.core.fusions import fused_bias_swiglu as core\ncore=importlib.reload(core)\nx=torch.linspace(-1,1,192).reshape(3,64).requires_grad_()\nb=torch.linspace(-0.2,0.2,64).requires_grad_()\ng=torch.linspace(-0.5,0.5,96).reshape(3,32)\nout=core.bias_swiglu_impl(x,b)\nout.backward(g)\nx_ref=x.detach().clone().requires_grad_();b_ref=b.detach().clone().requires_grad_()\ngate,value=(x_ref+b_ref).chunk(2,dim=-1)\nref=torch.nn.functional.silu(gate)*value\nref.backward(g)\nfor actual,expected in ((out,ref),(x.grad,x_ref.grad),(b.grad,b_ref.grad)):\n torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)\nassert graphs,'torch.compile must capture a graph, not silently run disabled'\nprint('SWIGLU_COMPILED_JSON='+json.dumps({'compiled_graphs':len(graphs),'backend':'eager'}))\n"
    env = dict(os.environ, MG_FL_PREFER='npu', PYTHONPATH=os.pathsep.join(sys.path))
    env.pop('TORCH_COMPILE_DISABLE', None)
    env.pop('TORCHDYNAMO_DISABLE', None)
    result = subprocess.run(
        [sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = next(
        (line for line in result.stdout.splitlines() if line.startswith('SWIGLU_COMPILED_JSON='))
    )
    metrics = json.loads(report.split('=', 1)[1])
    assert metrics['compiled_graphs'] > 0
    record_property('compiled_cpu_fallback', json.dumps(metrics))


DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@pytest.mark.parametrize('y_dtype,g_dtype,w_dtype', list(itertools.product(DTYPES, repeat=3)))
@pytest.mark.parametrize('width', [65, 128])
def test_weighted_dtype_rounding(
    dispatch, npu_ops, record_property, y_dtype, g_dtype, w_dtype, width
):
    y = _sample((7, width * 2), y_dtype, npu_ops.device)
    g = _sample((7, width), g_dtype, npu_ops.device, 72)
    w = _sample((7, 1), w_dtype, npu_ops.device, 73)
    with torch.no_grad():
        actual = dispatch.core.weighted_swiglu_back(g, y, w)
        with dispatch.reference():
            expected = dispatch.core.weighted_swiglu_back(g, y, w)
    for label, a, ref in zip(('dy', 'dw'), actual, expected):
        _close(a, ref, y_dtype, record_property, label)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('kind', ['strided', 'broadcast_g', 'broadcast_w', 'scalar_w'])
def test_weighted_shapes(dispatch, npu_ops, record_property, dtype, kind):
    y = _sample((3, 256), dtype, npu_ops.device)
    g = _sample((3, 128), dtype, npu_ops.device, 72)
    w = _sample((3, 1), torch.float32, npu_ops.device, 73)
    if kind == 'strided':
        y, g = (y[:, ::2], g[:, ::2])
    elif kind == 'broadcast_g':
        g = g[:1]
    elif kind == 'broadcast_w':
        w = w[:1]
    else:
        w = w[0, 0]
    with torch.no_grad():
        actual = dispatch.core.weighted_swiglu_back(g, y, w)
        with dispatch.reference():
            expected = dispatch.core.weighted_swiglu_back(g, y, w)
    for label, a, ref in zip(('dy', 'dw'), actual, expected):
        _close(a, ref, dtype, record_property, label)


@pytest.mark.parametrize('dtype', DTYPES)
def test_missing_backward_keeps_reference_rounding(
    dispatch, npu_ops, record_property, monkeypatch, dtype
):
    monkeypatch.setattr(npu_ops.api, 'npu_swiglu_backward', None)
    y = _sample((7, 256), dtype, npu_ops.device)
    g = _sample((7, 128), dtype, npu_ops.device, 72)
    w = _sample((7, 1), dtype, npu_ops.device, 73)
    with torch.no_grad():
        actual = dispatch.core.weighted_swiglu_back(g, y, w)
        with dispatch.reference():
            expected = dispatch.core.weighted_swiglu_back(g, y, w)
    for label, a, ref in zip(('dy', 'dw'), actual, expected):
        _close(a, ref, dtype, record_property, label)


@pytest.mark.parametrize('kind', ['empty', 'float64', 'higher_order', 'missing_forward'])
def test_weighted_fallback(dispatch, npu_ops, monkeypatch, kind):
    dtype = torch.float64 if kind == 'float64' else torch.float32
    rows = 0 if kind == 'empty' else 3
    y = _sample((rows, 128), dtype, npu_ops.device)
    g = _sample((rows, 64), dtype, npu_ops.device, 72)
    w = _sample((rows, 1), dtype, npu_ops.device, 73)
    if kind == 'higher_order':
        y.requires_grad_()
        g.requires_grad_()
        w.requires_grad_()
    if kind == 'missing_forward':
        monkeypatch.setattr(npu_ops.api, 'npu_swiglu', None)
    from megatron.plugin.Ascend.fusions import weighted_swiglu_kernels as kernels

    def fail(*args, **kwargs):
        raise AssertionError('The optimized kernel must not run')

    monkeypatch.setattr(kernels, 'weighted_swiglu_backward', fail)
    if kind == 'float64':
        for fn in (
            dispatch.core.weighted_swiglu_back.__wrapped__,
            dispatch.core.weighted_swiglu_back,
        ):
            with pytest.raises(RuntimeError, match='DT_DOUBLE'):
                fn(g, y, w)
        return
    actual = dispatch.core.weighted_swiglu_back(g, y, w)
    expected = dispatch.core.weighted_swiglu_back.__wrapped__(g, y, w)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert a.requires_grad == b.requires_grad


@pytest.mark.parametrize('missing', ['triton', 'unexpected_dependency', 'import_error'])
def test_dependency_failures(dispatch, npu_ops, monkeypatch, missing):
    y = _sample((3, 128), torch.float32, npu_ops.device)
    g = _sample((3, 64), y.dtype, npu_ops.device, 72)
    w = _sample((3, 1), y.dtype, npu_ops.device, 73)
    original_import = builtins.__import__

    def import_kernel(name, *args, **kwargs):
        if name == 'weighted_swiglu_kernels':
            if missing == 'import_error':
                raise ImportError('broken kernel module')
            raise ModuleNotFoundError('missing dependency', name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_kernel)
    if missing == 'triton':
        actual = dispatch.core.weighted_swiglu_back(g, y, w)
        expected = dispatch.core.weighted_swiglu_back.__wrapped__(g, y, w)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    else:
        with pytest.raises(ImportError):
            dispatch.core.weighted_swiglu_back(g, y, w)


def test_weighted_kernel_error_propagates(dispatch, npu_ops, monkeypatch):
    from megatron.plugin.Ascend.fusions import weighted_swiglu_kernels as kernels

    def fail(*args, **kwargs):
        raise RuntimeError('injected weighted kernel failure')

    monkeypatch.setattr(kernels, 'weighted_swiglu_backward', fail)
    y = _sample((3, 128), torch.float32, npu_ops.device)
    with pytest.raises(RuntimeError, match='injected weighted kernel failure'):
        dispatch.core.weighted_swiglu_back(y[:, :64], y, y[:, :1])


@pytest.fixture
def launches(monkeypatch):
    from megatron.plugin.Ascend.fusions import weighted_swiglu_kernels as kernels

    calls = []
    kernel = kernels._weighted_swiglu_backward

    class Counted:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                calls.append(grid)
                return kernel[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(kernels, '_weighted_swiglu_backward', Counted())
    return calls


@pytest.mark.parametrize('y_dtype,g_dtype,w_dtype', list(itertools.product(DTYPES, repeat=3)))
def test_width4352_dtype_matrix(
    dispatch, npu_ops, launches, record_property, y_dtype, g_dtype, w_dtype
):
    y = _sample((7, 8704), y_dtype, npu_ops.device)
    g = _sample((7, 4352), g_dtype, npu_ops.device, 72)
    w = _sample((7, 1), w_dtype, npu_ops.device, 73)
    record_property(
        'input_dtypes',
        json.dumps({'input': str(y.dtype), 'gradient': str(g.dtype), 'weights': str(w.dtype)}),
    )
    expected = dispatch.core.weighted_swiglu_back.__wrapped__(g, y, w)
    actual = dispatch.core.weighted_swiglu_back(g, y, w)
    for label, a, b in zip(('dy', 'dw'), actual, expected):
        _close(a, b, y_dtype, record_property, label)
    assert len(launches) == 1
    record_property('triton_launches', len(launches))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('layout', ['strided', 'broadcast'])
def test_width4352_layouts(dispatch, npu_ops, launches, record_property, dtype, layout):
    y = _sample((3, 17408 if layout == 'strided' else 8704), dtype, npu_ops.device)
    g = _sample((3, 8704 if layout == 'strided' else 4352), dtype, npu_ops.device, 72)
    w = _sample((3, 1), torch.float32, npu_ops.device, 73)
    if layout == 'strided':
        y, g = (y[:, ::2], g[:, ::2])
    else:
        g, w = (g[:1], w[:1])
    record_property(
        'input_dtypes',
        json.dumps({'input': str(y.dtype), 'gradient': str(g.dtype), 'weights': str(w.dtype)}),
    )
    expected = dispatch.core.weighted_swiglu_back.__wrapped__(g, y, w)
    actual = dispatch.core.weighted_swiglu_back(g, y, w)
    for label, a, b in zip(('dy', 'dw'), actual, expected):
        _close(a, b, dtype, record_property, label)
    assert len(launches) == 1
    record_property('triton_launches', len(launches))


def test_width4352_target_weighted_autograd(
    dispatch, npu_ops, launches, record_property, monkeypatch
):
    y = _sample((16384, 8704), torch.bfloat16, npu_ops.device)
    g = _sample((16384, 4352), y.dtype, npu_ops.device, 72)
    with monkeypatch.context() as context:
        context.setattr(
            dispatch.core, 'weighted_swiglu_back', dispatch.core.weighted_swiglu_back.__wrapped__
        )
        expected, refs = _evaluate(dispatch.core, y, g, 'weighted_fp32')
    actual, grads = _evaluate(dispatch.core, y, g, 'weighted_fp32')
    _close(actual, expected, y.dtype, record_property, 'output')
    for label, value in grads.items():
        _close(value, refs[label], y.dtype, record_property, label)
    assert len(launches) == 1
    record_property('triton_launches', len(launches))


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize(
    'width', (768, 1024, 1025, 2048, 4095, 4096, 4351, 4353, 8192, 8193, 17408, 32768)
)
def test_width_boundaries(dispatch, npu_ops, launches, record_property, dtype, width):
    y = _sample((7, width * 2), dtype, npu_ops.device)
    g = _sample((7, width), dtype, npu_ops.device, 72)
    w = _sample((7, 1), torch.float32, npu_ops.device, 73)
    record_property(
        'input_dtypes',
        json.dumps({'input': str(y.dtype), 'gradient': str(g.dtype), 'weights': str(w.dtype)}),
    )
    expected = dispatch.core.weighted_swiglu_back.__wrapped__(g, y, w)
    actual = dispatch.core.weighted_swiglu_back(g, y, w)
    for label, a, b in zip(('dy', 'dw'), actual, expected):
        _close(a, b, dtype, record_property, label)
    assert len(launches) == 1


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16, torch.float32))
def test_large_missing_backward_rounding(
    dispatch, npu_ops, launches, monkeypatch, record_property, dtype
):
    monkeypatch.setattr(npu_ops.api, 'npu_swiglu_backward', None)
    y = _sample((7, 8704), dtype, npu_ops.device)
    g = _sample((7, 4352), dtype, npu_ops.device, 72)
    w = _sample((7, 1), dtype, npu_ops.device, 73)
    record_property(
        'input_dtypes',
        json.dumps({'input': str(y.dtype), 'gradient': str(g.dtype), 'weights': str(w.dtype)}),
    )
    expected = dispatch.core.weighted_swiglu_back.__wrapped__(g, y, w)
    actual = dispatch.core.weighted_swiglu_back(g, y, w)
    for label, a, b in zip(('dy', 'dw'), actual, expected):
        _close(a, b, dtype, record_property, label)
    assert len(launches) == 1


@pytest.mark.parametrize('y_dtype,g_dtype,w_dtype', list(itertools.product(DTYPES, repeat=3)))
def test_weight_gradient_high_precision(
    dispatch, npu_ops, record_property, y_dtype, g_dtype, w_dtype
):
    """Check every row against FP64, with dot-product conditioning accounted for."""
    y = _sample((7, 8704), y_dtype, npu_ops.device, 177)
    g = _sample((7, 4352), g_dtype, npu_ops.device, 72)
    w = _sample((7, 1), w_dtype, npu_ops.device, 73)
    actual = dispatch.core.weighted_swiglu_back(g, y, w)[1].cpu().double()
    a, b = y.cpu().double().chunk(2, -1)
    upstream = g.to(w_dtype).cpu().double()
    output = torch.nn.functional.silu(a) * b
    expected = (output * upstream).sum(-1, keepdim=True)
    scale = output.norm(dim=-1, keepdim=True) * upstream.norm(dim=-1, keepdim=True)
    q = torch.promote_types(y_dtype, w_dtype)
    u32 = torch.finfo(torch.float32).eps / 2
    gamma = output.shape[-1] * u32 / (1 - output.shape[-1] * u32)
    # Forward rounding, promoted product, FP32 summation and final weight cast.
    # Cauchy-Schwarz bounds cancellation-sensitive dot products by row norms.
    bound = 2 * (
        torch.finfo(y_dtype).eps / 2
        + 8 * u32
        + torch.finfo(q).eps / 2
        + gamma
        + torch.finfo(w_dtype).eps / 2
    )
    normalized = (actual - expected).abs() / scale.clamp_min(1e-30)
    record_property(
        'input_dtypes',
        json.dumps({'input': str(y_dtype), 'gradient': str(g_dtype), 'weights': str(w_dtype)}),
    )
    record_property(
        'fp64_weight_gradient',
        json.dumps({'max_row_normalized_error': normalized.max().item(), 'bound': bound}),
    )
    assert torch.isfinite(actual).all()
    assert (normalized <= bound).all()
