# Adapted from flagos-ai/Megatron-LM-FL PR #153, head 9d4788a41d91fe8a13a2e1563eb39ea43184b674.
# See the repository LICENSE for license information.

import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

P = importlib.import_module('megatron.plugin.Ascend.ssm.causal_conv1d')
F = importlib.import_module('fla.modules.convolution')


def test_signature():
    adapter = inspect.signature(P.causal_conv1d).parameters
    upstream = inspect.signature(F.causal_conv1d).parameters
    dense_parameters = (
        'x',
        'weight',
        'bias',
        'residual',
        'initial_state',
        'output_final_state',
        'activation',
        'backend',
        'cu_seqlens',
    )
    assert list(adapter)[: len(dense_parameters)] == list(dense_parameters)
    for name in dense_parameters:
        assert adapter[name].kind == upstream[name].kind
        assert adapter[name].default == upstream[name].default
    assert adapter['kwargs'].kind == inspect.Parameter.VAR_KEYWORD
    # FLA 0.5.2 added keyword metadata. The adapter deliberately forwards it.
    for name in upstream:
        if name not in adapter:
            inspect.signature(P.causal_conv1d).bind(x=object(), **{name: object()})


@pytest.mark.parametrize(
    'options',
    [
        dict(backend='cuda'),
        dict(custom_option=3),
        dict(cu_seqlens=object()),
        dict(initial_state=object()),
        dict(output_final_state=True),
        dict(cu_seqlens_cpu=object()),
        dict(chunk_indices=object()),
        dict(cp_context=object()),
    ],
)
def test_unhandled_options_delegated(options):
    x = SimpleNamespace(device=SimpleNamespace(type='npu'), ndim=3)
    with patch.object(F, 'causal_conv1d', return_value=('sentinel', None)) as fn:
        assert P.causal_conv1d(x, weight=object(), **options)[0] == 'sentinel'
        for k, v in options.items():
            assert fn.call_args.kwargs[k] is v
