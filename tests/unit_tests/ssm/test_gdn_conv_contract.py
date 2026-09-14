import importlib
from types import SimpleNamespace
from unittest.mock import patch
import inspect
import pytest

P = importlib.import_module('megatron.plugin.Ascend.ssm.causal_conv1d')
F = importlib.import_module('fla.modules.convolution')


def test_signature():
    assert list(inspect.signature(P.causal_conv1d).parameters) == list(
        inspect.signature(F.causal_conv1d).parameters
    )


@pytest.mark.parametrize(
    'options',
    [
        dict(backend='cuda'),
        dict(custom_option=3),
        dict(cu_seqlens=object()),
        dict(initial_state=object()),
        dict(output_final_state=True),
    ],
)
def test_unhandled_options_delegated(options):
    x = SimpleNamespace(device=SimpleNamespace(type='npu'), ndim=3)
    with patch.object(F, 'causal_conv1d', return_value=('sentinel', None)) as fn:
        assert P.causal_conv1d(x, weight=object(), **options)[0] == 'sentinel'
        for k, v in options.items():
            assert fn.call_args.kwargs[k] is v
