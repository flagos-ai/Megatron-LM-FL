# Copyright (c) 2026, BAAI. All rights reserved.
"""_BaseDataParallel delegates forward and checkpoint state to the wrapped module; sync hooks are no-ops."""

import torch

from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.transformer.transformer_config import TransformerConfig


class _CheckpointableModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return {
            f'{prefix}ckpt.{key}': value
            for key, value in self.state_dict(keep_vars=keep_vars).items()
        }

    def forward(self, x):
        return self.linear(x)


def _make_base_data_parallel():
    config = TransformerConfig(num_attention_heads=1, num_layers=1)
    module = _CheckpointableModule(4, 3)
    return _BaseDataParallel(config=config, module=module)


def test_forward_calls_wrapped_module_forward():
    base = _make_base_data_parallel()
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    torch.testing.assert_close(base(x), base.module(x), rtol=0, atol=0)


def test_no_sync_runs_block_body_without_touching_module():
    entered = []
    with _make_base_data_parallel().no_sync():
        entered.append(1)
    assert entered == [1]


def test_grad_and_param_sync_stubs_leave_state_unchanged():
    base = _make_base_data_parallel()
    weight_before = base.module.linear.weight.detach().clone()
    base.start_grad_sync()
    base.scale_gradients(0.5)
    base.finish_grad_sync()
    base.zero_grad_buffer()
    base.broadcast_params()
    torch.testing.assert_close(base.module.linear.weight.detach(), weight_before, rtol=0, atol=0)


def test_state_dict_returns_module_state_dict_with_prefix_and_keep_vars():
    base = _make_base_data_parallel()
    state = base.state_dict(prefix='model.', keep_vars=True)
    assert sorted(state.keys()) == ['model.linear.bias', 'model.linear.weight']
    assert state['model.linear.weight'] is base.module.linear.weight


def test_state_dict_for_save_checkpoint_delegates_to_module():
    base = _make_base_data_parallel()
    state = base.state_dict_for_save_checkpoint(prefix='layer0.', keep_vars=False)
    assert sorted(state.keys()) == ['layer0.ckpt.linear.bias', 'layer0.ckpt.linear.weight']
    torch.testing.assert_close(
        state['layer0.ckpt.linear.weight'], base.module.linear.weight.detach(), rtol=0, atol=0
    )


def test_load_state_dict_copies_values_into_wrapped_module():
    base = _make_base_data_parallel()
    new_state = {
        'linear.weight': torch.full_like(base.module.linear.weight, 2.5),
        'linear.bias': torch.full_like(base.module.linear.bias, -1.0),
    }
    base.load_state_dict(new_state)
    torch.testing.assert_close(
        base.module.linear.weight.detach(), new_state['linear.weight'], rtol=0, atol=0
    )
    torch.testing.assert_close(
        base.module.linear.bias.detach(), new_state['linear.bias'], rtol=0, atol=0
    )
