# Copyright (c) 2026, BAAI. All rights reserved.
"""TE MXFP8 LM-head output projection: config copying, extra-state shims and guards."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TELMHeadColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


def _make_config(**overrides):
    overrides.setdefault('params_dtype', torch.bfloat16)
    overrides.setdefault('bf16', True)
    overrides.setdefault('gradient_accumulation_fusion', False)
    overrides.setdefault('tensor_model_parallel_size', 2)
    # TransformerConfig only accepts delay_wgrad_compute together with MoE expert-parallel overlap;
    # the TE layers read the flag from the built config, so set it after validation.
    delay_wgrad_compute = overrides.pop('delay_wgrad_compute', False)
    config = TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4, **overrides)
    config.delay_wgrad_compute = delay_wgrad_compute
    return config


def _det_input(seq, batch, hidden):
    values = torch.arange(seq * batch * hidden, dtype=torch.float32)
    return (((values % 11) - 5) / 4.0).to(torch.bfloat16).reshape(seq, batch, hidden)


def _make_lm_head(model_parallel):
    model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    config = _make_config(
        fp8='e4m3', fp8_recipe='mxfp8', fp8_output_proj=True, delay_wgrad_compute=True
    )
    head = TELMHeadColumnParallelLinear(
        16,
        8,
        config=config,
        init_method=lambda w: None,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
        tp_group=parallel_state.get_tensor_model_parallel_group(),
    )
    return config, head


def test_te_lm_head_forces_delay_wgrad_off_on_copied_config(model_parallel):
    config, head = _make_lm_head(model_parallel)
    assert config.delay_wgrad_compute is True
    assert head.config.delay_wgrad_compute is False
    assert head.output_size_per_partition == 4
    assert head.gather_output is False
    assert head.disable_grad_reduce is False
    assert head.embedding_activation_buffer is None


def test_te_lm_head_extra_state_shims_round_trip(model_parallel):
    _, head = _make_lm_head(model_parallel)
    weight_before = head.weight.detach().clone()
    head.load_state_dict(head.state_dict())
    assert torch.equal(head.weight.detach(), weight_before)
    assert head.get_extra_state() is None
    head.set_extra_state({'marker': 1})
    assert head.get_extra_state() is None


def test_te_lm_head_load_state_dict_adds_missing_extra_state(model_parallel):
    _, head = _make_lm_head(model_parallel)
    weight_before = head.weight.detach().clone()
    head.load_state_dict({'weight': head.weight.detach().clone()})
    assert torch.equal(head.weight.detach(), weight_before)
    assert head.get_extra_state() is None


def test_te_lm_head_forward_rejects_runtime_weight(model_parallel):
    _, head = _make_lm_head(model_parallel)
    x = _det_input(6, 2, 16).cuda()
    with pytest.raises(RuntimeError, match='does not support runtime weight'):
        head(x, weight=torch.empty(2, 2, dtype=torch.bfloat16, device='cuda'))
