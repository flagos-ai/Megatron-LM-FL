# Copyright (c) 2026, BAAI. All rights reserved.
"""TE-fused state_dict key rewrites performed by the ModelOpt GPT load pre-hook."""

import torch

from megatron.core.post_training.modelopt.gpt.state_dict_hooks import (
    mcore_gpt_load_te_state_dict_pre_hook,
)


def _run_pre_hook(state_dict):
    mcore_gpt_load_te_state_dict_pre_hook(state_dict, '', {}, True, [], [], [])
    return state_dict


def test_pre_hook_pops_the_modelopt_state_entry():
    weight = torch.ones(2)
    state_dict = {'modelopt_state': {'quant': 'fp8'}, 'decoder.weight': weight}

    _run_pre_hook(state_dict)

    assert set(state_dict) == {'decoder.weight'}
    assert state_dict['decoder.weight'] is weight


def test_pre_hook_rewrites_te_fused_layernorm_keys_to_unfused_names():
    qkv_norm = torch.ones(3)
    fc1_norm_bias = torch.zeros(3)
    mixer_norm = torch.ones(4)
    state_dict = {
        'decoder.layers.0.self_attention.linear_qkv.layer_norm_weight': qkv_norm,
        'decoder.layers.0.mlp.linear_fc1.layer_norm_bias': fc1_norm_bias,
        'mamba_layers.1.mixer.in_proj.layer_norm_weight': mixer_norm,
    }

    _run_pre_hook(state_dict)

    assert set(state_dict) == {
        'decoder.layers.0.input_layernorm.weight',
        'decoder.layers.0.pre_mlp_layernorm.bias',
        'mamba_layers.1.norm.weight',
    }
    assert state_dict['decoder.layers.0.input_layernorm.weight'] is qkv_norm
    assert state_dict['decoder.layers.0.pre_mlp_layernorm.bias'] is fc1_norm_bias
    assert state_dict['mamba_layers.1.norm.weight'] is mixer_norm


def test_pre_hook_leaves_unmatched_keys_untouched():
    weight = torch.ones(2, 2)
    state_dict = {'decoder.layers.0.self_attention.linear_proj.weight': weight}

    _run_pre_hook(state_dict)

    assert state_dict == {'decoder.layers.0.self_attention.linear_proj.weight': weight}
