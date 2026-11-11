# Copyright (c) 2026, BAAI. All rights reserved.
"""Model chunk schedule plans: stored inputs, 1f1b run callbacks and state release, hyper-connection and MoE-MTP forwards matching plain forward, and FSDP reshard hooks."""

import pytest
import torch

from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from megatron.core.models.gpt.fine_grained_callables import build_layer_callables
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, get_current_device


@pytest.fixture
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


@pytest.fixture
def nvte_auto_env(monkeypatch):
    monkeypatch.setenv('NVTE_FLASH_ATTN', '1')
    monkeypatch.setenv('NVTE_FUSED_ATTN', '1')
    monkeypatch.setenv('NVTE_UNFUSED_ATTN', '1')


def _config(**overrides):
    values = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _input_data(device):
    return {
        'input_ids': torch.randint(0, 128, (2, 4), device=device),
        'position_ids': torch.arange(4, device=device).unsqueeze(0).expand(2, 4),
        'attention_mask': torch.ones(2, 1, 4, 4, dtype=torch.bool, device=device),
        'labels': torch.randint(0, 128, (2, 4), device=device),
    }


def test_build_schedule_plan_stores_inputs_and_builds_layer_nodes(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    model = GPTModel(
        config=_config(),
        transformer_layer_spec=get_gpt_decoder_block_spec(_config(), use_transformer_engine=False),
        vocab_size=128,
        max_sequence_length=8,
    )
    data = _input_data(get_current_device())
    model.to(get_current_device())

    plan = model.build_schedule_plan(**data)

    assert plan.pre_process.gpt_model is model
    assert plan.post_process.gpt_model is model
    assert len(plan._transformer_layers) == model.config.num_layers
    assert plan.mhc_expand_input is False
    assert plan.state.input_ids is data['input_ids']
    assert plan.state.labels is data['labels']
    assert plan.state.attention_mask is data['attention_mask']
    assert plan.state.hc_head_fn is None
    assert plan.state.enable_mtp is False
    assert plan.state.decoder_input is None


def test_schedule_plan_run_invokes_pipeline_callbacks(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = _config()
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_decoder_block_spec(config, use_transformer_engine=False),
        vocab_size=128,
        max_sequence_length=8,
    ).eval()
    device = get_current_device()
    model.to(device)
    data = _input_data(device)
    set_streams()
    seen = []

    plan = model.build_schedule_plan(**data)
    f_input = TransformerModelChunkSchedulePlan.run(
        plan,
        None,
        pre_forward=lambda vp_stage: seen.append(('pre_forward', vp_stage)),
        post_forward=lambda tensor, vp_stage: seen.append(
            ('post_forward', tuple(tensor.shape), vp_stage)
        ),
    )
    TransformerModelChunkSchedulePlan.run(
        None,
        plan,
        b_grad=torch.ones_like(f_input),
        pre_backward=lambda vp_stage: seen.append(('pre_backward', vp_stage)),
        post_backward=lambda tensor, vp_stage: seen.append(
            ('post_backward', tuple(tensor.shape), vp_stage)
        ),
    )

    assert seen[0] == ('pre_forward', None)
    assert seen[1] == ('post_forward', (4, 2, model.config.hidden_size), None)
    assert seen[2] == ('pre_backward', None)
    assert seen[3] == ('post_backward', (4, 2, model.config.hidden_size), None)
    assert f_input.shape == (2, 4)


def test_schedule_plan_releases_state_after_backward(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = _config()
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_decoder_block_spec(config, use_transformer_engine=False),
        vocab_size=128,
        max_sequence_length=8,
    ).eval()
    device = get_current_device()
    model.to(device)
    data = _input_data(device)
    set_streams()

    plan = model.build_schedule_plan(**data)
    f_input = TransformerModelChunkSchedulePlan.run(plan, None)
    TransformerModelChunkSchedulePlan.run(None, plan, b_grad=torch.ones_like(f_input))

    assert plan.pre_process is None
    assert plan.post_process is None
    assert plan.state.model is None


def test_hyper_connection_schedule_plan_matches_plain_forward(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = _config(enable_hyper_connections=True, num_residual_streams=2)
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_decoder_block_spec(config, use_transformer_engine=False),
        vocab_size=128,
        max_sequence_length=8,
    ).eval()
    device = get_current_device()
    model.to(device)
    data = _input_data(device)
    set_streams()

    reference_loss = model.forward(**data)
    plan = model.build_schedule_plan(**data)
    assert plan.mhc_expand_input is True
    assert plan.state.hc_head_fn is model.decoder.hc_head_fn
    assert plan.state.num_residual_streams == 2

    f_input = TransformerModelChunkSchedulePlan.run(plan, None)
    assert f_input.shape == reference_loss.shape
    torch.testing.assert_close(f_input, reference_loss, rtol=1e-4, atol=1e-4)

    TransformerModelChunkSchedulePlan.run(None, plan, b_grad=torch.ones_like(f_input))
    assert plan.state.hc_head_fn is None
    assert plan.state.num_residual_streams is None
    assert plan.state.enable_mtp is None


def test_moe_mtp_schedule_plan_matches_plain_forward(model_parallel, nvte_auto_env):
    model_parallel(expert_model_parallel_size=4)
    model_parallel_cuda_manual_seed(123)
    config = _config(
        num_moe_experts=8,
        moe_token_dispatcher_type='alltoall',
        moe_grouped_gemm=False,
        moe_router_dtype='fp32',
        add_bias_linear=False,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=1.0,
    )
    block_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
    mtp_block_spec = get_gpt_mtp_block_spec(config, block_spec.layer_specs[-1], False)
    model = GPTModel(
        config=config,
        transformer_layer_spec=block_spec,
        mtp_block_spec=mtp_block_spec,
        vocab_size=128,
        max_sequence_length=8,
    ).eval()
    device = get_current_device()
    model.to(device)
    data = _input_data(device)
    set_streams()

    reference_loss = model.forward(**data)
    plan = model.build_schedule_plan(**data)
    assert len(plan._transformer_layers) == config.num_layers + 1
    assert plan.state.mtp_hidden_states is None

    f_input = TransformerModelChunkSchedulePlan.run(plan, None)
    assert f_input.shape == reference_loss.shape
    torch.testing.assert_close(f_input, reference_loss, rtol=1e-4, atol=1e-4)

    TransformerModelChunkSchedulePlan.run(None, plan, b_grad=torch.ones_like(f_input))


def test_fsdp_reshard_hooks_fire_on_forward_and_backward(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = _config()
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_decoder_block_spec(config, use_transformer_engine=False),
        vocab_size=128,
        max_sequence_length=8,
    ).eval()
    device = get_current_device()
    model.to(device)
    data = _input_data(device)
    set_streams()
    seen = []

    plan = model.build_schedule_plan(**data)
    plan._transformer_layers[0].set_fsdp_reshard_hooks(
        lambda module: seen.append(('forward_release', module)),
        lambda module: seen.append(('backward_release', module)),
    )
    f_input = TransformerModelChunkSchedulePlan.run(plan, None)
    TransformerModelChunkSchedulePlan.run(None, plan, b_grad=torch.ones_like(f_input))

    assert seen == [
        ('forward_release', model.decoder.layers[0]),
        ('backward_release', model.decoder.layers[0]),
    ]


def test_dense_layer_callables_raise_for_dispatch_and_combine(model_parallel, nvte_auto_env):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    config = _config()
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_decoder_block_spec(config, use_transformer_engine=False),
        vocab_size=128,
        max_sequence_length=8,
    ).eval()
    model.to(get_current_device())

    forward_funcs, backward_dw = build_layer_callables(model.decoder.layers[0])
    attn, dispatch, mlp, combine, post_process = forward_funcs

    assert post_process is None
    assert backward_dw == {
        'attn': model.decoder.layers[0].backward_dw_wrapper,
        'mlp': model.decoder.layers[0].mlp,
    }
    with pytest.raises(NotImplementedError, match='not implemented for Dense layer'):
        dispatch(None, None, None)
    with pytest.raises(NotImplementedError, match='not implemented for Dense layer'):
        combine(None, None)
