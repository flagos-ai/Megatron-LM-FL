# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import forward_step_calc_loss
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.internal


def _make_config(**kwargs):
    options = dict(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=16,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        rope_type="rope",
        normalization="RMSNorm",
        qk_layernorm=True,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        experimental_attention_variant="dsa",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=2048,
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_rotate_activation=False,
        gradient_accumulation_fusion=False,
    )
    options.update(kwargs)
    return MLATransformerConfig(**options)


@pytest.fixture(autouse=True)
def isolate_indexer_loss_state():
    previous_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
    previous_tracker = DSAIndexerLossLoggingHelper.tracker
    DSAIndexerLossAutoScaler.main_loss_backward_scale = None
    DSAIndexerLossLoggingHelper.tracker = {}
    yield
    Utils.destroy_model_parallel()
    DSAIndexerLossAutoScaler.main_loss_backward_scale = previous_scale
    DSAIndexerLossLoggingHelper.tracker = previous_tracker


def _build_gpt(config):
    return GPTModel(
        config=config,
        transformer_layer_spec=get_transformer_block_with_experimental_attention_variant_spec(
            config
        ),
        vocab_size=128,
        max_sequence_length=8,
        parallel_output=False,
        position_embedding_type="rope",
    ).cuda()


@pytest.mark.parametrize(
    "topk_freq",
    [
        1,
        # TODO: Remove with the config guard after selective core_attn sharing is fixed here
        # and seq_len > top-k, multi-microbatch output/gradient parity passes (#7107 upstream).
        pytest.param(
            2,
            marks=pytest.mark.skip(
                reason="DSA index sharing with selective core_attn is temporarily disabled."
            ),
        ),
    ],
)
def test_dsa_core_checkpoint_matches_eager_outputs_and_gradients(topk_freq):
    """Checkpoint x/qr and shared indices for two outstanding microbatches."""
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    eager = _build_gpt(_make_config(dsa_indexer_topk_freq=topk_freq))
    checkpointed = _build_gpt(
        _make_config(
            dsa_indexer_topk_freq=topk_freq,
            recompute_granularity="selective",
            recompute_modules=["core_attn"],
        )
    )
    checkpointed.load_state_dict(eager.state_dict(), strict=True)
    tokens = [torch.randint(0, 128, (1, 8), device="cuda") for _ in range(2)]
    positions = torch.arange(8, device="cuda").unsqueeze(0)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.ones(1, device="cuda"))

    expected = [eager(batch, positions, attention_mask=None) for batch in tokens]
    sum(output.square().mean() for output in expected).backward()
    actual = [checkpointed(batch, positions, attention_mask=None) for batch in tokens]
    sum(output.square().mean() for output in actual).backward()

    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, atol=2e-5, rtol=2e-3)
    eager_parameters = dict(eager.named_parameters())
    for name, parameter in checkpointed.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(
            parameter.grad, eager_parameters[name].grad, atol=2e-5, rtol=2e-3, msg=name
        )
    indexer = checkpointed.decoder.layers[0].self_attention.core_attention.indexer
    assert indexer.linear_wq_b.weight.grad.abs().max() > 0


def _run_cp_indexer_backward(cp_size, per_token, num_microbatches, state_dict=None):
    Utils.initialize_model_parallel(1, 1, context_parallel_size=cp_size)
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = _make_config(
        context_parallel_size=cp_size, cp_comm_type="allgather", calculate_per_token_loss=per_token
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    attention = build_module(
        get_dsa_module_spec_for_backend(config, TESpecProvider()),
        config=config,
        layer_number=1,
        cp_comm_type="allgather",
        pg_collection=pg_collection,
    ).cuda()
    if state_dict is not None:
        attention.load_state_dict(state_dict, strict=True)
    state_dict = copy.deepcopy(attention.state_dict())
    config.grad_scale_func = lambda value: value * 3.0
    model = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(overlap_grad_reduce=False, use_distributed_optimizer=False),
        attention,
    )
    model.zero_grad_buffer()
    torch.manual_seed(93)
    full_input = torch.randn(8, 1, config.hidden_size, device="cuda")
    positions = torch.arange(8, device="cuda")
    if cp_size == 2:
        positions = torch.tensor(
            [0, 1, 6, 7] if pg_collection.cp.rank() == 0 else [2, 3, 4, 5], device="cuda"
        )
    for _ in range(num_microbatches):
        output, _ = model(full_input[positions].clone().requires_grad_(True), attention_mask=None)
        forward_step_calc_loss(
            model=model,
            output_tensor=output,
            loss_func=None,
            config=config,
            vp_stage=None,
            collect_non_loss_data=False,
            num_microbatches=num_microbatches,
            forward_data_store=[],
            cp_group_size=cp_size,
            is_last_stage=True,
        )
        # Isolate the indexer auxiliary gradient from the main attention objective.
        (output.sum() * 0).backward()
    num_tokens = (
        torch.tensor(positions.numel() * num_microbatches, device="cuda") if per_token else None
    )
    finalize_model_grads([model], num_tokens=num_tokens)
    gradients = {
        name: parameter.main_grad.detach().clone()
        for name, parameter in attention.named_parameters()
        if "indexer" in name
    }
    DSAIndexerLossLoggingHelper.tracker.clear()
    Utils.destroy_model_parallel()
    return state_dict, gradients


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("num_microbatches", [1, 3])
def test_dsa_cp1_cp2_indexer_gradients_match(per_token, num_microbatches):
    """Compare the same batch after real CP/DDP reduction and token normalization."""
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("CP2 requires an even world size of at least two")
    state, expected = _run_cp_indexer_backward(1, per_token, num_microbatches)
    _, actual = _run_cp_indexer_backward(2, per_token, num_microbatches, state)
    assert expected and max(gradient.abs().max() for gradient in expected.values()) > 0
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=2e-5, rtol=2e-3, msg=name)
