# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.recompute import checkpointed_forward
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.experimental_attention_variant.dsa_index_sharing import (
    get_dsa_index_sharing_state,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize('method', ['uniform', 'block'])
@pytest.mark.parametrize('carrier', ['none', 'mask', 'packed'])
def test_shared_topk_survives_two_checkpointed_microbatches(method, carrier):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    config = TransformerConfig(
        num_layers=2,
        hidden_size=2,
        num_attention_heads=1,
        experimental_attention_variant='dsa',
        dsa_indexer_topk=1,
        dsa_indexer_topk_freq=2,
        dsa_indexer_loss_coeff=0.0,
    )
    config.recompute_method = method
    config.recompute_num_layers = 1 if method == 'uniform' else 2
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool, device='cuda') if carrier == 'mask' else None
    cu = torch.tensor([0, 4], dtype=torch.int32, device='cuda')
    packed = (
        PackedSeqParams(
            qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=4, max_seqlen_kv=4
        )
        if carrier == 'packed'
        else None
    )

    class Indexer(torch.nn.Module):
        def forward_before_topk(self, x, qr, packed):
            return torch.ones(4, 1, 1, 2, device=x.device), x.expand(4, 1, 2), torch.ones_like(x)

    with patch.object(dsa, 'build_module', side_effect=lambda *args, **kwargs: Indexer()):
        full, shared = [
            dsa.DSAttention(
                config,
                SimpleNamespace(indexer=None),
                layer,
                AttnMaskType.causal,
                'self',
                pg_collection=groups,
            )
            for layer in (1, 2)
        ]

    class Layer(torch.nn.Module):
        def __init__(self, attention, source):
            super().__init__()
            self.attention = attention
            self.source = source

        def forward(self, hidden_states, **kwargs):
            x = hidden_states
            q = torch.zeros(4, 1, 1, 2, device=x.device)
            value = x.unsqueeze(-1)
            if packed is not None:
                q, value = q.squeeze(1), value.squeeze(1)
            result = self.attention(
                q,
                q,
                value,
                mask,
                x,
                x,
                attn_mask_type=AttnMaskType.causal,
                packed_seq_params=packed,
            )
            if self.source:
                # Simulate fused top-k's auxiliary lengths, distinct per microbatch.
                holder = self.attention._get_index_share_topk_length_holder(packed, mask)
                holder[self.attention.layer_number] = x[0].detach().clone()
                return x * 1
            return result

    layers = [Layer(full, True), Layer(shared, False)]
    block = SimpleNamespace(
        config=config, layers=layers, num_layers_per_pipeline_rank=2, pg_collection=groups
    )
    original_sparse = dsa._run_sparse_attention
    lengths = []

    def capture_sparse(**kwargs):
        if kwargs['topk_length'] is not None:
            lengths.append(kwargs['topk_length'].detach().clone())
        return original_sparse(**kwargs)

    def inputs():
        return [
            torch.tensor(v, device='cuda').view(4, 1, 1).requires_grad_()
            for v in ([1.0, 4.0, 2.0, 3.0], [7.0, 1.0, 2.0, 3.0])
        ]

    try:
        with patch.object(dsa, '_run_sparse_attention', side_effect=capture_sparse):
            expected_inputs = inputs()
            expected_outputs = []
            for x in expected_inputs:
                y = layers[1](layers[0](x))
                expected_outputs.append(y.detach().clone())
                y.sum().backward()
            lengths.clear()
            actual_inputs = inputs()
            actual_outputs = [
                checkpointed_forward(block, x, mask, None, None, None, None, packed, False)
                for x in actual_inputs
            ]
            for y in actual_outputs:
                y.sum().backward()
            for actual, expected in zip(actual_outputs, expected_outputs):
                torch.testing.assert_close(actual, expected)
            for actual, expected in zip(actual_inputs, expected_inputs):
                torch.testing.assert_close(actual.grad, expected.grad)
            assert [int(v.item()) for v in lengths] == [1, 7, 1, 7]
            assert get_dsa_index_sharing_state() is None
            ordinary_input = inputs()[0]
            ordinary_output = layers[1](layers[0](ordinary_input))
            torch.testing.assert_close(ordinary_output, expected_outputs[0])
            assert get_dsa_index_sharing_state() is None
    finally:
        Utils.destroy_model_parallel()


def test_mtp_checkpoint_sharing_survives_two_microbatches():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    config = TransformerConfig(
        num_layers=2,
        hidden_size=2,
        num_attention_heads=1,
        experimental_attention_variant='dsa',
        dsa_indexer_topk=1,
        dsa_indexer_topk_freq=2,
        dsa_indexer_loss_coeff=0.0,
    )
    config.mtp_num_layers = 2
    config.recompute_method = 'uniform'
    config.recompute_num_layers = 1

    class Indexer(torch.nn.Module):
        def forward_before_topk(self, x, qr, packed):
            return torch.ones(4, 1, 1, 2, device=x.device), x.expand(4, 1, 2), torch.ones_like(x)

    with patch.object(dsa, 'build_module', side_effect=lambda *args, **kwargs: Indexer()):
        attentions = [
            dsa.DSAttention(
                config,
                SimpleNamespace(indexer=None),
                layer,
                AttnMaskType.causal,
                'self',
                pg_collection=groups,
                is_mtp_layer=True,
            )
            for layer in (1, 2)
        ]

    def embeddings(self, input_ids, position_ids, padding_mask, hidden_states, **kwargs):
        return input_ids, position_ids, padding_mask, hidden_states, hidden_states

    def projection(attention, source):
        def run(self, hidden_states, **kwargs):
            x = hidden_states
            q = torch.zeros(4, 1, 1, 2, device=x.device)
            out = attention(q, q, x.unsqueeze(-1), None, x, x, attn_mask_type=AttnMaskType.causal)
            if source:
                holder = attention._get_index_share_topk_length_holder(None, None)
                holder[attention.layer_number] = x[0].detach().clone()
                return x * 1
            return out

        return run

    layers = []
    for i, attention in enumerate(attentions):
        layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
        torch.nn.Module.__init__(layer)
        layer.config = config
        layer._get_embeddings = MethodType(embeddings, layer)
        layer._proj_and_transformer_layer = MethodType(projection(attention, i == 0), layer)
        layers.append(layer)
    block = SimpleNamespace(
        config=config, vp_stage=None, mtp_use_repeated_layer=False, layers=layers
    )
    original_sparse = dsa._run_sparse_attention
    lengths = []

    def capture_sparse(**kwargs):
        if kwargs['topk_length'] is not None:
            lengths.append(kwargs['topk_length'].detach().clone())
        return original_sparse(**kwargs)

    def inputs():
        return [
            torch.tensor(v, device='cuda').view(4, 1, 1).requires_grad_()
            for v in ([1.0, 4.0, 2.0, 3.0], [7.0, 1.0, 2.0, 3.0])
        ]

    def forward(x):
        return MultiTokenPredictionBlock.forward(block, None, None, x, None)[-4:]

    try:
        with patch.object(dsa, '_run_sparse_attention', side_effect=capture_sparse):
            config.recompute_granularity = None
            expected_inputs = inputs()
            expected_outputs = []
            for x in expected_inputs:
                y = forward(x)
                expected_outputs.append(y.detach().clone())
                y.sum().backward()
            config.recompute_granularity = 'full'
            lengths.clear()
            actual_inputs = inputs()
            actual_outputs = [forward(x) for x in actual_inputs]
            for y in actual_outputs:
                y.sum().backward()
            for actual, expected in zip(actual_outputs, expected_outputs):
                torch.testing.assert_close(actual, expected)
            for actual, expected in zip(actual_inputs, expected_inputs):
                torch.testing.assert_close(actual.grad, expected.grad)
            assert [int(v.item()) for v in lengths] == [1, 7, 1, 7]
            assert get_dsa_index_sharing_state() is None
    finally:
        Utils.destroy_model_parallel()
