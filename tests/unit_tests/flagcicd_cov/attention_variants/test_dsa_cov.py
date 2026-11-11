# Copyright (c) 2026, BAAI. All rights reserved.
"""DSAIndexer and DSAttention: process-group defaults, Yarn rotary indexer outputs, share-mode reuse of the previous layer's top-k indices, and masked sparse attention."""

import pytest
import torch

import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module
from megatron.core.models.common.embeddings import YarnRotaryEmbedding
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

_HADAMARD_MATRICES = {}


def _hadamard_matrix(size):
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)], dim=0
        )
    return matrix


def _hadamard_rotate(tensor, scale=1.0):
    """Deterministic Hadamard rotation used when fast_hadamard_transform is absent."""
    size = tensor.shape[-1]
    if size not in _HADAMARD_MATRICES:
        _HADAMARD_MATRICES[size] = _hadamard_matrix(size)
    matrix = _HADAMARD_MATRICES[size].to(device=tensor.device, dtype=tensor.dtype)
    return torch.matmul(tensor, matrix) * scale


@pytest.fixture(autouse=True)
def patch_hadamard_transform_when_missing(monkeypatch):
    if dsa_module.hadamard_transform is None:
        monkeypatch.setattr(dsa_module, 'hadamard_transform', _hadamard_rotate, raising=False)


def _make_config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=32,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=16,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


def _make_indexer_submodules():
    from megatron.core.extensions.transformer_engine import TELinear, TENorm

    return dsa_module.DSAIndexerSubmodules(
        linear_wq_b=ModuleSpec(module=TELinear),
        linear_wk=ModuleSpec(module=TELinear),
        k_norm=ModuleSpec(module=TENorm),
        linear_weights_proj=ModuleSpec(module=TELinear),
    )


def _make_ds_attention(config, layer_number, is_mtp_layer=False):
    indexer_spec = ModuleSpec(module=dsa_module.DSAIndexer, submodules=_make_indexer_submodules())
    return dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=indexer_spec),
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type='self',
        is_mtp_layer=is_mtp_layer,
    )


class TestDSAIndexerConstruction:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        request.cls.pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        yield
        Utils.destroy_model_parallel()

    def test_indexer_resolves_default_process_groups_and_builds_yarn_rotary(self):
        config = _make_config(rope_type='yarn')
        indexer = dsa_module.DSAIndexer(config, _make_indexer_submodules(), pg_collection=None)
        assert isinstance(indexer.rotary_pos_emb, YarnRotaryEmbedding)
        assert indexer.softmax_scale == config.dsa_indexer_head_dim**-0.5
        assert indexer.index_topk == config.dsa_indexer_topk
        assert indexer.index_n_heads == config.dsa_indexer_n_heads
        assert indexer.q_lora_rank == config.q_lora_rank

    def test_indexer_rejects_unsupported_rope_type(self):
        config = _make_config(rope_type='bogus')
        with pytest.raises(ValueError, match='Unsupported RoPE type'):
            dsa_module.DSAIndexer(config, _make_indexer_submodules(), pg_collection=self.pg)


class TestDSAIndexerYarnForward:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        request.cls.pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        yield
        Utils.destroy_model_parallel()

    def test_forward_before_topk_yarn_returns_rope_rotated_q_k_and_weights(self):
        config = _make_config(rope_type='yarn')
        indexer = dsa_module.DSAIndexer(
            config, _make_indexer_submodules(), pg_collection=self.pg
        ).cuda()
        rope_config = _make_config(rope_type='rope')
        rope_indexer = dsa_module.DSAIndexer(
            rope_config, _make_indexer_submodules(), pg_collection=self.pg
        )
        rope_indexer.load_state_dict(indexer.state_dict())
        rope_indexer = rope_indexer.cuda()

        device = torch.cuda.current_device()
        sq, b = 32, 2
        x = torch.randn(sq, b, config.hidden_size, dtype=torch.bfloat16, device=device)
        qr = torch.randn(sq, b, config.q_lora_rank, dtype=torch.bfloat16, device=device)

        q_yarn, k_yarn, weights_yarn = indexer.forward_before_topk(x, qr)
        q_rope, k_rope, weights_rope = rope_indexer.forward_before_topk(x, qr)

        assert q_yarn.shape == (sq, b, config.dsa_indexer_n_heads, config.dsa_indexer_head_dim)
        assert k_yarn.shape == (sq, b, config.dsa_indexer_head_dim)
        assert weights_yarn.shape == (sq, b, config.dsa_indexer_n_heads)
        assert q_yarn.dtype == torch.bfloat16
        assert k_yarn.dtype == torch.bfloat16
        torch.testing.assert_close(weights_yarn.float(), weights_rope.float(), rtol=0, atol=0)
        expected_weights = (
            torch.nn.functional.linear(x.float(), indexer.linear_weights_proj.weight.float())
            * config.dsa_indexer_n_heads**-0.5
            * indexer.softmax_scale
        )
        torch.testing.assert_close(weights_yarn.float(), expected_weights, rtol=1.6e-2, atol=1e-2)
        assert not torch.allclose(q_yarn.float(), q_rope.float(), rtol=1e-3, atol=1e-3)


class TestDSAttentionConstruction:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    def test_layer_number_shifts_by_num_layers_for_mtp_layer(self):
        config = _make_config(num_layers=2)
        attn = _make_ds_attention(config, layer_number=1, is_mtp_layer=True)
        assert attn.layer_number == 3

    def test_share_indexer_type_builds_no_indexer(self):
        config = _make_config(indexer_types=['full', 'share'])
        full_attn = _make_ds_attention(config, layer_number=1)
        share_attn = _make_ds_attention(config, layer_number=2)
        assert isinstance(full_attn.indexer, dsa_module.DSAIndexer)
        assert full_attn.indexer.index_topk == config.dsa_indexer_topk
        assert full_attn.indexer.softmax_scale == config.dsa_indexer_head_dim**-0.5
        assert share_attn.indexer is None
        assert list(share_attn.named_parameters()) == []
        assert {name for name, _ in full_attn.named_parameters()} == {
            f'indexer.{name}' for name, _ in full_attn.indexer.named_parameters()
        }
        assert share_attn.softmax_scale == full_attn.softmax_scale

    def test_softmax_scale_defaults_to_inverse_sqrt_of_kv_channels(self):
        config = _make_config()
        attn = _make_ds_attention(config, layer_number=1)
        assert attn.softmax_scale == pytest.approx(config.kv_channels**-0.5, rel=1e-12)


class TestDSAttentionShareForward:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    def _make_inputs(self, config, sq, b):
        device = torch.cuda.current_device()
        n_heads = config.num_attention_heads
        head_dim = config.kv_channels
        query = torch.randn(sq, b, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        key = torch.randn(sq, b, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        value = torch.randn(sq, b, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        x = torch.randn(sq, b, config.hidden_size, dtype=torch.bfloat16, device=device)
        qr = torch.randn(sq, b, config.q_lora_rank, dtype=torch.bfloat16, device=device)
        return query, key, value, x, qr

    def test_share_layer_requires_prev_topk_indices(self):
        config = _make_config(indexer_types=['full', 'share'])
        attn = _make_ds_attention(config, layer_number=2).cuda()
        sq, b = 16, 2
        query, key, value, x, qr = self._make_inputs(config, sq, b)
        with pytest.raises(AssertionError, match='no prev_topk_indices'):
            attn(query=query, key=key, value=value, attention_mask=None, x=x, qr=qr)

    def test_share_layer_reuses_prev_topk_indices_for_sparse_attention(self):
        config = _make_config(indexer_types=['full', 'share'])
        attn = _make_ds_attention(config, layer_number=2).cuda()
        sq, b, topk = 16, 2, 8
        query, key, value, x, qr = self._make_inputs(config, sq, b)
        device = query.device
        upper = torch.arange(sq, device=device).view(1, sq, 1) + 1
        prev_topk = (torch.rand(b, sq, topk, device=device) * upper).long()
        output = attn(
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            x=x,
            qr=qr,
            prev_topk_indices=prev_topk,
        )
        expected = dsa_module.unfused_dsa_fn(query, key, value, prev_topk, attn.softmax_scale)
        assert output.shape == (sq, b, config.num_attention_heads * config.kv_channels)
        assert torch.isfinite(output.float()).all()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert attn.current_topk_indices is prev_topk


class TestDSAttentionAttentionMaskForward:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    def test_forward_uses_attention_mask_when_attn_mask_type_is_none(self):
        config = _make_config(dsa_indexer_topk=8)
        attn = _make_ds_attention(config, layer_number=1).cuda()
        attn.eval()
        device = torch.cuda.current_device()
        sq, b, topk = 16, 2, 8
        n_heads = config.num_attention_heads
        head_dim = config.kv_channels
        query = torch.randn(sq, b, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        key = torch.randn(sq, b, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        value = torch.randn(sq, b, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        x = torch.randn(sq, b, config.hidden_size, dtype=torch.bfloat16, device=device)
        qr = torch.randn(sq, b, config.q_lora_rank, dtype=torch.bfloat16, device=device)
        attention_mask = torch.triu(
            torch.ones(b, 1, sq, sq, dtype=torch.bool, device=device), diagonal=1
        )

        output = attn(query=query, key=key, value=value, attention_mask=attention_mask, x=x, qr=qr)

        squeezed_mask = attention_mask.squeeze()
        float_mask = torch.zeros_like(squeezed_mask, dtype=torch.float32).masked_fill(
            squeezed_mask, float('-inf')
        )
        _, topk_indices = attn.indexer.forward_with_scores(x, qr, mask=float_mask)
        expected = dsa_module.unfused_dsa_fn(query, key, value, topk_indices, attn.softmax_scale)
        assert topk_indices.shape == (b, sq, min(topk, sq))
        assert output.shape == (sq, b, n_heads * head_dim)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
        assert torch.equal(attn.current_topk_indices, topk_indices)
