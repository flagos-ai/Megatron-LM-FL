# Copyright (c) 2026, BAAI. All rights reserved.
"""CompressedSparseAttention components: Compressor overlap layout and APE shape, token dropping past full groups, CSAIndexer wiring with rotating compressor, process-group defaults, MTP layer numbering, and uncompressed-KV fallback."""

import pytest
import torch

from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    CompressedSparseAttentionSubmodules,
    Compressor,
    CompressorSubmodules,
    CSAIndexer,
    CSAIndexerSubmodules,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import get_pg_size
from tests.unit_tests.test_utilities import Utils


def _make_config(**overrides):
    kwargs = dict(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=16,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_head_dim=32,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        csa_compress_ratios=[4, 128, 4, 128],
        csa_window_size=8,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=8,
    )
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


def _te_linear_spec():
    from megatron.core.extensions.transformer_engine import TELinear
    from megatron.core.transformer.spec_utils import ModuleSpec

    return ModuleSpec(module=TELinear)


def _make_compressor_submodules():
    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CompressorSubmodules(
        linear_wkv=_te_linear_spec(), linear_wgate=_te_linear_spec(), norm=ModuleSpec(module=TENorm)
    )


def _make_csa_indexer_submodules():
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CSAIndexerSubmodules(
        linear_wq_b=_te_linear_spec(),
        linear_weights_proj=_te_linear_spec(),
        compressor=ModuleSpec(module=Compressor, submodules=_make_compressor_submodules()),
    )


def _make_csa_submodules():
    from megatron.core.transformer.spec_utils import ModuleSpec

    return CompressedSparseAttentionSubmodules(
        compressor=ModuleSpec(module=Compressor, submodules=_make_compressor_submodules()),
        indexer=ModuleSpec(module=CSAIndexer, submodules=_make_csa_indexer_submodules()),
    )


class TestCompressorConstruction:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        'compress_ratio,expected_coff',
        [pytest.param(4, 2, id='overlap'), pytest.param(128, 1, id='non_overlap')],
    )
    def test_compressor_resolves_default_process_groups_and_ape_layout(
        self, compress_ratio, expected_coff
    ):
        config = _make_config()
        compressor = Compressor(
            config=config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=config.v_head_dim,
            rotate=False,
            rotary_pos_emb=None,
            pg_collection=None,
        )
        assert get_pg_size(compressor.pg_collection.tp) == 1
        assert compressor.compress_ratio == compress_ratio
        assert compressor.overlap is (compress_ratio == 4)
        assert compressor.coff == expected_coff
        assert compressor.ape.shape == (compress_ratio, expected_coff * config.v_head_dim)
        assert compressor.ape.dtype == torch.float32


class TestCompressorForward:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        cls = request.cls
        cls.config = _make_config()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        cls.rotary = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg.cp,
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        'compress_ratio,seq_len',
        [pytest.param(4, 66, id='ratio4'), pytest.param(128, 129, id='ratio128')],
    )
    def test_compressor_drops_tokens_beyond_full_groups(self, compress_ratio, seq_len):
        compressor = Compressor(
            config=self.config,
            submodules=_make_compressor_submodules(),
            compress_ratio=compress_ratio,
            head_dim=self.config.v_head_dim,
            rotate=False,
            rotary_pos_emb=self.rotary,
            pg_collection=self.pg,
        ).cuda()
        device = torch.cuda.current_device()
        batch_size = 2
        x = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16, device=device
        )
        full_len = (seq_len // compress_ratio) * compress_ratio
        output = compressor(x)
        output_full = compressor(x[:full_len])
        assert output.shape == (full_len // compress_ratio, batch_size, self.config.v_head_dim)
        assert output.dtype == torch.bfloat16
        torch.testing.assert_close(output, output_full, rtol=1.6e-2, atol=1e-2)


class TestCSAIndexerConstruction:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        cls = request.cls
        cls.config = _make_config()
        cls.rotary = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
        )
        yield
        Utils.destroy_model_parallel()

    def test_csa_indexer_resolves_default_process_groups_and_builds_rotating_compressor(self):
        indexer = CSAIndexer(
            config=self.config,
            submodules=_make_csa_indexer_submodules(),
            compress_ratio=4,
            rotary_pos_emb=self.rotary,
            pg_collection=None,
        )
        assert get_pg_size(indexer.pg_collection.tp) == 1
        assert indexer.compress_ratio == 4
        assert indexer.index_n_heads == self.config.dsa_indexer_n_heads
        assert indexer.index_head_dim == self.config.dsa_indexer_head_dim
        assert indexer.index_topk == self.config.dsa_indexer_topk
        assert indexer.softmax_scale == self.config.dsa_indexer_head_dim**-0.5
        assert indexer.compressor.rotate is True
        assert indexer.compressor.head_dim == self.config.dsa_indexer_head_dim


class TestCSAConstruction:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        cls = request.cls
        cls.config = _make_config()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        cls.rotary = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg.cp,
        )
        yield
        Utils.destroy_model_parallel()

    def test_csa_resolves_default_process_groups_and_builds_compressor_and_indexer(self):
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=None,
            rotary_pos_emb=self.rotary,
            compress_ratio=4,
        )
        assert get_pg_size(csa.pg_collection.tp) == 1
        assert csa.compress_ratio == 4
        assert csa.window_size == self.config.csa_window_size
        assert csa.v_head_dim == self.config.v_head_dim
        assert csa.softmax_scale == self.config.v_head_dim**-0.5
        assert isinstance(csa.compressor, Compressor)
        assert isinstance(csa.indexer, CSAIndexer)
        assert csa.attn_sink.shape == (self.config.num_attention_heads,)
        assert csa.attn_sink.dtype == torch.float32

    def test_csa_shifts_layer_number_by_num_layers_for_mtp_layer(self):
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg,
            rotary_pos_emb=self.rotary,
            compress_ratio=4,
            is_mtp_layer=True,
        )
        assert csa.layer_number == 1 + self.config.num_layers


class TestCSABuildKvFull:

    @pytest.fixture(scope='class', autouse=True)
    def setup_model_parallel(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        cls = request.cls
        cls.config = _make_config()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        cls.rotary = RotaryEmbedding(
            cls.config.qk_pos_emb_head_dim,
            rotary_percent=cls.config.rotary_percent,
            rotary_base=cls.config.rotary_base,
            cp_group=cls.pg.cp,
        )
        yield
        Utils.destroy_model_parallel()

    def test_build_kv_full_returns_uncompressed_kv_when_input_shorter_than_ratio(self):
        csa = CompressedSparseAttention(
            config=self.config,
            submodules=_make_csa_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg,
            rotary_pos_emb=self.rotary,
            compress_ratio=4,
        ).cuda()
        device = torch.cuda.current_device()
        batch_size = 2
        kv = torch.randn(3, batch_size, self.config.v_head_dim, dtype=torch.bfloat16, device=device)
        x = torch.randn(3, batch_size, self.config.hidden_size, dtype=torch.bfloat16, device=device)
        kv_full, compressed_kv, n_compressed = csa._build_kv_full(kv, x)
        assert kv_full is kv
        assert compressed_kv is None
        assert n_compressed == 0
