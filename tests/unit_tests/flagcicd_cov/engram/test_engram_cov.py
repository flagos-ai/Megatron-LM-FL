# Copyright (c) 2026, BAAI. All rights reserved.
"""Engram embeddings: vocab padding, seeded shard init, short conv, and offset-indexed lookups."""

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer import engram
from megatron.core.transformer.transformer_config import TransformerConfig

_UNK_TOKEN = "<unk>"
_PAD_TOKEN = "<pad>"
_WORDLEVEL_VOCAB = {
    _UNK_TOKEN: 0,
    _PAD_TOKEN: 1,
    "Hello": 2,
    "HELLO": 3,
    "Héllo": 4,
    "  Wide   Gap ": 5,
    "\ufffd": 6,
    "Mixed CASE": 7,
    "a": 8,
    "A": 9,
    "": 10,
}


@pytest.fixture
def tokenizer_path(tmp_path):
    tokenizer = Tokenizer(models.WordLevel(vocab=_WORDLEVEL_VOCAB, unk_token=_UNK_TOKEN))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token=_UNK_TOKEN, pad_token=_PAD_TOKEN
    )
    fast.save_pretrained(str(tmp_path))
    return str(tmp_path)


def _engram_config(tokenizer_path, **overrides):
    fields = dict(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        use_cpu_initialization=True,
        use_engram=True,
        engram_tokenizer_name_or_path=tokenizer_path,
        engram_vocab_size=[31, 37],
        max_ngram_size=3,
        n_embed_per_ngram=8,
        n_head_per_ngram=1,
        engram_layer_ids=[0],
        engram_pad_id=0,
        engram_seed=0,
        engram_kernel_size=3,
    )
    fields.update(overrides)
    return TransformerConfig(**fields)


@pytest.mark.parametrize(
    ("orig_vocab_size", "tp_size", "expected"),
    [
        pytest.param(16, 2, 16, id="already-divisible"),
        pytest.param(17, 2, 18, id="pads-up-to-multiple"),
        pytest.param(7, 4, 8, id="pads-seven-to-eight"),
        pytest.param(33, 8, 40, id="pads-by-large-tp"),
        pytest.param(1, 1, 1, id="single-partition-unchanged"),
    ],
)
def test_vocab_size_with_padding_rounds_up_to_tp_multiple(orig_vocab_size, tp_size, expected):
    assert engram._vocab_size_with_padding(orig_vocab_size, tp_size) == expected


def test_initialize_engram_weight_gpu_with_seed_is_deterministic_and_marks_shard():
    torch.manual_seed(3)
    weight_a = torch.empty(12, 4)
    weight_b = torch.empty(12, 4)
    weight_c = torch.empty(12, 4)
    engram._initialize_engram_weight_gpu_with_seed(weight_a, torch.nn.init.normal_, 4242)
    engram._initialize_engram_weight_gpu_with_seed(weight_b, torch.nn.init.normal_, 4242)
    engram._initialize_engram_weight_gpu_with_seed(weight_c, torch.nn.init.normal_, 4243)
    torch.testing.assert_close(weight_a, weight_b, rtol=0, atol=0)
    assert not torch.allclose(weight_a, weight_c)
    assert weight_a.tensor_model_parallel is True
    assert weight_a.partition_dim == 0
    assert weight_a.partition_stride == 1


def test_short_conv_builds_causal_depthwise_conv_over_rmsnorm_groups():
    no_activation = engram.ShortConv(
        hidden_size=8, kernel_size=3, dilation=2, norm_eps=1e-3, hc_mult=2, activation=False
    )
    assert no_activation.conv.groups == 16
    assert no_activation.conv.kernel_size == (3,)
    assert no_activation.conv.dilation == (2,)
    assert no_activation.conv.padding == (4,)
    assert no_activation.conv.bias is None
    assert len(no_activation.norms) == 2
    assert no_activation.norms[0].eps == 1e-3
    assert no_activation.norms[1].normalized_shape == (8,)
    assert no_activation.activation is False
    assert not hasattr(no_activation, "act_fn")
    with_activation = engram.ShortConv(hidden_size=8, kernel_size=3, hc_mult=2)
    assert isinstance(with_activation.act_fn, torch.nn.SiLU)


@pytest.mark.parametrize(
    "activation", [pytest.param(True, id="silu"), pytest.param(False, id="linear")]
)
def test_short_conv_forward_matches_reference_norms_and_conv(activation):
    torch.manual_seed(5)
    short_conv = engram.ShortConv(
        hidden_size=8, kernel_size=3, dilation=2, hc_mult=2, activation=activation
    )
    value = torch.randn(7, 3, 2, 8)
    output = short_conv(value)
    assert output.shape == (7, 3, 2, 8)
    permuted = value.permute(1, 0, 2, 3)
    normed = []
    for group in range(2):
        chunk = permuted[:, :, group, :]
        norm = chunk * torch.rsqrt(
            chunk.pow(2).mean(-1, keepdim=True) + short_conv.norms[group].eps
        )
        normed.append(norm * short_conv.norms[group].weight)
    flattened = torch.cat(normed, dim=-1).transpose(1, 2)
    conv_out = F.conv1d(
        flattened,
        short_conv.conv.weight,
        None,
        stride=1,
        padding=short_conv.conv.padding[0],
        dilation=short_conv.conv.dilation[0],
        groups=short_conv.conv.groups,
    )[..., :7]
    if activation:
        conv_out = F.silu(conv_out)
    expected = conv_out.transpose(1, 2).reshape(3, 7, 2, 8).permute(1, 0, 2, 3).contiguous()
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_short_conv_forward_rejects_mismatched_group_count():
    short_conv = engram.ShortConv(hidden_size=8, kernel_size=3, hc_mult=2)
    with pytest.raises(AssertionError, match="Input groups 3 != hc_mult 2"):
        short_conv(torch.randn(7, 3, 3, 8))


@pytest.mark.parametrize(
    "deterministic_mode",
    [pytest.param(True, id="deterministic"), pytest.param(False, id="f_embedding")],
)
def test_engram_embedding_lookup_matches_f_embedding_reference(deterministic_mode):
    torch.manual_seed(9)
    config = ModelParallelConfig(
        use_cpu_initialization=True,
        perform_initialization=True,
        deterministic_mode=deterministic_mode,
    )
    embedding = engram.EngramEmbedding(
        num_embeddings=16, embedding_dim=8, init_method=torch.nn.init.normal_, config=config
    )
    input_ids = torch.tensor([[3, 5, 1], [0, 15, 7]])
    output = embedding(input_ids)
    assert output.shape == (2, 3, 8)
    assert output.dtype == torch.float32
    torch.testing.assert_close(
        output, F.embedding(input_ids, embedding.weight.data), rtol=0, atol=0
    )


def test_engram_embedding_cpu_init_sets_shard_attributes():
    torch.manual_seed(11)
    embedding = engram.EngramEmbedding(
        num_embeddings=16,
        embedding_dim=8,
        init_method=torch.nn.init.normal_,
        config=ModelParallelConfig(use_cpu_initialization=True, perform_initialization=True),
    )
    assert tuple(embedding.weight.shape) == (16, 8)
    assert embedding.weight.tensor_model_parallel is True
    assert embedding.weight.partition_dim == 0
    assert embedding.weight.partition_stride == 1
    assert embedding.num_embeddings_per_partition == 16
    assert embedding.vocab_start_index == 0
    assert embedding.vocab_end_index == 16


def test_engram_embedding_enable_parallel_skips_flag_when_single_partition():
    embedding = engram.EngramEmbedding(
        num_embeddings=16,
        embedding_dim=8,
        init_method=torch.nn.init.normal_,
        config=ModelParallelConfig(use_cpu_initialization=True, perform_initialization=False),
    )
    assert embedding.embedding_parallel_size == 1
    embedding.enable_parallel()
    assert getattr(embedding.weight, "is_engram_embedding", False) is False
    assert getattr(embedding.weight, "allreduce", True) is True


def test_engram_embedding_enable_offloading_marks_weight():
    marked = engram.EngramEmbedding(
        num_embeddings=16,
        embedding_dim=8,
        init_method=torch.nn.init.normal_,
        config=ModelParallelConfig(use_cpu_initialization=True, perform_initialization=False),
    )
    unmarked = engram.EngramEmbedding(
        num_embeddings=16,
        embedding_dim=8,
        init_method=torch.nn.init.normal_,
        config=ModelParallelConfig(use_cpu_initialization=True, perform_initialization=False),
    )
    marked.enable_offloading()
    assert getattr(marked.weight, "is_offloading_candidate", False) is True
    assert getattr(unmarked.weight, "is_offloading_candidate", False) is False


def test_engram_embedding_sharded_state_dict_describes_single_partition():
    torch.manual_seed(13)
    embedding = engram.EngramEmbedding(
        num_embeddings=16,
        embedding_dim=8,
        init_method=torch.nn.init.normal_,
        config=ModelParallelConfig(use_cpu_initialization=True, perform_initialization=True),
    )
    sharded = embedding.sharded_state_dict(prefix="module.")
    assert list(sharded) == ["module.weight"]
    tensor = sharded["module.weight"]
    assert tensor.local_shape == (16, 8)
    assert tensor.global_shape == (16, 8)
    assert tensor.global_offset == (0, 0)
    assert tensor.replica_id == (0, 0, 0)
    assert tensor.axis_fragmentations == (1, 1)


def test_multi_head_embedding_shifts_each_head_by_its_vocab_offset(tokenizer_path):
    torch.manual_seed(15)
    embedding = engram.MultiHeadEmbedding(_engram_config(tokenizer_path), list_of_N=[7, 5], D=4)
    assert embedding.offsets.tolist() == [0, 7]
    assert embedding.num_heads == 2
    assert embedding.embedding_dim == 4
    input_ids = torch.tensor([[[1, 4], [3, 1], [2, 0], [0, 2]], [[6, 3], [5, 4], [4, 1], [2, 3]]])
    output = embedding(input_ids)
    assert output.shape == (4, 2, 2, 4)
    expected = F.embedding(input_ids + embedding.offsets, embedding.embedding.weight.data)
    torch.testing.assert_close(output, expected.transpose(0, 1).contiguous(), rtol=0, atol=0)


def test_multi_head_embedding_rejects_unknown_parallel_method(tokenizer_path):
    config = _engram_config(tokenizer_path, engram_embedding_parallel_method="bogus")
    with pytest.raises(ValueError, match="Unsupported engram_embedding_parallel_method: bogus"):
        engram.MultiHeadEmbedding(config, list_of_N=[3], D=2)
