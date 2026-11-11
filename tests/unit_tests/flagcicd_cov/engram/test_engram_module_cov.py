# Copyright (c) 2026, BAAI. All rights reserved.
"""EngramModule forward/backward: gated n-gram memory plus short conv, vs manual reference."""

import argparse
import math

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

import megatron.training
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


@pytest.fixture
def training_args(monkeypatch):
    monkeypatch.setattr(
        megatron.training, "get_args", lambda: argparse.Namespace(vocab_size=len(_WORDLEVEL_VOCAB))
    )


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


def _layer_hash_ids(tokenizer_path):
    mapping = engram.get_or_create_hash_mapping(
        engram_vocab_size=[31, 37],
        max_ngram_size=3,
        n_embed_per_ngram=8,
        n_head_per_ngram=1,
        layer_ids=[0],
        tokenizer_name_or_path=tokenizer_path,
        pad_id=0,
        seed=0,
    )
    torch.manual_seed(17)
    raw_ids = torch.randint(0, len(_WORDLEVEL_VOCAB), (2, 6))
    return mapping.hash(raw_ids)[0]


def _reference_short_conv(short_conv, value):
    seq_len, batch_size, groups, channels = value.shape
    permuted = value.permute(1, 0, 2, 3)
    normed = []
    for group in range(groups):
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
    )[..., :seq_len]
    if short_conv.activation:
        conv_out = F.silu(conv_out)
    return (
        conv_out.transpose(1, 2)
        .reshape(batch_size, seq_len, groups, channels)
        .permute(1, 0, 2, 3)
        .contiguous()
    )


def _reference_module_forward(module, hidden_states, hash_ids):
    seq_len, batch_size, expanded = hidden_states.shape
    hc_mult = module.hc_mult
    shifted = hash_ids + module.multi_head_embedding.offsets
    embeddings = F.embedding(shifted, module.multi_head_embedding.embedding.weight)
    embeddings = embeddings.transpose(0, 1).contiguous().flatten(start_dim=-2)
    scale = 1.0 / math.sqrt(module.config.hidden_size)
    gates = []
    for hc_idx in range(hc_mult):
        normed_key = module.norm1[hc_idx](module.key_projs[hc_idx](embeddings))
        query = hidden_states.view(seq_len, batch_size, hc_mult, -1)[:, :, hc_idx, :]
        normed_query = module.norm2[hc_idx](query)
        gate = torch.sum(normed_key * normed_query, dim=-1, keepdim=True) * scale
        gate = torch.sigmoid(torch.sign(gate) * torch.sqrt(torch.abs(gate).clamp_min(1e-6)))
        gates.append(gate)
    gates = torch.stack(gates, dim=2)
    value = gates * module.value_proj(embeddings).unsqueeze(2)
    output = value + _reference_short_conv(module.short_conv, value)
    return output.view(seq_len, batch_size, expanded)


def test_engram_module_builds_head_vocabs_from_layer_primes(tokenizer_path, training_args):
    module = engram.EngramModule(_engram_config(tokenizer_path), layer_id=0)
    assert module.multi_head_embedding.offsets.tolist() == [0, 31]
    assert module.multi_head_embedding.num_heads == 2
    assert module.multi_head_embedding.embedding_dim == 8
    assert module.hc_mult == 1
    assert module.value_proj.in_features == 16
    assert module.value_proj.out_features == 16
    assert len(module.key_projs) == 1
    assert len(module.norm1) == 1
    assert len(module.norm2) == 1
    assert module.short_conv.hc_mult == 1
    assert module.short_conv.conv.kernel_size == (3,)
    assert module.short_conv.conv.dilation == (3,)
    assert module.embedding_cache is None


@pytest.mark.parametrize(
    ("enable_hyper_connections", "num_residual_streams", "hc_mult"),
    [pytest.param(False, 1, 1, id="single_stream"), pytest.param(True, 2, 2, id="two_streams")],
)
def test_engram_module_forward_matches_manual_gated_conv_reference(
    tokenizer_path, training_args, enable_hyper_connections, num_residual_streams, hc_mult
):
    torch.manual_seed(61)
    config = _engram_config(
        tokenizer_path,
        enable_hyper_connections=enable_hyper_connections,
        num_residual_streams=num_residual_streams,
    )
    module = engram.EngramModule(config, layer_id=0)
    assert module.hc_mult == hc_mult
    hidden_states = torch.randn(6, 2, 16 * hc_mult)
    output = module(hidden_states, _layer_hash_ids(tokenizer_path))
    assert output.shape == (6, 2, 16 * hc_mult)
    expected = _reference_module_forward(module, hidden_states, _layer_hash_ids(tokenizer_path))
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("enable_hyper_connections", "num_residual_streams", "hc_mult"),
    [pytest.param(False, 1, 1, id="single_stream"), pytest.param(True, 2, 2, id="two_streams")],
)
def test_engram_module_backward_grads_match_manual_reference(
    tokenizer_path, training_args, enable_hyper_connections, num_residual_streams, hc_mult
):
    torch.manual_seed(71)
    config = _engram_config(
        tokenizer_path,
        enable_hyper_connections=enable_hyper_connections,
        num_residual_streams=num_residual_streams,
    )
    module = engram.EngramModule(config, layer_id=0)
    hash_ids = _layer_hash_ids(tokenizer_path)
    hidden_states = torch.randn(6, 2, 16 * hc_mult, requires_grad=True)
    output = module(hidden_states, hash_ids)
    output.sum().backward()
    actual_grads = {name: param.grad.clone() for name, param in module.named_parameters()}
    actual_hidden_grad = hidden_states.grad.clone()
    module.zero_grad(set_to_none=True)
    reference_hidden = hidden_states.detach().clone().requires_grad_(True)
    expected = _reference_module_forward(module, reference_hidden, hash_ids)
    expected.sum().backward()
    for name, param in module.named_parameters():
        torch.testing.assert_close(param.grad, actual_grads[name], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reference_hidden.grad, actual_hidden_grad, rtol=1e-5, atol=1e-6)


def test_engram_module_forward_rejects_missing_hash_ids_without_cache(
    tokenizer_path, training_args
):
    module = engram.EngramModule(_engram_config(tokenizer_path), layer_id=0)
    with pytest.raises(AssertionError, match="hash input ids can not be None for Engram"):
        module(torch.randn(6, 2, 16), None)
