# Copyright (c) 2026, BAAI. All rights reserved.
"""EngramModule on 8 GPUs: all-to-all parity, sequence-parallel scatter, embedding cache warmup."""

import argparse
import math

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

import megatron.training
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import engram
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

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
def model_parallel():
    def setup(**sizes):
        Utils.initialize_model_parallel(**sizes)

    yield setup
    Utils.destroy_model_parallel()


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
    return mapping.hash(raw_ids.cuda())[0]


def _gather_full_table(weight, group):
    world_size = torch.distributed.get_world_size(group)
    shards = [torch.empty_like(weight) for _ in range(world_size)]
    torch.distributed.all_gather(shards, weight, group=group)
    return torch.cat(shards, dim=0)


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


def _reference_module_forward(module, hidden_states, hash_ids, full_table):
    seq_len, batch_size, expanded = hidden_states.shape
    hc_mult = module.hc_mult
    shifted = hash_ids + module.multi_head_embedding.offsets
    embeddings = F.embedding(shifted, full_table)
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


def test_engram_module_alltoall_forward_and_grads_match_local_reference(
    model_parallel, tokenizer_path, training_args
):
    model_parallel(engram_embedding_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    module = engram.EngramModule(
        _engram_config(tokenizer_path, use_cpu_initialization=False), layer_id=0
    ).cuda()
    hash_ids = _layer_hash_ids(tokenizer_path)
    group = parallel_state.get_engram_embedding_parallel_group()
    full_table = _gather_full_table(module.multi_head_embedding.embedding.weight.data, group)
    torch.manual_seed(41)
    hidden_states = torch.randn(6, 2, 16, device="cuda", requires_grad=True)
    output = module(hidden_states, hash_ids)
    assert output.shape == (6, 2, 16)
    expected = _reference_module_forward(module, hidden_states, hash_ids, full_table)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    output.sum().backward()
    actual_grads = {name: param.grad.clone() for name, param in module.named_parameters()}
    actual_hidden_grad = hidden_states.grad.clone()
    module.zero_grad(set_to_none=True)
    reference_hidden = hidden_states.detach().clone().requires_grad_(True)
    reference = _reference_module_forward(module, reference_hidden, hash_ids, full_table)
    reference.sum().backward()
    for name, param in module.named_parameters():
        if name == "multi_head_embedding.embedding.weight":
            continue
        torch.testing.assert_close(param.grad, actual_grads[name], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(reference_hidden.grad, actual_hidden_grad, rtol=1e-5, atol=1e-5)


def test_engram_module_scatters_output_by_tp_rank_under_sequence_parallel(
    model_parallel, tokenizer_path, training_args
):
    model_parallel(tensor_model_parallel_size=2, engram_embedding_parallel_size=2)
    model_parallel_cuda_manual_seed(123)
    hash_ids = _layer_hash_ids(tokenizer_path)
    torch.manual_seed(21)
    module_sp = engram.EngramModule(
        _engram_config(tokenizer_path, sequence_parallel=True, tensor_model_parallel_size=2),
        layer_id=0,
    ).cuda()
    torch.manual_seed(21)
    module_reference = engram.EngramModule(
        _engram_config(tokenizer_path, sequence_parallel=False), layer_id=0
    ).cuda()
    torch.manual_seed(31)
    hidden_full = torch.randn(6, 2, 16)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    hidden_half = hidden_full[tp_rank * 3 : (tp_rank + 1) * 3].cuda().requires_grad_(True)
    hidden_reference = hidden_full.cuda().requires_grad_(True)
    output = module_sp(hidden_half, hash_ids)
    assert output.shape == (3, 2, 16)
    expected = module_reference(hidden_reference, hash_ids)
    torch.testing.assert_close(
        output, expected[tp_rank * 3 : (tp_rank + 1) * 3], rtol=1e-5, atol=1e-5
    )
    output.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        hidden_half.grad,
        hidden_reference.grad[tp_rank * 3 : (tp_rank + 1) * 3],
        rtol=1e-5,
        atol=1e-5,
    )


def test_engram_module_pre_compute_embedding_warms_cache_used_by_forward(
    model_parallel, tokenizer_path, training_args
):
    model_parallel()
    model_parallel_cuda_manual_seed(123)
    module = engram.EngramModule(
        _engram_config(tokenizer_path, use_cpu_initialization=False), layer_id=0
    ).cuda()
    hash_ids = _layer_hash_ids(tokenizer_path)
    torch.manual_seed(47)
    hidden_states = torch.randn(6, 2, 16, device="cuda")
    torch.cuda.synchronize()
    module.pre_compute_embedding(hash_ids)
    cached_embeddings, _ = module.embedding_cache
    assert cached_embeddings.shape == (6, 2, 16)
    cached = module(hidden_states, None)
    assert module.embedding_cache is None
    assert cached.shape == (6, 2, 16)
    recomputed = module(hidden_states, hash_ids)
    torch.testing.assert_close(cached, recomputed, rtol=1e-6, atol=1e-6)
