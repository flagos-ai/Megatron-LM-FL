# Copyright (c) 2026, BAAI. All rights reserved.
"""N-gram hashes: compressed IDs, per-layer multipliers, xor/remainder mixing, cache reuse."""

import argparse

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

import megatron.training
from megatron.core.transformer import engram

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


@pytest.mark.parametrize(
    ("start", "seen_primes", "expected"),
    [
        pytest.param(1, frozenset(), 2, id="first-prime"),
        pytest.param(10, frozenset({11}), 13, id="skips-seen-prime"),
        pytest.param(13, frozenset({17, 19}), 23, id="skips-several-seen"),
        pytest.param(30, frozenset({31}), 37, id="continues-past-seen"),
    ],
)
def test_find_next_prime_skips_seen_primes_and_composites(start, seen_primes, expected):
    assert engram.find_next_prime(start, seen_primes) == expected


def test_compressed_tokenizer_dedupes_normalized_spellings_into_shared_ids(
    tokenizer_path, training_args
):
    tokenizer = engram.CompressedTokenizer(tokenizer_path)
    assert len(tokenizer) == 8
    assert tokenizer.lookup_table.dtype == torch.long
    assert tokenizer.lookup_table.tolist() == [0, 1, 2, 2, 2, 3, 4, 5, 6, 6, 7]


def test_compressed_tokenizer_compress_maps_in_range_ids_and_passes_others_through(
    tokenizer_path, training_args
):
    tokenizer = engram.CompressedTokenizer(tokenizer_path)
    input_ids = torch.tensor([[2, 9, 10, -3, 100, 0]])
    assert tokenizer(input_ids).tolist() == [[2, 6, 7, -3, 100, 0]]


def test_get_or_create_hash_mapping_reuses_instance_for_identical_config(
    tokenizer_path, training_args, monkeypatch
):
    monkeypatch.setattr(engram, "_HASH_MAPPING_CACHE", {})
    arguments = dict(
        engram_vocab_size=[31],
        max_ngram_size=2,
        n_embed_per_ngram=4,
        n_head_per_ngram=1,
        layer_ids=[0],
        tokenizer_name_or_path=tokenizer_path,
        pad_id=None,
        seed=1,
    )
    first = engram.get_or_create_hash_mapping(**arguments)
    second = engram.get_or_create_hash_mapping(**arguments)
    assert first is second
    third = engram.get_or_create_hash_mapping(**dict(arguments, seed=2))
    assert third is not first
    assert len(engram._HASH_MAPPING_CACHE) == 2


def test_ngram_hash_mapping_assigns_distinct_primes_per_head_and_layer(
    tokenizer_path, training_args
):
    mapping = engram.NgramHashMapping(
        engram_vocab_size=[31, 37],
        max_ngram_size=3,
        n_embed_per_ngram=8,
        n_head_per_ngram=1,
        layer_ids=[0, 1],
        tokenizer_name_or_path=tokenizer_path,
        pad_id=4,
        seed=0,
    )
    assert mapping.vocab_size_across_layers == {0: [[31], [37]], 1: [[41], [43]]}
    primes = [
        prime
        for per_layer in mapping.vocab_size_across_layers.values()
        for per_ngram in per_layer
        for prime in per_ngram
    ]
    assert len(set(primes)) == 4
    assert mapping.pad_id == 2
    assert mapping.tokenizer_vocab_size == 8
    for layer_id in (0, 1):
        assert mapping.layer_multipliers[layer_id].shape == (3,)
        assert torch.all(mapping.layer_multipliers[layer_id] % 2 == 1)


def test_ngram_hash_mapping_keeps_pad_id_none_when_unset(tokenizer_path, training_args):
    mapping = engram.NgramHashMapping(
        engram_vocab_size=[31],
        max_ngram_size=2,
        n_embed_per_ngram=4,
        n_head_per_ngram=1,
        layer_ids=[5],
        tokenizer_name_or_path=tokenizer_path,
        pad_id=None,
        seed=3,
    )
    assert mapping.pad_id is None
    assert mapping.layer_ids == [5]
    assert mapping.vocab_size_across_layers == {5: [[31]]}
    assert mapping.max_ngram_size == 2
    assert mapping.n_embed_per_ngram == 4
    assert mapping.n_head_per_ngram == 1


def test_get_ngram_hashes_matches_manual_shift_xor_and_remainder(tokenizer_path, training_args):
    mapping = engram.NgramHashMapping(
        engram_vocab_size=[31, 37],
        max_ngram_size=3,
        n_embed_per_ngram=8,
        n_head_per_ngram=1,
        layer_ids=[0, 1],
        tokenizer_name_or_path=tokenizer_path,
        pad_id=0,
        seed=0,
    )
    input_ids = torch.tensor([[2, 3, 9, 1, 6, 8, 0, 4], [5, 5, 2, 2, 2, 9, 9, 1]])
    hashes = mapping._get_ngram_hashes(input_ids, layer_id=0)
    again = mapping._get_ngram_hashes(input_ids, layer_id=0)
    multipliers = mapping.layer_multipliers[0]
    batch, seq = input_ids.shape
    shifts = [input_ids]
    for k in range(1, mapping.max_ngram_size):
        pad = torch.full((batch, k), mapping.pad_id, dtype=torch.long)
        shifts.append(torch.cat([pad, input_ids], dim=1)[:, :seq])
    heads = []
    for n in range(2, mapping.max_ngram_size + 1):
        mix = shifts[0] * multipliers[0]
        for k in range(1, n):
            mix = torch.bitwise_xor(mix, shifts[k] * multipliers[k])
        for vocab_size in mapping.vocab_size_across_layers[0][n - 2]:
            heads.append(torch.remainder(mix, vocab_size))
    expected = torch.stack(heads, dim=2)
    assert hashes.shape == (2, 8, 2)
    assert hashes.dtype == torch.long
    torch.testing.assert_close(hashes, expected, rtol=0, atol=0)
    torch.testing.assert_close(again, expected, rtol=0, atol=0)


def test_get_ngram_hashes_rejects_missing_input(tokenizer_path, training_args):
    mapping = engram.NgramHashMapping(
        engram_vocab_size=[31],
        max_ngram_size=2,
        n_embed_per_ngram=4,
        n_head_per_ngram=1,
        layer_ids=[0],
        tokenizer_name_or_path=tokenizer_path,
        pad_id=None,
        seed=0,
    )
    with pytest.raises(AssertionError, match="input_ids can not be None in NgramHashMapping"):
        mapping._get_ngram_hashes(None, layer_id=0)


def test_hash_compresses_ids_then_hashes_for_every_layer(tokenizer_path, training_args):
    mapping = engram.NgramHashMapping(
        engram_vocab_size=[31, 37],
        max_ngram_size=3,
        n_embed_per_ngram=8,
        n_head_per_ngram=1,
        layer_ids=[0, 1],
        tokenizer_name_or_path=tokenizer_path,
        pad_id=0,
        seed=0,
    )
    hashes = mapping.hash(torch.tensor([[2, 3], [4, 3]]))
    assert sorted(hashes) == [0, 1]
    layer_zero = hashes[0]
    assert layer_zero.shape == (2, 2, 2)
    assert torch.equal(layer_zero[0], layer_zero[1])
