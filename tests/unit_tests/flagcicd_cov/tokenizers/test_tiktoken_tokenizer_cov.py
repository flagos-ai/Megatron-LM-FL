# Copyright (c) 2026, BAAI. All rights reserved.
"""TikTokenTokenizer: rank loading with the special shift, byte encoding, ids, offsets."""

import base64
import json

import pytest

from megatron.core.tokenizers.text.libraries.tiktoken_tokenizer import (
    SPECIAL_TOKENS,
    TikTokenTokenizer,
    reload_mergeable_ranks,
)

SPECIAL_TOKENS_UNDER_TEST = [
    '<unk>',
    '<s>',
    '</s>',
    '<mask>',
    '<pad>',
    '<cls>',
    '<sep>',
    '<custom>',
]
NUM_SPECIAL = 8


@pytest.fixture
def tiktoken_vocab_file(tmp_path):
    entries = [
        {
            'rank': rank,
            'token_bytes': base64.b64encode(bytes([rank])).decode(),
            'token_str': chr(rank) if 32 <= rank < 127 else '',
        }
        for rank in range(256)
    ]
    for rank, merge in enumerate([b'ab', b'hello', b'world']):
        entries.append(
            {
                'rank': 256 + rank,
                'token_bytes': base64.b64encode(merge).decode(),
                'token_str': merge.decode(),
            }
        )
    path = tmp_path / 'tiny.vocab.json'
    path.write_text(json.dumps(entries))
    return str(path)


@pytest.fixture
def tiktoken_tokenizer(tiktoken_vocab_file):
    return TikTokenTokenizer(
        tiktoken_vocab_file,
        special_tokens=SPECIAL_TOKENS_UNDER_TEST,
        num_special_tokens=NUM_SPECIAL,
        vocab_size=NUM_SPECIAL + 259,
    )


def test_reload_mergeable_ranks_rejects_paths_without_a_json_suffix(tiktoken_vocab_file):
    with pytest.raises(AssertionError):
        reload_mergeable_ranks(tiktoken_vocab_file[:-5] + '.txt', num_special_tokens=8)


def test_reload_mergeable_ranks_shifts_ranks_by_num_special_tokens(tiktoken_vocab_file):
    ranks = reload_mergeable_ranks(tiktoken_vocab_file, num_special_tokens=NUM_SPECIAL)
    assert len(ranks) == 259
    assert ranks[b'h'] == 104 + NUM_SPECIAL
    assert ranks[b'ab'] == 256 + NUM_SPECIAL
    assert set(ranks.values()) == set(range(NUM_SPECIAL, NUM_SPECIAL + 259))


def test_reload_mergeable_ranks_cuts_vocabulary_to_max_vocab(tiktoken_vocab_file):
    ranks = reload_mergeable_ranks(
        tiktoken_vocab_file, max_vocab=258, num_special_tokens=NUM_SPECIAL
    )
    assert set(ranks.values()) == set(range(NUM_SPECIAL, NUM_SPECIAL + 258))
    assert ranks[b'ab'] == 256 + NUM_SPECIAL
    assert ranks[b'hello'] == 257 + NUM_SPECIAL
    assert b'world' not in ranks


def test_init_raises_value_error_when_path_is_invalid():
    with pytest.raises(ValueError, match='tokenizer_path: /missing/vocab.json is invalid'):
        TikTokenTokenizer('/missing/vocab.json')


def test_init_raises_value_error_for_unknown_pattern(tiktoken_vocab_file):
    with pytest.raises(ValueError, match='Expected tiktoken pattern to be .v1. or .v2.'):
        TikTokenTokenizer(tiktoken_vocab_file, pattern='v3')


@pytest.mark.parametrize('pattern', ['v1', 'v2'], ids=['v1', 'v2'])
def test_init_accepts_both_split_patterns(tiktoken_vocab_file, pattern):
    tokenizer = TikTokenTokenizer(
        tiktoken_vocab_file,
        special_tokens=SPECIAL_TOKENS_UNDER_TEST,
        num_special_tokens=NUM_SPECIAL,
        vocab_size=NUM_SPECIAL + 259,
        pattern=pattern,
    )
    assert tokenizer.text_to_ids('hi') == [104 + NUM_SPECIAL, 105 + NUM_SPECIAL]


def test_init_uses_default_special_tokens_when_none_is_given(tiktoken_vocab_file):
    tokenizer = TikTokenTokenizer(tiktoken_vocab_file, num_special_tokens=7, vocab_size=7 + 259)
    assert tokenizer.special_tokens == SPECIAL_TOKENS
    assert (tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id) == (0, 1, 2)


def test_init_fills_special_tokens_up_to_num_special_tokens(tiktoken_vocab_file):
    tokenizer = TikTokenTokenizer(
        tiktoken_vocab_file, num_special_tokens=1000, vocab_size=1000 + 258
    )
    assert tokenizer.special_filler[0] == '<SPECIAL_7>'
    assert tokenizer.special_filler[-1] == '<SPECIAL_999>'
    assert len(tokenizer.special_tokens) == 1000
    assert tokenizer.vocab_size == 1258


def test_text_to_ids_encodes_bytes_and_special_tokens(tiktoken_tokenizer):
    assert tiktoken_tokenizer.text_to_ids('hi') == [112, 113]
    assert tiktoken_tokenizer.text_to_ids('<unk>I</s><custom>') == [0, 81, 2, 7]


def test_text_to_tokens_and_tokens_to_text_use_token_bytes(tiktoken_tokenizer):
    assert tiktoken_tokenizer.text_to_tokens('hi') == [b'h', b'i']
    assert tiktoken_tokenizer.tokens_to_text([b'h', b'i']) == 'hi'


def test_token_to_id_returns_the_index_of_special_tokens(tiktoken_tokenizer):
    assert tiktoken_tokenizer.token_to_id('<unk>') == 0
    assert tiktoken_tokenizer.token_to_id('<custom>') == 7


def test_ids_to_tokens_maps_special_and_merge_ids(tiktoken_tokenizer):
    assert tiktoken_tokenizer.id_to_token(0) == '<unk>'
    assert tiktoken_tokenizer.id_to_token(256 + NUM_SPECIAL) == 'ab'
    assert tiktoken_tokenizer.ids_to_tokens([0, 256 + NUM_SPECIAL]) == ['<unk>', 'ab']


def test_ids_to_text_keeps_special_tokens_by_default(tiktoken_tokenizer):
    assert tiktoken_tokenizer.ids_to_text([0, 113, 2]) == '<unk>i</s>'


def test_ids_to_text_strips_bos_eos_and_special_ids_when_requested(tiktoken_tokenizer):
    assert tiktoken_tokenizer.ids_to_text([0, 113, 2], remove_special_tokens=True) == 'i'
    assert tiktoken_tokenizer.ids_to_text([0, 2], remove_special_tokens=True) == ''


def test_offsets_returns_character_offsets(tiktoken_tokenizer):
    ascii_ids = tiktoken_tokenizer.text_to_ids('hi')
    assert tiktoken_tokenizer.offsets(ascii_ids, 'hi') == [0, 1]
    multi_byte_ids = [0xC3 + NUM_SPECIAL, 0xA9 + NUM_SPECIAL]
    assert tiktoken_tokenizer.offsets(multi_byte_ids, '\u00e9') == [0, 0]


def test_offsets_falls_back_when_bytes_are_not_valid_utf8(tiktoken_tokenizer):
    assert tiktoken_tokenizer.offsets([0xFF + NUM_SPECIAL], '\u00ff') == [0]


def test_additional_special_tokens_ids_exclude_unk_bos_eos_and_filler(tiktoken_tokenizer):
    assert tiktoken_tokenizer.additional_special_tokens_ids == [3, 4, 5, 6, 7]


def test_property_accessors_return_special_ids_and_vocabs(tiktoken_tokenizer):
    assert (tiktoken_tokenizer.bos_id, tiktoken_tokenizer.eos_id) == (1, 2)
    assert (tiktoken_tokenizer.eod, tiktoken_tokenizer.unk_id) == (2, 0)
    assert (tiktoken_tokenizer.mask_id, tiktoken_tokenizer.pad_id) == (3, 4)
    assert (tiktoken_tokenizer.cls_id, tiktoken_tokenizer.sep_id) == (5, 6)
    assert tiktoken_tokenizer.vocab_size == 267
    assert tiktoken_tokenizer.vocab is tiktoken_tokenizer.encoder
    assert tiktoken_tokenizer.decoder is tiktoken_tokenizer.inv_vocab
    assert tiktoken_tokenizer.vocab['<unk>'] == 0
    assert tiktoken_tokenizer.decoder[256 + 2 * NUM_SPECIAL] == b'ab'


def test_add_special_tokens_is_not_supported(tiktoken_tokenizer):
    with pytest.raises(NotImplementedError, match='not supported for TikToken'):
        tiktoken_tokenizer.add_special_tokens({'bos_token': '<new>'})
