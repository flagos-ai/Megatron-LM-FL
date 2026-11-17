# Copyright (c) 2026, BAAI. All rights reserved.
"""HuggingFaceTokenizer: local-directory loading, special-token wiring, id/text conversion."""

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast

from megatron.core.tokenizers.text.libraries.huggingface_tokenizer import HuggingFaceTokenizer

CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
GPT2_VOCAB = {
    '<pad>': 0,
    '<unk>': 1,
    '<s>': 2,
    '</s>': 3,
    'hello': 4,
    'world': 5,
    'h': 6,
    'e': 7,
    'l': 8,
    'o': 9,
    'w': 10,
    'r': 11,
    'd': 12,
    'he': 13,
    'llo': 14,
    'or': 15,
    'wor': 16,
    'ld': 17,
}
GPT2_MERGES = '#version: 0.2\nh e\nhe llo\nw or\nwor ld\n'


def _save_wordlevel(directory, special_kwargs):
    vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, 'hello': 4, 'world': 5}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    dollar = chr(36)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f'<s> {dollar}A </s>',
        pair=f'{dollar}A {dollar}B',
        special_tokens=[('<s>', 2), ('</s>', 3)],
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>', **special_kwargs)
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(str(directory))
    return str(directory)


@pytest.fixture
def wordlevel_tokenizer_dir(tmp_path):
    directory = tmp_path / 'wordlevel'
    directory.mkdir()
    (directory / 'vocab.json').write_text(json.dumps(GPT2_VOCAB))
    (directory / 'merges.txt').write_text(GPT2_MERGES)
    return _save_wordlevel(directory, dict(pad_token='<pad>', bos_token='<s>', eos_token='</s>'))


@pytest.fixture
def cls_sep_only_tokenizer_dir(tmp_path):
    directory = tmp_path / 'cls_sep_only'
    directory.mkdir()
    vocab = {'<unk>': 0, '<cls9>': 1, '<sep9>': 2, 'a': 3}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token='<unk>', cls_token='<cls9>', sep_token='<sep9>'
    )
    fast.save_pretrained(str(directory))
    return str(directory)


@pytest.fixture
def unk_only_tokenizer_dir(tmp_path):
    directory = tmp_path / 'unk_only'
    directory.mkdir()
    tokenizer = Tokenizer(models.WordLevel({'<unk>': 0, 'a': 1}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>').save_pretrained(
        str(directory)
    )
    return str(directory)


@pytest.fixture
def gpt2_tokenizer_dir(tmp_path):
    directory = tmp_path / 'gpt2'
    directory.mkdir()
    (directory / 'vocab.json').write_text(json.dumps(GPT2_VOCAB))
    (directory / 'merges.txt').write_text(GPT2_MERGES)
    slow = GPT2TokenizerFast(
        vocab_file=str(directory / 'vocab.json'),
        merges_file=str(directory / 'merges.txt'),
        unk_token='<unk>',
        pad_token='<pad>',
        bos_token='<s>',
        eos_token='</s>',
    )
    saved = tmp_path / 'gpt2_saved'
    slow.save_pretrained(str(saved))
    return str(saved)


@pytest.fixture
def no_special_tokens_dir(tmp_path):
    directory = tmp_path / 'no_special'
    directory.mkdir()
    tokenizer = Tokenizer(models.WordLevel({'<unk>': 0, 'a': 1}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer).save_pretrained(str(directory))
    return str(directory)


def test_init_raises_value_error_when_the_tokenizer_cannot_be_loaded():
    with pytest.raises(ValueError, match='Unable to instantiate HuggingFace AutoTokenizer'):
        HuggingFaceTokenizer('/nonexistent/tokenizer/path')


@pytest.mark.parametrize('merges_file', [None, 'merges.txt'], ids=['vocab-only', 'both'])
def test_init_accepts_explicit_vocab_and_merges_arguments(wordlevel_tokenizer_dir, merges_file):
    vocab_file = wordlevel_tokenizer_dir + '/vocab.json'
    if merges_file is None:
        tokenizer = HuggingFaceTokenizer(wordlevel_tokenizer_dir, vocab_file=vocab_file)
    else:
        tokenizer = HuggingFaceTokenizer(
            wordlevel_tokenizer_dir, vocab_file=vocab_file, merges_file=merges_file
        )
    assert tokenizer.vocab_size == 6
    assert tokenizer.text_to_ids('hello') == [2, 4, 3]


def test_init_applies_the_special_tokens_given_in_the_constructor(wordlevel_tokenizer_dir):
    tokenizer = HuggingFaceTokenizer(
        wordlevel_tokenizer_dir,
        unk_token='<unk2>',
        mask_token='<mask2>',
        pad_token='<pad2>',
        sep_token='<sep2>',
        eos_token='<eos2>',
        bos_token='<bos2>',
        cls_token='<cls2>',
    )
    assert (tokenizer.unk_id, tokenizer.mask_id, tokenizer.pad_id) == (6, 7, 8)
    assert (tokenizer.sep_id, tokenizer.eos_id, tokenizer.bos_id, tokenizer.cls_id) == (
        9,
        10,
        11,
        12,
    )


def test_init_aliases_bos_and_eos_to_cls_and_sep_when_only_the_latter_exist(
    cls_sep_only_tokenizer_dir,
):
    tokenizer = HuggingFaceTokenizer(cls_sep_only_tokenizer_dir)
    assert (tokenizer.bos_id, tokenizer.cls_id, tokenizer.eos_id, tokenizer.sep_id) == (1, 1, 2, 2)


def test_text_conversions_round_trip_vocabulary_words(wordlevel_tokenizer_dir):
    tokenizer = HuggingFaceTokenizer(wordlevel_tokenizer_dir)
    assert tokenizer.text_to_tokens('hello world') == ['hello', 'world']
    assert tokenizer.tokens_to_text(['hello', 'world']) == 'hello world'
    assert tokenizer.token_to_id('hello') == 4
    assert tokenizer.tokens_to_ids(['hello', '</s>']) == [4, 3]
    assert tokenizer.ids_to_tokens([4, 3]) == ['hello', '</s>']


@pytest.mark.parametrize(
    'include_special_tokens,expected',
    [(True, [2, 4, 3]), (False, [4])],
    ids=['included', 'excluded'],
)
def test_text_to_ids_adds_special_tokens_only_when_requested(
    wordlevel_tokenizer_dir, include_special_tokens, expected
):
    tokenizer = HuggingFaceTokenizer(
        wordlevel_tokenizer_dir, include_special_tokens=include_special_tokens
    )
    assert tokenizer.text_to_ids('hello') == expected


@pytest.mark.parametrize(
    'include_special_tokens,remove_special_tokens,expected',
    [(True, False, '<s> hello </s>'), (True, True, 'hello'), (False, None, 'hello')],
    ids=['keep-explicit', 'strip-explicit', 'strip-default'],
)
def test_ids_to_text_strips_special_tokens_only_when_requested(
    wordlevel_tokenizer_dir, include_special_tokens, remove_special_tokens, expected
):
    tokenizer = HuggingFaceTokenizer(
        wordlevel_tokenizer_dir, include_special_tokens=include_special_tokens
    )
    text = tokenizer.ids_to_text([2, 4, 3], remove_special_tokens=remove_special_tokens)
    assert text == expected


def test_special_token_ids_alias_missing_tokens_to_their_counterparts(wordlevel_tokenizer_dir):
    tokenizer = HuggingFaceTokenizer(wordlevel_tokenizer_dir)
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 2
    assert tokenizer.eos_id == 3
    assert tokenizer.eod == 3
    assert tokenizer.sep_id == 3
    assert tokenizer.cls_id == 2
    assert tokenizer.unk_id == 1
    assert tokenizer.mask_id is None


def test_special_token_ids_return_none_when_the_tokenizer_has_none(unk_only_tokenizer_dir):
    tokenizer = HuggingFaceTokenizer(unk_only_tokenizer_dir)
    assert tokenizer.tokenizer.pad_token is None
    assert tokenizer.unk_id == 0
    assert (
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eod,
        tokenizer.sep_id,
        tokenizer.cls_id,
        tokenizer.mask_id,
    ) == (None, None, None, None, None, None)


def test_unk_id_returns_none_when_the_tokenizer_has_no_unk_token(no_special_tokens_dir):
    tokenizer = HuggingFaceTokenizer(no_special_tokens_dir)
    assert tokenizer.tokenizer.unk_token is None
    assert tokenizer.unk_id is None


def test_vocab_and_inv_vocab_are_ordered_by_id_and_cached(wordlevel_tokenizer_dir):
    tokenizer = HuggingFaceTokenizer(wordlevel_tokenizer_dir)
    assert tokenizer.vocab == ['<pad>', '<unk>', '<s>', '</s>', 'hello', 'world']
    assert tokenizer.inv_vocab[4] == 'hello'
    assert tokenizer.inv_vocab is tokenizer.inv_vocab
    assert tokenizer.vocab_size == 6


def test_additional_special_tokens_ids_convert_the_constructor_tokens(wordlevel_tokenizer_dir):
    with_additional = HuggingFaceTokenizer(
        wordlevel_tokenizer_dir, additional_special_tokens=['<extra1>']
    )
    assert with_additional.additional_special_tokens_ids == [6]
    without_additional = HuggingFaceTokenizer(wordlevel_tokenizer_dir)
    assert without_additional.additional_special_tokens_ids == []


def test_add_special_tokens_returns_the_number_of_tokens_actually_added(wordlevel_tokenizer_dir):
    tokenizer = HuggingFaceTokenizer(wordlevel_tokenizer_dir)
    assert tokenizer.add_special_tokens({'additional_special_tokens': ['<extra2>']}) == 1
    assert tokenizer.add_special_tokens({'additional_special_tokens': ['<extra2>']}) == 0
    assert tokenizer.tokenizer.get_vocab()['<extra2>'] == 6


def test_add_special_tokens_keeps_the_constructor_list_for_additional_ids(wordlevel_tokenizer_dir):
    tokenizer = HuggingFaceTokenizer(
        wordlevel_tokenizer_dir, additional_special_tokens=['<extra1>']
    )
    assert tokenizer.add_special_tokens({'additional_special_tokens': ['<extra2>']}) == 1
    assert tokenizer.additional_special_tokens_ids == [6]


def test_new_special_tokens_extend_the_vocabulary_when_missing(wordlevel_tokenizer_dir):
    tokenizer = HuggingFaceTokenizer(wordlevel_tokenizer_dir, mask_token='<maskx>')
    assert tokenizer.mask_id == 6
    assert tokenizer.vocab_size == 7


def test_space_sensitive_is_false_when_whitespace_is_not_encoded(wordlevel_tokenizer_dir):
    tokenizer = HuggingFaceTokenizer(wordlevel_tokenizer_dir)
    assert tokenizer.text_to_tokens('x y') == tokenizer.text_to_tokens('x') + (
        tokenizer.text_to_tokens('y')
    )
    assert tokenizer.space_sensitive is False


def test_apply_chat_template_delegates_to_the_underlying_tokenizer(wordlevel_tokenizer_dir):
    tokenizer = HuggingFaceTokenizer(wordlevel_tokenizer_dir)
    conversation = [{'role': 'user', 'content': 'hello'}]
    rendered = tokenizer.apply_chat_template(
        conversation=conversation, chat_template=tokenizer.chat_template, tokenize=False
    )
    assert rendered == 'hello'


def test_save_vocabulary_writes_vocab_and_merges_files(gpt2_tokenizer_dir, tmp_path):
    tokenizer = HuggingFaceTokenizer(gpt2_tokenizer_dir)
    out = tmp_path / 'legacy'
    out.mkdir()
    vocab_files = tokenizer.save_vocabulary(str(out))
    assert [Path(part).name for part in vocab_files] == ['vocab.json', 'merges.txt']
    assert all(Path(part).is_file() for part in vocab_files)


def test_save_pretrained_writes_the_tokenizer_directory(gpt2_tokenizer_dir, tmp_path):
    tokenizer = HuggingFaceTokenizer(gpt2_tokenizer_dir)
    out = tmp_path / 'pretrained'
    out.mkdir()
    saved = tokenizer.save_pretrained(str(out))
    assert (out / 'tokenizer.json').is_file()
    assert sorted(Path(path).name for path in saved) == [
        'added_tokens.json',
        'merges.txt',
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json',
    ]
    assert (out / 'vocab.json').is_file() and (out / 'merges.txt').is_file()
