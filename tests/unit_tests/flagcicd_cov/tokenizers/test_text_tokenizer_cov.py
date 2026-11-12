# Copyright (c) 2026, BAAI. All rights reserved.
"""MegatronTokenizerText wrapper: template resolution, library delegation, save/restore."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from megatron.core.tokenizers import MegatronTokenizer

CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"


@pytest.fixture
def wordlevel_tokenizer_dir(tmp_path):
    directory = tmp_path / 'wordlevel'
    directory.mkdir()
    vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3, 'hello': 4, 'world': 5}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    dollar = chr(36)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f'<s> {dollar}A </s>',
        pair=f'{dollar}A {dollar}B',
        special_tokens=[('<s>', 2), ('</s>', 3)],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token='<unk>',
        pad_token='<pad>',
        bos_token='<s>',
        eos_token='</s>',
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(str(directory))
    return str(directory)


@pytest.fixture
def bare_tokenizer_dir(tmp_path):
    directory = tmp_path / 'bare'
    directory.mkdir()
    tokenizer = Tokenizer(models.WordLevel({'<unk>': 0, 'a': 1}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>').save_pretrained(
        str(directory)
    )
    return str(directory)


def _hf_wrapper(path, **kwargs):
    return MegatronTokenizer.from_pretrained(
        tokenizer_path=path, metadata_path={'library': 'huggingface'}, **kwargs
    )


@pytest.mark.parametrize(
    'metadata,chat_template,expected',
    [
        ({'library': 'huggingface', 'chat_template': 'FROM_CONFIG'}, 'KW', 'FROM_CONFIG'),
        ({'library': 'huggingface'}, 'KW', 'KW'),
        ({'library': 'huggingface'}, None, CHAT_TEMPLATE),
    ],
    ids=['config-wins', 'kwargs-wins', 'tokenizer-default'],
)
def test_wrapper_chat_template_prefers_config_then_kwargs_then_tokenizer(
    wordlevel_tokenizer_dir, metadata, chat_template, expected
):
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=wordlevel_tokenizer_dir, metadata_path=metadata, chat_template=chat_template
    )
    assert tokenizer.chat_template == expected


def test_wrapper_tokenize_includes_special_tokens_by_default(wordlevel_tokenizer_dir):
    tokenizer = _hf_wrapper(wordlevel_tokenizer_dir)
    assert tokenizer.tokenize('hello') == [2, 4, 3]


@pytest.mark.parametrize(
    'skip_special_tokens,expected',
    [(True, 'hello'), (False, '<s> hello </s>')],
    ids=['stripped', 'kept'],
)
def test_wrapper_detokenize_strips_special_tokens_when_requested(
    wordlevel_tokenizer_dir, skip_special_tokens, expected
):
    tokenizer = _hf_wrapper(wordlevel_tokenizer_dir)
    assert tokenizer.detokenize([2, 4, 3], skip_special_tokens=skip_special_tokens) == expected


def test_wrapper_detokenize_ignores_skip_special_tokens_for_libraries_without_it():
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={'library': 'null-text'}, vocab_size=9
    )
    assert tokenizer.detokenize([1, 20], skip_special_tokens=True) == '1 20'


def test_wrapper_apply_chat_template_uses_the_stored_template(wordlevel_tokenizer_dir):
    tokenizer = _hf_wrapper(wordlevel_tokenizer_dir)
    conversation = [{'role': 'user', 'content': 'hello'}]
    assert tokenizer.apply_chat_template(conversation, tokenize=False) == 'hello'
    assert tokenizer.apply_chat_template(conversation) == [4]


def test_wrapper_apply_chat_template_requires_a_stored_template(bare_tokenizer_dir):
    tokenizer = _hf_wrapper(bare_tokenizer_dir)
    with pytest.raises(AssertionError, match='`chat_template` was not specified'):
        tokenizer.apply_chat_template([{'role': 'user', 'content': 'a'}])


def test_wrapper_tokenize_conversation_requires_the_sft_library(wordlevel_tokenizer_dir):
    tokenizer = _hf_wrapper(wordlevel_tokenizer_dir)
    with pytest.raises(NotImplementedError, match='supported only for SFTTokenizer'):
        tokenizer.tokenize_conversation([], return_target=False, add_generation_prompt=False)


def test_wrapper_save_pretrained_round_trips_the_hf_tokenizer(wordlevel_tokenizer_dir, tmp_path):
    tokenizer = _hf_wrapper(wordlevel_tokenizer_dir)
    out = tmp_path / 'saved'
    out.mkdir()
    tokenizer.save_pretrained(str(out))
    reloaded = _hf_wrapper(str(out))
    assert reloaded.tokenize('hello') == [2, 4, 3]


def test_wrapper_save_pretrained_rejects_non_hf_libraries(tmp_path):
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={'library': 'byte-level'}, vocab_size=300
    )
    with pytest.raises(ValueError, match='not supported with byte-level library'):
        tokenizer.save_pretrained(str(tmp_path / 'unused'))


def test_wrapper_add_special_tokens_extends_the_underlying_vocabulary(wordlevel_tokenizer_dir):
    tokenizer = _hf_wrapper(wordlevel_tokenizer_dir)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<new>']})
    assert tokenizer.vocab[6] == '<new>'
    assert tokenizer.vocab_size == 7


def test_wrapper_offsets_delegate_to_the_underlying_tokenizer():
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={'library': 'null-text'}, vocab_size=9
    )
    assert tokenizer.offsets([1, 20], '1 20') == [0, 2]


def test_wrapper_space_sensitive_exposes_the_underlying_flag(wordlevel_tokenizer_dir):
    tokenizer = _hf_wrapper(wordlevel_tokenizer_dir)
    assert tokenizer._tokenizer.text_to_tokens('x y') == tokenizer._tokenizer.text_to_tokens(
        'x'
    ) + tokenizer._tokenizer.text_to_tokens('y')
    assert tokenizer.space_sensitive is False


def test_wrapper_space_sensitive_rejects_unsupported_libraries():
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={'library': 'byte-level'}, vocab_size=300
    )
    with pytest.raises(NotImplementedError, match='not supported for byte-level tokenizers'):
        tokenizer.space_sensitive


def test_wrapper_additional_special_tokens_ids_delegate_to_the_library(wordlevel_tokenizer_dir):
    tokenizer = _hf_wrapper(wordlevel_tokenizer_dir, additional_special_tokens=['<extra1>'])
    assert tokenizer.additional_special_tokens_ids == [6]


def test_wrapper_exposes_the_underlying_special_token_ids(wordlevel_tokenizer_dir):
    tokenizer = _hf_wrapper(wordlevel_tokenizer_dir)
    assert (
        tokenizer.pad,
        tokenizer.pad_id,
        tokenizer.bos,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.eod,
        tokenizer.unk,
        tokenizer.unk_id,
        tokenizer.cls,
        tokenizer.cls_id,
        tokenizer.sep,
        tokenizer.sep_id,
    ) == (0, 0, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3)
    assert tokenizer.mask_id is None
    assert tokenizer.mask is None


def test_wrapper_exposes_the_underlying_vocab_and_file_arguments(wordlevel_tokenizer_dir):
    tokenizer = _hf_wrapper(
        wordlevel_tokenizer_dir, vocab_file='vocab.json', merges_file='merges.txt'
    )
    assert tokenizer.vocab == ['<pad>', '<unk>', '<s>', '</s>', 'hello', 'world']
    assert tokenizer.inv_vocab[4] == 'hello'
    assert tokenizer.vocab_size == 6
    assert tokenizer.vocab_file == 'vocab.json'
    assert tokenizer.merges_file == 'merges.txt'
