# Copyright (c) 2026, BAAI. All rights reserved.
"""build_tokenizer: tokenizer_type dispatch, argument plumbing, padded vocab size."""

import base64
import json
from types import SimpleNamespace

import pytest
import sentencepiece
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

import megatron.core.tokenizers.text.libraries.megatron_hf_tokenizer as megatron_hf
import megatron.core.tokenizers.text.models as text_models
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer, vocab_size_with_padding

CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
TIKTOKEN_SPECIAL_TOKENS = ['<unk>', '<s>', '</s>', '<mask>', '<pad>', '<cls>', '<sep>', '<custom>']


@pytest.fixture
def sentencepiece_model(tmp_path):
    prefix = str(tmp_path / 'tiny')
    sentences = [
        'hello world',
        'hello there',
        'world of tokenizers',
        'special words appear here',
        'a second sentence for training',
        'unicode caf\u00e9 works',
    ]
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_prefix=prefix,
        vocab_size=40,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        hard_vocab_limit=False,
    )
    return prefix + '.model'


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
def megatron_hf_local_configs(monkeypatch, wordlevel_tokenizer_dir):
    monkeypatch.setitem(
        megatron_hf.MEGATRON_CONFIG_MAP,
        'GPT2BPETokenizer',
        {'vocab': None, 'merges_file': None, 'tokenizer_name': wordlevel_tokenizer_dir},
    )
    monkeypatch.setitem(
        megatron_hf.MEGATRON_CONFIG_MAP,
        'BertWordPieceCase',
        {'vocab': None, 'tokenizer_name': wordlevel_tokenizer_dir},
    )


def _args(tokenizer_type, **overrides):
    values = dict(
        tokenizer_type=tokenizer_type,
        special_tokens=None,
        vocab_file=None,
        merge_file=None,
        tokenizer_hf_no_use_fast=False,
        trust_remote_code=False,
        tokenizer_hf_no_include_special_tokens=False,
        tokenizer_model=None,
        chat_template=None,
        tokenizer_sentencepiece_ignore_extra_whitespaces=True,
        tokenizer_sentencepiece_legacy=False,
        tiktoken_pattern=None,
        vocab_size=None,
        tiktoken_num_special_tokens=8,
        tokenizer_prompt_format='chatml',
        image_tag_type='',
        force_system_message=False,
        sft_tokenizer_prompt_format='identity',
        metadata_path=None,
        pad_vocab_size=False,
        make_vocab_size_divisible_by=8,
        tensor_model_parallel_size=1,
        rank=0,
        padded_vocab_size=None,
        null_tokenizer_eod_id=None,
        null_tokenizer_pad_id=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_tokenizer_rejects_unsupported_types():
    with pytest.raises(ValueError, match='tokenizer_type bogus is not supported'):
        build_tokenizer(_args('bogus'))


def test_build_tokenizer_builds_sentencepiece_with_the_args_flags(sentencepiece_model):
    tokenizer = build_tokenizer(
        _args('SentencePieceTokenizer', tokenizer_model=sentencepiece_model)
    )
    assert tokenizer.library == 'sentencepiece'
    assert tokenizer.vocab_size == 40
    assert tokenizer._tokenizer.legacy is False
    assert tokenizer._tokenizer.ignore_extra_whitespaces is True
    assert tokenizer.chat_template is None


def test_build_tokenizer_builds_tiktoken_with_the_requested_pattern(tiktoken_vocab_file):
    args = _args(
        'TikTokenizer',
        tokenizer_model=tiktoken_vocab_file,
        tiktoken_pattern='v1',
        vocab_size=8 + 259,
        tiktoken_num_special_tokens=8,
        special_tokens=TIKTOKEN_SPECIAL_TOKENS,
    )
    tokenizer = build_tokenizer(args)
    assert tokenizer.library == 'tiktoken'
    assert tokenizer.vocab_size == 267
    assert tokenizer.tokenize('hi') == [8 + 104, 8 + 105]


def test_build_tokenizer_builds_huggingface_from_the_local_directory(wordlevel_tokenizer_dir):
    tokenizer = build_tokenizer(
        _args('HuggingFaceTokenizer', tokenizer_model=wordlevel_tokenizer_dir)
    )
    assert tokenizer.library == 'huggingface'
    assert tokenizer.tokenize('hello') == [2, 4, 3]
    assert tokenizer.chat_template == CHAT_TEMPLATE
    assert tokenizer._tokenizer.include_special_tokens is True


def test_build_tokenizer_pads_the_vocabulary_on_the_general_path(wordlevel_tokenizer_dir):
    args = _args(
        'HuggingFaceTokenizer', tokenizer_model=wordlevel_tokenizer_dir, pad_vocab_size=True
    )
    tokenizer = build_tokenizer(args)
    assert tokenizer.vocab_size == 6
    assert args.padded_vocab_size == 8


def test_build_tokenizer_uses_the_metadata_path_when_given(wordlevel_tokenizer_dir):
    tokenizer = build_tokenizer(
        _args(
            'HuggingFaceTokenizer',
            tokenizer_model=wordlevel_tokenizer_dir,
            metadata_path={'library': 'huggingface', 'model_type': 'gpt'},
        )
    )
    assert tokenizer.library == 'huggingface'
    assert tokenizer.model_type == 'gpt'


def test_build_tokenizer_builds_sft_tokenizers(wordlevel_tokenizer_dir):
    tokenizer = build_tokenizer(_args('SFTTokenizer', tokenizer_model=wordlevel_tokenizer_dir))
    assert tokenizer.library == 'sft'
    assert tokenizer.tokenize_conversation(
        [{'role': 'user', 'content': 'hello'}], return_target=False, add_generation_prompt=False
    ).tolist() == [4]


@pytest.mark.parametrize(
    'tokenizer_type,vocab_size,expected_vocab_size,expected_eod',
    [('NullTokenizer', 31, 31, 29), ('NullMultimodalTokenizer', 11, 12, 11)],
    ids=['null-text', 'null-multimodal'],
)
def test_build_tokenizer_builds_null_tokenizers(
    tokenizer_type, vocab_size, expected_vocab_size, expected_eod
):
    tokenizer = build_tokenizer(
        _args(tokenizer_type, vocab_size=vocab_size, null_tokenizer_eod_id=29)
    )
    assert tokenizer.vocab_size == expected_vocab_size
    assert tokenizer.eod == expected_eod


def test_build_tokenizer_applies_the_null_eod_and_pad_ids():
    tokenizer = build_tokenizer(
        _args('NullTokenizer', vocab_size=31, null_tokenizer_eod_id=29, null_tokenizer_pad_id=3)
    )
    assert (tokenizer.vocab_size, tokenizer.eod, tokenizer.pad_id) == (31, 29, 3)


def test_build_tokenizer_pads_the_vocabulary_when_requested():
    args = _args('NullTokenizer', vocab_size=31, pad_vocab_size=True)
    tokenizer = build_tokenizer(args)
    assert tokenizer.vocab_size == 31
    assert args.padded_vocab_size == 32


@pytest.mark.parametrize(
    'orig,expected,tp',
    [(31, 32, 1), (33, 40, 1), (17, 32, 2), (16, 16, 1)],
    ids=['round-up', 'next-multiple', 'tensor-parallel', 'already-aligned'],
)
def test_vocab_size_with_padding_rounds_up_to_the_model_parallel_multiple(orig, expected, tp):
    args = _args('NullTokenizer', tensor_model_parallel_size=tp)
    assert vocab_size_with_padding(orig, args, logging_enabled=False) == expected


def test_build_tokenizer_builds_megatron_gpt_from_the_config_map(megatron_hf_local_configs):
    tokenizer = build_tokenizer(_args('GPT2BPETokenizer'))
    assert tokenizer.library == 'megatron'
    assert isinstance(tokenizer, text_models.DefaultTokenizerText)
    assert tokenizer.vocab_size == 6
    assert tokenizer.additional_special_tokens_ids == []
    assert tokenizer.chat_template == CHAT_TEMPLATE


def test_build_tokenizer_adds_extra_ids_for_bert_word_piece_case(megatron_hf_local_configs):
    tokenizer = build_tokenizer(_args('BertWordPieceCase'))
    assert tokenizer.library == 'megatron'
    assert tokenizer.vocab_size == 106
    assert tokenizer.additional_special_tokens_ids == list(range(6, 106))
