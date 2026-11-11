# Copyright (c) 2026, BAAI. All rights reserved.
"""SentencePieceTokenizer: legacy special tokens, whitespace handling, id/text round trips."""

import numpy as np
import pytest
import sentencepiece
import torch

from megatron.core.tokenizers.text.libraries.sentencepiece_tokenizer import SentencePieceTokenizer


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


def test_init_raises_value_error_when_model_path_does_not_exist():
    with pytest.raises(ValueError, match='tokenizer_path: /missing/model is invalid'):
        SentencePieceTokenizer('/missing/model')


def test_init_raises_value_error_when_special_tokens_given_without_legacy(sentencepiece_model):
    with pytest.raises(ValueError, match='Special tokens must be None when legacy'):
        SentencePieceTokenizer(sentencepiece_model, special_tokens=['<extra>'])


def test_init_reads_vocab_and_special_ids_from_the_model(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model)
    assert tokenizer.vocab_size == tokenizer.original_vocab_size == 40
    assert tokenizer.removed_extra_spaces is True
    assert tokenizer.space_sensitive is False
    assert tokenizer.legacy is False
    assert (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id) == (0, 1, 2)
    assert tokenizer.eod == tokenizer.eos_id
    assert tokenizer.unk_id == tokenizer.tokenizer.unk_id() == 3
    assert tokenizer.special_token_to_id == {}


def test_text_to_tokens_returns_pieces_and_ids_round_trip(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model)
    pieces = tokenizer.text_to_tokens('hello world')
    ids = tokenizer.text_to_ids('hello world')
    assert pieces == ['▁he', 'll', 'o', '▁wor', 'ld']
    assert ids == [11, 15, 21, 7, 14]
    assert tokenizer.ids_to_tokens(ids) == pieces
    assert tokenizer.tokens_to_ids(pieces) == ids
    assert tokenizer.tokens_to_text(pieces) == 'hello world'
    assert tokenizer.ids_to_text(ids) == 'hello world'
    assert tokenizer.token_to_id('▁he') == 11
    assert tokenizer.inv_vocab == {i: token for i, token in enumerate(tokenizer.vocab)}


def test_text_to_ids_raises_value_error_when_input_is_not_a_string(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model)
    with pytest.raises(ValueError, match='Expected str input'):
        tokenizer.text_to_ids(['hello'])


def test_decoding_accepts_numpy_arrays_and_torch_tensors(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model)
    ids = tokenizer.text_to_ids('hello world')
    assert tokenizer.ids_to_text(np.asarray(ids)) == 'hello world'
    assert tokenizer.ids_to_text(torch.tensor(ids)) == 'hello world'
    assert tokenizer.tokens_to_text(np.asarray(['▁he', 'll', 'o', '▁wor', 'ld'])) == ('hello world')


def test_text_to_ids_with_sampling_decodes_to_the_same_text(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model)
    sampled_ids = tokenizer.text_to_ids('hello', sample_alpha=0.5)
    assert tokenizer.ids_to_text(sampled_ids) == 'hello'


def test_text_to_ids_keeps_double_space_when_extra_whitespaces_are_not_ignored(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model, ignore_extra_whitespaces=False)
    assert tokenizer.text_to_tokens('a  a') == ['▁a', '▁', '▁a']
    assert tokenizer.text_to_ids('a  a') == [18, 19, 18]
    assert tokenizer.ids_to_text(tokenizer.text_to_ids('a  a')) == 'a  a'


def test_text_to_ids_with_extra_spaces_uses_sampling_when_alpha_is_given(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model, ignore_extra_whitespaces=False)
    sampled_ids = tokenizer.text_to_ids('a  a', sample_alpha=0.5)
    assert tokenizer.ids_to_text(sampled_ids) == 'a  a'


def test_text_to_ids_with_extra_spaces_skips_the_empty_trailing_segment(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model, ignore_extra_whitespaces=False)
    assert tokenizer.text_to_ids('a ') == [18, 19]
    assert tokenizer.ids_to_text([18, 19]) == 'a '


def test_sep_cls_and_mask_ids_raise_name_error_when_not_legacy(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model)
    for name in ('sep_id', 'cls_id', 'mask_id'):
        with pytest.raises(NameError, match='Use function token_to_id'):
            getattr(tokenizer, name)


def test_legacy_special_tokens_extend_vocabulary_and_are_found_in_text(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(
        sentencepiece_model,
        special_tokens={
            'pad_token': '<padx>',
            'bos_token': '<bosx>',
            'eos_token': '<eosx>',
            'sep_token': '<sepx>',
            'cls_token': '<clsx>',
            'mask_token': '<maskx>',
        },
        legacy=True,
    )
    assert tokenizer.vocab_size == 46
    assert (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id) == (40, 41, 42)
    assert (tokenizer.sep_id, tokenizer.cls_id, tokenizer.mask_id) == (43, 44, 45)
    assert tokenizer.token_to_id('<eosx>') == 42
    assert tokenizer.ids_to_tokens([42, 45]) == ['<eosx>', '<maskx>']
    assert tokenizer.vocab[40:] == ['<padx>', '<bosx>', '<eosx>', '<sepx>', '<clsx>', '<maskx>']
    assert tokenizer.inv_vocab[45] == '<maskx>'
    ids = tokenizer.text_to_ids('hello <eosx> world')
    assert ids == [11, 15, 21, 42, 7, 14]
    assert tokenizer.ids_to_text(ids) == 'hello <eosx> world'


def test_legacy_special_token_ids_are_parsed_in_order_and_trim_separator_afterwards(
    sentencepiece_model,
):
    tokenizer = SentencePieceTokenizer(
        sentencepiece_model, special_tokens={'eos_token': '<eosx>'}, legacy=True
    )
    text = 'hello <eosx> q <eosx> world'
    assert tokenizer.text_to_tokens(text) == [
        '▁he',
        'll',
        'o',
        '<eosx>',
        'q',
        '<eosx>',
        '▁wor',
        'ld',
    ]
    assert tokenizer.text_to_ids(text) == [11, 15, 21, 40, 3, 40, 7, 14]


def test_legacy_extra_space_encoding_is_used_for_the_tail_segment(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(
        sentencepiece_model,
        special_tokens={'eos_token': '<eosx>'},
        legacy=True,
        ignore_extra_whitespaces=False,
    )
    assert tokenizer.text_to_ids('a  a') == [18, 19, 18]
    assert tokenizer.text_to_ids('hello <eosx> a  a') == [11, 15, 21, 40, 18, 19, 18]


def test_add_special_tokens_maps_tokens_already_in_the_vocabulary(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model, legacy=True)
    tokenizer.add_special_tokens(['</s>'])
    assert tokenizer.special_token_to_id == {'</s>': 2}
    assert tokenizer.vocab_size == 40


def test_add_special_tokens_maps_dict_tokens_already_in_the_vocabulary(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model, legacy=True)
    tokenizer.add_special_tokens({'eos_token': '</s>', 'bos_token': '<s>'})
    assert tokenizer.special_token_to_id == {'</s>': 2, '<s>': 1}
    assert tokenizer.vocab_size == 40
    assert (tokenizer.eos_id, tokenizer.bos_id) == (2, 1)


def test_add_special_tokens_rejects_other_types(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model, legacy=True)
    with pytest.raises(ValueError, match='Expected special_tokens to be a list or a dict'):
        tokenizer.add_special_tokens(1)


def test_add_special_tokens_raises_attribute_error_when_not_legacy(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model)
    with pytest.raises(AttributeError, match='legacy is set to False'):
        tokenizer.add_special_tokens(['<extra>'])


def test_tokens_to_ids_converts_a_single_token_and_skips_requested_tokens(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model, legacy=True)
    tokenizer.add_special_tokens(['<extra>', '<another>'])
    assert tokenizer.tokens_to_ids('<extra>') == [40]
    assert tokenizer.tokens_to_ids(['<extra>', '<another>'], tokens_to_skip=['<another>']) == [40]


def test_additional_special_tokens_ids_exclude_the_named_special_tokens(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model, legacy=True)
    tokenizer.add_special_tokens(
        {
            'pad_token': '<padx>',
            'bos_token': '<bosx>',
            'eos_token': '<eosx>',
            'sep_token': '<sepx>',
            'cls_token': '<clsx>',
            'mask_token': '<maskx>',
        }
    )
    tokenizer.add_special_tokens(['<extra1>', '<extra2>'])
    assert tokenizer.additional_special_tokens_ids == [46, 47]


def test_apply_chat_template_renders_conversation_when_tokenize_is_disabled(sentencepiece_model):
    template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    tokenizer = SentencePieceTokenizer(sentencepiece_model, chat_template=template)
    conversation = [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'world'}]
    rendered = tokenizer.apply_chat_template(conversation, template, tokenize=False)
    assert rendered == 'hello world '


def test_apply_chat_template_returns_ids_and_truncates_when_tokenize_is_enabled(
    sentencepiece_model,
):
    template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    tokenizer = SentencePieceTokenizer(sentencepiece_model, chat_template=template)
    conversation = [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'world'}]
    ids = tokenizer.apply_chat_template(conversation, template, tokenize=True)
    assert ids == tokenizer.text_to_ids('hello world ')
    truncated = tokenizer.apply_chat_template(
        conversation, template, tokenize=True, truncation=True, max_length=2
    )
    assert truncated == ids[:2]


def test_apply_chat_template_requires_max_length_when_truncation_is_enabled(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model)
    with pytest.raises(AssertionError, match='max_length must be specified'):
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'hello'}], 'template', tokenize=True, truncation=True
        )


def test_apply_chat_template_requires_a_non_empty_template(sentencepiece_model):
    tokenizer = SentencePieceTokenizer(sentencepiece_model)
    with pytest.raises(AssertionError, match='Chat template is not defined'):
        tokenizer.apply_chat_template([{'role': 'user', 'content': 'hello'}], '')
