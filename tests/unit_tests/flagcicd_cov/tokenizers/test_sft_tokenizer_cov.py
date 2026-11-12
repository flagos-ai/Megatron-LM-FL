# Copyright (c) 2026, BAAI. All rights reserved.
"""SFTTokenizer: per-format prompt configs, non-assistant masking, backend result handling."""

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.tokenizers.text.libraries.sft_tokenizer import SFTTokenizer

CONVERSATION = [
    {'role': 'system', 'content': 'sys'},
    {'role': 'user', 'content': 'hello'},
    {'role': 'assistant', 'content': 'world'},
]


@pytest.fixture
def sft_tokenizer_dir(tmp_path):
    directory = tmp_path / 'sft'
    directory.mkdir()
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<s>': 2,
        '</s>': 3,
        'hello': 4,
        'world': 5,
        'System': 6,
        'User': 7,
        'Assistant': 8,
        '<SPECIAL_10>': 9,
        '<SPECIAL_11>': 10,
        '<SPECIAL_12>': 11,
        '<SPECIAL_233>': 12,
        'sys': 13,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token='<unk>',
        pad_token='<pad>',
        bos_token='<s>',
        eos_token='</s>',
        additional_special_tokens=['<SPECIAL_10>', '<SPECIAL_11>', '<SPECIAL_12>', '<SPECIAL_233>'],
    )
    fast.chat_template = "{{ bos_token }}{% for m in messages %}{{ m['content'] }} {% endfor %}"
    fast.save_pretrained(str(directory))
    return str(directory)


def test_init_rejects_unknown_prompt_formats(sft_tokenizer_dir):
    with pytest.raises(NotImplementedError, match='unknown SFT prompt format'):
        SFTTokenizer(sft_tokenizer_dir, prompt_format='bogus')


@pytest.mark.parametrize(
    'prompt_format,expected_pad',
    [('nemotron-nano-v2', 1), ('nemotron-h-aligned', 12), ('identity', 1), ('default', 0)],
    ids=['nano-v2', 'h-aligned', 'identity', 'default'],
)
def test_pad_id_uses_the_token_of_each_prompt_format(
    sft_tokenizer_dir, prompt_format, expected_pad
):
    assert SFTTokenizer(sft_tokenizer_dir, prompt_format).pad_id == expected_pad


def test_init_exposes_the_underlying_tokenizer_ids(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='identity')
    assert tokenizer.vocab_size == 14
    assert tokenizer.bos_id == 2
    assert tokenizer.eod == 3


def test_tokenize_conversation_masks_non_assistant_turns_for_nemotron_nano_v2(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='nemotron-nano-v2')
    tokens, target = tokenizer.tokenize_conversation(
        CONVERSATION, return_target=True, add_generation_prompt=False
    )
    assert tokens.tolist() == [9, 6, 13, 10, 7, 4, 10, 8, 5, 11]
    assert target.tolist() == [-100] * 9 + [11]


def test_tokenize_conversation_masks_the_assistant_prefix_for_nemotron_h_aligned(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='nemotron-h-aligned')
    tokens, target = tokenizer.tokenize_conversation(
        CONVERSATION, return_target=True, add_generation_prompt=False
    )
    assert tokens.tolist() == [9, 6, 13, 10, 7, 4, 10, 8, 5]
    assert target.tolist() == [-100] * 8 + [5]


def test_tokenize_conversation_returns_tokens_only_when_no_target_is_requested(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='nemotron-nano-v2')
    tokens = tokenizer.tokenize_conversation(
        CONVERSATION, return_target=False, add_generation_prompt=False
    )
    assert tokens.tolist() == [9, 6, 13, 10, 7, 4, 10, 8, 5, 11]


def test_tokenize_conversation_returns_an_unmasked_target_for_identity(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='identity')
    tokens, target = tokenizer.tokenize_conversation(
        [{'role': 'user', 'content': 'hello'}], return_target=True, add_generation_prompt=False
    )
    assert tokens.tolist() == [4]
    assert target.tolist() == [4]


def test_tokenize_conversation_returns_an_unmasked_target_for_default(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='default')
    tokens, target = tokenizer.tokenize_conversation(
        CONVERSATION, return_target=True, add_generation_prompt=False
    )
    assert tokens.tolist() == [2, 13, 4, 5]
    assert target.tolist() == [2, 13, 4, 5]


def test_tokenize_conversation_rejects_an_empty_assistant_turn(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='nemotron-nano-v2')
    with pytest.raises(ValueError, match='empty assistant turn in conversation'):
        tokenizer.tokenize_conversation(
            [{'role': 'assistant', 'content': ''}], return_target=True, add_generation_prompt=False
        )


def test_tokenize_conversation_requires_a_user_turn_before_each_assistant_turn(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='nemotron-nano-v2')
    conversation = [
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': 'world'},
        {'role': 'assistant', 'content': 'again'},
    ]
    with pytest.raises(AssertionError):
        tokenizer.tokenize_conversation(
            conversation, return_target=True, add_generation_prompt=False
        )


def test_tokenize_conversation_rejects_unknown_roles(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='nemotron-nano-v2')
    with pytest.raises(ValueError, match='Wrong role value'):
        tokenizer.tokenize_conversation(
            [{'role': 'unknown', 'content': 'hello'}],
            return_target=True,
            add_generation_prompt=False,
        )


def test_text_to_ids_encodes_plain_strings(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='identity')
    assert tokenizer.text_to_ids('hello world') == [4, 5]


def test_text_to_ids_tokenizes_conversations_for_list_inputs(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='nemotron-nano-v2')
    assert tokenizer.text_to_ids(CONVERSATION) == [9, 6, 13, 10, 7, 4, 10, 8, 5, 11]


def test_tokens_to_ids_and_ids_to_text_round_trip(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='identity')
    assert tokenizer.tokens_to_ids(['hello', '<SPECIAL_233>']) == [4, 12]
    assert tokenizer.ids_to_text([4, 5]) == 'hello world'


@pytest.mark.parametrize(
    'method', ['ids_to_tokens', 'text_to_tokens', 'tokens_to_text', 'add_special_tokens']
)
def test_unsupported_conversions_are_rejected(sft_tokenizer_dir, method):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='identity')
    with pytest.raises(NotImplementedError, match='not supported for SFTTokenizer'):
        getattr(tokenizer, method)()


def test_get_special_tokens_returns_the_added_vocabulary(sft_tokenizer_dir):
    tokenizer = SFTTokenizer(sft_tokenizer_dir, prompt_format='identity')
    assert tokenizer.get_special_tokens() == {
        '<pad>': 0,
        '<unk>': 1,
        '<s>': 2,
        '</s>': 3,
        '<SPECIAL_10>': 9,
        '<SPECIAL_11>': 10,
        '<SPECIAL_12>': 11,
        '<SPECIAL_233>': 12,
    }


@pytest.mark.parametrize(
    'result_factory',
    [
        lambda fast: fast('hello'),
        lambda fast: fast('hello', return_tensors='np'),
        lambda fast: fast._tokenizer.encode('hello'),
        lambda fast: [4],
        lambda fast: np.array([[4]]),
    ],
    ids=['batch-encoding', 'batch-encoding-2d', 'encoding-object', 'plain-list', '2d-ndarray'],
)
def test_extract_token_ids_normalises_every_backend_result(sft_tokenizer_dir, result_factory):
    fast = AutoTokenizer.from_pretrained(sft_tokenizer_dir)
    assert SFTTokenizer._extract_token_ids(result_factory(fast)).tolist() == [4]


def test_sft_tokenizer_works_behind_the_text_wrapper(sft_tokenizer_dir):
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=sft_tokenizer_dir, metadata_path={'library': 'sft'}, prompt_format='identity'
    )
    assert tokenizer.tokenize('hello world') == [4, 5]
    assert tokenizer.tokenize_conversation(
        [{'role': 'user', 'content': 'hello'}], return_target=False, add_generation_prompt=False
    ).tolist() == [4]
