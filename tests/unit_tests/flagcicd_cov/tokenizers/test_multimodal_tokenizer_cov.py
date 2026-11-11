# Copyright (c) 2026, BAAI. All rights reserved.
"""MegatronMultimodalTokenizer: prompt formats, target masking, image-tag rewriting."""

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Tokenizer, PreTrainedTokenizerFast

from megatron.core.tokenizers.vision.libraries.multimodal_tokenizer import (
    MegatronMultimodalTokenizer,
)

CHAT_TEMPLATE = "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\n{% endfor %}"

QWEN_CONVERSATION = [
    {'role': 'system', 'content': 'sys'},
    {'role': 'user', 'content': 'hello'},
    {'role': 'assistant', 'content': 'world'},
]


@pytest.fixture
def multimodal_tokenizer_dir(tmp_path):
    directory = tmp_path / 'multimodal'
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
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
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
def qwen_multimodal(multimodal_tokenizer_dir):
    return MegatronMultimodalTokenizer(
        multimodal_tokenizer_dir,
        prompt_format='qwen2p0',
        special_tokens=['<image>'],
        image_tag_type='nvlm',
    )


def test_init_rejects_unknown_prompt_format(multimodal_tokenizer_dir):
    with pytest.raises(NotImplementedError, match='unknown multimodal tokenizer type'):
        MegatronMultimodalTokenizer(
            multimodal_tokenizer_dir, prompt_format='bogus', special_tokens=[], image_tag_type=''
        )


@pytest.fixture
def slow_gpt2_tokenizer_dir(tmp_path):
    source = tmp_path / 'slow_source'
    source.mkdir()
    vocab = {
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
    (source / 'vocab.json').write_text(json.dumps(vocab))
    (source / 'merges.txt').write_text('#version: 0.2\nh e\nhe llo\nw or\nwor ld\n')
    slow = GPT2Tokenizer(
        vocab_file=str(source / 'vocab.json'),
        merges_file=str(source / 'merges.txt'),
        unk_token='<unk>',
        pad_token='<pad>',
        bos_token='<s>',
        eos_token='</s>',
    )
    saved = tmp_path / 'slow_saved'
    slow.save_pretrained(str(saved))
    return str(saved)


def test_init_converts_slow_tokenizers_for_nvlm_yi_34b(slow_gpt2_tokenizer_dir):
    tokenizer = MegatronMultimodalTokenizer(
        slow_gpt2_tokenizer_dir,
        prompt_format='nvlm-yi-34b',
        special_tokens=['<image>'],
        image_tag_type='',
    )
    assert tokenizer.pad == 0
    assert tokenizer.tokenize('hello') == [2, 13, 8, 8, 9]


@pytest.mark.parametrize(
    'prompt_format,expected_pad',
    [
        ('mistral', 1),
        ('llama3', 1),
        ('llama3p1', 1),
        ('llama3p2', 1),
        ('chatml', 0),
        ('nemotron5', 1),
        ('nemotron5-aligned', 1),
    ],
)
def test_init_selects_the_prompt_config_for_each_format(
    multimodal_tokenizer_dir, prompt_format, expected_pad
):
    tokenizer = MegatronMultimodalTokenizer(
        multimodal_tokenizer_dir,
        prompt_format=prompt_format,
        special_tokens=['<image>'],
        image_tag_type='',
    )
    assert tokenizer.pad == expected_pad


def test_tokenize_encodes_text_and_applies_the_image_tag(qwen_multimodal):
    assert qwen_multimodal.tokenize('hello') == [4]
    assert qwen_multimodal.tokenize('hello <image> world') == [4, 1, 1, 1, 9, 1, 1, 1, 5]


def test_tokenize_accepts_a_conversation_as_a_list(qwen_multimodal):
    tokens = qwen_multimodal.tokenize(QWEN_CONVERSATION)
    assert (
        tokens
        == qwen_multimodal.tokenize_conversation(
            QWEN_CONVERSATION, return_target=False, add_generation_prompt=True
        ).tolist()
    )


def test_tokenize_conversation_masks_system_and_user_turns_in_the_target(qwen_multimodal):
    tokens, target = qwen_multimodal.tokenize_conversation(
        QWEN_CONVERSATION, return_target=True, add_generation_prompt=False
    )
    assert tokens.tolist() == [1] * 12 + [4] + [1] * 7 + [5] + [1] * 3
    assert target.tolist() == [-100] * 19 + [1, 5, 1, 1, 1]


def test_tokenize_conversation_masks_the_bos_once_with_bos_formats(multimodal_tokenizer_dir):
    tokenizer = MegatronMultimodalTokenizer(
        multimodal_tokenizer_dir,
        prompt_format='nemotron5',
        special_tokens=['<image>'],
        image_tag_type='',
    )
    tokens, target = tokenizer.tokenize_conversation(
        QWEN_CONVERSATION, return_target=True, add_generation_prompt=False
    )
    assert tokens.tolist() == [2] + [1] * 12 + [4] + [1] * 7 + [5] + [1] * 3
    assert target.tolist() == [-100] * 20 + [1, 5, 1, 1, 1]


def test_tokenize_conversation_skips_system_when_format_has_no_system_role(
    multimodal_tokenizer_dir,
):
    tokenizer = MegatronMultimodalTokenizer(
        multimodal_tokenizer_dir,
        prompt_format='mistral',
        special_tokens=['<image>'],
        image_tag_type='',
    )
    conversation = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'hello'}]
    tokens, target = tokenizer.tokenize_conversation(
        conversation, return_target=True, add_generation_prompt=False
    )
    assert tokens.tolist() == [2, 1, 1, 1, 4, 1, 1, 1]
    assert target.tolist() == [-100, -100, -100, -100, -100, -100, -100, -100]


def test_tokenize_conversation_forces_the_default_system_message_when_enabled(
    multimodal_tokenizer_dir,
):
    tokenizer = MegatronMultimodalTokenizer(
        multimodal_tokenizer_dir,
        prompt_format='qwen2p5',
        special_tokens=['<image>'],
        image_tag_type='',
        force_system_message=True,
    )
    tokens, target = tokenizer.tokenize_conversation(
        [{'role': 'user', 'content': 'hello'}], return_target=True, add_generation_prompt=False
    )
    assert tokens.tolist() == [1] * 25 + [4] + [1] * 3
    assert target.tolist() == [-100] * 29


def test_tokenize_conversation_replaces_an_existing_system_message_when_forced(
    multimodal_tokenizer_dir,
):
    tokenizer = MegatronMultimodalTokenizer(
        multimodal_tokenizer_dir,
        prompt_format='qwen2p5',
        special_tokens=['<image>'],
        image_tag_type='',
        force_system_message=True,
    )
    tokens = tokenizer.tokenize_conversation(
        [{'role': 'system', 'content': 'mine'}], return_target=True, add_generation_prompt=False
    )[0]
    assert tokens.tolist() == [1] * 21


def test_tokenize_conversation_capitalizes_roles_for_the_aligned_format(multimodal_tokenizer_dir):
    tokenizer = MegatronMultimodalTokenizer(
        multimodal_tokenizer_dir,
        prompt_format='nemotron5-aligned',
        special_tokens=['<image>'],
        image_tag_type='',
    )
    tokens = tokenizer.tokenize_conversation(
        [{'role': 'user', 'content': 'hi'}], return_target=False, add_generation_prompt=False
    )
    assert tokens.tolist() == [2, 7, 1, 1, 1, 1]


def test_tokenize_conversation_uses_the_tokenizer_template_for_chatml(multimodal_tokenizer_dir):
    tokenizer = MegatronMultimodalTokenizer(
        multimodal_tokenizer_dir,
        prompt_format='chatml',
        special_tokens=['<image>'],
        image_tag_type='',
    )
    tokens = tokenizer.tokenize_conversation(
        [{'role': 'user', 'content': 'hello'}], return_target=False, add_generation_prompt=False
    )
    assert tokens.tolist() == [1, 1, 4]


def test_tokenize_conversation_rejects_empty_turns(qwen_multimodal):
    with pytest.raises(ValueError, match='empty turn in conversation'):
        qwen_multimodal.tokenize_conversation(
            [{'role': 'user', 'content': ''}], return_target=True, add_generation_prompt=False
        )


def test_tokenize_conversation_rejects_image_tokens_in_assistant_content(qwen_multimodal):
    with pytest.raises(RuntimeError, match='not allowed in assistant content'):
        qwen_multimodal.tokenize_conversation(
            [{'role': 'assistant', 'content': 'look <image>'}],
            return_target=True,
            add_generation_prompt=False,
        )


def test_apply_image_tag_wraps_the_image_token_for_strings_and_turns(qwen_multimodal):
    assert qwen_multimodal._apply_image_tag('a <image> b') == 'a <Image><image></Image> b'
    tagged = qwen_multimodal._apply_image_tag([{'role': 'user', 'content': 'x <image> y'}])
    assert tagged == [{'role': 'user', 'content': 'x <Image><image></Image> y'}]


def test_apply_image_tag_leaves_text_unchanged_when_tag_type_is_empty(multimodal_tokenizer_dir):
    tokenizer = MegatronMultimodalTokenizer(
        multimodal_tokenizer_dir,
        prompt_format='qwen2p0',
        special_tokens=['<image>'],
        image_tag_type='',
    )
    assert tokenizer._apply_image_tag('a <image> b') == 'a <image> b'


def test_detokenize_decodes_ids_back_to_text(qwen_multimodal):
    assert qwen_multimodal.detokenize(qwen_multimodal.tokenize('hello')) == 'hello'


def test_convert_tokens_to_ids_maps_vocabulary_and_special_tokens(qwen_multimodal):
    assert qwen_multimodal.convert_tokens_to_ids(['hello', '<image>']) == [4, 9]


def test_add_special_tokens_extends_the_added_vocabulary(qwen_multimodal):
    qwen_multimodal.add_special_tokens(['<newtok>'])
    assert qwen_multimodal.get_special_tokens()['<newtok>'] == 10


def test_get_special_tokens_returns_the_added_vocabulary(qwen_multimodal):
    assert qwen_multimodal.get_special_tokens() == {
        '<pad>': 0,
        '<unk>': 1,
        '<s>': 2,
        '</s>': 3,
        '<image>': 9,
    }


def test_vocab_size_and_vocab_expose_the_underlying_tokenizer(qwen_multimodal):
    assert qwen_multimodal.vocab_size == 9
    assert len(qwen_multimodal.vocab) == 10
    assert qwen_multimodal.vocab['hello'] == 4
    assert qwen_multimodal.eod == 3
