# Copyright (c) 2026, BAAI. All rights reserved.
"""MegatronTokenizerVision wrapper: restoring multimodal and null-multimodal libraries."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.tokenizers.vision.libraries.null_multimodal_tokenizer import (
    MegatronNullMultimodalTokenizer,
)
from megatron.core.tokenizers.vision.models.default_tokenizer import DefaultTokenizerVision


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
    fast.chat_template = "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\n{% endfor %}"
    fast.save_pretrained(str(directory))
    return str(directory)


def test_null_multimodal_tokenizes_and_detokenizes_integer_strings():
    tokenizer = MegatronNullMultimodalTokenizer(10)
    assert tokenizer.tokenize('1 22 333') == [1, 22, 333]
    assert tokenizer.detokenize([1, 22, 333]) == '1 22 333'


def test_null_multimodal_vocab_size_reserves_one_extra_id_for_eod():
    tokenizer = MegatronNullMultimodalTokenizer(10)
    assert tokenizer.vocab_size == 11
    assert tokenizer.eod == 10


def test_null_multimodal_reports_minus_one_for_cls_sep_and_mask():
    tokenizer = MegatronNullMultimodalTokenizer(10)
    assert (tokenizer.cls, tokenizer.sep, tokenizer.mask) == (-1, -1, -1)


def test_null_multimodal_additional_special_tokens_ids_is_none():
    tokenizer = MegatronNullMultimodalTokenizer(10, image_token='<pic>', image_token_id=99)
    assert tokenizer.additional_special_tokens_ids is None


def test_null_multimodal_offsets_advance_by_the_width_of_each_id():
    tokenizer = MegatronNullMultimodalTokenizer(10)
    assert tokenizer.offsets([1, 20, 3], '1 20 3') == [0, 2, 5]


def test_null_multimodal_converts_the_image_token_to_its_placeholder_id():
    tokenizer = MegatronNullMultimodalTokenizer(10, image_token='<pic>', image_token_id=99)
    assert tokenizer.convert_tokens_to_ids('<pic>  4') == [99, 4]
    assert tokenizer.convert_tokens_to_ids('6') == 6


def test_vision_wrapper_restores_the_null_multimodal_library():
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={'library': 'null-multimodal'},
        vocab_size=11,
        image_token='<pic>',
        image_token_id=99,
    )
    assert isinstance(tokenizer, DefaultTokenizerVision)
    assert tokenizer.tokenize('1 2') == [1, 2]
    assert tokenizer.detokenize([1, 2]) == '1 2'
    assert tokenizer.offsets([1, 20, 3], '1 20 3') == [0, 2, 5]
    assert tokenizer.convert_tokens_to_ids('<pic>  4') == [99, 4]
    assert tokenizer.vocab_size == 12
    assert tokenizer.eod == 11


def test_vision_wrapper_restores_the_multimodal_library(multimodal_tokenizer_dir):
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=multimodal_tokenizer_dir,
        metadata_path={'library': 'multimodal'},
        prompt_format='qwen2p0',
        special_tokens=['<image>'],
        image_tag_type='nvlm',
    )
    assert isinstance(tokenizer, DefaultTokenizerVision)
    assert tokenizer.tokenize('hello') == [4]
    assert tokenizer.tokenize_conversation(
        [{'role': 'user', 'content': 'hello'}], return_target=False, add_generation_prompt=False
    ).tolist() == [1, 1, 1, 1, 4, 1, 1, 1]
    assert tokenizer.detokenize([4]) == 'hello'
    assert tokenizer.convert_tokens_to_ids(['hello', '<image>']) == [4, 9]
    assert tokenizer.get_special_tokens()['<image>'] == 9
    assert tokenizer.vocab_size == 9
    assert tokenizer.vocab['hello'] == 4
    assert (tokenizer.pad, tokenizer.eod) == (0, 3)
    tokenizer.add_special_tokens(['<newtok>'])
    assert tokenizer.get_special_tokens()['<newtok>'] == 10
    with pytest.raises(NotImplementedError, match='not supported for vision tokenizers'):
        tokenizer.apply_chat_template()
