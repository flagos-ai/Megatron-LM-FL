# Copyright (c) 2026, BAAI. All rights reserved.
"""MegatronTokenizer metadata write/restore, byte-level/null-text libraries, model classes."""

import json
from types import SimpleNamespace

import pytest

import megatron.core.tokenizers.text.models as text_models
import megatron.core.tokenizers.vision.models as vision_models
from megatron.core.tokenizers.base_tokenizer import MegatronTokenizerBase
from megatron.core.tokenizers.megatron_tokenizer import (
    MegatronTokenizer,
    _get_metadata_path,
    _get_tokenizer_model_class,
)
from megatron.core.tokenizers.text.libraries.abstract_tokenizer import MegatronTokenizerTextAbstract
from megatron.core.tokenizers.text.libraries.bytelevel_tokenizer import ByteLevelTokenizer
from megatron.core.tokenizers.text.libraries.null_tokenizer import NullTokenizer
from megatron.core.tokenizers.text.models.bert_tokenizer import BertTokenizer
from megatron.core.tokenizers.text.models.gpt_tokenizer import GPTTokenizer
from megatron.core.tokenizers.text.models.mamba_tokenizer import MambaTokenizer
from megatron.core.tokenizers.text.models.t5_tokenizer import T5Tokenizer


class _PlaceholderTokenizer(MegatronTokenizerBase):
    """Concrete subclass used to exercise the base constructor's config handling."""

    tokenize = None
    detokenize = None
    vocab = None
    vocab_size = None
    apply_chat_template = None


def test_base_config_entries_are_set_as_attributes():
    tokenizer = _PlaceholderTokenizer('model', {'library': 'null-text', 'model_type': 'gpt'})
    assert tokenizer.path == 'model'
    assert tokenizer.library == 'null-text'
    assert tokenizer.model_type == 'gpt'


def test_text_abstract_property_aliases_return_the_available_ids():
    holder = SimpleNamespace(cls_id=1, sep_id=2, pad_id=3, eod_id=4, bos_id=5, eos_id=6, mask_id=7)
    assert MegatronTokenizerTextAbstract.cls_id.fget(holder) == 1
    assert MegatronTokenizerTextAbstract.sep_id.fget(holder) == 2
    assert MegatronTokenizerTextAbstract.pad_id.fget(holder) == 3
    assert MegatronTokenizerTextAbstract.eod.fget(holder) == 4
    assert MegatronTokenizerTextAbstract.bos_id.fget(holder) == 5
    assert MegatronTokenizerTextAbstract.eos_id.fget(holder) == 6
    assert MegatronTokenizerTextAbstract.mask_id.fget(holder) == 7


def test_text_abstract_eod_falls_back_to_eos_id_when_eod_id_is_missing():
    assert MegatronTokenizerTextAbstract.eod.fget(SimpleNamespace(eos_id=9)) == 9


def test_text_abstract_property_aliases_raise_when_no_id_is_available():
    for name in ('cls_id', 'sep_id', 'pad_id', 'eod', 'bos_id', 'eos_id', 'mask_id'):
        with pytest.raises(AttributeError, match='has no attribute'):
            getattr(MegatronTokenizerTextAbstract, name).fget(SimpleNamespace())


def test_megatron_tokenizer_rejects_direct_instantiation():
    with pytest.raises(EnvironmentError, match='MegatronTokenizer.from_pretrained'):
        MegatronTokenizer()


def test_write_metadata_writes_default_metadata_next_to_the_model(tmp_path):
    model_file = tmp_path / 'model.bin'
    model_file.write_text('model')
    MegatronTokenizer.write_metadata(str(model_file), 'null-text')
    assert json.loads((tmp_path / 'tokenizer_metadata.json').read_text()) == {
        'library': 'null-text',
        'class_name': None,
        'class_path': None,
        'model_type': 'default-text',
        'chat_template': None,
    }


def test_write_metadata_records_the_given_class_model_type_and_template(tmp_path):
    model_file = tmp_path / 'model.bin'
    model_file.write_text('model')
    MegatronTokenizer.write_metadata(
        str(model_file),
        'huggingface',
        model_type='gpt',
        tokenizer_class=GPTTokenizer,
        chat_template='{{ template }}',
        overwrite=True,
    )
    assert json.loads((tmp_path / 'tokenizer_metadata.json').read_text()) == {
        'library': 'huggingface',
        'class_name': 'GPTTokenizer',
        'class_path': 'megatron.core.tokenizers.text.models.gpt_tokenizer',
        'model_type': 'gpt',
        'chat_template': '{{ template }}',
    }


def test_write_metadata_requires_an_existing_tokenizer_model_path(tmp_path):
    with pytest.raises(AssertionError, match="Tokenizer path doesn't exist"):
        MegatronTokenizer.write_metadata(str(tmp_path / 'missing'), 'null-text')


def test_write_metadata_rejects_an_unsupported_library(tmp_path):
    model_file = tmp_path / 'model.bin'
    model_file.write_text('model')
    with pytest.raises(AssertionError, match='Tokenizer library is not supported'):
        MegatronTokenizer.write_metadata(str(model_file), 'unsupported', overwrite=True)


def test_write_metadata_refuses_to_overwrite_an_existing_file(tmp_path):
    model_file = tmp_path / 'model.bin'
    model_file.write_text('model')
    MegatronTokenizer.write_metadata(str(model_file), 'null-text')
    with pytest.raises(ValueError, match='Metadata file already exists'):
        MegatronTokenizer.write_metadata(str(model_file), 'null-text')


def test_write_metadata_overwrites_the_file_when_requested(tmp_path):
    model_file = tmp_path / 'model.bin'
    model_file.write_text('model')
    MegatronTokenizer.write_metadata(str(model_file), 'null-text')
    MegatronTokenizer.write_metadata(str(model_file), 'multimodal', overwrite=True)
    assert json.loads((tmp_path / 'tokenizer_metadata.json').read_text()) == {
        'library': 'multimodal',
        'class_name': None,
        'class_path': None,
        'model_type': 'default-vision',
        'chat_template': None,
    }


def test_get_metadata_path_lives_next_to_files_and_inside_directories(tmp_path):
    model_file = tmp_path / 'model.bin'
    model_file.write_text('model')
    expected = str(tmp_path / 'tokenizer_metadata.json')
    assert _get_metadata_path(str(model_file)) == expected
    assert _get_metadata_path(str(tmp_path)) == expected


@pytest.mark.parametrize(
    'library,model_type,expected_class',
    [
        ('null-text', 'gpt', GPTTokenizer),
        ('null-text', 'bert', BertTokenizer),
        ('null-text', 'mamba', MambaTokenizer),
        ('null-text', 't5', T5Tokenizer),
        ('null-text', None, text_models.DefaultTokenizerText),
        ('null-multimodal', None, vision_models.DefaultTokenizerVision),
    ],
    ids=['gpt', 'bert', 'mamba', 't5', 'default-text', 'default-vision'],
)
def test_get_tokenizer_model_class_resolves_model_types(library, model_type, expected_class):
    assert _get_tokenizer_model_class(library, {'model_type': model_type}) is expected_class


def test_from_pretrained_builds_the_class_named_by_hand_written_metadata():
    """Hand-written dict metadata names the model class directly with the 'tokenizer_class'/
    'tokenizer_class_path'/'tokenizer_class_name' keys; write_metadata writes 'class_name'/
    'class_path' instead, so it never produces these keys (see SUSPECTED SOURCE BUGS)."""
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={
            'library': 'byte-level',
            'tokenizer_class': GPTTokenizer,
            'tokenizer_class_path': text_models,
            'tokenizer_class_name': 'GPTTokenizer',
        },
        vocab_size=260,
    )
    assert type(tokenizer) is GPTTokenizer
    assert tokenizer.class_name == 'GPTTokenizer'
    assert tokenizer.class_path == 'megatron.core.tokenizers.text.models.gpt_tokenizer'
    assert tokenizer.vocab_size == 260


def test_from_pretrained_loads_the_metadata_written_next_to_the_model(tmp_path):
    model_file = tmp_path / 'model.bin'
    model_file.write_text('model')
    MegatronTokenizer.write_metadata(str(model_file), 'null-text')
    tokenizer = MegatronTokenizer.from_pretrained(str(model_file), vocab_size=19)
    assert tokenizer.library == 'null-text'
    assert tokenizer.vocab_size == 19


def test_from_pretrained_loads_metadata_from_a_json_file(tmp_path):
    metadata_file = tmp_path / 'metadata.json'
    metadata_file.write_text(json.dumps({'library': 'null-text', 'model_type': 'gpt'}))
    tokenizer = MegatronTokenizer.from_pretrained(metadata_path=str(metadata_file), vocab_size=7)
    assert tokenizer.model_type == 'gpt'
    assert tokenizer.tokenize('3 11') == [3, 11]


def test_from_pretrained_rejects_a_missing_metadata_file(tmp_path):
    with pytest.raises(AssertionError, match="metadata file doesn't exist"):
        MegatronTokenizer.from_pretrained(
            str(tmp_path / 'model'), metadata_path=str(tmp_path / 'missing.json')
        )


def test_from_pretrained_rejects_metadata_paths_of_other_types():
    with pytest.raises(ValueError, match='Expected metadata_path to be str or dict'):
        MegatronTokenizer.from_pretrained('path', metadata_path=1)


def test_from_pretrained_requires_a_tokenizer_path_for_file_backed_libraries():
    with pytest.raises(AssertionError, match='Tokenizer path must be specified'):
        MegatronTokenizer.from_pretrained(metadata_path={'library': 'huggingface'})


@pytest.mark.parametrize(
    'kwargs,match',
    [
        ({}, 'Prompt format'),
        ({'prompt_format': 'chatml'}, 'Special tokens'),
        ({'prompt_format': 'chatml', 'special_tokens': []}, 'Image tag type'),
    ],
    ids=['prompt-format', 'special-tokens', 'image-tag-type'],
)
def test_from_pretrained_requires_all_multimodal_arguments(kwargs, match):
    with pytest.raises(AssertionError, match=match):
        MegatronTokenizer.from_pretrained('path', metadata_path={'library': 'multimodal'}, **kwargs)


def test_from_pretrained_restores_the_null_text_library():
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={'library': 'null-text'}, vocab_size=19, eod_id=12, pad_id=4
    )
    assert tokenizer.tokenize('1 20 -3') == [1, 20, -3]
    assert tokenizer.detokenize([1, 20, -3]) == '1 20 -3'
    assert (tokenizer.vocab_size, tokenizer.eod, tokenizer.pad_id) == (19, 12, 4)


def test_from_pretrained_restores_the_byte_level_library():
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={'library': 'byte-level'}, vocab_size=300, special_tokens=['<x>'], _eos_id=2
    )
    assert tokenizer.tokenize('A') == [65]
    assert tokenizer._tokenizer.token_to_id('<x>') == 300


def test_bytelevel_maps_special_tokens_to_ids_at_the_end_of_the_vocabulary():
    tokenizer = ByteLevelTokenizer(
        special_tokens=['<extra>', '<second>'], vocab_size=300, _eos_id=2, _pad_id=1, _bos_id=3
    )
    assert tokenizer.vocab_size == 302
    assert tokenizer.token_to_id('<extra>') == 301
    assert tokenizer.id_to_token(300) == '<second>'
    assert tokenizer.special_token_to_id == {1: 1, 3: 3, 2: 2, '<extra>': 301, '<second>': 300}


def test_bytelevel_defaults_to_a_512_entry_byte_vocabulary():
    tokenizer = ByteLevelTokenizer()
    assert tokenizer.vocab_size == 512
    assert tokenizer.token_to_id(200) == 200
    assert tokenizer.id_to_token(200) == 200


def test_bytelevel_encodes_text_as_utf8_bytes_and_decodes_it_back():
    tokenizer = ByteLevelTokenizer(vocab_size=300)
    ids = tokenizer.text_to_ids('hé ')
    assert ids == [104, 195, 169, 32]
    assert tokenizer.text_to_tokens('hé ') == ids
    assert tokenizer.ids_to_text(ids) == 'hé'
    assert tokenizer.tokens_to_text(ids) == 'hé'


def test_bytelevel_drops_special_ids_while_decoding():
    tokenizer = ByteLevelTokenizer(special_tokens=['<extra>'], vocab_size=300)
    assert tokenizer.ids_to_text([104, 300, 105]) == 'hi'


def test_bytelevel_converts_a_single_token_string_to_a_one_element_list():
    tokenizer = ByteLevelTokenizer(special_tokens=['<extra>'], vocab_size=300)
    assert tokenizer.tokens_to_ids('<extra>') == [300]
    assert tokenizer.tokens_to_ids(['<extra>', 65]) == [300, 65]


def test_bytelevel_converts_a_single_id_to_a_one_element_list():
    tokenizer = ByteLevelTokenizer(special_tokens=['<extra>'], vocab_size=300)
    assert tokenizer.ids_to_tokens(300) == ['<extra>']
    assert tokenizer.ids_to_tokens([300, 65]) == ['<extra>', 65]


def test_bytelevel_returns_the_configured_special_ids():
    tokenizer = ByteLevelTokenizer(vocab_size=300, _eos_id=2, _pad_id=1, _bos_id=3)
    assert (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.eod) == (1, 3, 2, 2)
    assert tokenizer.unk_id == 259


def test_bytelevel_add_special_tokens_is_not_supported():
    tokenizer = ByteLevelTokenizer(vocab_size=300)
    with pytest.raises(NotImplementedError, match='not supported for byte-level tokenizers'):
        tokenizer.add_special_tokens([])


def test_null_converts_space_separated_numbers_round_trip():
    tokenizer = NullTokenizer(17, eod_id=12, pad_id=4)
    assert tokenizer.text_to_ids('1 20 -3') == [1, 20, -3]
    assert tokenizer.ids_to_text([1, 20, -3]) == '1 20 -3'


def test_null_converts_tokens_between_strings_and_integers():
    tokenizer = NullTokenizer(17)
    assert tokenizer.tokens_to_ids(['1', '20']) == [1, 20]
    assert tokenizer.ids_to_tokens([1, 20]) == ['1', '20']


def test_null_offsets_advance_by_the_printed_width_of_each_id():
    tokenizer = NullTokenizer(17)
    assert tokenizer.offsets([1, 20, -3], '1 20 -3') == [0, 2, 5]


def test_null_unique_identifiers_describe_the_configuration():
    tokenizer = NullTokenizer(17, eod_id=12, pad_id=4)
    assert tokenizer.unique_identifiers == {
        'class': 'megatron.core.tokenizers.text.libraries.null_tokenizer.NullTokenizer',
        'vocab_size': 17,
        'eod_id': 12,
        'pad_id': 4,
    }


def test_null_reports_minus_one_for_cls_sep_and_mask():
    tokenizer = NullTokenizer(17)
    assert (tokenizer.cls, tokenizer.sep, tokenizer.mask) == (-1, -1, -1)


def test_null_vocab_access_is_not_implemented():
    tokenizer = NullTokenizer(17)
    with pytest.raises(NotImplementedError) as caught:
        tokenizer.vocab
    assert str(caught.value) == ''
    with pytest.raises(NotImplementedError) as caught:
        tokenizer.inv_vocab
    assert str(caught.value) == ''


def test_null_additional_special_tokens_ids_is_none():
    tokenizer = NullTokenizer(17, eod_id=12, pad_id=4)
    assert tokenizer.additional_special_tokens_ids is None


@pytest.mark.parametrize('model_class', [BertTokenizer, GPTTokenizer, MambaTokenizer, T5Tokenizer])
def test_model_classes_record_their_names_and_paths_in_the_config(model_class):
    config = {'library': 'byte-level'}
    instance = model_class(None, config, vocab_size=260)
    assert instance.class_name == model_class.__name__
    assert instance.class_path == model_class.__module__
    assert instance.library == 'byte-level'
    assert instance.vocab_size == 260
