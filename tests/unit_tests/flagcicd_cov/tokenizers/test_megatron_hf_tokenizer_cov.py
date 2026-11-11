# Copyright (c) 2026, BAAI. All rights reserved.
"""MegatronHFTokenizer: config-map lookup, vocab/merges resolution, download-cache reuse."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

import megatron.core.tokenizers.text.libraries.megatron_hf_tokenizer as megatron_hf_module
from megatron.core.tokenizers.text.libraries.megatron_hf_tokenizer import MegatronHFTokenizer


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
    fast.save_pretrained(str(directory))
    return str(directory)


@pytest.fixture
def megatron_hf_env(monkeypatch, tmp_path, wordlevel_tokenizer_dir):
    cache = tmp_path / 'cache'
    cache.mkdir()
    monkeypatch.setattr(megatron_hf_module, 'MEGATRON_CACHE', str(cache))
    monkeypatch.setitem(
        megatron_hf_module.MEGATRON_CONFIG_MAP,
        'megatron-bert-uncased',
        {
            'vocab': 'https://example.invalid/bert-vocab.txt',
            'tokenizer_name': wordlevel_tokenizer_dir,
        },
    )
    monkeypatch.setitem(
        megatron_hf_module.MEGATRON_CONFIG_MAP,
        'megatron-gpt-345m',
        {
            'vocab': 'https://example.invalid/gpt-vocab.json',
            'merges_file': 'https://example.invalid/gpt-merges.txt',
            'tokenizer_name': wordlevel_tokenizer_dir,
        },
    )
    (cache / 'megatron-bert-uncased_vocab').write_text('cached-bert-vocab')
    return megatron_hf_module


def _install_fake_wget(monkeypatch, module, download):
    monkeypatch.setattr(module, 'HAVE_WGET', True, raising=False)
    monkeypatch.setattr(module, 'wget', SimpleNamespace(download=download), raising=False)


def test_megatron_hf_rejects_unknown_tokenizer_names(megatron_hf_env):
    with pytest.raises(ValueError, match='The name of the tokenizer is incorrect'):
        MegatronHFTokenizer('not-a-model')


def test_megatron_hf_uses_the_cached_vocabulary_without_downloading(megatron_hf_env):
    tokenizer = MegatronHFTokenizer('megatron-bert-uncased')
    assert tokenizer.vocab_size == 6
    cache = Path(megatron_hf_env.MEGATRON_CACHE)
    assert (cache / 'megatron-bert-uncased_vocab').read_text() == 'cached-bert-vocab'
    assert 'megatron-gpt-345m' in tokenizer._get_available_models_list()


def test_megatron_hf_accepts_explicit_vocab_and_merges_files(
    megatron_hf_env, wordlevel_tokenizer_dir
):
    tokenizer = MegatronHFTokenizer(
        'megatron-gpt-345m',
        vocab_file=wordlevel_tokenizer_dir + '/vocab.json',
        merges_file=wordlevel_tokenizer_dir + '/merges.txt',
    )
    assert tokenizer.vocab_size == 6
    assert tokenizer.pad_id == 0


def test_megatron_hf_downloads_vocab_and_merges_into_the_cache(megatron_hf_env, monkeypatch):
    module = megatron_hf_env
    cache = Path(module.MEGATRON_CACHE)
    assert not (cache / 'megatron-gpt-345m_vocab').exists()

    def fake_download(url):
        downloaded = cache / 'downloaded'
        downloaded.write_text('downloaded:' + url)
        return str(downloaded)

    _install_fake_wget(monkeypatch, module, fake_download)
    tokenizer = MegatronHFTokenizer('megatron-gpt-345m')
    assert tokenizer.vocab_size == 6
    assert (cache / 'megatron-gpt-345m_vocab').read_text() == (
        'downloaded:https://example.invalid/gpt-vocab.json'
    )
    assert (cache / 'megatron-gpt-345m_merges').read_text() == (
        'downloaded:https://example.invalid/gpt-merges.txt'
    )


def test_megatron_hf_waits_at_a_barrier_after_rank_zero_downloads(megatron_hf_env, monkeypatch):
    module = megatron_hf_env
    cache = Path(module.MEGATRON_CACHE)
    barriers = []

    def fake_download(url):
        downloaded = cache / 'downloaded'
        downloaded.write_text('downloaded:' + url)
        return str(downloaded)

    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(torch.distributed, 'barrier', lambda: barriers.append(True))
    _install_fake_wget(monkeypatch, module, fake_download)
    tokenizer = MegatronHFTokenizer('megatron-gpt-345m')
    assert tokenizer.vocab_size == 6
    assert barriers == [True, True]


def test_megatron_hf_raises_when_the_downloaded_file_is_missing(megatron_hf_env, monkeypatch):
    module = megatron_hf_env
    missing = Path(module.MEGATRON_CACHE) / 'nowhere'
    _install_fake_wget(monkeypatch, module, lambda url: str(missing))
    with pytest.raises(FileNotFoundError, match='Downloaded file not found'):
        MegatronHFTokenizer('megatron-gpt-345m')


def test_megatron_hf_raises_when_wget_is_missing_and_the_cache_is_empty(
    megatron_hf_env, monkeypatch
):
    module = megatron_hf_env
    monkeypatch.setitem(
        module.MEGATRON_CONFIG_MAP,
        'megatron-bert-cased',
        {
            'vocab': 'https://example.invalid/cased.txt',
            'tokenizer_name': module.MEGATRON_CONFIG_MAP['megatron-bert-uncased']['tokenizer_name'],
        },
    )
    monkeypatch.setattr(module, 'HAVE_WGET', False, raising=False)
    with pytest.raises(ModuleNotFoundError, match='wget library should be'):
        MegatronHFTokenizer('megatron-bert-cased')


def test_megatron_hf_skips_the_download_when_no_url_is_defined(megatron_hf_env, monkeypatch):
    module = megatron_hf_env
    monkeypatch.setitem(
        module.MEGATRON_CONFIG_MAP,
        'megatron-bert-uncased',
        {
            'vocab': None,
            'tokenizer_name': module.MEGATRON_CONFIG_MAP['megatron-bert-uncased']['tokenizer_name'],
        },
    )
    cached = Path(module.MEGATRON_CACHE) / 'megatron-bert-uncased_vocab'
    cached.unlink()
    tokenizer = MegatronHFTokenizer('megatron-bert-uncased')
    assert tokenizer.vocab_size == 6
    assert not cached.exists()
