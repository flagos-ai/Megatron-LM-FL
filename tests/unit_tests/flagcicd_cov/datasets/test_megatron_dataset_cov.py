# Copyright (c) 2026, BAAI. All rights reserved.
"""MegatronDataset base class: pad-token collision handling and the abstract contracts."""

import pytest

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import Split


class _Tokenizer:
    def __init__(self, pad, eos=None, eod=None, special_tokens_dict=None):
        self.pad = pad
        self.eos = eos
        self.eod = eod
        self.special_tokens_dict = special_tokens_dict
        self.unique_identifiers = {"class": "TinyTokenizer"}


class _Dataset(MegatronDataset):
    @staticmethod
    def numel_low_level_dataset(low_level_dataset):
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path, config):
        return []

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        return super().__getitem__(idx)


def _dataset(tokenizer, **config_overrides):
    config = BlendedMegatronDatasetConfig(
        random_seed=3, sequence_length=4, tokenizer=tokenizer, **config_overrides
    )
    return _Dataset([], "tiny", None, None, Split.train, config)


def test_megatron_dataset_resets_a_colliding_pad_token_id_and_warns():
    tokenizer = _Tokenizer(pad=7, eos=7, eod=7, special_tokens_dict={"eos_token": 7})

    with pytest.warns(UserWarning, match="collides with another special token id"):
        dataset = _dataset(tokenizer)

    assert dataset._pad_token_id == -1


def test_megatron_dataset_keeps_a_colliding_pad_token_id_when_the_user_opts_in():
    tokenizer = _Tokenizer(pad=7, eos=7, eod=7, special_tokens_dict={"eos_token": 7})

    with pytest.warns(UserWarning, match="collides with another special token id"):
        dataset = _dataset(tokenizer, allow_ambiguous_pad_tokens=True)

    assert dataset._pad_token_id == 7


def test_megatron_dataset_checks_eos_and_eod_when_no_special_tokens_dict_exists():
    tokenizer = _Tokenizer(pad=5, eos=5, eod=6, special_tokens_dict=None)

    with pytest.warns(UserWarning, match="collides with another special token id"):
        dataset = _dataset(tokenizer)

    assert dataset._pad_token_id == -1


def test_megatron_dataset_keeps_the_pad_token_id_when_nothing_collides():
    tokenizer = _Tokenizer(pad=5, eos=6, eod=7, special_tokens_dict={"eos_token": 6})

    dataset = _dataset(tokenizer)

    assert dataset._pad_token_id == 5


def test_megatron_dataset_falls_back_to_the_reserved_pad_token_id_without_a_tokenizer():
    class _AttributelessTokenizer:
        unique_identifiers = {"class": "AttributelessTokenizer"}

    dataset = _dataset(_AttributelessTokenizer())

    assert dataset._pad_token_id == -1


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: MegatronDataset.numel_low_level_dataset([]), id="numel"),
        pytest.param(lambda: MegatronDataset.build_low_level_dataset("unused", None), id="build"),
    ],
)
def test_megatron_dataset_static_builders_are_abstract(call):
    with pytest.raises(NotImplementedError) as caught:
        call()

    assert caught.value.args == ()


def test_megatron_dataset_len_and_getitem_have_no_default_behaviour():
    dataset = _dataset(_Tokenizer(pad=0, eos=1, eod=2, special_tokens_dict={"pad_token": 0}))

    assert dataset.__len__() is None
    assert dataset[0] is None
