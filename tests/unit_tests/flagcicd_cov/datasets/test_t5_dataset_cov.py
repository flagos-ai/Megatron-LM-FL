# Copyright (c) 2026, BAAI. All rights reserved.
"""T5 WordPiece dataset: sentinel replacement, truncation, masks by TE version."""

import hashlib
import json
import os
from collections import OrderedDict

import numpy
import pytest
import torch

from megatron.core.datasets import t5_dataset as t5_module
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.t5_dataset import (
    T5MaskedWordPieceDataset,
    T5MaskedWordPieceDatasetConfig,
)
from megatron.core.datasets.utils import Split


class _TinyT5Tokenizer:
    def __init__(self):
        self.bos = 1
        self.eos = 2
        self.cls = -1
        self.sep = -1
        self.mask = 3
        self.pad = 0
        self.vocab_size = 50
        self.additional_special_tokens_ids = [100, 101, 102, 103, 104]
        self.inv_vocab = {
            0: "[PAD]",
            1: "[BOS]",
            2: "[EOS]",
            3: "<mask>",
            10: "apple",
            11: "##s",
            12: "banana",
            13: "##ing",
            14: "pear",
        }
        self.special_tokens_dict = {"pad_token": 0, "bos_token": 1, "eos_token": 2}
        self.unique_identifiers = {"class": "TinyT5Tokenizer"}


def _config(**overrides):
    kwargs = dict(
        random_seed=42,
        sequence_length=16,
        sequence_length_decoder=16,
        tokenizer=_TinyT5Tokenizer(),
        masking_probability=0.4,
        short_sequence_probability=0.1,
        masking_max_ngram=1,
        masking_do_full_word=False,
        masking_do_permutation=False,
        masking_use_longer_ngrams=False,
        masking_use_geometric_distribution=False,
    )
    kwargs.update(overrides)
    return T5MaskedWordPieceDatasetConfig(**kwargs)


def _build_dataset(tmp_path, documents, sample_index, config, dataset_path):
    prefix = os.path.join(str(tmp_path), "t5_data")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=numpy.int32)
    for tokens in documents:
        builder.add_item(torch.tensor(tokens, dtype=torch.int64))
        builder.end_document()
    builder.finalize(prefix + ".idx")

    unique_identifiers = OrderedDict()
    unique_identifiers["class"] = "T5MaskedWordPieceDataset"
    unique_identifiers["dataset_path"] = dataset_path
    unique_identifiers["num_samples"] = None
    unique_identifiers["index_split"] = Split.train.name
    for attr in T5MaskedWordPieceDataset._key_config_attributes():
        unique_identifiers[attr] = getattr(config, attr)
    description = json.dumps(
        unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
    )
    description_hash = hashlib.md5(description.encode("utf-8"), usedforsecurity=False).hexdigest()
    cache_dir = config.path_to_cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(
        os.path.join(cache_dir, f"{description_hash}-T5MaskedWordPieceDataset-description.txt"), "w"
    ) as writer:
        writer.write(description)
    numpy.save(
        os.path.join(cache_dir, f"{description_hash}-T5MaskedWordPieceDataset-sample_index.npy"),
        sample_index,
    )
    return T5MaskedWordPieceDataset(
        IndexedDataset(prefix, mmap=False),
        dataset_path,
        numpy.arange(len(documents), dtype=numpy.int64),
        None,
        Split.train,
        config,
    )


def test_t5_dataset_config_requires_a_decoder_sequence_length():
    with pytest.raises(AssertionError):
        _config(sequence_length_decoder=None)


def test_t5_dataset_config_requires_sentinel_tokens():
    tokenizer = _TinyT5Tokenizer()
    tokenizer.additional_special_tokens_ids = []

    with pytest.raises(AssertionError):
        _config(tokenizer=tokenizer)


def test_t5_dataset_exposes_the_vocabulary_order_for_random_masks(tmp_path):
    config = _config(path_to_cache=str(tmp_path / "cache"))
    dataset = _build_dataset(
        tmp_path, [[10, 11, 12, 13]], numpy.array([[0, 1, 12]], dtype=numpy.int32), config, "vocab"
    )

    assert dataset.token_lookup == [0, 1, 2, 3, 10, 11, 12, 13, 14]
    assert dataset._get_token_mask(numpy.random.RandomState(0)) == 3


def test_t5_dataset_replaces_masked_spans_with_sentinels(tmp_path):
    config = _config(path_to_cache=str(tmp_path / "cache"))
    dataset = _build_dataset(
        tmp_path,
        [[10, 11, 12, 13]],
        numpy.array([[0, 1, 12]], dtype=numpy.int32),
        config,
        "sentinels",
    )

    sample = dataset[0]

    numpy.testing.assert_array_equal(
        sample["text_enc"], [10, 100, 12, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["text_dec"], [1, 100, 11, 101, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["labels"], [100, 11, 101, 13, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    )
    numpy.testing.assert_array_equal(
        sample["loss_mask"], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["enc_mask"], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["dec_mask"], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    assert sample["truncated"] == 0


def test_t5_dataset_truncates_long_samples_to_the_target_length(tmp_path):
    config = _config(path_to_cache=str(tmp_path / "cache"))
    dataset = _build_dataset(
        tmp_path,
        [[10, 11, 12, 13]],
        numpy.array([[0, 1, 3]], dtype=numpy.int32),
        config,
        "truncated",
    )

    sample = dataset[0]

    numpy.testing.assert_array_equal(
        sample["text_enc"], [100, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["text_dec"], [1, 100, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["labels"], [100, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    )
    numpy.testing.assert_array_equal(
        sample["loss_mask"], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    assert sample["truncated"] == 1


def _attention_mask_inputs():
    return (
        torch.tensor([[10, 0]], dtype=torch.int64),
        torch.tensor([[100, 101]], dtype=torch.int64),
        torch.tensor([[1, 0]], dtype=torch.int64),
        torch.tensor([[1, 1]], dtype=torch.int64),
    )


def test_t5_dataset_config_attention_mask_builds_b1ss_masks_for_the_local_implementation():
    encoder_tokens, decoder_tokens, encoder_mask, decoder_mask = _attention_mask_inputs()

    enc_mask, dec_mask, cross_mask = T5MaskedWordPieceDataset.config_attention_mask(
        encoder_tokens, decoder_tokens, encoder_mask, decoder_mask, use_local=True
    )

    torch.testing.assert_close(
        enc_mask, torch.tensor([[[[False, True], [True, True]]]], dtype=torch.bool)
    )
    torch.testing.assert_close(
        dec_mask, torch.tensor([[[[False, True], [False, False]]]], dtype=torch.bool)
    )
    torch.testing.assert_close(
        cross_mask, torch.tensor([[[[False, True], [False, True]]]], dtype=torch.bool)
    )


@pytest.mark.parametrize(
    ("te_version", "expected_encoder_form"),
    [
        pytest.param("1.10.0", "padding", id="te-1-10-padding-masks"),
        pytest.param("1.8.0", "b1ss", id="te-1-8-b1ss-masks"),
    ],
)
def test_t5_dataset_config_attention_mask_varies_with_the_te_version(
    monkeypatch, te_version, expected_encoder_form
):
    monkeypatch.delenv("NVTE_FLASH_ATTN", raising=False)
    monkeypatch.delenv("NVTE_FUSED_ATTN", raising=False)
    encoder_tokens, decoder_tokens, encoder_mask, decoder_mask = _attention_mask_inputs()

    enc_mask, dec_mask, cross_mask = T5MaskedWordPieceDataset.config_attention_mask(
        encoder_tokens,
        decoder_tokens,
        encoder_mask,
        decoder_mask,
        use_local=False,
        test_te_version=te_version,
    )

    if expected_encoder_form == "padding":
        torch.testing.assert_close(enc_mask, torch.tensor([[[[1, 0]]]], dtype=torch.int64))
        assert dec_mask is None
        torch.testing.assert_close(cross_mask[0], torch.tensor([[[[1, 1]]]], dtype=torch.int64))
        torch.testing.assert_close(cross_mask[1], torch.tensor([[[[1, 0]]]], dtype=torch.int64))
    else:
        torch.testing.assert_close(
            enc_mask, torch.tensor([[[[False, True], [True, True]]]], dtype=torch.bool)
        )
        assert dec_mask is None
        torch.testing.assert_close(
            cross_mask, torch.tensor([[[[False, True], [False, True]]]], dtype=torch.bool)
        )


def test_t5_dataset_config_attention_mask_builds_b1ss_masks_without_flash_attn_for_old_te(
    monkeypatch,
):
    monkeypatch.setattr(t5_module, "get_te_version", lambda: t5_module.PkgVersion("0.0.0"))
    monkeypatch.delenv("NVTE_FLASH_ATTN", raising=False)
    monkeypatch.delenv("NVTE_FUSED_ATTN", raising=False)
    encoder_tokens, decoder_tokens, encoder_mask, decoder_mask = _attention_mask_inputs()

    enc_mask, dec_mask, cross_mask = T5MaskedWordPieceDataset.config_attention_mask(
        encoder_tokens, decoder_tokens, encoder_mask, decoder_mask, use_local=False
    )

    torch.testing.assert_close(
        enc_mask, torch.tensor([[[[False, True], [True, True]]]], dtype=torch.bool)
    )
    assert dec_mask is None
    torch.testing.assert_close(
        cross_mask, torch.tensor([[[[False, True], [False, True]]]], dtype=torch.bool)
    )


def test_t5_dataset_config_attention_mask_keeps_padding_masks_for_flash_attn_between_te_1_7_and_1_10(
    monkeypatch,
):
    monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
    encoder_tokens, decoder_tokens, encoder_mask, decoder_mask = _attention_mask_inputs()

    enc_mask, dec_mask, cross_mask = T5MaskedWordPieceDataset.config_attention_mask(
        encoder_tokens,
        decoder_tokens,
        encoder_mask,
        decoder_mask,
        use_local=False,
        test_te_version="1.8.0",
    )

    torch.testing.assert_close(enc_mask, torch.tensor([[[[1, 0]]]], dtype=torch.int64))
    assert dec_mask is None
    torch.testing.assert_close(cross_mask[0], torch.tensor([[[[1, 1]]]], dtype=torch.int64))
    torch.testing.assert_close(cross_mask[1], torch.tensor([[[[1, 0]]]], dtype=torch.int64))


def test_t5_dataset_config_attention_mask_rejects_flash_attn_below_te_1_7(monkeypatch):
    monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
    encoder_tokens, decoder_tokens, encoder_mask, decoder_mask = _attention_mask_inputs()

    with pytest.raises(AssertionError, match="Flash and fused attention is not supported"):
        T5MaskedWordPieceDataset.config_attention_mask(
            encoder_tokens,
            decoder_tokens,
            encoder_mask,
            decoder_mask,
            use_local=False,
            test_te_version="1.6.0",
        )
