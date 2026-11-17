# Copyright (c) 2026, BAAI. All rights reserved.
"""BERT WordPiece dataset: two-segment sampling, truncation, masks, labels, padding masks."""

import hashlib
import json
import os
from collections import OrderedDict

import numpy
import pytest
import torch

from megatron.core.datasets.bert_dataset import (
    BERTMaskedWordPieceDataset,
    BERTMaskedWordPieceDatasetConfig,
)
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.utils import Split


class _TinyTokenizer:
    def __init__(self):
        self.cls = 1
        self.sep = 2
        self.mask = 3
        self.pad = 0
        self.vocab_size = 50
        self.inv_vocab = {
            0: "[PAD]",
            1: "[CLS]",
            2: "[SEP]",
            3: "<mask>",
            10: "apple",
            11: "##s",
            12: "banana",
            13: "##ing",
            14: "pear",
        }
        self.special_tokens_dict = {"pad_token": 0, "cls_token": 1, "sep_token": 2}
        self.unique_identifiers = {"class": "TinyTokenizer"}


def _config(**overrides):
    kwargs = dict(
        random_seed=42,
        sequence_length=16,
        tokenizer=_TinyTokenizer(),
        masking_probability=0.2,
        short_sequence_probability=0.1,
        masking_max_ngram=3,
        masking_do_full_word=True,
        masking_do_permutation=False,
        masking_use_longer_ngrams=False,
        masking_use_geometric_distribution=False,
        classification_head=True,
    )
    kwargs.update(overrides)
    return BERTMaskedWordPieceDatasetConfig(**kwargs)


def _build_dataset(tmp_path, documents, sample_index, config, dataset_path):
    prefix = os.path.join(str(tmp_path), "bert_data")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=numpy.int32)
    for tokens in documents:
        builder.add_item(torch.tensor(tokens, dtype=torch.int64))
        builder.end_document()
    builder.finalize(prefix + ".idx")

    unique_identifiers = OrderedDict()
    unique_identifiers["class"] = "BERTMaskedWordPieceDataset"
    unique_identifiers["dataset_path"] = dataset_path
    unique_identifiers["num_samples"] = None
    unique_identifiers["index_split"] = Split.train.name
    for attr in BERTMaskedWordPieceDataset._key_config_attributes():
        unique_identifiers[attr] = getattr(config, attr)
    description = json.dumps(
        unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
    )
    description_hash = hashlib.md5(description.encode("utf-8"), usedforsecurity=False).hexdigest()
    cache_dir = config.path_to_cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(
        os.path.join(cache_dir, f"{description_hash}-BERTMaskedWordPieceDataset-description.txt"),
        "w",
    ) as writer:
        writer.write(description)
    numpy.save(
        os.path.join(cache_dir, f"{description_hash}-BERTMaskedWordPieceDataset-sample_index.npy"),
        sample_index,
    )
    return BERTMaskedWordPieceDataset(
        IndexedDataset(prefix, mmap=False),
        dataset_path,
        numpy.arange(len(documents), dtype=numpy.int64),
        None,
        Split.train,
        config,
    )


def test_bert_dataset_config_requires_a_classification_head_decision():
    with pytest.raises(AssertionError):
        _config(classification_head=None)


def test_bert_dataset_samples_two_segments_and_masks_one_token_with_a_classification_head(tmp_path):
    config = _config(path_to_cache=str(tmp_path / "cache"))
    dataset = _build_dataset(
        tmp_path,
        [[10, 11, 12], [13, 14]],
        numpy.array([[0, 2, 8]], dtype=numpy.int32),
        config,
        "segmented",
    )

    sample = dataset[0]

    numpy.testing.assert_array_equal(
        sample["text"], [1, 13, 3, 2, 10, 11, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["types"], [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["labels"], [-1, -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    )
    numpy.testing.assert_array_equal(
        sample["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["loss_mask"], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    assert sample["is_random"] == 1
    assert sample["truncated"] == 0


def test_bert_dataset_samples_a_single_segment_without_a_classification_head(tmp_path):
    config = _config(classification_head=False, path_to_cache=str(tmp_path / "cache"))
    dataset = _build_dataset(
        tmp_path, [[10, 11, 12]], numpy.array([[0, 1, 8]], dtype=numpy.int32), config, "single"
    )

    sample = dataset[0]

    numpy.testing.assert_array_equal(
        sample["text"], [1, 10, 11, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["types"], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["labels"], [-1, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    )
    assert sample["is_random"] == 0
    assert sample["truncated"] == 0


def test_bert_dataset_trims_segments_from_both_ends_to_the_target_length(tmp_path):
    config = _config(random_seed=7, path_to_cache=str(tmp_path / "cache"))
    dataset = _build_dataset(
        tmp_path,
        [[10, 11, 12, 13, 14], [10, 11, 12, 13, 14]],
        numpy.array([[0, 2, 5]], dtype=numpy.int32),
        config,
        "truncated",
    )

    sample = dataset[0]

    numpy.testing.assert_array_equal(
        sample["text"], [1, 3, 12, 13, 2, 10, 11, 2, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["types"], [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["labels"], [-1, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    )
    assert sample["is_random"] == 1
    assert sample["truncated"] == 1


def test_bert_dataset_picks_the_segment_pivot_randomly_for_three_or_more_sentences(tmp_path):
    config = _config(path_to_cache=str(tmp_path / "cache"))
    dataset = _build_dataset(
        tmp_path,
        [[10, 11], [12, 13], [14, 10]],
        numpy.array([[0, 3, 8]], dtype=numpy.int32),
        config,
        "three_sentences",
    )

    sample = dataset[0]

    numpy.testing.assert_array_equal(
        sample["text"], [1, 10, 11, 2, 12, 13, 14, 3, 2, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["types"], [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    )
    numpy.testing.assert_array_equal(
        sample["labels"], [-1, -1, -1, -1, -1, -1, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1]
    )
    assert sample["is_random"] == 0
    assert sample["truncated"] == 0


def test_bert_dataset_requires_two_sentences_for_the_classification_head(tmp_path):
    config = _config(path_to_cache=str(tmp_path / "cache"))
    dataset = _build_dataset(
        tmp_path, [[10, 11, 12]], numpy.array([[0, 1, 8]], dtype=numpy.int32), config, "lonely"
    )

    with pytest.raises(AssertionError, match="the sample must contain at least two sentences"):
        dataset[0]


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        pytest.param(0, 3, id="replaces-with-the-mask-token"),
        pytest.param(4, 1, id="replaces-with-a-random-vocab-token"),
        pytest.param(6, None, id="keeps-the-token"),
    ],
)
def test_bert_dataset_token_mask_selects_between_mask_random_and_keep(tmp_path, seed, expected):
    config = _config(classification_head=False, path_to_cache=str(tmp_path / "cache"))
    dataset = _build_dataset(
        tmp_path, [[10, 11, 12]], numpy.array([[0, 1, 8]], dtype=numpy.int32), config, "masking"
    )

    assert dataset._get_token_mask(numpy.random.RandomState(seed)) == expected
